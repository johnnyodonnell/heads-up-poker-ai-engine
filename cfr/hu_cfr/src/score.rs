//! Advantage scoring: how much better live solving plays than the table,
//! measured decision by decision instead of from the hands' outcomes.
//!
//! In a match between the engine with live solving and a table opponent,
//! both players' strategies are known exactly at every decision: the live
//! solve's probabilities, `Engine::table_play` for the table (which the bot
//! would have played instead) and for the opponent. So at each live decision
//! `t` the bot's *advantage* over its table can be scored directly:
//!
//! ```text
//! A(t) = Σ_a (σ_live(a) − σ_table(a)) · Q(t, a)
//! ```
//!
//! where `Q(t, a)` is what the bot expects to win if it takes real action `a`
//! at `t` and both players play their tables from there on. Summed over a
//! hand's live decisions, this is the performance difference lemma (Kakade &
//! Langford 2002) in the game seen from the bot's side, where the opponent's
//! hand is a hidden state distributed as its exact range: the hand's score has
//! exactly the same expectation as the difference between the bot's result
//! playing live and its result playing the table, against this opponent, for
//! any opponent table and whatever the live solve does. It has none of the
//! luck of the cards still to come, the opponent's actual hand or which
//! actions were sampled, so far fewer deals measure the same difference.
//!
//! - **Range** (`g`): the opponent's hand weighted by its exact range: how
//!   likely each hand was to play the hand so far as the opponent's table
//!   plays (`Engine::table_likelihood`).
//! - **Real hand** (`full`): the opponent's actual hand instead. No range
//!   involved, noisier. Both are unbiased, so they must agree (`hu scorecheck`
//!   and the eval's summary compare them).
//!
//! `Q` is exact: the real game tree below `t` is walked with both tables'
//! real decisions (the same translation, rounding and fallback as in play),
//! every river card dealt, showdowns and all-in run-outs scored exactly.

use hu_core::allin_ev::is_runout;
use hu_core::cards::Card;
use hu_core::hand_eval::eval7;
use hu_core::rules::{Action, ActionType, Phase, State, Status, Street};

use crate::engine::Engine;
use crate::live::showdown::{hand, holes, H};

/// What scoring found at one or more of the bot's decisions, in chips.
#[derive(Clone, Copy, Debug, Default)]
pub struct Adv {
    /// Advantage over the table against the opponent's exact range.
    pub g: f64,
    /// The same against the opponent's real hand.
    pub full: f64,
    /// Decisions scored.
    pub decisions: u32,
    /// Of those, how many had no exact opponent range (an opponent action
    /// its table couldn't have taken); `g` used the real hand there.
    pub no_range: u32,
    /// Decisions where the opponent's real hand had zero weight in its exact
    /// range (should never happen).
    pub range_misses: u32,
}

impl std::ops::AddAssign for Adv {
    fn add_assign(&mut self, o: Adv) {
        self.g += o.g;
        self.full += o.full;
        self.decisions += o.decisions;
        self.no_range += o.no_range;
        self.range_misses += o.range_misses;
    }
}

/// Both tables playing out a hand from a real state, for the bot in `seat`
/// holding `hole`.
pub struct Scorer<'a> {
    /// The bot's table (the engine played without live solving).
    pub me: &'a Engine,
    /// The opponent's table.
    pub opp: &'a Engine,
    pub seat: usize,
    pub hole: [Card; 2],
    /// The hand's starting stacks, button and seed (as in the bots' views).
    pub stacks: [i32; 2],
    pub button: usize,
    pub seed: u64,
}

fn strength(hole: [Card; 2], board: &[Card]) -> u32 {
    let mut c = [0u8; 7];
    c[0] = hole[0];
    c[1] = hole[1];
    c[2..2 + board.len()].copy_from_slice(board);
    eval7(&c[..2 + board.len()])
}

impl Scorer<'_> {
    /// Σ_h w[h] × the bot's chips won from `s` on (against its starting
    /// stack), both players playing their tables and the opponent holding
    /// hand `h` (`holes()` order). `w` must be 0 on hands touching the bot's
    /// cards or the board.
    pub fn value(&self, s: &State, w: &[f64]) -> f64 {
        match s.phase {
            Phase::Betting => {
                let seat = s.to_act as usize;
                let (opts, p) = if seat == self.seat {
                    let mut one = vec![0f64; H];
                    one[hand(self.hole)] = 1.0;
                    self.me.table_play_all(s, self.stacks, self.button, self.seed, &one)
                } else {
                    self.opp.table_play_all(s, self.stacks, self.button, self.seed, w)
                };
                let mut v = 0.0;
                for (o, &a) in opts.iter().enumerate() {
                    let row = &p[o * H..(o + 1) * H];
                    let w2: Vec<f64> = if seat == self.seat {
                        let q = row[hand(self.hole)];
                        if q == 0.0 {
                            continue;
                        }
                        w.iter().map(|x| x * q).collect()
                    } else {
                        w.iter().zip(row).map(|(x, q)| x * q).collect()
                    };
                    if w2.iter().all(|&x| x == 0.0) {
                        continue;
                    }
                    let mut t = s.clone();
                    t.apply(a);
                    v += self.value(&t, &w2);
                }
                v
            }
            Phase::StreetComplete => self.street_end(s, w),
            p => panic!("scoring reached {p:?}"),
        }
    }

    /// Betting on the street is over: a fold, a showdown, an all-in run-out,
    /// or the next street's card.
    fn street_end(&self, s: &State, w: &[f64]) -> f64 {
        let me = &s.players[self.seat];
        let op = &s.players[1 - self.seat];
        let total: f64 = w.iter().sum();
        if me.status == Status::Folded {
            return -(me.commit as f64) * total;
        }
        if op.status == Status::Folded {
            return op.commit as f64 * total;
        }
        let m = me.commit.min(op.commit) as f64;
        if s.street == Street::River {
            return m * self.showdown(&s.community, w);
        }
        if is_runout(s) {
            return m * self.runout(&s.community, w);
        }
        assert!(s.community.len() >= 3, "scoring deals no flop");
        // Each opponent hand meets the cards outside both hands and the board.
        let n = (52 - 4 - s.community.len()) as f64;
        let mut v = 0.0;
        for c in 0..52u8 {
            if self.hole.contains(&c) || s.community.contains(&c) {
                continue;
            }
            let w2: Vec<f64> = holes().iter().zip(w).map(|(&[a, b], &x)| if a == c || b == c { 0.0 } else { x / n }).collect();
            if w2.iter().all(|&x| x == 0.0) {
                continue;
            }
            let mut t = s.clone();
            t.deck.push(c);
            t.advance_street();
            v += self.value(&t, &w2);
        }
        v
    }

    /// Σ_h w[h] × (+1 win, −1 loss, 0 tie) on a complete board.
    fn showdown(&self, board: &[Card], w: &[f64]) -> f64 {
        let mine = strength(self.hole, board);
        let mut v = 0.0;
        for (h, &x) in w.iter().enumerate() {
            if x != 0.0 {
                let theirs = strength(holes()[h], board);
                if mine > theirs {
                    v += x;
                } else if mine < theirs {
                    v -= x;
                }
            }
        }
        v
    }

    /// The same averaged over every run-out of the board (each opponent hand
    /// over the cards it leaves).
    fn runout(&self, board: &[Card], w: &[f64]) -> f64 {
        let free: Vec<Card> = (0..52u8).filter(|c| !self.hole.contains(c) && !board.contains(c)).collect();
        let need = 5 - board.len();
        let left = (52 - 4 - board.len()) as f64;
        let mut v = 0.0;
        let mut add = |cards: &[Card], per: f64| {
            let mut b = board.to_vec();
            b.extend_from_slice(cards);
            let w2: Vec<f64> = holes().iter().zip(w).map(|(&[a, x], &y)| if cards.contains(&a) || cards.contains(&x) { 0.0 } else { y / per }).collect();
            v += self.showdown(&b, &w2);
        };
        match need {
            1 => free.iter().for_each(|&c| add(&[c], left)),
            2 => {
                let pairs = left * (left - 1.0) / 2.0;
                for i in 0..free.len() {
                    for j in i + 1..free.len() {
                        add(&[free[i], free[j]], pairs);
                    }
                }
            }
            _ => panic!("scoring runs out the turn or the flop only"),
        }
        v
    }
}

/// The resulting (kind, to) of a real action at `s`.
fn record(s: &State, a: Action) -> (ActionType, i32) {
    let mut t = s.clone();
    t.apply(a);
    let r = t.actions.last().expect("applied");
    (r.kind, r.to)
}

/// The advantage of the bot (`seat`, to act at `s`) over its table at this
/// decision, where it sampled from `dist`: Σ_a (σ(a) − σ_table(a)) × Q(a),
/// against the opponent's exact range and against its real hand. `init` is
/// the hand's state before any action (with its real deck); `me` is the
/// bot's engine (its table is what the advantage is over), `opp` the
/// opponent's (a table).
#[allow(clippy::too_many_arguments)]
pub fn advantage(me: &Engine, opp: &Engine, s: &State, init: &State, seat: usize, stacks: [i32; 2], seed: u64, dist: &[(Action, f64)]) -> Adv {
    let button = s.button as usize;
    let hole = [s.players[seat].hole[0], s.players[seat].hole[1]];
    let opp_hole = [s.players[1 - seat].hole[0], s.players[1 - seat].hole[1]];
    let sc = Scorer { me, opp, seat, hole, stacks, button, seed };
    let mut one = vec![0f64; H];
    one[hand(hole)] = 1.0;
    let (opts, p) = me.table_play_all(s, stacks, button, seed, &one);

    // Both distributions over the real actions they lead to.
    let mut rows: Vec<((ActionType, i32), Action, f64, f64)> = Vec::new();
    let live_total: f64 = dist.iter().map(|x| x.1).sum();
    let table_total: f64 = (0..opts.len()).map(|o| p[o * H + hand(hole)]).sum();
    let mut put = |a: Action, live: f64, table: f64| {
        let r = record(s, a);
        match rows.iter_mut().find(|x| x.0 == r) {
            Some(x) => {
                x.2 += live;
                x.3 += table;
            }
            None => rows.push((r, a, live, table)),
        }
    };
    for &(a, x) in dist {
        put(a, x / live_total, 0.0);
    }
    for (o, &a) in opts.iter().enumerate() {
        put(a, 0.0, p[o * H + hand(hole)] / table_total);
    }

    let mut dead = vec![hole[0], hole[1]];
    dead.extend_from_slice(&s.community);
    let range = opp.table_likelihood(init, &s.actions, 1 - seat, stacks, button, seed, &dead);
    let mut real = vec![0f64; H];
    real[hand(opp_hole)] = 1.0;
    let mut adv = Adv { decisions: 1, ..Adv::default() };
    let range_total = range.as_ref().map(|r| r.iter().sum::<f64>()).unwrap_or(0.0);
    match &range {
        Some(r) if r[hand(opp_hole)] == 0.0 => adv.range_misses = 1,
        None => adv.no_range = 1,
        _ => {}
    }
    for &(_, a, live, table) in &rows {
        let d = live - table;
        if d.abs() < 1e-12 {
            continue;
        }
        let mut t = s.clone();
        t.apply(a);
        let q_full = sc.value(&t, &real);
        adv.full += d * q_full;
        adv.g += d * match &range {
            Some(r) if range_total > 0.0 => sc.value(&t, r) / range_total,
            _ => q_full,
        };
    }
    adv
}
