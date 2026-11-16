//! Play-time decisions from trained strategies.
//!
//! For each decision the engine:
//! 1. replays the hand so far through the rules port (the exact real state,
//!    and a parity check on every recorded action);
//! 2. picks the strategy whose stack depth is nearest the hand's effective
//!    stack (in ratio);
//! 3. walks the abstract tree along the real betting history, translating
//!    each real bet/raise to an abstract size with the pseudo-harmonic
//!    mapping (Ganzfried & Sandholm 2013), deterministically per hand;
//! 4. looks up the bot's bucket and the strategy at that node, maps each
//!    abstract action to a legal real action (pot fractions of the *real*
//!    pot, rounded and clamped), and samples one.
//!
//! When the abstract tree can't follow the real hand (a fourth raise on one
//! street, a real raise mapped to an abstract all-in, ...) the engine falls
//! back to a simple equity heuristic and says so in the debug output.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;

use hu_core::cards::{id, Card};
use hu_core::hand_eval::eval7;
use hu_core::rules::{Action, ActionType, Legal, Phase, State, Street};

use crate::abs::Abstraction;
use crate::game::{Act, MAX_ACTIONS, TERMINAL};
use crate::strategy::Strategy;

/// One betting action of the hand so far.
#[derive(Clone, Copy, Debug)]
pub struct HistAction {
    pub seat: usize,
    /// fold / check / call / bet / raise (an all-in is the bet/raise/call it
    /// amounts to).
    pub kind: ActionType,
    /// The actor's street bet after the action (the raise-to for bet/raise).
    pub to: i32,
}

/// What the bot knows when it must act.
#[derive(Clone, Debug)]
pub struct View {
    pub bot: usize,
    /// The button's seat (it posts the small blind).
    pub button: usize,
    /// Both stacks at the start of the hand, before the blinds.
    pub stacks: [i32; 2],
    pub hole: [Card; 2],
    pub board: Vec<Card>,
    pub actions: Vec<HistAction>,
    /// Per-hand seed: keeps the translation of earlier actions stable across
    /// the bot's decisions in one hand.
    pub seed: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Opt {
    /// The abstract action.
    pub label: String,
    /// The real action it becomes.
    pub action: String,
    pub prob: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Debug {
    pub effective_bb: f64,
    pub strategy_bb: f64,
    pub street: String,
    pub bucket: Option<u32>,
    /// The bot's hand group, readable: its preflop class, or "group n of N".
    pub bucket_label: Option<String>,
    pub node: Option<u32>,
    /// The abstract betting history the real one was translated to.
    pub path: Vec<String>,
    pub options: Vec<Opt>,
    pub chosen: String,
    /// Why the equity heuristic was used, if it was.
    pub fallback: Option<String>,
}

pub struct Decision {
    pub action: Action,
    pub debug: Debug,
}

pub struct Engine {
    pub abs: Abstraction,
    /// Sorted by stack.
    pub strategies: Vec<Strategy>,
}

/// Abstract player of a real seat: 0 = small blind / button, 1 = big blind.
#[inline]
fn abs_player(seat: usize, button: usize) -> usize {
    usize::from(seat != button)
}

fn street_name(s: Street) -> &'static str {
    match s {
        Street::Preflop => "preflop",
        Street::Flop => "flop",
        Street::Turn => "turn",
        Street::River => "river",
    }
}

pub fn action_label(a: &Action) -> String {
    match a.kind {
        ActionType::Fold => "fold".into(),
        ActionType::Check => "check".into(),
        ActionType::Call => "call".into(),
        ActionType::Bet => format!("bet {}", a.amount.unwrap_or(0)),
        ActionType::Raise => format!("raise to {}", a.amount.unwrap_or(0)),
        ActionType::AllIn => "all-in".into(),
    }
}

/// A full deck ordered so `start_hand` deals the bot its cards and the
/// streets deal the known board (the opponent and future cards are filler).
fn deck_for(v: &View) -> Result<Vec<Card>, String> {
    let mut used = [false; 52];
    for &c in v.hole.iter().chain(&v.board) {
        if c >= 52 || used[c as usize] {
            return Err(format!("bad or repeated card {c}"));
        }
        used[c as usize] = true;
    }
    let mut spare = (0..52u8).filter(|&c| !used[c as usize]);
    let opp = [spare.next().unwrap(), spare.next().unwrap()];
    let mut holes = [[0u8; 2]; 2];
    holes[v.bot] = v.hole;
    holes[1 - v.bot] = opp;
    let mut order: Vec<Card> = vec![holes[0][0], holes[0][1], holes[1][0], holes[1][1]];
    order.extend_from_slice(&v.board);
    order.extend(spare);
    order.reverse(); // the rules pop cards off the end
    Ok(order)
}

/// The real state at the bot's decision, rebuilt by replaying the hand.
pub fn replay(v: &View) -> Result<State, String> {
    if v.bot > 1 || v.button > 1 {
        return Err("seats are 0 and 1".into());
    }
    if v.stacks.iter().any(|&s| s <= 0) {
        return Err(format!("bad stacks {:?}", v.stacks));
    }
    let mut s = State::new_table(v.stacks);
    s.button = (1 - v.button) as i32; // start_hand moves the button on one seat
    s.start_hand(deck_for(v)?);
    if s.button as usize != v.button {
        return Err("button mismatch".into());
    }
    for (i, a) in v.actions.iter().enumerate() {
        while s.phase == Phase::StreetComplete {
            s.advance_street();
        }
        if s.phase != Phase::Betting || s.to_act < 0 || s.to_act as usize != a.seat {
            return Err(format!("action {i}: seat {} is not to act", a.seat));
        }
        let act = match a.kind {
            ActionType::Bet | ActionType::Raise => Action::to(a.kind, a.to),
            k => Action::new(k),
        };
        s.apply(act);
        let rec = s.actions.last().unwrap();
        if rec.kind != a.kind || rec.to != a.to {
            return Err(format!("action {i}: replayed as {:?} to {} (sent {:?} to {})", rec.kind, rec.to, a.kind, a.to));
        }
    }
    while s.phase == Phase::StreetComplete && s.community.len() < v.board.len() {
        s.advance_street();
    }
    if s.phase != Phase::Betting || s.to_act < 0 || s.to_act as usize != v.bot {
        return Err("it is not the bot's turn".into());
    }
    if s.community != v.board {
        return Err("board mismatch".into());
    }
    Ok(s)
}

/// The abstract action a real one is translated to (index into `acts`).
fn translate(acts: &[Act], ha: &hu_core::rules::HandAction, rng: &mut SmallRng) -> usize {
    let find = |pred: &dyn Fn(&Act) -> bool| acts.iter().position(|a| pred(a));
    let passive = || find(&|a| matches!(a, Act::Check | Act::Call)).unwrap_or(0);
    match ha.kind {
        ActionType::Fold => find(&|a| matches!(a, Act::Fold)).unwrap_or_else(passive),
        ActionType::Check | ActionType::Call => passive(),
        _ => {
            let allin = find(&|a| matches!(a, Act::AllIn { .. }));
            if ha.allin {
                if let Some(k) = allin {
                    return k;
                }
            }
            let bet_before = ha.to - ha.added;
            let pac = ha.pot_before + (ha.current_bet_before - bet_before);
            let f = (ha.to - ha.current_bet_before) as f64 / pac.max(1) as f64;
            let mut sizes: Vec<(usize, f64)> = acts
                .iter()
                .enumerate()
                .filter(|(_, a)| matches!(a, Act::Bet { .. } | Act::Raise { .. }))
                .map(|(i, a)| (i, a.frac().unwrap() as f64))
                .collect();
            // All-in is the upper neighbour for oversized bets: a 3x-pot bet
            // is mapped between pot and all-in, not down to pot.
            if let Some(k) = allin {
                sizes.push((k, acts[k].frac().unwrap() as f64));
            }
            if sizes.is_empty() {
                return allin.unwrap_or_else(passive);
            }
            sizes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            if f <= sizes[0].1 {
                return sizes[0].0;
            }
            if f >= sizes[sizes.len() - 1].1 {
                return sizes[sizes.len() - 1].0;
            }
            let j = sizes.iter().position(|s| s.1 >= f).unwrap();
            let ((ia, a), (ib, b)) = (sizes[j - 1], sizes[j]);
            // Pseudo-harmonic mapping: probability of mapping down to A.
            let p_a = (b - f) * (1.0 + a) / ((b - a) * (1.0 + f));
            if rng.gen::<f64>() < p_a { ia } else { ib }
        }
    }
}

/// The real action an abstract one becomes at the current real state.
fn to_real(a: &Act, legal: &Legal, current_bet: i32, pac: i32) -> Action {
    let passive = || Action::new(if legal.can_check { ActionType::Check } else { ActionType::Call });
    match *a {
        Act::Fold => {
            if legal.can_fold {
                Action::new(ActionType::Fold)
            } else {
                passive()
            }
        }
        Act::Check | Act::Call => passive(),
        Act::Bet { frac, .. } | Act::Raise { frac, .. } => {
            if !legal.can_raise {
                return passive();
            }
            let to = (current_bet as f64 + frac as f64 * pac as f64).round() as i32;
            let to = to.clamp(legal.min_raise_to, legal.max_raise_to);
            if to >= legal.max_raise_to {
                Action::new(ActionType::AllIn)
            } else {
                Action::to(if legal.opening { ActionType::Bet } else { ActionType::Raise }, to)
            }
        }
        Act::AllIn { .. } => {
            if legal.can_raise {
                Action::new(ActionType::AllIn)
            } else {
                passive()
            }
        }
    }
}

fn hand_seed(seed: u64, i: usize) -> u64 {
    seed ^ (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Equity against a random hand (Monte Carlo), for the fallback.
fn equity_vs_random(hole: [Card; 2], board: &[Card], rng: &mut SmallRng) -> f64 {
    let mut dead = 0u64;
    for &c in hole.iter().chain(board) {
        dead |= 1 << c;
    }
    let mut cards = [0u8; 7];
    let n = 2000;
    let mut won = 0.0;
    for _ in 0..n {
        let mut used = dead;
        let mut draw = || loop {
            let c = rng.gen_range(0..52u8);
            if used & (1 << c) == 0 {
                used |= 1 << c;
                return c;
            }
        };
        let opp = [draw(), draw()];
        let mut full = [0u8; 5];
        full[..board.len()].copy_from_slice(board);
        for slot in full[board.len()..].iter_mut() {
            *slot = draw();
        }
        cards[2..].copy_from_slice(&full);
        cards[0] = hole[0];
        cards[1] = hole[1];
        let me = eval7(&cards);
        cards[0] = opp[0];
        cards[1] = opp[1];
        let them = eval7(&cards);
        won += if me > them { 1.0 } else if me == them { 0.5 } else { 0.0 };
    }
    won / n as f64
}

impl Engine {
    /// The strategy whose stack is nearest `eff` chips in ratio.
    pub fn pick(&self, eff: i32) -> &Strategy {
        let e = eff.max(1) as f64;
        self.strategies
            .iter()
            .min_by(|a, b| {
                let da = (a.meta.stack as f64 / e).ln().abs();
                let db = (b.meta.stack as f64 / e).ln().abs();
                da.partial_cmp(&db).unwrap()
            })
            .expect("no strategies loaded")
    }

    pub fn decide(&self, v: &View, rng: &mut SmallRng) -> Result<Decision, String> {
        let s = replay(v)?;
        let eff = v.stacks[0].min(v.stacks[1]);
        let strat = self.pick(eff);
        let tree = &strat.tree;
        let mut dbg = Debug {
            effective_bb: eff as f64 / 2.0,
            strategy_bb: strat.meta.stack as f64 / 2.0,
            street: street_name(s.street).into(),
            ..Debug::default()
        };
        let who = |seat: usize| if seat == v.bot { "bot" } else { "you" };

        // Walk the abstract tree along the real history.
        let mut node = 0u32;
        for (i, ha) in s.actions.iter().enumerate() {
            let nd = &tree.nodes[node as usize];
            if nd.player == TERMINAL {
                dbg.fallback = Some("the abstract hand ended before the real one".into());
                break;
            }
            if nd.player as usize != abs_player(ha.seat, v.button) || nd.street as usize != ha.street.index() {
                dbg.fallback = Some("the abstract and real betting diverged".into());
                break;
            }
            let acts = tree.actions(nd);
            let mut trng = SmallRng::seed_from_u64(hand_seed(v.seed, i));
            let k = translate(acts, ha, &mut trng);
            dbg.path.push(format!("{}: {}", who(ha.seat), acts[k].label()));
            node = tree.children(nd)[k];
        }
        let nd = &tree.nodes[node as usize];
        if dbg.fallback.is_none() {
            if nd.player == TERMINAL {
                dbg.fallback = Some("the abstract hand already ended".into());
            } else if nd.player as usize != abs_player(v.bot, v.button) || nd.street as usize != s.street.index() {
                dbg.fallback = Some("the abstract and real turns diverged".into());
            }
        }

        let legal = s.legal().ok_or("no legal actions")?;
        let me = &s.players[v.bot];
        let pot_total = s.pot + s.players.iter().map(|p| p.bet).sum::<i32>();
        let pac = pot_total + (s.current_bet - me.bet);

        if dbg.fallback.is_some() {
            let eq = equity_vs_random(v.hole, &s.community, rng);
            let to_call = legal.to_call;
            let action = if to_call > 0 {
                let odds = to_call as f64 / (pot_total + to_call) as f64;
                if eq > 0.8 && legal.can_raise {
                    Action::new(ActionType::AllIn)
                } else if eq >= odds + 0.05 {
                    Action::new(ActionType::Call)
                } else {
                    Action::new(ActionType::Fold)
                }
            } else if eq > 0.75 && legal.can_raise {
                to_real(&Act::Bet { to: 0, frac: 1.0 }, &legal, s.current_bet, pac)
            } else {
                Action::new(ActionType::Check)
            };
            dbg.chosen = format!("{} (equity {:.0}%)", action_label(&action), eq * 100.0);
            return Ok(Decision { action, debug: dbg });
        }

        let bucket = self.abs.bucket(s.street.index(), v.hole, &s.community);
        let n = nd.n as usize;
        let mut p = [0f64; MAX_ACTIONS];
        strat.probs(node, bucket, &mut p[..n]);
        let acts = tree.actions(nd);
        let mut merged: Vec<(Action, f64)> = Vec::new();
        for a in 0..n {
            let real = to_real(&acts[a], &legal, s.current_bet, pac);
            dbg.options.push(Opt { label: acts[a].label(), action: action_label(&real), prob: p[a] });
            match merged.iter_mut().find(|m| m.0 == real) {
                Some(m) => m.1 += p[a],
                None => merged.push((real, p[a])),
            }
        }
        let total: f64 = merged.iter().map(|m| m.1).sum();
        let mut x = rng.gen::<f64>() * total;
        let mut action = merged.last().unwrap().0;
        for (a, w) in &merged {
            if x < *w {
                action = *a;
                break;
            }
            x -= w;
        }
        dbg.bucket = Some(bucket);
        dbg.bucket_label = Some(if s.street == Street::Preflop {
            crate::classes::class_name(bucket)
        } else {
            format!("hand group {} of {} (0 = weakest)", bucket, self.abs.buckets()[s.street.index()])
        });
        dbg.node = Some(node);
        dbg.chosen = action_label(&action);
        Ok(Decision { action, debug: dbg })
    }

    /// Load an abstraction and every strategy directory under `dir`.
    pub fn load(abs_dir: &std::path::Path, strategy_dirs: &[std::path::PathBuf]) -> std::io::Result<Engine> {
        let abs = Abstraction::load(abs_dir)?;
        let mut strategies = Vec::new();
        for d in strategy_dirs {
            let s = Strategy::load(d)?;
            if s.meta.abstraction != abs.meta.name {
                return Err(std::io::Error::other(format!(
                    "{} was trained on abstraction {}, not {}",
                    d.display(),
                    s.meta.abstraction,
                    abs.meta.name
                )));
            }
            strategies.push(s);
        }
        strategies.sort_by_key(|s| s.meta.stack);
        if strategies.is_empty() {
            return Err(std::io::Error::other("no strategies"));
        }
        Ok(Engine { abs, strategies })
    }
}

/// The bot's view of a hu_core state (used by the evaluator; the server
/// builds the same thing from the app's request).
pub fn view_of(s: &State, bot: usize, start_stacks: [i32; 2], seed: u64) -> View {
    View {
        bot,
        button: s.button as usize,
        stacks: start_stacks,
        hole: [s.players[bot].hole[0], s.players[bot].hole[1]],
        board: s.community.clone(),
        actions: s.actions.iter().map(|a| HistAction { seat: a.seat, kind: a.kind, to: a.to }).collect(),
        seed,
    }
}

/// Card ids for logs/debug.
pub fn cards_str(cards: &[Card]) -> String {
    cards.iter().map(|&c| id(c)).collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::Bot;
    use hu_core::table::deal_random_hand;

    /// The view the server gets at each decision must replay to the exact
    /// real state: random hands from random stack splits, random actions.
    #[test]
    fn replay_reproduces_real_states() {
        let mut rng = SmallRng::seed_from_u64(11);
        let mut checked = 0;
        for h in 0..3000 {
            let mut s = deal_random_hand(&mut rng);
            let start = [s.players[0].stack + s.players[0].commit, s.players[1].stack + s.players[1].commit];
            loop {
                match s.phase {
                    Phase::Betting => {
                        let seat = s.to_act as usize;
                        let r = replay(&view_of(&s, seat, start, h)).expect("replay");
                        assert_eq!((r.pot, r.current_bet, r.min_raise, r.to_act), (s.pot, s.current_bet, s.min_raise, s.to_act));
                        assert_eq!(r.community, s.community);
                        assert_eq!(r.players[seat].hole, s.players[seat].hole);
                        for i in 0..2 {
                            assert_eq!((r.players[i].stack, r.players[i].bet, r.players[i].commit), (s.players[i].stack, s.players[i].bet, s.players[i].commit));
                        }
                        assert_eq!(r.legal(), s.legal());
                        checked += 1;
                        let a = Bot::Random.act(&s, &mut rng);
                        s.apply(a);
                    }
                    Phase::StreetComplete => s.advance_street(),
                    _ => break,
                }
            }
        }
        assert!(checked > 5000, "{checked}");
    }
}
