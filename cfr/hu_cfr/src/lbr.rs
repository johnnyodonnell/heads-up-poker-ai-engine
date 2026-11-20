//! Local best response (LBR; Lisý & Bowling 2017): an exploiter that knows
//! the bot's strategy and looks for the bets the bot answers badly. Its win
//! rate is a lower bound on how exploitable the bot is, which matches
//! against fixed opponents can't measure.
//!
//! At each decision on the streets it is set to play, LBR holds the bot's
//! exact range (how likely each of the 1,326 hands is to have played the hand
//! so far, from the bot's own strategy, the same path `Engine::decide` takes)
//! and takes the action worth the most, assuming the hand is checked and
//! called down after it unless the bot folds right away (the paper's
//! Figure 1):
//!
//! ```text
//! U(call)  = wp·pot − (1 − wp)·asked
//! U(bet a) = fp·pot + (1 − fp)·(wp′·(pot + a) − (1 − wp′)·(asked + a))
//! ```
//!
//! `wp` is LBR's showdown share against the range with every remaining board
//! card dealt, `fp` the bot's probability of folding to the bet, `wp′` the
//! share against the hands that don't fold, `a` the chips the bet adds beyond
//! a call. It folds when nothing is worth more than folding. On the other
//! streets it checks and calls.
//!
//! What LBR knows is the bot's strategy, not its luck. The bot translates
//! off-menu sizes at random, seeded per hand; LBR's model averages over
//! `model_seeds` independent draws of that translation on streets the bot
//! plays from its table (knowing the real draws, it could bet 2× pot exactly
//! when the bot will read it as all-in). On streets the bot solves live,
//! where every answer costs a solve, it uses one draw, sampled by how well
//! each explains the bot's actions on the earlier streets.
//!
//! Scoring: besides the chips, every hand is scored against the bot's exact
//! range at the end, with the bot's real draws (the scorer may know them): a
//! showdown or all-in is worth LBR's expected share against every hand the
//! bot could hold, every remaining board card dealt. That has the same
//! expectation as the chips, without the luck of the bot's cards.

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use rand::rngs::SmallRng;
use rand::seq::index::sample;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use hu_core::allin_ev::{expected_stacks, is_runout};
use hu_core::cards::Card;
use hu_core::hand_eval::eval7;
use hu_core::rules::{Action, ActionType, HandAction, Phase, State, Status, Street};

use crate::engine::{view_of, Engine};
use crate::eval::{deal_spec, summarize, DealSpec, Stacks};
use crate::live::showdown::{holes, H};
use crate::live::{self, LiveMode};

/// A bet or raise size LBR considers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Size {
    /// The smallest legal bet or raise.
    Min,
    /// A fraction of the pot after calling (the menus' convention).
    Pot(f32),
    AllIn,
}

impl Size {
    pub fn parse(s: &str) -> Option<Size> {
        match s {
            "min" => Some(Size::Min),
            "allin" | "all-in" => Some(Size::AllIn),
            x => x.parse::<f32>().ok().filter(|f| *f > 0.0).map(Size::Pot),
        }
    }

    pub fn parse_list(s: &str) -> Option<Vec<Size>> {
        s.split(',').filter(|x| !x.trim().is_empty()).map(|x| Size::parse(x.trim())).collect()
    }

    pub fn label(&self) -> String {
        match self {
            Size::Min => "min".into(),
            Size::Pot(f) => format!("{f}"),
            Size::AllIn => "all-in".into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct LbrCfg {
    /// Streets LBR plays (index 0 = preflop); on the others it checks and
    /// calls.
    pub streets: [bool; 4],
    /// Bet and raise sizes it considers where the bot plays from its table.
    pub sizes: Vec<Size>,
    /// The same where the bot solves live (each one costs the bot a solve).
    pub live_sizes: Vec<Size>,
    /// Draws of the bot's translation its model averages over.
    pub model_seeds: usize,
    /// Boards sampled for preflop showdown shares.
    pub preflop_boards: usize,
}

impl Default for LbrCfg {
    fn default() -> LbrCfg {
        LbrCfg {
            streets: [false, true, true, true],
            sizes: Size::parse_list("min,0.33,0.5,0.75,1,1.5,2,3,allin").unwrap(),
            live_sizes: Size::parse_list("min,0.5,1,2,allin").unwrap(),
            model_seeds: 16,
            preflop_boards: 2000,
        }
    }
}

impl LbrCfg {
    /// "234" = flop, turn and river; "1234" adds preflop; "none" = never.
    pub fn parse_streets(s: &str) -> Option<[bool; 4]> {
        if s == "none" {
            return Some([false; 4]);
        }
        let mut out = [false; 4];
        for ch in s.chars() {
            let i = ch.to_digit(10)? as usize;
            if !(1..=4).contains(&i) {
                return None;
            }
            out[i - 1] = true;
        }
        Some(out)
    }

    pub fn describe(&self) -> String {
        let names = ["preflop", "flop", "turn", "river"];
        let on: Vec<&str> = (0..4).filter(|&i| self.streets[i]).map(|i| names[i]).collect();
        let list = |v: &[Size]| v.iter().map(Size::label).collect::<Vec<_>>().join(",");
        format!(
            "plays {}; sizes {} (live streets {}); model averages {} translation draws",
            if on.is_empty() { "nothing (checks and calls)".to_string() } else { on.join("/") },
            list(&self.sizes),
            list(&self.live_sizes),
            self.model_seeds
        )
    }
}

pub struct Lbr<'a> {
    /// The bot under test, solving live per `mode`.
    pub bot: &'a Engine,
    pub mode: LiveMode,
    pub cfg: LbrCfg,
}

/// One LBR decision, for the match's statistics.
#[derive(Clone, Debug)]
pub struct Pick {
    pub street: usize,
    /// "check", "call", "fold", "bet 0.5", "raise min", "all-in", ...
    pub label: String,
    /// For bets and raises: the bot's fold probability under LBR's model,
    /// and whether it really folded.
    pub fp: Option<f64>,
    pub folded: Option<bool>,
    /// The chosen action's utility (chips) under LBR's assumption.
    pub u: f64,
}

/// LBR's scratch for one hand.
struct HandCtx {
    rng: SmallRng,
    /// Showdown shares for the board length they were computed at.
    shares: Option<(usize, Vec<f64>)>,
    picks: Vec<Pick>,
    /// Decisions where no model draw explained the bot's play (LBR then used
    /// the bot's real draw).
    model_misses: usize,
}

/// LBR's share of the pot (1 win, ½ tie, 0 loss) against each bot hand
/// (`holes()` order), averaged over every completion of `board`; preflop
/// over `boards` sampled boards (seeded). Hands touching LBR's cards or the
/// board get 0 and must have zero weight.
pub fn shares(me: [Card; 2], board: &[Card], boards: usize, seed: u64) -> Vec<f64> {
    let hs = holes();
    let mut dead = 0u64;
    for &c in me.iter().chain(board) {
        dead |= 1 << c;
    }
    let rest: Vec<Card> = (0..52u8).filter(|&c| dead & (1 << c) == 0).collect();
    let mut win = vec![0f64; H];
    let mut cnt = vec![0u32; H];
    let mut full = [0u8; 5];
    full[..board.len()].copy_from_slice(board);
    let mut run = |full: &[Card; 5]| {
        let mut cards = [0u8; 7];
        cards[2..].copy_from_slice(full);
        cards[0] = me[0];
        cards[1] = me[1];
        let mine = eval7(&cards);
        let mut used = dead;
        for &c in full {
            used |= 1 << c;
        }
        for (h, &[a, b]) in hs.iter().enumerate() {
            if used & ((1 << a) | (1 << b)) != 0 {
                continue;
            }
            cards[0] = a;
            cards[1] = b;
            let theirs = eval7(&cards);
            win[h] += if mine > theirs {
                1.0
            } else if mine == theirs {
                0.5
            } else {
                0.0
            };
            cnt[h] += 1;
        }
    };
    match 5 - board.len() {
        0 => run(&full),
        1 => {
            for &r in &rest {
                full[4] = r;
                run(&full);
            }
        }
        2 => {
            for i in 0..rest.len() {
                for j in i + 1..rest.len() {
                    full[3] = rest[i];
                    full[4] = rest[j];
                    run(&full);
                }
            }
        }
        5 => {
            let mut rng = SmallRng::seed_from_u64(seed);
            for _ in 0..boards.max(1) {
                for (k, i) in sample(&mut rng, rest.len(), 5).into_iter().enumerate() {
                    full[k] = rest[i];
                }
                run(&full);
            }
        }
        n => panic!("shares: {n} board cards to come"),
    }
    win.iter().zip(&cnt).map(|(&w, &n)| if n > 0 { w / n as f64 } else { 0.0 }).collect()
}

fn act_of(ha: &HandAction) -> Action {
    match ha.kind {
        ActionType::Bet | ActionType::Raise => Action::to(ha.kind, ha.to),
        k => Action::new(k),
    }
}

/// All chips in the middle (gathered pot plus every street bet).
fn pot_total(s: &State) -> i32 {
    s.pot + s.players.iter().map(|p| p.bet).sum::<i32>()
}

/// Model draw `j` of the translation for a hand whose real seed is `seed`:
/// an independent stream (LBR never sees the real draws).
fn model_seed(seed: u64, j: usize) -> u64 {
    let mut z = seed ^ 0x4C42_5221_D00D_F00D ^ (j as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Index of the first action on a street the bot solves live (or the end).
fn first_live(actions: &[HandAction], mode: LiveMode) -> usize {
    actions.iter().position(|a| mode.covers(a.street)).unwrap_or(actions.len())
}

impl<'a> Lbr<'a> {
    /// The bot's play at `s` for every hand (option-major probabilities,
    /// `p[o * H + h]`) as it plays with translation seed `seed`: its live
    /// solve on the streets it solves live, its table otherwise. Hands with
    /// zero weight in `w` may come back 0 (table streets).
    fn bot_play_all(&self, s: &State, start: [i32; 2], seed: u64, w: &[f64]) -> Result<(Vec<Action>, Vec<f64>), String> {
        if self.mode.covers(s.street) {
            let v = view_of(s, s.to_act as usize, start, seed);
            live::play_all(self.bot, &v, s, self.mode)
        } else {
            Ok(self.bot.table_play_all(s, start, s.button as usize, seed, w))
        }
    }

    /// Each hand's likelihood of the bot (seat `bot`) having taken its
    /// actions in `actions`, replayed from `init`, with translation seed
    /// `seed`: the product of its probabilities of each. Hands touching
    /// `dead` get 0. Under a translation draw other than the bot's real one,
    /// a real action can be impossible (the draw put the bot at another node
    /// of its tree): every hand's likelihood is then 0; with the real draw
    /// (`real`), that is an error.
    pub fn bot_range(&self, init: &State, actions: &[HandAction], bot: usize, start: [i32; 2], seed: u64, dead: &[Card]) -> Result<Vec<f64>, String> {
        self.range_with(init, actions, bot, start, seed, dead, false)
    }

    #[allow(clippy::too_many_arguments)]
    fn range_with(&self, init: &State, actions: &[HandAction], bot: usize, start: [i32; 2], seed: u64, dead: &[Card], real: bool) -> Result<Vec<f64>, String> {
        let hs = holes();
        let mut w: Vec<f64> = hs.iter().map(|&[a, b]| if dead.contains(&a) || dead.contains(&b) { 0.0 } else { 1.0 }).collect();
        let mut s = init.clone();
        for ha in actions {
            while s.phase == Phase::StreetComplete {
                s.advance_street();
            }
            if ha.seat == bot {
                for (h, x) in w.iter_mut().enumerate() {
                    let [a, b] = hs[h];
                    if s.community.contains(&a) || s.community.contains(&b) {
                        *x = 0.0;
                    }
                }
                let (opts, p) = self.bot_play_all(&s, start, seed, &w)?;
                let took: Vec<usize> = (0..opts.len())
                    .filter(|&o| {
                        let mut t = s.clone();
                        t.apply(opts[o]);
                        let r = t.actions.last().unwrap();
                        (r.kind, r.to) == (ha.kind, ha.to)
                    })
                    .collect();
                if took.is_empty() {
                    if real {
                        return Err(format!("the bot's {:?} to {} is not among its options {opts:?}", ha.kind, ha.to));
                    }
                    return Ok(vec![0.0; H]);
                }
                for (h, x) in w.iter_mut().enumerate() {
                    if *x != 0.0 {
                        *x *= took.iter().map(|&o| p[o * H + h]).sum::<f64>();
                    }
                }
            }
            s.apply(act_of(ha));
        }
        Ok(w)
    }

    fn shares_at(&self, ctx: &mut HandCtx, me: [Card; 2], board: &[Card], seed: u64) -> Vec<f64> {
        if let Some((n, sh)) = &ctx.shares {
            if *n == board.len() {
                return sh.clone();
            }
        }
        let sh = shares(me, board, self.cfg.preflop_boards, seed ^ 0x5348_4152_4553);
        ctx.shares = Some((board.len(), sh.clone()));
        sh
    }

    /// The real action a size becomes at `s` (None if no raise is legal).
    fn real(size: Size, s: &State) -> Option<Action> {
        let l = s.legal()?;
        if !l.can_raise {
            return None;
        }
        let me = &s.players[s.to_act as usize];
        let to = match size {
            Size::AllIn => return Some(Action::new(ActionType::AllIn)),
            Size::Min => l.min_raise_to,
            Size::Pot(f) => {
                let pac = pot_total(s) + (s.current_bet - me.bet);
                ((s.current_bet as f64 + f as f64 * pac as f64).round() as i32).clamp(l.min_raise_to, l.max_raise_to)
            }
        };
        Some(if to >= l.max_raise_to {
            Action::new(ActionType::AllIn)
        } else {
            Action::to(if l.opening { ActionType::Bet } else { ActionType::Raise }, to)
        })
    }

    /// LBR's action at `s` (its turn), the hand's real translation seed
    /// `seed` (used only when no model draw can explain the bot's play).
    fn act(&self, s: &State, init: &State, start: [i32; 2], seed: u64, ctx: &mut HandCtx) -> Result<Action, String> {
        let me = s.to_act as usize;
        let bot = 1 - me;
        let legal = s.legal().ok_or("no legal actions")?;
        let passive = Action::new(if legal.can_check { ActionType::Check } else { ActionType::Call });
        let passive_label = if legal.can_check { "check" } else { "call" };
        let st = s.street.index();
        if !self.cfg.streets[st] {
            ctx.picks.push(Pick { street: st, label: passive_label.into(), fp: None, folded: None, u: 0.0 });
            return Ok(passive);
        }
        let hole = [s.players[me].hole[0], s.players[me].hole[1]];
        let mut dead = s.community.clone();
        dead.extend_from_slice(&hole);
        let live = self.mode.covers(s.street);

        // The model: draws of the bot's translation, and its range under each.
        let draws: Vec<u64> = (0..self.cfg.model_seeds.max(1)).map(|j| model_seed(seed, j)).collect();
        let (mut keys, mut ranges): (Vec<u64>, Vec<Vec<f64>>) = if !live {
            let r = draws.iter().map(|&k| self.bot_range(init, &s.actions, bot, start, k, &dead)).collect::<Result<Vec<_>, _>>()?;
            (draws, r)
        } else {
            // One draw (each answer costs a live solve), sampled by how well
            // each explains the bot's play before the live streets.
            let k0 = first_live(&s.actions, self.mode);
            let fit = draws
                .iter()
                .map(|&k| self.bot_range(init, &s.actions[..k0], bot, start, k, &dead).map(|w| w.iter().sum::<f64>()))
                .collect::<Result<Vec<f64>, _>>()?;
            let total: f64 = fit.iter().sum();
            let mut keys = Vec::new();
            if total > 0.0 {
                let mut x = ctx.rng.gen::<f64>() * total;
                let mut pick = fit.iter().rposition(|&f| f > 0.0).unwrap();
                for (i, &f) in fit.iter().enumerate() {
                    if x < f {
                        pick = i;
                        break;
                    }
                    x -= f;
                }
                keys.push(draws[pick]);
            }
            let r = keys.iter().map(|&k| self.bot_range(init, &s.actions, bot, start, k, &dead)).collect::<Result<Vec<_>, _>>()?;
            (keys, r)
        };
        let mut mass: f64 = ranges.iter().map(|r| r.iter().sum::<f64>()).sum();
        if mass <= 0.0 {
            // No draw explains the bot's play: use its real one.
            ctx.model_misses += 1;
            keys = vec![seed];
            ranges = vec![self.range_with(init, &s.actions, bot, start, seed, &dead, true)?];
            mass = ranges[0].iter().sum();
            if mass <= 0.0 {
                return Err("the bot's range is empty under its own strategy".into());
            }
        }

        let sh = self.shares_at(ctx, hole, &s.community, seed);
        let wp: f64 = ranges.iter().map(|r| r.iter().zip(&sh).map(|(w, x)| w * x).sum::<f64>()).sum::<f64>() / mass;
        let pot = pot_total(s);
        let asked = legal.to_call;
        // Calling all-in for less: the part of the bot's bet LBR can't match
        // goes back.
        let unmatched = (s.current_bet - s.players[me].bet - asked).max(0);
        let u_pass = wp * (pot - unmatched) as f64 - (1.0 - wp) * asked as f64;
        let mut best = (u_pass, passive, passive_label.to_string(), None::<f64>);

        let sizes = if live { &self.cfg.live_sizes } else { &self.cfg.sizes };
        let mut seen: Vec<Action> = Vec::new();
        for &size in sizes {
            let Some(a) = Self::real(size, s) else { continue };
            if seen.contains(&a) {
                continue;
            }
            seen.push(a);
            let mut s2 = s.clone();
            s2.apply(a);
            if s2.phase != Phase::Betting || s2.to_act as usize != bot {
                continue;
            }
            // What the bot can match of the bet (the rest would go back).
            let to = s2.players[me].bet.min(s2.players[bot].bet + s2.players[bot].stack);
            let add = (to - s.current_bet) as f64;
            let (mut fold, mut keep, mut keep_wp) = (0f64, 0f64, 0f64);
            for (&k, r) in keys.iter().zip(&ranges) {
                let (opts, p) = self.bot_play_all(&s2, start, k, r)?;
                let fi = opts.iter().position(|x| x.kind == ActionType::Fold);
                for h in 0..H {
                    if r[h] == 0.0 {
                        continue;
                    }
                    let f = fi.map_or(0.0, |i| p[i * H + h]);
                    fold += r[h] * f;
                    keep += r[h] * (1.0 - f);
                    keep_wp += r[h] * (1.0 - f) * sh[h];
                }
            }
            let fp = fold / mass;
            let wp2 = if keep > 0.0 { keep_wp / keep } else { 0.0 };
            let u = fp * pot as f64 + (1.0 - fp) * (wp2 * (pot as f64 + add) - (1.0 - wp2) * (asked as f64 + add));
            if u > best.0 {
                let label = match a.kind {
                    ActionType::AllIn => "all-in".to_string(),
                    _ => format!("{} {}", if legal.opening { "bet" } else { "raise" }, size.label()),
                };
                best = (u, a, label, Some(fp));
            }
        }
        let (u, action, label, fp) = if best.0 <= 0.0 && legal.can_fold {
            (0.0, Action::new(ActionType::Fold), "fold".to_string(), None)
        } else {
            best
        };
        ctx.picks.push(Pick { street: st, label, fp, folded: None, u });
        Ok(action)
    }

    /// LBR's expected result (chips) at a hand's showdown or all-in: its
    /// share against the bot's exact range (real translation draws), every
    /// remaining board card dealt.
    fn range_score(&self, s: &State, init: &State, me: usize, start: [i32; 2], seed: u64, ctx: &mut HandCtx) -> Result<f64, String> {
        let bot = 1 - me;
        let hole = [s.players[me].hole[0], s.players[me].hole[1]];
        let mut dead = s.community.clone();
        dead.extend_from_slice(&hole);
        let pi = self.range_with(init, &s.actions, bot, start, seed, &dead, true)?;
        let mass: f64 = pi.iter().sum();
        if mass <= 0.0 {
            return Err("the bot's range is empty at the showdown".into());
        }
        let sh = self.shares_at(ctx, hole, &s.community, seed);
        let e = pi.iter().zip(&sh).map(|(w, x)| w * x).sum::<f64>() / mass;
        let m = s.players[me].commit.min(s.players[bot].commit) as f64;
        Ok(m * (2.0 * e - 1.0))
    }

    /// One play of a deal with LBR in seat `me` and the bot in the other:
    /// LBR's chips (runouts at their expectation, as `eval` scores them) and
    /// its range-scored result.
    fn play(&self, deal: &DealSpec, me: usize, c: &Counters) -> Result<(f64, f64, Vec<Pick>, usize), String> {
        // The bot's draws (and the runouts'), exactly as in `eval`.
        let mut rng = SmallRng::seed_from_u64(deal.pseed);
        let mut ctx = HandCtx {
            rng: SmallRng::seed_from_u64(deal.pseed ^ 0x4C42_5200 ^ me as u64),
            shares: None,
            picks: Vec::new(),
            model_misses: 0,
        };
        let mut s = State::new_table(deal.stacks);
        s.button = (1 - deal.button) as i32;
        s.start_hand(deal.deck.clone());
        let init = s.clone();
        let start = deal.stacks;
        let mut scored: Option<f64> = None;
        loop {
            match s.phase {
                Phase::Betting => {
                    let seat = s.to_act as usize;
                    let action = if seat == me {
                        self.act(&s, &init, start, deal.pseed, &mut ctx)?
                    } else {
                        let v = view_of(&s, seat, start, deal.pseed);
                        let d = self.bot.decide_mode(&v, &mut rng, self.mode)?;
                        c.decisions.fetch_add(1, Relaxed);
                        if d.debug.fallback.is_some() {
                            c.fallbacks.fetch_add(1, Relaxed);
                        }
                        if let Some(p) = ctx.picks.last_mut() {
                            if p.fp.is_some() && p.folded.is_none() {
                                p.folded = Some(d.action.kind == ActionType::Fold);
                            }
                        }
                        d.action
                    };
                    s.apply(action);
                }
                Phase::StreetComplete => {
                    let both_in = s.players.iter().all(|p| matches!(p.status, Status::Active | Status::AllIn));
                    if is_runout(&s) {
                        let ev = expected_stacks(&s, &mut rng);
                        let chips = ev[me] - start[me] as f64;
                        let sc = self.range_score(&s, &init, me, start, deal.pseed, &mut ctx)?;
                        return Ok((chips, sc, ctx.picks, ctx.model_misses));
                    }
                    if both_in && s.street == Street::River {
                        scored = Some(self.range_score(&s, &init, me, start, deal.pseed, &mut ctx)?);
                    }
                    s.advance_street();
                }
                Phase::HandOver | Phase::MatchOver => {
                    let chips = (s.players[me].stack - start[me]) as f64;
                    return Ok((chips, scored.unwrap_or(chips), ctx.picks, ctx.model_misses));
                }
                Phase::Waiting => unreachable!(),
            }
        }
    }
}

struct Counters {
    decisions: AtomicUsize,
    fallbacks: AtomicUsize,
}

/// Counts a finished deal when dropped; a line on stderr every 10%.
struct Progress<'a> {
    done: &'a AtomicUsize,
    of: usize,
    t0: std::time::Instant,
}

impl Drop for Progress<'_> {
    fn drop(&mut self) {
        let k = self.done.fetch_add(1, Relaxed) + 1;
        let step = (self.of / 10).max(1);
        if k % step == 0 && k < self.of {
            eprintln!("  {k}/{} deals, {:.0}s", self.of, self.t0.elapsed().as_secs_f64());
        }
    }
}

/// A match's results, from LBR's side.
pub struct LbrResult {
    /// bb/100 with a 95% interval: chips, and range-scored.
    pub chips: (f64, f64),
    pub scored: (f64, f64),
    pub hands: usize,
    /// Per deal (mean of its two plays), chips and range-scored.
    pub per_deal: Vec<(f64, f64)>,
    pub picks: Vec<Pick>,
    pub bot_decisions: usize,
    pub bot_fallbacks: usize,
    pub model_misses: usize,
    /// Deals that failed (an error in the model or the bot), skipped.
    pub errors: Vec<(usize, String)>,
}

impl<'a> Lbr<'a> {
    /// LBR against the bot over `deals` duplicate deals from deal `from`:
    /// the same deals, seats and draws as `eval::play_match` with this seed
    /// (so an LBR that plays no street is exactly the `station` bot).
    pub fn play_match(&self, from: usize, deals: usize, stacks: Stacks, seed: u64) -> LbrResult {
        let c = Counters { decisions: AtomicUsize::new(0), fallbacks: AtomicUsize::new(0) };
        let done = AtomicUsize::new(0);
        let t0 = std::time::Instant::now();
        let rows: Vec<Result<((f64, f64), Vec<Pick>, usize), (usize, String)>> = (from..from + deals)
            .into_par_iter()
            .map(|d| {
                let _progress = Progress { done: &done, of: deals, t0 };
                let deal = deal_spec(seed, d, stacks);
                let (mut chips, mut sc, mut picks, mut misses) = (0.0, 0.0, Vec::new(), 0);
                // The bot in seat 0 first, as player A in `eval`.
                for bot_seat in 0..2 {
                    let (x, y, p, m) = self.play(&deal, 1 - bot_seat, &c).map_err(|e| (d, e))?;
                    chips += x;
                    sc += y;
                    picks.extend(p);
                    misses += m;
                }
                Ok(((chips / 2.0, sc / 2.0), picks, misses))
            })
            .collect();
        let mut per_deal = Vec::new();
        let mut picks = Vec::new();
        let mut model_misses = 0;
        let mut errors = Vec::new();
        for r in rows {
            match r {
                Ok((x, p, m)) => {
                    per_deal.push(x);
                    picks.extend(p);
                    model_misses += m;
                }
                Err(e) => errors.push(e),
            }
        }
        let chips = summarize(&per_deal.iter().map(|x| x.0).collect::<Vec<_>>());
        let scored = summarize(&per_deal.iter().map(|x| x.1).collect::<Vec<_>>());
        LbrResult {
            chips,
            scored,
            hands: per_deal.len() * 2,
            per_deal,
            picks,
            bot_decisions: c.decisions.load(Relaxed),
            bot_fallbacks: c.fallbacks.load(Relaxed),
            model_misses,
            errors,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hu_core::cards::parse;
    use hu_core::equity::equity;

    fn cards(s: &str) -> Vec<Card> {
        s.split_whitespace().map(|x| parse(x).unwrap()).collect()
    }

    /// Exact shares against single hands agree with the Monte Carlo equity
    /// on the turn, the flop and the river.
    #[test]
    fn shares_match_monte_carlo_equity() {
        let mut rng = SmallRng::seed_from_u64(3);
        let me = [parse("Ah").unwrap(), parse("Kh").unwrap()];
        for board in ["Qh 7d 2c", "Qh 7d 2c 9h", "Qh 7d 2c 9h 3s"] {
            let b = cards(board);
            let sh = shares(me, &b, 0, 1);
            for opp in ["Qs Qc", "7h 6h", "Ad Kd", "2d 2h"] {
                let o = cards(opp);
                let h = crate::live::showdown::hand([o[0], o[1]]);
                let mc = equity(&me, &[&o], &b, &[], 40_000, &mut rng);
                assert!((sh[h] - mc).abs() < 0.01, "{board} vs {opp}: exact {} mc {mc}", sh[h]);
            }
        }
    }

    /// Preflop shares (sampled boards) are close to known equities.
    #[test]
    fn preflop_shares_are_close() {
        let me = [parse("Ah").unwrap(), parse("As").unwrap()];
        let sh = shares(me, &[], 20_000, 9);
        let kk = crate::live::showdown::hand([parse("Kh").unwrap(), parse("Ks").unwrap()]);
        assert!((sh[kk] - 0.82).abs() < 0.015, "AA vs KK {}", sh[kk]);
        // Hands sharing a card with LBR's never meet it.
        let ax = crate::live::showdown::hand([parse("Ah").unwrap(), parse("2c").unwrap()]);
        assert_eq!(sh[ax], 0.0);
    }

    #[test]
    fn sizes_parse() {
        assert_eq!(Size::parse_list("min,0.5,allin").unwrap(), vec![Size::Min, Size::Pot(0.5), Size::AllIn]);
        assert!(Size::parse_list("min,x").is_none());
        assert_eq!(LbrCfg::parse_streets("234").unwrap(), [false, true, true, true]);
        assert_eq!(LbrCfg::parse_streets("none").unwrap(), [false; 4]);
        assert!(LbrCfg::parse_streets("5").is_none());
    }

    /// Bet sizes become legal real actions: the minimum, pot fractions of
    /// the pot after calling (clamped), all-in.
    #[test]
    fn sizes_become_legal_actions() {
        let mut s = State::new_table([200, 200]);
        s.start_hand((0..52).collect());
        // Small blind to act: 1 in, 2 to match; pot after calling = 4.
        assert_eq!(Lbr::real(Size::Min, &s), Some(Action::to(ActionType::Raise, 4)));
        assert_eq!(Lbr::real(Size::Pot(1.0), &s), Some(Action::to(ActionType::Raise, 6)));
        assert_eq!(Lbr::real(Size::Pot(0.1), &s), Some(Action::to(ActionType::Raise, 4)));
        assert_eq!(Lbr::real(Size::Pot(100.0), &s), Some(Action::new(ActionType::AllIn)));
        assert_eq!(Lbr::real(Size::AllIn, &s), Some(Action::new(ActionType::AllIn)));
    }
}
