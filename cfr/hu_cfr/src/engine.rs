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
//! back to a simple equity heuristic and says so in the debug output. The
//! heuristic is deterministic (exact equity on the turn and river, seeded
//! Monte Carlo before), so every decision's action probabilities are known
//! exactly (`table_play`, `table_play_all`): what advantage scoring
//! (`crate::score`) needs.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;

use hu_core::cards::{id, Card};
use hu_core::hand_eval::eval7;
use hu_core::rules::{Action, ActionType, Legal, Phase, State, Street};

use crate::abs::Abstraction;
use crate::game::{Act, TreeGame, MAX_ACTIONS, TERMINAL};
use crate::live::showdown::{hand, holes, Board, H};
use crate::live::{Cache, LiveCfg, LiveDebug, LiveMode};
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
    /// Set when the decision was solved live.
    pub live: Option<LiveDebug>,
}

pub struct Decision {
    pub action: Action,
    pub debug: Debug,
    /// The probability of each real action the decision sampled from
    /// (summing to 1 up to rounding); the chosen action is one of them.
    pub dist: Vec<(Action, f64)>,
}

/// A table decision before sampling.
pub struct TablePlay {
    /// The probability of each real action (abstract actions that become the
    /// same real action merged), in the table's order.
    pub dist: Vec<(Action, f64)>,
    pub debug: Debug,
}

pub struct Engine {
    pub abs: Abstraction,
    /// Sorted by stack.
    pub strategies: Vec<Strategy>,
    /// Which streets `decide` solves live (`crate::live`).
    pub live: LiveMode,
    pub live_cfg: LiveCfg,
    /// Live solves of recent hands.
    pub cache: Cache,
}

/// Abstract player of a real seat: 0 = small blind / button, 1 = big blind.
#[inline]
pub fn abs_player(seat: usize, button: usize) -> usize {
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
    let mut s = replay_n(v, v.actions.len())?;
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

/// The state at the start of `street` (betting open), whose history is the
/// hand's first `n` actions.
pub fn state_at_street(v: &View, n: usize, street: Street) -> Result<State, String> {
    let mut s = replay_n(v, n)?;
    while s.phase == Phase::StreetComplete && s.street < street {
        s.advance_street();
    }
    if s.street != street || s.phase != Phase::Betting {
        return Err(format!("no betting at the start of the {street:?}"));
    }
    Ok(s)
}

/// The state after the hand's first `n` actions, each checked to replay
/// exactly as recorded.
fn replay_n(v: &View, n: usize) -> Result<State, String> {
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
    for (i, a) in v.actions[..n].iter().enumerate() {
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

/// Equity against a random hand (Monte Carlo).
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

/// Every hand's exact equity against a random hand (ties half) on a 4- or
/// 5-card board, `holes()` order; hands touching the board get 0.
pub fn equity_all(board: &[Card]) -> Vec<f64> {
    assert!(board.len() == 4 || board.len() == 5, "equity_all needs the turn or river");
    let mut any = vec![1f32; H];
    crate::live::showdown::mask(&mut any, board);
    let mut acc = vec![0f64; H];
    let mut out = vec![0f32; H];
    let rivers: Vec<Card> = if board.len() == 5 { vec![board[4]] } else { (0..52u8).filter(|c| !board.contains(c)).collect() };
    for &r in &rivers {
        let cards = [board[0], board[1], board[2], board[3], r];
        let mut w = any.clone();
        crate::live::showdown::mask(&mut w, &[r]);
        Board::new(cards).showdown(&w, 1.0, &mut out);
        // 990 opponent hands avoid each hand and the five board cards.
        for (h, &[a, b]) in holes().iter().enumerate() {
            if w[h] != 0.0 && a != r && b != r {
                acc[h] += 0.5 + out[h] as f64 / (2.0 * 990.0);
            }
        }
    }
    // Each hand off the 4-card board meets 46 river cards.
    let n = if board.len() == 5 { 1.0 } else { 46.0 };
    acc.iter_mut().for_each(|x| *x /= n);
    acc
}

/// The fallback's equity for `hole`: exact on the turn and river, Monte
/// Carlo seeded by the hand (deterministic) before.
fn fallback_equity(hole: [Card; 2], board: &[Card], seed: u64, n_actions: usize) -> f64 {
    if board.len() >= 4 {
        equity_all(board)[hand(hole)]
    } else {
        let mut rng = SmallRng::seed_from_u64(hand_seed(seed, 0x1_0000 + n_actions) ^ hand(hole) as u64);
        equity_vs_random(hole, board, &mut rng)
    }
}

/// The equity heuristic's action.
fn fallback_action(eq: f64, legal: &Legal, s: &State) -> Action {
    let pot_total = s.pot + s.players.iter().map(|p| p.bet).sum::<i32>();
    let to_call = legal.to_call;
    if to_call > 0 {
        let odds = to_call as f64 / (pot_total + to_call) as f64;
        if eq > 0.8 && legal.can_raise {
            Action::new(ActionType::AllIn)
        } else if eq >= odds + 0.05 {
            Action::new(ActionType::Call)
        } else {
            Action::new(ActionType::Fold)
        }
    } else if eq > 0.75 && legal.can_raise {
        let me = &s.players[s.to_act as usize];
        let pac = pot_total + (s.current_bet - me.bet);
        to_real(&Act::Bet { to: 0, frac: 1.0 }, legal, s.current_bet, pac)
    } else {
        Action::new(ActionType::Check)
    }
}

/// The table's node at the current decision of `s`: the abstract tree walked
/// along the real history, each real bet/raise translated (deterministically
/// per hand: `seed`); or why the table can't play it (the equity fallback).
/// `path` gets (seat, abstract action) per step.
fn table_node_at(tree: &TreeGame, seed: u64, button: usize, s: &State, path: &mut Vec<(usize, String)>) -> Result<u32, &'static str> {
    let mut node = 0u32;
    for (i, ha) in s.actions.iter().enumerate() {
        let nd = &tree.nodes[node as usize];
        if nd.player == TERMINAL {
            return Err("the abstract hand ended before the real one");
        }
        if nd.player as usize != abs_player(ha.seat, button) || nd.street as usize != ha.street.index() {
            return Err("the abstract and real betting diverged");
        }
        let acts = tree.actions(nd);
        let mut trng = SmallRng::seed_from_u64(hand_seed(seed, i));
        let k = translate(acts, ha, &mut trng);
        path.push((ha.seat, acts[k].label()));
        node = tree.children(nd)[k];
    }
    let nd = &tree.nodes[node as usize];
    if nd.player == TERMINAL {
        return Err("the abstract hand already ended");
    }
    if nd.player as usize != abs_player(s.to_act as usize, button) || nd.street as usize != s.street.index() {
        return Err("the abstract and real turns diverged");
    }
    Ok(node)
}

/// The action `x` (uniform in [0, 1)) picks from a distribution.
pub fn sample(dist: &[(Action, f64)], x: f64) -> Action {
    let total: f64 = dist.iter().map(|m| m.1).sum();
    let mut x = x * total;
    for (a, w) in dist {
        if x < *w {
            return *a;
        }
        x -= w;
    }
    dist.last().expect("no actions").0
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

    /// The bot's decision, with live solving per `self.live`.
    pub fn decide(&self, v: &View, rng: &mut SmallRng) -> Result<Decision, String> {
        self.decide_mode(v, rng, self.live)
    }

    /// The bot's decision with the given live-solving mode.
    pub fn decide_mode(&self, v: &View, rng: &mut SmallRng, mode: LiveMode) -> Result<Decision, String> {
        let s = replay(v)?;
        if mode.covers(s.street) {
            return crate::live::decide(self, v, &s, mode, rng);
        }
        let TablePlay { dist, mut debug } = self.table_play(&s, v.hole, v.stacks, v.button, v.seed)?;
        let action = sample(&dist, rng.gen::<f64>());
        if debug.fallback.is_none() {
            debug.chosen = action_label(&action);
        }
        Ok(Decision { action, debug, dist })
    }

    /// The table's play at real state `s` for the seat to act holding
    /// `hole`, before sampling (what `decide` samples from): `stacks` are
    /// the hand's starting stacks, `seed` its seed (translation, fallback).
    pub fn table_play(&self, s: &State, hole: [Card; 2], stacks: [i32; 2], button: usize, seed: u64) -> Result<TablePlay, String> {
        if s.phase != Phase::Betting || s.to_act < 0 {
            return Err("no decision at this state".into());
        }
        let seat = s.to_act as usize;
        let eff = stacks[0].min(stacks[1]);
        let strat = self.pick(eff);
        let tree = &strat.tree;
        let mut dbg = Debug {
            effective_bb: eff as f64 / 2.0,
            strategy_bb: strat.meta.stack as f64 / 2.0,
            street: street_name(s.street).into(),
            ..Debug::default()
        };
        let mut path = Vec::new();
        let spot = table_node_at(tree, seed, button, s, &mut path);
        dbg.path = path.into_iter().map(|(q, l)| format!("{}: {l}", if q == seat { "bot" } else { "you" })).collect();
        let legal = s.legal().ok_or("no legal actions")?;
        let node = match spot {
            Ok(node) => node,
            Err(why) => {
                dbg.fallback = Some(why.into());
                let eq = fallback_equity(hole, &s.community, seed, s.actions.len());
                let action = fallback_action(eq, &legal, s);
                dbg.chosen = format!("{} (equity {:.0}%)", action_label(&action), eq * 100.0);
                return Ok(TablePlay { dist: vec![(action, 1.0)], debug: dbg });
            }
        };
        let nd = &tree.nodes[node as usize];
        let me = &s.players[seat];
        let pot_total = s.pot + s.players.iter().map(|p| p.bet).sum::<i32>();
        let pac = pot_total + (s.current_bet - me.bet);
        let bucket = self.abs.bucket(s.street.index(), hole, &s.community);
        let n = nd.n as usize;
        let mut p = [0f64; MAX_ACTIONS];
        strat.probs(node, bucket, &mut p[..n]);
        let acts = tree.actions(nd);
        let mut dist: Vec<(Action, f64)> = Vec::new();
        for a in 0..n {
            let real = to_real(&acts[a], &legal, s.current_bet, pac);
            dbg.options.push(Opt { label: acts[a].label(), action: action_label(&real), prob: p[a] });
            match dist.iter_mut().find(|m| m.0 == real) {
                Some(m) => m.1 += p[a],
                None => dist.push((real, p[a])),
            }
        }
        dbg.bucket = Some(bucket);
        dbg.bucket_label = Some(if s.street == Street::Preflop {
            crate::classes::class_name(bucket)
        } else {
            format!("hand group {} of {} (0 = weakest)", bucket, self.abs.buckets()[s.street.index()])
        });
        dbg.node = Some(node);
        Ok(TablePlay { dist, debug: dbg })
    }

    /// `table_play` for every hand at once (`holes()` order): the real
    /// options of the seat to act at `s` and each hand's probability of each
    /// (option-major: `p[o * H + h]`). Hands with weight 0 in `w` are
    /// skipped (all 0); so must be every hand touching the board.
    pub fn table_play_all(&self, s: &State, stacks: [i32; 2], button: usize, seed: u64, w: &[f64]) -> (Vec<Action>, Vec<f64>) {
        let seat = s.to_act as usize;
        let strat = self.pick(stacks[0].min(stacks[1]));
        let tree = &strat.tree;
        let legal = s.legal().expect("no legal actions");
        let hs = holes();
        let mut opts: Vec<Action> = Vec::new();
        let index_of = |a: Action, opts: &mut Vec<Action>| match opts.iter().position(|x| *x == a) {
            Some(i) => i,
            None => {
                opts.push(a);
                opts.len() - 1
            }
        };
        match table_node_at(tree, seed, button, s, &mut Vec::new()) {
            Err(_) => {
                // The fallback, hand by hand.
                let board = &s.community;
                let exact = if board.len() >= 4 { Some(equity_all(board)) } else { None };
                let mut pick = vec![usize::MAX; H];
                for h in 0..H {
                    if w[h] == 0.0 {
                        continue;
                    }
                    let eq = match &exact {
                        Some(e) => e[h],
                        None => fallback_equity(hs[h], board, seed, s.actions.len()),
                    };
                    pick[h] = index_of(fallback_action(eq, &legal, s), &mut opts);
                }
                let mut p = vec![0f64; opts.len() * H];
                for h in 0..H {
                    if pick[h] != usize::MAX {
                        p[pick[h] * H + h] = 1.0;
                    }
                }
                (opts, p)
            }
            Ok(node) => {
                let nd = &tree.nodes[node as usize];
                let me = &s.players[seat];
                let pot_total = s.pot + s.players.iter().map(|p| p.bet).sum::<i32>();
                let pac = pot_total + (s.current_bet - me.bet);
                let n = nd.n as usize;
                let acts = tree.actions(nd);
                let to: Vec<usize> = (0..n).map(|a| index_of(to_real(&acts[a], &legal, s.current_bet, pac), &mut opts)).collect();
                let mut p = vec![0f64; opts.len() * H];
                let mut q = [0f64; MAX_ACTIONS];
                let st = s.street.index();
                for h in 0..H {
                    if w[h] == 0.0 {
                        continue;
                    }
                    strat.probs(node, self.abs.bucket(st, hs[h], &s.community), &mut q[..n]);
                    for a in 0..n {
                        p[to[a] * H + h] += q[a];
                    }
                }
                (opts, p)
            }
        }
    }

    /// Each hand's likelihood (`holes()` order) of seat `seat` having played
    /// its actions in `actions` the way this engine's table plays: the
    /// product over them of the probability of the real action it took
    /// (`table_play_all`), replayed from `init` (the hand's state before any
    /// action, with its real deck). Hands touching `dead` get 0. None when
    /// one of the actions isn't one the table could have taken.
    #[allow(clippy::too_many_arguments)]
    pub fn table_likelihood(&self, init: &State, actions: &[hu_core::rules::HandAction], seat: usize, stacks: [i32; 2], button: usize, seed: u64, dead: &[Card]) -> Option<Vec<f64>> {
        let hs = holes();
        let mut w: Vec<f64> = hs.iter().map(|&[a, b]| if dead.contains(&a) || dead.contains(&b) { 0.0 } else { 1.0 }).collect();
        let mut s = init.clone();
        for ha in actions {
            while s.phase == Phase::StreetComplete {
                s.advance_street();
            }
            let act = match ha.kind {
                ActionType::Bet | ActionType::Raise => Action::to(ha.kind, ha.to),
                k => Action::new(k),
            };
            if ha.seat == seat {
                for (h, x) in w.iter_mut().enumerate() {
                    let [a, b] = hs[h];
                    if s.community.contains(&a) || s.community.contains(&b) {
                        *x = 0.0;
                    }
                }
                let (opts, p) = self.table_play_all(&s, stacks, button, seed, &w);
                let took: Vec<usize> = (0..opts.len())
                    .filter(|&o| {
                        let mut t = s.clone();
                        t.apply(opts[o]);
                        let r = t.actions.last().unwrap();
                        (r.kind, r.to) == (ha.kind, ha.to)
                    })
                    .collect();
                if took.is_empty() {
                    return None;
                }
                for (h, x) in w.iter_mut().enumerate() {
                    *x *= took.iter().map(|&o| p[o * H + h]).sum::<f64>();
                }
            }
            s.apply(act);
        }
        Some(w)
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
        Ok(Engine { abs, strategies, live: LiveMode::Off, live_cfg: LiveCfg::default(), cache: Cache::new(256) })
    }

    /// The table's node after the hand's first `upto` actions (the same
    /// translated walk `decide` and `table_ranges` do), or None where the
    /// real hand leaves the abstract tree.
    pub fn table_node(&self, v: &View, actions: &[hu_core::rules::HandAction], upto: usize) -> Option<u32> {
        let tree = &self.pick(v.stacks[0].min(v.stacks[1])).tree;
        let mut node = 0u32;
        for (i, ha) in actions[..upto].iter().enumerate() {
            let nd = &tree.nodes[node as usize];
            if nd.player == TERMINAL || nd.player as usize != abs_player(ha.seat, v.button) || nd.street as usize != ha.street.index() {
                return None;
            }
            let mut trng = SmallRng::seed_from_u64(hand_seed(v.seed, i));
            let k = translate(tree.actions(nd), ha, &mut trng);
            node = tree.children(nd)[k];
        }
        Some(node)
    }

    /// Each seat's range after the hand's first `upto` actions as the table
    /// would have played them: per hand (`holes()` order), the product of
    /// the table's probabilities of each of that seat's actions, along the
    /// same translated history walk `decide` does (the opponent's
    /// probabilities floored at `live_cfg.opp_floor`). Stops (leaving the rest
    /// uninformative) where the real hand leaves the abstract tree. Hands
    /// that clash with the board are not zeroed here.
    pub fn table_ranges(&self, v: &View, actions: &[hu_core::rules::HandAction], upto: usize) -> [Vec<f32>; 2] {
        let strat = self.pick(v.stacks[0].min(v.stacks[1]));
        let tree = &strat.tree;
        let hs = holes();
        let mut r = [vec![1f32; H], vec![1f32; H]];
        let mut node = 0u32;
        let mut p = [0f64; MAX_ACTIONS];
        for (i, ha) in actions[..upto].iter().enumerate() {
            let nd = &tree.nodes[node as usize];
            if nd.player == TERMINAL || nd.player as usize != abs_player(ha.seat, v.button) || nd.street as usize != ha.street.index() {
                break;
            }
            let acts = tree.actions(nd);
            let mut trng = SmallRng::seed_from_u64(hand_seed(v.seed, i));
            let k = translate(acts, ha, &mut trng);
            let st = ha.street.index();
            let board = &v.board[..[0, 3, 4, 5][st]];
            let n = nd.n as usize;
            let opp = ha.seat != v.bot;
            let floor = if opp { self.live_cfg.opp_floor } else { 0.0 };
            // The abstract actions whose probabilities make the likelihood:
            // the translated one, or (opp_sizes) every bet/raise for a
            // bet/raise.
            let aggressive = |a: &Act| matches!(a, Act::Bet { .. } | Act::Raise { .. } | Act::AllIn { .. });
            let take: Vec<usize> = if opp && self.live_cfg.opp_sizes && aggressive(&acts[k]) {
                (0..n).filter(|&a| aggressive(&acts[a])).collect()
            } else {
                vec![k]
            };
            for (h, w) in r[ha.seat].iter_mut().enumerate() {
                if *w == 0.0 {
                    continue;
                }
                let [a, b] = hs[h];
                if board.contains(&a) || board.contains(&b) {
                    *w = 0.0;
                    continue;
                }
                strat.probs(node, self.abs.bucket(st, [a, b], board), &mut p[..n]);
                let like = (take.iter().map(|&a| p[a]).sum::<f64>() as f32).max(floor);
                *w *= if opp && self.live_cfg.opp_temper != 1.0 { like.powf(self.live_cfg.opp_temper) } else { like };
            }
            node = tree.children(nd)[k];
        }
        r
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
