//! Live solving (v2): at a turn or river decision, solve the rest of the hand
//! from the start of the current street instead of looking the decision up.
//!
//! - The subgame has the real pot, stacks and bet sizes (`subgame`): the
//!   street's real actions are in the tree as they happened, so nothing is
//!   translated or rounded, and every hand is exact (no hand groups).
//! - Each player's range at the root is how likely each of the 1,326 hands
//!   is to have played the hand this way: from the table on streets the
//!   table played (the same history walk `Engine::decide` does), from the
//!   solve itself on streets that were solved.
//! - The bot's own earlier actions on this street are frozen to the strategy
//!   it actually played (kept per hand), so its range stays consistent.
//!   The opponent's are free: the solve works out how the opponent's range
//!   would play them. (Re-solving without the "safe" gadget: the opponent's
//!   values aren't constrained to the table's.)
//! - Solves are deterministic (fixed iterations, ordered sums), so anything
//!   missing from the per-hand cache is recomputed exactly.

pub mod showdown;
pub mod solver;
pub mod subgame;

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rand::rngs::SmallRng;
use rand::Rng;
use serde::Serialize;

use hu_core::rules::{HandAction, State, Street};

use crate::engine::{action_label, state_at_street, Debug, Decision, Engine, HistAction, Opt, View};
use crate::menu::LiveMenu;
use showdown::{hand, mask, H};
use solver::Solver;
use subgame::{Kind, Subgame};

/// Which streets are solved live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiveMode {
    Off,
    River,
    /// Turn and river.
    Turn,
}

impl LiveMode {
    pub fn parse(s: &str) -> Option<LiveMode> {
        match s {
            "off" => Some(LiveMode::Off),
            "river" => Some(LiveMode::River),
            "turn" => Some(LiveMode::Turn),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            LiveMode::Off => "off",
            LiveMode::River => "river",
            LiveMode::Turn => "turn",
        }
    }

    pub fn covers(self, street: Street) -> bool {
        match self {
            LiveMode::Off => false,
            LiveMode::River => street == Street::River,
            LiveMode::Turn => matches!(street, Street::Turn | Street::River),
        }
    }
}

#[derive(Clone, Debug)]
pub struct LiveCfg {
    pub river_iters: u32,
    pub turn_iters: u32,
    /// Solve the 48 river cards of a turn solve in parallel.
    pub parallel: bool,
}

impl Default for LiveCfg {
    fn default() -> LiveCfg {
        LiveCfg { river_iters: 400, turn_iters: 200, parallel: true }
    }
}

/// What a decision's live solving did (in the debug output).
#[derive(Clone, Debug, Default, Serialize)]
pub struct LiveDebug {
    pub street: String,
    /// Solves run for this decision (more than one when earlier solves of
    /// the hand had to be recomputed), and their total time.
    pub solves: u32,
    pub ms: u64,
    pub iterations: u32,
    pub decision_nodes: usize,
    /// Of the solve this decision used, in % of the pot at its root.
    pub exploitability_pct: f64,
    /// Root range sizes, in hand combos (out of 1,326).
    pub bot_range: f64,
    pub opp_range: f64,
}

/// A finished solve: the tree and the average strategy at its current-street
/// decision nodes.
pub struct Solved {
    pub g: Subgame,
    pub strat: HashMap<u32, Vec<f32>>,
    /// Root ranges per seat.
    pub ranges: [Vec<f32>; 2],
    pub iterations: u32,
    pub ms: u64,
    pub exploitability: f64,
}

type HandKey = (u64, usize);
type SolveKey = (usize, usize, usize); // (live mode, street index, real actions on the street)

/// Recent hands' solves, so later decisions in a hand reuse them.
pub struct Cache {
    inner: Mutex<(HashMap<HandKey, HashMap<SolveKey, Arc<Solved>>>, VecDeque<HandKey>)>,
    hands: usize,
}

impl Cache {
    pub fn new(hands: usize) -> Cache {
        Cache { inner: Mutex::new((HashMap::new(), VecDeque::new())), hands }
    }

    fn get(&self, hk: HandKey, sk: SolveKey) -> Option<Arc<Solved>> {
        self.inner.lock().unwrap().0.get(&hk).and_then(|m| m.get(&sk)).cloned()
    }

    fn put(&self, hk: HandKey, sk: SolveKey, s: Arc<Solved>) {
        let mut g = self.inner.lock().unwrap();
        let (map, order) = &mut *g;
        if !map.contains_key(&hk) {
            order.push_back(hk);
            while order.len() > self.hands {
                if let Some(old) = order.pop_front() {
                    map.remove(&old);
                }
            }
        }
        map.entry(hk).or_default().insert(sk, s);
    }
}

#[derive(Default)]
struct Stats {
    solves: u32,
    ms: u64,
}

fn board_len(street: Street) -> usize {
    match street {
        Street::Preflop => 0,
        Street::Flop => 3,
        Street::Turn => 4,
        Street::River => 5,
    }
}

/// Index of the street's first action in the hand history.
fn street_start(actions: &[HandAction], street: Street) -> usize {
    actions.iter().position(|a| a.street == street).unwrap_or(actions.len())
}

/// Ranges at the start of `street`, per seat, board hands zeroed.
fn street_ranges(e: &Engine, v: &View, s: &State, mode: LiveMode, key: HandKey, street: Street, st: &mut Stats) -> Result<[Vec<f32>; 2], String> {
    let k = street_start(&s.actions, street);
    let board = &v.board[..board_len(street)];
    if street == Street::River && mode.covers(Street::Turn) {
        let kt = street_start(&s.actions, Street::Turn);
        let turn: &[HistAction] = &v.actions[kt..k];
        if let Some(j) = turn.iter().rposition(|a| a.seat == v.bot) {
            // The last turn solve covers the whole turn: the bot's actions
            // (frozen or chosen there) and the opponent's replies.
            let prev = solve(e, v, s, mode, key, Street::Turn, j, st)?;
            let g = &prev.g;
            let mut r = prev.ranges.clone();
            let mut node = 0u32;
            for a in turn {
                let nd = &g.nodes[node as usize];
                let Kind::Decision(q) = nd.kind else {
                    return Err("the turn's actions left the turn solve's tree".into());
                };
                if q as usize != a.seat {
                    return Err("turn solve: wrong seat to act".into());
                }
                let edge = g.records(nd).iter().position(|x| *x == (a.kind, a.to)).ok_or("turn action not in the turn solve's tree")?;
                let sg = prev.strat.get(&node).ok_or("no strategy at a turn node")?;
                for (h, w) in r[a.seat].iter_mut().enumerate() {
                    *w *= sg[edge * H + h];
                }
                node = g.children(nd)[edge];
            }
            for x in r.iter_mut() {
                mask(x, board);
            }
            return Ok(r);
        }
    }
    let mut r = e.table_ranges(v, &s.actions, k);
    for x in r.iter_mut() {
        mask(x, board);
    }
    Ok(r)
}

/// The solve of `street` with its first `n` real actions taken (cached).
#[allow(clippy::too_many_arguments)]
fn solve(e: &Engine, v: &View, s: &State, mode: LiveMode, key: HandKey, street: Street, n: usize, st: &mut Stats) -> Result<Arc<Solved>, String> {
    let sk = (mode as usize, street.index(), n);
    if let Some(x) = e.cache.get(key, sk) {
        return Ok(x);
    }
    let k = street_start(&s.actions, street);
    let forced: Vec<HistAction> = v.actions[k..k + n].to_vec();
    let start = state_at_street(v, k, street)?;
    let ranges = street_ranges(e, v, s, mode, key, street, st)?;
    // The bot's own earlier actions on this street, frozen.
    let mut frozen: Vec<(usize, Vec<f32>)> = Vec::new();
    for (j, a) in forced.iter().enumerate() {
        if a.seat == v.bot {
            let prev = solve(e, v, s, mode, key, street, j, st)?;
            let node = prev.g.path[j];
            frozen.push((j, prev.strat.get(&node).ok_or("no strategy at an earlier decision")?.clone()));
        }
    }
    let menu = if street == Street::River { LiveMenu::river() } else { LiveMenu::turn() };
    let g = subgame::build(&start, &forced, &menu, &LiveMenu::river_in_turn())?;
    let mut fz = Vec::with_capacity(frozen.len());
    for (j, sg) in frozen {
        let node = g.path[j];
        if g.nodes[node as usize].n as usize * H != sg.len() {
            return Err(format!("the bot's decision {j} on this street has a different set of options than when it was made"));
        }
        fz.push((node, sg));
    }
    let iters = if street == Street::River { e.live_cfg.river_iters } else { e.live_cfg.turn_iters };
    let t0 = Instant::now();
    let board = &v.board[..board_len(street)];
    let (strat, ranges, exploitability) = {
        let mut sv = Solver::new(&g, board, ranges, fz, e.live_cfg.parallel);
        sv.run(iters);
        let x = sv.exploitability();
        let mut strat = HashMap::new();
        for (i, nd) in g.nodes.iter().enumerate() {
            if matches!(nd.kind, Kind::Decision(_)) && !nd.river {
                strat.insert(i as u32, sv.strategy(i as u32));
            }
        }
        (strat, sv.ranges.clone(), x)
    };
    let ms = t0.elapsed().as_millis() as u64;
    st.solves += 1;
    st.ms += ms;
    let solved = Arc::new(Solved { g, strat, ranges, iterations: iters, ms, exploitability });
    e.cache.put(key, sk, solved.clone());
    Ok(solved)
}

/// The bot's decision at `s` (the replayed state of `v`), solved live.
pub fn decide(e: &Engine, v: &View, s: &State, mode: LiveMode, rng: &mut SmallRng) -> Result<Decision, String> {
    let street = s.street;
    let k = street_start(&s.actions, street);
    let n = s.actions.len() - k;
    let key = (v.seed, v.bot);
    let mut st = Stats::default();
    let solved = solve(e, v, s, mode, key, street, n, &mut st)?;
    let g = &solved.g;
    if g.path.len() != n + 1 {
        return Err("live solve: no decision at the current node".into());
    }
    let node = g.current();
    let nd = &g.nodes[node as usize];
    if nd.kind != Kind::Decision(v.bot as u8) {
        return Err("live solve: the current node is not the bot's".into());
    }
    let sg = solved.strat.get(&node).ok_or("live solve: no strategy at the current node")?;
    let h = hand(v.hole);
    let acts = g.actions(nd);
    let probs: Vec<f64> = (0..nd.n as usize).map(|a| sg[a * H + h] as f64).collect();
    let total: f64 = probs.iter().sum();
    let mut x = rng.gen::<f64>() * total;
    let mut pick = probs.len() - 1;
    for (a, p) in probs.iter().enumerate() {
        if x < *p {
            pick = a;
            break;
        }
        x -= p;
    }
    let action = acts[pick];

    let pot_now = s.pot + s.players.iter().map(|p| p.bet).sum::<i32>();
    let label = |a: &hu_core::rules::Action| -> String {
        match a.amount {
            Some(to) if a.kind != hu_core::rules::ActionType::AllIn => {
                let me = &s.players[v.bot];
                let pac = pot_now + (s.current_bet - me.bet);
                format!("{} ({:.0}% pot)", action_label(a), (to - s.current_bet) as f64 / pac.max(1) as f64 * 100.0)
            }
            _ => action_label(a),
        }
    };
    let eff = v.stacks[0].min(v.stacks[1]);
    let mut dbg = Debug {
        effective_bb: eff as f64 / 2.0,
        strategy_bb: eff as f64 / 2.0,
        street: format!("{:?}", street).to_lowercase(),
        bucket_label: Some("exact hand (live solve)".into()),
        node: Some(node),
        path: v.actions[k..].iter().map(|a| format!("{}: {:?} to {}", if a.seat == v.bot { "bot" } else { "you" }, a.kind, a.to).to_lowercase()).collect(),
        ..Debug::default()
    };
    for (a, p) in acts.iter().zip(&probs) {
        dbg.options.push(Opt { label: label(a), action: action_label(a), prob: *p });
    }
    dbg.chosen = label(&action);
    dbg.live = Some(LiveDebug {
        street: dbg.street.clone(),
        solves: st.solves,
        ms: st.ms,
        iterations: solved.iterations,
        decision_nodes: g.decisions(),
        exploitability_pct: solved.exploitability / g.pot.max(1) as f64 * 100.0,
        bot_range: solved.ranges[v.bot].iter().map(|&x| x as f64).sum(),
        opp_range: solved.ranges[1 - v.bot].iter().map(|&x| x as f64).sum(),
    });
    Ok(Decision { action, debug: dbg })
}

/// One solve of the bot's current decision with `iters` iterations, outside
/// the cache (for benchmarks): (ms, exploitability in % of the pot,
/// decision nodes, solver floats).
pub fn bench(e: &Engine, v: &View, mode: LiveMode, iters: u32) -> Result<(u64, f64, usize, usize), String> {
    let s = crate::engine::replay(v)?;
    let street = s.street;
    let k = street_start(&s.actions, street);
    let n = s.actions.len() - k;
    let mut st = Stats::default();
    let key = (v.seed ^ 0xBE4C_0000_0000_0000, v.bot);
    // Ranges (and frozen strategies) through the normal path, then time
    // just this solve.
    let ranges = street_ranges(e, v, &s, mode, key, street, &mut st)?;
    let mut fz = Vec::new();
    for (j, a) in v.actions[k..k + n].iter().enumerate() {
        if a.seat == v.bot {
            let prev = solve(e, v, &s, mode, key, street, j, &mut st)?;
            fz.push((j, prev.strat[&prev.g.path[j]].clone()));
        }
    }
    let start = state_at_street(v, k, street)?;
    let menu = if street == Street::River { LiveMenu::river() } else { LiveMenu::turn() };
    let g = subgame::build(&start, &v.actions[k..k + n], &menu, &LiveMenu::river_in_turn())?;
    let fz = fz.into_iter().map(|(j, sg)| (g.path[j], sg)).collect();
    let t0 = Instant::now();
    let mut sv = Solver::new(&g, &v.board[..board_len(street)], ranges, fz, e.live_cfg.parallel);
    sv.run(iters);
    let ms = t0.elapsed().as_millis() as u64;
    let x = sv.exploitability();
    Ok((ms, x / g.pot.max(1) as f64 * 100.0, g.decisions(), sv.size()))
}

#[cfg(test)]
mod tests;
