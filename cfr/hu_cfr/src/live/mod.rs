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

pub mod safe;
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
pub use safe::SafeMode;
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
    /// Least likelihood any opponent action gives a hand when estimating
    /// the opponent's range (0 = trust the model fully). Keeps hands the
    /// model says it would never play this way, for opponents who don't
    /// play like our model.
    pub opp_floor: f32,
    /// Share of the opponent's range at each solved street's start that is
    /// "any hand" (uniform over the hands the board allows); the rest is
    /// the estimated range. Hedges against opponents who don't play like
    /// our model, whose estimated ranges come out far too narrow.
    pub opp_mix: f32,
    /// Least probability with which every opponent hand takes each action
    /// the opponent really took on the street being solved (0 = leave it
    /// to the solve). Without it the solve assumes only the hands that
    /// "should" take a line took it, and against opponents who bet and
    /// raise with anything the bot folds far too much.
    pub opp_line: f32,
    /// Estimate the opponent's range from how likely each hand is to bet or
    /// raise at all, not at the abstract size its real size translates to
    /// (the table splits its own betting range over sizes in ways that say
    /// little about another player's sizes).
    pub opp_sizes: bool,
    /// Exponent on the likelihood of each opponent action when estimating
    /// its range from the table (1 = trust it fully, smaller = trust it
    /// less; after `opp_floor`).
    pub opp_temper: f32,
    /// The turn's menu in turn solves.
    pub turn_menu: LiveMenu,
    /// Safe re-solving (`safe`): every opponent hand is held to what it
    /// would get against the table, instead of trusting the estimated range.
    /// Falls back to the estimate where the hand has left the table's tree.
    pub safe: SafeMode,
    /// How strictly safe solves hold hands to their alternative values:
    /// the weight of "every hand that would beat its alternative" next to
    /// the estimated range at the root (`Solver::set_gadget`); None = pure
    /// Resolve (no estimate at all).
    pub safe_weight: Option<f32>,
}

impl Default for LiveCfg {
    fn default() -> LiveCfg {
        LiveCfg {
            river_iters: 400,
            // ~2 s on the box with the medium turn menu, ~4% of the pot
            // from equilibrium on real 200bb spots (`hu livebench`).
            turn_iters: 150,
            parallel: true,
            // The opponent hedges, fit so the ranges predict the hands
            // Slumbot really held (it shows them after every hand) as well
            // as possible and are calibrated: `hu slumbot-diag` on one 10k
            // hand log, checked on another. Trusting the table fully
            // predicted those hands worse than "any hand" (river start: -2.8
            // nats/hand vs uniform; these: +0.33, and +0.5 after the
            // opponent's river actions).
            opp_floor: 0.02,
            opp_mix: 0.1,
            opp_line: 0.2,
            opp_sizes: true,
            opp_temper: 0.5,
            turn_menu: LiveMenu::turn_by_name("medium").unwrap(),
            // Safe re-solving against best responses to the table, with the
            // likely hands counted too (`hu livebench --check-safe` on real
            // 200bb spots: hands beat their table value by 9x less than in
            // unsafe solves on the turn, 5x less on the river, and the
            // solves do no worse against the estimated range).
            safe: SafeMode::Best,
            safe_weight: Some(2.0),
        }
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
    /// Root range sizes, in hand combos (out of 1,326): the bot's range
    /// (each hand weighted by how often it plays this way), and the
    /// effective number of hands in the opponent's estimated range.
    pub bot_range: f64,
    pub opp_range: f64,
    /// Whether the solve was safe (held every opponent hand to its value
    /// against the table) or trusted the estimated opponent range.
    pub safe: bool,
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
    pub safe: bool,
}

type HandKey = (u64, usize);
/// (live mode, street, real actions on the street, hash of everything the
/// solve depends on). The hash makes the cache content-addressed: hands
/// that share a seed (e.g. the same deals replayed against another
/// opponent) never reuse each other's solves.
type SolveKey = (usize, usize, usize, u64);

/// FNV-1a over the stacks, button, bot, board up to `street` and the
/// actions up to `upto`.
fn solve_hash(v: &View, street: Street, upto: usize) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |x: u64| {
        for b in x.to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    eat(v.stacks[0] as u64);
    eat(v.stacks[1] as u64);
    eat(v.button as u64);
    eat(v.bot as u64);
    for &c in &v.board[..board_len(street)] {
        eat(c as u64);
    }
    for a in &v.actions[..upto] {
        eat(a.seat as u64);
        eat(a.kind as u64);
        eat(a.to as u64);
    }
    h
}

/// (Σw)² / Σw²: how many hands a weighted range amounts to.
fn effective_hands(w: &[f32]) -> f64 {
    let (s, sq) = w.iter().fold((0f64, 0f64), |(s, q), &x| (s + x as f64, q + (x as f64).powi(2)));
    if sq > 0.0 {
        s * s / sq
    } else {
        0.0
    }
}

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

/// Ranges at the start of `street`, per seat, board hands zeroed, the
/// opponent's mixed with "any hand" per `opp_mix`.
fn street_ranges(e: &Engine, v: &View, s: &State, mode: LiveMode, key: HandKey, street: Street, st: &mut Stats) -> Result<[Vec<f32>; 2], String> {
    let mut r = estimated_ranges(e, v, s, mode, key, street, st)?;
    let mix = e.live_cfg.opp_mix;
    if mix > 0.0 {
        let board = &v.board[..board_len(street)];
        let mut any = vec![1f32; H];
        mask(&mut any, board);
        let opp = &mut r[1 - v.bot];
        let total: f32 = opp.iter().sum();
        let valid: f32 = any.iter().sum();
        if total > 0.0 {
            for (x, u) in opp.iter_mut().zip(&any) {
                *x = (1.0 - mix) * *x / total * valid + mix * u;
            }
        } else {
            *opp = any;
        }
    }
    Ok(r)
}

/// The ranges before mixing.
fn estimated_ranges(e: &Engine, v: &View, s: &State, mode: LiveMode, key: HandKey, street: Street, st: &mut Stats) -> Result<[Vec<f32>; 2], String> {
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
                // The opponent's real turn actions get the floor its real
                // actions get inside a solve (the last ones came after the
                // solve, so it couldn't apply it).
                let floor = if a.seat == v.bot { 0.0 } else { e.live_cfg.opp_line.max(e.live_cfg.opp_floor) };
                for (h, w) in r[a.seat].iter_mut().enumerate() {
                    *w *= sg[edge * H + h].max(floor);
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

/// Floor the opponent's real actions on the street (`LiveCfg::opp_line`).
fn floor_opponent_line(sv: &mut Solver, g: &Subgame, forced: &[HistAction], bot: usize, p: f32) {
    if p <= 0.0 {
        return;
    }
    for (j, a) in forced.iter().enumerate() {
        if a.seat != bot {
            sv.floor(g.path[j], g.path_edges[j] as usize, p);
        }
    }
}

/// The solve of `street` with its first `n` real actions taken (cached).
#[allow(clippy::too_many_arguments)]
fn solve(e: &Engine, v: &View, s: &State, mode: LiveMode, key: HandKey, street: Street, n: usize, st: &mut Stats) -> Result<Arc<Solved>, String> {
    let k = street_start(&s.actions, street);
    let sk = (mode as usize * 4 + e.live_cfg.safe as usize, street.index(), n, solve_hash(v, street, k + n));
    if let Some(x) = e.cache.get(key, sk) {
        return Ok(x);
    }
    let forced: Vec<HistAction> = v.actions[k..k + n].to_vec();
    let start = state_at_street(v, k, street)?;
    let ranges = street_ranges(e, v, s, mode, key, street, st)?;
    // The bot's own earlier actions on this street, frozen.
    let mut frozen: Vec<(usize, Vec<f32>)> = Vec::new();
    let mut prev_recs: Vec<Vec<(hu_core::rules::ActionType, i32)>> = Vec::new();
    for (j, a) in forced.iter().enumerate() {
        if a.seat == v.bot {
            let prev = solve(e, v, s, mode, key, street, j, st)?;
            let node = prev.g.path[j];
            frozen.push((j, prev.strat.get(&node).ok_or("no strategy at an earlier decision")?.clone()));
            prev_recs.push(prev.g.records(&prev.g.nodes[node as usize]).to_vec());
        }
    }
    let menu = if street == Street::River { LiveMenu::river() } else { e.live_cfg.turn_menu.clone() };
    let g = subgame::build(&start, &forced, &menu, &LiveMenu::river_in_turn())?;
    let mut fz = Vec::with_capacity(frozen.len());
    for (i, (j, sg)) in frozen.into_iter().enumerate() {
        let node = g.path[j];
        if g.nodes[node as usize].n as usize * H != sg.len() {
            return Err(format!(
                "the bot's decision {j} on this street has a different set of options than when it was made: {:?} then, {:?} now (real actions {:?})",
                prev_recs[i],
                g.records(&g.nodes[node as usize]),
                forced.iter().map(|a| (a.seat, a.kind, a.to)).collect::<Vec<_>>()
            ));
        }
        fz.push((node, sg));
    }
    let iters = if street == Street::River { e.live_cfg.river_iters } else { e.live_cfg.turn_iters };
    let t0 = Instant::now();
    let board = &v.board[..board_len(street)];
    let (strat, ranges, exploitability, safe) = {
        let mut sv = Solver::new(&g, board, ranges, fz, e.live_cfg.parallel);
        floor_opponent_line(&mut sv, &g, &forced, v.bot, e.live_cfg.opp_line);
        if e.live_cfg.safe != SafeMode::Off {
            let alt = safe::table_values(e, v, &s.actions, k, &sv.ranges[v.bot], sv.boards(), sv.river_cards(), e.live_cfg.safe == SafeMode::Best, g.pot, e.live_cfg.parallel);
            if let Some(alt) = alt {
                sv.set_gadget(1 - v.bot, alt, e.live_cfg.safe_weight);
            }
        }
        sv.run(iters);
        let safe = sv.is_safe();
        let x = sv.exploitability();
        let mut strat = HashMap::new();
        for (i, nd) in g.nodes.iter().enumerate() {
            if matches!(nd.kind, Kind::Decision(_)) && !nd.river {
                strat.insert(i as u32, sv.strategy(i as u32));
            }
        }
        (strat, sv.ranges.clone(), x, safe)
    };
    let ms = t0.elapsed().as_millis() as u64;
    st.solves += 1;
    st.ms += ms;
    let solved = Arc::new(Solved { g, strat, ranges, iterations: iters, ms, exploitability, safe });
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
        opp_range: effective_hands(&solved.ranges[1 - v.bot]),
        safe: solved.safe,
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
    let menu = if street == Street::River { LiveMenu::river() } else { e.live_cfg.turn_menu.clone() };
    let g = subgame::build(&start, &v.actions[k..k + n], &menu, &LiveMenu::river_in_turn())?;
    let fz = fz.into_iter().map(|(j, sg)| (g.path[j], sg)).collect();
    let t0 = Instant::now();
    let mut sv = Solver::new(&g, &v.board[..board_len(street)], ranges, fz, e.live_cfg.parallel);
    floor_opponent_line(&mut sv, &g, &v.actions[k..k + n], v.bot, e.live_cfg.opp_line);
    if e.live_cfg.safe != SafeMode::Off {
        let alt = safe::table_values(e, v, &s.actions, k, &sv.ranges[v.bot], sv.boards(), sv.river_cards(), e.live_cfg.safe == SafeMode::Best, g.pot, e.live_cfg.parallel);
        if let Some(alt) = alt {
            sv.set_gadget(1 - v.bot, alt, e.live_cfg.safe_weight);
        }
    }
    sv.run(iters);
    let ms = t0.elapsed().as_millis() as u64;
    let x = sv.exploitability();
    Ok((ms, x / g.pot.max(1) as f64 * 100.0, g.decisions(), sv.size()))
}

/// What `safe_check` measured on one spot.
#[derive(Debug, Default)]
pub struct SafeCheck {
    /// Time to compute both kinds of alternative values.
    pub alt_ms: u64,
    /// Share of hands whose best-response value against the table is below
    /// its value playing the table (should be 0).
    pub best_below_table: f64,
    /// How much the opponent's hands, every hand the board allows counted
    /// alike, get above their best against the table: against an unsafe
    /// solve and a safe one (`Best` values), in % of the pot per hand pair.
    pub gain_unsafe: f64,
    pub gain_safe: f64,
    /// Share of those hands the safe solve's gadget enters with.
    pub enter: f64,
    /// What a best-responding opponent with the estimated range wins
    /// against each solve, in % of the pot per hand pair (the unsafe solve
    /// minimizes this).
    pub est_unsafe: f64,
    pub est_safe: f64,
}

/// Safe vs unsafe solves of the bot's current decision with `iters`
/// iterations (outside the cache).
pub fn safe_check(e: &Engine, v: &View, mode: LiveMode, iters: u32) -> Result<SafeCheck, String> {
    let s = crate::engine::replay(v)?;
    let street = s.street;
    let k = street_start(&s.actions, street);
    let n = s.actions.len() - k;
    let mut st = Stats::default();
    let key = (v.seed ^ 0x5AFE_0000_0000_0000, v.bot);
    let ranges = street_ranges(e, v, &s, mode, key, street, &mut st)?;
    let mut fz = Vec::new();
    for (j, a) in v.actions[k..k + n].iter().enumerate() {
        if a.seat == v.bot {
            let prev = solve(e, v, &s, mode, key, street, j, &mut st)?;
            fz.push((j, prev.strat[&prev.g.path[j]].clone()));
        }
    }
    let start = state_at_street(v, k, street)?;
    let menu = if street == Street::River { LiveMenu::river() } else { e.live_cfg.turn_menu.clone() };
    let g = subgame::build(&start, &v.actions[k..k + n], &menu, &LiveMenu::river_in_turn())?;
    let fz: Vec<(u32, Vec<f32>)> = fz.into_iter().map(|(j, sg)| (g.path[j], sg)).collect();
    let board = &v.board[..board_len(street)];
    let opp = 1 - v.bot;
    let new_solver = || {
        let mut sv = Solver::new(&g, board, ranges.clone(), fz.clone(), e.live_cfg.parallel);
        floor_opponent_line(&mut sv, &g, &v.actions[k..k + n], v.bot, e.live_cfg.opp_line);
        sv
    };
    let mut un = new_solver();
    let rb = un.ranges[v.bot].clone();
    let t0 = Instant::now();
    let par = e.live_cfg.parallel;
    let alt_t = safe::table_values(e, v, &s.actions, k, &rb, un.boards(), un.river_cards(), false, g.pot, par).ok_or("the hand left the table's tree")?;
    let alt_b = safe::table_values(e, v, &s.actions, k, &rb, un.boards(), un.river_cards(), true, g.pot, par).ok_or("the hand left the table's tree")?;
    let alt_ms = t0.elapsed().as_millis() as u64;
    let hs = showdown::holes();
    let valid: Vec<usize> = (0..H).filter(|&h| !board.contains(&hs[h][0]) && !board.contains(&hs[h][1])).collect();
    let below = valid.iter().filter(|&&h| alt_b[h] < alt_t[h] - 1e-3 * alt_t[h].abs().max(1.0)).count();
    let bot_mass: f64 = rb.iter().map(|&x| x as f64).sum();
    let gain = |br: &[f32]| -> f64 {
        let total: f64 = valid.iter().map(|&h| (br[h] - alt_b[h]).max(0.0) as f64).sum();
        total / valid.len() as f64 / bot_mass.max(1e-9) / g.pot.max(1) as f64 * 100.0
    };
    let est = un.ranges[opp].clone();
    let pairs = showdown::pair_weight(&est, &rb).max(1e-9);
    let est_value = |br: &[f32]| -> f64 { est.iter().zip(br).map(|(&r, &x)| r as f64 * x as f64).sum::<f64>() / pairs / g.pot.max(1) as f64 * 100.0 };
    un.run(iters);
    let br = un.best_response_values(opp);
    let (gain_unsafe, est_unsafe) = (gain(&br), est_value(&br));
    let mut sf = new_solver();
    let alt = if e.live_cfg.safe == SafeMode::Table { alt_t } else { alt_b.clone() };
    sf.set_gadget(opp, alt, e.live_cfg.safe_weight);
    sf.run(iters);
    let br = sf.best_response_values(opp);
    let (gain_safe, est_safe) = (gain(&br), est_value(&br));
    let enter = sf.gadget_enter().unwrap();
    Ok(SafeCheck {
        alt_ms,
        best_below_table: below as f64 / valid.len() as f64,
        gain_unsafe,
        gain_safe,
        enter: valid.iter().map(|&h| enter[h] as f64).sum::<f64>() / valid.len() as f64,
        est_unsafe,
        est_safe,
    })
}

#[cfg(test)]
mod tests;
