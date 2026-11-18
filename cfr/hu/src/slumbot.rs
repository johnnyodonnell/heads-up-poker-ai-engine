//! Play a trained run against Slumbot through its public API
//! (https://slumbot.com, API section): heads-up NLHE with blinds 50/100 and
//! 200bb stacks that reset every hand.
//!
//!   hu slumbot --run v1c [--hands 1000] [--sessions 1] [--live river] [--tag NAME]
//!
//! Slumbot chips are 50 engine chips' worth of ours (blinds 1/2), so its bet
//! sizes are rounded to the nearest engine chip (and clamped to legal) when
//! they are replayed through the rules port; the engine's own sizes go back
//! exactly. Each session is an anonymous Slumbot session played one request
//! at a time. Every finished hand is appended to
//! $HU_HOME/logs/slumbot-TAG.jsonl (TAG defaults to RUN, or RUN-live-MODE).
//!
//!   hu slumbot-diag --run v1c --log FILE [--street turn|river] [live flags]
//!
//! checks a log's live decisions against the hands Slumbot really held (it
//! shows its cards after every hand): how much weight the opponent ranges the
//! solves assumed gave them. `--table-only` (with `--street flop`) checks the
//! table's range estimates at each street's start instead, without solving;
//! `--selfplay N` does that for the table playing itself, where they should
//! be calibrated.

use std::fs::OpenOptions;
use std::io::Write;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde_json::{json, Value};

use hu_cfr::engine::{Engine, HistAction, View};
use hu_core::cards::{parse, Card};
use hu_core::rules::{Action, ActionType, Phase, State};

use crate::play::{live_flags, load_engine};
use crate::util::{flag, hu_home, log, num};

const API: &str = "https://slumbot.com/slumbot/api/";
/// Slumbot chips per engine chip.
const SCALE: i32 = 50;
const STACK: i32 = 20_000;
const BIG_BLIND: i32 = 100;

fn post(endpoint: &str, body: &Value) -> Result<Value, String> {
    let mut last = String::new();
    for attempt in 0..6 {
        if attempt > 0 {
            std::thread::sleep(Duration::from_secs(1 << attempt));
        }
        let out = Command::new("curl")
            .args(["-sS", "--max-time", "30", "-X", "POST", "-H", "Content-Type: application/json", "-d"])
            .arg(body.to_string())
            .arg(format!("{API}{endpoint}"))
            .output();
        match out {
            Ok(o) if o.status.success() => match serde_json::from_slice::<Value>(&o.stdout) {
                Ok(v) => {
                    if let Some(e) = v.get("error_msg") {
                        return Err(format!("slumbot {endpoint}: {e}"));
                    }
                    return Ok(v);
                }
                Err(e) => last = format!("bad JSON ({e}): {}", String::from_utf8_lossy(&o.stdout)),
            },
            Ok(o) => last = String::from_utf8_lossy(&o.stderr).trim().to_string(),
            Err(e) => last = e.to_string(),
        }
    }
    Err(format!("slumbot {endpoint}: {last}"))
}

/// Slumbot writes ten as "T"; the rules port as "10".
fn card(s: &str) -> Result<Card, String> {
    parse(&s.replacen('T', "10", 1)).ok_or_else(|| format!("bad card {s:?}"))
}

fn cards(v: &Value, key: &str) -> Result<Vec<Card>, String> {
    v.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("no {key}"))?
        .iter()
        .map(|c| card(c.as_str().unwrap_or("")))
        .collect()
}

/// Slumbot's betting state on the current street, in its chips
/// (as its sample client's ParseAction tracks it).
#[derive(Clone, Copy, Debug)]
struct Street {
    /// Chips the last bettor has put in on this street.
    bet_to: i32,
    /// Chips the last bettor has put in over the whole hand.
    total: i32,
    /// Size of the last bet or raise (0 once called or checked).
    last_size: i32,
}

/// Replays Slumbot's action string through the rules port with the engine
/// at seat 0. Returns the state, the recorded actions and Slumbot's view of
/// the current street.
fn replay(action: &str, button: usize) -> Result<(State, Vec<HistAction>, Street), String> {
    let mut s = State::new_table([STACK / SCALE; 2]);
    s.button = (1 - button) as i32;
    s.start_hand((0..52).collect());
    let mut st = Street { bet_to: BIG_BLIND, total: BIG_BLIND, last_size: BIG_BLIND / 2 };
    let mut hist = Vec::new();
    let b = action.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        i += 1;
        if c == '/' {
            continue;
        }
        while s.phase == Phase::StreetComplete {
            s.advance_street();
        }
        let legal = s.legal().ok_or_else(|| format!("{action:?}: no one to act at {}", i - 1))?;
        let act = match c {
            'f' => Action::new(ActionType::Fold),
            'k' => Action::new(ActionType::Check),
            'c' => {
                st.last_size = 0;
                Action::new(ActionType::Call)
            }
            'b' => {
                let j = i;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
                let to: i32 = action[j..i].parse().map_err(|_| format!("{action:?}: bad bet size"))?;
                let size = to - st.bet_to;
                st.total += size;
                st.bet_to = to;
                st.last_size = size;
                let ours = ((to as f64) / SCALE as f64).round() as i32;
                if st.total >= STACK || ours >= legal.max_raise_to {
                    Action::new(ActionType::AllIn)
                } else {
                    let ours = ours.clamp(legal.min_raise_to, legal.max_raise_to);
                    Action::to(if legal.opening { ActionType::Bet } else { ActionType::Raise }, ours)
                }
            }
            other => return Err(format!("{action:?}: unexpected {other:?}")),
        };
        let street = s.street;
        s.apply(act);
        let rec = s.actions.last().ok_or("action not recorded")?;
        hist.push(HistAction { seat: rec.seat, kind: rec.kind, to: rec.to });
        if s.phase != Phase::Betting || s.street != street {
            st.bet_to = 0; // the street is over
        }
    }
    while s.phase == Phase::StreetComplete && !s.hand_done() {
        s.advance_street();
    }
    Ok((s, hist, st))
}

/// The engine's action as a Slumbot increment.
fn incr(a: &Action, st: &Street) -> String {
    let remaining = STACK - st.total;
    match a.kind {
        ActionType::Fold => "f".into(),
        ActionType::Check => "k".into(),
        ActionType::Call => "c".into(),
        ActionType::AllIn => format!("b{}", st.bet_to + remaining),
        ActionType::Bet | ActionType::Raise => {
            let min = st.last_size.max(BIG_BLIND).min(remaining);
            let to = (a.amount.unwrap_or(0) * SCALE).clamp(st.bet_to + min, st.bet_to + remaining);
            format!("b{to}")
        }
    }
}

/// Mean and 95% interval (bb/100) of per-hand results in bb.
#[derive(Default)]
struct Stat {
    n: f64,
    sum: f64,
    sum_sq: f64,
}

impl Stat {
    fn add(&mut self, x: f64) {
        self.n += 1.0;
        self.sum += x;
        self.sum_sq += x * x;
    }
    fn show(&self) -> String {
        let n = self.n.max(2.0);
        let mean = self.sum / n;
        let sd = ((self.sum_sq / n - mean * mean).max(0.0) * n / (n - 1.0)).sqrt();
        format!("{:+.1} ± {:.1} bb/100", mean * 100.0, 1.96 * sd / n.sqrt() * 100.0)
    }
}

struct Tally {
    hands: usize,
    /// What the engine won.
    raw: Stat,
    /// What it won minus what Slumbot would have won with the same cards in
    /// the same seat (Slumbot's `baseline_winnings`): much less luck.
    vs_baseline: Stat,
    decisions: usize,
    fallbacks: usize,
}

/// Plays one hand; returns (winnings in Slumbot chips, the final response).
fn play_hand(engine: &Engine, token: &mut Option<String>, rng: &mut SmallRng, tally: &Mutex<Tally>) -> Result<(i64, Value), String> {
    let body = match token {
        Some(t) => json!({ "token": t }),
        None => json!({}),
    };
    let mut r = post("new_hand", &body)?;
    let seed: u64 = rng.gen();
    loop {
        if let Some(t) = r.get("token").and_then(Value::as_str) {
            *token = Some(t.to_string());
        }
        if let Some(w) = r.get("winnings").and_then(Value::as_i64) {
            return Ok((w, r));
        }
        let action = r.get("action").and_then(Value::as_str).unwrap_or("");
        // client_pos 1: we are the small blind, i.e. the button.
        let button = if r.get("client_pos").and_then(Value::as_i64) == Some(1) { 0 } else { 1 };
        let (s, hist, st) = replay(action, button)?;
        if s.to_act != 0 {
            return Err(format!("{action:?}: replayed with seat {} to act, not the engine", s.to_act));
        }
        let hole = cards(&r, "hole_cards")?;
        let view = View {
            bot: 0,
            button,
            stacks: [STACK / SCALE; 2],
            hole: [hole[0], hole[1]],
            board: cards(&r, "board")?,
            actions: hist,
            seed,
        };
        let d = engine.decide(&view, rng).map_err(|e| format!("{action:?}: {e}"))?;
        {
            let mut t = tally.lock().unwrap();
            t.decisions += 1;
            t.fallbacks += usize::from(d.debug.fallback.is_some());
        }
        let inc = incr(&d.action, &st);
        r = post("act", &json!({ "token": token.as_deref().unwrap_or(""), "incr": inc }))?;
    }
}

pub fn slumbot_cmd(args: &[String]) {
    let run = flag(args, "--run").expect("--run NAME");
    let hands: usize = num(args, "--hands", 1000);
    let sessions: usize = num(args, "--sessions", 1);
    let mut engine = load_engine(&run);
    live_flags(args, &mut engine);
    let tag = flag(args, "--tag").unwrap_or_else(|| if engine.live == hu_cfr::live::LiveMode::Off { run.clone() } else { format!("{run}-live-{}", engine.live.name()) });
    let path = hu_home().join("logs").join(format!("slumbot-{tag}.jsonl"));
    let out = Mutex::new(OpenOptions::new().create(true).append(true).open(&path).expect("open hand log"));
    let tally = Mutex::new(Tally { hands: 0, raw: Stat::default(), vs_baseline: Stat::default(), decisions: 0, fallbacks: 0 });
    let next = AtomicUsize::new(0);
    let t0 = Instant::now();
    log(format!("{tag} vs Slumbot: {hands} hands over {sessions} session(s); hands logged to {}", path.display()));

    std::thread::scope(|sc| {
        for k in 0..sessions {
            let (engine, out, tally, next) = (&engine, &out, &tally, &next);
            sc.spawn(move || {
                let mut rng = SmallRng::from_entropy();
                let mut token: Option<String> = None;
                let mut errors = 0;
                while next.fetch_add(1, Ordering::Relaxed) < hands {
                    match play_hand(engine, &mut token, &mut rng, tally) {
                        Ok((w, r)) => {
                            errors = 0;
                            writeln!(out.lock().unwrap(), "{}", json!({ "session": k, "winnings": w, "final": r })).ok();
                            let mut t = tally.lock().unwrap();
                            t.hands += 1;
                            t.raw.add(w as f64 / BIG_BLIND as f64);
                            if let Some(b) = r.get("baseline_winnings").and_then(Value::as_i64) {
                                t.vs_baseline.add((w - b) as f64 / BIG_BLIND as f64);
                            }
                            if t.hands % 100 == 0 {
                                log(format!(
                                    "{} hands: {} (vs Slumbot's baseline {})  {} decisions, {} fallbacks, {:.2} s/hand",
                                    t.hands,
                                    t.raw.show(),
                                    t.vs_baseline.show(),
                                    t.decisions,
                                    t.fallbacks,
                                    t0.elapsed().as_secs_f64() / t.hands as f64 * sessions as f64
                                ));
                            }
                        }
                        Err(e) => {
                            log(format!("session {k}: hand failed: {e}"));
                            errors += 1;
                            token = None; // start a fresh session
                            if errors >= 5 {
                                log(format!("session {k}: giving up after 5 failed hands in a row"));
                                return;
                            }
                        }
                    }
                }
            });
        }
    });
    let t = tally.lock().unwrap();
    log(format!(
        "{tag} vs Slumbot: {} (vs Slumbot's baseline {}) over {} hands ({} decisions, {} fallbacks) in {:.1} min",
        t.raw.show(),
        t.vs_baseline.show(),
        t.hands,
        t.decisions,
        t.fallbacks,
        t0.elapsed().as_secs_f64() / 60.0
    ));
}

/// End offsets of each complete action in a Slumbot action string.
fn action_ends(action: &str) -> Vec<usize> {
    let b = action.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        i += 1;
        if c == b'/' {
            continue;
        }
        if c == b'b' {
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
        }
        out.push(i);
    }
    out
}

/// How well an estimated opponent range predicted the hand the opponent
/// really held.
#[derive(Clone, Copy, Default)]
struct Calib {
    /// The real hand's probability × the number of possible hands
    /// (1 = an average hand).
    ratio: f64,
    /// Range mass below the real hand's probability (ties half): uniform on
    /// [0, 1] when the range is calibrated.
    pit: f64,
    /// The range's own probability that the real hand's ratio falls below
    /// each of `THRESHOLDS`, and whether it did.
    expect: [f64; 3],
    below: [bool; 3],
    /// ln p(real hand) minus the range's expected ln p (0 on average when
    /// calibrated; negative when the range is overconfident).
    surprise: f64,
    /// 1 / Σp²: the range's effective number of hands.
    ess: f64,
    /// ln(p(real hand) × N): what the range knew about the real hand, in
    /// nats over "any hand" (a proper score: higher is better).
    score: f64,
}

const THRESHOLDS: [f64; 3] = [0.01, 0.1, 0.3];

fn calib(range: &[f32], dead: &[Card], real: [Card; 2]) -> Option<Calib> {
    let hs = hu_cfr::live::showdown::holes();
    let ok = |h: usize| !dead.contains(&hs[h][0]) && !dead.contains(&hs[h][1]);
    let total: f64 = (0..range.len()).filter(|&h| ok(h)).map(|h| range[h] as f64).sum();
    if total <= 0.0 {
        return None;
    }
    let p: Vec<f64> = (0..range.len()).map(|h| if ok(h) { range[h] as f64 / total } else { -1.0 }).collect();
    let n = p.iter().filter(|&&x| x >= 0.0).count() as f64;
    let a = hu_cfr::live::showdown::hand(real);
    let pa = p[a].max(0.0);
    let mut c = Calib { ratio: pa * n, ..Calib::default() };
    let mut expected_ln = 0.0;
    let mut sq = 0.0;
    for &x in p.iter().filter(|&&x| x >= 0.0) {
        if x < pa {
            c.pit += x;
        } else if x == pa {
            c.pit += 0.5 * x;
        }
        for (t, &thr) in THRESHOLDS.iter().enumerate() {
            if x * n < thr {
                c.expect[t] += x;
            }
        }
        if x > 0.0 {
            expected_ln += x * x.ln();
        }
        sq += x * x;
    }
    for (t, &thr) in THRESHOLDS.iter().enumerate() {
        c.below[t] = c.ratio < thr;
    }
    c.surprise = pa.max(1e-9).ln() - expected_ln;
    c.score = (pa * n).max(1e-9).ln();
    c.ess = 1.0 / sq;
    Some(c)
}

#[derive(Default)]
struct CalibSum {
    n: usize,
    expect: [f64; 3],
    below: [usize; 3],
    pit: f64,
    pit_low: usize,
    surprise: f64,
    ess: f64,
    score: f64,
}

impl CalibSum {
    fn add(&mut self, c: &Calib) {
        self.n += 1;
        for t in 0..3 {
            self.expect[t] += c.expect[t];
            self.below[t] += usize::from(c.below[t]);
        }
        self.pit += c.pit;
        self.pit_low += usize::from(c.pit < 0.1);
        self.surprise += c.surprise;
        self.ess += c.ess;
        self.score += c.score;
    }
    fn show(&self, name: &str) -> String {
        let n = self.n.max(1) as f64;
        let mut s = format!("{name:44} n={:5}  ", self.n);
        for (t, thr) in THRESHOLDS.iter().enumerate() {
            s += &format!("p<{thr}/N: {:4.1}% (model {:4.1}%)  ", self.below[t] as f64 / n * 100.0, self.expect[t] / n * 100.0);
        }
        s += &format!(
            "PIT mean {:.2}, <0.1: {:4.1}%  surprise {:+.2}  range ~{:.0} hands  score {:+.3}",
            self.pit / n,
            self.pit_low as f64 / n * 100.0,
            self.surprise / n,
            self.ess / n,
            self.score / n
        );
        s
    }
}

/// Mix "any hand" into a range as `live::street_ranges` does.
fn mix_any(r: &mut [f32], board: &[Card], mix: f32) {
    if mix <= 0.0 {
        return;
    }
    let mut any = vec![1f32; r.len()];
    hu_cfr::live::showdown::mask(&mut any, board);
    hu_cfr::live::showdown::mask(r, board);
    let total: f32 = r.iter().sum();
    let valid: f32 = any.iter().sum();
    for (x, u) in r.iter_mut().zip(&any) {
        *x = if total > 0.0 { (1.0 - mix) * *x / total * valid + mix * u } else { *u };
    }
}

/// `hu slumbot-diag --run NAME --log FILE [--street turn|river] [--hands N]
/// [live flags]`: replay a Slumbot hand log's live decisions on one street
/// and check the opponent ranges the solves assumed against the hands
/// Slumbot really held (it shows them after every hand).
pub fn diag_cmd(args: &[String]) {
    use hu_core::rules::Street as Rs;
    use rayon::prelude::*;
    let run = flag(args, "--run").expect("--run NAME");
    let path = flag(args, "--log").expect("--log FILE");
    let street = match flag(args, "--street").as_deref() {
        Some("flop") => Rs::Flop,
        Some("turn") => Rs::Turn,
        None | Some("river") => Rs::River,
        Some(x) => panic!("--street flop|turn|river, not {x}"),
    };
    let max: usize = num(args, "--hands", 1000);
    let show: usize = num(args, "--show", 12);
    let mut engine = load_engine(&run);
    live_flags(args, &mut engine);
    engine.live_cfg.parallel = false;
    let mode = engine.live;
    let text = std::fs::read_to_string(&path).expect("read the log");
    let hands: Vec<Value> = text
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter_map(|v| v.get("final").cloned())
        .filter(|f| f.get("board").and_then(Value::as_array).map_or(0, |b| b.len()) >= [0, 3, 4, 5][street.index()])
        .take(max)
        .collect();
    log(format!("{} hands reach the {street:?}; live {}", hands.len(), mode.name()));

    if let Some(n) = flag(args, "--selfplay") {
        // The same check on the table playing itself at Slumbot's depth,
        // where its range estimates should be right.
        use hu_core::rules::Status;
        let n: usize = n.parse().expect("--selfplay N");
        // `--opp-run NAME`: seat 1 is another run's table, so the check
        // says how far that run's play is from this run's model of it.
        let other = flag(args, "--opp-run").map(|r| load_engine(&r));
        let mut sums: Vec<CalibSum> = (0..4).map(|_| CalibSum::default()).collect();
        let mut rng = SmallRng::seed_from_u64(7);
        for i in 0..n {
            let mut deck: Vec<Card> = (0..52).collect();
            rand::seq::SliceRandom::shuffle(&mut deck[..], &mut rng);
            let stack = STACK / SCALE;
            let mut s = State::new_table([stack; 2]);
            s.button = rng.gen_range(0..2);
            s.start_hand(deck);
            let seed: u64 = rng.gen();
            loop {
                match s.phase {
                    Phase::Betting => {
                        let seat = s.to_act as usize;
                        let v = hu_cfr::engine::view_of(&s, seat, [stack; 2], seed);
                        let who = if seat == 1 { other.as_ref().unwrap_or(&engine) } else { &engine };
                        let d = who.decide_mode(&v, &mut rng, hu_cfr::live::LiveMode::Off).expect("decide");
                        s.apply(d.action);
                    }
                    Phase::StreetComplete if !hu_core::allin_ev::is_runout(&s) && s.players.iter().all(|p| p.status != Status::Folded) => s.advance_street(),
                    _ => break,
                }
            }
            let view = hu_cfr::engine::view_of(&s, 0, [stack; 2], seed);
            let opp = [s.players[1].hole[0], s.players[1].hole[1]];
            for st in [Rs::Flop, Rs::Turn, Rs::River] {
                let nb = [0, 3, 4, 5][st.index()];
                if s.community.len() < nb {
                    break;
                }
                let Some(k) = s.actions.iter().position(|a| a.street == st) else { break };
                let r = engine.table_ranges(&view, &s.actions, k);
                let mut dead = vec![view.hole[0], view.hole[1]];
                dead.extend_from_slice(&s.community[..nb]);
                if let Some(c) = calib(&r[1], &dead, [opp[0], opp[1]]) {
                    sums[st.index()].add(&c);
                }
            }
            if i % 1000 == 999 {
                log(format!("{} self-play hands", i + 1));
            }
        }
        for st in [Rs::Flop, Rs::Turn, Rs::River] {
            println!("{}", sums[st.index()].show(&format!("{st:?} start (table estimate, self-play)")));
        }
        return;
    }

    if args.iter().any(|a| a == "--table-only") {
        // The table's estimate of the opponent's range at the start of each
        // street, no solving.
        let mut sums: Vec<CalibSum> = (0..4).map(|_| CalibSum::default()).collect();
        let mut zero_from = [0usize; 4];
        for (i, f) in hands.iter().enumerate() {
            let action = f["action"].as_str().unwrap_or("");
            let button = if f["client_pos"].as_i64() == Some(1) { 0 } else { 1 };
            let (Ok(hole), Ok(opp), Ok(board), Ok((s, hist, _))) = (cards(f, "hole_cards"), cards(f, "bot_hole_cards"), cards(f, "board"), replay(action, button)) else {
                continue;
            };
            let view = View { bot: 0, button, stacks: [STACK / SCALE; 2], hole: [hole[0], hole[1]], board: board.clone(), actions: hist, seed: 0x5EED ^ i as u64 };
            let mut first_zero = None;
            for st in [Rs::Flop, Rs::Turn, Rs::River] {
                let nb = [0, 3, 4, 5][st.index()];
                if board.len() < nb {
                    break;
                }
                let Some(k) = s.actions.iter().position(|a| a.street == st) else { break };
                let mut r = engine.table_ranges(&view, &s.actions, k);
                let mut dead = vec![hole[0], hole[1]];
                dead.extend_from_slice(&board[..nb]);
                mix_any(&mut r[1], &board[..nb], engine.live_cfg.opp_mix);
                if let Some(c) = calib(&r[1], &dead, [opp[0], opp[1]]) {
                    sums[st.index()].add(&c);
                    if c.ratio == 0.0 && first_zero.is_none() {
                        first_zero = Some(st.index());
                    }
                }
            }
            if let Some(z) = first_zero {
                zero_from[z] += 1;
            }
        }
        for st in [Rs::Flop, Rs::Turn, Rs::River] {
            println!("{}", sums[st.index()].show(&format!("{st:?} start (table estimate)")));
        }
        println!("real hand first given zero weight at the start of: flop {}, turn {}, river {}", zero_from[1], zero_from[2], zero_from[3]);
        return;
    }

    struct Row {
        start: Option<Calib>,
        now: Option<Calib>,
        /// The opponent's last action this street: (added off the menu,
        /// share of its range the solve has taking it).
        last_opp: Option<(bool, f64)>,
        expl: f64,
        text: String,
    }
    let t0 = Instant::now();
    let rows: Vec<Row> = hands
        .par_iter()
        .enumerate()
        .flat_map_iter(|(i, f)| {
            let mut out = Vec::new();
            let action = f["action"].as_str().unwrap_or("").to_string();
            let button = if f["client_pos"].as_i64() == Some(1) { 0 } else { 1 };
            let (Ok(hole), Ok(opp), Ok(board)) = (cards(f, "hole_cards"), cards(f, "bot_hole_cards"), cards(f, "board")) else {
                return out.into_iter();
            };
            for end in std::iter::once(0).chain(action_ends(&action)) {
                let Ok((s, hist, _)) = replay(&action[..end], button) else { continue };
                if s.phase != Phase::Betting || s.to_act != 0 || s.street != street {
                    continue;
                }
                let nb = [0, 3, 4, 5][street.index()];
                let view = View { bot: 0, button, stacks: [STACK / SCALE; 2], hole: [hole[0], hole[1]], board: board[..nb].to_vec(), actions: hist, seed: 0x5EED ^ i as u64 };
                let ins = match hu_cfr::live::inspect(&engine, &view, mode) {
                    Ok(x) => x,
                    Err(e) => {
                        log(format!("hand {i} {:?}: {e}", &action[..end]));
                        continue;
                    }
                };
                let mut dead = vec![hole[0], hole[1]];
                dead.extend_from_slice(&board[..nb]);
                let real = [opp[0], opp[1]];
                let last_opp = ins.actions.iter().rev().find(|a| a.0 != 0).map(|a| (a.1, a.2));
                let now = calib(&ins.opp_now, &dead, real);
                let text = format!(
                    "{:<28} bot {} vs {} on {}: ratio {:.3} (start {:.3}) last opp action {} | {}",
                    &action[..end],
                    hu_cfr::engine::cards_str(&hole),
                    hu_cfr::engine::cards_str(&opp),
                    hu_cfr::engine::cards_str(&board[..nb]),
                    now.map_or(f64::NAN, |c| c.ratio),
                    calib(&ins.opp_start, &dead, real).map_or(f64::NAN, |c| c.ratio),
                    last_opp.map_or("-".into(), |(a, s)| format!("{} {:.1}% of range", if a { "inserted" } else { "on menu" }, s * 100.0)),
                    ins.options.iter().map(|(l, p)| format!("{l} {:.0}%", p * 100.0)).collect::<Vec<_>>().join(", ")
                );
                out.push(Row { start: calib(&ins.opp_start, &dead, real), now, last_opp, expl: ins.exploitability_pct, text });
            }
            out.into_iter()
        })
        .collect();
    log(format!("{} decisions in {:.0}s", rows.len(), t0.elapsed().as_secs_f64()));

    let mut start = CalibSum::default();
    let mut groups: Vec<(&str, CalibSum)> = vec![("now: no opponent action this street", CalibSum::default()), ("now: after an on-menu opponent action", CalibSum::default()), ("now: after an inserted opponent action", CalibSum::default())];
    let mut shares = [(0usize, 0f64), (0usize, 0f64)];
    let mut expl = 0.0;
    for r in &rows {
        if let Some(c) = &r.start {
            start.add(c);
        }
        let g = match r.last_opp {
            None => 0,
            Some((false, _)) => 1,
            Some((true, _)) => 2,
        };
        if let Some(c) = &r.now {
            groups[g].1.add(c);
        }
        if let Some((ins, s)) = r.last_opp {
            let k = usize::from(ins);
            shares[k].0 += 1;
            shares[k].1 += s;
        }
        expl += r.expl;
    }
    println!("{}", start.show(&format!("{street:?} start (estimated range)")));
    for (name, c) in &groups {
        println!("{}", c.show(name));
    }
    for (k, name) in ["on-menu", "inserted"].iter().enumerate() {
        println!("opponent's last {name} actions: {} of them, the solve had {:.1}% of its range taking them on average", shares[k].0, shares[k].1 / shares[k].0.max(1) as f64 * 100.0);
    }
    println!("mean exploitability of the solves: {:.2}% of the pot", expl / rows.len().max(1) as f64);
    let mut worst: Vec<&Row> = rows.iter().filter(|r| r.now.is_some()).collect();
    worst.sort_by(|a, b| a.now.unwrap().ratio.partial_cmp(&b.now.unwrap().ratio).unwrap());
    println!("\nleast expected real hands:");
    for r in worst.iter().take(show) {
        println!("  {}", r.text);
    }
}
