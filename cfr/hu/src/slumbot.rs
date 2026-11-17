//! Play a trained run against Slumbot through its public API
//! (https://slumbot.com, API section): heads-up NLHE with blinds 50/100 and
//! 200bb stacks that reset every hand.
//!
//!   hu slumbot --run v1c [--hands 1000] [--sessions 1]
//!
//! Slumbot chips are 50 engine chips' worth of ours (blinds 1/2), so its bet
//! sizes are rounded to the nearest engine chip (and clamped to legal) when
//! they are replayed through the rules port; the engine's own sizes go back
//! exactly. Each session is an anonymous Slumbot session played one request
//! at a time. Every finished hand is appended to
//! $HU_HOME/logs/slumbot-RUN.jsonl.

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

use crate::play::load_engine;
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
    let engine = load_engine(&run);
    let path = hu_home().join("logs").join(format!("slumbot-{run}.jsonl"));
    let out = Mutex::new(OpenOptions::new().create(true).append(true).open(&path).expect("open hand log"));
    let tally = Mutex::new(Tally { hands: 0, raw: Stat::default(), vs_baseline: Stat::default(), decisions: 0, fallbacks: 0 });
    let next = AtomicUsize::new(0);
    let t0 = Instant::now();
    log(format!("{run} vs Slumbot: {hands} hands over {sessions} session(s); hands logged to {}", path.display()));

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
        "{run} vs Slumbot: {} (vs Slumbot's baseline {}) over {} hands ({} decisions, {} fallbacks) in {:.1} min",
        t.raw.show(),
        t.vs_baseline.show(),
        t.hands,
        t.decisions,
        t.fallbacks,
        t0.elapsed().as_secs_f64() / 60.0
    ));
}
