//! Snapshot evaluation: bb/100 against earlier snapshots (the rating ladder in
//! <run-dir>/pool.json) plus the random bot and the scripted panel (reported
//! separately).
//!
//! A match is duplicate heads-up: each deal (stacks from the training sampler,
//! button, deck) is played twice, with the candidate in each seat, so both
//! players see the same cards from both sides. Results are all-in-EV adjusted
//! and neural agents sample from the policy, as the browser bot does. The
//! result is the candidate's bb/100 with a 95% CI from the per-deal variance.
//!
//! The ladder fits ratings to the snapshot-vs-snapshot bb/100 results by least
//! squares (rating(a) - rating(b) ~ bb/100 of a vs b), with the first snapshot
//! pinned at 0 — the poker stand-in for fox-lite's Bradley-Terry Elo. Results
//! against random and the panel stay out of the fit (a sound, tight policy
//! wins *less* against random maniacs than a loose one does) and are kept per
//! model as `vs_random` / `vs_panel`. Each new snapshot plays the previous
//! snapshot, the top `--n-top` rated snapshots, and `--n-anchors` snapshots
//! spread across the rating range (fox-lite's opponent selection).
//!
//!   hu eval --run-dir DIR --candidate snap.safetensors [--deals 30000]
//!   hu match --a snap.safetensors|random|station|... --b ... [--deals 10000]
//!   hu best --run-dir DIR

use std::collections::HashMap;
use std::path::Path;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde_json::{json, Value};
use tch::nn::VarStore;
use tch::{Device, Kind, Tensor};

use hu_core::actions::{to_action, NUM_TYPES};
use hu_core::allin_ev::{expected_stacks, is_runout};
use hu_core::cards::full_deck;
use hu_core::encode::{encode_into, INPUT_SIZE};
use hu_core::rules::{Phase, State, BIG_BLIND, SEATS};
use hu_core::table::sample_stacks;

use crate::bots::{Bot, PANEL, RANDOM};
use crate::net::HuNet;
use crate::selfplay::sample_policy;
use crate::util::{flag, has_flag, num};

pub enum Agent {
    // The VarStore owns the weights the net uses.
    Net(#[allow(dead_code)] VarStore, HuNet),
    Bot(Bot),
}

pub fn load_agent(spec: &str, dev: Device) -> Agent {
    if let Some(b) = Bot::parse(spec) {
        return Agent::Bot(b);
    }
    let mut vs = VarStore::new(dev);
    let net = HuNet::new(&vs.root());
    vs.load(spec).unwrap_or_else(|e| panic!("load {spec}: {e}"));
    Agent::Net(vs, net)
}

fn is_fixed(name: &str) -> bool {
    name == RANDOM || PANEL.contains(&name)
}

struct Game {
    st: State,
    cand: usize,
    start: i32,
    result: Option<f64>, // candidate's chip result (EV-adjusted)
}

impl Game {
    /// Advance to the next decision (true) or the end of the hand (false).
    fn advance(&mut self, rng: &mut StdRng) -> bool {
        loop {
            match self.st.phase {
                Phase::Betting => return true,
                Phase::HandOver | Phase::MatchOver => {
                    self.result = Some((self.st.players[self.cand].stack - self.start) as f64);
                    return false;
                }
                _ if is_runout(&self.st) => {
                    let ev = expected_stacks(&self.st, rng);
                    self.result = Some(ev[self.cand] - self.start as f64);
                    return false;
                }
                _ => self.st.advance_street(),
            }
        }
    }
}

/// Candidate `a` vs `b`, `deals` duplicate deals. Returns (bb/100 for a, 95%
/// CI half-width).
pub fn play_match(a: &Agent, b: &Agent, deals: usize, seed: u64, dev: Device) -> (f64, f64) {
    const CHUNK: usize = 8192;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut deal_sums = vec![0f64; deals];
    let total = deals * SEATS;
    let mut next = 0;
    let mut x = vec![0f32; INPUT_SIZE];
    while next < total {
        let n = CHUNK.min(total - next);
        let mut games: Vec<Game> = (next..next + n)
            .map(|h| {
                let (d, cand) = (h / SEATS, h % SEATS);
                let mut drng = StdRng::seed_from_u64(seed ^ (d as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let stacks = sample_stacks(&mut drng);
                let button = drng.gen_range(0..SEATS as i32);
                let mut deck = full_deck();
                deck.shuffle(&mut drng);
                let mut st = State::new_table(stacks);
                st.button = button; // start_hand moves it on one seat
                let start = stacks[cand];
                st.start_hand(deck);
                Game { st, cand, start, result: None }
            })
            .collect();
        loop {
            let live: Vec<usize> = (0..n).filter(|&i| games[i].result.is_none() && games[i].advance(&mut rng)).collect();
            if live.is_empty() {
                break;
            }
            for (agent, is_cand) in [(a, true), (b, false)] {
                let mine: Vec<usize> =
                    live.iter().copied().filter(|&i| (games[i].st.to_act as usize == games[i].cand) == is_cand).collect();
                if mine.is_empty() {
                    continue;
                }
                match agent {
                    Agent::Bot(bot) => {
                        for &i in &mine {
                            let act = bot.act(&games[i].st, &mut rng);
                            games[i].st.apply(act);
                        }
                    }
                    Agent::Net(_, net) => {
                        let mut buf = Vec::with_capacity(mine.len() * INPUT_SIZE);
                        let mut spaces = Vec::with_capacity(mine.len());
                        for &i in &mine {
                            spaces.push(encode_into(&games[i].st, &mut x).expect("decision"));
                            buf.extend_from_slice(&x);
                        }
                        let xs = Tensor::from_slice(&buf).view([mine.len() as i64, INPUT_SIZE as i64]).to_device(dev);
                        let (tl, ab) = tch::no_grad(|| net.policy(&xs));
                        let to_vec = |t: &Tensor| {
                            Vec::<f32>::try_from(&t.to_kind(Kind::Float).to_device(Device::Cpu).contiguous().view([-1]))
                                .unwrap()
                        };
                        let (tl, ab) = (to_vec(&tl), to_vec(&ab));
                        for (k, &i) in mine.iter().enumerate() {
                            let sp = &spaces[k];
                            let (ty, u, _, _) =
                                sample_policy(sp, &tl[k * NUM_TYPES..(k + 1) * NUM_TYPES], &ab[k * 2..k * 2 + 2], &mut rng);
                            games[i].st.apply(to_action(sp, ty, u));
                        }
                    }
                }
            }
        }
        for (k, g) in games.iter().enumerate() {
            deal_sums[(next + k) / SEATS] += g.result.unwrap();
        }
        next += n;
    }
    // Per-deal mean result per hand, in big blinds; then bb/100.
    let per: Vec<f64> = deal_sums.iter().map(|s| s / SEATS as f64 / BIG_BLIND as f64).collect();
    let mean = per.iter().sum::<f64>() / deals as f64;
    let var = per.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (deals.max(2) - 1) as f64;
    (mean * 100.0, 1.96 * (var / deals as f64).sqrt() * 100.0)
}

// ---- pool / ladder ---------------------------------------------------------------

fn load_pool(path: &Path) -> Value {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| json!({"models": {}, "results": []}))
}

/// Least-squares ratings from snapshot-vs-snapshot bb/100 (Gauss-Seidel), with
/// the earliest snapshot pinned at 0. Results against the fixed bots go to
/// `vs_random` / `vs_panel` instead.
fn refit(pool: &mut Value) {
    let mut names: Vec<String> = pool["models"].as_object().unwrap().keys().cloned().collect();
    names.sort(); // snap_hHHHHH_cCCCCCCC: chronological
    let all: Vec<(String, String, f64)> = pool["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| (r["a"].as_str().unwrap().to_string(), r["b"].as_str().unwrap().to_string(), r["bb100"].as_f64().unwrap()))
        .collect();
    let mut results: Vec<(String, String, f64)> = Vec::new();
    for (a, b, y) in all {
        if b == RANDOM {
            pool["models"][&a]["vs_random"] = json!(y);
        } else if PANEL.contains(&b.as_str()) {
            pool["models"][&a]["vs_panel"][&b] = json!(y);
        } else {
            results.push((a, b, y));
        }
    }
    let mut rating: HashMap<String, f64> = names.iter().map(|n| (n.clone(), 0.0)).collect();
    for _ in 0..500 {
        for n in names.iter().skip(1) {
            let (mut num, mut den) = (0.0, 0.0);
            for (a, b, y) in &results {
                if a == n && rating.contains_key(b) {
                    num += rating[b] + y;
                    den += 1.0;
                } else if b == n && rating.contains_key(a) {
                    num += rating[a] - y;
                    den += 1.0;
                }
            }
            if den > 0.0 {
                rating.insert(n.clone(), num / den);
            }
        }
    }
    for n in &names {
        pool["models"][n]["rating"] = json!((rating[n] * 10.0).round() / 10.0);
    }
}

/// Evaluate a new snapshot and update pool.json. The snapshots named in `must`
/// (e.g. the previous one) are always among the opponents. Returns the
/// candidate's (bb/100, 95% CI) against every opponent played.
pub fn evaluate(
    run_dir: &str,
    candidate: &Path,
    must: &[String],
    deals: usize,
    n_top: usize,
    n_anchors: usize,
    seed: u64,
    dev: Device,
) -> HashMap<String, (f64, f64)> {
    let pool_path = Path::new(run_dir).join("pool.json");
    let mut pool = load_pool(&pool_path);
    let name = candidate.file_stem().unwrap().to_string_lossy().into_owned();

    // Snapshot opponents: `must`, the top-rated, and anchors spread over the range.
    let mut rated: Vec<(String, f64)> = pool["models"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(k, _)| **k != name)
        .map(|(k, v)| (k.clone(), v["rating"].as_f64().unwrap_or(0.0)))
        .collect();
    rated.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut opps: Vec<String> = Vec::new();
    for p in must {
        if pool["models"].get(p).is_some() && !opps.contains(p) {
            opps.push(p.clone());
        }
    }
    for r in rated.iter().take(n_top) {
        if !opps.contains(&r.0) {
            opps.push(r.0.clone());
        }
    }
    if let Some(top) = rated.first().map(|r| r.1) {
        for k in 1..=n_anchors {
            let target = top * k as f64 / (n_anchors + 1) as f64;
            if let Some(best) = rated
                .iter()
                .filter(|r| !opps.contains(&r.0))
                .min_by(|a, b| (a.1 - target).abs().partial_cmp(&(b.1 - target).abs()).unwrap())
            {
                opps.push(best.0.clone());
            }
        }
    }
    // The fixed bots, every time.
    opps.push(RANDOM.to_string());
    opps.extend(PANEL.iter().map(|s| s.to_string()));

    let mut results = HashMap::new();
    let cand = load_agent(&candidate.to_string_lossy(), dev);
    pool["models"][&name] = json!({"rating": 0.0, "path": candidate.to_string_lossy()});
    for o in &opps {
        let spec = if is_fixed(o) { o.clone() } else { pool["models"][o]["path"].as_str().unwrap().to_string() };
        let opp = load_agent(&spec, dev);
        let t = std::time::Instant::now();
        let (bb, ci) = play_match(&cand, &opp, deals, seed, dev);
        println!(
            "[eval] {name} vs {o}: {bb:+.1} ± {ci:.1} bb/100 ({} hands, {:.1}s)",
            deals * SEATS,
            t.elapsed().as_secs_f64()
        );
        results.insert(o.clone(), (bb, ci));
        pool["results"].as_array_mut().unwrap().push(json!({
            "a": name, "b": o, "bb100": (bb * 10.0).round() / 10.0, "ci": (ci * 10.0).round() / 10.0, "deals": deals,
        }));
    }
    refit(&mut pool);
    let mut ladder: Vec<(String, f64)> =
        pool["models"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v["rating"].as_f64().unwrap())).collect();
    ladder.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top: Vec<String> = ladder.iter().take(5).map(|(k, r)| format!("{k}={r:+.1}")).collect();
    println!("[eval] ladder: {}", top.join("  "));
    let m = &pool["models"][&name];
    println!("[eval] {name} rating={} vs_random={} vs_panel={}", m["rating"], m["vs_random"], m["vs_panel"]);
    std::fs::write(&pool_path, serde_json::to_string_pretty(&pool).unwrap()).expect("write pool.json");
    results
}

pub fn device() -> Device {
    if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    }
}

/// Called by the training daemon after each snapshot (see `evaluate`).
pub fn after_snapshot(run_dir: &str, snapshot: &Path, must: &[String], args: &[String]) {
    if has_flag(args, "--no-eval") {
        return;
    }
    let seed = snapshot.to_string_lossy().bytes().fold(7u64, |h, b| h.wrapping_mul(31).wrapping_add(b as u64));
    evaluate(
        run_dir,
        snapshot,
        must,
        num(args, "--eval-deals", 30_000),
        num(args, "--n-top", 2),
        num(args, "--n-anchors", 3),
        seed,
        device(),
    );
}

/// `hu eval`
pub fn eval_cmd(args: &[String]) {
    let run_dir = flag(args, "--run-dir").expect("--run-dir");
    let cand = flag(args, "--candidate").expect("--candidate snap.safetensors");
    evaluate(
        &run_dir,
        Path::new(&cand),
        &[],
        num(args, "--deals", 30_000),
        num(args, "--n-top", 2),
        num(args, "--n-anchors", 3),
        num(args, "--seed", 1),
        device(),
    );
}

/// `hu match`
pub fn match_cmd(args: &[String]) {
    let dev = device();
    let a = flag(args, "--a").expect("--a");
    let b = flag(args, "--b").expect("--b");
    let deals: usize = num(args, "--deals", 10_000);
    let t = std::time::Instant::now();
    let (bb, ci) = play_match(&load_agent(&a, dev), &load_agent(&b, dev), deals, num(args, "--seed", 1), dev);
    println!(
        "{a} vs {b}: {bb:+.1} ± {ci:.1} bb/100 ({} hands, {:.1}s)",
        deals * SEATS,
        t.elapsed().as_secs_f64()
    );
}

/// `hu best --run-dir DIR`: print the path of the top-rated snapshot.
pub fn best_cmd(args: &[String]) {
    let run_dir = flag(args, "--run-dir").expect("--run-dir");
    let mut pool = load_pool(&Path::new(&run_dir).join("pool.json"));
    refit(&mut pool);
    let best = pool["models"]
        .as_object()
        .unwrap()
        .iter()
        .max_by(|a, b| a.1["rating"].as_f64().partial_cmp(&b.1["rating"].as_f64()).unwrap());
    match best {
        Some((_, v)) => println!("{}", v["path"].as_str().unwrap()),
        None => {
            eprintln!("no rated snapshots in {run_dir}/pool.json yet");
            std::process::exit(1);
        }
    }
}
