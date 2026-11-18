//! Loading a run's strategies into an engine; `hu eval`, `hu probe` and
//! `hu livebench`.
//!
//!   hu eval --run NAME [--vs all|station,maniac,...|run:OTHER] [--deals 20000]
//!       [--stacks full|sampled|200bb] [--seed 1] [--save DIR [--tag NAME]] [live flags]
//!   hu compare BASE.txt OTHER.txt ...   paired differences between saved
//!       matches against the same opponent (same seed, deals, stacks)
//!   hu probe --run NAME [--stack 200]   preflop charts: the small blind's first
//!       action and the big blind facing a 2.5bb open
//!   hu livebench --run NAME [--street river|turn] [--spots 20] [--iters 100,200,400]
//!       [--stack 400] [--check-safe]   time live solves of real spots and
//!       measure how close to equilibrium they get (or compare safe and
//!       unsafe solves of them)
//!
//! Live flags (eval, serve, slumbot, slumbot-diag, livebench): --live
//! off|river|turn, --river-iters N, --turn-iters N, --turn-menu
//! full|medium|lean, and the opponent-range hedges (defaults in
//! `hu_cfr::live::LiveCfg`): --opp-floor F (least likelihood of an opponent
//! action when estimating its range from the table), --opp-temper F
//! (exponent on those likelihoods), --opp-sizes on|off (a bet's likelihood is
//! betting at any size), --opp-mix F (share of "any hand" in the opponent's
//! range at a solve's start), --opp-line F (least probability of every
//! opponent hand taking the opponent's real actions on the solved street),
//! --safe off|table|best (safe re-solving: hold every opponent hand to its
//! value against the table, the opponent playing the table or best-responding),
//! --safe-weight W|resolve (how strictly: weight of the hands that would beat
//! their table value next to the estimated range; resolve = no estimate).
//! `run:OTHER` rivals always play their table only (`run:NAME` with the same
//! run: the run without live solving).

use std::path::PathBuf;
use std::time::Instant;

use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use hu_cfr::bots::{Bot, PANEL};
use hu_cfr::classes::{class_hands, N_CLASSES};
use hu_cfr::engine::{view_of, Engine};
use hu_cfr::eval::{play_match, Player, Stacks};
use hu_cfr::game::{Act, MAX_ACTIONS};
use hu_cfr::live::{self, LiveMode};
use hu_cfr::strategy::StrategyMeta;
use hu_core::allin_ev::is_runout;
use hu_core::rules::{Phase, State, Street};

use crate::util::{flag, hu_home, log, num};

/// The strategy directories of a run (strategies/NAME/s*/).
pub fn strategy_dirs(run: &str) -> Vec<PathBuf> {
    let root = hu_home().join("strategies").join(run);
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("{}: {e}", root.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("meta.json").exists() && p.join("probs.u8").exists())
        .collect();
    dirs.sort();
    dirs
}

/// Load a run's strategies (all trained on one abstraction) into an engine.
pub fn load_engine(run: &str) -> Engine {
    let dirs = strategy_dirs(run);
    assert!(!dirs.is_empty(), "no strategies for run {run}");
    let meta: StrategyMeta = serde_json::from_str(&std::fs::read_to_string(dirs[0].join("meta.json")).unwrap()).unwrap();
    let t = Instant::now();
    let e = Engine::load(&hu_home().join("data").join(&meta.abstraction), &dirs).expect("load engine");
    log(format!(
        "engine {run}: abstraction {} {:?}, stacks {:?} ({:.1}s to load)",
        meta.abstraction,
        e.abs.buckets(),
        e.strategies.iter().map(|s| s.meta.stack).collect::<Vec<_>>(),
        t.elapsed().as_secs_f64()
    ));
    e
}

/// Apply the live-solving flags to an engine.
pub fn live_flags(args: &[String], e: &mut Engine) {
    if let Some(m) = flag(args, "--live") {
        e.live = LiveMode::parse(&m).unwrap_or_else(|| panic!("--live off|river|turn, not {m}"));
    }
    e.live_cfg.river_iters = num(args, "--river-iters", e.live_cfg.river_iters);
    e.live_cfg.turn_iters = num(args, "--turn-iters", e.live_cfg.turn_iters);
    if let Some(m) = flag(args, "--turn-menu") {
        e.live_cfg.turn_menu = hu_cfr::menu::LiveMenu::turn_by_name(&m).unwrap_or_else(|| panic!("--turn-menu full|medium|lean, not {m}"));
    }
    if let Some(f) = flag(args, "--opp-line") {
        e.live_cfg.opp_line = f.parse().expect("--opp-line 0..1");
    }
    if let Some(f) = flag(args, "--opp-mix") {
        e.live_cfg.opp_mix = f.parse().expect("--opp-mix 0..1");
    }
    if let Some(f) = flag(args, "--opp-floor") {
        e.live_cfg.opp_floor = f.parse().expect("--opp-floor 0..1");
    }
    if let Some(f) = flag(args, "--opp-temper") {
        e.live_cfg.opp_temper = f.parse().expect("--opp-temper 0..1");
    }
    if let Some(m) = flag(args, "--safe") {
        e.live_cfg.safe = hu_cfr::live::SafeMode::parse(&m).unwrap_or_else(|| panic!("--safe off|table|best, not {m}"));
    }
    if let Some(w) = flag(args, "--safe-weight") {
        e.live_cfg.safe_weight = if w == "resolve" { None } else { Some(w.parse().expect("--safe-weight W|resolve")) };
    }
    if let Some(f) = flag(args, "--opp-sizes") {
        e.live_cfg.opp_sizes = match f.as_str() {
            "on" => true,
            "off" => false,
            x => panic!("--opp-sizes on|off, not {x}"),
        };
    }
}

/// "full", "sampled" or "NNbb".
fn parse_stacks(s: Option<&str>) -> Stacks {
    match s {
        None | Some("full") => Stacks::Full,
        Some("sampled") => Stacks::Sampled,
        Some(x) => match x.strip_suffix("bb").and_then(|n| n.parse::<i32>().ok()) {
            Some(bb) => Stacks::Fixed(bb * 2),
            None => panic!("--stacks full|sampled|NNbb, not {x}"),
        },
    }
}

fn stacks_name(s: Stacks) -> String {
    match s {
        Stacks::Full => "100bb".into(),
        Stacks::Sampled => "sampled".into(),
        Stacks::Fixed(c) => format!("{}bb", c / 2),
    }
}

/// `hu compare A.txt B.txt ...`: paired differences between saved matches
/// (`hu eval --save`) against the same opponent with the same seed, deals
/// and stacks: each file after the first minus the first, deal by deal.
pub fn compare_cmd(args: &[String]) {
    let files: Vec<&String> = args.iter().skip(2).filter(|a| !a.starts_with("--")).collect();
    assert!(files.len() >= 2, "hu compare BASE.txt OTHER.txt ...");
    let read = |p: &str| -> Vec<f64> { std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{p}: {e}")).lines().map(|l| l.parse().expect("number")).collect() };
    let base = read(files[0]);
    let (m, c) = hu_cfr::eval::summarize(&base);
    println!("{:70} {m:+8.1} ± {c:5.1} bb/100", files[0]);
    for f in &files[1..] {
        let x = read(f);
        let (m, c) = hu_cfr::eval::summarize(&x);
        let (d, dc) = hu_cfr::eval::compare(&x, &base);
        println!("{:70} {m:+8.1} ± {c:5.1} bb/100   minus base: {d:+7.2} ± {dc:5.2}", f);
    }
}

pub fn eval_cmd(args: &[String]) {
    let run = flag(args, "--run").expect("--run NAME");
    let deals: usize = num(args, "--deals", 20_000);
    let seed: u64 = num(args, "--seed", 1);
    let stacks = parse_stacks(flag(args, "--stacks").as_deref());
    let vs = flag(args, "--vs").unwrap_or("all".into());
    let names: Vec<String> = if vs == "all" { PANEL.iter().map(|s| s.to_string()).collect() } else { vs.split(',').map(String::from).collect() };
    let mut engine = load_engine(&run);
    live_flags(args, &mut engine);
    // Batch runs parallelize across hands, not inside a solve.
    engine.live_cfg.parallel = false;
    let me = Player::Engine(&engine, engine.live);
    let label = if engine.live == LiveMode::Off { run.clone() } else { format!("{run} (live {})", engine.live.name()) };
    // `run:NAME` plays against another trained run (its table only) instead
    // of a scripted bot.
    let rivals: Vec<(String, Engine)> =
        names.iter().filter_map(|n| n.strip_prefix("run:")).filter(|r| *r != run).map(|r| (r.to_string(), load_engine(r))).collect();
    for name in names {
        let t = Instant::now();
        let opp = match (Bot::parse(&name), name.strip_prefix("run:")) {
            (Some(b), _) => Player::Bot(b),
            (None, Some(r)) if r == run => Player::Engine(&engine, LiveMode::Off),
            (None, Some(r)) => Player::Engine(&rivals.iter().find(|(n, _)| n == r).unwrap().1, LiveMode::Off),
            (None, None) => panic!("unknown bot {name}"),
        };
        let r = play_match(&me, &opp, deals, stacks, seed);
        if let Some(dir) = flag(args, "--save") {
            // Per-deal results, for `hu compare` against another player's
            // match with the same opponent, seed, deals and stacks.
            let tag = flag(args, "--tag").unwrap_or_else(|| label.replace([' ', '(', ')'], ""));
            let path = std::path::Path::new(&dir).join(format!("{tag}--vs-{}--{}-{}-{seed}.txt", name.replace(':', "_"), stacks_name(stacks), deals));
            std::fs::create_dir_all(&dir).expect("create --save dir");
            let text: String = r.per_deal.iter().map(|x| format!("{x}\n")).collect();
            std::fs::write(&path, text).expect("write per-deal results");
        }
        log(format!(
            "{label} vs {name:<8} {:+8.1} ± {:5.1} bb/100  ({} hands, {:?} stacks, {} decisions, {} fallbacks, {:.1}s)",
            r.bb100,
            r.ci95,
            r.hands,
            stacks,
            r.decisions,
            r.fallbacks,
            t.elapsed().as_secs_f64()
        ));
    }
}

fn chart(title: &str, cols: &[&str], rows: &[[f64; MAX_ACTIONS]]) {
    // Combo-weighted totals, then the 13x13 grid of the main column.
    println!("{title}");
    let hands = class_hands();
    let w: f64 = 1326.0;
    let totals: Vec<String> = cols
        .iter()
        .enumerate()
        .map(|(i, c)| format!("{c} {:.1}%", (0..N_CLASSES).map(|k| rows[k][i] * hands[k].len() as f64).sum::<f64>() / w * 100.0))
        .collect();
    println!("  overall: {}", totals.join(", "));
    for (i, c) in cols.iter().enumerate() {
        println!("  {c} % by hand (suited above the diagonal):");
        print!("     ");
        for col in "AKQJT98765432".chars() {
            print!("{col:>4}");
        }
        println!();
        for (r, ch) in "AKQJT98765432".chars().enumerate() {
            print!("  {ch}  ");
            for col in 0..13 {
                print!("{:>4.0}", rows[r * 13 + col][i] * 100.0);
            }
            println!();
        }
    }
}

pub fn probe_cmd(args: &[String]) {
    let run = flag(args, "--run").expect("--run NAME");
    let stack: i32 = num(args, "--stack", 200);
    let engine = load_engine(&run);
    let s = engine.pick(stack);
    let tree = &s.tree;
    // The small blind's first action, grouped: fold / limp / raise / all-in.
    let root = &tree.nodes[0];
    let acts = tree.actions(root);
    let mut rows = vec![[0f64; MAX_ACTIONS]; N_CLASSES];
    let mut p = [0f64; MAX_ACTIONS];
    for c in 0..N_CLASSES {
        s.probs(0, c as u32, &mut p);
        for (a, act) in acts.iter().enumerate() {
            let col = match act {
                Act::Fold => 0,
                Act::Call | Act::Check => 1,
                Act::AllIn { .. } => 3,
                _ => 2,
            };
            rows[c][col] += p[a];
        }
    }
    chart(&format!("small blind first action at {}bb ({} iterations)", s.meta.stack / 2, s.meta.iterations), &["fold", "limp", "raise", "all-in"], &rows);

    // The big blind facing the 2.5bb open (raise to 5).
    let open = acts.iter().position(|a| matches!(a, Act::Raise { to: 5, .. }));
    if let Some(k) = open {
        let node = tree.children(root)[k];
        let nd = &tree.nodes[node as usize];
        let acts = tree.actions(nd);
        let mut rows = vec![[0f64; MAX_ACTIONS]; N_CLASSES];
        for c in 0..N_CLASSES {
            s.probs(node, c as u32, &mut p);
            for (a, act) in acts.iter().enumerate() {
                let col = match act {
                    Act::Fold => 0,
                    Act::Call | Act::Check => 1,
                    _ => 2,
                };
                rows[c][col] += p[a];
            }
        }
        chart("big blind facing a 2.5bb open", &["fold", "call", "3-bet"], &rows);
    }
}

pub fn livebench_cmd(args: &[String]) {
    let run = flag(args, "--run").expect("--run NAME");
    let street = match flag(args, "--street").as_deref() {
        None | Some("river") => Street::River,
        Some("turn") => Street::Turn,
        Some(x) => panic!("--street river|turn, not {x}"),
    };
    let n: usize = num(args, "--spots", 20);
    let stack: i32 = num(args, "--stack", 400);
    let seed: u64 = num(args, "--seed", 1);
    let iters: Vec<u32> = flag(args, "--iters").unwrap_or("100,200,400".into()).split(',').map(|x| x.parse().expect("--iters")).collect();
    let mut engine = load_engine(&run);
    live_flags(args, &mut engine);
    let mode = if street == Street::River { LiveMode::River } else { LiveMode::Turn };

    // Spots: the table plays both seats until someone acts on `street`.
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut spots = Vec::new();
    while spots.len() < n {
        let mut deck: Vec<u8> = (0..52).collect();
        deck.shuffle(&mut rng);
        let mut s = State::new_table([stack; 2]);
        s.button = rng.gen_range(0..2);
        s.start_hand(deck);
        let hseed: u64 = rng.gen();
        loop {
            match s.phase {
                Phase::Betting => {
                    let seat = s.to_act as usize;
                    let v = view_of(&s, seat, [stack; 2], hseed);
                    if s.street == street {
                        spots.push(v);
                        break;
                    }
                    let d = engine.decide_mode(&v, &mut rng, LiveMode::Off).expect("decide");
                    s.apply(d.action);
                }
                Phase::StreetComplete if !is_runout(&s) => s.advance_street(),
                _ => break,
            }
        }
    }
    log(format!("{} {:?} spots at {}bb (live {})", spots.len(), street, stack / 2, mode.name()));
    if args.iter().any(|a| a == "--show") {
        // Table vs live, spot by spot.
        for v in &spots {
            let mut r = SmallRng::seed_from_u64(1);
            let t = engine.decide_mode(v, &mut r, LiveMode::Off).expect("table");
            let l = engine.decide_mode(v, &mut r, mode).expect("live");
            let show = |d: &hu_cfr::engine::Debug| d.options.iter().map(|o| format!("{} {:.0}%", o.label, o.prob * 100.0)).collect::<Vec<_>>().join(", ");
            println!(
                "[{} | {}] {}",
                hu_cfr::engine::cards_str(&v.hole),
                hu_cfr::engine::cards_str(&v.board),
                v.actions.iter().map(|a| format!("{}{:?}{}", if a.seat == v.bot { "B:" } else { "o:" }, a.kind, a.to)).collect::<Vec<_>>().join(" ")
            );
            println!("    table ({}): {}", t.debug.bucket_label.clone().unwrap_or_default(), show(&t.debug));
            let ld = l.debug.live.as_ref().unwrap();
            println!(
                "    live  ({} ms, {:.2}% pot, ranges bot {:.0} opp {:.0}): {}",
                ld.ms, ld.exploitability_pct, ld.bot_range, ld.opp_range, show(&l.debug)
            );
        }
    }
    if args.iter().any(|a| a == "--check-safe") {
        // Safe vs unsafe solves on each spot, with the last --iters count.
        let it = *iters.last().unwrap();
        let mut rows = Vec::new();
        for v in &spots {
            match live::safe_check(&engine, v, mode, it) {
                Ok(c) => rows.push(c),
                Err(e) => log(format!("spot skipped: {e}")),
            }
        }
        let mean = |f: &dyn Fn(&live::SafeCheck) -> f64| rows.iter().map(f).sum::<f64>() / rows.len().max(1) as f64;
        let max = |f: &dyn Fn(&live::SafeCheck) -> f64| rows.iter().map(f).fold(0.0, f64::max);
        log(format!(
            "{} spots, {it} iterations: alternative values {:.0} ms mean | best-response values below table values on {:.2}% of hands (max {:.2}%) | opponent hands above their best against the table: unsafe {:.3}% of the pot per hand pair (max {:.3}), safe {:.3}% (max {:.3}) | gadget enters with {:.0}% of hands | estimated range best-responding wins: unsafe {:.2}% of the pot, safe {:.2}%",
            rows.len(),
            mean(&|c| c.alt_ms as f64),
            mean(&|c| c.best_below_table) * 100.0,
            max(&|c| c.best_below_table) * 100.0,
            mean(&|c| c.gain_unsafe),
            max(&|c| c.gain_unsafe),
            mean(&|c| c.gain_safe),
            max(&|c| c.gain_safe),
            mean(&|c| c.enter) * 100.0,
            mean(&|c| c.est_unsafe),
            mean(&|c| c.est_safe)
        ));
        return;
    }
    for &it in &iters {
        let mut ms = Vec::new();
        let mut pct = Vec::new();
        let mut nodes = 0;
        let mut floats = 0;
        for v in &spots {
            let (t, x, nd, fl) = live::bench(&engine, v, mode, it).expect("bench");
            ms.push(t as f64);
            pct.push(x);
            nodes = nodes.max(nd);
            floats = floats.max(fl);
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let mean_ms = ms.iter().sum::<f64>() / ms.len() as f64;
        let max_ms = ms.iter().cloned().fold(0.0, f64::max);
        let mean_pct = pct.iter().sum::<f64>() / pct.len() as f64;
        let max_pct = pct.iter().cloned().fold(0.0, f64::max);
        log(format!(
            "{it:>5} iterations: {mean_ms:6.0} ms mean, {:6.0} median, {max_ms:6.0} max | exploitability {mean_pct:.2}% of the pot mean, {:.2}% median, {max_pct:.2}% max | up to {nodes} decision nodes, {:.0} MB",
            med(&mut ms),
            med(&mut pct),
            floats as f64 * 4.0 / 1e6
        ));
    }
}
