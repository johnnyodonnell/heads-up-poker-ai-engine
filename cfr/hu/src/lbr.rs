//! `hu lbr`: local best response against a run (`hu_cfr::lbr`): a lower
//! bound on how exploitable it is, from an exploiter that knows its strategy.
//!
//!   hu lbr --run NAME [--live off|river|turn] [--deals 2000] [--from 0]
//!       [--stacks full|sampled|NNbb] [--seed 1] [--streets 234|1234|34|none]
//!       [--sizes min,0.33,0.5,0.75,1,1.5,2,3,allin] [--live-sizes min,0.5,1,2,allin]
//!       [--model-seeds 16] [--preflop-boards 2000] [--save DIR [--tag NAME]]
//!       [live flags, as for eval]
//!
//! Prints LBR's win rate (bb/100, its side) in chips and range-scored, and
//! what it did street by street: how often it chose each action, the bot's
//! fold rate its model predicted for its bets, and how often the bot really
//! folded. `--streets none` checks and calls everywhere: exactly the
//! `station` bot of `hu eval` with the same seed, deal by deal.

use std::collections::BTreeMap;
use std::time::Instant;

use hu_cfr::lbr::{Lbr, LbrCfg, Size};
use hu_cfr::live::{self, Cache};

use crate::play::{live_flags, load_engine, parse_stacks, stacks_name};
use crate::util::{flag, log, num};

pub fn lbr_cmd(args: &[String]) {
    let run = flag(args, "--run").expect("--run NAME");
    let deals: usize = num(args, "--deals", 2000);
    let from: usize = num(args, "--from", 0);
    let seed: u64 = num(args, "--seed", 1);
    let stacks = parse_stacks(flag(args, "--stacks").as_deref());
    let mut cfg = LbrCfg::default();
    if let Some(s) = flag(args, "--streets") {
        cfg.streets = LbrCfg::parse_streets(&s).unwrap_or_else(|| panic!("--streets like 234, 1234 or none, not {s}"));
    }
    if let Some(s) = flag(args, "--sizes") {
        cfg.sizes = Size::parse_list(&s).unwrap_or_else(|| panic!("--sizes like min,0.5,1,allin, not {s}"));
    }
    if let Some(s) = flag(args, "--live-sizes") {
        cfg.live_sizes = Size::parse_list(&s).unwrap_or_else(|| panic!("--live-sizes like min,0.5,1,allin, not {s}"));
    }
    cfg.model_seeds = num(args, "--model-seeds", cfg.model_seeds);
    cfg.preflop_boards = num(args, "--preflop-boards", cfg.preflop_boards);

    let mut engine = load_engine(&run);
    live_flags(args, &mut engine);
    // Batch runs parallelize across hands, not inside a solve; LBR's model
    // solves under its own translation draws too, so keep more hands' solves.
    engine.live_cfg.parallel = false;
    engine.cache = Cache::new(2048);
    let mode = engine.live;
    let label = if mode == live::LiveMode::Off { run.clone() } else { format!("{run} (live {})", mode.name()) };
    let lbr = Lbr { bot: &engine, mode, cfg };
    log(format!("LBR vs {label}: {}; {deals} deals from {from}, {} stacks, seed {seed}", lbr.cfg.describe(), stacks_name(stacks)));

    use std::sync::atomic::Ordering::Relaxed;
    let turn0 = live::TURN_SOLVES.load(Relaxed);
    let t = Instant::now();
    let r = lbr.play_match(from, deals, stacks, seed);
    let secs = t.elapsed().as_secs_f64();
    let turns = live::TURN_SOLVES.load(Relaxed) - turn0;

    // What LBR did, per street.
    let names = ["preflop", "flop", "turn", "river"];
    let mut by: BTreeMap<(usize, String), (usize, f64, usize, usize)> = BTreeMap::new();
    let mut per_street = [0usize; 4];
    for p in &r.picks {
        per_street[p.street] += 1;
        let e = by.entry((p.street, p.label.clone())).or_default();
        e.0 += 1;
        if let Some(fp) = p.fp {
            e.1 += fp;
            if let Some(f) = p.folded {
                e.2 += 1;
                e.3 += usize::from(f);
            }
        }
    }
    for st in 0..4 {
        if per_street[st] == 0 {
            continue;
        }
        let mut rows: Vec<(&String, &(usize, f64, usize, usize))> = by.iter().filter(|((s, _), _)| *s == st).map(|((_, l), v)| (l, v)).collect();
        rows.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
        let line: Vec<String> = rows
            .iter()
            .map(|(l, (n, fp, seen, folded))| {
                let share = 100.0 * *n as f64 / per_street[st] as f64;
                if *seen > 0 {
                    format!("{l} {share:.0}% (fold: model {:.0}%, real {:.0}%)", 100.0 * fp / *n as f64, 100.0 * *folded as f64 / *seen as f64)
                } else {
                    format!("{l} {share:.0}%")
                }
            })
            .collect();
        log(format!("  LBR {:<7} ({} decisions): {}", names[st], per_street[st], line.join(", ")));
    }
    for (d, e) in r.errors.iter().take(5) {
        log(format!("  deal {d} failed: {e}"));
    }
    if let Some(dir) = flag(args, "--save") {
        // Per deal: LBR's chips and range-scored result (mean of the plays).
        let streets: String = (0..4).filter(|&i| lbr.cfg.streets[i]).map(|i| char::from(b'1' + i as u8)).collect();
        let tag = flag(args, "--tag").unwrap_or_else(|| label.replace([' ', '(', ')'], ""));
        let span = if from == 0 { deals.to_string() } else { format!("{from}+{deals}") };
        let path = std::path::Path::new(&dir).join(format!("{tag}--lbr{}--{}-{span}-{seed}.lbr", if streets.is_empty() { "none".into() } else { streets }, stacks_name(stacks)));
        std::fs::create_dir_all(&dir).expect("create --save dir");
        let text: String = r.per_deal.iter().map(|(c, s)| format!("{c} {s}\n")).collect();
        std::fs::write(&path, text).expect("write per-deal results");
        log(format!("  saved {}", path.display()));
    }
    let solves = if turns > 0 { format!(", {turns} turn solves") } else { String::new() };
    log(format!(
        "LBR vs {label}: LBR wins {:+.1} ± {:.1} bb/100 in chips, {:+.1} ± {:.1} range-scored  ({} hands, {:.0}s; bot {} decisions, {} fallbacks{solves}; {} model misses, {} failed deals)",
        r.chips.0,
        r.chips.1,
        r.scored.0,
        r.scored.1,
        r.hands,
        secs,
        r.bot_decisions,
        r.bot_fallbacks,
        r.model_misses,
        r.errors.len()
    ));
}
