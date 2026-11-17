//! Loading a run's strategies into an engine; `hu eval` and `hu probe`.
//!
//!   hu eval --run NAME [--vs all|station,maniac,...] [--deals 20000] [--stacks full|sampled]
//!   hu probe --run NAME [--stack 200]   preflop charts: the small blind's first
//!       action and the big blind facing a 2.5bb open

use std::path::PathBuf;
use std::time::Instant;

use hu_cfr::bots::{Bot, PANEL};
use hu_cfr::classes::{class_hands, N_CLASSES};
use hu_cfr::engine::Engine;
use hu_cfr::eval::{play_match, Player, Stacks};
use hu_cfr::game::{Act, MAX_ACTIONS};
use hu_cfr::strategy::StrategyMeta;

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

pub fn eval_cmd(args: &[String]) {
    let run = flag(args, "--run").expect("--run NAME");
    let deals: usize = num(args, "--deals", 20_000);
    let seed: u64 = num(args, "--seed", 1);
    let stacks = match flag(args, "--stacks").as_deref() {
        None | Some("full") => Stacks::Full,
        Some("sampled") => Stacks::Sampled,
        Some(other) => panic!("--stacks full|sampled, not {other}"),
    };
    let vs = flag(args, "--vs").unwrap_or("all".into());
    let names: Vec<String> = if vs == "all" { PANEL.iter().map(|s| s.to_string()).collect() } else { vs.split(',').map(String::from).collect() };
    let engine = load_engine(&run);
    let me = Player::Engine(&engine);
    // `run:NAME` plays against another trained run instead of a scripted bot.
    let rivals: Vec<(String, Engine)> = names.iter().filter_map(|n| n.strip_prefix("run:")).map(|r| (r.to_string(), load_engine(r))).collect();
    for name in names {
        let t = Instant::now();
        let opp = match (Bot::parse(&name), name.strip_prefix("run:")) {
            (Some(b), _) => Player::Bot(b),
            (None, Some(r)) => Player::Engine(&rivals.iter().find(|(n, _)| n == r).unwrap().1),
            (None, None) => panic!("unknown bot {name}"),
        };
        let r = play_match(&me, &opp, deals, stacks, seed);
        log(format!(
            "{run} vs {name:<8} {:+8.1} ± {:5.1} bb/100  ({} hands, {:?} stacks, {} decisions, {} fallbacks, {:.1}s)",
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
