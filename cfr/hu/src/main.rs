//! `hu` — the CFR engine's command line.
//!
//! Subcommands:
//!   tree --stack 200 [--menu lean] [--buckets 169,2000,2000,1000]
//!       size of the abstract betting tree and its regret tables
//!   pushfold [--stack 20] [--iters 2e7] [--samples 20000] [--exact-iters 20000]
//!       solver check on real cards: MCCFR vs an exact solve of heads-up
//!       push/fold, with the resulting shove/call charts
//!   abstraction --name NAME --buckets 169,F,T,R [--fit-boards 1500,2000,2000] [--seed 1]
//!       build the hand-group tables into $HU_HOME/data/NAME/
//!   train --run NAME --abs ABS --stack 200 --iters 1e9   train one stack depth (resumable)
//!   export --run NAME --abs ABS --stack 200              export from the latest checkpoint
//!   eval --run NAME [--vs all|bot,...|run:OTHER] [--deals 20000] [--stacks full|sampled|200bb]
//!   compare BASE.txt OTHER.txt ...                       paired differences of saved evals
//!   probe --run NAME [--stack 200]                       preflop charts
//!   livebench --run NAME [--street river|turn] [--iters 100,200,400]  time live solves
//!   serve --run NAME [--port 8790] [--web DIR]           the engine over HTTP + the web app
//!   rangecheck --run NAME [--opp-run NAME] [--hands N]   the table's range estimates in self-play
//!   scorecheck --run NAME [--opp-run NAME] [--spots N]   checks of advantage scoring on real spots
//!   turncheck --run NAME [--opp-run A,B] [--configs ...]  live turn decisions vs the table, per setting
//!
//! eval, serve and livebench take --live off|river|turn (solve those
//! streets live; default off) and --river-iters / --turn-iters.

mod play;
mod rangecheck;
mod scorecheck;
mod serve;
mod turncheck;
mod train;
mod util;

use std::time::Instant;

use hu_cfr::classes::{class_name, combos, N_CLASSES};
use hu_cfr::mccfr::{Config, Solver};
use hu_cfr::menu::Menu;
use hu_cfr::pushfold;
use util::{flag, hu_home, list, log, num, threads};

fn tree_cmd(args: &[String]) {
    let stack: i32 = num(args, "--stack", 200);
    let menu = Menu::by_name(&flag(args, "--menu").unwrap_or("lean".into())).expect("unknown menu");
    let b = list(args, "--buckets").unwrap_or(vec![169, 2000, 2000, 1000]);
    let g = hu_cfr::tree::build(stack, &menu, [b[0], b[1], b[2], b[3]]);
    let c = g.street_counts();
    let e = g.street_entries();
    println!(
        "stack {stack} chips ({}bb), menu {}: {} nodes ({} terminal)",
        stack / 2,
        menu.name,
        g.nodes.len(),
        g.nodes.len() - c.iter().sum::<usize>()
    );
    println!("decision nodes pre/flop/turn/river: {c:?}");
    println!("entries pre/flop/turn/river: {e:?}  total {} ({:.2} GiB of f64 regrets + strategy sums)", g.n_entries, g.n_entries as f64 * 16.0 / (1u64 << 30) as f64);
}

fn chart(title: &str, p: &[f64]) {
    println!("{title}");
    print!("     ");
    for col in "AKQJT98765432".chars() {
        print!("{col:>4}");
    }
    println!();
    for (row, ch) in "AKQJT98765432".chars().enumerate() {
        print!("  {ch}  ");
        for col in 0..13 {
            print!("{:>4.0}", p[row * 13 + col] * 100.0);
        }
        println!();
    }
}

fn range_pct(p: &[f64]) -> f64 {
    (0..N_CLASSES).map(|c| p[c] * combos(c as u32) as f64).sum::<f64>() / 1326.0 * 100.0
}

fn pushfold_cmd(args: &[String]) {
    let stack: i32 = num(args, "--stack", 20);
    let iters: u64 = num(args, "--iters", 20_000_000);
    let samples: usize = num(args, "--samples", 20_000);
    let exact_iters: usize = num(args, "--exact-iters", 20_000);
    println!("push/fold at {stack} chips ({:.1}bb), {} threads", stack as f64 / 2.0, threads());

    let t = Instant::now();
    let m = pushfold::matchups(samples, 1);
    println!("matchups: {samples} samples per class pair in {:.1}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let (ex_s, ex_c) = pushfold::solve_exact(stack, &m, exact_iters);
    let ex_expl = pushfold::exploitability(stack, &m, &ex_s, &ex_c);
    println!(
        "exact CFR+ ({exact_iters} iterations, {:.1}s): shove {:.1}%, call {:.1}%, exploitability {:.3} mbb/hand",
        t.elapsed().as_secs_f64(),
        range_pct(&ex_s),
        range_pct(&ex_c),
        ex_expl / 2.0 * 1000.0
    );

    let g = pushfold::tree(stack);
    let mut solver = Solver::new(&g);
    let cfg = Config { threads: threads(), discount_every: iters / 100, discount_until: iters / 2, ..Config::default() };
    let t = Instant::now();
    solver.train(&g, &pushfold::PushFoldDealer, iters, iters / 10, &cfg, 7, |_| true);
    let secs = t.elapsed().as_secs_f64();
    let mut mc_s = vec![0.0; N_CLASSES];
    let mut mc_c = vec![0.0; N_CLASSES];
    let mut buf = [0f64; 8];
    for c in 0..N_CLASSES {
        solver.average(&g, 0, c as u32, &mut buf);
        mc_s[c] = buf[1];
        solver.average(&g, pushfold::BB_NODE, c as u32, &mut buf);
        mc_c[c] = buf[1];
    }
    let mc_expl = pushfold::exploitability(stack, &m, &mc_s, &mc_c);
    println!(
        "MCCFR ({iters} iterations, {secs:.1}s, {:.2}M iterations/s): shove {:.1}%, call {:.1}%, exploitability {:.3} mbb/hand",
        iters as f64 / secs / 1e6,
        range_pct(&mc_s),
        range_pct(&mc_c),
        mc_expl / 2.0 * 1000.0
    );
    let diff = |a: &[f64], b: &[f64]| {
        let mut worst = (0.0, 0);
        let mut mean = 0.0;
        for c in 0..N_CLASSES {
            let d = (a[c] - b[c]).abs();
            mean += d * combos(c as u32) as f64 / 1326.0;
            if d > worst.0 {
                worst = (d, c);
            }
        }
        (mean, worst)
    };
    let (ms, ws) = diff(&mc_s, &ex_s);
    let (mcall, wc) = diff(&mc_c, &ex_c);
    println!(
        "MCCFR vs exact: shove mean |diff| {:.3} (worst {} {:.2}), call mean |diff| {:.3} (worst {} {:.2})",
        ms,
        class_name(ws.1 as u32),
        ws.0,
        mcall,
        class_name(wc.1 as u32),
        wc.0
    );
    chart("small blind shove % (MCCFR); suited above the diagonal", &mc_s);
    chart("big blind call % (MCCFR)", &mc_c);
}

fn abstraction_cmd(args: &[String]) {
    let name = flag(args, "--name").expect("--name NAME");
    let b = list(args, "--buckets").expect("--buckets 169,F,T,R");
    let fit = list(args, "--fit-boards").unwrap_or(vec![1500, 2000, 2000]);
    let seed: u64 = num(args, "--seed", 1);
    let dir = hu_home().join("data").join(&name);
    log(format!("building abstraction {name} {b:?} (fit boards {fit:?}) -> {}", dir.display()));
    let t = Instant::now();
    let abs = hu_cfr::abs::build(&name, [b[0], b[1], b[2], b[3]], [fit[0] as usize, fit[1] as usize, fit[2] as usize], seed, &log);
    abs.save(&dir).expect("save abstraction");
    log(format!("done in {:.1} min", t.elapsed().as_secs_f64() / 60.0));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("tree") => tree_cmd(&args),
        Some("pushfold") => pushfold_cmd(&args),
        Some("abstraction") => abstraction_cmd(&args),
        Some("train") => train::train_cmd(&args),
        Some("export") => train::export_cmd(&args),
        Some("eval") => play::eval_cmd(&args),
        Some("compare") => play::compare_cmd(&args),
        Some("probe") => play::probe_cmd(&args),
        Some("livebench") => play::livebench_cmd(&args),
        Some("serve") => serve::serve_cmd(&args),
        Some("rangecheck") => rangecheck::rangecheck_cmd(&args),
        Some("scorecheck") => scorecheck::scorecheck_cmd(&args),
        Some("turncheck") => turncheck::turncheck_cmd(&args),
        other => {
            eprintln!("unknown subcommand {other:?}; expected: tree | pushfold | abstraction | train | export | eval | compare | probe | livebench | serve | rangecheck | scorecheck | turncheck");
            std::process::exit(2);
        }
    }
}
