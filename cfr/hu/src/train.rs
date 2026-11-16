//! `hu train` / `hu export`: train one stack depth with MCCFR (resumable),
//! then export its strategy for the engine.
//!
//!   hu train --run NAME --abs ABS --stack 200 --iters 1e9 [--menu lean] [--seed 1]
//!       runs/NAME/s<stack>/solver.ckpt   (checkpoint, every 30 min + at the end)
//!       strategies/NAME/s<stack>/        (exported at the end)
//!   hu export --run NAME --abs ABS --stack 200 [--menu lean]
//!       export from the latest checkpoint (e.g. while training continues)

use std::path::PathBuf;
use std::time::Instant;

use hu_cfr::abs::{Abstraction, HuDealer};
use hu_cfr::mccfr::{Config, Solver};
use hu_cfr::menu::Menu;
use hu_cfr::strategy::{Strategy, StrategyMeta};
use hu_cfr::tree;

use crate::util::{flag, hu_home, log, num, threads};

struct Setup {
    run: String,
    abs_name: String,
    stack: i32,
    menu: Menu,
    run_dir: PathBuf,
    out_dir: PathBuf,
}

fn setup(args: &[String]) -> Setup {
    let run = flag(args, "--run").expect("--run NAME");
    let abs_name = flag(args, "--abs").expect("--abs ABSTRACTION");
    let stack: i32 = num(args, "--stack", 200);
    let menu = Menu::by_name(&flag(args, "--menu").unwrap_or("lean".into())).expect("unknown menu");
    let run_dir = hu_home().join("runs").join(&run).join(format!("s{stack}"));
    let out_dir = hu_home().join("strategies").join(&run).join(format!("s{stack}"));
    Setup { run, abs_name, stack, menu, run_dir, out_dir }
}

fn export(st: &Setup, tree: hu_cfr::game::TreeGame, solver: &Solver) {
    let t = Instant::now();
    let meta = StrategyMeta {
        stack: st.stack,
        menu: st.menu.name.clone(),
        abstraction: st.abs_name.clone(),
        iterations: solver.iterations,
    };
    let s = Strategy::from_solver(meta, tree, solver);
    s.save(&st.out_dir).expect("save strategy");
    log(format!("exported {} ({} entries) in {:.1}s", st.out_dir.display(), s.probs.len(), t.elapsed().as_secs_f64()));
}

pub fn train_cmd(args: &[String]) {
    let st = setup(args);
    let iters: u64 = num(args, "--iters", 100_000_000);
    let seed: u64 = num(args, "--seed", 1);
    std::fs::create_dir_all(&st.run_dir).expect("run dir");
    let abs = Abstraction::load(&hu_home().join("data").join(&st.abs_name)).expect("load abstraction");
    let tree = tree::build(st.stack, &st.menu, abs.buckets());
    let ckpt = st.run_dir.join("solver.ckpt");
    let mut solver = if ckpt.exists() {
        let s = Solver::load(&tree, &ckpt).expect("load checkpoint");
        log(format!("resumed {} at {} iterations", ckpt.display(), s.iterations));
        s
    } else {
        Solver::new(&tree)
    };
    // Pluribus-style schedule, in iterations: discount for the first 20%,
    // prune negative-regret actions after that.
    let stack = st.stack as f64;
    let cfg = Config {
        threads: threads(),
        discount_every: (iters / 200).max(1),
        discount_until: iters / 5,
        prune_after: iters / 5,
        prune_threshold: -500.0 * stack,
        regret_floor: -515.0 * stack,
    };
    let c = tree.street_counts();
    log(format!(
        "run {} stack {} ({}bb), abstraction {} {:?}, menu {}: {:?} decision nodes, {} entries ({:.2} GiB), {} threads, target {iters} iterations",
        st.run,
        st.stack,
        st.stack / 2,
        st.abs_name,
        abs.buckets(),
        st.menu.name,
        c,
        tree.n_entries,
        tree.n_entries as f64 * 16.0 / (1u64 << 30) as f64,
        cfg.threads
    ));
    let dealer = HuDealer { abs: &abs };
    let t0 = Instant::now();
    let start_iters = solver.iterations;
    let mut last_ckpt = Instant::now();
    solver.train(&tree, &dealer, iters, (iters / 100).max(1), &cfg, seed, |s| {
        let done = s.iterations - start_iters;
        let secs = t0.elapsed().as_secs_f64();
        let rate = done as f64 / secs;
        let eta = (iters - s.iterations) as f64 / rate.max(1.0);
        log(format!(
            "{:>5.1}%  {} iterations  {:.0}/s  elapsed {:.1} min  eta {:.1} min  discounts {}{}",
            s.iterations as f64 / iters as f64 * 100.0,
            s.iterations,
            rate,
            secs / 60.0,
            eta / 60.0,
            s.discounts,
            if s.iterations >= cfg.prune_after { "  pruning" } else { "" }
        ));
        if last_ckpt.elapsed().as_secs() > 1800 {
            s.save(&ckpt).expect("save checkpoint");
            log(format!("checkpoint {}", ckpt.display()));
            last_ckpt = Instant::now();
        }
        true
    });
    solver.save(&ckpt).expect("save checkpoint");
    log(format!("trained {} iterations in {:.1} min", solver.iterations - start_iters, t0.elapsed().as_secs_f64() / 60.0));
    export(&st, tree, &solver);
}

pub fn export_cmd(args: &[String]) {
    let st = setup(args);
    let abs = Abstraction::load(&hu_home().join("data").join(&st.abs_name)).expect("load abstraction");
    let tree = tree::build(st.stack, &st.menu, abs.buckets());
    let solver = Solver::load(&tree, &st.run_dir.join("solver.ckpt")).expect("load checkpoint");
    export(&st, tree, &solver);
}
