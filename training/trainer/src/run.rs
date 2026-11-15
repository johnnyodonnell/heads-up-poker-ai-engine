//! The training daemon: fox-lite's strict-synchronous loop, in one process
//! with the model resident on the GPU (no cohort files, no worker protocol):
//!
//!   1. self-play a cohort with the current weights (on-policy)
//!   2. one REINFORCE step over it
//!   3. save `latest` (weights + AdamW moments + counters) for resume
//!   4. if a snapshot is due: save it, then evaluate and probe it
//!   5. log one JSON line
//!
//!   hu train --run-dir ~/Workspace/heads-up-poker-ai-engine/runs/run1
//!       [--hands 32768] [--slots 4096] [--micro-batch 8192]
//!       [--lr 3e-4] [--warmup 100] [--weight-decay 1e-4] [--c-value 1.0]
//!       [--c-type-ent 0.01] [--c-size-ent 0.001] [--grad-clip 1.0]
//!       [--snapshot-every 4h] [--seed 42] [--max-cohorts 0]
//!       [--eval-deals 30000] [--n-top 2] [--n-anchors 3] [--no-eval]
//!
//! Resumes from <run-dir>/latest.* if present. Snapshots are taken every
//! --snapshot-every of training time; the next scheduled snapshot is kept in
//! latest.json, so restarting with a new interval keeps that one and spaces
//! the later ones by the new interval.

use std::path::Path;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::json;
use tch::nn::VarStore;
use tch::Device;

use crate::net::{n_params, HuNet};
use crate::selfplay::run_cohort;
use crate::train::{reinforce_step, AdamW, Batch, TrainCfg};
use crate::util::{flag, num, parse_duration};

#[derive(Serialize, Deserialize, Default)]
pub struct Meta {
    pub elapsed_sec: f64,
    pub cohorts: u64,
    pub hands: u64,
    pub rows: u64,
    pub next_snapshot_at: f64,
    #[serde(default)]
    pub last_snapshot: Option<String>,
}

fn r(x: f64, digits: i32) -> f64 {
    let m = 10f64.powi(digits);
    (x * m).round() / m
}

pub fn train(args: &[String]) {
    let run_dir = flag(args, "--run-dir").expect("--run-dir DIR");
    let hands: usize = num(args, "--hands", 32_768);
    let slots: usize = num(args, "--slots", 4096);
    let seed: u64 = num(args, "--seed", 42);
    let max_cohorts: u64 = num(args, "--max-cohorts", 0);
    // Linear LR warmup over the first cohorts: Adam's first steps move every
    // weight by ~lr.
    let warmup: u64 = num(args, "--warmup", 100);
    let snapshot_every = parse_duration(&flag(args, "--snapshot-every").unwrap_or("4h".into()));
    let cfg = TrainCfg {
        lr: num(args, "--lr", 3e-4),
        weight_decay: num(args, "--weight-decay", 1e-4),
        c_value: num(args, "--c-value", 1.0),
        c_type_ent: num(args, "--c-type-ent", 0.01),
        c_size_ent: num(args, "--c-size-ent", 0.001),
        grad_clip: num(args, "--grad-clip", 1.0),
        micro_batch: num(args, "--micro-batch", 8192),
    };

    let dir = Path::new(&run_dir);
    std::fs::create_dir_all(dir.join("snapshots")).expect("run dir");
    let p = |name: &str| dir.join(name).to_string_lossy().into_owned();

    let dev = if tch::Cuda::is_available() { Device::Cuda(0) } else { Device::Cpu };
    println!("device={dev:?}");
    let mut vs = VarStore::new(dev);
    tch::manual_seed(seed as i64);
    let net = HuNet::new(&vs.root());
    let mut opt = AdamW::new(&vs);
    let mut meta = Meta { next_snapshot_at: snapshot_every, ..Default::default() };
    if Path::new(&p("latest.json")).exists() {
        meta = serde_json::from_str(&std::fs::read_to_string(p("latest.json")).unwrap()).expect("latest.json");
        vs.load(p("latest.safetensors")).expect("load latest weights");
        opt.load(&p("latest_opt.safetensors"), dev);
        println!(
            "resumed {run_dir} (elapsed={:.2}h cohorts={} hands={} next_snapshot_at={:.2}h)",
            meta.elapsed_sec / 3600.0,
            meta.cohorts,
            meta.hands,
            meta.next_snapshot_at / 3600.0
        );
    } else {
        println!("cold start (seed={seed})");
    }
    println!(
        "net: {} params; hands/cohort={hands} slots={slots} lr={} snapshot_every={}s",
        n_params(&vs),
        cfg.lr,
        snapshot_every
    );

    let start = Instant::now();
    let base_elapsed = meta.elapsed_sec;
    loop {
        let t0 = Instant::now();
        let cohort_seed = seed.wrapping_add(meta.cohorts.wrapping_mul(7919)).wrapping_add(1);
        let cohort = run_cohort(&net, dev, slots, hands, cohort_seed);
        let sp_sec = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let batch = Batch::from_rows(&cohort.rows, dev);
        let z_mean = cohort.rows.iter().map(|r| r.z as f64).sum::<f64>() / cohort.rows.len().max(1) as f64;
        let lr = cfg.lr * ((meta.cohorts + 1) as f64 / warmup.max(1) as f64).min(1.0);
        let stats = reinforce_step(&net, &vs, &mut opt, &batch, &cfg, lr);
        let tr_sec = t1.elapsed().as_secs_f64();
        drop(batch);

        meta.cohorts += 1;
        meta.hands += cohort.hands as u64;
        meta.rows += cohort.rows.len() as u64;
        meta.elapsed_sec = base_elapsed + start.elapsed().as_secs_f64();

        // Latest checkpoint: write temps, then rename.
        vs.save(p("latest.tmp.safetensors")).expect("save weights");
        opt.save(&p("latest_opt.tmp.safetensors"));
        let mut snapshot = None;
        if meta.elapsed_sec >= meta.next_snapshot_at {
            meta.next_snapshot_at += snapshot_every;
            let name = format!("snap_h{:05}_c{:07}.safetensors", (meta.elapsed_sec / 3600.0) as u64, meta.cohorts);
            let path = dir.join("snapshots").join(&name);
            vs.save(&path).expect("save snapshot");
            snapshot = Some(path);
        }
        std::fs::write(p("latest.tmp.json"), serde_json::to_string(&meta).unwrap()).unwrap();
        for (tmp, f) in [
            ("latest.tmp.safetensors", "latest.safetensors"),
            ("latest_opt.tmp.safetensors", "latest_opt.safetensors"),
            ("latest.tmp.json", "latest.json"),
        ] {
            std::fs::rename(p(tmp), p(f)).expect("rename checkpoint");
        }

        println!(
            "{}",
            json!({
                "t": r(meta.elapsed_sec, 1),
                "cohorts": meta.cohorts,
                "hands": meta.hands,
                "rows": cohort.rows.len(),
                "lr": lr,
                "sp_sec": r(sp_sec, 2),
                "tr_sec": r(tr_sec, 2),
                "hands_per_s": (cohort.hands as f64 / sp_sec).round(),
                "z_mean": r(z_mean, 4),
                "loss": r(stats.loss, 4),
                "policy": r(stats.policy, 4),
                "value": r(stats.value, 4),
                "type_ent": r(stats.type_entropy, 4),
                "size_ent": r(stats.size_entropy, 4),
                "beta_ab": [r(stats.beta_a, 2), r(stats.beta_b, 2)],
                "beta_mean": r(stats.beta_mean, 3),
                "grad_norm": r(stats.grad_norm, 3),
                "logp_gap": r(stats.logp_gap, 6),
                "kl": r(stats.kl, 6),
                "hud": cohort.hud.to_json(),
            })
        );
        if let Some(path) = snapshot {
            println!("[snapshot] {}", path.display());
            let name = path.file_stem().unwrap().to_string_lossy().into_owned();
            println!("[probe] {name} {}", crate::probe::run(&net, dev));
            let must: Vec<String> = meta.last_snapshot.iter().cloned().collect();
            crate::eval::after_snapshot(&run_dir, &path, &must, args);
            meta.last_snapshot = Some(name);
            std::fs::write(p("latest.tmp.json"), serde_json::to_string(&meta).unwrap()).unwrap();
            std::fs::rename(p("latest.tmp.json"), p("latest.json")).expect("rename meta");
        }

        if max_cohorts > 0 && meta.cohorts >= max_cohorts {
            println!("reached --max-cohorts; stopping");
            break;
        }
    }
}
