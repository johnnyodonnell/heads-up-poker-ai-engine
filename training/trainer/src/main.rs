//! `hu` — the heads-up poker trainer CLI.
//!
//! Subcommands:
//!   gpu-check                       libtorch sees the GPU and can train on it
//!   export --out m.onnx [--weights w.safetensors | --init SEED]
//!          [--fixture fx.json] [--n 256]
//!                                   write the browser ONNX (+ a parity fixture:
//!                                   real encoded positions and tch's outputs)
//!   train --run-dir DIR [...]       the self-play REINFORCE daemon (see run.rs)
//!   selfplay [--weights w.safetensors] [--hands 32768] [--slots 4096] [--seed 1]
//!                                   one self-play cohort on its own (profiling
//!                                   + sanity stats)

mod hud;
mod net;
mod onnx;
mod run;
mod selfplay;
mod train;
mod util;

use tch::nn::{self, Module, OptimizerConfig};
use tch::{Device, Kind, Tensor};

use hu_core::encode::INPUT_SIZE;
use util::{flag, has_flag, random_positions};

/// CUDA visible, a small MLP's loss drops under AdamW with forward + backward
/// on the GPU, and the poker net runs a forward pass there.
fn gpu_check() -> bool {
    let cuda = tch::Cuda::is_available();
    println!("cuda available: {cuda} (devices: {})", tch::Cuda::device_count());
    let dev = if cuda { Device::Cuda(0) } else { Device::Cpu };

    let vs = nn::VarStore::new(dev);
    let root = vs.root();
    let mlp = nn::seq()
        .add(nn::linear(&root / "l1", 64, 256, Default::default()))
        .add_fn(|x| x.gelu("none"))
        .add(nn::linear(&root / "l2", 256, 1, Default::default()));
    let mut opt = nn::AdamW::default().build(&vs, 1e-3).expect("optimizer");

    tch::manual_seed(0);
    let x = Tensor::randn([8192, 64], (Kind::Float, dev));
    let y = x.sum_dim_intlist(1, true, Kind::Float).sin();
    let mut first = f64::NAN;
    let mut last = f64::NAN;
    for step in 0..300 {
        let loss = mlp.forward(&x).mse_loss(&y, tch::Reduction::Mean);
        opt.backward_step(&loss);
        let l = loss.double_value(&[]);
        if step == 0 {
            first = l;
        }
        last = l;
    }
    println!("device {dev:?}: loss {first:.4} -> {last:.4}");

    let vs2 = nn::VarStore::new(dev);
    let hnet = net::HuNet::new(&vs2.root());
    let xs = Tensor::from_slice(&random_positions(256, 1).concat()).view([256, INPUT_SIZE as i64]).to_device(dev);
    let out = hnet.forward(&xs);
    println!(
        "hu net: {} params, type_logits {:?}, size_ab {:?}, value {:?}",
        net::n_params(&vs2),
        out.type_logits.size(),
        out.size_ab.size(),
        out.value.size()
    );

    let ok = cuda && last < 0.5 * first;
    println!("{}", if ok { "GPU-CHECK OK" } else { "GPU-CHECK FAILED" });
    ok
}

fn export(args: &[String]) {
    let out = flag(args, "--out").expect("--out model.onnx");
    let mut vs = nn::VarStore::new(Device::Cpu);
    if let Some(seed) = flag(args, "--init") {
        tch::manual_seed(seed.parse().expect("--init SEED"));
    }
    let hnet = net::HuNet::new(&vs.root());
    if let Some(w) = flag(args, "--weights") {
        vs.load(&w).unwrap_or_else(|e| panic!("load {w}: {e}"));
    } else if !has_flag(args, "--init") {
        panic!("pass --weights w.safetensors or --init SEED");
    }
    std::fs::write(&out, hnet.export_onnx()).expect("write onnx");
    println!("wrote {out} ({} params)", net::n_params(&vs));

    if let Some(fx) = flag(args, "--fixture") {
        let n: usize = util::num(args, "--n", 256);
        let rows = random_positions(n, 12345);
        let xs = Tensor::from_slice(&rows.concat()).view([n as i64, INPUT_SIZE as i64]);
        let (tl, ab) = tch::no_grad(|| hnet.policy(&xs));
        let v = |t: &Tensor| Vec::<f32>::try_from(&t.contiguous().view([-1])).unwrap();
        let json = serde_json::json!({
            "n": n,
            "inputs": rows,
            "type_logits": v(&tl),
            "size_ab": v(&ab),
        });
        std::fs::write(&fx, json.to_string()).expect("write fixture");
        println!("wrote {fx} ({n} positions)");
    }
}

fn device() -> Device {
    if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    }
}

fn selfplay_cmd(args: &[String]) {
    let dev = device();
    let mut vs = nn::VarStore::new(dev);
    let hnet = net::HuNet::new(&vs.root());
    if let Some(w) = flag(args, "--weights") {
        vs.load(&w).unwrap_or_else(|e| panic!("load {w}: {e}"));
    }
    let hands: usize = util::num(args, "--hands", 32_768);
    let slots: usize = util::num(args, "--slots", 4096);
    let seed: u64 = util::num(args, "--seed", 1);
    let t0 = std::time::Instant::now();
    let c = selfplay::run_cohort(&hnet, dev, slots, hands, seed);
    let secs = t0.elapsed().as_secs_f64();
    let n = c.rows.len();
    let zs: Vec<f32> = c.rows.iter().map(|r| r.z).collect();
    let zmin = zs.iter().cloned().fold(f32::INFINITY, f32::min);
    let zmax = zs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let zabs = zs.iter().map(|z| z.abs() as f64).sum::<f64>() / n.max(1) as f64;
    assert!(zmin >= -1.0 && zmax <= 1.0, "z out of [-1, 1]: [{zmin}, {zmax}]");
    let sized = c.rows.iter().filter(|r| r.sized).count();
    println!(
        "{}",
        serde_json::json!({
            "hands": c.hands, "rows": n, "rows_per_hand": (n as f64 / c.hands as f64 * 100.0).round() / 100.0,
            "sized_rows": sized, "forwards": c.forwards, "secs": (secs * 100.0).round() / 100.0,
            "hands_per_sec": (c.hands as f64 / secs).round(), "z_min": zmin, "z_max": zmax,
            "z_mean_abs": (zabs * 1000.0).round() / 1000.0, "hud": c.hud.to_json(),
        })
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("gpu-check") => std::process::exit(if gpu_check() { 0 } else { 1 }),
        Some("export") => export(&args),
        Some("selfplay") => selfplay_cmd(&args),
        Some("train") => run::train(&args),
        other => {
            eprintln!("unknown subcommand {other:?}; expected: gpu-check | export | selfplay | train");
            std::process::exit(2);
        }
    }
}
