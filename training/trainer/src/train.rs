//! REINFORCE update over one on-policy cohort (fox-lite's recipe): one
//! optimizer step per cohort, the gradient accumulated over micro-batches.
//!
//!   log pi(a|s) = log_softmax(masked type logits)[type]
//!               + [sized bet/raise] * log Beta(u; a, b)
//!   advantage   = z - V(s)                      (V detached: a baseline)
//!   loss        = -mean(advantage * log pi) + c_value * MSE(V, z)
//!                 - c_type_ent * H(type) - c_size_ent * H(Beta)
//!
//! H(type) is averaged over all rows; H(Beta) over rows where a sized
//! bet/raise was legal. The Beta's entropy is at most 0 (uniform), so its
//! bonus can't run away the way a Gaussian's log-sigma bonus can.
//!
//! AdamW is implemented directly on tensors so its moments can be saved in
//! checkpoints (tch's built-in optimizer state can't be serialized).

use std::collections::HashMap;

use tch::nn::VarStore;
use tch::{Device, Kind, Tensor};

use hu_core::actions::{NUM_TYPES, RAISE};
use hu_core::encode::INPUT_SIZE;

use crate::net::HuNet;
use crate::selfplay::Row;

const MASK_NEG: f64 = 1e9;

pub struct TrainCfg {
    pub lr: f64,
    pub weight_decay: f64,
    pub c_value: f64,
    pub c_type_ent: f64,
    pub c_size_ent: f64,
    pub grad_clip: f64,
    pub micro_batch: usize,
}

/// The cohort as device tensors.
pub struct Batch {
    pub n: i64,
    x: Tensor,
    mask: Tensor,
    ty: Tensor,
    u: Tensor,
    sized: Tensor,
    z: Tensor,
    logp_b: Tensor,
}

impl Batch {
    pub fn from_rows(rows: &[Row], dev: Device) -> Batch {
        let n = rows.len();
        let mut x = Vec::with_capacity(n * INPUT_SIZE);
        let mut mask = Vec::with_capacity(n * NUM_TYPES);
        for r in rows {
            x.extend_from_slice(&r.x);
            mask.extend_from_slice(&r.mask);
        }
        let col = |f: &dyn Fn(&Row) -> f32| Tensor::from_slice(&rows.iter().map(f).collect::<Vec<f32>>()).to_device(dev);
        Batch {
            n: n as i64,
            x: Tensor::from_slice(&x).view([n as i64, INPUT_SIZE as i64]).to_device(dev),
            mask: Tensor::from_slice(&mask).view([n as i64, NUM_TYPES as i64]).to_device(dev),
            ty: Tensor::from_slice(&rows.iter().map(|r| r.ty as i64).collect::<Vec<i64>>()).to_device(dev),
            u: col(&|r| r.u),
            sized: col(&|r| r.sized as u8 as f32),
            z: col(&|r| r.z),
            logp_b: col(&|r| r.logp),
        }
    }
}

/// ln B(a, b).
fn ln_beta(a: &Tensor, b: &Tensor) -> Tensor {
    a.lgamma() + b.lgamma() - (a + b).lgamma()
}

/// Per-row log pi(a|s), the masked type log-probs [B,4], and the Beta's
/// entropy [B], from net outputs.
fn log_probs(
    type_logits: &Tensor,
    size_ab: &Tensor,
    mask: &Tensor,
    ty: &Tensor,
    u: &Tensor,
    sized: &Tensor,
) -> (Tensor, Tensor, Tensor) {
    let masked = type_logits + (mask - 1.0) * MASK_NEG;
    let logp_all = masked.log_softmax(1, Kind::Float);
    let logp_ty = logp_all.gather(1, &ty.unsqueeze(1), false).squeeze_dim(1);
    let a = size_ab.select(1, 0);
    let b = size_ab.select(1, 1);
    let lb = ln_beta(&a, &b);
    // Rows without a sized raise carry u = 0: keep the log finite, then zero it.
    let uu = u.clamp(1e-6, 1.0 - 1e-6);
    let size_logp = (&a - 1.0) * uu.log() + (&b - 1.0) * (-&uu).log1p() - &lb;
    let size_logp = size_logp * sized;
    let ab = &a + &b;
    let ent = &lb - (&a - 1.0) * a.digamma() - (&b - 1.0) * b.digamma() + (&ab - 2.0) * ab.digamma();
    (logp_ty + size_logp, logp_all, ent)
}

pub struct AdamW {
    pub step: i64,
    names: Vec<String>,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
}

impl AdamW {
    const B1: f64 = 0.9;
    const B2: f64 = 0.999;
    const EPS: f64 = 1e-8;

    pub fn new(vs: &VarStore) -> AdamW {
        let vars = vs.variables();
        let mut names: Vec<String> = vars.keys().cloned().collect();
        names.sort();
        let m = names.iter().map(|k| vars[k].zeros_like()).collect();
        let v = names.iter().map(|k| vars[k].zeros_like()).collect();
        AdamW { step: 0, names, m, v }
    }

    /// Clip the global grad norm to `clip`, then take one AdamW step.
    /// Returns the pre-clip grad norm.
    pub fn step(&mut self, vs: &VarStore, lr: f64, wd: f64, clip: f64) -> f64 {
        let vars = vs.variables();
        let params: Vec<&Tensor> = self.names.iter().map(|k| &vars[k]).collect();
        tch::no_grad(|| {
            let sq: Vec<Tensor> = params.iter().map(|p| p.grad().pow_tensor_scalar(2).sum(Kind::Float)).collect();
            let norm = Tensor::stack(&sq, 0).sum(Kind::Float).sqrt().double_value(&[]);
            let scale = if norm > clip { clip / (norm + 1e-6) } else { 1.0 };
            self.step += 1;
            let bc1 = 1.0 - Self::B1.powi(self.step as i32);
            let bc2 = 1.0 - Self::B2.powi(self.step as i32);
            for (i, p) in params.iter().enumerate() {
                let g = p.grad() * scale;
                let m = &mut self.m[i];
                let v = &mut self.v[i];
                *m = &*m * Self::B1 + &g * (1.0 - Self::B1);
                *v = &*v * Self::B2 + (&g * &g) * (1.0 - Self::B2);
                let upd = (&*m / bc1) / ((&*v / bc2).sqrt() + Self::EPS);
                let mut p = p.shallow_clone();
                let _ = p.f_mul_scalar_(1.0 - lr * wd).unwrap();
                let _ = p.f_sub_(&(upd * lr)).unwrap();
            }
            norm
        })
    }

    pub fn save(&self, path: &str) {
        let mut named: Vec<(String, Tensor)> = Vec::new();
        for (i, k) in self.names.iter().enumerate() {
            named.push((format!("m.{k}"), self.m[i].to_device(Device::Cpu)));
            named.push((format!("v.{k}"), self.v[i].to_device(Device::Cpu)));
        }
        named.push(("step".into(), Tensor::from_slice(&[self.step])));
        let refs: Vec<(&str, &Tensor)> = named.iter().map(|(k, t)| (k.as_str(), t)).collect();
        Tensor::write_safetensors(&refs, path).expect("save optimizer");
    }

    pub fn load(&mut self, path: &str, dev: Device) {
        let map: HashMap<String, Tensor> = Tensor::read_safetensors(path).expect("load optimizer").into_iter().collect();
        for (i, k) in self.names.iter().enumerate() {
            self.m[i] = map[&format!("m.{k}")].to_device(dev);
            self.v[i] = map[&format!("v.{k}")].to_device(dev);
        }
        self.step = map["step"].int64_value(&[0]);
    }
}

#[derive(Default, Debug)]
pub struct StepStats {
    pub loss: f64,
    pub policy: f64,
    pub value: f64,
    pub type_entropy: f64,
    pub size_entropy: f64,
    /// Mean Beta (a, b) and mean band position a / (a + b) where a sized
    /// bet/raise was legal.
    pub beta_a: f64,
    pub beta_b: f64,
    pub beta_mean: f64,
    pub grad_norm: f64,
    /// mean |log pi(trainer) - log pi(sampler)| before the update: should be
    /// ~0 (the on-policy sanity check).
    pub logp_gap: f64,
    /// mean(log pi_behavior - log pi_new) after the update: approx KL.
    pub kl: f64,
}

pub fn reinforce_step(net: &HuNet, vs: &VarStore, opt: &mut AdamW, b: &Batch, cfg: &TrainCfg, lr: f64) -> StepStats {
    let n = b.n;
    let mut st = StepStats::default();
    for p in vs.trainable_variables() {
        let mut p = p.shallow_clone();
        p.zero_grad();
    }
    let mb = cfg.micro_batch as i64;
    let raise_rows = b.mask.select(1, RAISE as i64).sum(Kind::Float).double_value(&[]).max(1.0);
    let mut start = 0;
    while start < n {
        let len = mb.min(n - start);
        let sl = |t: &Tensor| t.narrow(0, start, len);
        let out = net.forward(&sl(&b.x));
        let (logp, logp_all, ent) =
            log_probs(&out.type_logits, &out.size_ab, &sl(&b.mask), &sl(&b.ty), &sl(&b.u), &sl(&b.sized));
        let z = sl(&b.z);
        let adv = &z - out.value.detach();
        let policy = -(adv * &logp).sum(Kind::Float) / n as f64;
        let value = (&out.value - &z).square().sum(Kind::Float) / n as f64;
        let type_entropy = -(logp_all.exp() * &logp_all).sum(Kind::Float) / n as f64;
        let raise_legal = sl(&b.mask).select(1, RAISE as i64);
        let size_entropy = (&ent * &raise_legal).sum(Kind::Float) / raise_rows;
        let loss = &policy + cfg.c_value * &value - cfg.c_type_ent * &type_entropy - cfg.c_size_ent * &size_entropy;
        loss.backward();
        st.loss += loss.double_value(&[]);
        st.policy += policy.double_value(&[]);
        st.value += value.double_value(&[]);
        st.type_entropy += type_entropy.double_value(&[]);
        st.size_entropy += size_entropy.double_value(&[]);
        tch::no_grad(|| {
            let a = out.size_ab.select(1, 0);
            let bb = out.size_ab.select(1, 1);
            let w = |t: &Tensor| (t * &raise_legal).sum(Kind::Float).double_value(&[]) / raise_rows;
            st.beta_a += w(&a);
            st.beta_b += w(&bb);
            st.beta_mean += w(&(&a / (&a + &bb)));
            st.logp_gap += (logp.detach() - sl(&b.logp_b)).abs().sum(Kind::Float).double_value(&[]) / n as f64;
        });
        start += len;
    }
    st.grad_norm = opt.step(vs, lr, cfg.weight_decay, cfg.grad_clip);

    // Approximate KL(behavior || new) on the cohort.
    tch::no_grad(|| {
        let mut start = 0;
        while start < n {
            let len = mb.min(n - start);
            let sl = |t: &Tensor| t.narrow(0, start, len);
            let (tl, ab) = net.policy(&sl(&b.x));
            let (logp, _, _) = log_probs(&tl, &ab, &sl(&b.mask), &sl(&b.ty), &sl(&b.u), &sl(&b.sized));
            st.kl += (sl(&b.logp_b) - logp).sum(Kind::Float).double_value(&[]) / n as f64;
            start += len;
        }
    });
    st
}
