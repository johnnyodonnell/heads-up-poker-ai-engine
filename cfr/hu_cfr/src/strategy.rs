//! A trained strategy for one stack depth, as served: the abstract tree and
//! each information set's average strategy, quantized to one byte per action.
//!
//! Files in a strategy directory: `meta.json`, `tree.json`, `probs.u8`
//! (indexed like the solver's tables: node.base + bucket * n + action).

use std::path::Path;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::game::{TreeGame, MAX_ACTIONS, TERMINAL};
use crate::mccfr::Solver;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StrategyMeta {
    /// Effective stack in chips (blinds 1/2).
    pub stack: i32,
    pub menu: String,
    pub abstraction: String,
    pub iterations: u64,
}

pub struct Strategy {
    pub meta: StrategyMeta,
    pub tree: TreeGame,
    pub probs: Vec<u8>,
}

impl Strategy {
    /// Quantize the solver's average strategy.
    pub fn from_solver(meta: StrategyMeta, tree: TreeGame, solver: &Solver) -> Strategy {
        let mut probs = vec![0u8; tree.n_entries as usize];
        // Fill per decision node; nodes own disjoint ranges of `probs`.
        let mut ranges: Vec<(u32, usize, usize)> = Vec::new();
        for (id, nd) in tree.nodes.iter().enumerate() {
            if nd.player != TERMINAL {
                let len = tree.buckets[nd.street as usize] as usize * nd.n as usize;
                ranges.push((id as u32, nd.base as usize, len));
            }
        }
        ranges.sort_by_key(|r| r.1);
        let mut rest: &mut [u8] = &mut probs;
        let mut pos = 0usize;
        let mut chunks: Vec<(u32, &mut [u8])> = Vec::with_capacity(ranges.len());
        for (id, base, len) in ranges {
            let (_, tail) = rest.split_at_mut(base - pos);
            let (mine, tail) = tail.split_at_mut(len);
            chunks.push((id, mine));
            rest = tail;
            pos = base + len;
        }
        chunks.into_par_iter().for_each(|(id, out)| {
            let n = tree.nodes[id as usize].n as usize;
            let mut p = [0f64; MAX_ACTIONS];
            for (b, row) in out.chunks_mut(n).enumerate() {
                solver.average(&tree, id, b as u32, &mut p);
                quantize(&p[..n], row);
            }
        });
        Strategy { meta, tree, probs }
    }

    /// Action probabilities at (node, bucket), renormalized.
    pub fn probs(&self, node: u32, bucket: u32, out: &mut [f64]) {
        let nd = &self.tree.nodes[node as usize];
        let n = nd.n as usize;
        let base = (nd.base + bucket as u64 * n as u64) as usize;
        let q = &self.probs[base..base + n];
        let total: u32 = q.iter().map(|&x| x as u32).sum();
        for a in 0..n {
            out[a] = if total == 0 { 1.0 / n as f64 } else { q[a] as f64 / total as f64 };
        }
    }

    pub fn save(&self, dir: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("meta.json"), serde_json::to_string_pretty(&self.meta).unwrap())?;
        std::fs::write(dir.join("tree.json"), serde_json::to_string(&self.tree).unwrap())?;
        let tmp = dir.join("probs.u8.tmp");
        std::fs::write(&tmp, &self.probs)?;
        std::fs::rename(&tmp, dir.join("probs.u8"))
    }

    pub fn load(dir: &Path) -> std::io::Result<Strategy> {
        let bad = |e: serde_json::Error| std::io::Error::other(format!("{}: {e}", dir.display()));
        let meta: StrategyMeta = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json"))?).map_err(bad)?;
        let tree: TreeGame = serde_json::from_str(&std::fs::read_to_string(dir.join("tree.json"))?).map_err(bad)?;
        let probs = std::fs::read(dir.join("probs.u8"))?;
        if probs.len() as u64 != tree.n_entries {
            return Err(std::io::Error::other(format!("{}: {} probs for {} entries", dir.display(), probs.len(), tree.n_entries)));
        }
        Ok(Strategy { meta, tree, probs })
    }
}

/// Probabilities -> bytes summing to ~255 (largest remainders get the slack).
fn quantize(p: &[f64], out: &mut [u8]) {
    let n = p.len();
    let mut q = [0u32; MAX_ACTIONS];
    let mut rem = [0f64; MAX_ACTIONS];
    let mut used = 0u32;
    for a in 0..n {
        let x = p[a].max(0.0) * 255.0;
        q[a] = x.floor() as u32;
        rem[a] = x - x.floor();
        used += q[a];
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| rem[y].partial_cmp(&rem[x]).unwrap());
    for &a in order.iter().take(255u32.saturating_sub(used) as usize) {
        if rem[a] > 0.0 {
            q[a] += 1;
        }
    }
    for a in 0..n {
        out[a] = q[a].min(255) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::quantize;

    #[test]
    fn quantize_sums_to_255() {
        for p in [vec![1.0, 0.0, 0.0], vec![0.333, 0.333, 0.334], vec![0.5, 0.25, 0.125, 0.125], vec![0.999, 0.001]] {
            let mut out = vec![0u8; p.len()];
            quantize(&p, &mut out);
            let s: u32 = out.iter().map(|&x| x as u32).sum();
            assert!((254..=255).contains(&s), "{p:?} -> {out:?}");
        }
    }
}
