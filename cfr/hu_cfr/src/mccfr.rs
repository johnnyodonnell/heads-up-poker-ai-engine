//! External-sampling Monte Carlo CFR over a `TreeGame`, multithreaded.
//!
//! Each iteration samples one deal and traverses the tree once per player:
//! the traverser explores all of its actions and updates their regrets; the
//! opponent samples one action from its current strategy and adds that
//! strategy to its running average (stochastically-weighted averaging, which
//! is unbiased for external sampling).
//!
//! Following Pluribus's blueprint recipe:
//! - Linear-CFR-style discounting early on: every `discount_every`
//!   iterations, until `discount_until`, regrets and strategy sums are scaled
//!   by k/(k+1) (k = discounts so far + 1), which down-weights the noisy first
//!   iterations.
//! - Negative-regret pruning once `prune_after` iterations have run: on 95%
//!   of traversals, actions whose regret is below `prune_threshold` are not
//!   explored, unless they end the hand or are on the last street.
//! - Regrets are floored at `regret_floor` so pruned actions can recover.
//!
//! Threads share the regret and strategy tables without locks: entries are
//! atomics holding f64 bits, updated with compare-and-swap adds (no lost
//! updates), while a traversal may read slightly stale values. f64 (not f32)
//! because the busiest strategy sums grow past 2^24, where f32 can no longer
//! add a probability.

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::game::{Deal, Dealer, TreeGame, MAX_ACTIONS, TERMINAL};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub threads: usize,
    pub discount_every: u64,
    pub discount_until: u64,
    pub prune_after: u64,
    pub prune_threshold: f64,
    pub regret_floor: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
            discount_every: 0,
            discount_until: 0,
            prune_after: u64::MAX,
            prune_threshold: f64::NEG_INFINITY,
            regret_floor: -1e9,
        }
    }
}

#[inline]
fn ld(a: &AtomicU64) -> f64 {
    f64::from_bits(a.load(Relaxed))
}

#[inline]
fn st(a: &AtomicU64, v: f64) {
    a.store(v.to_bits(), Relaxed)
}

/// Atomically add `d` (then clamp at `floor`): a compare-and-swap loop, so
/// concurrent updates to the hottest entries (the preflop root, touched by
/// every thread every iteration) are never lost.
#[inline]
fn add(a: &AtomicU64, d: f64, floor: f64) {
    let _ = a.fetch_update(Relaxed, Relaxed, |bits| Some((f64::from_bits(bits) + d).max(floor).to_bits()));
}

/// A zero-filled table of atomics (allocated as zeroed u64s: cheap for big
/// tables, since 0u64 is the bit pattern of 0.0f64).
fn zeroed(n: usize) -> Box<[AtomicU64]> {
    let v: Box<[u64]> = vec![0u64; n].into_boxed_slice();
    // SAFETY: AtomicU64 has the same size and bit validity as u64, and its
    // alignment (8) is what the allocation for u64 already guarantees on the
    // 64-bit targets this runs on.
    unsafe { Box::from_raw(Box::into_raw(v) as *mut [AtomicU64]) }
}

/// Regret matching: positive regrets normalized, uniform if none are positive.
#[inline]
pub fn regret_matching(r: &[AtomicU64], out: &mut [f64]) {
    let mut sum = 0.0f64;
    for (o, x) in out.iter_mut().zip(r) {
        let v = ld(x).max(0.0);
        *o = v;
        sum += v;
    }
    if sum > 0.0 {
        for o in out.iter_mut() {
            *o /= sum;
        }
    } else {
        let u = 1.0 / out.len() as f64;
        out.fill(u);
    }
}

#[inline]
fn sample(p: &[f64], rng: &mut SmallRng) -> usize {
    let mut x: f64 = rng.gen();
    for (i, &w) in p.iter().enumerate() {
        if x < w {
            return i;
        }
        x -= w;
    }
    // Rounding left x slightly above the total: take the last action with mass.
    p.iter().rposition(|&w| w > 0.0).unwrap_or(p.len() - 1)
}

pub struct Solver {
    pub regret: Box<[AtomicU64]>,
    pub strat: Box<[AtomicU64]>,
    pub iterations: u64,
    pub discounts: u64,
}

struct Ctx<'a> {
    game: &'a TreeGame,
    regret: &'a [AtomicU64],
    strat: &'a [AtomicU64],
    prune: bool,
    prune_threshold: f64,
    regret_floor: f64,
    last_street: u8,
}

fn traverse(cx: &Ctx, node: u32, p: usize, deal: &Deal, rng: &mut SmallRng) -> f64 {
    let g = cx.game;
    let nd = &g.nodes[node as usize];
    if nd.player == TERMINAL {
        return g.utility(nd, p, deal.showdown);
    }
    let q = nd.player as usize;
    let n = nd.n as usize;
    let base = (nd.base + deal.bucket[q][nd.street as usize] as u64 * n as u64) as usize;
    let mut sigma = [0f64; MAX_ACTIONS];
    regret_matching(&cx.regret[base..base + n], &mut sigma[..n]);
    let kids = g.children(nd);
    if q == p {
        let mut v = [0f64; MAX_ACTIONS];
        let mut explored = [false; MAX_ACTIONS];
        let mut any = false;
        let mut vn = 0f64;
        let can_prune = cx.prune && nd.street < cx.last_street;
        for a in 0..n {
            if can_prune
                && ld(&cx.regret[base + a]) < cx.prune_threshold
                && g.nodes[kids[a] as usize].player != TERMINAL
            {
                continue;
            }
            v[a] = traverse(cx, kids[a], p, deal, rng);
            explored[a] = true;
            any = true;
            vn += sigma[a] * v[a];
        }
        if !any {
            // Everything pruned (shouldn't happen): fall back to a full pass.
            for a in 0..n {
                v[a] = traverse(cx, kids[a], p, deal, rng);
                explored[a] = true;
                vn += sigma[a] * v[a];
            }
        }
        for a in 0..n {
            if explored[a] {
                add(&cx.regret[base + a], v[a] - vn, cx.regret_floor);
            }
        }
        vn
    } else {
        for a in 0..n {
            add(&cx.strat[base + a], sigma[a], f64::NEG_INFINITY);
        }
        let a = sample(&sigma[..n], rng);
        traverse(cx, kids[a], p, deal, rng)
    }
}

impl Solver {
    pub fn new(game: &TreeGame) -> Solver {
        let n = game.n_entries as usize;
        Solver { regret: zeroed(n), strat: zeroed(n), iterations: 0, discounts: 0 }
    }

    fn last_street(game: &TreeGame) -> u8 {
        game.nodes.iter().map(|n| n.street).max().unwrap_or(0)
    }

    /// Run `iters` more iterations (split across `cfg.threads`), without
    /// applying any discounting (see `train` for the full schedule).
    pub fn run(&mut self, game: &TreeGame, dealer: &dyn Dealer, iters: u64, cfg: &Config, seed: u64) {
        let threads = cfg.threads.max(1) as u64;
        let prune = self.iterations >= cfg.prune_after;
        let last_street = Self::last_street(game);
        let start = self.iterations;
        let (regret, strat) = (&self.regret[..], &self.strat[..]);
        std::thread::scope(|sc| {
            for t in 0..threads {
                let share = iters / threads + u64::from(t < iters % threads);
                if share == 0 {
                    continue;
                }
                let tseed = seed
                    ^ start.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ (t + 1).wrapping_mul(0xD1B5_4A32_D192_ED03);
                sc.spawn(move || {
                    let mut rng = SmallRng::seed_from_u64(tseed);
                    for _ in 0..share {
                        let deal = dealer.deal(&mut rng);
                        let cx = Ctx {
                            game,
                            regret,
                            strat,
                            prune: prune && rng.gen::<f64>() < 0.95,
                            prune_threshold: cfg.prune_threshold,
                            regret_floor: cfg.regret_floor,
                            last_street,
                        };
                        for p in 0..2 {
                            traverse(&cx, 0, p, &deal, &mut rng);
                        }
                    }
                });
            }
        });
        self.iterations += iters;
    }

    /// Scale every regret and strategy sum by `factor`.
    pub fn discount(&mut self, factor: f64) {
        for table in [&self.regret, &self.strat] {
            table.par_chunks(1 << 16).for_each(|c| {
                for x in c {
                    st(x, ld(x) * factor);
                }
            });
        }
    }

    /// Run up to `target` total iterations following the discount schedule,
    /// calling `report` after each chunk of at most `chunk` iterations.
    /// `report` returns false to stop early.
    pub fn train(
        &mut self,
        game: &TreeGame,
        dealer: &dyn Dealer,
        target: u64,
        chunk: u64,
        cfg: &Config,
        seed: u64,
        mut report: impl FnMut(&Solver) -> bool,
    ) {
        while self.iterations < target {
            let mut next = (self.iterations + chunk.max(1)).min(target);
            let discounting = cfg.discount_every > 0 && self.iterations < cfg.discount_until;
            if discounting {
                let boundary = (self.iterations / cfg.discount_every + 1) * cfg.discount_every;
                next = next.min(boundary);
            }
            // Pruning switches on at a chunk boundary.
            if self.iterations < cfg.prune_after && cfg.prune_after < next {
                next = cfg.prune_after;
            }
            self.run(game, dealer, next - self.iterations, cfg, seed);
            if discounting && self.iterations % cfg.discount_every == 0 {
                self.discounts += 1;
                let k = self.discounts as f64;
                self.discount(k / (k + 1.0));
            }
            if !report(self) {
                break;
            }
        }
    }

    /// The average strategy at (node, bucket): normalized strategy sums,
    /// falling back to the current strategy, then uniform.
    pub fn average(&self, game: &TreeGame, node: u32, bucket: u32, out: &mut [f64]) {
        let nd = &game.nodes[node as usize];
        let n = nd.n as usize;
        let base = (nd.base + bucket as u64 * n as u64) as usize;
        let mut sum = 0.0;
        for a in 0..n {
            out[a] = ld(&self.strat[base + a]).max(0.0);
            sum += out[a];
        }
        if sum > 0.0 {
            for o in out[..n].iter_mut() {
                *o /= sum;
            }
        } else {
            regret_matching(&self.regret[base..base + n], &mut out[..n]);
        }
    }

    /// Total strategy weight at (node, bucket): how much the average has seen.
    pub fn visits(&self, game: &TreeGame, node: u32, bucket: u32) -> f64 {
        let nd = &game.nodes[node as usize];
        let n = nd.n as usize;
        let base = (nd.base + bucket as u64 * n as u64) as usize;
        (0..n).map(|a| ld(&self.strat[base + a])).sum()
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let tmp = path.with_extension("tmp");
        {
            let mut w = BufWriter::with_capacity(1 << 24, std::fs::File::create(&tmp)?);
            w.write_all(b"HUCFR002")?;
            w.write_all(&self.iterations.to_le_bytes())?;
            w.write_all(&self.discounts.to_le_bytes())?;
            w.write_all(&(self.regret.len() as u64).to_le_bytes())?;
            for table in [&self.regret, &self.strat] {
                for c in table.chunks(1 << 20) {
                    let bytes: Vec<u8> = c.iter().flat_map(|x| x.load(Relaxed).to_le_bytes()).collect();
                    w.write_all(&bytes)?;
                }
            }
            w.flush()?;
        }
        std::fs::rename(&tmp, path)
    }

    pub fn load(game: &TreeGame, path: &Path) -> std::io::Result<Solver> {
        let mut r = BufReader::with_capacity(1 << 24, std::fs::File::open(path)?);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != b"HUCFR002" {
            return Err(std::io::Error::other("not a solver checkpoint"));
        }
        let mut u = [0u8; 8];
        r.read_exact(&mut u)?;
        let iterations = u64::from_le_bytes(u);
        r.read_exact(&mut u)?;
        let discounts = u64::from_le_bytes(u);
        r.read_exact(&mut u)?;
        let n = u64::from_le_bytes(u) as usize;
        if n as u64 != game.n_entries {
            return Err(std::io::Error::other(format!("checkpoint has {n} entries, tree has {}", game.n_entries)));
        }
        let mut s = Solver { regret: zeroed(n), strat: zeroed(n), iterations, discounts };
        let mut buf = vec![0u8; 8 << 20];
        for table in [&mut s.regret, &mut s.strat] {
            for c in table.chunks_mut(1 << 20) {
                let bytes = &mut buf[..c.len() * 8];
                r.read_exact(bytes)?;
                for (x, b) in c.iter_mut().zip(bytes.chunks_exact(8)) {
                    *x.get_mut() = u64::from_le_bytes(b.try_into().unwrap());
                }
            }
        }
        Ok(s)
    }
}
