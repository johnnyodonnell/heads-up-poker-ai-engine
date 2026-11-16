//! Heads-up push/fold: the small blind (player 0, 1 chip) shoves all-in or
//! folds; the big blind (player 1, 2 chips) calls or folds. Stacks are equal.
//!
//! Two ways to solve it, used to check the solver on real cards:
//! - `tree` + `PushFoldDealer` for the MCCFR solver (sampled deals, 169
//!   preflop classes as buckets);
//! - `solve_exact`: full-width CFR+ on the 169×169 class matchup matrix
//!   (combo-pair counts with card removal, Monte Carlo equities).

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use hu_core::cards::Card;
use hu_core::hand_eval::eval7;

use crate::classes::{class_hands, class_of, N_CLASSES};
use crate::game::{Act, Deal, Dealer, Terminal, TreeGame};

/// The push/fold tree for `stack` chips each. Node 0 is the small blind's
/// decision (fold, all-in); `BB_NODE` is the big blind facing the shove.
pub fn tree(stack: i32) -> TreeGame {
    let mut g = TreeGame::new([N_CLASSES as u32, 0, 0, 0]);
    let root = g.decision(0, 0, [1, 2]);
    let fold = g.terminal(0, [1, 2], Terminal::Fold { winner: 1 });
    let bb = g.decision(1, 0, [stack, 2]);
    let bb_fold = g.terminal(0, [stack, 2], Terminal::Fold { winner: 0 });
    let call = g.terminal(0, [stack, stack], Terminal::Showdown);
    g.set_children(bb, &[(Act::Fold, bb_fold), (Act::Call, call)]);
    let frac = (stack - 2) as f32 / 4.0;
    g.set_children(root, &[(Act::Fold, fold), (Act::AllIn { to: stack, frac }, bb)]);
    g.finalize();
    assert_eq!((root, bb), (0, BB_NODE));
    g
}

pub const BB_NODE: u32 = 2;

/// Deal 9 distinct cards from a fresh deck into `out` (partial Fisher–Yates).
#[inline]
pub fn deal_cards(rng: &mut SmallRng, out: &mut [Card]) {
    let mut deck: [Card; 52] = std::array::from_fn(|i| i as Card);
    for i in 0..out.len() {
        let j = rng.gen_range(i..52);
        deck.swap(i, j);
        out[i] = deck[i];
    }
}

/// Showdown result (+1 player 0 wins, -1 player 1, 0 tie) for two hole-card
/// pairs on a 5-card board.
#[inline]
pub fn showdown(h0: [Card; 2], h1: [Card; 2], board: &[Card]) -> i8 {
    let mut c = [0u8; 7];
    c[2..].copy_from_slice(board);
    c[0] = h0[0];
    c[1] = h0[1];
    let v0 = eval7(&c);
    c[0] = h1[0];
    c[1] = h1[1];
    let v1 = eval7(&c);
    (v0.cmp(&v1) as i8).signum()
}

pub struct PushFoldDealer;

impl Dealer for PushFoldDealer {
    fn deal(&self, rng: &mut SmallRng) -> Deal {
        let mut c = [0u8; 9];
        deal_cards(rng, &mut c);
        let mut d = Deal::default();
        d.bucket[0][0] = class_of(c[0], c[1]);
        d.bucket[1][0] = class_of(c[2], c[3]);
        d.showdown = showdown([c[0], c[1]], [c[2], c[3]], &c[4..9]);
        d
    }
}

/// Class-vs-class matchups for the small blind (row) against the big blind
/// (column): combo-pair counts with card removal, and Monte Carlo equity.
pub struct Matchups {
    pub w: Vec<f64>,
    pub eq: Vec<f64>,
}

pub fn matchups(samples: usize, seed: u64) -> Matchups {
    let hands = class_hands();
    let rows: Vec<(Vec<f64>, Vec<f64>)> = (0..N_CLASSES)
        .into_par_iter()
        .map(|h| {
            let mut rng = SmallRng::seed_from_u64(seed ^ (h as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut w = vec![0.0; N_CLASSES];
            let mut eq = vec![0.0; N_CLASSES];
            for g in 0..N_CLASSES {
                let pairs: Vec<((Card, Card), (Card, Card))> = hands[h]
                    .iter()
                    .flat_map(|&a| hands[g].iter().map(move |&b| (a, b)))
                    .filter(|&((a0, a1), (b0, b1))| a0 != b0 && a0 != b1 && a1 != b0 && a1 != b1)
                    .collect();
                w[g] = pairs.len() as f64;
                let mut won = 0.0;
                for _ in 0..samples {
                    let ((a0, a1), (b0, b1)) = pairs[rng.gen_range(0..pairs.len())];
                    let mut used: u64 = (1 << a0) | (1 << a1) | (1 << b0) | (1 << b1);
                    let mut board = [0u8; 5];
                    for slot in board.iter_mut() {
                        loop {
                            let c = rng.gen_range(0..52u8);
                            if used & (1 << c) == 0 {
                                used |= 1 << c;
                                *slot = c;
                                break;
                            }
                        }
                    }
                    won += match showdown([a0, a1], [b0, b1], &board) {
                        1 => 1.0,
                        0 => 0.5,
                        _ => 0.0,
                    };
                }
                eq[g] = won / samples as f64;
            }
            (w, eq)
        })
        .collect();
    let mut m = Matchups { w: Vec::with_capacity(N_CLASSES * N_CLASSES), eq: Vec::with_capacity(N_CLASSES * N_CLASSES) };
    for (w, eq) in rows {
        m.w.extend(w);
        m.eq.extend(eq);
    }
    m
}

/// Per-class action values: small blind (fold, shove) given the big blind's
/// call probabilities, and big blind (fold, call) given the shove
/// probabilities. Values are combo-weighted chip totals.
fn values(stack: f64, m: &Matchups, shove: &[f64], call: &[f64]) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let n = N_CLASSES;
    let mut sb = vec![[0.0; 2]; n];
    let mut bb = vec![[0.0; 2]; n];
    for h in 0..n {
        for g in 0..n {
            let w = m.w[h * n + g];
            if w == 0.0 {
                continue;
            }
            let showdown = stack * (2.0 * m.eq[h * n + g] - 1.0);
            sb[h][0] += w * -1.0;
            sb[h][1] += w * ((1.0 - call[g]) * 2.0 + call[g] * showdown);
            // The big blind only decides when the small blind shoves.
            bb[g][0] += w * shove[h] * -2.0;
            bb[g][1] += w * shove[h] * -showdown;
        }
    }
    (sb, bb)
}

/// Exact push/fold equilibrium by full-width CFR+ with linear averaging:
/// (shove probability per class, call probability per class).
pub fn solve_exact(stack: i32, m: &Matchups, iters: usize) -> (Vec<f64>, Vec<f64>) {
    let n = N_CLASSES;
    let s = stack as f64;
    let (mut rs, mut rc) = (vec![[0.0f64; 2]; n], vec![[0.0f64; 2]; n]);
    let (mut shove, mut call) = (vec![0.5; n], vec![0.5; n]);
    let (mut avg_s, mut avg_c, mut wsum) = (vec![0.0; n], vec![0.0; n], 0.0);
    let rm = |r: &[f64; 2]| {
        let (a, b) = (r[0].max(0.0), r[1].max(0.0));
        if a + b > 0.0 { b / (a + b) } else { 0.5 }
    };
    for t in 1..=iters {
        let (sb, _) = values(s, m, &shove, &call);
        for h in 0..n {
            let v = (1.0 - shove[h]) * sb[h][0] + shove[h] * sb[h][1];
            for a in 0..2 {
                rs[h][a] = (rs[h][a] + sb[h][a] - v).max(0.0);
            }
            shove[h] = rm(&rs[h]);
        }
        let (_, bb) = values(s, m, &shove, &call);
        for g in 0..n {
            let v = (1.0 - call[g]) * bb[g][0] + call[g] * bb[g][1];
            for a in 0..2 {
                rc[g][a] = (rc[g][a] + bb[g][a] - v).max(0.0);
            }
            call[g] = rm(&rc[g]);
        }
        let wt = t as f64;
        wsum += wt;
        for i in 0..n {
            avg_s[i] += wt * shove[i];
            avg_c[i] += wt * call[i];
        }
    }
    (avg_s.iter().map(|x| x / wsum).collect(), avg_c.iter().map(|x| x / wsum).collect())
}

/// Exploitability of (shove, call) in chips per hand: the average of the two
/// best-response values.
pub fn exploitability(stack: i32, m: &Matchups, shove: &[f64], call: &[f64]) -> f64 {
    let n = N_CLASSES;
    let total: f64 = m.w.iter().sum();
    let (sb, bb) = values(stack as f64, m, shove, call);
    let br_sb: f64 = sb.iter().map(|v| v[0].max(v[1])).sum::<f64>() / total;
    // The big blind also collects the small blind when it folds.
    let mut folds = 0.0;
    for h in 0..n {
        for g in 0..n {
            folds += m.w[h * n + g] * (1.0 - shove[h]);
        }
    }
    let br_bb: f64 = (folds + bb.iter().map(|v| v[0].max(v[1])).sum::<f64>()) / total;
    (br_sb + br_bb) / 2.0
}
