//! Terminal payoffs for every hand at once against a weighted opponent
//! range, with card removal (two hands that share a card never meet).
//!
//! Hands are indexed by `abs::index::hole_index` (1,326 of them). Showdowns
//! sort the board's hands by strength once, then sweep up and down keeping
//! the opponent weight below/above and the part of it holding each card;
//! the same idea as `abs::equity::river_ehs`, with weights.
//!
//! Long sums are taken in a fixed chunked order (`chunked_prefix`, fold's
//! total) that the GPU solver (`super::gpu`) can take in parallel, so both
//! give bit-identical results.

use std::sync::OnceLock;

use hu_core::cards::Card;
use hu_core::hand_eval::eval7;

use crate::abs::index::{hole_cards, hole_index, N_HOLES};

pub const H: usize = N_HOLES;

/// The two cards of every hand index.
pub fn holes() -> &'static [[Card; 2]; H] {
    static HOLES: OnceLock<[[Card; 2]; H]> = OnceLock::new();
    HOLES.get_or_init(|| {
        let mut t = [[0u8; 2]; H];
        for (h, c) in t.iter_mut().enumerate() {
            *c = hole_cards(h as u32);
        }
        t
    })
}

/// The 51 hands holding each card.
pub fn with_card() -> &'static [[u16; 51]; 52] {
    static T: OnceLock<[[u16; 51]; 52]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [[0u16; 51]; 52];
        let mut n = [0usize; 52];
        for (h, &[a, b]) in holes().iter().enumerate() {
            for c in [a, b] {
                t[c as usize][n[c as usize]] = h as u16;
                n[c as usize] += 1;
            }
        }
        t
    })
}

/// Index of the hand `[a, b]`.
#[inline]
pub fn hand(cards: [Card; 2]) -> usize {
    hole_index(cards[0], cards[1]) as usize
}

/// Zero the weight of every hand holding one of `cards`.
pub fn mask(w: &mut [f32], cards: &[Card]) {
    let hs = holes();
    for (h, x) in w.iter_mut().enumerate() {
        let [a, b] = hs[h];
        if cards.contains(&a) || cards.contains(&b) {
            *x = 0.0;
        }
    }
}

/// Hands per chunk of fold's total (the GPU sums the same chunks).
pub const FOLD_CHUNK: usize = 64;
/// Positions per chunk of the showdown sweeps' sums.
pub const CHUNK: usize = 32;

/// Exclusive prefix sums of `x(0..n)` into `out[..n]`, in a fixed order the
/// GPU solver takes in parallel: `CHUNK`-element chunks each summed in
/// order, the chunk sums accumulated in order (each chunk's start), and
/// within a chunk its start plus its elements in order.
pub fn chunked_prefix(n: usize, x: impl Fn(usize) -> f32, out: &mut [f32]) {
    let chunks = n.div_ceil(CHUNK);
    let mut sums = [0f32; H / CHUNK + 1];
    for (c, s) in sums.iter_mut().enumerate().take(chunks) {
        let mut t = 0f32;
        for j in c * CHUNK..((c + 1) * CHUNK).min(n) {
            t += x(j);
        }
        *s = t;
    }
    let mut start = 0f32;
    for (c, s) in sums.iter().enumerate().take(chunks) {
        let mut t = start;
        for j in c * CHUNK..((c + 1) * CHUNK).min(n) {
            out[j] = t;
            t += x(j);
        }
        start += s;
    }
}

/// `out[h] = v ×` (opponent weight of hands disjoint from `h`).
pub fn fold(opp: &[f32], v: f32, out: &mut [f32]) {
    let hs = holes();
    let mut card = [0f32; 52];
    for h in 0..H {
        let w = opp[h];
        if w != 0.0 {
            let [a, b] = hs[h];
            card[a as usize] += w;
            card[b as usize] += w;
        }
    }
    let mut total = 0f32;
    for chunk in opp.chunks(FOLD_CHUNK) {
        let mut t = 0f32;
        for &w in chunk {
            t += w;
        }
        total += t;
    }
    for h in 0..H {
        let [a, b] = hs[h];
        out[h] = v * (total - card[a as usize] - card[b as usize] + opp[h]);
    }
}

/// Total weight of disjoint hand pairs: Σ_h a[h] × (b's weight disjoint from h).
pub fn pair_weight(a: &[f32], b: &[f32]) -> f64 {
    let mut d = vec![0f32; H];
    fold(b, 1.0, &mut d);
    a.iter().zip(&d).map(|(&x, &y)| x as f64 * y as f64).sum()
}

/// One complete (5-card) board: its hands sorted by strength, stored as
/// parallel arrays with the boundaries of equal-strength groups.
pub struct Board {
    pub cards: [Card; 5],
    /// Hand index of each sorted position (hands touching the board left out).
    idx: Vec<u16>,
    /// Its two cards.
    ca: Vec<u8>,
    cb: Vec<u8>,
    /// Start of each group of equal strength, then the number of hands.
    groups: Vec<u32>,
    /// The positions with groups descending, ascending within a group (the
    /// down sweep's order).
    down: Vec<u16>,
}

impl Board {
    pub fn new(cards: [Card; 5]) -> Board {
        let mut seven = [0u8; 7];
        seven[2..].copy_from_slice(&cards);
        let mut hands = Vec::with_capacity(1081);
        for (h, &[a, b]) in holes().iter().enumerate() {
            if cards.contains(&a) || cards.contains(&b) {
                continue;
            }
            seven[0] = a;
            seven[1] = b;
            hands.push((eval7(&seven), h as u16, a, b));
        }
        hands.sort_unstable_by_key(|x| (x.0, x.1));
        let mut groups = Vec::new();
        for (i, x) in hands.iter().enumerate() {
            if i == 0 || hands[i - 1].0 != x.0 {
                groups.push(i as u32);
            }
        }
        groups.push(hands.len() as u32);
        let mut down = Vec::with_capacity(hands.len());
        for g in groups.windows(2).rev() {
            down.extend(g[0] as u16..g[1] as u16);
        }
        Board {
            cards,
            idx: hands.iter().map(|x| x.1).collect(),
            ca: hands.iter().map(|x| x.2).collect(),
            cb: hands.iter().map(|x| x.3).collect(),
            groups,
            down,
        }
    }

    /// The sorted hands' indices and cards, and the group boundaries (for
    /// the GPU solver).
    pub fn parts(&self) -> (&[u16], &[u8], &[u8], &[u32], &[u16]) {
        (&self.idx, &self.ca, &self.cb, &self.groups, &self.down)
    }

    /// Showdown for `s` chips each: `out[h] = s × (opponent weight `h` beats
    /// − weight that beats `h`)`, disjoint hands only, ties paying nothing.
    /// Hands touching the board get 0.
    pub fn showdown(&self, opp: &[f32], s: f32, out: &mut [f32]) {
        let n = self.idx.len();
        let (idx, ca, cb) = (&self.idx[..], &self.ca[..], &self.cb[..]);
        let mut w = [0f32; H];
        for k in 0..n {
            w[k] = opp[idx[k] as usize];
        }
        // The weight below each position and above each (in the down
        // order), as chunked sums; read at each group's first position.
        let mut below = [0f32; H];
        chunked_prefix(n, |k| w[k], &mut below);
        let mut above = [0f32; H];
        chunked_prefix(n, |j| w[self.down[j] as usize], &mut above);
        // Up: minus each card's part of the weight below (hands sharing a
        // card never meet).
        let mut net = [0f32; H];
        let mut card = [0f32; 52];
        for g in self.groups.windows(2) {
            let (i, j) = (g[0] as usize, g[1] as usize);
            for k in i..j {
                net[k] = below[i] - card[ca[k] as usize] - card[cb[k] as usize];
            }
            for k in i..j {
                card[ca[k] as usize] += w[k];
                card[cb[k] as usize] += w[k];
            }
        }
        // Down: minus the weight strictly above, likewise.
        let mut card = [0f32; 52];
        for g in self.groups.windows(2).rev() {
            let (i, j) = (g[0] as usize, g[1] as usize);
            let a = above[n - j];
            for k in i..j {
                net[k] -= a - card[ca[k] as usize] - card[cb[k] as usize];
            }
            for k in i..j {
                card[ca[k] as usize] += w[k];
                card[cb[k] as usize] += w[k];
            }
        }
        out.fill(0.0);
        for k in 0..n {
            out[idx[k] as usize] = s * net[k];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::seq::SliceRandom;
    use rand::{Rng, SeedableRng};

    fn disjoint(x: [Card; 2], y: [Card; 2]) -> bool {
        !x.contains(&y[0]) && !x.contains(&y[1])
    }

    #[test]
    fn sweeps_match_brute_force() {
        let mut rng = SmallRng::seed_from_u64(5);
        let hs = holes();
        for _ in 0..4 {
            let mut deck: Vec<Card> = (0..52).collect();
            deck.shuffle(&mut rng);
            let cards = [deck[0], deck[1], deck[2], deck[3], deck[4]];
            let board = Board::new(cards);
            let mut opp: Vec<f32> = (0..H).map(|_| if rng.gen_bool(0.3) { 0.0 } else { rng.gen::<f32>() }).collect();
            mask(&mut opp, &cards);
            let mut sd = vec![0f32; H];
            let mut fd = vec![0f32; H];
            board.showdown(&opp, 3.0, &mut sd);
            fold(&opp, 2.0, &mut fd);
            let strength = |x: [Card; 2]| {
                let mut c = vec![x[0], x[1]];
                c.extend_from_slice(&cards);
                eval7(&c)
            };
            for h in (0..H).step_by(7) {
                let x = hs[h];
                if cards.contains(&x[0]) || cards.contains(&x[1]) {
                    assert_eq!(sd[h], 0.0);
                    continue;
                }
                let (mut net, mut tot) = (0f64, 0f64);
                for h2 in 0..H {
                    let y = hs[h2];
                    if !disjoint(x, y) {
                        continue;
                    }
                    let w = opp[h2] as f64;
                    tot += w;
                    let (sx, sy) = (strength(x), strength(y));
                    if sx > sy {
                        net += w;
                    } else if sx < sy {
                        net -= w;
                    }
                }
                assert!((sd[h] as f64 - 3.0 * net).abs() < 1e-2, "showdown {h}: {} vs {}", sd[h], 3.0 * net);
                assert!((fd[h] as f64 - 2.0 * tot).abs() < 1e-2, "fold {h}: {} vs {}", fd[h], 2.0 * tot);
            }
        }
    }
}
