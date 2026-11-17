//! Terminal payoffs for every hand at once against a weighted opponent
//! range, with card removal (two hands that share a card never meet).
//!
//! Hands are indexed by `abs::index::hole_index` (1,326 of them). Showdowns
//! sort the board's hands by strength once, then sweep up and down keeping
//! the opponent weight below/above and the part of it holding each card;
//! the same idea as `abs::equity::river_ehs`, with weights.

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

/// `out[h] = v ×` (opponent weight of hands disjoint from `h`).
pub fn fold(opp: &[f32], v: f32, out: &mut [f32]) {
    let hs = holes();
    let mut card = [0f32; 52];
    let mut total = 0f32;
    for h in 0..H {
        let w = opp[h];
        if w != 0.0 {
            let [a, b] = hs[h];
            card[a as usize] += w;
            card[b as usize] += w;
            total += w;
        }
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

/// One complete (5-card) board: its hands sorted by strength.
pub struct Board {
    pub cards: [Card; 5],
    /// (strength, hand, card a, card b), weakest first; hands touching the
    /// board are left out.
    hands: Vec<(u32, u16, u8, u8)>,
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
        Board { cards, hands }
    }

    /// Showdown for `s` chips each: `out[h] = s × (opponent weight `h` beats
    /// − weight that beats `h`)`, disjoint hands only, ties paying nothing.
    /// Hands touching the board get 0.
    pub fn showdown(&self, opp: &[f32], s: f32, out: &mut [f32]) {
        out.fill(0.0);
        let hs = &self.hands;
        let n = hs.len();
        // Up: weight strictly below each hand.
        let mut below = 0f32;
        let mut card = [0f32; 52];
        let mut i = 0;
        while i < n {
            let mut j = i;
            while j < n && hs[j].0 == hs[i].0 {
                j += 1;
            }
            for &(_, h, a, b) in &hs[i..j] {
                out[h as usize] = below - card[a as usize] - card[b as usize];
            }
            for &(_, h, a, b) in &hs[i..j] {
                let w = opp[h as usize];
                below += w;
                card[a as usize] += w;
                card[b as usize] += w;
            }
            i = j;
        }
        // Down: weight strictly above.
        let mut above = 0f32;
        let mut card = [0f32; 52];
        let mut j = n;
        while j > 0 {
            let mut i = j;
            while i > 0 && hs[i - 1].0 == hs[j - 1].0 {
                i -= 1;
            }
            for &(_, h, a, b) in &hs[i..j] {
                let o = &mut out[h as usize];
                *o = s * (*o - (above - card[a as usize] - card[b as usize]));
            }
            for &(_, h, a, b) in &hs[i..j] {
                let w = opp[h as usize];
                above += w;
                card[a as usize] += w;
                card[b as usize] += w;
            }
            j = i;
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
