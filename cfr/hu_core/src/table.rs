//! Random hands for evaluation matches: each hand starts from a freshly
//! sampled stack split, so a match covers every stack depth the app produces.
//!
//! The app is a heads-up freeze-out: 2 × 200 chips to start, then the stacks
//! drift, but they always sum to 400. So 40% of hands start at 200/200; the
//! rest split the 400 chips with a Beta(1.5, 1.5) share (each player keeps at
//! least 1 chip). The button is random.

use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::{Beta, Distribution};

use crate::cards::full_deck;
use crate::rules::{State, SEATS, STARTING_STACK, TOTAL_CHIPS};

pub const FRESH_TABLE_PROB: f64 = 0.4;
/// Beta concentration of the stack split (lower = more lopsided).
const SPLIT_ALPHA: f64 = 1.5;

/// Stacks for a sampled hand.
pub fn sample_stacks(rng: &mut impl Rng) -> [i32; SEATS] {
    if rng.gen_bool(FRESH_TABLE_PROB) {
        return [STARTING_STACK; SEATS];
    }
    let share = Beta::new(SPLIT_ALPHA, SPLIT_ALPHA).unwrap().sample(rng);
    let a = 1 + (share * (TOTAL_CHIPS - 2) as f64).floor() as i32;
    let a = a.clamp(1, TOTAL_CHIPS - 1);
    [a, TOTAL_CHIPS - a]
}

/// A new hand: sampled stacks, random button, shuffled deck, dealt.
pub fn deal_random_hand(rng: &mut impl Rng) -> State {
    let mut s = State::new_table(sample_stacks(rng));
    s.button = rng.gen_range(0..SEATS as i32); // start_hand moves it on one seat
    let mut deck = full_deck();
    deck.shuffle(rng);
    s.start_hand(deck);
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn stacks_always_sum_to_400() {
        let mut rng = StdRng::seed_from_u64(3);
        let (mut fresh, mut short) = (0, 0);
        for _ in 0..20000 {
            let st = sample_stacks(&mut rng);
            assert_eq!(st.iter().sum::<i32>(), TOTAL_CHIPS);
            assert!(st.iter().all(|&x| x >= 1));
            if st == [STARTING_STACK; SEATS] {
                fresh += 1;
            }
            if st.iter().any(|&x| x <= 20) {
                short += 1;
            }
        }
        assert!((7000..9000).contains(&fresh), "fresh {fresh}");
        assert!(short > 300, "short-stack hands {short}");
    }

    #[test]
    fn buttons_are_random() {
        let mut rng = StdRng::seed_from_u64(5);
        let mut seat0 = 0;
        for _ in 0..2000 {
            if deal_random_hand(&mut rng).button == 0 {
                seat0 += 1;
            }
        }
        assert!((800..1200).contains(&seat0), "{seat0}");
    }
}
