//! All-in EV adjustment: once no player can make another decision, replace the
//! luck of the remaining board cards with its expectation.
//!
//! When betting is closed for the rest of the hand (at most one player can
//! still act and the street's betting is done), each seat's final stack
//! depends only on the cards still to come. Averaging the payout over those
//! runouts gives the same expected reward with far less variance. The deck
//! left in the state is exactly the set of undealt cards, so runouts are drawn
//! from it: enumerated when one or two cards remain, sampled preflop.

use rand::seq::index::sample;
use rand::Rng;

use crate::cards::Card;
use crate::hand_eval::eval7;
use crate::rules::{award_pots, build_pots, Phase, State, Status, SEATS};

/// Number of random runouts used when the whole board is still to come.
pub const PREFLOP_SAMPLES: usize = 2000;

/// True when no further decisions can happen this hand but the outcome still
/// depends on board cards: betting just closed, at least two players remain,
/// at most one of them can still act, and the board isn't complete.
pub fn is_runout(s: &State) -> bool {
    if s.phase != Phase::StreetComplete || s.community.len() >= 5 {
        return false;
    }
    let in_hand = s
        .players
        .iter()
        .filter(|p| matches!(p.status, Status::Active | Status::AllIn))
        .count();
    let can_act = s.players.iter().filter(|p| p.status == Status::Active).count();
    in_hand >= 2 && can_act <= 1
}

/// Expected final stacks for a state where `is_runout` holds.
pub fn expected_stacks(s: &State, rng: &mut impl Rng) -> [f64; SEATS] {
    let pots = build_pots(&s.players);
    let need = 5 - s.community.len();
    let deck = &s.deck;
    let mut board: Vec<Card> = s.community.clone();
    let mut total = [0f64; SEATS];
    let mut n = 0usize;

    let payout = |board: &[Card], total: &mut [f64; SEATS]| {
        let mut strength = [None; SEATS];
        let mut cards = [0u8; 7];
        cards[2..7].copy_from_slice(board);
        for (i, p) in s.players.iter().enumerate() {
            if matches!(p.status, Status::Active | Status::AllIn) {
                cards[0] = p.hole[0];
                cards[1] = p.hole[1];
                strength[i] = Some(eval7(&cards));
            }
        }
        let (_, won) = award_pots(&pots, &strength, s.button);
        for i in 0..SEATS {
            total[i] += won[i] as f64;
        }
    };

    match need {
        1 => {
            for &a in deck {
                board.push(a);
                payout(&board, &mut total);
                board.pop();
                n += 1;
            }
        }
        2 => {
            for i in 0..deck.len() {
                for j in i + 1..deck.len() {
                    board.push(deck[i]);
                    board.push(deck[j]);
                    payout(&board, &mut total);
                    board.truncate(board.len() - 2);
                    n += 1;
                }
            }
        }
        _ => {
            for _ in 0..PREFLOP_SAMPLES {
                for k in sample(rng, deck.len(), need) {
                    board.push(deck[k]);
                }
                payout(&board, &mut total);
                board.truncate(board.len() - need);
                n += 1;
            }
        }
    }

    let mut out = [0f64; SEATS];
    for i in 0..SEATS {
        out[i] = s.players[i].stack as f64 + total[i] / n as f64;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::{full_deck, parse};
    use crate::rules::{Action, ActionType};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn c(id: &str) -> Card {
        parse(id).unwrap()
    }

    /// A hand with the given hole cards (seat 0 first) and stacks; the button
    /// (small blind) is seat 0.
    fn deal(holes: [[&str; 2]; 2], stacks: [i32; SEATS]) -> State {
        let mut s = State::new_table(stacks);
        // Build a deck whose pops deal the requested hole cards in seat order.
        let wanted: Vec<Card> = holes.iter().flat_map(|h| [c(h[0]), c(h[1])]).collect();
        let mut deck: Vec<Card> = full_deck().into_iter().filter(|x| !wanted.contains(x)).collect();
        for &w in wanted.iter().rev() {
            deck.push(w);
        }
        s.start_hand(deck);
        assert_eq!(s.button, 0);
        s
    }

    /// Exact average over runouts by dealing each one through the rules
    /// engine (brute-force reference for `expected_stacks`).
    fn brute_force(s: &State) -> [f64; SEATS] {
        let need = 5 - s.community.len();
        let deck = s.deck.clone();
        let mut total = [0f64; SEATS];
        let mut n = 0;
        let mut run = |cards: &[Card]| {
            let mut t = s.clone();
            // advance_street pops from the end: put the runout last, in order.
            let mut d: Vec<Card> = deck.iter().copied().filter(|x| !cards.contains(x)).collect();
            d.extend(cards.iter().rev());
            t.deck = d;
            while !t.hand_done() {
                t.advance_street();
            }
            for i in 0..SEATS {
                total[i] += t.players[i].stack as f64;
            }
            n += 1;
        };
        match need {
            1 => deck.iter().for_each(|&a| run(&[a])),
            2 => {
                for i in 0..deck.len() {
                    for j in i + 1..deck.len() {
                        run(&[deck[i], deck[j]]);
                    }
                }
            }
            _ => unreachable!(),
        }
        total.map(|t| t / n as f64)
    }

    #[test]
    fn flop_and_turn_allins_match_brute_force() {
        // Unequal stacks: the bigger stack's uncalled excess comes back.
        let mut s = deal([["Ah", "Kd"], ["7h", "8h"]], [300, 100]);
        s.apply(Action::new(ActionType::Call)); // SB limps
        s.apply(Action::new(ActionType::Check)); // BB checks
        s.advance_street(); // flop: BB first
        s.apply(Action::to(ActionType::Bet, 10));
        s.apply(Action::new(ActionType::AllIn)); // SB shoves 298
        s.apply(Action::new(ActionType::Call)); // BB calls all-in for 98
        assert!(is_runout(&s), "phase {:?}", s.phase);
        let mut rng = StdRng::seed_from_u64(1);
        let ev = expected_stacks(&s, &mut rng);
        let bf = brute_force(&s);
        for i in 0..SEATS {
            assert!((ev[i] - bf[i]).abs() < 1e-9, "seat {i}: {} vs {}", ev[i], bf[i]);
        }
        assert!((ev.iter().sum::<f64>() - 400.0).abs() < 1e-9);
        assert!(ev[0] >= 200.0, "the SB keeps its uncalled 200 at worst");

        let mut t = s.clone();
        t.advance_street(); // turn
        assert!(is_runout(&t));
        let ev = expected_stacks(&t, &mut rng);
        let bf = brute_force(&t);
        for i in 0..SEATS {
            assert!((ev[i] - bf[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn preflop_aces_vs_kings() {
        let mut s = deal([["Ah", "As"], ["Kh", "Ks"]], [200, 200]);
        s.apply(Action::new(ActionType::AllIn));
        s.apply(Action::new(ActionType::Call));
        assert!(is_runout(&s));
        let mut rng = StdRng::seed_from_u64(7);
        let ev = expected_stacks(&s, &mut rng);
        // AA vs KK (two suits shared) wins about 82% of the time.
        assert!((ev[0] / 400.0 - 0.82).abs() < 0.03, "AA equity {}", ev[0] / 400.0);
        assert!((ev[0] + ev[1] - 400.0).abs() < 1e-9);
    }

    #[test]
    fn blind_allin_is_a_runout() {
        // SB all-in for 1 chip posting the blind: no decision, straight to
        // the run-out.
        let s = deal([["Ah", "As"], ["Kh", "Ks"]], [1, 399]);
        assert_eq!(s.phase, Phase::StreetComplete);
        assert!(is_runout(&s));
        let mut rng = StdRng::seed_from_u64(3);
        let ev = expected_stacks(&s, &mut rng);
        assert!((ev[0] + ev[1] - 400.0).abs() < 1e-9);
        assert!(ev[0] > 1.5 && ev[0] < 2.0, "SB wins 2 chips ~82%: {}", ev[0]);
    }
}
