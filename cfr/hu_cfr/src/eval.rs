//! Matches between the engine and scripted bots (or another engine), played
//! on the real rules with the exact decision path the server uses.
//!
//! Duplicate format: every deal (stacks, button, deck) is played twice with
//! the players' seats swapped, so both see the same cards from both sides.
//! Both plays of a deal also use the same random draws, so two players that
//! decide alike play the two hands identically and their luck cancels within
//! the deal; only the decisions where they differ leave a trace. Once no
//! more decisions are possible, the board's remaining luck is replaced by
//! its expectation (all-in EV). Result: player A's bb/100 with a 95%
//! confidence interval from the per-deal variance.
//!
//! Deals and draws depend only on the seed and the deal's index, so matches
//! of different players against the same opponent with the same seed are
//! paired deal by deal (`compare`): the opponent sees the same cards and
//! makes the same random choices until the two players' decisions differ.

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use hu_core::allin_ev::{expected_stacks, is_runout};
use hu_core::rules::{Phase, State, BIG_BLIND, STARTING_STACK};
use hu_core::table::sample_stacks;

use crate::bots::Bot;
use crate::engine::{view_of, Engine};
use crate::live::LiveMode;

pub enum Player<'a> {
    /// An engine, solving live per the mode.
    Engine(&'a Engine, LiveMode),
    Bot(Bot),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stacks {
    /// Both players start every hand with 100bb.
    Full,
    /// The app's freeze-out spread: 40% 100bb, else a random split of 400 chips.
    Sampled,
    /// Both players start every hand with this many chips (e.g. 400 = 200bb,
    /// Slumbot's depth).
    Fixed(i32),
}

#[derive(Debug)]
pub struct MatchResult {
    pub bb100: f64,
    pub ci95: f64,
    pub hands: usize,
    /// Engine decisions, and how many used the equity fallback.
    pub decisions: usize,
    pub fallbacks: usize,
    /// Player A's result per deal, in chips (mean of the two plays).
    pub per_deal: Vec<f64>,
}

/// Mean and 95% interval (bb/100) of per-deal chip results.
pub fn summarize(per_deal: &[f64]) -> (f64, f64) {
    let n = per_deal.len() as f64;
    let mean = per_deal.iter().sum::<f64>() / n;
    let var = per_deal.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let to_bb100 = 100.0 / BIG_BLIND as f64;
    (mean * to_bb100, 1.96 * (var / n).sqrt() * to_bb100)
}

/// A − B per deal, for two matches with the same seed and deal count
/// (paired: same deals, same draws): mean and 95% interval in bb/100.
pub fn compare(a: &[f64], b: &[f64]) -> (f64, f64) {
    assert_eq!(a.len(), b.len(), "paired matches need the same deals");
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    summarize(&d)
}

struct Counters {
    decisions: AtomicUsize,
    fallbacks: AtomicUsize,
}

/// Chips won by the player in seat `a_seat` in one hand.
fn play_hand(players: [&Player; 2], stacks: [i32; 2], button: usize, deck: Vec<u8>, a_seat: usize, seed: u64, c: &Counters) -> f64 {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut s = State::new_table(stacks);
    s.button = (1 - button) as i32;
    s.start_hand(deck);
    let start = stacks;
    loop {
        match s.phase {
            Phase::Betting => {
                let seat = s.to_act as usize;
                let action = match players[seat] {
                    Player::Bot(b) => b.act(&s, &mut rng),
                    Player::Engine(e, mode) => {
                        let v = view_of(&s, seat, start, seed);
                        let d = e.decide_mode(&v, &mut rng, *mode).unwrap_or_else(|err| panic!("engine error: {err}"));
                        c.decisions.fetch_add(1, Relaxed);
                        if d.debug.fallback.is_some() {
                            c.fallbacks.fetch_add(1, Relaxed);
                        }
                        d.action
                    }
                };
                s.apply(action);
            }
            Phase::StreetComplete => {
                if is_runout(&s) {
                    let ev = expected_stacks(&s, &mut rng);
                    return ev[a_seat] - start[a_seat] as f64;
                }
                s.advance_street();
            }
            Phase::HandOver | Phase::MatchOver => return (s.players[a_seat].stack - start[a_seat]) as f64,
            Phase::Waiting => unreachable!(),
        }
    }
}

/// Player A vs player B over `deals` duplicate deals.
pub fn play_match(a: &Player, b: &Player, deals: usize, stacks: Stacks, seed: u64) -> MatchResult {
    let c = Counters { decisions: AtomicUsize::new(0), fallbacks: AtomicUsize::new(0) };
    let per_deal: Vec<f64> = (0..deals)
        .into_par_iter()
        .map(|d| {
            let dseed = seed ^ (d as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut rng = SmallRng::seed_from_u64(dseed);
            let st = match stacks {
                Stacks::Full => [STARTING_STACK; 2],
                Stacks::Sampled => sample_stacks(&mut rng),
                Stacks::Fixed(c) => [c; 2],
            };
            let button = rng.gen_range(0..2usize);
            let mut deck: Vec<u8> = (0..52).collect();
            deck.shuffle(&mut rng);
            // The plays' draws: shared by both plays, independent of the
            // deal's own draws.
            let pseed = SmallRng::seed_from_u64(dseed ^ 0x5DEE_CE66_D1CE_4E5B).gen::<u64>();
            let mut total = 0.0;
            for a_seat in 0..2 {
                let players: [&Player; 2] = if a_seat == 0 { [a, b] } else { [b, a] };
                total += play_hand(players, st, button, deck.clone(), a_seat, pseed, &c);
            }
            total / 2.0
        })
        .collect();
    let (bb100, ci95) = summarize(&per_deal);
    MatchResult { bb100, ci95, hands: deals * 2, decisions: c.decisions.load(Relaxed), fallbacks: c.fallbacks.load(Relaxed), per_deal }
}
