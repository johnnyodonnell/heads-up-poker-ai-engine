//! Fixed scripted opponents for evaluation. None of them is good, but each is
//! exploitable in a different direction, so a sound policy should beat all of
//! them — and a policy drifting toward nit or maniac play shows up as losing
//! (or winning much less) against one of them, which snapshot-vs-snapshot
//! ratings can hide. Reported separately from the ladder.
//!
//!   random   port of src/engine/random.ts (the app's original bot)
//!   station  always checks or calls; never folds, never raises
//!   maniac   pot-size bet/raise whenever allowed, otherwise calls
//!   nit      plays only premium hands preflop; after the flop continues only
//!            with top pair / overpair or better (bets 2/3 pot, else calls)
//!   shover   all-in preflop with any two cards; calls an all-in

use rand::Rng;

use hu_core::cards::{rank, Card};
use hu_core::hand_eval::{category, eval7, PAIR, TWO_PAIR};
use hu_core::rules::{Action, ActionType, Legal, State, Street};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bot {
    Random,
    Station,
    Maniac,
    Nit,
    Shover,
}

pub const RANDOM: &str = "random";
/// The scripted panel (random is reported on its own, as fox-lite's collapse alarm).
pub const PANEL: [&str; 4] = ["station", "maniac", "nit", "shover"];

impl Bot {
    pub fn parse(name: &str) -> Option<Bot> {
        Some(match name {
            "random" => Bot::Random,
            "station" => Bot::Station,
            "maniac" => Bot::Maniac,
            "nit" => Bot::Nit,
            "shover" => Bot::Shover,
            _ => return None,
        })
    }

    pub fn act(self, st: &State, rng: &mut impl Rng) -> Action {
        let l = st.legal().expect("legal");
        match self {
            Bot::Random => random_action(&l, rng),
            Bot::Station => passive(&l),
            Bot::Maniac => {
                if l.can_raise {
                    raise_to(&l, st, 1.0)
                } else {
                    passive(&l)
                }
            }
            Bot::Nit => nit(st, &l),
            Bot::Shover => {
                if st.street == Street::Preflop && l.can_raise {
                    Action::new(ActionType::AllIn)
                } else {
                    passive(&l)
                }
            }
        }
    }
}

/// Port of src/engine/random.ts: a uniform legal action type, and a uniform
/// raise-to between the min-raise and all-in.
fn random_action(l: &Legal, rng: &mut impl Rng) -> Action {
    let mut kinds = Vec::with_capacity(3);
    if l.can_fold {
        kinds.push(ActionType::Fold);
    }
    kinds.push(if l.can_check { ActionType::Check } else { ActionType::Call });
    if l.can_raise {
        kinds.push(if l.opening { ActionType::Bet } else { ActionType::Raise });
    }
    let k = kinds[rng.gen_range(0..kinds.len())];
    match k {
        ActionType::Bet | ActionType::Raise => Action::to(k, rng.gen_range(l.min_raise_to..=l.max_raise_to)),
        _ => Action::new(k),
    }
}

fn passive(l: &Legal) -> Action {
    Action::new(if l.can_check { ActionType::Check } else { ActionType::Call })
}

fn check_or_fold(l: &Legal) -> Action {
    Action::new(if l.can_check { ActionType::Check } else { ActionType::Fold })
}

/// Bet/raise by `frac` of the pot after calling (clamped to the legal range).
fn raise_to(l: &Legal, st: &State, frac: f64) -> Action {
    let p = &st.players[st.to_act as usize];
    let pot = st.pot + st.players.iter().map(|q| q.bet).sum::<i32>();
    let pot_after_call = pot + (st.current_bet - p.bet);
    let to = (st.current_bet as f64 + frac * pot_after_call as f64).round() as i32;
    let kind = if l.opening { ActionType::Bet } else { ActionType::Raise };
    Action::to(kind, to.clamp(l.min_raise_to, l.max_raise_to))
}

/// Pairs 88+, AK, AQ, AJs, KQs.
fn premium(hole: &[Card]) -> bool {
    let (a, b) = (rank(hole[0]), rank(hole[1]));
    let (hi, lo) = (a.max(b), a.min(b));
    let suited = hole[0] / 13 == hole[1] / 13;
    (hi == lo && hi >= 8) || (hi == 14 && lo >= 12) || (hi == 14 && lo == 11 && suited) || (hi == 13 && lo == 12 && suited)
}

/// Top pair (using a hole card) / overpair, or two pair and better.
fn strong_made_hand(hole: &[Card], board: &[Card]) -> bool {
    let mut cards = hole.to_vec();
    cards.extend_from_slice(board);
    let v = eval7(&cards);
    let cat = category(v);
    if cat >= TWO_PAIR {
        return true;
    }
    if cat != PAIR {
        return false;
    }
    let pair_rank = ((v >> 16) & 0xF) as u8;
    let top = board.iter().map(|&c| rank(c)).max().unwrap_or(0);
    hole.iter().any(|&c| rank(c) == pair_rank) && pair_rank >= top
}

fn nit(st: &State, l: &Legal) -> Action {
    let hole = &st.players[st.to_act as usize].hole;
    let strong = if st.street == Street::Preflop {
        premium(hole)
    } else {
        strong_made_hand(hole, &st.community)
    };
    if !strong {
        return check_or_fold(l);
    }
    if l.can_raise {
        raise_to(l, st, if st.street == Street::Preflop { 1.0 } else { 0.67 })
    } else {
        passive(l)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hu_core::cards::parse;

    fn cards(ids: &str) -> Vec<Card> {
        ids.split_whitespace().map(|s| parse(s).unwrap()).collect()
    }

    #[test]
    fn nit_hand_reading() {
        assert!(premium(&cards("Ah Kd")));
        assert!(premium(&cards("8c 8d")));
        assert!(!premium(&cards("7c 7d")));
        assert!(!premium(&cards("Ah Jd")));
        assert!(premium(&cards("Ah Jh")));
        assert!(strong_made_hand(&cards("Ah 2d"), &cards("Ac 9s 4h"))); // top pair
        assert!(strong_made_hand(&cards("Qh Qd"), &cards("Jc 9s 4h"))); // overpair
        assert!(!strong_made_hand(&cards("9h 2d"), &cards("Ac 9s 4h"))); // middle pair
        assert!(!strong_made_hand(&cards("Kh Qd"), &cards("Ac 9s 4h"))); // air
        assert!(strong_made_hand(&cards("9h 4d"), &cards("Ac 9s 4h"))); // two pair
    }
}
