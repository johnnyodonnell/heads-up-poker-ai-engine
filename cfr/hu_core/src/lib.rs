//! Heads-up No-Limit Hold'em core: the game rules, cards, a 7-card hand
//! evaluator, equity and all-in EV. No I/O.
//!
//! The rules are a faithful port of `src/engine/game.ts` (the browser game is
//! the source of truth); parity is checked against traces dumped from the TS
//! engine (`cfr/scripts/dump_fixtures.ts`).

pub mod allin_ev;
pub mod cards;
pub mod equity;
pub mod hand_eval;
pub mod rules;
pub mod table;
