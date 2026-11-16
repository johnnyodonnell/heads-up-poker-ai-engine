//! The CFR engine: card abstraction, abstract betting trees and bet-size
//! menus, external-sampling MCCFR, exported strategies, the play-time engine
//! (history translation and action mapping), evaluation matches against
//! scripted bots, and the validation games (push/fold, Kuhn, Leduc).

pub mod abs;
pub mod bots;
pub mod br;
pub mod classes;
pub mod engine;
pub mod eval;
pub mod game;
pub mod mccfr;
pub mod menu;
pub mod pushfold;
pub mod strategy;
pub mod toy;
pub mod tree;
