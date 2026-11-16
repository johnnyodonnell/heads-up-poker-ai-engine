//! The CFR engine's solver side: abstract betting trees, bet-size menus,
//! preflop classes, external-sampling MCCFR, exact best responses for small
//! games, and the push/fold and Kuhn/Leduc validation games.

pub mod br;
pub mod classes;
pub mod game;
pub mod mccfr;
pub mod menu;
pub mod pushfold;
pub mod toy;
pub mod tree;
