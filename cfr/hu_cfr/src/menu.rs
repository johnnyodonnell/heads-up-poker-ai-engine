//! Bet-size menus for the abstract game.
//!
//! A size is a fraction of the pot *after calling* (the convention the app's
//! old action space used): raise-to = current bet + f × (pot + to-call). The
//! k-th bet/raise on a street uses list k (the last list repeats). All-in is
//! always offered when a raise is legal, and counts as a raise; after
//! `max_raises` bets/raises on a street only fold/call remain, or (with
//! `allin_past_cap`) fold/call/all-in.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Menu {
    pub name: String,
    pub preflop: Vec<Vec<f64>>,
    pub postflop: Vec<Vec<f64>>,
    pub max_raises: u32,
    /// Per street after the flop (flop, turn, river): lists that replace
    /// `postflop` there, when given.
    #[serde(default)]
    pub by_street: Option<[Vec<Vec<f64>>; 3]>,
    /// All-in stays an option past `max_raises` (so a 4-bet can be 5-bet
    /// all-in, and a raise beyond the cap has a node to land on).
    #[serde(default)]
    pub allin_past_cap: bool,
}

impl Menu {
    /// v1: a lean menu plus the sizes humans commonly use.
    ///
    /// Preflop, the first raise (an open, or a raise over a limp) is to 2, 2.5,
    /// 3 or 4 big blinds (0.5, 0.75, 1, 1.5 × the 4-chip pot after calling);
    /// re-raises are ~2.5× and ~3.5× the raise; a 4-bet ~2.2×. After the flop:
    /// bets of a third, two thirds or a full pot, then pot-size raises.
    pub fn lean() -> Menu {
        Menu {
            name: "lean".into(),
            preflop: vec![vec![0.5, 0.75, 1.0, 1.5], vec![0.75, 1.25], vec![0.75]],
            postflop: vec![vec![0.33, 0.66, 1.0], vec![1.0]],
            max_raises: 3,
            by_street: None,
            allin_past_cap: false,
        }
    }

    /// Candidates for v2, aimed at the holes local best response found in
    /// the lean menu (`hu lbr`). Each adds to the one before:
    /// - `cap`: lean, with all-in past the raise cap (5-bets, fourth raises).
    /// - `wide`: also small raises (half pot) and overbets (1.5×, 2.5× pot)
    ///   on the flop and turn, a 6bb open and a 2× 3-bet preflop; the river
    ///   stays lean (it is solved live, the table only supplies its values).
    /// - `wide-river`: `wide` with the river as wide as the turn.
    /// - `wide-flop`: `wide` before the turn only; the turn and river lean
    ///   (served with live turn, the table's turn and river play no hands:
    ///   they shape the flop's training and the live solves' safe values).
    pub fn candidate(name: &str) -> Option<Menu> {
        let mut m = Menu::lean();
        m.name = name.into();
        m.allin_past_cap = true;
        match name {
            "cap" => {}
            "wide" | "wide-river" | "wide-flop" => {
                m.preflop = vec![vec![0.5, 0.75, 1.0, 1.5, 2.5], vec![0.75, 1.25, 2.0], vec![0.75]];
                let rich = vec![vec![0.33, 0.66, 1.0, 1.5, 2.5], vec![0.5, 1.0]];
                let lean = m.postflop.clone();
                m.by_street = Some(match name {
                    "wide" => [rich.clone(), rich, lean],
                    "wide-river" => [rich.clone(), rich.clone(), rich],
                    _ => [rich, lean.clone(), lean],
                });
            }
            _ => return None,
        }
        Some(m)
    }

    pub fn by_name(name: &str) -> Option<Menu> {
        match name {
            "lean" => Some(Menu::lean()),
            other => Menu::candidate(other),
        }
    }

    /// Sizes for the `raises`-th bet/raise on `street` (0 = preflop).
    pub fn sizes(&self, street: usize, raises: u32) -> &[f64] {
        let lists = match (street, &self.by_street) {
            (0, _) => &self.preflop,
            (s, Some(b)) => &b[s - 1],
            (_, None) => &self.postflop,
        };
        &lists[(raises as usize).min(lists.len() - 1)]
    }
}

/// Bet sizes for live solving (`crate::live`), as fractions of the pot after
/// calling. All-in is always an option when raising is legal, even past
/// `max_raises`; the opponent's real sizes are added to the tree as they
/// happen, so these only shape the bot's own options and the future.
#[derive(Clone, Debug)]
pub struct LiveMenu {
    /// The first bet of a street.
    pub bets: Vec<f64>,
    /// Every raise after it.
    pub raises: Vec<f64>,
    /// Bets + raises per street before only all-in is left.
    pub max_raises: u32,
}

impl LiveMenu {
    /// A river solve: ¼ to 1½ pot, raises of a pot.
    pub fn river() -> LiveMenu {
        LiveMenu { bets: vec![0.25, 0.5, 0.75, 1.0, 1.5], raises: vec![1.0], max_raises: 3 }
    }

    /// The turn of a turn solve.
    pub fn turn() -> LiveMenu {
        LiveMenu { bets: vec![0.33, 0.66, 1.0, 1.5], raises: vec![1.0], max_raises: 3 }
    }

    /// The river inside a turn solve (48 copies of it, so lean).
    pub fn river_in_turn() -> LiveMenu {
        LiveMenu { bets: vec![0.5, 1.0], raises: vec![], max_raises: 2 }
    }

    /// The river inside turn solves by name: "lean" (`river_in_turn`),
    /// "raises" (the same bets, pot raises, 2 per street) or "full" (a river
    /// solve's, `river`).
    pub fn river_in_turn_by_name(name: &str) -> Option<LiveMenu> {
        match name {
            "lean" => Some(LiveMenu::river_in_turn()),
            "raises" => Some(LiveMenu { bets: vec![0.5, 1.0], raises: vec![1.0], max_raises: 2 }),
            "full" => Some(LiveMenu::river()),
            _ => None,
        }
    }

    /// The turn of a turn solve by name: "full" (`turn`), "medium" (bets of
    /// ⅓ and ¾ pot, pot raises, 2 per street) or "lean" (the same bets,
    /// raises only all-in).
    pub fn turn_by_name(name: &str) -> Option<LiveMenu> {
        match name {
            "full" => Some(LiveMenu::turn()),
            "medium" => Some(LiveMenu { bets: vec![0.33, 0.75], raises: vec![1.0], max_raises: 2 }),
            "lean" => Some(LiveMenu { bets: vec![0.33, 0.75], raises: vec![], max_raises: 2 }),
            _ => None,
        }
    }
}
