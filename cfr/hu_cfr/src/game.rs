//! A two-player zero-sum betting tree whose chance events are dealt up front.
//!
//! Every game the solver handles (heads-up NLHE abstractions, push/fold, and
//! the Kuhn/Leduc test games) is a fixed tree of betting nodes plus a *deal*:
//! each player's bucket on each street and who wins a showdown. A player's
//! information set at a decision node is (node, their bucket on the node's
//! street), so regrets live in one flat array indexed by
//! `node.base + bucket * n_actions + action`.
//!
//! Utilities are chips. A terminal stores each player's total contribution;
//! a fold loses the folder's contribution to the other player, and a showdown
//! moves min(contributions) to the winner (equal on a tie: zero).

use serde::{Deserialize, Serialize};

pub const MAX_STREETS: usize = 4;
/// The most actions any node may have (fold, call, four sizes, all-in = 7).
pub const MAX_ACTIONS: usize = 8;
/// `Node::player` at a terminal.
pub const TERMINAL: u8 = u8::MAX;

/// An abstract action, as labelled for logs, translation and the debug view.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Act {
    Fold,
    Check,
    Call,
    /// Opening bet to `to` chips; `frac` = bet / pot.
    Bet { to: i32, frac: f32 },
    /// Raise to `to` chips; `frac` = (to - current bet) / pot after calling.
    Raise { to: i32, frac: f32 },
    /// All-in for `to` chips (a bet or raise); `frac` as for Raise.
    AllIn { to: i32, frac: f32 },
}

impl Act {
    pub fn label(&self) -> String {
        match *self {
            Act::Fold => "fold".into(),
            Act::Check => "check".into(),
            Act::Call => "call".into(),
            Act::Bet { to, frac } => format!("bet {to} ({:.0}% pot)", frac * 100.0),
            Act::Raise { to, frac } => format!("raise to {to} ({:.0}% pot)", frac * 100.0),
            Act::AllIn { to, .. } => format!("all-in {to}"),
        }
    }

    /// Pot fraction of a bet/raise/all-in (None for fold/check/call).
    pub fn frac(&self) -> Option<f32> {
        match *self {
            Act::Bet { frac, .. } | Act::Raise { frac, .. } | Act::AllIn { frac, .. } => Some(frac),
            _ => None,
        }
    }

    pub fn is_aggressive(&self) -> bool {
        self.frac().is_some()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Terminal {
    Fold { winner: u8 },
    Showdown,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    /// Player to act (0 or 1), or TERMINAL.
    pub player: u8,
    pub street: u8,
    /// Children (and their actions) are `first..first + n` in `TreeGame::child`
    /// and `TreeGame::acts`.
    pub first: u32,
    pub n: u8,
    /// Offset of (bucket 0, action 0) in the regret/strategy arrays.
    pub base: u64,
    /// Each player's total chips committed when this node is reached.
    pub contrib: [i32; 2],
    pub terminal: Option<Terminal>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TreeGame {
    pub nodes: Vec<Node>,
    pub child: Vec<u32>,
    pub acts: Vec<Act>,
    /// Buckets per street (information-set granularity).
    pub buckets: [u32; MAX_STREETS],
    /// Total (infoset, action) entries.
    pub n_entries: u64,
}

impl TreeGame {
    pub fn new(buckets: [u32; MAX_STREETS]) -> TreeGame {
        TreeGame { nodes: Vec::new(), child: Vec::new(), acts: Vec::new(), buckets, n_entries: 0 }
    }

    /// Add a terminal node; returns its id.
    pub fn terminal(&mut self, street: u8, contrib: [i32; 2], t: Terminal) -> u32 {
        self.nodes.push(Node { player: TERMINAL, street, first: 0, n: 0, base: 0, contrib, terminal: Some(t) });
        (self.nodes.len() - 1) as u32
    }

    /// Reserve a decision node (children filled in later with `set_children`).
    pub fn decision(&mut self, player: u8, street: u8, contrib: [i32; 2]) -> u32 {
        self.nodes.push(Node { player, street, first: 0, n: 0, base: 0, contrib, terminal: None });
        (self.nodes.len() - 1) as u32
    }

    pub fn set_children(&mut self, node: u32, items: &[(Act, u32)]) {
        assert!(!items.is_empty() && items.len() <= MAX_ACTIONS, "bad action count {}", items.len());
        let first = self.child.len() as u32;
        for &(a, c) in items {
            self.acts.push(a);
            self.child.push(c);
        }
        let nd = &mut self.nodes[node as usize];
        nd.first = first;
        nd.n = items.len() as u8;
    }

    /// Assign every decision node its block of regret entries.
    pub fn finalize(&mut self) {
        let mut off = 0u64;
        for nd in self.nodes.iter_mut() {
            if nd.player != TERMINAL {
                nd.base = off;
                off += self.buckets[nd.street as usize] as u64 * nd.n as u64;
            }
        }
        self.n_entries = off;
    }

    #[inline]
    pub fn children(&self, nd: &Node) -> &[u32] {
        &self.child[nd.first as usize..nd.first as usize + nd.n as usize]
    }

    #[inline]
    pub fn actions(&self, nd: &Node) -> &[Act] {
        &self.acts[nd.first as usize..nd.first as usize + nd.n as usize]
    }

    /// Chips won by `p` at terminal `nd`; `showdown` is +1 if player 0 wins
    /// the showdown, -1 if player 1 does, 0 on a tie.
    #[inline]
    pub fn utility(&self, nd: &Node, p: usize, showdown: i8) -> f64 {
        match nd.terminal {
            Some(Terminal::Fold { winner }) => {
                let loser = 1 - winner as usize;
                let v = nd.contrib[loser] as f64;
                if p == winner as usize { v } else { -v }
            }
            Some(Terminal::Showdown) => {
                let s = nd.contrib[0].min(nd.contrib[1]) as f64;
                let u0 = s * showdown as f64;
                if p == 0 { u0 } else { -u0 }
            }
            None => unreachable!("utility of a decision node"),
        }
    }

    /// Decision-node counts per street.
    pub fn street_counts(&self) -> [usize; MAX_STREETS] {
        let mut c = [0; MAX_STREETS];
        for nd in &self.nodes {
            if nd.player != TERMINAL {
                c[nd.street as usize] += 1;
            }
        }
        c
    }

    /// (infoset, action) entries per street.
    pub fn street_entries(&self) -> [u64; MAX_STREETS] {
        let mut c = [0u64; MAX_STREETS];
        for nd in &self.nodes {
            if nd.player != TERMINAL {
                c[nd.street as usize] += self.buckets[nd.street as usize] as u64 * nd.n as u64;
            }
        }
        c
    }
}

/// One deal: each player's bucket on each street, and the showdown winner.
#[derive(Clone, Copy, Debug, Default)]
pub struct Deal {
    pub bucket: [[u32; MAX_STREETS]; 2],
    /// +1 if player 0 wins a showdown, -1 if player 1 does, 0 on a tie.
    pub showdown: i8,
}

/// Samples deals for the solver.
pub trait Dealer: Sync {
    fn deal(&self, rng: &mut rand::rngs::SmallRng) -> Deal;
}
