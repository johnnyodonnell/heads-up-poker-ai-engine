//! The betting tree of a live solve: from the start of the current street
//! (turn or river) with the real pot and stacks, through the street's real
//! actions so far (inserted at their real sizes when they aren't on the
//! menu), to the end of the hand. A turn tree has one chance node per line
//! that reaches the river, shared by all 48 river cards (the betting doesn't
//! depend on the card; the solver keeps a separate copy of the river regrets
//! per card).
//!
//! Built by driving the rules port, like `crate::tree`, so every action is
//! legal under the app's rules and records exactly as the real hand does.

use hu_core::allin_ev::is_runout;
use hu_core::rules::{Action, ActionType, Phase, State, Status, Street};

use crate::engine::HistAction;
use crate::menu::LiveMenu;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The seat to act.
    Decision(u8),
    /// The river card.
    Chance,
    /// The seat that wins the pot.
    Fold(u8),
    Showdown,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub kind: Kind,
    /// After the chance node: one copy per river card.
    pub river: bool,
    /// Chips each seat has put in over the whole hand.
    pub contrib: [i32; 2],
    /// Children are `kids[first..first + n]`.
    pub first: u32,
    pub n: u32,
}

pub struct Subgame {
    pub nodes: Vec<Node>,
    pub kids: Vec<u32>,
    /// The real action of each edge (parallel to `kids`; unused for chance).
    pub acts: Vec<Action>,
    /// How each edge records in the hand history: (kind, to).
    pub recs: Vec<(ActionType, i32)>,
    pub street: Street,
    /// `path[i]` is the node where the street's real action `i` was taken;
    /// the last entry is the node to act now (if the street is still on).
    pub path: Vec<u32>,
    /// The edge (0..n) each real action took at `path[i]`.
    pub path_edges: Vec<u32>,
    /// Pot at the root (both seats' contributions).
    pub pot: i32,
}

impl Subgame {
    #[inline]
    pub fn children(&self, nd: &Node) -> &[u32] {
        &self.kids[nd.first as usize..(nd.first + nd.n) as usize]
    }

    #[inline]
    pub fn actions(&self, nd: &Node) -> &[Action] {
        &self.acts[nd.first as usize..(nd.first + nd.n) as usize]
    }

    #[inline]
    pub fn records(&self, nd: &Node) -> &[(ActionType, i32)] {
        &self.recs[nd.first as usize..(nd.first + nd.n) as usize]
    }

    /// The node to act now (after all the street's real actions).
    pub fn current(&self) -> u32 {
        *self.path.last().expect("path")
    }

    pub fn decisions(&self) -> usize {
        self.nodes.iter().filter(|n| matches!(n.kind, Kind::Decision(_))).count()
    }
}

struct Builder<'a> {
    g: Subgame,
    forced: &'a [HistAction],
    menu: &'a LiveMenu,
    river_menu: &'a LiveMenu,
    err: Option<String>,
}

/// The tree from `start` (a state at the start of a turn or river, betting
/// open) with the street's real actions `forced` taken so far.
pub fn build(start: &State, forced: &[HistAction], menu: &LiveMenu, river_menu: &LiveMenu) -> Result<Subgame, String> {
    if start.phase != Phase::Betting || !matches!(start.street, Street::Turn | Street::River) {
        return Err(format!("live solves start on the turn or river with betting open, not {:?} {:?}", start.street, start.phase));
    }
    let pot = start.pot + start.players.iter().map(|p| p.bet).sum::<i32>();
    let mut b = Builder {
        g: Subgame {
            nodes: Vec::new(),
            kids: Vec::new(),
            acts: Vec::new(),
            recs: Vec::new(),
            street: start.street,
            path: vec![u32::MAX; forced.len() + 1],
            path_edges: vec![u32::MAX; forced.len()],
            pot,
        },
        forced,
        menu,
        river_menu,
        err: None,
    };
    let root = b.decision(start.clone(), 0, Some(0), false);
    debug_assert_eq!(root, 0);
    if let Some(e) = b.err {
        return Err(e);
    }
    // The street may have ended with the last real action (then there is no
    // node to act on this street); otherwise every path entry is set.
    if b.g.path[..forced.len()].contains(&u32::MAX) {
        return Err("the real actions left the tree".into());
    }
    if b.g.path[forced.len()] == u32::MAX {
        b.g.path.pop();
    }
    Ok(b.g)
}

impl Builder<'_> {
    fn push(&mut self, kind: Kind, river: bool, s: &State) -> u32 {
        self.g.nodes.push(Node { kind, river, contrib: [s.players[0].commit, s.players[1].commit], first: 0, n: 0 });
        (self.g.nodes.len() - 1) as u32
    }

    fn set_children(&mut self, id: u32, items: Vec<(Action, (ActionType, i32), u32)>) {
        let nd = &mut self.g.nodes[id as usize];
        nd.first = self.g.kids.len() as u32;
        nd.n = items.len() as u32;
        for (a, r, k) in items {
            self.g.acts.push(a);
            self.g.recs.push(r);
            self.g.kids.push(k);
        }
    }

    fn decision(&mut self, s: State, raises: u32, depth: Option<usize>, river: bool) -> u32 {
        let legal = s.legal().expect("a decision");
        let seat = s.to_act as usize;
        let id = self.push(Kind::Decision(seat as u8), river, &s);
        let menu = if river { self.river_menu } else { self.menu };

        let mut opts: Vec<Action> = Vec::new();
        if legal.can_fold {
            opts.push(Action::new(ActionType::Fold));
        }
        opts.push(Action::new(if legal.can_check { ActionType::Check } else { ActionType::Call }));
        if legal.can_raise {
            if raises < menu.max_raises {
                let me = &s.players[seat];
                let pot_total = s.pot + s.players.iter().map(|p| p.bet).sum::<i32>();
                let pac = pot_total + (s.current_bet - me.bet);
                let sizes = if raises == 0 { &menu.bets } else { &menu.raises };
                let mut seen: Vec<i32> = Vec::new();
                for &f in sizes {
                    let to = ((s.current_bet as f64 + f * pac as f64).round() as i32).max(legal.min_raise_to);
                    if to >= legal.max_raise_to || seen.contains(&to) {
                        continue;
                    }
                    seen.push(to);
                    opts.push(Action::to(if legal.opening { ActionType::Bet } else { ActionType::Raise }, to));
                }
            }
            opts.push(Action::new(ActionType::AllIn));
        }
        // Each option applied: how it records, and the state after it.
        let mut items: Vec<(Action, (ActionType, i32), State)> = opts
            .into_iter()
            .map(|a| {
                let mut s2 = s.clone();
                s2.apply(a);
                let r = s2.actions.last().expect("recorded");
                (a, (r.kind, r.to), s2)
            })
            .collect();

        // The real action taken here, if this node is on the real path.
        let mut path_edge = None;
        if let Some(d) = depth {
            self.g.path[d] = id;
            if let Some(f) = self.forced.get(d) {
                if f.seat != seat {
                    self.err.get_or_insert(format!("real action {d}: seat {} acted, the rules say seat {seat}", f.seat));
                } else if let Some(k) = items.iter().position(|it| it.1 == (f.kind, f.to)) {
                    path_edge = Some(k);
                } else {
                    let a = match f.kind {
                        ActionType::Bet | ActionType::Raise => Action::to(f.kind, f.to),
                        k => Action::new(k),
                    };
                    let mut s2 = s.clone();
                    s2.apply(a);
                    let r = s2.actions.last().expect("recorded");
                    if (r.kind, r.to) != (f.kind, f.to) {
                        self.err.get_or_insert(format!("real action {d} ({:?} to {}) is not legal here", f.kind, f.to));
                    } else {
                        items.push((a, (f.kind, f.to), s2));
                        path_edge = Some(items.len() - 1);
                    }
                }
                if let Some(k) = path_edge {
                    self.g.path_edges[d] = k as u32;
                }
            }
        }

        let mut kids = Vec::with_capacity(items.len());
        for (k, (a, r, s2)) in items.into_iter().enumerate() {
            let aggressive = matches!(r.0, ActionType::Bet | ActionType::Raise);
            let next_depth = depth.filter(|_| path_edge == Some(k)).map(|d| d + 1);
            let child = self.after(s2, raises + u32::from(aggressive), next_depth, river);
            kids.push((a, r, child));
        }
        self.set_children(id, kids);
        id
    }

    /// The node reached after an action.
    fn after(&mut self, mut s: State, raises: u32, depth: Option<usize>, river: bool) -> u32 {
        match s.phase {
            Phase::Betting => self.decision(s, raises, depth, river),
            Phase::StreetComplete => {
                let live: Vec<usize> = (0..2).filter(|&i| !matches!(s.players[i].status, Status::Folded | Status::Out)).collect();
                if live.len() == 1 {
                    return self.push(Kind::Fold(live[0] as u8), river, &s);
                }
                if s.street == Street::River {
                    return self.push(Kind::Showdown, river, &s);
                }
                // The turn is over: deal the river.
                let id = self.push(Kind::Chance, false, &s);
                let child = if is_runout(&s) {
                    self.push(Kind::Showdown, true, &s)
                } else {
                    s.advance_street();
                    self.decision(s, 0, None, true)
                };
                self.set_children(id, vec![(Action::new(ActionType::Check), (ActionType::Check, 0), child)]);
                id
            }
            other => unreachable!("unexpected phase {other:?}"),
        }
    }
}
