//! The abstract heads-up NLHE betting tree for one stack depth.
//!
//! Built by driving the real rules (`hu_core::rules`, the port of the app's
//! game.ts) from a fresh hand with both stacks at `stack` chips: at every
//! decision the legal actions come from the rules engine, and the menu's
//! sizes are rounded to whole chips, raised to the legal minimum if needed,
//! deduplicated, and dropped if they would be all-in (all-in is its own
//! action). So the tree contains exactly the app's betting rules (blinds 1/2,
//! min-raise by increment, no raising an all-in player).
//!
//! Player 0 is the small blind / button, player 1 the big blind. The effective
//! stack is all that matters heads-up, so both stacks are equal.

use hu_core::allin_ev::is_runout;
use hu_core::rules::{Action, ActionType, Phase, State, Status, Street, BIG_BLIND};

use crate::game::{Act, Terminal, TreeGame, MAX_STREETS};
use crate::menu::Menu;

pub fn build(stack: i32, menu: &Menu, buckets: [u32; MAX_STREETS]) -> TreeGame {
    assert!(stack > BIG_BLIND, "stack {stack} too short for a betting tree");
    let mut g = TreeGame::new(buckets);
    let mut s = State::new_table([stack, stack]);
    s.start_hand((0..52).collect());
    assert_eq!(s.phase, Phase::Betting);
    assert_eq!(s.to_act, 0, "the small blind acts first preflop");
    let root = decision(&mut g, s, 0, menu);
    assert_eq!(root, 0);
    g.finalize();
    g
}

fn decision(g: &mut TreeGame, s: State, raises: u32, menu: &Menu) -> u32 {
    let legal = s.legal().expect("a decision");
    let i = s.to_act as usize;
    let contrib = [s.players[0].commit, s.players[1].commit];
    let id = g.decision(i as u8, s.street.index() as u8, contrib);

    let mut opts: Vec<(Act, Action)> = Vec::new();
    if legal.can_fold {
        opts.push((Act::Fold, Action::new(ActionType::Fold)));
    }
    if legal.can_check {
        opts.push((Act::Check, Action::new(ActionType::Check)));
    } else {
        opts.push((Act::Call, Action::new(ActionType::Call)));
    }
    if legal.can_raise && raises < menu.max_raises {
        let me = &s.players[i];
        let pot_total = s.pot + s.players.iter().map(|p| p.bet).sum::<i32>();
        let pac = pot_total + (s.current_bet - me.bet);
        let mut seen: Vec<i32> = Vec::new();
        for &f in menu.sizes(s.street.index(), raises) {
            let to = ((s.current_bet as f64 + f * pac as f64).round() as i32).max(legal.min_raise_to);
            if to >= legal.max_raise_to || seen.contains(&to) {
                continue;
            }
            seen.push(to);
            let frac = (to - s.current_bet) as f32 / pac as f32;
            let (act, kind) = if legal.opening {
                (Act::Bet { to, frac }, ActionType::Bet)
            } else {
                (Act::Raise { to, frac }, ActionType::Raise)
            };
            opts.push((act, Action::to(kind, to)));
        }
        let to = legal.max_raise_to;
        let frac = (to - s.current_bet) as f32 / pac as f32;
        opts.push((Act::AllIn { to, frac }, Action::new(ActionType::AllIn)));
    } else if legal.can_raise && menu.allin_past_cap {
        let me = &s.players[i];
        let pot_total = s.pot + s.players.iter().map(|p| p.bet).sum::<i32>();
        let pac = pot_total + (s.current_bet - me.bet);
        let to = legal.max_raise_to;
        let frac = (to - s.current_bet) as f32 / pac as f32;
        opts.push((Act::AllIn { to, frac }, Action::new(ActionType::AllIn)));
    }

    let mut kids = Vec::with_capacity(opts.len());
    for (act, action) in opts {
        let mut s2 = s.clone();
        s2.apply(action);
        let r2 = if act.is_aggressive() { raises + 1 } else { raises };
        kids.push((act, after(g, s2, r2, menu)));
    }
    g.set_children(id, &kids);
    id
}

/// The node reached after an action: the next decision, possibly on a new
/// street, or a terminal.
fn after(g: &mut TreeGame, mut s: State, mut raises: u32, menu: &Menu) -> u32 {
    loop {
        match s.phase {
            Phase::Betting => return decision(g, s, raises, menu),
            Phase::StreetComplete => {
                let contrib = [s.players[0].commit, s.players[1].commit];
                let street = s.street.index() as u8;
                let live: Vec<usize> =
                    (0..2).filter(|&i| !matches!(s.players[i].status, Status::Folded | Status::Out)).collect();
                if live.len() == 1 {
                    return g.terminal(street, contrib, Terminal::Fold { winner: live[0] as u8 });
                }
                if s.street == Street::River || is_runout(&s) {
                    return g.terminal(street, contrib, Terminal::Showdown);
                }
                s.advance_street();
                raises = 0;
            }
            other => unreachable!("unexpected phase {other:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::TERMINAL;

    #[test]
    fn lean_tree_is_consistent() {
        for stack in [12, 24, 50, 100, 200] {
            let g = build(stack, &Menu::lean(), [169, 10, 10, 10]);
            let counts = g.street_counts();
            assert!(counts[0] > 0);
            for (id, nd) in g.nodes.iter().enumerate() {
                if nd.player == TERMINAL {
                    // Zero-sum and never more than the stack at stake.
                    for showdown in [-1i8, 0, 1] {
                        let u0 = g.utility(nd, 0, showdown);
                        let u1 = g.utility(nd, 1, showdown);
                        assert_eq!(u0 + u1, 0.0);
                        assert!(u0.abs() <= stack as f64, "node {id} pays {u0}");
                    }
                } else {
                    for (&c, a) in g.children(nd).iter().zip(g.actions(nd)) {
                        assert!(c as usize > id, "children come after their parent");
                        if let Some(f) = a.frac() {
                            assert!(f > 0.0, "{a:?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn opens_are_two_to_four_big_blinds() {
        let g = build(200, &Menu::lean(), [169, 1, 1, 1]);
        let root = &g.nodes[0];
        let labels: Vec<String> = g.actions(root).iter().map(|a| a.label()).collect();
        assert_eq!(
            labels,
            ["fold", "call", "raise to 4 (50% pot)", "raise to 5 (75% pot)", "raise to 6 (100% pot)", "raise to 8 (150% pot)", "all-in 200"]
        );
    }
}
