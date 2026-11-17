use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use hu_core::cards::{parse, Card};
use hu_core::rules::{Action, ActionType, Phase, State, Street};

use super::showdown::{hand, H};
use super::solver::Solver;
use super::subgame::{self, Kind, Node, Subgame};
use crate::engine::HistAction;
use crate::menu::LiveMenu;

fn card(s: &str) -> Card {
    parse(s).unwrap()
}

/// A state at the start of `street` after both players only checked and
/// called (stacks 400 chips = 200bb), with a shuffled deck.
fn at_street(street: Street, seed: u64) -> State {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut deck: Vec<Card> = (0..52).collect();
    deck.shuffle(&mut rng);
    let mut s = State::new_table([400, 400]);
    s.start_hand(deck);
    loop {
        match s.phase {
            Phase::Betting => {
                let l = s.legal().unwrap();
                s.apply(Action::new(if l.can_check { ActionType::Check } else { ActionType::Call }));
            }
            Phase::StreetComplete => {
                s.advance_street();
                if s.street == street {
                    return s;
                }
            }
            p => panic!("{p:?}"),
        }
    }
}

fn random_ranges(rng: &mut SmallRng) -> [Vec<f32>; 2] {
    let mut r = || (0..H).map(|_| if rng.gen_bool(0.4) { 0.0 } else { rng.gen::<f32>() }).collect::<Vec<f32>>();
    [r(), r()]
}

/// Bettor (seat 0) with sets of kings and 6-5 air against a bluff-catcher
/// (aces), one pot-size bet: the bettor bets every set and bluffs half as
/// often as it value-bets; the catcher calls half the time.
#[test]
fn polarized_river_reaches_the_known_equilibrium() {
    let node = |kind, contrib, first, n| Node { kind, river: false, contrib, first, n };
    let g = Subgame {
        nodes: vec![
            node(Kind::Decision(0), [10, 10], 0, 2),
            node(Kind::Showdown, [10, 10], 0, 0),
            node(Kind::Decision(1), [30, 10], 2, 2),
            node(Kind::Fold(0), [30, 10], 0, 0),
            node(Kind::Showdown, [30, 30], 0, 0),
        ],
        kids: vec![1, 2, 3, 4],
        acts: vec![Action::new(ActionType::Check); 4],
        recs: vec![(ActionType::Check, 0); 4],
        street: Street::River,
        path: vec![0],
        path_edges: vec![],
        pot: 20,
    };
    let board = ["Kh", "Qd", "7c", "4s", "2h"].map(card);
    let mut r0 = vec![0f32; H];
    let mut r1 = vec![0f32; H];
    let kings: Vec<usize> = [("Kd", "Kc"), ("Kd", "Ks"), ("Kc", "Ks")].iter().map(|(a, b)| hand([card(a), card(b)])).collect();
    let mut air = Vec::new();
    for s6 in ["c", "d", "h", "s"] {
        for s5 in ["c", "d", "h", "s"] {
            air.push(hand([card(&format!("6{s6}")), card(&format!("5{s5}"))]));
        }
    }
    for &h in kings.iter().chain(&air) {
        r0[h] = 1.0;
    }
    let aces = ["Ac", "Ad", "Ah", "As"];
    for i in 0..4 {
        for j in i + 1..4 {
            r1[hand([card(aces[i]), card(aces[j])])] = 1.0;
        }
    }
    let mut sv = Solver::new(&g, &board, [r0, r1.clone()], vec![], false);
    sv.run(4000);
    let bet = sv.strategy(0);
    let call = sv.strategy(2);
    for &h in &kings {
        assert!(bet[H + h] > 0.98, "sets bet {}", bet[H + h]);
    }
    let bluffs: f32 = air.iter().map(|&h| bet[H + h]).sum();
    assert!((bluffs - 1.5).abs() < 0.15, "bluff combos {bluffs} (want 1.5)");
    let calls: Vec<f32> = (0..H).filter(|&h| r1[h] > 0.0).map(|h| call[H + h]).collect();
    let mean = calls.iter().sum::<f32>() / calls.len() as f32;
    assert!((mean - 0.5).abs() < 0.05, "call frequency {mean} (want 0.5)");
    let x = sv.exploitability();
    assert!(x < 0.1, "exploitability {x} chips");
}

#[test]
fn river_solves_converge_and_are_deterministic() {
    let mut rng = SmallRng::seed_from_u64(9);
    for seed in 0..3 {
        let s = at_street(Street::River, seed);
        let g = subgame::build(&s, &[], &LiveMenu::river(), &LiveMenu::river_in_turn()).unwrap();
        assert!(g.decisions() > 10);
        let ranges = random_ranges(&mut rng);
        let mut a = Solver::new(&g, &s.community, ranges.clone(), vec![], false);
        a.run(50);
        let early = a.exploitability();
        a.run(550);
        let late = a.exploitability();
        let pct = late / g.pot as f64 * 100.0;
        assert!(late < early, "{early} -> {late}");
        assert!(pct < 1.0, "exploitability {pct}% of the pot after 600 iterations");
        let mut b = Solver::new(&g, &s.community, ranges, vec![], false);
        b.run(600);
        assert_eq!(a.strategy(0), b.strategy(0), "same inputs, same strategy");
    }
}

#[test]
fn real_off_menu_actions_are_inserted() {
    let s = at_street(Street::River, 3);
    let first = s.to_act as usize;
    // The pot is 4 chips: the menu bets 2, 3, 4 or 6; the real bet is 5.
    let forced = [HistAction { seat: first, kind: ActionType::Bet, to: 5 }];
    let g = subgame::build(&s, &forced, &LiveMenu::river(), &LiveMenu::river_in_turn()).unwrap();
    assert_eq!(g.path.len(), 2);
    let root = &g.nodes[g.path[0] as usize];
    assert_eq!(g.records(root)[g.path_edges[0] as usize], (ActionType::Bet, 5));
    assert_eq!(g.nodes[g.current() as usize].kind, Kind::Decision((1 - first) as u8));
    // An on-menu action is matched, not duplicated.
    let forced = [HistAction { seat: first, kind: ActionType::Bet, to: 4 }];
    let g2 = subgame::build(&s, &forced, &LiveMenu::river(), &LiveMenu::river_in_turn()).unwrap();
    assert_eq!(g2.nodes[0].n + 1, g.nodes[0].n);
}

#[test]
fn turn_solves_converge() {
    let s = at_street(Street::Turn, 4);
    let g = subgame::build(&s, &[], &LiveMenu::turn(), &LiveMenu::river_in_turn()).unwrap();
    let mut rng = SmallRng::seed_from_u64(1);
    let ranges = random_ranges(&mut rng);
    let mut sv = Solver::new(&g, &s.community, ranges.clone(), vec![], true);
    sv.run(20);
    let early = sv.exploitability();
    sv.run(130);
    let late = sv.exploitability();
    let pct = late / g.pot as f64 * 100.0;
    assert!(late < early * 0.5, "{early} -> {late}");
    // A 4-chip pot with 200bb behind and random ranges: slow in pot terms
    // (about 1/iterations; see convergence_curves).
    assert!(pct < 20.0, "exploitability {pct}% of the pot after 150 iterations");
    // Threading doesn't change the result.
    let mut one = Solver::new(&g, &s.community, ranges, vec![], false);
    one.run(150);
    assert_eq!(sv.strategy(0), one.strategy(0));
}

/// Convergence curves (run with --ignored --nocapture).
#[test]
#[ignore]
fn convergence_curves() {
    let mut rng = SmallRng::seed_from_u64(1);
    for (street, seed) in [(Street::River, 4), (Street::Turn, 4)] {
        let s = at_street(street, seed);
        let g = subgame::build(&s, &[], &LiveMenu::turn(), &LiveMenu::river_in_turn()).unwrap();
        let ranges = random_ranges(&mut rng);
        let mut sv = Solver::new(&g, &s.community, ranges, vec![], true);
        let t0 = std::time::Instant::now();
        let mut done = 0;
        for target in [25, 50, 100, 200, 400, 800] {
            sv.run(target - done);
            done = target;
            let x = sv.exploitability();
            println!("{street:?} pot {} nodes {}: {target} iters  {:.3}% pot  ({:.1}s)", g.pot, g.decisions(), x / g.pot as f64 * 100.0, t0.elapsed().as_secs_f64());
        }
    }
}
