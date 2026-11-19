use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use hu_core::cards::{parse, Card};
use hu_core::hand_eval::eval7;
use hu_core::rules::{Action, ActionType, Phase, State, Status, Street};

use super::showdown::{hand, holes, mask, H};
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
        path_inserted: vec![],
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
        // Single solves wander: over 64 random river spots, 400 iterations
        // averaged 1.4% of the pot (median 1.1%) but some spots took
        // excursions to several percent, whatever the summation order.
        assert!(pct < 3.0, "exploitability {pct}% of the pot after 600 iterations");
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
    assert!(g.path_inserted[0]);
    assert_eq!(g.nodes[g.current() as usize].kind, Kind::Decision((1 - first) as u8));
    // An on-menu action is matched, not duplicated.
    let forced = [HistAction { seat: first, kind: ActionType::Bet, to: 4 }];
    let g2 = subgame::build(&s, &forced, &LiveMenu::river(), &LiveMenu::river_in_turn()).unwrap();
    assert_eq!(g2.nodes[0].n + 1, g.nodes[0].n);
    assert!(!g2.path_inserted[0]);
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

/// Turn tree sizes and iteration cost for candidate menus (run with
/// --ignored --nocapture).
#[test]
#[ignore]
fn turn_menu_costs() {
    let m = |bets: &[f64], raises: &[f64], max_raises| LiveMenu { bets: bets.to_vec(), raises: raises.to_vec(), max_raises };
    let menus = [
        ("current", LiveMenu::turn(), LiveMenu::river_in_turn()),
        ("half/pot, raise pot, 2", m(&[0.5, 1.0], &[1.0], 2), m(&[0.5, 1.0], &[], 2)),
        ("third/3q, jam raises", m(&[0.33, 0.75], &[], 2), m(&[0.5, 1.0], &[], 2)),
        ("third/3q, jam raises, river 3q", m(&[0.33, 0.75], &[], 2), m(&[0.75], &[], 1)),
    ];
    let mut rng = SmallRng::seed_from_u64(1);
    let ranges = random_ranges(&mut rng);
    for seed in [4, 7] {
        let s = at_street(Street::Turn, seed);
        for (name, turn, river) in &menus {
            let g = subgame::build(&s, &[], turn, river).unwrap();
            let river_nodes = g.nodes.iter().filter(|n| n.river && matches!(n.kind, Kind::Decision(_))).count();
            let mut sv = Solver::new(&g, &s.community, ranges.clone(), vec![], true);
            let t0 = std::time::Instant::now();
            sv.run(20);
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / 20.0;
            println!("{name:32} {} decision nodes ({} river): {ms:.1} ms/iteration", g.decisions(), river_nodes);
        }
    }
}


/// Safe re-solving holds every opponent hand to its alternative value even
/// when the estimated range leaves hands out; unsafe re-solving doesn't.
/// Blueprint: the subgame's equilibrium with the opponent's true (wide)
/// range; alternative values: each opponent hand's best response to it.
/// Estimate: the true range with every weak hand removed.
#[test]
fn safe_resolving_holds_every_hand_to_its_alternative() {
    use super::showdown::{pair_weight, Board};
    let s = at_street(Street::River, 21);
    let g = subgame::build(&s, &[], &LiveMenu::river(), &LiveMenu::river_in_turn()).unwrap();
    let board = s.community.clone();
    let mut rng = SmallRng::seed_from_u64(3);
    let ranges = random_ranges(&mut rng);
    let bot = g.nodes[0].kind == Kind::Decision(0);
    let (bot, opp) = if bot { (0usize, 1usize) } else { (1, 0) };
    // The blueprint and each opponent hand's best response to it.
    let mut bp = Solver::new(&g, &board, ranges.clone(), vec![], false);
    bp.run(800);
    let alt = bp.best_response_values(opp);
    // The estimate: drop the opponent's weaker half of hands (by strength
    // on this board), i.e. the bluffs and bluff-catchers it really has.
    let b = Board::new([board[0], board[1], board[2], board[3], board[4]]);
    let mut strength = vec![0f32; H];
    let mut uniform = vec![1f32; H];
    super::showdown::mask(&mut uniform, &board);
    b.showdown(&uniform, 1.0, &mut strength);
    let mut est = ranges.clone();
    let mut order: Vec<usize> = (0..H).filter(|&h| est[opp][h] > 0.0 && uniform[h] > 0.0).collect();
    order.sort_by(|&x, &y| strength[x].partial_cmp(&strength[y]).unwrap());
    for &h in &order[..order.len() / 2] {
        est[opp][h] = 0.0;
    }
    // How much the opponent's true range gains over its alternatives
    // against a strategy, per hand pair.
    let gain = |sv: &mut Solver| -> f64 {
        let br = sv.best_response_values(opp);
        let num: f64 = (0..H).map(|h| ranges[opp][h] as f64 * (br[h] - alt[h]).max(0.0) as f64).sum();
        num / pair_weight(&ranges[opp], &ranges[bot])
    };
    let mut unsafe_sv = Solver::new(&g, &board, est.clone(), vec![], false);
    unsafe_sv.run(800);
    let unsafe_gain = gain(&mut unsafe_sv);
    let mut safe_sv = Solver::new(&g, &board, est, vec![], false);
    safe_sv.set_gadget(opp, alt.clone(), None);
    safe_sv.run(800);
    let safe_gain = gain(&mut safe_sv);
    let pot = g.pot as f64;
    println!("gain over the alternatives: unsafe {:.2}% of the pot, safe {:.2}%", unsafe_gain / pot * 100.0, safe_gain / pot * 100.0);
    assert!(unsafe_gain / pot > 0.05, "the unsafe solve should be exploitable by the hands it left out ({unsafe_gain})");
    assert!(safe_gain / pot < 0.01, "the safe solve should hold every hand to its alternative ({safe_gain})");
    assert!(safe_gain < unsafe_gain / 10.0);
    // Every hand (weighted or not) gets to choose: the gadget enters with
    // hands the estimate dropped too.
    let enter = safe_sv.gadget_enter().unwrap();
    assert!(order[..order.len() / 2].iter().any(|&h| enter[h] > 0.0));
}

/// A turn spot with a real pot: a raised preflop pot and a bet and call on
/// the flop (stacks 400 chips = 200bb), with a shuffled deck.
fn raised_turn(seed: u64) -> State {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut deck: Vec<Card> = (0..52).collect();
    deck.shuffle(&mut rng);
    let mut s = State::new_table([400, 400]);
    s.start_hand(deck);
    s.apply(Action::to(ActionType::Raise, 6));
    s.apply(Action::new(ActionType::Call));
    s.advance_street();
    s.apply(Action::to(ActionType::Bet, 8));
    s.apply(Action::new(ActionType::Call));
    s.advance_street();
    assert_eq!((s.street, s.phase), (Street::Turn, Phase::Betting));
    s
}

/// What `p` holding `h` wins at `node` (state `s`), best-responding to an
/// opponent who plays uniformly at random with reach `ro` (per hand,
/// `holes()` order): every action through the rules port, every river card
/// dealt from the deck, every hand settled by the state's commitments and
/// `eval7`. `strength[c][h']` is hand `h'`'s strength with river card `c`.
fn brute_br(g: &Subgame, s: &State, node: u32, p: usize, h: [Card; 2], ro: &[f64], strength: &[Vec<u32>]) -> f64 {
    let nd = &g.nodes[node as usize];
    let kids = g.children(nd);
    let disjoint = |x: [Card; 2]| x[0] != h[0] && x[0] != h[1] && x[1] != h[0] && x[1] != h[1];
    match nd.kind {
        Kind::Decision(q) => {
            assert_eq!((s.phase, s.to_act), (Phase::Betting, q as i32), "the tree's seat to act");
            let vals: Vec<f64> = g
                .actions(nd)
                .iter()
                .zip(kids)
                .map(|(&a, &k)| {
                    let mut t = s.clone();
                    t.apply(a);
                    brute_br(g, &t, k, p, h, ro, strength)
                })
                .collect();
            if q as usize == p {
                vals.iter().copied().fold(f64::MIN, f64::max)
            } else {
                vals.iter().sum::<f64>() / vals.len() as f64
            }
        }
        Kind::Chance => {
            assert_eq!(s.phase, Phase::StreetComplete);
            let mut total = 0.0;
            for c in 0..52u8 {
                if s.community.contains(&c) || h.contains(&c) {
                    continue;
                }
                let ro2: Vec<f64> = holes().iter().zip(ro).map(|(&[a, b], &x)| if a == c || b == c { 0.0 } else { x }).collect();
                let mut t = s.clone();
                t.deck.retain(|&x| x != c);
                t.deck.push(c);
                t.advance_street();
                assert_eq!(t.community.last(), Some(&c));
                total += brute_br(g, &t, kids[0], p, h, &ro2, strength);
            }
            // Each pair of hands meets the 44 cards outside both and the board.
            total / 44.0
        }
        Kind::Fold(w) => {
            let folded = (0..2).find(|&i| s.players[i].status == Status::Folded).expect("someone folded");
            assert_eq!(1 - folded, w as usize);
            let v = if folded == p { -(s.players[p].commit as f64) } else { s.players[1 - p].commit as f64 };
            let reach: f64 = holes().iter().zip(ro).filter(|(&x, _)| disjoint(x)).map(|(_, &r)| r).sum();
            v * reach
        }
        Kind::Showdown => {
            assert_eq!(s.community.len(), 5);
            let c = s.community[4] as usize;
            let mut cards = h.to_vec();
            cards.extend_from_slice(&s.community);
            let mine = eval7(&cards);
            let m = s.players[0].commit.min(s.players[1].commit) as f64;
            let mut net = 0.0;
            for (k, (&x, &r)) in holes().iter().zip(ro).enumerate() {
                if r != 0.0 && disjoint(x) {
                    net += r * match mine.cmp(&strength[c][k]) {
                        std::cmp::Ordering::Greater => 1.0,
                        std::cmp::Ordering::Less => -1.0,
                        std::cmp::Ordering::Equal => 0.0,
                    };
                }
            }
            m * net
        }
    }
}

/// The turn solver's game is the real game: each hand's best response to
/// an untrained turn solve (whose average strategy is uniformly random
/// everywhere, river included), from the solver (CPU, and the GPU when the
/// `gpu` feature finds one) and from an independent walk of the same tree
/// that plays every action through the rules port, deals every river card
/// and settles every hand by its commitments and `eval7` (`brute_br`).
/// Covers the tree, fold and showdown payoffs, all-in run-outs, the river
/// cards and card removal between the hands, the river and the board.
#[test]
fn turn_best_responses_match_the_rules_engine() {
    let spots = [(at_street(Street::Turn, 3), "medium"), (raised_turn(5), "medium"), (raised_turn(9), "full")];
    for (i, (s, menu)) in spots.iter().enumerate() {
        let g = subgame::build(s, &[], &LiveMenu::turn_by_name(menu).unwrap(), &LiveMenu::river_in_turn()).unwrap();
        let mut rng = SmallRng::seed_from_u64(i as u64 + 40);
        let mut ranges = random_ranges(&mut rng);
        for r in ranges.iter_mut() {
            mask(r, &s.community);
        }
        let strength: Vec<Vec<u32>> = (0..52u8)
            .map(|c| {
                holes()
                    .iter()
                    .map(|&[a, b]| {
                        if s.community.contains(&c) || [a, b].contains(&c) || s.community.contains(&a) || s.community.contains(&b) {
                            return 0;
                        }
                        let mut x = vec![a, b];
                        x.extend_from_slice(&s.community);
                        x.push(c);
                        eval7(&x)
                    })
                    .collect()
            })
            .collect();
        let ours: Vec<usize> = (0..H).filter(|&h| ranges[0][h] > 0.0 && ranges[1][h] > 0.0).step_by(37).collect();
        assert!(ours.len() >= 8);
        let mut checked = [0; 2];
        for gpu in [false, true] {
            let mut sv = Solver::new(&g, &s.community, ranges.clone(), vec![], true);
            if gpu && !sv.use_gpu() {
                continue;
            }
            for p in 0..2 {
                let br = sv.best_response_values(p);
                let ro: Vec<f64> = ranges[1 - p].iter().map(|&x| x as f64).collect();
                let want: Vec<f64> = ours.iter().map(|&h| brute_br(&g, s, 0, p, holes()[h], &ro, &strength)).collect();
                let scale = want.iter().fold(1.0f64, |m, x| m.max(x.abs()));
                for (&h, &x) in ours.iter().zip(&want) {
                    let y = br[h] as f64;
                    assert!((x - y).abs() <= 1e-4 * scale, "spot {i} ({menu}) gpu {gpu} seat {p} hand {h}: solver {y}, rules engine {x}");
                    checked[gpu as usize] += 1;
                }
            }
        }
        assert_eq!(checked[0], 2 * ours.len());
        eprintln!("spot {i} ({menu}): {} values on the CPU, {} on the GPU, against the rules engine", checked[0], checked[1]);
    }
}
