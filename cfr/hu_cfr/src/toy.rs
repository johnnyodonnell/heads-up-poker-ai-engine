//! Kuhn and Leduc poker as `TreeGame`s with enumerated deals: small enough for
//! exact exploitability, so they verify the solver end to end.

use rand::rngs::SmallRng;
use rand::Rng;

use crate::game::{Act, Deal, Dealer, Terminal, TreeGame};

/// Samples uniformly from an enumerated list of equally likely deals.
pub struct EnumDealer(pub Vec<Deal>);

impl Dealer for EnumDealer {
    fn deal(&self, rng: &mut SmallRng) -> Deal {
        self.0[rng.gen_range(0..self.0.len())]
    }
}

fn bet(to: i32) -> Act {
    Act::Bet { to, frac: 0.0 }
}

fn raise(to: i32) -> Act {
    Act::Raise { to, frac: 0.0 }
}

/// Kuhn poker: cards J < Q < K, ante 1, one bet of 1. Player 0's equilibrium
/// value is -1/18.
pub fn kuhn() -> (TreeGame, Vec<(Deal, f64)>) {
    let mut g = TreeGame::new([3, 0, 0, 0]);
    let root = g.decision(0, 0, [1, 1]);
    // Player 0 checks.
    let n_check = g.decision(1, 0, [1, 1]);
    let sd_checks = g.terminal(0, [1, 1], Terminal::Showdown);
    let n_check_bet = g.decision(0, 0, [1, 2]);
    let f = g.terminal(0, [1, 2], Terminal::Fold { winner: 1 });
    let c = g.terminal(0, [2, 2], Terminal::Showdown);
    g.set_children(n_check_bet, &[(Act::Fold, f), (Act::Call, c)]);
    g.set_children(n_check, &[(Act::Check, sd_checks), (bet(2), n_check_bet)]);
    // Player 0 bets.
    let n_bet = g.decision(1, 0, [2, 1]);
    let f = g.terminal(0, [2, 1], Terminal::Fold { winner: 0 });
    let c = g.terminal(0, [2, 2], Terminal::Showdown);
    g.set_children(n_bet, &[(Act::Fold, f), (Act::Call, c)]);
    g.set_children(root, &[(Act::Check, n_check), (bet(2), n_bet)]);
    g.finalize();
    assert_eq!(root, 0);

    let mut deals = Vec::new();
    for a in 0..3u32 {
        for b in 0..3u32 {
            if a != b {
                let mut d = Deal::default();
                d.bucket[0][0] = a;
                d.bucket[1][0] = b;
                d.showdown = if a > b { 1 } else { -1 };
                deals.push((d, 1.0 / 6.0));
            }
        }
    }
    (g, deals)
}

/// Leduc hold'em: a six-card deck (two each of J, Q, K), ante 1, one private
/// card each, one public card after the first round; bets of 2 then 4, at
/// most two bets/raises per round; player 0 acts first in both rounds. A pair
/// with the public card wins, otherwise the higher card.
pub fn leduc() -> (TreeGame, Vec<(Deal, f64)>) {
    fn round(g: &mut TreeGame, street: u8, contrib: [i32; 2], actor: u8, raises: u8, facing: bool, first: bool) -> u32 {
        let size = if street == 0 { 2 } else { 4 };
        let id = g.decision(actor, street, contrib);
        let other = 1 - actor;
        let mut kids = Vec::new();
        if !facing {
            let c = if first {
                round(g, street, contrib, other, raises, false, false)
            } else {
                round_end(g, street, contrib)
            };
            kids.push((Act::Check, c));
            let mut c2 = contrib;
            c2[actor as usize] += size;
            let b = round(g, street, c2, other, raises + 1, true, false);
            kids.push((bet(c2[actor as usize]), b));
        } else {
            let f = g.terminal(street, contrib, Terminal::Fold { winner: other });
            kids.push((Act::Fold, f));
            let mut c2 = contrib;
            c2[actor as usize] = contrib[other as usize];
            let c = round_end(g, street, c2);
            kids.push((Act::Call, c));
            if raises < 2 {
                let mut c3 = contrib;
                c3[actor as usize] = contrib[other as usize] + size;
                let r = round(g, street, c3, other, raises + 1, true, false);
                kids.push((raise(c3[actor as usize]), r));
            }
        }
        g.set_children(id, &kids);
        id
    }
    fn round_end(g: &mut TreeGame, street: u8, contrib: [i32; 2]) -> u32 {
        if street == 0 {
            round(g, 1, contrib, 0, 0, false, true)
        } else {
            g.terminal(1, contrib, Terminal::Showdown)
        }
    }

    let mut g = TreeGame::new([3, 9, 0, 0]);
    let root = round(&mut g, 0, [1, 1], 0, 0, false, true);
    g.finalize();
    assert_eq!(root, 0);

    let rank = |c: usize| (c / 2) as u32;
    let mut deals = Vec::new();
    for i in 0..6 {
        for j in 0..6 {
            for k in 0..6 {
                if i == j || i == k || j == k {
                    continue;
                }
                let (a, b, p) = (rank(i), rank(j), rank(k));
                let mut d = Deal::default();
                d.bucket[0][0] = a;
                d.bucket[0][1] = a * 3 + p;
                d.bucket[1][0] = b;
                d.bucket[1][1] = b * 3 + p;
                d.showdown = if a == p {
                    1
                } else if b == p {
                    -1
                } else {
                    (a as i32 - b as i32).signum() as i8
                };
                deals.push((d, 1.0 / 120.0));
            }
        }
    }
    (g, deals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::br::{exploitability, expected_value};
    use crate::mccfr::{Config, Solver};

    fn solve_with(g: &TreeGame, deals: &[(Deal, f64)], iters: u64, threads: usize) -> (Solver, f64, f64) {
        let dealer = EnumDealer(deals.iter().map(|(d, _)| *d).collect());
        let cfg = Config { threads, discount_every: iters / 100, discount_until: iters / 2, ..Config::default() };
        let mut s = Solver::new(g);
        s.train(g, &dealer, iters, iters / 20, &cfg, 7, |_| true);
        let sigma = |node: u32, bucket: u32, out: &mut [f64]| {
            let mut f = [0f64; 8];
            s.average(g, node, bucket, &mut f);
            out.copy_from_slice(&f[..out.len()]);
        };
        let expl = exploitability(g, deals, &sigma);
        let ev = expected_value(g, deals, &sigma);
        (s, expl, ev)
    }

    fn solve(g: &TreeGame, deals: &[(Deal, f64)], iters: u64) -> (Solver, f64, f64) {
        solve_with(g, deals, iters, 1)
    }

    /// Eight threads all hammering Kuhn's 12 information sets: the worst case
    /// for contention. Every update must still land.
    #[test]
    fn mccfr_converges_multithreaded() {
        let (g, deals) = kuhn();
        let (s, expl, ev) = solve_with(&g, &deals, 16_000_000, 8);
        println!("kuhn, 8 threads: exploitability {expl:.5}, value {ev:.5}");
        assert_eq!(s.iterations, 16_000_000);
        assert!(expl < 0.002, "kuhn (8 threads) exploitability {expl}");
    }

    #[test]
    fn uniform_kuhn_is_exploitable() {
        let (g, deals) = kuhn();
        let uniform = |_: u32, _: u32, out: &mut [f64]| {
            let n = out.len() as f64;
            out.fill(1.0 / n);
        };
        assert!(exploitability(&g, &deals, &uniform) > 0.2);
    }

    #[test]
    fn mccfr_solves_kuhn() {
        let (g, deals) = kuhn();
        // External sampling converges like 1/sqrt(T): check it keeps improving.
        let (_, e1, _) = solve(&g, &deals, 1_000_000);
        let (_, expl, ev) = solve(&g, &deals, 16_000_000);
        println!("kuhn: exploitability {e1:.5} at 1M -> {expl:.5} at 16M, value {ev:.5} (equilibrium -1/18 = {:.5})", -1.0 / 18.0);
        assert!(expl < 0.5 * e1, "kuhn exploitability {e1} -> {expl}");
        assert!(expl < 0.002, "kuhn exploitability {expl}");
        assert!((ev + 1.0 / 18.0).abs() < 0.01, "kuhn value {ev}");
    }

    #[test]
    fn mccfr_solves_leduc() {
        let (g, deals) = leduc();
        let (_, expl, ev) = solve(&g, &deals, 4_000_000);
        println!("leduc: {} nodes, exploitability {expl:.5}, value {ev:.5} (equilibrium ≈ -0.0856)", g.nodes.len());
        assert!(expl < 0.02, "leduc exploitability {expl}");
        assert!((ev + 0.0856).abs() < 0.02, "leduc value {ev}");
    }

    #[test]
    fn checkpoint_round_trip() {
        let (g, deals) = kuhn();
        let (s, _, _) = solve(&g, &deals, 50_000);
        let dir = std::env::temp_dir().join(format!("hucfr-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("kuhn.ckpt");
        s.save(&path).unwrap();
        let t = Solver::load(&g, &path).unwrap();
        assert_eq!(s.iterations, t.iterations);
        let (mut a, mut b) = ([0f64; 8], [0f64; 8]);
        for node in 0..g.nodes.len() as u32 {
            if g.nodes[node as usize].player != crate::game::TERMINAL {
                for bucket in 0..3 {
                    s.average(&g, node, bucket, &mut a);
                    t.average(&g, node, bucket, &mut b);
                    assert_eq!(a, b);
                }
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
