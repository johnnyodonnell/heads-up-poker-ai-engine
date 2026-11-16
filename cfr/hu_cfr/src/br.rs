//! Exact best response and exploitability for small games whose deals can be
//! enumerated (Kuhn, Leduc). Used to verify the solver.
//!
//! The best-responding player's information sets are (node, bucket), so this
//! is exact when buckets are lossless and the tree remembers all actions
//! (perfect recall), as in the test games.

use crate::game::{Deal, TreeGame, MAX_ACTIONS, TERMINAL};

/// A strategy: fills `out[..n]` with the action probabilities at (node, bucket).
pub type Strategy<'a> = dyn Fn(u32, u32, &mut [f64]) + 'a;

/// Per-deal values for `p` of best-responding below `node`, where `reach[i]`
/// is chance × opponent reach of deal i.
fn br(game: &TreeGame, node: u32, p: usize, reach: &[f64], deals: &[(Deal, f64)], sigma: &Strategy) -> Vec<f64> {
    let nd = &game.nodes[node as usize];
    if nd.player == TERMINAL {
        return deals.iter().map(|(d, _)| game.utility(nd, p, d.showdown)).collect();
    }
    let q = nd.player as usize;
    let n = nd.n as usize;
    let street = nd.street as usize;
    let kids = game.children(nd);
    let nd_deals = deals.len();
    if q != p {
        let mut sig = vec![[0f64; MAX_ACTIONS]; nd_deals];
        for (i, (d, _)) in deals.iter().enumerate() {
            sigma(node, d.bucket[q][street], &mut sig[i][..n]);
        }
        let mut val = vec![0.0; nd_deals];
        for a in 0..n {
            let r2: Vec<f64> = (0..nd_deals).map(|i| reach[i] * sig[i][a]).collect();
            let v = br(game, kids[a], p, &r2, deals, sigma);
            for i in 0..nd_deals {
                val[i] += sig[i][a] * v[i];
            }
        }
        val
    } else {
        let vs: Vec<Vec<f64>> = (0..n).map(|a| br(game, kids[a], p, reach, deals, sigma)).collect();
        let nb = game.buckets[street] as usize;
        let mut score = vec![[0f64; MAX_ACTIONS]; nb];
        for (i, (d, _)) in deals.iter().enumerate() {
            let b = d.bucket[p][street] as usize;
            for a in 0..n {
                score[b][a] += reach[i] * vs[a][i];
            }
        }
        let best: Vec<usize> = score
            .iter()
            .map(|s| (0..n).max_by(|&x, &y| s[x].partial_cmp(&s[y]).unwrap()).unwrap())
            .collect();
        deals.iter().enumerate().map(|(i, (d, _))| vs[best[d.bucket[p][street] as usize]][i]).collect()
    }
}

/// Value to player `p` of a best response against `sigma`.
pub fn best_response_value(game: &TreeGame, p: usize, deals: &[(Deal, f64)], sigma: &Strategy) -> f64 {
    let reach: Vec<f64> = deals.iter().map(|(_, w)| *w).collect();
    let v = br(game, 0, p, &reach, deals, sigma);
    v.iter().zip(&reach).map(|(x, w)| x * w).sum()
}

/// Expected value to player 0 when both play `sigma`.
pub fn expected_value(game: &TreeGame, deals: &[(Deal, f64)], sigma: &Strategy) -> f64 {
    fn ev(game: &TreeGame, node: u32, d: &Deal, sigma: &Strategy) -> f64 {
        let nd = &game.nodes[node as usize];
        if nd.player == TERMINAL {
            return game.utility(nd, 0, d.showdown);
        }
        let n = nd.n as usize;
        let mut s = [0f64; MAX_ACTIONS];
        sigma(node, d.bucket[nd.player as usize][nd.street as usize], &mut s[..n]);
        game.children(nd).iter().zip(&s[..n]).map(|(&c, &w)| if w > 0.0 { w * ev(game, c, d, sigma) } else { 0.0 }).sum()
    }
    deals.iter().map(|(d, w)| w * ev(game, 0, d, sigma)).sum()
}

/// Exploitability: the average of the two best-response values (0 at a Nash
/// equilibrium), in chips per game.
pub fn exploitability(game: &TreeGame, deals: &[(Deal, f64)], sigma: &Strategy) -> f64 {
    let b0 = best_response_value(game, 0, deals, sigma);
    let b1 = best_response_value(game, 1, deals, sigma);
    (b0 + b1) / 2.0
}
