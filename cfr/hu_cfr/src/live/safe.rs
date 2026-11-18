//! Safe re-solving (Burch, Johanson & Bowling 2014; Moravčík et al. 2017;
//! Brown & Sandholm 2017): instead of trusting an estimate of the
//! opponent's range at a live solve's root, give every opponent hand the
//! choice between playing the solve and taking what it would get against
//! the table from here (its "alternative value"). The solver learns that
//! choice per hand along with its own strategy (`Solver::set_gadget`), so
//! its strategy has to hold every hand the opponent could have to no more
//! than it gets against the table, including hands the range estimate says
//! the opponent can't have.
//!
//! The alternative values come from the table itself: translate the hand so
//! far onto the table's betting tree, then walk that tree's subtree from the
//! start of the street over all 1,326 hands, with the bot's side playing the
//! table and the bot's exact range at the root. The opponent's side either
//! plays the table too (`Table`: what the hand gets if both keep playing the
//! table) or best-responds to it (`Best`: the most the hand could get
//! against the table, the strictly safe target). Values are in the table's
//! chips, rescaled to the real pot.

use rayon::prelude::*;

use hu_core::cards::Card;

use super::showdown::{fold, holes, with_card, Board, H};
use crate::engine::{abs_player, Engine, View};
use crate::game::{Terminal, MAX_ACTIONS, TERMINAL};
use crate::strategy::Strategy;

/// River cards consistent with a given pair of hands on a 4-card board.
const RIVERS_PER_PAIR: f32 = 44.0;
const NO_BUCKET: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SafeMode {
    /// Trust the estimated opponent range (unsafe re-solving).
    Off,
    /// Alternative values: the opponent plays the table too.
    Table,
    /// Alternative values: the opponent best-responds to the table.
    Best,
}

impl SafeMode {
    pub fn parse(s: &str) -> Option<SafeMode> {
        match s {
            "off" => Some(SafeMode::Off),
            "table" => Some(SafeMode::Table),
            "best" => Some(SafeMode::Best),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            SafeMode::Off => "off",
            SafeMode::Table => "table",
            SafeMode::Best => "best",
        }
    }
}

struct Ctx<'a> {
    strat: &'a Strategy,
    /// Abstract players of the bot and the opponent.
    pb: u8,
    po: u8,
    best: bool,
    /// Street index of the root (2 = turn, 3 = river).
    root_street: u8,
    /// The river boards: one (river root) or one per river card (turn root).
    boards: &'a [Board],
    cards: &'a [Card],
    /// Each hand's bucket on the root street (NO_BUCKET if it touches the board).
    root_buckets: Vec<u32>,
    /// Turn root: each hand's river bucket per river card.
    river_buckets: Vec<Vec<u32>>,
    /// Walk the river cards in parallel.
    parallel: bool,
}

/// The opponent's alternative value of each hand at the start of the
/// current street (`k` = index of the street's first action in `actions`),
/// against the bot's root range `rb` (board hands zeroed): Σ over bot hands
/// of their weight × the opponent hand's winnings, card removal included,
/// in real chips. `boards`/`cards` are the solver's (`Solver::boards`).
/// None when the real hand has left the table's tree.
#[allow(clippy::too_many_arguments)]
pub fn table_values(
    e: &Engine,
    v: &View,
    actions: &[hu_core::rules::HandAction],
    k: usize,
    rb: &[f32],
    boards: &[Board],
    cards: &[Card],
    best: bool,
    real_pot: i32,
    parallel: bool,
) -> Option<Vec<f32>> {
    let strat = e.pick(v.stacks[0].min(v.stacks[1]));
    let tree = &strat.tree;
    let root = e.table_node(v, actions, k)?;
    let nd = &tree.nodes[root as usize];
    let root_street = if cards.is_empty() { 3u8 } else { 2u8 };
    if nd.player == TERMINAL || nd.street != root_street {
        return None;
    }
    let abs_pot = nd.contrib[0] + nd.contrib[1];
    if abs_pot <= 0 {
        return None;
    }
    let pb = abs_player(v.bot, v.button) as u8;
    let hs = holes();
    let root_board: Vec<Card> = if root_street == 3 { boards[0].cards.to_vec() } else { boards[0].cards[..4].to_vec() };
    let bucket_on = |street: usize, board: &[Card]| -> Vec<u32> {
        hs.iter()
            .map(|&[a, b]| if board.contains(&a) || board.contains(&b) { NO_BUCKET } else { e.abs.bucket(street, [a, b], board) })
            .collect()
    };
    let root_buckets = bucket_on(root_street as usize, &root_board);
    let river_buckets = if root_street != 2 {
        Vec::new()
    } else if parallel {
        boards.par_iter().map(|b| bucket_on(3, &b.cards)).collect()
    } else {
        boards.iter().map(|b| bucket_on(3, &b.cards)).collect()
    };
    let cx = Ctx { strat, pb, po: 1 - pb, best, root_street, boards, cards, root_buckets, river_buckets, parallel };
    let mut out = walk(&cx, root, rb, None);
    let scale = real_pot as f32 / abs_pot as f32;
    for (h, x) in out.iter_mut().enumerate() {
        *x = if cx.root_buckets[h] == NO_BUCKET { 0.0 } else { *x * scale };
    }
    Some(out)
}

fn zero_card(v: &mut [f32], card: Card) {
    for &h in &with_card()[card as usize] {
        v[h as usize] = 0.0;
    }
}

/// Average over the river cards of `f(card index, bot reach without the
/// card's hands)`, each hand's value counted over the cards it allows
/// (summed in card order, so the result doesn't depend on threading).
fn over_rivers(cx: &Ctx, rb: &[f32], f: impl Fn(usize, &[f32]) -> Vec<f32> + Sync) -> Vec<f32> {
    let one = |c: usize| {
        let card = cx.cards[c];
        let mut r = rb.to_vec();
        zero_card(&mut r, card);
        let mut v = f(c, &r);
        zero_card(&mut v, card);
        v
    };
    let vals: Vec<Vec<f32>> = if cx.parallel { (0..cx.cards.len()).into_par_iter().map(one).collect() } else { (0..cx.cards.len()).map(one).collect() };
    let mut acc = vec![0f32; H];
    for v in &vals {
        for (a, x) in acc.iter_mut().zip(v) {
            *a += x;
        }
    }
    acc.iter_mut().for_each(|x| *x /= RIVERS_PER_PAIR);
    acc
}

/// The opponent's values below `node` against bot reach `rb`; `slot` is the
/// river card's index once a turn root's walk has dealt it.
fn walk(cx: &Ctx, node: u32, rb: &[f32], slot: Option<usize>) -> Vec<f32> {
    if rb.iter().all(|&x| x == 0.0) {
        return vec![0f32; H];
    }
    let tree = &cx.strat.tree;
    let nd = &tree.nodes[node as usize];
    // A turn root's walk reaching the river: deal it.
    if cx.root_street == 2 && nd.street == 3 && slot.is_none() {
        return over_rivers(cx, rb, |c, r| walk(cx, node, r, Some(c)));
    }
    let (pb, po) = (cx.pb as usize, cx.po as usize);
    match nd.terminal {
        Some(Terminal::Fold { winner }) => {
            let v = if winner == cx.po { nd.contrib[pb] } else { -nd.contrib[po] };
            let mut out = vec![0f32; H];
            fold(rb, v as f32, &mut out);
            out
        }
        Some(Terminal::Showdown) => {
            let s = nd.contrib[0].min(nd.contrib[1]) as f32;
            if cx.root_street == 3 || slot.is_some() {
                let mut out = vec![0f32; H];
                cx.boards[slot.unwrap_or(0)].showdown(rb, s, &mut out);
                out
            } else {
                // All in on the turn: every river card.
                over_rivers(cx, rb, |c, r| {
                    let mut out = vec![0f32; H];
                    cx.boards[c].showdown(r, s, &mut out);
                    out
                })
            }
        }
        None => {
            let n = nd.n as usize;
            let buckets = match slot {
                Some(c) if nd.street == 3 => &cx.river_buckets[c],
                _ => &cx.root_buckets,
            };
            // The table's strategy per hand (action-major).
            let mut sigma = vec![0f32; n * H];
            let mut p = [0f64; MAX_ACTIONS];
            for (h, &b) in buckets.iter().enumerate() {
                if b == NO_BUCKET {
                    continue;
                }
                cx.strat.probs(node, b, &mut p[..n]);
                for a in 0..n {
                    sigma[a * H + h] = p[a] as f32;
                }
            }
            let kids = tree.children(nd);
            if nd.player as usize == pb {
                let mut v = vec![0f32; H];
                for (a, &kid) in kids.iter().enumerate() {
                    let r: Vec<f32> = rb.iter().zip(&sigma[a * H..(a + 1) * H]).map(|(x, s)| x * s).collect();
                    let c = walk(cx, kid, &r, slot);
                    for (x, y) in v.iter_mut().zip(&c) {
                        *x += y;
                    }
                }
                v
            } else {
                let vals: Vec<Vec<f32>> = kids.iter().map(|&kid| walk(cx, kid, rb, slot)).collect();
                let mut v = vec![0f32; H];
                for h in 0..H {
                    v[h] = if cx.best {
                        vals.iter().map(|c| c[h]).fold(f32::NEG_INFINITY, f32::max)
                    } else {
                        (0..n).map(|a| sigma[a * H + h] * vals[a][h]).sum()
                    };
                }
                v
            }
        }
    }
}
