//! Full-width Discounted CFR (Brown & Sandholm 2019: α = 1.5, β = 0, γ = 2)
//! on a live subgame, vectorized over all 1,326 hands of both players.
//!
//! Each traversal carries both players' reach vectors and returns the
//! traverser's counterfactual values per hand. Updates alternate between the
//! players within an iteration. A turn solve's river is a chance node: each
//! of the 48 river cards has its own copy of the river regrets, solved in
//! parallel and summed in card order, so results don't depend on threading.
//!
//! Nodes can be frozen to a given strategy (the bot's own earlier actions on
//! this street keep the strategy it actually played).

use rayon::prelude::*;

use hu_core::cards::Card;

use super::showdown::{fold, holes, mask, pair_weight, Board, H};
use super::subgame::{Kind, Subgame};

const ALPHA: f64 = 1.5;
const GAMMA: f64 = 2.0;
/// River cards consistent with a given pair of hands on a 4-card board.
const RIVERS_PER_PAIR: f32 = 44.0;

#[derive(Default)]
struct Arena {
    regret: Vec<f32>,
    sum: Vec<f32>,
}

impl Arena {
    fn new(n: usize) -> Arena {
        Arena { regret: vec![0.0; n], sum: vec![0.0; n] }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Train,
    /// Best response against the average strategy.
    Best,
}

#[derive(Clone, Copy)]
struct Disc {
    pos: f32,
    neg: f32,
    avg: f32,
}

struct Ctx<'a> {
    g: &'a Subgame,
    boards: &'a [Board],
    cards: &'a [Card],
    off: &'a [usize],
    frozen: &'a [Option<Vec<f32>>],
    mode: Mode,
    disc: Disc,
    parallel: bool,
}

pub struct Solver<'a> {
    pub g: &'a Subgame,
    boards: Vec<Board>,
    /// The river card of each board (turn solves).
    cards: Vec<Card>,
    off: Vec<usize>,
    main: Arena,
    per_card: Vec<Arena>,
    frozen: Vec<Option<Vec<f32>>>,
    /// Reach of each hand at the root, per seat.
    pub ranges: [Vec<f32>; 2],
    pub iterations: u32,
    pub parallel: bool,
}

impl<'a> Solver<'a> {
    /// `board`: the 5 river cards (river solve) or 4 turn cards (turn
    /// solve). `ranges` are per seat; hands touching the board are zeroed.
    /// `frozen` fixes the strategy (n × H, action-major) at some nodes.
    pub fn new(g: &'a Subgame, board: &[Card], mut ranges: [Vec<f32>; 2], frozen: Vec<(u32, Vec<f32>)>, parallel: bool) -> Solver<'a> {
        for r in ranges.iter_mut() {
            mask(r, board);
        }
        let (boards, cards) = if board.len() == 5 {
            (vec![Board::new([board[0], board[1], board[2], board[3], board[4]])], Vec::new())
        } else {
            assert_eq!(board.len(), 4, "a turn or river board");
            let cards: Vec<Card> = (0..52u8).filter(|c| !board.contains(c)).collect();
            let boards = cards.iter().map(|&c| Board::new([board[0], board[1], board[2], board[3], c])).collect();
            (boards, cards)
        };
        let mut off = vec![usize::MAX; g.nodes.len()];
        let (mut main_len, mut card_len) = (0usize, 0usize);
        for (i, nd) in g.nodes.iter().enumerate() {
            if let Kind::Decision(_) = nd.kind {
                let len = nd.n as usize * H;
                if nd.river && !cards.is_empty() {
                    off[i] = card_len;
                    card_len += len;
                } else {
                    off[i] = main_len;
                    main_len += len;
                }
            }
        }
        let per_card = if cards.is_empty() { Vec::new() } else { (0..cards.len()).map(|_| Arena::new(card_len)).collect() };
        let mut fz = vec![None; g.nodes.len()];
        for (node, s) in frozen {
            assert_eq!(s.len(), g.nodes[node as usize].n as usize * H, "frozen strategy size");
            fz[node as usize] = Some(s);
        }
        Solver { g, boards, cards, off, main: Arena::new(main_len), per_card, frozen: fz, ranges, iterations: 0, parallel }
    }

    /// Floats of solver state (regrets and strategy sums).
    pub fn size(&self) -> usize {
        2 * (self.main.regret.len() + self.per_card.iter().map(|a| a.regret.len()).sum::<usize>())
    }

    fn walk_root(&mut self, p: usize, mode: Mode, disc: Disc) -> Vec<f32> {
        let cx = Ctx {
            g: self.g,
            boards: &self.boards,
            cards: &self.cards,
            off: &self.off,
            frozen: &self.frozen,
            mode,
            disc,
            parallel: self.parallel,
        };
        walk(&cx, &mut self.main, &mut self.per_card, 0, p, &self.ranges[p], &self.ranges[1 - p], 0)
    }

    /// Run `iters` more DCFR iterations.
    pub fn run(&mut self, iters: u32) {
        for _ in 0..iters {
            self.iterations += 1;
            let t = (self.iterations - 1) as f64; // iterations already accumulated
            let tp = t.powf(ALPHA);
            let disc = Disc {
                pos: (tp / (tp + 1.0)) as f32,
                neg: 0.5, // β = 0
                avg: ((t / (t + 1.0)).powf(GAMMA)) as f32,
            };
            for p in 0..2 {
                self.walk_root(p, Mode::Train, disc);
            }
        }
    }

    /// Average strategy at a decision node (n × H, action-major; uniform
    /// where a hand has none). Frozen nodes return their fixed strategy.
    pub fn strategy(&self, node: u32) -> Vec<f32> {
        if let Some(f) = &self.frozen[node as usize] {
            return f.clone();
        }
        let nd = &self.g.nodes[node as usize];
        assert!(!(nd.river && !self.cards.is_empty()), "average of a per-card node");
        let n = nd.n as usize;
        let off = self.off[node as usize];
        let mut out = vec![0f32; n * H];
        normalize(&self.main.sum[off..off + n * H], n, &mut out);
        out
    }

    /// Exploitability of the average strategies, in chips per hand pair,
    /// within this subgame (frozen nodes stay fixed for both players).
    pub fn exploitability(&mut self) -> f64 {
        let disc = Disc { pos: 1.0, neg: 1.0, avg: 1.0 };
        let mut total = 0.0;
        for p in 0..2 {
            let v = self.walk_root(p, Mode::Best, disc);
            total += self.ranges[p].iter().zip(&v).map(|(&r, &x)| r as f64 * x as f64).sum::<f64>();
        }
        let pairs = pair_weight(&self.ranges[0], &self.ranges[1]);
        if pairs > 0.0 {
            total / pairs / 2.0
        } else {
            0.0
        }
    }

    /// Expected value per hand for `p` under the average strategies
    /// (counterfactual: not divided by the opponent's reach).
    pub fn values(&mut self, p: usize) -> Vec<f32> {
        let disc = Disc { pos: 1.0, neg: 1.0, avg: 1.0 };
        let cx = Ctx {
            g: self.g,
            boards: &self.boards,
            cards: &self.cards,
            off: &self.off,
            frozen: &self.frozen,
            mode: Mode::Best,
            disc,
            parallel: self.parallel,
        };
        expect(&cx, &self.main, &self.per_card, 0, p, &self.ranges[1 - p], 0)
    }
}

/// Regret matching (n × H, action-major).
fn regret_match(regret: &[f32], n: usize, out: &mut [f32]) {
    for h in 0..H {
        let mut total = 0f32;
        for a in 0..n {
            total += regret[a * H + h].max(0.0);
        }
        if total > 0.0 {
            for a in 0..n {
                out[a * H + h] = regret[a * H + h].max(0.0) / total;
            }
        } else {
            for a in 0..n {
                out[a * H + h] = 1.0 / n as f32;
            }
        }
    }
}

fn normalize(sum: &[f32], n: usize, out: &mut [f32]) {
    for h in 0..H {
        let mut total = 0f32;
        for a in 0..n {
            total += sum[a * H + h];
        }
        for a in 0..n {
            out[a * H + h] = if total > 0.0 { sum[a * H + h] / total } else { 1.0 / n as f32 };
        }
    }
}

fn terminal(cx: &Ctx, node: u32, p: usize, ro: &[f32], slot: usize) -> Vec<f32> {
    let nd = &cx.g.nodes[node as usize];
    let mut out = vec![0f32; H];
    match nd.kind {
        Kind::Fold(w) => {
            let v = if w as usize == p { nd.contrib[1 - p] } else { -nd.contrib[p] };
            fold(ro, v as f32, &mut out);
        }
        Kind::Showdown => {
            let s = nd.contrib[0].min(nd.contrib[1]) as f32;
            cx.boards[slot].showdown(ro, s, &mut out);
        }
        _ => unreachable!(),
    }
    out
}

/// Deal each river card: `f(card index, arena)` per card, summed in card
/// order and divided by the rivers each hand pair can see.
fn chance<F>(cx: &Ctx, per_card: &mut [Arena], f: F) -> Vec<f32>
where
    F: Fn(usize, &mut Arena) -> Vec<f32> + Sync + Send,
{
    let results: Vec<Vec<f32>> = if cx.parallel {
        per_card.par_iter_mut().enumerate().map(|(c, ar)| f(c, ar)).collect()
    } else {
        per_card.iter_mut().enumerate().map(|(c, ar)| f(c, ar)).collect()
    };
    let mut v = vec![0f32; H];
    for r in &results {
        for (x, y) in v.iter_mut().zip(r) {
            *x += y;
        }
    }
    for x in v.iter_mut() {
        *x /= RIVERS_PER_PAIR;
    }
    v
}

fn masked(r: &[f32], card: Card) -> Vec<f32> {
    let mut v = r.to_vec();
    mask(&mut v, &[card]);
    v
}

fn zero_card(v: &mut [f32], card: Card) {
    let hs = holes();
    for (h, x) in v.iter_mut().enumerate() {
        if hs[h][0] == card || hs[h][1] == card {
            *x = 0.0;
        }
    }
}

/// Counterfactual values for `p`; updates regrets and averages in Train.
#[allow(clippy::too_many_arguments)]
fn walk(cx: &Ctx, ar: &mut Arena, per_card: &mut [Arena], node: u32, p: usize, rp: &[f32], ro: &[f32], slot: usize) -> Vec<f32> {
    let g = cx.g;
    let nd = &g.nodes[node as usize];
    match nd.kind {
        Kind::Fold(_) | Kind::Showdown => terminal(cx, node, p, ro, slot),
        Kind::Chance => {
            let child = g.children(nd)[0];
            chance(cx, per_card, |c, car| {
                let card = cx.cards[c];
                let (rp2, ro2) = (masked(rp, card), masked(ro, card));
                let mut v = walk(cx, car, &mut [], child, p, &rp2, &ro2, c);
                zero_card(&mut v, card);
                v
            })
        }
        Kind::Decision(q) => {
            let q = q as usize;
            let n = nd.n as usize;
            let kids = g.children(nd);
            let off = cx.off[node as usize];
            let frozen = cx.frozen[node as usize].as_ref();
            let mut sigma = vec![0f32; n * H];
            match (frozen, cx.mode) {
                (Some(f), _) => sigma.copy_from_slice(f),
                (None, Mode::Train) => regret_match(&ar.regret[off..off + n * H], n, &mut sigma),
                (None, Mode::Best) => normalize(&ar.sum[off..off + n * H], n, &mut sigma),
            }
            let mut v = vec![0f32; H];
            if q == p {
                let mut cv: Vec<Vec<f32>> = Vec::with_capacity(n);
                for (a, &kid) in kids.iter().enumerate() {
                    let child_rp: Vec<f32> = match cx.mode {
                        Mode::Train => rp.iter().zip(&sigma[a * H..(a + 1) * H]).map(|(r, s)| r * s).collect(),
                        Mode::Best => rp.to_vec(),
                    };
                    cv.push(walk(cx, ar, per_card, kid, p, &child_rp, ro, slot));
                }
                if cx.mode == Mode::Best && frozen.is_none() {
                    v.copy_from_slice(&cv[0]);
                    for c in &cv[1..] {
                        for (x, y) in v.iter_mut().zip(c) {
                            *x = x.max(*y);
                        }
                    }
                } else {
                    for (a, c) in cv.iter().enumerate() {
                        let s = &sigma[a * H..(a + 1) * H];
                        for h in 0..H {
                            v[h] += s[h] * c[h];
                        }
                    }
                }
                if cx.mode == Mode::Train && frozen.is_none() {
                    let Disc { pos, neg, avg } = cx.disc;
                    for (a, c) in cv.iter().enumerate() {
                        let reg = &mut ar.regret[off + a * H..off + (a + 1) * H];
                        for h in 0..H {
                            let r = reg[h];
                            reg[h] = if r > 0.0 { r * pos } else { r * neg } + c[h] - v[h];
                        }
                        let sum = &mut ar.sum[off + a * H..off + (a + 1) * H];
                        let s = &sigma[a * H..(a + 1) * H];
                        for h in 0..H {
                            sum[h] = sum[h] * avg + rp[h] * s[h];
                        }
                    }
                }
            } else {
                for (a, &kid) in kids.iter().enumerate() {
                    let s = &sigma[a * H..(a + 1) * H];
                    if s.iter().zip(ro).all(|(x, r)| *x == 0.0 || *r == 0.0) {
                        continue; // the opponent never gets here
                    }
                    let child_ro: Vec<f32> = ro.iter().zip(s).map(|(r, s)| r * s).collect();
                    let c = walk(cx, ar, per_card, kid, p, rp, &child_ro, slot);
                    for (x, y) in v.iter_mut().zip(&c) {
                        *x += y;
                    }
                }
            }
            v
        }
    }
}

/// Values for `p` with both players on their average strategies.
fn expect(cx: &Ctx, ar: &Arena, per_card: &[Arena], node: u32, p: usize, ro: &[f32], slot: usize) -> Vec<f32> {
    let g = cx.g;
    let nd = &g.nodes[node as usize];
    match nd.kind {
        Kind::Fold(_) | Kind::Showdown => terminal(cx, node, p, ro, slot),
        Kind::Chance => {
            let child = g.children(nd)[0];
            let mut v = vec![0f32; H];
            for (c, car) in per_card.iter().enumerate() {
                let card = cx.cards[c];
                let mut x = expect(cx, car, &[], child, p, &masked(ro, card), c);
                zero_card(&mut x, card);
                for (a, b) in v.iter_mut().zip(&x) {
                    *a += b / RIVERS_PER_PAIR;
                }
            }
            v
        }
        Kind::Decision(q) => {
            let n = nd.n as usize;
            let off = cx.off[node as usize];
            let mut sigma = vec![0f32; n * H];
            match cx.frozen[node as usize].as_ref() {
                Some(f) => sigma.copy_from_slice(f),
                None => normalize(&ar.sum[off..off + n * H], n, &mut sigma),
            }
            let mut v = vec![0f32; H];
            for (a, &kid) in g.children(nd).iter().enumerate() {
                let s = &sigma[a * H..(a + 1) * H];
                if q as usize == p {
                    let c = expect(cx, ar, per_card, kid, p, ro, slot);
                    for h in 0..H {
                        v[h] += s[h] * c[h];
                    }
                } else {
                    let child_ro: Vec<f32> = ro.iter().zip(s).map(|(r, s)| r * s).collect();
                    let c = expect(cx, ar, per_card, kid, p, &child_ro, slot);
                    for (x, y) in v.iter_mut().zip(&c) {
                        *x += y;
                    }
                }
            }
            v
        }
    }
}
