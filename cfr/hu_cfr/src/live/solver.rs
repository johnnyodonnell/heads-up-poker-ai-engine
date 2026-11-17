//! Full-width Discounted CFR (Brown & Sandholm 2019: α = 1.5, β = 0, γ = 2)
//! on a live subgame, vectorized over all 1,326 hands of both players.
//!
//! Each traversal carries both players' reach vectors and returns the
//! traverser's counterfactual values per hand. Updates alternate between the
//! players within an iteration.
//!
//! A river solve is one recursive traversal. A turn solve deals the river at
//! each line that reaches it (a chance node), and every (chance node, river
//! card) pair has its own copy of the river regrets. Its traversal runs in
//! three passes so all of that river work is one parallel batch:
//!   1. top-down over the turn's nodes: reach vectors and strategies;
//!   2. every (chance node, card) river subtree, in parallel;
//!   3. bottom-up over the turn's nodes: values and regret updates.
//! Results are summed in a fixed order, so they don't depend on threading.
//!
//! Nodes can be frozen to a given strategy (the bot's own earlier actions on
//! this street keep the strategy it actually played).

use rayon::prelude::*;

use hu_core::cards::Card;

use super::showdown::{fold, mask, pair_weight, with_card, Board, H};
use super::subgame::{Kind, Subgame};

const ALPHA: f64 = 1.5;
const GAMMA: f64 = 2.0;
/// River cards consistent with a given pair of hands on a 4-card board.
const RIVERS_PER_PAIR: f32 = 44.0;
const NONE: usize = usize::MAX;

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

/// What the recursive traversal needs besides the arena it updates.
struct Ctx<'a> {
    g: &'a Subgame,
    boards: &'a [Board],
    off: &'a [usize],
    frozen: &'a [Option<Vec<f32>>],
    mode: Mode,
    disc: Disc,
}

pub struct Solver<'a> {
    pub g: &'a Subgame,
    boards: Vec<Board>,
    /// The river card of each board (turn solves; empty for a river solve).
    cards: Vec<Card>,
    /// Offset of each decision node in its arena: the main arena for the
    /// solve's own street, its (chance node, card) arena on a turn solve's
    /// river.
    off: Vec<usize>,
    /// The turn's chance nodes, in id order, and each node's index among
    /// them (NONE elsewhere).
    chances: Vec<usize>,
    chance_of: Vec<usize>,
    /// For a turn solve, the nodes before the river, in id order.
    turn_nodes: Vec<usize>,
    main: Arena,
    /// `river[i * cards + c]`: chance node i's river with card c.
    river: Vec<Arena>,
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
        let nn = g.nodes.len();
        // Each chance node's river: its child's subtree, a contiguous run of
        // ids (the tree is built depth-first).
        let mut chances = Vec::new();
        let mut chance_of = vec![NONE; nn];
        let mut owner = vec![NONE; nn];
        let mut spans = Vec::new();
        for (i, nd) in g.nodes.iter().enumerate() {
            if nd.kind == Kind::Chance && owner[i] == NONE {
                let child = g.children(nd)[0] as usize;
                let end = subtree_end(g, child);
                chance_of[i] = chances.len();
                for o in &mut owner[child..end] {
                    *o = chances.len();
                }
                chances.push(i);
                spans.push((child, end));
            }
        }
        let turn_nodes: Vec<usize> = if cards.is_empty() { Vec::new() } else { (0..nn).filter(|&i| owner[i] == NONE).collect() };
        let mut off = vec![NONE; nn];
        let mut main_len = 0;
        for (i, nd) in g.nodes.iter().enumerate() {
            if matches!(nd.kind, Kind::Decision(_)) && owner[i] == NONE {
                off[i] = main_len;
                main_len += nd.n as usize * H;
            }
        }
        let mut river = Vec::new();
        for &(start, end) in &spans {
            let mut len = 0;
            for i in start..end {
                if matches!(g.nodes[i].kind, Kind::Decision(_)) {
                    off[i] = len;
                    len += g.nodes[i].n as usize * H;
                }
            }
            for _ in 0..cards.len() {
                river.push(Arena::new(len));
            }
        }
        let mut fz = vec![None; nn];
        for (node, s) in frozen {
            assert_eq!(s.len(), g.nodes[node as usize].n as usize * H, "frozen strategy size");
            assert_eq!(owner[node as usize], NONE, "frozen node on the river of a turn solve");
            fz[node as usize] = Some(s);
        }
        Solver {
            g,
            boards,
            cards,
            off,
            chances,
            chance_of,
            turn_nodes,
            main: Arena::new(main_len),
            river,
            frozen: fz,
            ranges,
            iterations: 0,
            parallel,
        }
    }

    /// Floats of solver state (regrets and strategy sums).
    pub fn size(&self) -> usize {
        2 * (self.main.regret.len() + self.river.iter().map(|a| a.regret.len()).sum::<usize>())
    }

    fn walk_root(&mut self, p: usize, mode: Mode, disc: Disc) -> Vec<f32> {
        if self.cards.is_empty() {
            let cx = Ctx { g: self.g, boards: &self.boards, off: &self.off, frozen: &self.frozen, mode, disc };
            walk(&cx, &mut self.main, 0, p, &self.ranges[p], &self.ranges[1 - p], 0)
        } else {
            self.turn_pass(p, mode, disc)
        }
    }

    /// One traversal of a turn solve (see the module docs).
    fn turn_pass(&mut self, p: usize, mode: Mode, disc: Disc) -> Vec<f32> {
        let g = self.g;
        let nn = g.nodes.len();
        let mut rp: Vec<Option<Vec<f32>>> = vec![None; nn];
        let mut ro: Vec<Option<Vec<f32>>> = vec![None; nn];
        let mut sig: Vec<Option<Vec<f32>>> = vec![None; nn];
        // Nodes the opponent never reaches (values 0, nothing to update).
        let mut dead = vec![false; nn];
        rp[0] = Some(self.ranges[p].clone());
        ro[0] = Some(self.ranges[1 - p].clone());

        // 1. Top-down: reach and strategy at each turn node.
        for &i in &self.turn_nodes {
            let nd = &g.nodes[i];
            let Kind::Decision(q) = nd.kind else { continue };
            if dead[i] {
                for &k in g.children(nd) {
                    dead[k as usize] = true;
                }
                continue;
            }
            let n = nd.n as usize;
            let off = self.off[i];
            let mut sigma = vec![0f32; n * H];
            match (&self.frozen[i], mode) {
                (Some(f), _) => sigma.copy_from_slice(f),
                (None, Mode::Train) => regret_match(&self.main.regret[off..off + n * H], n, &mut sigma),
                (None, Mode::Best) => normalize(&self.main.sum[off..off + n * H], n, &mut sigma),
            }
            let (rpi, roi) = (rp[i].clone().unwrap(), ro[i].clone().unwrap());
            for (a, &k) in g.children(nd).iter().enumerate() {
                let s = &sigma[a * H..(a + 1) * H];
                let k = k as usize;
                if q as usize == p {
                    rp[k] = Some(match mode {
                        Mode::Train => rpi.iter().zip(s).map(|(r, s)| r * s).collect(),
                        Mode::Best => rpi.clone(),
                    });
                    ro[k] = Some(roi.clone());
                } else {
                    let child: Vec<f32> = roi.iter().zip(s).map(|(r, s)| r * s).collect();
                    dead[k] = child.iter().all(|&x| x == 0.0);
                    rp[k] = Some(rpi.clone());
                    ro[k] = Some(child);
                }
            }
            sig[i] = Some(sigma);
        }

        // 2. Every live (chance node, card) river, in parallel.
        let nc = self.cards.len();
        let cx = Ctx { g, boards: &self.boards, off: &self.off, frozen: &self.frozen, mode, disc };
        let (chances, cards, rp_ref, ro_ref, dead_ref) = (&self.chances, &self.cards, &rp, &ro, &dead);
        let task = |idx: usize, ar: &mut Arena| -> Option<Vec<f32>> {
            let ch = chances[idx / nc];
            if dead_ref[ch] {
                return None;
            }
            let c = idx % nc;
            let card = cards[c];
            let (mut a, mut b) = (rp_ref[ch].clone().unwrap(), ro_ref[ch].clone().unwrap());
            zero_card(&mut a, card);
            zero_card(&mut b, card);
            let child = g.children(&g.nodes[ch])[0];
            let mut v = walk(&cx, ar, child, p, &a, &b, c);
            zero_card(&mut v, card);
            Some(v)
        };
        let results: Vec<Option<Vec<f32>>> = if self.parallel {
            self.river.par_iter_mut().enumerate().map(|(i, ar)| task(i, ar)).collect()
        } else {
            self.river.iter_mut().enumerate().map(|(i, ar)| task(i, ar)).collect()
        };

        // 3. Bottom-up: values, regrets and averages.
        let mut val: Vec<Option<Vec<f32>>> = vec![None; nn];
        for &i in self.turn_nodes.iter().rev() {
            let nd = &g.nodes[i];
            if dead[i] {
                val[i] = Some(vec![0f32; H]);
                continue;
            }
            let v = match nd.kind {
                Kind::Fold(_) | Kind::Showdown => terminal(&cx, i as u32, p, ro[i].as_ref().unwrap(), 0),
                Kind::Chance => {
                    let base = self.chance_of[i] * nc;
                    let mut v = vec![0f32; H];
                    for r in results[base..base + nc].iter().flatten() {
                        for (x, y) in v.iter_mut().zip(r) {
                            *x += y;
                        }
                    }
                    v.iter_mut().for_each(|x| *x /= RIVERS_PER_PAIR);
                    v
                }
                Kind::Decision(q) => {
                    let n = nd.n as usize;
                    let sigma = sig[i].as_ref().unwrap();
                    let kids = g.children(nd);
                    let cv: Vec<Vec<f32>> = kids.iter().map(|&k| val[k as usize].take().unwrap()).collect();
                    if q as usize == p {
                        let frozen = self.frozen[i].is_some();
                        let v = combine(&cv, sigma, mode == Mode::Best && !frozen);
                        if mode == Mode::Train && !frozen {
                            let off = self.off[i];
                            update(&mut self.main, off, n, &cv, &v, sigma, rp[i].as_ref().unwrap(), disc);
                        }
                        v
                    } else {
                        let mut v = vec![0f32; H];
                        for c in &cv {
                            for (x, y) in v.iter_mut().zip(c) {
                                *x += y;
                            }
                        }
                        v
                    }
                }
            };
            val[i] = Some(v);
        }
        val[0].take().unwrap()
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

    /// Average strategy at a decision node of the solve's own street
    /// (n × H, action-major; uniform where a hand has none). Frozen nodes
    /// return their fixed strategy.
    pub fn strategy(&self, node: u32) -> Vec<f32> {
        if let Some(f) = &self.frozen[node as usize] {
            return f.clone();
        }
        let nd = &self.g.nodes[node as usize];
        assert!(!(nd.river && !self.cards.is_empty()), "average of a river node in a turn solve");
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
}

/// One past the last id of `node`'s subtree (subtrees are contiguous).
fn subtree_end(g: &Subgame, node: usize) -> usize {
    let mut end = node + 1;
    let mut stack = vec![node];
    while let Some(i) = stack.pop() {
        end = end.max(i + 1);
        for &k in g.children(&g.nodes[i]) {
            stack.push(k as usize);
        }
    }
    end
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

/// The traverser's node value from its children's: the strategy's mix, or
/// the best child per hand (best response).
fn combine(cv: &[Vec<f32>], sigma: &[f32], best: bool) -> Vec<f32> {
    let mut v = vec![0f32; H];
    if best {
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
    v
}

/// DCFR regret and average-strategy update at one of the traverser's nodes.
#[allow(clippy::too_many_arguments)]
fn update(ar: &mut Arena, off: usize, n: usize, cv: &[Vec<f32>], v: &[f32], sigma: &[f32], rp: &[f32], disc: Disc) {
    let Disc { pos, neg, avg } = disc;
    for (a, c) in cv.iter().enumerate().take(n) {
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

fn zero_card(v: &mut [f32], card: Card) {
    for &h in &with_card()[card as usize] {
        v[h as usize] = 0.0;
    }
}

/// Counterfactual values for `p` below `node` (no chance nodes below);
/// updates regrets and averages in Train.
fn walk(cx: &Ctx, ar: &mut Arena, node: u32, p: usize, rp: &[f32], ro: &[f32], slot: usize) -> Vec<f32> {
    let g = cx.g;
    let nd = &g.nodes[node as usize];
    match nd.kind {
        Kind::Fold(_) | Kind::Showdown => terminal(cx, node, p, ro, slot),
        Kind::Chance => unreachable!("chance nodes are handled by the turn pass"),
        Kind::Decision(q) => {
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
            if q as usize == p {
                let mut cv: Vec<Vec<f32>> = Vec::with_capacity(n);
                for (a, &kid) in kids.iter().enumerate() {
                    let child_rp: Vec<f32> = match cx.mode {
                        Mode::Train => rp.iter().zip(&sigma[a * H..(a + 1) * H]).map(|(r, s)| r * s).collect(),
                        Mode::Best => rp.to_vec(),
                    };
                    cv.push(walk(cx, ar, kid, p, &child_rp, ro, slot));
                }
                let v = combine(&cv, &sigma, cx.mode == Mode::Best && frozen.is_none());
                if cx.mode == Mode::Train && frozen.is_none() {
                    update(ar, off, n, &cv, &v, &sigma, rp, cx.disc);
                }
                v
            } else {
                let mut v = vec![0f32; H];
                for (a, &kid) in kids.iter().enumerate() {
                    let s = &sigma[a * H..(a + 1) * H];
                    if s.iter().zip(ro).all(|(x, r)| *x == 0.0 || *r == 0.0) {
                        continue; // the opponent never gets here
                    }
                    let child_ro: Vec<f32> = ro.iter().zip(s).map(|(r, s)| r * s).collect();
                    let c = walk(cx, ar, kid, p, rp, &child_ro, slot);
                    for (x, y) in v.iter_mut().zip(&c) {
                        *x += y;
                    }
                }
                v
            }
        }
    }
}
