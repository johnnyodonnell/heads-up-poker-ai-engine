//! The river half of turn solves on the GPU (`cuda/river.cu`, built with the
//! `gpu` feature).
//!
//! Each pass of a turn solve walks a river subtree for every (chance node,
//! river card) pair. With the GPU, the CPU still walks the turn's own nodes
//! (top-down for reaches, bottom-up for values and updates) and the GPU
//! walks all those river subtrees at once, one thread block each, keeping
//! their regrets in GPU memory for the whole solve. The kernel does the
//! CPU's arithmetic in the CPU's order, so results are bit-identical
//! (`hu livebench --compare-gpu` checks it).

use std::ffi::c_void;

use hu_core::cards::Card;

use super::showdown::{holes, Board, H};
use super::subgame::{Kind, Subgame};

#[repr(C)]
struct HuGpuDesc {
    nch: i32,
    nc: i32,
    t_arena0: *const i64,
    t_node0: *const i32,
    t_nnodes: *const i32,
    t_arena_len: *const i32,
    t_vp0: *const i64,
    t_vp_len: *const i32,
    levels: i32,
    n_nodes: i32,
    kind: *const i8,
    who: *const i8,
    nk: *const i8,
    first: *const i32,
    off: *const i32,
    voff: *const i32,
    c0: *const i32,
    c1: *const i32,
    n_kids: i32,
    kids: *const i32,
    arena_total: i64,
    vp_total: i64,
    cards: *const u8,
    b_n: *const i32,
    b_off: *const i32,
    b_ngroups: *const i32,
    b_goff: *const i32,
    idx_total: i32,
    idx: *const u16,
    ca: *const u8,
    cb: *const u8,
    groups_total: i32,
    groups: *const u32,
    grp: *const u16,
    dl: *const u16,
    cu: *const u16,
    cd: *const u16,
    clen: *const u8,
    gs: *const u8,
    gsd: *const u8,
    wcard: *const u16,
    ha: *const u8,
    hb: *const u8,
}

#[cfg(feature = "gpu")]
extern "C" {
    fn hu_gpu_available() -> i32;
    fn hu_gpu_new(d: *const HuGpuDesc) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    fn hu_gpu_pass(ctx: *mut c_void, p: i32, best: i32, pos: f32, neg: f32, avg: f32, rp: *const f32, ro: *const f32, live: *const u8, out: *mut f32) -> i32;
    fn hu_gpu_free(ctx: *mut c_void);
    fn hu_gpu_kernel_ms() -> f64;
}

/// Total GPU kernel time so far, process-wide (ms; approximate when solves
/// run concurrently).
pub fn kernel_ms() -> f64 {
    #[cfg(feature = "gpu")]
    unsafe {
        hu_gpu_kernel_ms()
    }
    #[cfg(not(feature = "gpu"))]
    0.0
}

/// Whether this build has the GPU solver and a GPU to run it on.
pub fn available() -> bool {
    #[cfg(feature = "gpu")]
    {
        static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *OK.get_or_init(|| unsafe { hu_gpu_available() } != 0)
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// Time spent in GPU passes (ns) and passes run, process-wide (profiling).
pub static PASS_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PASSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Turn solves on the GPU right now, process-wide.
static IN_USE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// A claim on one of the GPU's solve slots; released when dropped.
pub struct Slot(());

impl Slot {
    /// One of `limit` slots if one is free (no waiting: a solve that finds
    /// the GPU busy runs on the CPU, with the same result).
    pub fn try_take(limit: usize) -> Option<Slot> {
        use std::sync::atomic::Ordering::*;
        let mut cur = IN_USE.load(Relaxed);
        while cur < limit {
            match IN_USE.compare_exchange_weak(cur, cur + 1, AcqRel, Relaxed) {
                Ok(_) => return Some(Slot(())),
                Err(x) => cur = x,
            }
        }
        None
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        IN_USE.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

/// One turn solve's river work on the GPU.
pub struct GpuRiver {
    ctx: *mut c_void,
    nch: usize,
    #[allow(dead_code)]
    nc: usize,
}

// The context is used by one thread at a time (its solver's).
unsafe impl Send for GpuRiver {}

impl GpuRiver {
    /// For a turn solve: each chance node's river subtree is the node range
    /// `spans[i]`, its decision nodes at `off` in a per-card arena of
    /// `lens[i]` floats; `boards`/`cards` are the solver's river boards.
    /// None if there is no GPU or it fails to set up.
    pub fn new(g: &Subgame, spans: &[(usize, usize)], off: &[usize], lens: &[usize], boards: &[Board], cards: &[Card]) -> Option<GpuRiver> {
        if !available() || spans.is_empty() {
            return None;
        }
        let nch = spans.len();
        let nc = cards.len();
        let (mut t_arena0, mut t_node0, mut t_nnodes, mut t_len) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut t_vp0, mut t_vp_len, mut voff) = (Vec::new(), Vec::new(), Vec::new());
        let mut vp_total = 0i64;
        let (mut kind, mut who, mut nk, mut first, mut offs, mut c0, mut c1, mut kids) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut arena = 0i64;
        let mut levels = 1usize;
        for (i, &(start, end)) in spans.iter().enumerate() {
            t_arena0.push(arena);
            t_node0.push(kind.len() as i32);
            t_nnodes.push((end - start) as i32);
            t_len.push(lens[i] as i32);
            arena += (nc * lens[i]) as i64;
            let decisions = (start..end).filter(|&id| matches!(g.nodes[id].kind, Kind::Decision(_))).count();
            t_vp0.push(vp_total);
            t_vp_len.push((decisions * H) as i32);
            vp_total += (nc * decisions * H) as i64;
            let mut dix = 0usize;
            let mut depth = vec![0usize; end - start];
            for id in start..end {
                let nd = &g.nodes[id];
                let (k, w) = match nd.kind {
                    Kind::Decision(q) => (0i8, q as i8),
                    Kind::Fold(w) => (1, w as i8),
                    Kind::Showdown => (2, 0),
                    Kind::Chance => return None,
                };
                kind.push(k);
                who.push(w);
                nk.push(if k == 0 { nd.n as i8 } else { 0 });
                first.push(kids.len() as i32);
                offs.push(if k == 0 { off[id] as i32 } else { 0 });
                voff.push(if k == 0 { (dix * H) as i32 } else { 0 });
                if k == 0 {
                    dix += 1;
                }
                c0.push(nd.contrib[0]);
                c1.push(nd.contrib[1]);
                for &kid in g.children(nd) {
                    let local = kid as usize - start;
                    depth[local] = depth[id - start] + 1;
                    levels = levels.max(depth[local] + 1);
                    kids.push(local as i32);
                }
            }
        }
        let (mut b_n, mut b_off, mut b_ng, mut b_goff) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut idx, mut ca, mut cb, mut groups) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut grp, mut dl, mut cu, mut cd, mut clen, mut gs, mut gsd) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for b in boards {
            let (i, a, c, gr, down) = b.parts();
            b_n.push(i.len() as i32);
            b_off.push(idx.len() as i32);
            b_ng.push(gr.len() as i32 - 1);
            b_goff.push(groups.len() as i32);
            idx.extend_from_slice(i);
            ca.extend_from_slice(a);
            cb.extend_from_slice(c);
            groups.extend_from_slice(gr);
            // Each sorted position's group, and each card's positions.
            for gi in 0..gr.len() - 1 {
                for _ in gr[gi]..gr[gi + 1] {
                    grp.push(gi as u16);
                }
            }
            dl.extend_from_slice(down);
            // Group starts, positions ascending and in the down order.
            let g_of = |k: usize| grp[grp.len() - i.len() + k];
            for k in 0..i.len() {
                gs.push(u8::from(k == 0 || g_of(k) != g_of(k - 1)));
            }
            for j in 0..down.len() {
                gsd.push(u8::from(j == 0 || g_of(down[j] as usize) != g_of(down[j - 1] as usize)));
            }
            // Each card's hands, packed: position, whether it starts a new
            // group in the chain, and whether the card is the hand's second.
            let pack = |order: &mut dyn Iterator<Item = usize>, out: &mut Vec<u16>| {
                let mut lists = vec![Vec::<u16>::new(); 52];
                let mut last = [u16::MAX; 52];
                for k in order {
                    for (slot, card) in [(0u16, a[k]), (1, c[k])] {
                        let cc = card as usize;
                        let new = u16::from(last[cc] != g_of(k));
                        last[cc] = g_of(k);
                        lists[cc].push(k as u16 | new << 11 | slot << 12);
                    }
                }
                for l in &lists {
                    let mut row = l.clone();
                    row.resize(51, 0);
                    out.extend_from_slice(&row);
                }
            };
            pack(&mut (0..i.len()), &mut cu);
            pack(&mut down.iter().map(|&k| k as usize), &mut cd);
            let mut n_card = [0u8; 52];
            for k in 0..i.len() {
                n_card[a[k] as usize] += 1;
                n_card[c[k] as usize] += 1;
            }
            clen.extend_from_slice(&n_card);
        }
        let wcard: Vec<u16> = super::showdown::with_card().iter().flat_map(|r| r.iter().copied()).collect();
        let ha: Vec<u8> = holes().iter().map(|h| h[0]).collect();
        let hb: Vec<u8> = holes().iter().map(|h| h[1]).collect();
        let d = HuGpuDesc {
            nch: nch as i32,
            nc: nc as i32,
            t_arena0: t_arena0.as_ptr(),
            t_node0: t_node0.as_ptr(),
            t_nnodes: t_nnodes.as_ptr(),
            t_arena_len: t_len.as_ptr(),
            t_vp0: t_vp0.as_ptr(),
            t_vp_len: t_vp_len.as_ptr(),
            levels: levels as i32,
            n_nodes: kind.len() as i32,
            kind: kind.as_ptr(),
            who: who.as_ptr(),
            nk: nk.as_ptr(),
            first: first.as_ptr(),
            off: offs.as_ptr(),
            voff: voff.as_ptr(),
            c0: c0.as_ptr(),
            c1: c1.as_ptr(),
            n_kids: kids.len() as i32,
            kids: kids.as_ptr(),
            arena_total: arena,
            vp_total,
            cards: cards.as_ptr(),
            b_n: b_n.as_ptr(),
            b_off: b_off.as_ptr(),
            b_ngroups: b_ng.as_ptr(),
            b_goff: b_goff.as_ptr(),
            idx_total: idx.len() as i32,
            idx: idx.as_ptr(),
            ca: ca.as_ptr(),
            cb: cb.as_ptr(),
            groups_total: groups.len() as i32,
            groups: groups.as_ptr(),
            grp: grp.as_ptr(),
            dl: dl.as_ptr(),
            cu: cu.as_ptr(),
            cd: cd.as_ptr(),
            clen: clen.as_ptr(),
            gs: gs.as_ptr(),
            gsd: gsd.as_ptr(),
            wcard: wcard.as_ptr(),
            ha: ha.as_ptr(),
            hb: hb.as_ptr(),
        };
        Self::create(&d, nch, nc)
    }

    #[cfg(feature = "gpu")]
    fn create(d: &HuGpuDesc, nch: usize, nc: usize) -> Option<GpuRiver> {
        let ctx = unsafe { hu_gpu_new(d) };
        (!ctx.is_null()).then_some(GpuRiver { ctx, nch, nc })
    }

    #[cfg(not(feature = "gpu"))]
    fn create(_: &HuGpuDesc, _: usize, _: usize) -> Option<GpuRiver> {
        None
    }

    /// One pass's river work for traverser `p`: `rp`/`ro` are the reaches at
    /// each chance node (nch × H), `live[i]` whether chance node i is
    /// reached; returns each live chance node's value per hand (its rivers'
    /// root values averaged as the CPU does; nch × H, dead ones 0).
    #[allow(clippy::too_many_arguments)]
    pub fn pass(&mut self, p: usize, best: bool, pos: f32, neg: f32, avg: f32, rp: &[f32], ro: &[f32], live: &[u8]) -> Vec<f32> {
        assert_eq!(rp.len(), self.nch * H);
        assert_eq!(ro.len(), self.nch * H);
        assert_eq!(live.len(), self.nch);
        let t = std::time::Instant::now();
        let mut out = vec![0f32; self.nch * H];
        self.run(p, best, pos, neg, avg, rp, ro, live, &mut out);
        PASS_NS.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        PASSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        out
    }

    #[cfg(feature = "gpu")]
    #[allow(clippy::too_many_arguments)]
    fn run(&mut self, p: usize, best: bool, pos: f32, neg: f32, avg: f32, rp: &[f32], ro: &[f32], live: &[u8], out: &mut [f32]) {
        let err = unsafe { hu_gpu_pass(self.ctx, p as i32, best as i32, pos, neg, avg, rp.as_ptr(), ro.as_ptr(), live.as_ptr(), out.as_mut_ptr()) };
        assert_eq!(err, 0, "GPU river pass failed");
    }

    #[cfg(not(feature = "gpu"))]
    #[allow(clippy::too_many_arguments)]
    fn run(&mut self, _: usize, _: bool, _: f32, _: f32, _: f32, _: &[f32], _: &[f32], _: &[u8], _: &mut [f32]) {
        unreachable!("no GPU in this build")
    }
}

impl Drop for GpuRiver {
    fn drop(&mut self) {
        #[cfg(feature = "gpu")]
        unsafe {
            hu_gpu_free(self.ctx)
        }
    }
}
