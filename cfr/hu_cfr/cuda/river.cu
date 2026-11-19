// The river half of a turn solve on the GPU (see hu_cfr/src/live/gpu.rs).
//
// A turn solve's traversal has a river subtree for every (chance node,
// river card) pair, each with its own regrets ("instances"). Each pass of
// the solve, this kernel walks all of them at once: one thread block per
// instance, its threads owning fixed hands, depth-first through the subtree
// exactly like the CPU's `walk` (live/solver.rs); a second kernel averages
// each chance node's rivers. The turn's own nodes stay on the CPU.
//
// The arithmetic is the CPU's, operation for operation and in the same
// order, so GPU and CPU solves give bit-identical results: built with
// -fmad=false (no fused multiply-adds); the cross-hand sums at fold and
// showdown terminals follow the CPU's fixed chunked order (chunk sums in
// parallel, the chain over chunks and each card's chain by single threads,
// see live/showdown.rs); and a regret update's final subtraction is applied
// when the regret is next read (`vp`), with the same two roundings.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#define H 1326
#define TPB 256
#define HPT 6  // hands per thread: TPB * HPT >= H
#define MAXL 32
#define MAXB 1081  // hands on a complete board

enum { K_DEC = 0, K_FOLD = 1, K_SHOW = 2 };

struct Tmpl {
    long long arena0;  // card 0's arena in the regret/sum arrays
    long long vp0;     // card 0's pending values (see `vp`)
    int node0;         // first node in the node arrays
    int nnodes;
    int arena_len;     // floats per card
    int vp_len;        // floats per card
};

struct Board {
    int n;       // hands not touching the board (sorted by strength)
    int off;     // into idx / ca / cb / grp
    int ngroups;
    int goff;    // into groups (ngroups + 1 entries)
    int loff;    // into clist (52 × 51) and clen (52)
};

struct Args {
    const int* inst;
    int nc;
    const Tmpl* tmpl;
    const int8_t* kind;
    const int8_t* who;
    const int8_t* nk;
    const int* first;
    const int* kids;
    const int* off;
    const int* voff;   // each decision node's pending values (H) in its card's vp arena
    const int* c0;
    const int* c1;
    const uint8_t* cards;
    const Board* boards;
    const uint16_t* idx;
    const uint8_t* ca;
    const uint8_t* cb;
    const uint32_t* groups;
    const uint16_t* grp;    // each sorted hand's group
    const uint16_t* dl;     // sorted positions, groups descending (ascending within)
    const uint16_t* cu;     // per board and card: its hands' positions ascending, packed
    const uint16_t* cd;     // the same in the down order
    const uint8_t* clen;
    const uint8_t* gs;      // per position: starts a group
    const uint8_t* gsd;     // per down-order position: starts a group
    const uint16_t* wcard;  // the 51 hands holding each card, ascending
    const uint8_t* ha;
    const uint8_t* hb;
    float* regret;
    float* sum;
    // The traverser's regrets are stored as disc(r) + cv and each decision
    // node's value v kept here, the regret being (stored − v): the CPU's
    // `disc(r) + cv − v` with the same two roundings, without a second pass
    // over the regrets to subtract v.
    float* vp;
    const float* rp;
    const float* ro;
    float* out;
    float* scratch;
    long long stride;  // scratch floats per block
    int levels;
    int p;
    int best;
    float pos, neg, avg;
    unsigned long long* prof;  // per category cycle sums (profiling), or null
};

// Shared memory: ro/out [H] f32; w, net, below and above at each group,
// card parts A/B [MAXB] f32 each; card sums [53], chunk sums and starts
// (below, above) [4 × 64] f32; idx, grp, down order [MAXB] u16; card chains
// up/down [2 × 52 × 51] u16; ha/hb [H] u8; group starts (up, down) [MAXB] u8
// each; clen [52] u8.
#define SH_F (H + 6 * MAXB + 53 + 4 * 64)
#define SH_BYTES (4 * SH_F + 2 * (3 * MAXB + 2 * 52 * 51) + 2 * H + 2 * MAXB + 52)
#define CHUNK 32
#define FOLD_CHUNK 64

#define FOR_HANDS _Pragma("unroll") for (int j = 0; j < HPT; j++)

__global__ void __launch_bounds__(TPB, 4) river_pass(Args A) {
    extern __shared__ __align__(16) unsigned char shm[];
    float* s_ro = (float*)shm;
    float* s_w = s_ro + H;
    float* s_net = s_w + MAXB;
    float* s_gu = s_net + MAXB;
    float* s_gd = s_gu + MAXB;
    float* s_au = s_gd + MAXB;
    float* s_bu = s_au + MAXB;
    float* s_card = s_bu + MAXB;
    float* s_cs = s_card + 53;
    float* s_ps = s_cs + 64;
    float* s_csd = s_ps + 64;
    float* s_psd = s_csd + 64;
    uint16_t* s_idx = (uint16_t*)(s_psd + 64);
    uint16_t* s_grp = s_idx + MAXB;
    uint16_t* s_dl = s_grp + MAXB;
    uint16_t* s_cu = s_dl + MAXB;
    uint16_t* s_cd = s_cu + 52 * 51;
    uint8_t* s_ha = (uint8_t*)(s_cd + 52 * 51);
    uint8_t* s_hb = s_ha + H;
    uint8_t* s_gs = s_hb + H;
    uint8_t* s_gsd = s_gs + MAXB;
    uint8_t* s_clen = s_gsd + MAXB;

    const int tid = threadIdx.x;
    long long t_start = clock64(), t_fold = 0, t_show = 0, t_mark;
    const int inst = A.inst[blockIdx.x];
    const int ch = inst / A.nc;
    const int c = inst % A.nc;
    const Tmpl T = A.tmpl[ch];
    const int card = A.cards[c];
    const Board B = A.boards[c];
    float* __restrict__ reg = A.regret + T.arena0 + (long long)c * T.arena_len;
    float* __restrict__ sum = A.sum + T.arena0 + (long long)c * T.arena_len;
    float* __restrict__ vp = A.vp + T.vp0 + (long long)c * T.vp_len;
    float* __restrict__ S = A.scratch + (long long)blockIdx.x * A.stride;
    float* RET = S + (long long)4 * A.levels * H;
#define RP(L) (S + (long long)(4 * (L) + 0) * H)
#define RO(L) (S + (long long)(4 * (L) + 1) * H)
#define VV(L) (S + (long long)(4 * (L) + 2) * H)
#define TOT(L) (S + (long long)(4 * (L) + 3) * H)
#define HAND (tid + j * TPB)

    // The board and the hands' cards, staged once.
    for (int k = tid; k < B.n; k += TPB) {
        s_idx[k] = A.idx[B.off + k];
        s_grp[k] = A.grp[B.off + k];
        s_dl[k] = A.dl[B.off + k];
        s_gs[k] = A.gs[B.off + k];
        s_gsd[k] = A.gsd[B.off + k];
    }
    for (int k = tid; k < 52 * 51; k += TPB) {
        s_cu[k] = A.cu[(long long)B.loff * 51 + k];
        s_cd[k] = A.cd[(long long)B.loff * 51 + k];
    }
    for (int k = tid; k < 52; k += TPB) s_clen[k] = A.clen[B.loff + k];
    for (int h = tid; h < H; h += TPB) {
        s_ha[h] = A.ha[h];
        s_hb[h] = A.hb[h];
    }
    __syncthreads();

    // The root's reach: the chance node's, without the card's hands.
    FOR_HANDS {
        const int h = HAND;
        if (h < H) {
            const bool hc = s_ha[h] == card || s_hb[h] == card;
            RP(0)[h] = hc ? 0.0f : A.rp[(long long)ch * H + h];
            RO(0)[h] = hc ? 0.0f : A.ro[(long long)ch * H + h];
        }
    }

    const int p = A.p;
    const bool best = A.best != 0;
    const float* __restrict__ src = best ? sum : reg;  // what strategies come from
    int stk_node[MAXL];
    int stk_next[MAXL];
    int d = 0;
    int node = 0;
    enum { ENTER, NEXT, UP } state = ENTER;

    for (;;) {
        if (state == ENTER) {
            const int g = T.node0 + node;
            const int kd = A.kind[g];
            t_mark = clock64();
            if (kd != K_DEC) {
                // A terminal: the traverser's values against RO(d).
                const float* ro = RO(d);
                FOR_HANDS {
                    const int h = HAND;
                    if (h < H) s_ro[h] = ro[h];
                }
                __syncthreads();
                if (kd == K_FOLD) {
                    const int w = A.who[g];
                    const float v = (float)(w == p ? (p == 0 ? A.c1[g] : A.c0[g]) : -(p == 0 ? A.c0[g] : A.c1[g]));
                    // The total in chunks (as the CPU sums it) and each
                    // card's sum (hands ascending), as independent chains.
                    if (tid < 32) {
                        const int nch = (H + FOLD_CHUNK - 1) / FOLD_CHUNK;
                        if (tid < nch) {
                            float t = 0.0f;
                            const int h1 = min(H, (tid + 1) * FOLD_CHUNK);
                            for (int h = tid * FOLD_CHUNK; h < h1; h++) t += s_ro[h];
                            s_cs[tid] = t;
                        }
                        __syncwarp();
                        if (tid == 0) {
                            float total = 0.0f;
                            for (int i = 0; i < nch; i++) total += s_cs[i];
                            s_card[52] = total;
                        }
                    } else if (tid >= 32 && tid < 32 + 52) {
                        const int cc = tid - 32;
                        const uint16_t* wc = A.wcard + cc * 51;
                        float acc = 0.0f;
#pragma unroll 17
                        for (int j = 0; j < 51; j++) {
                            const float x = s_ro[__ldg(wc + j)];
                            if (x != 0.0f) acc += x;
                        }
                        s_card[cc] = acc;
                    }
                    __syncthreads();
                    const float total = s_card[52];
                    FOR_HANDS {
                        const int h = HAND;
                        if (h < H) RET[h] = v * (total - s_card[s_ha[h]] - s_card[s_hb[h]] + s_ro[h]);
                    }
                } else {
                    const float sv = (float)(A.c0[g] < A.c1[g] ? A.c0[g] : A.c1[g]);
                    const int n = B.n;
                    for (int k = tid; k < n; k += TPB) s_w[k] = s_ro[s_idx[k]];
                    __syncthreads();
                    // In the CPU's order (`chunked_prefix`): the weight below
                    // each group (warp 0) and above it (warp 1, the down
                    // order) as chunked sums, and each card's part of the
                    // weight below (52 threads), all at once.
                    const int nch = (n + CHUNK - 1) / CHUNK;
                    if (tid < 64) {
                        const int lane = tid & 31;
                        const bool up = tid < 32;
                        float* cs = up ? s_cs : s_csd;
                        float* ps = up ? s_ps : s_psd;
                        for (int ci = lane; ci < nch; ci += 32) {
                            float t = 0.0f;
                            const int j1 = min(n, (ci + 1) * CHUNK);
                            for (int j = ci * CHUNK; j < j1; j++) t += s_w[up ? j : s_dl[j]];
                            cs[ci] = t;
                        }
                        __syncwarp();
                        if (lane == 0) {
                            float st = 0.0f;
                            for (int i = 0; i < nch; i++) {
                                ps[i] = st;
                                st += cs[i];
                            }
                        }
                        __syncwarp();
                        const uint8_t* starts = up ? s_gs : s_gsd;
                        float* gv = up ? s_gu : s_gd;
                        for (int ci = lane; ci < nch; ci += 32) {
                            float t = ps[ci];
                            const int j1 = min(n, (ci + 1) * CHUNK);
                            for (int j = ci * CHUNK; j < j1; j++) {
                                const int k = up ? j : s_dl[j];
                                if (starts[j]) gv[s_grp[k]] = t;
                                t += s_w[k];
                            }
                        }
                    } else if (tid >= 64 && tid < 64 + 52) {
                        // Each card's part of the weight below: its hands in
                        // position order, the value before each group.
                        const int cc = tid - 64;
                        const uint16_t* L = s_cu + cc * 51;
                        const int len = s_clen[cc];
                        float run = 0.0f, before = 0.0f;
                        for (int j = 0; j < len; j++) {
                            const int e = L[j];
                            const int k = e & 0x7FF;
                            if (e & 0x800) before = run;
                            ((e & 0x1000) ? s_bu : s_au)[k] = before;
                            run += s_w[k];
                        }
                    }
                    for (int h = tid; h < H; h += TPB) s_ro[h] = 0.0f;
                    __syncthreads();
                    for (int k = tid; k < n; k += TPB) s_net[k] = s_gu[s_grp[k]] - s_au[k] - s_bu[k];
                    __syncthreads();
                    // Each card's part of the weight above (groups descending,
                    // positions ascending within).
                    if (tid >= 64 && tid < 64 + 52) {
                        const int cc = tid - 64;
                        const uint16_t* L = s_cd + cc * 51;
                        const int len = s_clen[cc];
                        float run = 0.0f, before = 0.0f;
                        for (int j = 0; j < len; j++) {
                            const int e = L[j];
                            const int k = e & 0x7FF;
                            if (e & 0x800) before = run;
                            ((e & 0x1000) ? s_bu : s_au)[k] = before;
                            run += s_w[k];
                        }
                    }
                    __syncthreads();
                    for (int k = tid; k < n; k += TPB) {
                        float net = s_net[k];
                        net -= s_gd[s_grp[k]] - s_au[k] - s_bu[k];
                        s_ro[s_idx[k]] = sv * net;
                    }
                    __syncthreads();
                    FOR_HANDS {
                        const int h = HAND;
                        if (h < H) RET[h] = s_ro[h];
                    }
                }
                __syncthreads();
                if (kd == K_FOLD) t_fold += clock64() - t_mark; else t_show += clock64() - t_mark;
                state = UP;
            } else {
                // A decision: its strategy's normalizer, and a fresh value.
                const int n = A.nk[g];
                const long long o = A.off[g];
                const int vo = A.voff[g];
                float t[HPT], pv[HPT];
                FOR_HANDS {
                    const int h = HAND;
                    t[j] = 0.0f;
                    pv[j] = (h < H && !best) ? vp[vo + h] : 0.0f;
                }
                for (int a = 0; a < n; a++) {
                    float x[HPT];
                    FOR_HANDS {
                        const int h = HAND;
                        x[j] = h < H ? src[o + (long long)a * H + h] : 0.0f;
                    }
                    FOR_HANDS t[j] += best ? x[j] : fmaxf(x[j] - pv[j], 0.0f);
                }
                float* tot = TOT(d);
                float* vv = VV(d);
                FOR_HANDS {
                    const int h = HAND;
                    if (h < H) {
                        tot[h] = t[j];
                        vv[h] = 0.0f;
                    }
                }
                stk_node[d] = node;
                stk_next[d] = 0;
                state = NEXT;
            }
        } else if (state == NEXT) {
            node = stk_node[d];
            const int g = T.node0 + node;
            const int a = stk_next[d];
            const int n = A.nk[g];
            const long long o = A.off[g];
            if (a == n) {
                // Done with this node: finish the regret update, pass the value up.
                const bool mine = A.who[g] == p;
                const float* vv = VV(d);
                float v[HPT];
                FOR_HANDS {
                    const int h = HAND;
                    v[j] = h < H ? vv[h] : 0.0f;
                }
                if (mine && !best) {
                    const int vo = A.voff[g];
                    FOR_HANDS {
                        const int h = HAND;
                        if (h < H) vp[vo + h] = v[j];
                    }
                }
                FOR_HANDS {
                    const int h = HAND;
                    if (h < H) RET[h] = v[j];
                }
                state = UP;
                continue;
            }
            const int q = A.who[g];
            const float uni = 1.0f / (float)n;
            const float* tot = TOT(d);
            const float* rpd = RP(d);
            const float* rod = RO(d);
            float* rpn = RP(d + 1);
            float* ron = RO(d + 1);
            const int vo = A.voff[g];
            float t[HPT], x[HPT], rp[HPT], ro[HPT];
            FOR_HANDS {
                const int h = HAND;
                const bool in = h < H;
                t[j] = in ? tot[h] : 0.0f;
                x[j] = in ? (best ? src[o + (long long)a * H + h] : src[o + (long long)a * H + h] - vp[vo + h]) : 0.0f;
                rp[j] = in ? rpd[h] : 0.0f;
                ro[j] = in ? rod[h] : 0.0f;
            }
            int any = 0;
            FOR_HANDS {
                const int h = HAND;
                if (h < H) {
                    const float sg = t[j] > 0.0f ? (best ? x[j] : fmaxf(x[j], 0.0f)) / t[j] : uni;
                    if (q == p) {
                        rpn[h] = best ? rp[j] : rp[j] * sg;
                        ron[h] = ro[j];
                    } else {
                        any |= (sg != 0.0f && ro[j] != 0.0f);
                        ron[h] = ro[j] * sg;
                        rpn[h] = rp[j];
                    }
                }
            }
            if (q != p && !__syncthreads_or(any)) {
                stk_next[d] = a + 1;  // the opponent never gets here
                continue;
            }
            node = A.kids[A.first[g] + a];
            d += 1;
            state = ENTER;
        } else {
            // UP: RET is the value of the node at level d.
            if (d == 0) break;
            d -= 1;
            node = stk_node[d];
            const int g = T.node0 + node;
            const int a = stk_next[d];
            const int n = A.nk[g];
            const long long o = A.off[g];
            const int q = A.who[g];
            const float uni = 1.0f / (float)n;
            float* vv = VV(d);
            if (q == p && !best) {
                const float* tot = TOT(d);
                const float* rpd = RP(d);
                const int vo = A.voff[g];
                float cv[HPT], t[HPT], r[HPT], v[HPT], sm[HPT], rp[HPT];
                FOR_HANDS {
                    const int h = HAND;
                    const bool in = h < H;
                    cv[j] = in ? RET[h] : 0.0f;
                    t[j] = in ? tot[h] : 0.0f;
                    r[j] = in ? reg[o + (long long)a * H + h] - vp[vo + h] : 0.0f;
                    v[j] = in ? vv[h] : 0.0f;
                    sm[j] = in ? sum[o + (long long)a * H + h] : 0.0f;
                    rp[j] = in ? rpd[h] : 0.0f;
                }
                FOR_HANDS {
                    const int h = HAND;
                    if (h < H) {
                        const float sg = t[j] > 0.0f ? fmaxf(r[j], 0.0f) / t[j] : uni;
                        vv[h] = v[j] + sg * cv[j];
                        reg[o + (long long)a * H + h] = (r[j] > 0.0f ? r[j] * A.pos : r[j] * A.neg) + cv[j];
                        sum[o + (long long)a * H + h] = sm[j] * A.avg + rp[j] * sg;
                    }
                }
            } else {
                float cv[HPT], v[HPT];
                FOR_HANDS {
                    const int h = HAND;
                    const bool in = h < H;
                    cv[j] = in ? RET[h] : 0.0f;
                    v[j] = in ? vv[h] : 0.0f;
                }
                FOR_HANDS {
                    const int h = HAND;
                    if (h < H) vv[h] = q == p ? (a == 0 ? cv[j] : fmaxf(v[j], cv[j])) : v[j] + cv[j];
                }
            }
            stk_next[d] = a + 1;
            state = NEXT;
        }
    }

    FOR_HANDS {
        const int h = HAND;
        if (h < H) {
            const bool hc = s_ha[h] == card || s_hb[h] == card;
            A.out[(long long)inst * H + h] = hc ? 0.0f : RET[h];
        }
    }
    if (A.prof && tid == 0) {
        const long long tot = clock64() - t_start;
        atomicAdd(&A.prof[0], (unsigned long long)tot);
        atomicAdd(&A.prof[1], (unsigned long long)t_fold);
        atomicAdd(&A.prof[2], (unsigned long long)t_show);
        atomicAdd(&A.prof[3], 1ull);
    }
}

// Each live chance node's value per hand: its rivers' root values summed in
// card order, over the 44 river cards each hand pair allows (as the CPU).
__global__ void chance_avg(const int* live_ch, int nlive, int nc, const float* out, float* val) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nlive * H) return;
    const int ch = live_ch[i / H];
    const int h = i % H;
    float v = 0.0f;
    for (int c = 0; c < nc; c++) v += out[((long long)ch * nc + c) * H + h];
    val[(long long)ch * H + h] = v / 44.0f;
}

// ---- host side -------------------------------------------------------------

extern "C" {

typedef struct {
    int nch;  // chance nodes (one river subtree template each)
    int nc;   // river cards
    // per template
    const long long* t_arena0;
    const int* t_node0;
    const int* t_nnodes;
    const int* t_arena_len;
    const long long* t_vp0;
    const int* t_vp_len;
    int levels;  // the deepest subtree's depth + 1
    // per node (all templates, concatenated; child links are template-local)
    int n_nodes;
    const int8_t* kind;
    const int8_t* who;
    const int8_t* nk;
    const int* first;
    const int* off;
    const int* voff;
    const int* c0;
    const int* c1;
    int n_kids;
    const int* kids;
    long long arena_total;  // floats in all arenas (each of regret, sum)
    long long vp_total;     // floats in all pending-value arenas
    // per card
    const uint8_t* cards;
    const int* b_n;
    const int* b_off;
    const int* b_ngroups;
    const int* b_goff;
    int idx_total;
    const uint16_t* idx;
    const uint8_t* ca;
    const uint8_t* cb;
    int groups_total;
    const uint32_t* groups;
    const uint16_t* grp;    // idx_total entries
    const uint16_t* dl;     // idx_total entries
    const uint16_t* cu;     // nc × 52 × 51: position | new group << 11 | B slot << 12
    const uint16_t* cd;     // nc × 52 × 51, the down order
    const uint8_t* clen;    // nc × 52
    const uint8_t* gs;      // idx_total
    const uint8_t* gsd;     // idx_total
    const uint16_t* wcard;  // 52 × 51
    // the hands' cards
    const uint8_t* ha;
    const uint8_t* hb;
} HuGpuDesc;

}  // extern "C"

struct Ctx {
    cudaStream_t st;
    int nch, nc, levels;
    long long stride;
    Args a;
    // device buffers to free
    void* bufs[64];
    int nbufs;
    int* d_inst;
    float* d_rp;
    float* d_ro;
    float* d_out;
    float* d_val;
    int* d_live;
    int* h_inst;
    int* h_live;
    cudaEvent_t e0, e1;
    unsigned long long* d_prof;
};

static double g_kernel_ms = 0.0;
extern "C" double hu_gpu_kernel_ms(void) { return g_kernel_ms; }

static int check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "hu gpu: %s: %s\n", what, cudaGetErrorString(e));
        return 1;
    }
    return 0;
}

template <typename T>
static T* upload(Ctx* x, const T* src, size_t n, int* err) {
    T* d = nullptr;
    if (*err) return nullptr;
    size_t bytes = n * sizeof(T) > 0 ? n * sizeof(T) : 4;
    if (check(cudaMalloc((void**)&d, bytes), "cudaMalloc")) {
        *err = 1;
        return nullptr;
    }
    if (x->nbufs == 64) abort();
    x->bufs[x->nbufs++] = d;
    if (n > 0 && check(cudaMemcpyAsync(d, src, n * sizeof(T), cudaMemcpyHostToDevice, x->st), "upload")) *err = 1;
    return d;
}

template <typename T>
static T* alloc(Ctx* x, size_t n, int* err) {
    T* d = nullptr;
    if (*err) return nullptr;
    if (check(cudaMalloc((void**)&d, (n > 0 ? n : 1) * sizeof(T)), "cudaMalloc")) {
        *err = 1;
        return nullptr;
    }
    if (x->nbufs == 64) abort();
    x->bufs[x->nbufs++] = d;
    return d;
}

extern "C" void hu_gpu_free(void* ctx) {
    Ctx* x = (Ctx*)ctx;
    if (!x) return;
    cudaStreamSynchronize(x->st);
    if (x->d_prof) {
        unsigned long long h[4];
        cudaMemcpy(h, x->d_prof, sizeof(h), cudaMemcpyDeviceToHost);
        double n = (double)(h[3] ? h[3] : 1);
        fprintf(stderr, "gpu blocks %llu: cycles per block %.0f (fold %.0f, showdown %.0f, other %.0f)\n", h[3], h[0] / n, h[1] / n, h[2] / n, (h[0] - h[1] - h[2]) / n);
    }
    for (int i = 0; i < x->nbufs; i++) cudaFree(x->bufs[i]);
    if (x->e0) cudaEventDestroy(x->e0);
    if (x->e1) cudaEventDestroy(x->e1);
    cudaStreamDestroy(x->st);
    free(x->h_inst);
    free(x->h_live);
    free(x);
}

extern "C" int hu_gpu_available(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n > 0;
}

extern "C" void* hu_gpu_new(const HuGpuDesc* D) {
    if (D->levels > MAXL - 1) {
        fprintf(stderr, "hu gpu: river subtree too deep (%d levels)\n", D->levels);
        return nullptr;
    }
    Ctx* x = (Ctx*)calloc(1, sizeof(Ctx));
    int err = check(cudaStreamCreateWithFlags(&x->st, cudaStreamNonBlocking), "stream");
    if (err) {
        free(x);
        return nullptr;
    }
    x->nch = D->nch;
    x->nc = D->nc;
    x->levels = D->levels;
    x->stride = (long long)(4 * D->levels + 1) * H;
    std::vector<Tmpl> t(D->nch);
    for (int i = 0; i < D->nch; i++) {
        t[i].arena0 = D->t_arena0[i];
        t[i].node0 = D->t_node0[i];
        t[i].nnodes = D->t_nnodes[i];
        t[i].arena_len = D->t_arena_len[i];
        t[i].vp0 = D->t_vp0[i];
        t[i].vp_len = D->t_vp_len[i];
    }
    std::vector<Board> b(D->nc);
    for (int i = 0; i < D->nc; i++) {
        b[i].n = D->b_n[i];
        b[i].off = D->b_off[i];
        b[i].ngroups = D->b_ngroups[i];
        b[i].goff = D->b_goff[i];
        b[i].loff = i * 52;
    }
    Args& a = x->a;
    a.nc = D->nc;
    a.tmpl = upload(x, t.data(), t.size(), &err);
    a.kind = upload(x, D->kind, D->n_nodes, &err);
    a.who = upload(x, D->who, D->n_nodes, &err);
    a.nk = upload(x, D->nk, D->n_nodes, &err);
    a.first = upload(x, D->first, D->n_nodes, &err);
    a.off = upload(x, D->off, D->n_nodes, &err);
    a.voff = upload(x, D->voff, D->n_nodes, &err);
    a.c0 = upload(x, D->c0, D->n_nodes, &err);
    a.c1 = upload(x, D->c1, D->n_nodes, &err);
    a.kids = upload(x, D->kids, D->n_kids, &err);
    a.cards = upload(x, D->cards, D->nc, &err);
    a.boards = upload(x, b.data(), b.size(), &err);
    a.idx = upload(x, D->idx, D->idx_total, &err);
    a.ca = upload(x, D->ca, D->idx_total, &err);
    a.cb = upload(x, D->cb, D->idx_total, &err);
    a.groups = upload(x, D->groups, D->groups_total, &err);
    a.grp = upload(x, D->grp, D->idx_total, &err);
    a.dl = upload(x, D->dl, D->idx_total, &err);
    a.cu = upload(x, D->cu, (size_t)D->nc * 52 * 51, &err);
    a.cd = upload(x, D->cd, (size_t)D->nc * 52 * 51, &err);
    a.gs = upload(x, D->gs, D->idx_total, &err);
    a.gsd = upload(x, D->gsd, D->idx_total, &err);
    a.clen = upload(x, D->clen, (size_t)D->nc * 52, &err);
    a.wcard = upload(x, D->wcard, 52 * 51, &err);
    a.ha = upload(x, D->ha, H, &err);
    a.hb = upload(x, D->hb, H, &err);
    a.regret = alloc<float>(x, D->arena_total, &err);
    a.sum = alloc<float>(x, D->arena_total, &err);
    a.vp = alloc<float>(x, D->vp_total, &err);
    const long long ninst = (long long)D->nch * D->nc;
    a.scratch = alloc<float>(x, (size_t)(ninst * x->stride), &err);
    x->d_rp = alloc<float>(x, (size_t)D->nch * H, &err);
    x->d_ro = alloc<float>(x, (size_t)D->nch * H, &err);
    x->d_out = alloc<float>(x, (size_t)ninst * H, &err);
    x->d_inst = alloc<int>(x, (size_t)ninst, &err);
    x->d_val = alloc<float>(x, (size_t)D->nch * H, &err);
    if (getenv("HU_GPU_PROFILE")) {
        x->d_prof = alloc<unsigned long long>(x, 4, &err);
        if (!err) cudaMemsetAsync(x->d_prof, 0, 4 * sizeof(unsigned long long), x->st);
    }
    x->d_live = alloc<int>(x, (size_t)D->nch, &err);
    if (!err) err |= check(cudaEventCreate(&x->e0), "event");
    if (!err) err |= check(cudaEventCreate(&x->e1), "event");
    if (!err) err |= check(cudaMemsetAsync(a.regret, 0, D->arena_total * sizeof(float), x->st), "memset");
    if (!err) err |= check(cudaMemsetAsync(a.sum, 0, D->arena_total * sizeof(float), x->st), "memset");
    if (!err) err |= check(cudaMemsetAsync(a.vp, 0, D->vp_total * sizeof(float), x->st), "memset");
    if (!err) err |= check(cudaMemsetAsync(x->d_out, 0, (size_t)ninst * H * sizeof(float), x->st), "memset");
    if (!err) err |= check(cudaFuncSetAttribute(river_pass, cudaFuncAttributeMaxDynamicSharedMemorySize, SH_BYTES), "smem");
    if (!err) err |= check(cudaStreamSynchronize(x->st), "init sync");
    x->h_inst = (int*)malloc(sizeof(int) * (ninst > 0 ? ninst : 1));
    x->h_live = (int*)malloc(sizeof(int) * (D->nch > 0 ? D->nch : 1));
    a.rp = x->d_rp;
    a.ro = x->d_ro;
    a.out = x->d_out;
    a.inst = x->d_inst;
    a.stride = x->stride;
    a.levels = D->levels;
    if (err) {
        hu_gpu_free(x);
        return nullptr;
    }
    return x;
}

// One pass's river work: `rp`/`ro` are the reaches at each chance node
// (nch × H), `live` says which chance nodes the opponent reaches; `val`
// (nch × H) gets each live chance node's value (dead ones untouched).
extern "C" int hu_gpu_pass(void* ctx, int p, int best, float pos, float neg, float avg, const float* rp, const float* ro, const uint8_t* live, float* val) {
    Ctx* x = (Ctx*)ctx;
    int n = 0, nl = 0;
    for (int ch = 0; ch < x->nch; ch++)
        if (live[ch]) {
            x->h_live[nl++] = ch;
            for (int c = 0; c < x->nc; c++) x->h_inst[n++] = ch * x->nc + c;
        }
    if (n == 0) return 0;
    int err = 0;
    err |= check(cudaMemcpyAsync(x->d_rp, rp, (size_t)x->nch * H * sizeof(float), cudaMemcpyHostToDevice, x->st), "rp");
    err |= check(cudaMemcpyAsync(x->d_ro, ro, (size_t)x->nch * H * sizeof(float), cudaMemcpyHostToDevice, x->st), "ro");
    err |= check(cudaMemcpyAsync(x->d_inst, x->h_inst, (size_t)n * sizeof(int), cudaMemcpyHostToDevice, x->st), "inst");
    err |= check(cudaMemcpyAsync(x->d_live, x->h_live, (size_t)nl * sizeof(int), cudaMemcpyHostToDevice, x->st), "live");
    if (err) return 1;
    Args a = x->a;
    a.p = p;
    a.best = best;
    a.pos = pos;
    a.neg = neg;
    a.avg = avg;
    a.prof = x->d_prof;
    cudaEventRecord(x->e0, x->st);
    river_pass<<<n, TPB, SH_BYTES, x->st>>>(a);
    err |= check(cudaGetLastError(), "launch");
    chance_avg<<<(nl * H + 255) / 256, 256, 0, x->st>>>(x->d_live, nl, x->nc, x->d_out, x->d_val);
    err |= check(cudaGetLastError(), "launch avg");
    cudaEventRecord(x->e1, x->st);
    for (int i = 0; i < nl; i++) {
        const int ch = x->h_live[i];
        err |= check(cudaMemcpyAsync(val + (size_t)ch * H, x->d_val + (size_t)ch * H, H * sizeof(float), cudaMemcpyDeviceToHost, x->st), "val");
    }
    err |= check(cudaStreamSynchronize(x->st), "pass sync");
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, x->e0, x->e1) == cudaSuccess) g_kernel_ms += ms;
    return err;
}
