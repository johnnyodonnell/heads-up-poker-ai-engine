//! Poker-sense probes of a policy (run on every snapshot, and by `hu probe
//! --weights W`): does it know which hands are strong? All at 100bb.
//!
//!   btn_open  the button's first decision, for all 169 hand classes: how
//!             often it plays (doesn't fold) / limps / raises / shoves, the
//!             rank correlation of fold probability with equity, AA's fold and
//!             raise rates, and the typical opening size in big blinds;
//!   bb_defend the big blind facing a 2.5bb open, for all 169 classes: how
//!             often it defends, the rank correlation of fold probability with
//!             equity, and AA's fold rate;
//!   flop      the button facing a half-pot lead after open / call, over random
//!             flops and hands: fold rates for sets, top pair and air, and the
//!             rank correlation of fold probability with equity.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde_json::{json, Value};
use tch::{Device, Kind, Tensor};

use hu_core::actions::{raise_to_for, ActionSpace, ALLIN, CALL, FOLD, NUM_TYPES, RAISE};
use hu_core::cards::{full_deck, make, rank, Card};
use hu_core::encode::{encode_into, INPUT_SIZE};
use hu_core::equity::equity;
use hu_core::rules::{Action, ActionType, State};

use crate::net::HuNet;

const MASK_NEG: f64 = 1e9;

/// Masked type probabilities [n * 4] and Beta (a, b) [n * 2].
fn policy(net: &HuNet, xs: &[f32], masks: &[f32], dev: Device) -> (Vec<f32>, Vec<f32>) {
    let n = (xs.len() / INPUT_SIZE) as i64;
    let x = Tensor::from_slice(xs).view([n, INPUT_SIZE as i64]).to_device(dev);
    let m = Tensor::from_slice(masks).view([n, NUM_TYPES as i64]).to_device(dev);
    let (p, ab) = tch::no_grad(|| {
        let (tl, ab) = net.policy(&x);
        ((tl + (m - 1.0) * MASK_NEG).softmax(1, Kind::Float), ab)
    });
    let v = |t: &Tensor| Vec::<f32>::try_from(&t.to_device(Device::Cpu).contiguous().view([-1])).unwrap();
    (v(&p), v(&ab))
}

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let rank = |v: &[f64]| {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap());
        let mut r = vec![0f64; v.len()];
        for (k, &i) in idx.iter().enumerate() {
            r[i] = k as f64;
        }
        r
    };
    let (ra, rb) = (rank(a), rank(b));
    let m = (a.len() - 1) as f64 / 2.0;
    let cov: f64 = ra.iter().zip(&rb).map(|(x, y)| (x - m) * (y - m)).sum();
    let var: f64 = ra.iter().map(|x| (x - m).powi(2)).sum();
    cov / var
}

/// A fresh 200/200 hand with the button on seat 0.
fn fresh(rng: &mut StdRng) -> State {
    let mut st = State::new_table([200, 200]);
    st.button = 1; // start_hand moves it to seat 0
    let mut deck = full_deck();
    deck.shuffle(rng);
    st.start_hand(deck);
    st
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// The 169 preflop hand classes as (high rank, low rank, suited).
fn classes() -> Vec<(u8, u8, bool)> {
    let mut out = Vec::new();
    for hi in (2..=14u8).rev() {
        for lo in (2..=hi).rev() {
            out.push((hi, lo, false));
            if hi != lo {
                out.push((hi, lo, true));
            }
        }
    }
    out
}

fn combos(c: &(u8, u8, bool)) -> f64 {
    if c.0 == c.1 {
        6.0
    } else if c.2 {
        4.0
    } else {
        12.0
    }
}

/// Encode `st` once per hand class (the mover's hole cards swapped in).
fn per_class(st: &State, cls: &[(u8, u8, bool)]) -> (Vec<f32>, Vec<f32>, Vec<ActionSpace>) {
    let seat = st.to_act as usize;
    let (mut xs, mut masks, mut spaces) = (Vec::new(), Vec::new(), Vec::new());
    for &(h, l, suited) in cls {
        let mut s2 = st.clone();
        s2.players[seat].hole = vec![make(h, 0), make(l, if suited { 0 } else { 1 })];
        let mut x = vec![0f32; INPUT_SIZE];
        let sp = encode_into(&s2, &mut x).expect("decision");
        xs.extend_from_slice(&x);
        masks.extend_from_slice(&sp.mask);
        spaces.push(sp);
    }
    (xs, masks, spaces)
}

pub fn run(net: &HuNet, dev: Device) -> Value {
    let mut rng = StdRng::seed_from_u64(11);
    let cls = classes();
    // Preflop equity of each class vs one random hand.
    let eqs: Vec<f64> = cls
        .iter()
        .map(|&(h, l, suited)| {
            let hole = [make(h, 0), make(l, if suited { 0 } else { 1 })];
            let mut rest: Vec<Card> = (0..52u8).filter(|c| !hole.contains(c)).collect();
            let trials = 2000;
            let mut tot = 0.0;
            for _ in 0..trials {
                let (pick, _) = rest.partial_shuffle(&mut rng, 2);
                let opp = [pick[0], pick[1]];
                tot += equity(&hole, &[&opp], &[], &[], 1, &mut rng);
            }
            tot / trials as f64
        })
        .collect();
    let w: f64 = cls.iter().map(combos).sum();
    let aa = cls.iter().position(|c| *c == (14, 14, false)).unwrap();
    let mut out = json!({});

    // --- the button's first decision ---
    let st = fresh(&mut rng);
    let (xs, masks, spaces) = per_class(&st, &cls);
    let (p, ab) = policy(net, &xs, &masks, dev);
    let pr = |i: usize, t: usize| p[i * NUM_TYPES + t] as f64;
    let fold: Vec<f64> = (0..cls.len()).map(|i| pr(i, FOLD)).collect();
    let rate = |t: usize| cls.iter().enumerate().map(|(i, c)| combos(c) * pr(i, t)).sum::<f64>() / w;
    // Typical open: the raise-to at the Beta's mean band position, in bb,
    // weighted by how often each class raises.
    let (mut open_bb, mut open_w) = (0.0, 0.0);
    for (i, c) in cls.iter().enumerate() {
        let (a, b) = (ab[i * 2] as f64, ab[i * 2 + 1] as f64);
        let to = raise_to_for(&spaces[i], a / (a + b));
        let wr = combos(c) * pr(i, RAISE);
        open_bb += wr * to as f64 / 2.0;
        open_w += wr;
    }
    out["btn_open"] = json!({
        "play": round2(1.0 - rate(FOLD)),
        "limp": round2(rate(CALL)),
        "raise": round2(rate(RAISE)),
        "allin": round2(rate(ALLIN)),
        "rho_fold_equity": round2(spearman(&fold, &eqs)),
        "aa_fold": round2(pr(aa, FOLD)),
        "aa_raise": round2(pr(aa, RAISE) + pr(aa, ALLIN)),
        "open_size_bb": round2(open_bb / open_w.max(1e-9)),
    });

    // --- the big blind facing a 2.5bb open ---
    let mut st = fresh(&mut rng);
    st.apply(Action::to(ActionType::Raise, 5));
    let (xs, masks, _) = per_class(&st, &cls);
    let (p, _) = policy(net, &xs, &masks, dev);
    let pr = |i: usize, t: usize| p[i * NUM_TYPES + t] as f64;
    let fold: Vec<f64> = (0..cls.len()).map(|i| pr(i, FOLD)).collect();
    let defend = cls.iter().enumerate().map(|(i, c)| combos(c) * (1.0 - fold[i])).sum::<f64>() / w;
    let threebet = cls.iter().enumerate().map(|(i, c)| combos(c) * (pr(i, RAISE) + pr(i, ALLIN))).sum::<f64>() / w;
    out["bb_defend"] = json!({
        "defend": round2(defend),
        "3bet": round2(threebet),
        "rho_fold_equity": round2(spearman(&fold, &eqs)),
        "aa_fold": round2(pr(aa, FOLD)),
    });

    // --- flop: BTN opens to 5, BB calls and leads 5 into 10 ---
    let mut st = fresh(&mut rng);
    st.apply(Action::to(ActionType::Raise, 5));
    st.apply(Action::new(ActionType::Call));
    st.advance_street();
    st.apply(Action::to(ActionType::Bet, 5));
    let seat = st.to_act as usize;
    let (mut xs, mut masks, mut feqs, mut kind) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..300 {
        let mut d = full_deck();
        d.shuffle(&mut rng);
        let flop = [d[0], d[1], d[2]];
        for h in 0..24 {
            let hole = [d[3 + 2 * h], d[4 + 2 * h]];
            let mut s2 = st.clone();
            s2.community = flop.to_vec();
            s2.players[seat].hole = hole.to_vec();
            let mut x = vec![0f32; INPUT_SIZE];
            let sp = encode_into(&s2, &mut x).expect("decision");
            xs.extend_from_slice(&x);
            masks.extend_from_slice(&sp.mask);
            let mut rest: Vec<Card> = (0..52u8).filter(|c| !hole.contains(c) && !flop.contains(c)).collect();
            let mut tot = 0.0;
            for _ in 0..100 {
                let (pick, _) = rest.partial_shuffle(&mut rng, 2);
                let opp = [pick[0], pick[1]];
                tot += equity(&hole, &[&opp], &flop, &[], 1, &mut rng);
            }
            feqs.push(tot / 100.0);
            let br: Vec<u8> = flop.iter().map(|&c| rank(c)).collect();
            let top = *br.iter().max().unwrap();
            let pocket = rank(hole[0]) == rank(hole[1]);
            let paired = hole.iter().any(|&c| br.contains(&rank(c)));
            kind.push(if pocket && br.contains(&rank(hole[0])) {
                0 // set
            } else if !pocket && hole.iter().any(|&c| rank(c) == top) {
                1 // top pair
            } else if !pocket && !paired {
                2 // unpaired
            } else {
                3
            });
        }
    }
    let (p, _) = policy(net, &xs, &masks, dev);
    let fold: Vec<f64> = (0..feqs.len()).map(|i| p[i * NUM_TYPES + FOLD] as f64).collect();
    let rate = |sel: &dyn Fn(usize) -> bool| {
        let v: Vec<f64> = (0..fold.len()).filter(|&i| sel(i)).map(|i| fold[i]).collect();
        round2(v.iter().sum::<f64>() / v.len().max(1) as f64)
    };
    out["flop"] = json!({
        "rho_fold_equity": round2(spearman(&fold, &feqs)),
        "fold_set": rate(&|i| kind[i] == 0),
        "fold_top_pair": rate(&|i| kind[i] == 1),
        "fold_air": rate(&|i| kind[i] == 2 && feqs[i] < 0.35),
    });
    out
}
