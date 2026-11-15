//! Search-free self-play: both seats play the current network.
//!
//! `n_slots` hands are in flight. Each step:
//!   1. (parallel, CPU) every slot advances to its next decision — finishing
//!      hands and dealing new ones as needed — and encodes it;
//!   2. (GPU) one batched forward over the slots waiting on a decision;
//!   3. (parallel, CPU) each of those slots samples from the policy at
//!      temperature 1 (masked softmax over the action types, then a Beta band
//!      position for a bet/raise), records the decision, and applies it.
//! When a hand ends, each seat's chip result (with the board's run-out
//! replaced by its expected value when no decisions are left) / 200 becomes
//! `z` for all of that seat's decisions in the hand.
//!
//! A cohort is a fixed number of hands, all played to completion (as in
//! fox-lite): each slot plays its share and then idles, so nothing is
//! discarded and long hands aren't under-represented.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use tch::{Device, Kind, Tensor};

use hu_core::actions::{beta_logpdf, sample_u, to_action, ActionSpace, NUM_TYPES, RAISE};
use hu_core::allin_ev::{expected_stacks, is_runout};
use hu_core::encode::{encode_into, INPUT_SIZE};
use hu_core::rules::{Phase, State, SEATS};
use hu_core::table::deal_training_hand;

use crate::hud::Hud;
use crate::net::HuNet;

/// Chips per unit of reward (the 100bb starting stack).
pub const Z_SCALE: f64 = 200.0;

/// One recorded decision.
pub struct Row {
    pub x: Vec<f32>,
    pub mask: [f32; NUM_TYPES],
    pub ty: u8,
    /// Band position of a bet/raise (0 otherwise).
    pub u: f32,
    /// Whether the size is part of the action (a bet/raise with a band wider
    /// than one amount): its Beta log-density counts in log pi.
    pub sized: bool,
    /// Behavior log-prob of (type, u) under the sampling policy.
    pub logp: f32,
    pub z: f32,
}

/// Masked softmax over the action types (in f64): log-probs, -inf if illegal.
pub fn type_logprobs(mask: &[f32; NUM_TYPES], logits: &[f32]) -> [f64; NUM_TYPES] {
    let mut lp = [f64::NEG_INFINITY; NUM_TYPES];
    let mx = (0..NUM_TYPES).filter(|&t| mask[t] > 0.0).map(|t| logits[t] as f64).fold(f64::NEG_INFINITY, f64::max);
    let z: f64 = (0..NUM_TYPES).filter(|&t| mask[t] > 0.0).map(|t| (logits[t] as f64 - mx).exp()).sum();
    for t in 0..NUM_TYPES {
        if mask[t] > 0.0 {
            lp[t] = logits[t] as f64 - mx - z.ln();
        }
    }
    lp
}

/// Sample an action from policy outputs: (type, u, sized, log-prob).
pub fn sample_policy(sp: &ActionSpace, logits: &[f32], ab: &[f32], rng: &mut impl Rng) -> (usize, f64, bool, f64) {
    let lp = type_logprobs(&sp.mask, logits);
    let r: f64 = rng.gen();
    let mut acc = 0.0;
    let mut ty = (0..NUM_TYPES).rev().find(|&t| sp.mask[t] > 0.0).expect("a legal type");
    for t in 0..NUM_TYPES {
        if sp.mask[t] > 0.0 {
            acc += lp[t].exp();
            if r < acc {
                ty = t;
                break;
            }
        }
    }
    let mut logp = lp[ty];
    let sized = ty == RAISE && sp.s_hi > sp.s_lo;
    let mut u = 0.0;
    if sized {
        let (a, b) = (ab[0] as f64, ab[1] as f64);
        u = sample_u(a, b, rng);
        logp += beta_logpdf(u, a, b);
    }
    (ty, u, sized, logp)
}

struct Slot {
    st: State,
    rng: StdRng,
    /// Hands this slot still has to start (after the current one).
    quota: usize,
    /// Whether the slot has a hand in progress.
    live: bool,
    start: [i32; SEATS],
    pending: Vec<(usize, Row)>,
    done: Vec<Row>,
    sp: Option<ActionSpace>,
    hud: Hud,
}

impl Slot {
    fn new(seed: u64, hands: usize) -> Slot {
        let mut rng = StdRng::seed_from_u64(seed);
        let st = deal_training_hand(&mut rng);
        let mut s = Slot {
            st,
            rng,
            quota: hands.saturating_sub(1),
            live: hands > 0,
            start: [0; SEATS],
            pending: Vec::new(),
            done: Vec::new(),
            sp: None,
            hud: Hud::default(),
        };
        s.mark_start();
        s
    }

    fn mark_start(&mut self) {
        for (i, p) in self.st.players.iter().enumerate() {
            self.start[i] = p.stack + p.commit; // stacks before the blinds
        }
    }

    fn finish(&mut self, final_stacks: [f64; SEATS]) {
        // Chips are conserved (all-in EV included): the seats' z sum to 0.
        let before: i32 = self.start.iter().sum();
        let after: f64 = final_stacks.iter().sum();
        assert!((after - before as f64).abs() < 1e-6, "chips not conserved: {before} -> {after}");
        self.hud.record_hand(&self.st);
        for (seat, mut row) in self.pending.drain(..) {
            row.z = ((final_stacks[seat] - self.start[seat] as f64) / Z_SCALE) as f32;
            self.done.push(row);
        }
        if self.quota > 0 {
            self.quota -= 1;
            self.st = deal_training_hand(&mut self.rng);
            self.mark_start();
        } else {
            self.live = false;
        }
    }

    /// Advance to the next decision and encode it into `x`. Returns false if
    /// the slot has played all its hands.
    fn advance_and_encode(&mut self, x: &mut [f32]) -> bool {
        while self.live {
            match self.st.phase {
                Phase::Betting => {
                    self.sp = Some(encode_into(&self.st, x).expect("decision"));
                    return true;
                }
                Phase::StreetComplete => {
                    if is_runout(&self.st) {
                        let ev = expected_stacks(&self.st, &mut self.rng);
                        self.finish(ev);
                    } else {
                        self.st.advance_street();
                    }
                }
                Phase::HandOver | Phase::MatchOver => {
                    let f = std::array::from_fn(|i| self.st.players[i].stack as f64);
                    self.finish(f);
                }
                Phase::Waiting => unreachable!("training hands are dealt on creation"),
            }
        }
        false
    }

    /// Sample from the policy outputs, record the decision, apply it.
    fn act(&mut self, x: &[f32], logits: &[f32], ab: &[f32]) {
        let sp = self.sp.take().expect("pending decision");
        let (ty, u, sized, logp) = sample_policy(&sp, logits, ab, &mut self.rng);
        assert!(sp.mask[ty] > 0.0, "sampled an illegal action type {ty}");
        let action = to_action(&sp, ty, u);
        self.hud.record_decision(ty, &sp, action.amount);
        let seat = self.st.to_act as usize;
        self.pending.push((
            seat,
            Row { x: x.to_vec(), mask: sp.mask, ty: ty as u8, u: u as f32, sized, logp: logp as f32, z: 0.0 },
        ));
        self.st.apply(action);
    }
}

pub struct Cohort {
    pub rows: Vec<Row>,
    pub hud: Hud,
    pub hands: usize,
    pub forwards: usize,
}

/// Play `hands` self-play hands with `net` (`n_slots` in flight), all to
/// completion.
pub fn run_cohort(net: &HuNet, dev: Device, n_slots: usize, hands: usize, seed: u64) -> Cohort {
    let n_slots = n_slots.min(hands).max(1);
    let mut slots: Vec<Slot> = (0..n_slots)
        .into_par_iter()
        .map(|i| {
            let share = hands / n_slots + (i < hands % n_slots) as usize;
            Slot::new(seed.wrapping_mul(1_000_003).wrapping_add(i as u64), share)
        })
        .collect();
    let mut buf = vec![0f32; n_slots * INPUT_SIZE];
    let mut forwards = 0;
    loop {
        let waiting: Vec<bool> = slots
            .par_iter_mut()
            .zip(buf.par_chunks_mut(INPUT_SIZE))
            .map(|(slot, x)| slot.advance_and_encode(x))
            .collect();
        let idx: Vec<usize> = (0..n_slots).filter(|&i| waiting[i]).collect();
        if idx.is_empty() {
            break;
        }
        // Gather the waiting slots' encodings into one batch.
        let mut xs = Vec::with_capacity(idx.len() * INPUT_SIZE);
        for &i in &idx {
            xs.extend_from_slice(&buf[i * INPUT_SIZE..(i + 1) * INPUT_SIZE]);
        }
        let xt = Tensor::from_slice(&xs).view([idx.len() as i64, INPUT_SIZE as i64]).to_device(dev);
        let (tl, ab) = tch::no_grad(|| net.policy(&xt));
        let to_vec = |t: &Tensor| {
            Vec::<f32>::try_from(&t.to_kind(Kind::Float).to_device(Device::Cpu).contiguous().view([-1])).unwrap()
        };
        let logits = to_vec(&tl);
        let ab = to_vec(&ab);
        forwards += 1;
        let mut pos = vec![usize::MAX; n_slots];
        for (k, &i) in idx.iter().enumerate() {
            pos[i] = k;
        }
        slots.par_iter_mut().enumerate().for_each(|(i, slot)| {
            let k = pos[i];
            if k != usize::MAX {
                let x = &xs[k * INPUT_SIZE..(k + 1) * INPUT_SIZE];
                slot.act(x, &logits[k * NUM_TYPES..(k + 1) * NUM_TYPES], &ab[k * 2..k * 2 + 2]);
            }
        });
    }
    let mut hud = Hud::default();
    let mut rows = Vec::new();
    for s in slots {
        hud.merge(&s.hud);
        rows.extend(s.done);
    }
    Cohort { rows, hands: hud.hands as usize, hud, forwards }
}
