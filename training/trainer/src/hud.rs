//! "HUD" statistics of the self-play policy — a poker-native early warning for
//! the failure modes self-play REINFORCE can hit in poker: swinging between
//! nit and maniac (VPIP / PFR / 3-bet / fold rates moving in waves) and the
//! bet size collapsing to min-raises (size quantiles).
//!
//! Per position (BTN = the button / small blind, BB): VPIP (voluntarily put
//! chips in preflop) and PFR (raised preflop). Plus 3-bet (re-raised a single
//! preflop raise), c-bet (the preflop aggressor bets the flop when not bet
//! into), fold to c-bet, showdown rate, the action-type mix, and bet/raise
//! sizes as fractions of the pot after calling.

use hu_core::actions::{ActionSpace, ALLIN, CALL, FOLD, NUM_TYPES, RAISE};
use hu_core::rules::{ActionType, State, Status, Street};
use serde_json::{json, Value};

const MAX_SIZES: usize = 200_000;

#[derive(Clone, Default)]
pub struct Hud {
    pub hands: u64,
    /// [BTN, BB]
    pub seats: [u64; 2],
    pub vpip: [u64; 2],
    pub pfr: [u64; 2],
    pub threebet_opp: u64,
    pub threebet: u64,
    pub cbet_opp: u64,
    pub cbet: u64,
    pub fold_cbet_opp: u64,
    pub fold_cbet: u64,
    pub showdowns: u64,
    pub decisions: u64,
    pub types: [u64; NUM_TYPES],
    /// Bet/raise sizes as fractions of the pot after calling.
    pub sizes: Vec<f32>,
}

impl Hud {
    pub fn record_decision(&mut self, ty: usize, sp: &ActionSpace, amount: Option<i32>) {
        self.decisions += 1;
        self.types[ty] += 1;
        if let (RAISE, Some(to)) = (ty, amount) {
            if self.sizes.len() < MAX_SIZES {
                self.sizes.push((to - sp.current_bet) as f32 / sp.pot_after_call as f32);
            }
        }
    }

    /// Tally a finished hand from its action history.
    pub fn record_hand(&mut self, st: &State) {
        self.hands += 1;
        let pos = |seat: usize| if seat as i32 == st.button { 0 } else { 1 };
        let pre: Vec<_> = st.actions.iter().filter(|a| a.street == Street::Preflop).collect();
        for seat in 0..st.players.len() {
            let p = pos(seat);
            self.seats[p] += 1;
            let mine = pre.iter().filter(|a| a.seat == seat);
            let (mut v, mut r) = (false, false);
            for a in mine {
                v |= matches!(a.kind, ActionType::Call | ActionType::Bet | ActionType::Raise);
                r |= matches!(a.kind, ActionType::Bet | ActionType::Raise);
            }
            self.vpip[p] += v as u64;
            self.pfr[p] += r as u64;
        }
        // 3-bet: decisions facing exactly one preflop raise.
        let mut raises = 0;
        for a in &pre {
            if raises == 1 {
                self.threebet_opp += 1;
                self.threebet += (a.kind == ActionType::Raise) as u64;
            }
            if a.kind == ActionType::Raise {
                raises += 1;
            }
        }
        // C-bet: the last preflop raiser's first flop action, if not bet into;
        // then the opponent's response to that c-bet.
        if let Some(agg) = pre.iter().rev().find(|a| a.kind == ActionType::Raise).map(|a| a.seat) {
            let flop: Vec<_> = st.actions.iter().filter(|a| a.street == Street::Flop).collect();
            let mut bet_seen = false;
            for (k, a) in flop.iter().enumerate() {
                if a.seat == agg {
                    if !bet_seen {
                        self.cbet_opp += 1;
                        if a.kind == ActionType::Bet {
                            self.cbet += 1;
                            if let Some(resp) = flop.get(k + 1) {
                                self.fold_cbet_opp += 1;
                                self.fold_cbet += (resp.kind == ActionType::Fold) as u64;
                            }
                        }
                    }
                    break;
                }
                bet_seen |= matches!(a.kind, ActionType::Bet | ActionType::Raise);
            }
        }
        let in_hand = st.players.iter().filter(|p| matches!(p.status, Status::Active | Status::AllIn)).count();
        self.showdowns += (in_hand >= 2) as u64;
    }

    pub fn merge(&mut self, o: &Hud) {
        self.hands += o.hands;
        for p in 0..2 {
            self.seats[p] += o.seats[p];
            self.vpip[p] += o.vpip[p];
            self.pfr[p] += o.pfr[p];
        }
        self.threebet_opp += o.threebet_opp;
        self.threebet += o.threebet;
        self.cbet_opp += o.cbet_opp;
        self.cbet += o.cbet;
        self.fold_cbet_opp += o.fold_cbet_opp;
        self.fold_cbet += o.fold_cbet;
        self.showdowns += o.showdowns;
        self.decisions += o.decisions;
        for i in 0..NUM_TYPES {
            self.types[i] += o.types[i];
        }
        let room = MAX_SIZES.saturating_sub(self.sizes.len());
        self.sizes.extend(o.sizes.iter().take(room));
    }

    pub fn to_json(&self) -> Value {
        let pct = |a: u64, b: u64| if b == 0 { 0.0 } else { (1000.0 * a as f64 / b as f64).round() / 10.0 };
        let mut s = self.sizes.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| {
            if s.is_empty() {
                0.0
            } else {
                let v = s[((p * (s.len() - 1) as f64) as usize).min(s.len() - 1)] as f64;
                (v * 100.0).round() / 100.0
            }
        };
        json!({
            "vpip_btn": pct(self.vpip[0], self.seats[0]),
            "vpip_bb": pct(self.vpip[1], self.seats[1]),
            "pfr_btn": pct(self.pfr[0], self.seats[0]),
            "pfr_bb": pct(self.pfr[1], self.seats[1]),
            "3bet": pct(self.threebet, self.threebet_opp),
            "cbet": pct(self.cbet, self.cbet_opp),
            "fold_cbet": pct(self.fold_cbet, self.fold_cbet_opp),
            "wtsd": pct(self.showdowns, self.hands),
            "fold": pct(self.types[FOLD], self.decisions),
            "call": pct(self.types[CALL], self.decisions),
            "raise": pct(self.types[RAISE], self.decisions),
            "allin": pct(self.types[ALLIN], self.decisions),
            "size_q10_50_90": [q(0.1), q(0.5), q(0.9)],
        })
    }
}
