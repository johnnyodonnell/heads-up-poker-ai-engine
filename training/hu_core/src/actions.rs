//! The policy's action space — mirror of `src/engine/actions.ts` (the source of
//! truth; see that file for the design). Four action types — fold,
//! check/call, bet/raise, all-in — and for a bet/raise a band position
//! u in [0, 1] (drawn from the network's Beta), mapped onto the legal raises on
//! a log pot-fraction scale: s = sLo + u (sHi - sLo), s = ln(increment / pot
//! after calling).

use rand::Rng;
use rand_distr::{Beta, Distribution};

use crate::rules::{Action, ActionType, Legal, State};

pub const FOLD: usize = 0;
pub const CALL: usize = 1; // check or call
pub const RAISE: usize = 2; // bet or raise, sized by u
pub const ALLIN: usize = 3;
pub const NUM_TYPES: usize = 4;

/// u is kept strictly inside (0, 1) so its Beta log-density stays finite.
pub const U_EPS: f64 = 1e-6;

#[derive(Clone, Debug)]
pub struct ActionSpace {
    pub legal: Legal,
    /// 1 = allowed, per action type.
    pub mask: [f32; NUM_TYPES],
    pub current_bet: i32,
    /// Gathered pot + street bets + the mover's call.
    pub pot_after_call: i32,
    /// Smallest legal non-all-in raise-to (0 if none).
    pub min_raise_to: i32,
    /// Largest legal non-all-in raise-to: all-in minus one (0 if none).
    pub max_raise_to: i32,
    pub s_lo: f64,
    pub s_hi: f64,
}

/// TS `actionSpace`.
pub fn action_space(st: &State) -> Option<ActionSpace> {
    let legal = st.legal()?;
    let p = &st.players[st.to_act as usize];
    let pot_total = st.pot + st.players.iter().map(|q| q.bet).sum::<i32>();
    let pot_after_call = pot_total + (st.current_bet - p.bet);
    let sized = legal.can_raise && legal.min_raise_to < legal.max_raise_to;
    let min_to = legal.min_raise_to;
    let max_to = legal.max_raise_to - 1;
    let s = |to: i32| ((to - st.current_bet) as f64 / pot_after_call as f64).ln();
    Some(ActionSpace {
        mask: [legal.can_fold as u8 as f32, 1.0, sized as u8 as f32, legal.can_raise as u8 as f32],
        current_bet: st.current_bet,
        pot_after_call,
        min_raise_to: if sized { min_to } else { 0 },
        max_raise_to: if sized { max_to } else { 0 },
        s_lo: if sized { s(min_to) } else { 0.0 },
        s_hi: if sized { s(max_to) } else { 0.0 },
        legal,
    })
}

/// Raise-to for band position u (rounded to a chip, clamped to the band).
/// TS `raiseToFor`.
pub fn raise_to_for(sp: &ActionSpace, u: f64) -> i32 {
    let s = sp.s_lo + u * (sp.s_hi - sp.s_lo);
    let to = (sp.current_bet as f64 + s.exp() * sp.pot_after_call as f64).round() as i32;
    to.max(sp.min_raise_to).min(sp.max_raise_to)
}

/// The engine action for an action type (+ u for RAISE). TS `toEngineAction`.
pub fn to_action(sp: &ActionSpace, ty: usize, u: f64) -> Action {
    match ty {
        FOLD => Action::new(ActionType::Fold),
        CALL => Action::new(if sp.legal.can_check { ActionType::Check } else { ActionType::Call }),
        RAISE => Action::to(
            if sp.current_bet == 0 { ActionType::Bet } else { ActionType::Raise },
            raise_to_for(sp, u),
        ),
        _ => Action::new(ActionType::AllIn),
    }
}

/// A Beta(a, b) band position, kept strictly inside (0, 1).
pub fn sample_u(a: f64, b: f64, rng: &mut impl Rng) -> f64 {
    let u = Beta::new(a, b).expect("beta params").sample(rng);
    u.clamp(U_EPS, 1.0 - U_EPS)
}

/// ln Beta(u; a, b) density.
pub fn beta_logpdf(u: f64, a: f64, b: f64) -> f64 {
    (a - 1.0) * u.ln() + (b - 1.0) * (1.0 - u).ln() - ln_beta(a, b)
}

fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// ln Γ(x) for x > 0 (Lanczos, g = 7, n = 9; |rel err| ~1e-15).
pub fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = G[0];
    let t = x + 7.5;
    for (i, &g) in G.iter().enumerate().skip(1) {
        a += g / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn ln_gamma_known_values() {
        assert!((ln_gamma(1.0)).abs() < 1e-12);
        assert!((ln_gamma(5.0) - 24f64.ln()).abs() < 1e-12);
        assert!((ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-12);
    }

    #[test]
    fn beta_density_integrates_to_one() {
        for (a, b) in [(1.0, 1.0), (2.0, 5.0), (1.3, 1.1), (8.0, 3.0)] {
            let n = 200_000;
            let sum: f64 = (0..n).map(|k| beta_logpdf((k as f64 + 0.5) / n as f64, a, b).exp()).sum();
            assert!((sum / n as f64 - 1.0).abs() < 1e-3, "Beta({a},{b}) integral {}", sum / n as f64);
        }
    }

    #[test]
    fn samples_match_the_mean() {
        let mut rng = StdRng::seed_from_u64(1);
        let (a, b) = (2.0, 6.0);
        let n = 100_000;
        let mean: f64 = (0..n).map(|_| sample_u(a, b, &mut rng)).sum::<f64>() / n as f64;
        assert!((mean - a / (a + b)).abs() < 0.005, "mean {mean}");
    }
}
