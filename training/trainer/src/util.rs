//! Small shared helpers: CLI flags and random test positions.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use hu_core::actions::{to_action, NUM_TYPES};
use hu_core::encode::encode;
use hu_core::rules::Phase;
use hu_core::table::deal_training_hand;

/// `--key value` lookup.
pub fn flag(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).cloned()
}

pub fn has_flag(args: &[String], key: &str) -> bool {
    args.iter().any(|a| a == key)
}

/// `--key value` parsed, or `default`.
pub fn num<T: std::str::FromStr>(args: &[String], key: &str, default: T) -> T
where
    T::Err: std::fmt::Debug,
{
    flag(args, key).map(|v| v.parse().unwrap_or_else(|e| panic!("{key} {v}: {e:?}"))).unwrap_or(default)
}

/// "30m" / "4h" / "90s" / "120" (seconds) -> seconds.
pub fn parse_duration(s: &str) -> f64 {
    let (num, mult) = match s.chars().last() {
        Some('h') => (&s[..s.len() - 1], 3600.0),
        Some('m') => (&s[..s.len() - 1], 60.0),
        Some('s') => (&s[..s.len() - 1], 1.0),
        _ => (s, 1.0),
    };
    num.parse::<f64>().unwrap_or_else(|_| panic!("bad duration {s:?}")) * mult
}

/// `n` encoded decision points from random play on sampled training hands
/// (uniform legal action type, uniform band position) — varied, realistic
/// inputs for parity fixtures and smoke tests.
pub fn random_positions(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let mut s = deal_training_hand(&mut rng);
        while !s.hand_done() && out.len() < n {
            match s.phase {
                Phase::Betting => {
                    let (x, sp) = encode(&s).expect("encode");
                    out.push(x);
                    let legal: Vec<usize> = (0..NUM_TYPES).filter(|&t| sp.mask[t] > 0.0).collect();
                    let ty = legal[rng.gen_range(0..legal.len())];
                    s.apply(to_action(&sp, ty, rng.gen::<f64>()));
                }
                _ => s.advance_street(),
            }
        }
    }
    out
}
