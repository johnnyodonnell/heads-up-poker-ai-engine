//! How well the table's estimate of an opponent's range predicts the hand
//! the opponent really holds, in self-play (the bot's model of its opponent
//! is the table itself):
//!
//!   hu rangecheck --run v1c [--hands 10000] [--opp-run v1a] [--stack 400] [live flags]
//!
//! Seat 0 plays `--run`'s table, seat 1 the same table or `--opp-run`'s, at
//! `--stack` chips each (400 = 200bb). At the start of the flop, turn and
//! river, seat 0's estimate of seat 1's range (`Engine::table_ranges`) is
//! scored against seat 1's real hand. Against itself the estimate should be
//! calibrated (PIT uniform, about 10% below 0.1); a run whose play differs
//! from the model shows up as an overconfident estimate, as Slumbot did.
//! The live flags' range hedges (`--opp-floor`, `--opp-temper`,
//! `--opp-sizes`) apply, so `--opp-floor 0 --opp-temper 1 --opp-sizes off`
//! scores the table's raw estimate.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use hu_core::cards::Card;
use hu_core::rules::{Phase, State, Status, Street};

use crate::play::{live_flags, load_engine};
use crate::util::{flag, log, num};

/// How well an estimated opponent range predicted the hand the opponent
/// really held.
#[derive(Clone, Copy, Default)]
struct Calib {
    /// The real hand's probability × the number of possible hands
    /// (1 = an average hand).
    ratio: f64,
    /// Range mass below the real hand's probability (ties half): uniform on
    /// [0, 1] when the range is calibrated.
    pit: f64,
    /// The range's own probability that the real hand's ratio falls below
    /// each of `THRESHOLDS`, and whether it did.
    expect: [f64; 3],
    below: [bool; 3],
    /// ln p(real hand) minus the range's expected ln p (0 on average when
    /// calibrated; negative when the range is overconfident).
    surprise: f64,
    /// 1 / Σp²: the range's effective number of hands.
    ess: f64,
    /// ln(p(real hand) × N): what the range knew about the real hand, in
    /// nats over "any hand" (a proper score: higher is better).
    score: f64,
}

const THRESHOLDS: [f64; 3] = [0.01, 0.1, 0.3];

fn calib(range: &[f32], dead: &[Card], real: [Card; 2]) -> Option<Calib> {
    let hs = hu_cfr::live::showdown::holes();
    let ok = |h: usize| !dead.contains(&hs[h][0]) && !dead.contains(&hs[h][1]);
    let total: f64 = (0..range.len()).filter(|&h| ok(h)).map(|h| range[h] as f64).sum();
    if total <= 0.0 {
        return None;
    }
    let p: Vec<f64> = (0..range.len()).map(|h| if ok(h) { range[h] as f64 / total } else { -1.0 }).collect();
    let n = p.iter().filter(|&&x| x >= 0.0).count() as f64;
    let a = hu_cfr::live::showdown::hand(real);
    let pa = p[a].max(0.0);
    let mut c = Calib { ratio: pa * n, ..Calib::default() };
    let mut expected_ln = 0.0;
    let mut sq = 0.0;
    for &x in p.iter().filter(|&&x| x >= 0.0) {
        if x < pa {
            c.pit += x;
        } else if x == pa {
            c.pit += 0.5 * x;
        }
        for (t, &thr) in THRESHOLDS.iter().enumerate() {
            if x * n < thr {
                c.expect[t] += x;
            }
        }
        if x > 0.0 {
            expected_ln += x * x.ln();
        }
        sq += x * x;
    }
    for (t, &thr) in THRESHOLDS.iter().enumerate() {
        c.below[t] = c.ratio < thr;
    }
    c.surprise = pa.max(1e-9).ln() - expected_ln;
    c.score = (pa * n).max(1e-9).ln();
    c.ess = 1.0 / sq;
    Some(c)
}

#[derive(Default)]
struct CalibSum {
    n: usize,
    expect: [f64; 3],
    below: [usize; 3],
    pit: f64,
    pit_low: usize,
    surprise: f64,
    ess: f64,
    score: f64,
}

impl CalibSum {
    fn add(&mut self, c: &Calib) {
        self.n += 1;
        for t in 0..3 {
            self.expect[t] += c.expect[t];
            self.below[t] += usize::from(c.below[t]);
        }
        self.pit += c.pit;
        self.pit_low += usize::from(c.pit < 0.1);
        self.surprise += c.surprise;
        self.ess += c.ess;
        self.score += c.score;
    }
    fn show(&self, name: &str) -> String {
        let n = self.n.max(1) as f64;
        let mut s = format!("{name:44} n={:5}  ", self.n);
        for (t, thr) in THRESHOLDS.iter().enumerate() {
            s += &format!("p<{thr}/N: {:4.1}% (model {:4.1}%)  ", self.below[t] as f64 / n * 100.0, self.expect[t] / n * 100.0);
        }
        s += &format!(
            "PIT mean {:.2}, <0.1: {:4.1}%  surprise {:+.2}  range ~{:.0} hands  score {:+.3}",
            self.pit / n,
            self.pit_low as f64 / n * 100.0,
            self.surprise / n,
            self.ess / n,
            self.score / n
        );
        s
    }
}

/// `hu rangecheck`: see the module docs.
pub fn rangecheck_cmd(args: &[String]) {
    let run = flag(args, "--run").expect("--run NAME");
    let n: usize = num(args, "--hands", 10_000);
    let stack: i32 = num(args, "--stack", 400);
    let mut engine = load_engine(&run);
    live_flags(args, &mut engine);
    let other = flag(args, "--opp-run").map(|r| load_engine(&r));
    let mut sums: Vec<CalibSum> = (0..4).map(|_| CalibSum::default()).collect();
    let mut rng = SmallRng::seed_from_u64(7);
    for i in 0..n {
        let mut deck: Vec<Card> = (0..52).collect();
        rand::seq::SliceRandom::shuffle(&mut deck[..], &mut rng);
        let mut s = State::new_table([stack; 2]);
        s.button = rng.gen_range(0..2);
        s.start_hand(deck);
        let seed: u64 = rng.gen();
        loop {
            match s.phase {
                Phase::Betting => {
                    let seat = s.to_act as usize;
                    let v = hu_cfr::engine::view_of(&s, seat, [stack; 2], seed);
                    let who = if seat == 1 { other.as_ref().unwrap_or(&engine) } else { &engine };
                    let d = who.decide_mode(&v, &mut rng, hu_cfr::live::LiveMode::Off).expect("decide");
                    s.apply(d.action);
                }
                Phase::StreetComplete if !hu_core::allin_ev::is_runout(&s) && s.players.iter().all(|p| p.status != Status::Folded) => s.advance_street(),
                _ => break,
            }
        }
        let view = hu_cfr::engine::view_of(&s, 0, [stack; 2], seed);
        let opp = [s.players[1].hole[0], s.players[1].hole[1]];
        for st in [Street::Flop, Street::Turn, Street::River] {
            let nb = [0, 3, 4, 5][st.index()];
            if s.community.len() < nb {
                break;
            }
            let Some(k) = s.actions.iter().position(|a| a.street == st) else { break };
            let r = engine.table_ranges(&view, &s.actions, k);
            let mut dead = vec![view.hole[0], view.hole[1]];
            dead.extend_from_slice(&s.community[..nb]);
            if let Some(c) = calib(&r[1], &dead, opp) {
                sums[st.index()].add(&c);
            }
        }
        if i % 1000 == 999 {
            log(format!("{} self-play hands", i + 1));
        }
    }
    let who = flag(args, "--opp-run").unwrap_or(run);
    for st in [Street::Flop, Street::Turn, Street::River] {
        println!("{}", sums[st.index()].show(&format!("{st:?} start (estimate of {who})")));
    }
}
