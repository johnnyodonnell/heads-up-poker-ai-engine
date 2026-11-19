//! `hu scorecheck`: checks of advantage scoring (`hu_cfr::score`) on real
//! spots of table-vs-table play, before trusting it with test results.
//!
//!   hu scorecheck --run NAME [--opp-run OTHER] [--spots 200] [--playouts 0]
//!       [--stack 400] [--seed 3] [--verbose]
//!
//! On turn and river decisions of `--run` (seat 0) against `--opp-run`:
//!
//! 1. `table_play_all` (what scoring walks) gives every hand exactly what
//!    `table_play` gives it (what `decide` samples from).
//! 2. Each action's value Q from the scored walk of the game tree equals an
//!    independent exact walk that shares none of the scorer's code: every
//!    decision through `decide_mode` (the served path), every undealt card
//!    dealt, every hand settled by the rules engine. Optionally
//!    (`--playouts N`) also Monte Carlo playouts.
//! 3. The opponent's exact range: each hand's likelihood equals the product
//!    of `decide_mode`'s probabilities of the opponent's real actions with
//!    that hand, and the range is calibrated: where the opponent's real hand
//!    ranks in it (by strength on the board) should be uniform, 10% of spots
//!    in each tenth.
//! 4. How long scoring one decision takes, river and turn.

use std::time::Instant;

use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use hu_cfr::engine::{view_of, Engine};
use hu_cfr::live::showdown::{hand, holes, H};
use hu_cfr::live::LiveMode;
use hu_cfr::score::{advantage, Scorer};
use hu_core::allin_ev::{expected_stacks, is_runout};
use hu_core::cards::Card;
use hu_core::hand_eval::eval7;
use hu_core::rules::{Action, Phase, State, Street};

use crate::play::load_engine;
use crate::util::{flag, log, num};

struct Spot {
    s: State,
    init: State,
    seed: u64,
}

/// Seat 0's turn and river decisions in table-vs-table hands, one per hand.
fn spots(me: &Engine, opp: &Engine, n: usize, stack: i32, seed: u64) -> Vec<Spot> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut out = Vec::new();
    while out.len() < n {
        let mut deck: Vec<u8> = (0..52).collect();
        deck.shuffle(&mut rng);
        let mut s = State::new_table([stack; 2]);
        s.button = rng.gen_range(0..2);
        s.start_hand(deck);
        let init = s.clone();
        let hseed = rng.gen::<u64>();
        loop {
            match s.phase {
                Phase::Betting => {
                    let seat = s.to_act as usize;
                    if seat == 0 && s.street >= Street::Turn && rng.gen_bool(0.5) {
                        out.push(Spot { s: s.clone(), init: init.clone(), seed: hseed });
                        break;
                    }
                    let e = if seat == 0 { me } else { opp };
                    let d = e.decide_mode(&view_of(&s, seat, [stack; 2], hseed), &mut rng, LiveMode::Off).expect("decide");
                    s.apply(d.action);
                }
                Phase::StreetComplete => {
                    if is_runout(&s) {
                        break;
                    }
                    s.advance_street();
                }
                _ => break,
            }
        }
    }
    out
}

/// Seat 0's chips won playing `s` out with both tables, future cards shuffled.
fn playout(me: &Engine, opp: &Engine, s: &State, stack: i32, seed: u64, rng: &mut SmallRng) -> f64 {
    let mut t = s.clone();
    t.deck.shuffle(rng);
    loop {
        match t.phase {
            Phase::Betting => {
                let seat = t.to_act as usize;
                let e = if seat == 0 { me } else { opp };
                let d = e.decide_mode(&view_of(&t, seat, [stack; 2], seed), rng, LiveMode::Off).expect("decide");
                t.apply(d.action);
            }
            Phase::StreetComplete => {
                if is_runout(&t) {
                    return expected_stacks(&t, rng)[0] - stack as f64;
                }
                t.advance_street();
            }
            _ => return (t.players[0].stack - stack) as f64,
        }
    }
}

/// Seat 0's expected chips from `s`, both tables playing, computed without
/// any of the scorer's code: every decision through `decide_mode` (the
/// served path, replaying the hand from the bot's view), every undealt card
/// dealt from the real deck, every hand settled by the rules engine.
fn exact(me: &Engine, opp: &Engine, s: &State, stack: i32, seed: u64) -> f64 {
    match s.phase {
        Phase::Betting => {
            let seat = s.to_act as usize;
            let e = if seat == 0 { me } else { opp };
            let mut rng = SmallRng::seed_from_u64(0);
            let d = e.decide_mode(&view_of(s, seat, [stack; 2], seed), &mut rng, LiveMode::Off).expect("decide");
            let total: f64 = d.dist.iter().map(|x| x.1).sum();
            d.dist
                .iter()
                .filter(|x| x.1 > 0.0)
                .map(|&(a, p)| {
                    let mut t = s.clone();
                    t.apply(a);
                    p / total * exact(me, opp, &t, stack, seed)
                })
                .sum()
        }
        Phase::StreetComplete => {
            if is_runout(s) {
                let mut rng = SmallRng::seed_from_u64(0);
                return expected_stacks(s, &mut rng)[0] - stack as f64;
            }
            let folded = s.players.iter().any(|p| p.status == hu_core::rules::Status::Folded);
            if folded || s.street == Street::River {
                let mut t = s.clone();
                t.advance_street();
                return exact(me, opp, &t, stack, seed);
            }
            let cards = s.deck.clone();
            cards
                .iter()
                .map(|&c| {
                    let mut t = s.clone();
                    t.deck.retain(|&x| x != c);
                    t.deck.push(c);
                    t.advance_street();
                    exact(me, opp, &t, stack, seed)
                })
                .sum::<f64>()
                / cards.len() as f64
        }
        _ => (s.players[0].stack - stack) as f64,
    }
}

fn strength(hole: [Card; 2], board: &[Card]) -> u32 {
    let mut c = hole.to_vec();
    c.extend_from_slice(board);
    eval7(&c)
}

pub fn scorecheck_cmd(args: &[String]) {
    let run = flag(args, "--run").expect("--run NAME");
    let opp_run = flag(args, "--opp-run").unwrap_or_else(|| run.clone());
    let n: usize = num(args, "--spots", 200);
    let playouts: usize = num(args, "--playouts", 0);
    let stack: i32 = num(args, "--stack", 400);
    let seed: u64 = num(args, "--seed", 3);
    let verbose = args.iter().any(|a| a == "--verbose");
    let me = load_engine(&run);
    let other = if opp_run == run { None } else { Some(load_engine(&opp_run)) };
    let opp = other.as_ref().unwrap_or(&me);
    let t0 = Instant::now();
    let spots = spots(&me, opp, n, stack, seed);
    log(format!("{} spots ({} river) in {:.1}s", spots.len(), spots.iter().filter(|x| x.s.street == Street::River).count(), t0.elapsed().as_secs_f64()));

    struct Out {
        play_diff: f64,
        zs: Vec<f64>,
        diffs: Vec<f64>,
        like_diff: f64,
        pit: Option<f64>,
        ms: f64,
        river: bool,
    }
    let outs: Vec<Out> = spots
        .par_iter()
        .enumerate()
        .map(|(i, sp)| {
            let s = &sp.s;
            let button = s.button as usize;
            let stacks = [stack; 2];
            let hs = holes();
            // 1. table_play_all vs table_play, every hand.
            let all: Vec<f64> = hs.iter().map(|&[a, b]| if s.community.contains(&a) || s.community.contains(&b) { 0.0 } else { 1.0 }).collect();
            let (opts, p) = me.table_play_all(s, stacks, button, sp.seed, &all);
            let mut play_diff = 0f64;
            for h in (0..H).filter(|&h| all[h] > 0.0) {
                let tp = me.table_play(s, hs[h], stacks, button, sp.seed).expect("table_play");
                for (o, a) in opts.iter().enumerate() {
                    let x = tp.dist.iter().filter(|d| d.0 == *a).map(|d| d.1).sum::<f64>();
                    play_diff = play_diff.max((x - p[o * H + h]).abs());
                }
                for d in &tp.dist {
                    if !opts.contains(&d.0) && d.1 > 0.0 {
                        play_diff = play_diff.max(d.1);
                    }
                }
            }
            // 2. Q of each of the table's options against playouts.
            let hole = [s.players[0].hole[0], s.players[0].hole[1]];
            let opp_hole = [s.players[1].hole[0], s.players[1].hole[1]];
            let sc = Scorer { me: &me, opp, seat: 0, hole, stacks, button, seed: sp.seed };
            let mut real = vec![0f64; H];
            real[hand(opp_hole)] = 1.0;
            let mut rng = SmallRng::seed_from_u64(seed ^ (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut zs = Vec::new();
            let mut diffs = Vec::new();
            let mut seen: Vec<Action> = Vec::new();
            for &a in &opts {
                if seen.contains(&a) {
                    continue;
                }
                seen.push(a);
                let mut t = s.clone();
                t.apply(a);
                let q = sc.value(&t, &real);
                let x = exact(&me, opp, &t, stack, sp.seed);
                diffs.push((q - x).abs() / x.abs().max(1.0));
                if (q - x).abs() > 1e-6 * x.abs().max(1.0) && verbose {
                    let hist: Vec<String> = t.actions.iter().map(|x| format!("{}:{:?}{}{}", x.seat, x.kind, if x.allin { "*" } else { "" }, x.to)).collect();
                    log(format!(
                        "  mismatch: {:?} board {} holes {} | {}: Q {q:.6} exact {x:.6}\n    {}",
                        s.street,
                        hu_cfr::engine::cards_str(&s.community),
                        hu_cfr::engine::cards_str(&hole),
                        hu_cfr::engine::cards_str(&opp_hole),
                        hist.join(" ")
                    ));
                }
                if playouts > 0 {
                    let xs: Vec<f64> = (0..playouts).map(|_| playout(&me, opp, &t, stack, sp.seed, &mut rng)).collect();
                    let m = xs.iter().sum::<f64>() / xs.len() as f64;
                    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
                    let se = (var / xs.len() as f64).sqrt();
                    if se > 1e-9 {
                        zs.push((m - q) / se);
                    }
                }
            }
            // 3. The opponent's exact range: the rank of its real hand.
            let mut dead = vec![hole[0], hole[1]];
            dead.extend_from_slice(&s.community);
            let pit = opp.table_likelihood(&sp.init, &s.actions, 1, stacks, button, sp.seed, &dead).and_then(|r| {
                let total: f64 = r.iter().sum();
                if total <= 0.0 {
                    return None;
                }
                let mine = strength(opp_hole, &s.community);
                let (mut below, mut tie) = (0f64, 0f64);
                for h in 0..H {
                    if r[h] > 0.0 {
                        let x = strength(hs[h], &s.community);
                        if x < mine {
                            below += r[h];
                        } else if x == mine {
                            tie += r[h];
                        }
                    }
                }
                Some((below + rng.gen::<f64>() * tie) / total)
            });
            // 3b. The range's likelihoods, hand by hand, through decide_mode:
            // the probability of each of the opponent's real actions with
            // that hand in its seat, multiplied.
            let mut like_diff = 0f64;
            if let Some(r) = opp.table_likelihood(&sp.init, &s.actions, 1, stacks, button, sp.seed, &dead) {
                for h in (0..H).step_by(37) {
                    let [a, b] = hs[h];
                    if dead.contains(&a) || dead.contains(&b) {
                        continue;
                    }
                    let mut l = 1f64;
                    for (k, x) in s.actions.iter().enumerate().filter(|(_, x)| x.seat == 1) {
                        let n_board = [0, 3, 4, 5][x.street.index()];
                        let v = hu_cfr::engine::View {
                            bot: 1,
                            button,
                            stacks,
                            hole: [a, b],
                            board: s.community[..n_board].to_vec(),
                            actions: s.actions[..k].iter().map(|y| hu_cfr::engine::HistAction { seat: y.seat, kind: y.kind, to: y.to }).collect(),
                            seed: sp.seed,
                        };
                        let st = hu_cfr::engine::replay(&v).expect("replay");
                        let mut r0 = SmallRng::seed_from_u64(0);
                        let d = opp.decide_mode(&v, &mut r0, LiveMode::Off).expect("decide");
                        let total: f64 = d.dist.iter().map(|y| y.1).sum();
                        l *= d
                            .dist
                            .iter()
                            .filter(|(act, _)| {
                                let mut t = st.clone();
                                t.apply(*act);
                                let rec = t.actions.last().unwrap();
                                (rec.kind, rec.to) == (x.kind, x.to)
                            })
                            .map(|y| y.1)
                            .sum::<f64>()
                            / total;
                    }
                    like_diff = like_diff.max((l - r[h]).abs());
                }
            }
            // 4. Time to score one decision (against a uniform mix of the
            // table's options, so every option is walked).
            let k = opts.len() as f64;
            let dist: Vec<(Action, f64)> = opts.iter().map(|&a| (a, 1.0 / k)).collect();
            let t1 = Instant::now();
            let _ = advantage(&me, opp, s, &sp.init, 0, stacks, sp.seed, &dist);
            let ms = t1.elapsed().as_secs_f64() * 1000.0;
            Out { play_diff, zs, diffs, pit, like_diff, ms, river: s.street == Street::River }
        })
        .collect();

    let worst = outs.iter().map(|o| o.play_diff).fold(0f64, f64::max);
    log(format!("1. table_play_all vs table_play: largest difference in any hand's action probability {worst:.2e} (should be ~0)"));
    let diffs: Vec<f64> = outs.iter().flat_map(|o| o.diffs.iter().copied()).collect();
    let worst = diffs.iter().copied().fold(0f64, f64::max);
    let off = diffs.iter().filter(|&&d| d > 1e-6).count();
    log(format!("2. Q vs an independent exact walk (decide_mode + the rules engine's payouts): {} actions, largest relative difference {worst:.2e}, {off} above 1e-6 (should be 0)", diffs.len()));
    let zs: Vec<f64> = outs.iter().flat_map(|o| o.zs.iter().copied()).collect();
    if !zs.is_empty() {
        let mean = zs.iter().sum::<f64>() / zs.len() as f64;
        let rms = (zs.iter().map(|z| z * z).sum::<f64>() / zs.len() as f64).sqrt();
        log(format!("   and vs {playouts} playouts per action: z mean {mean:+.3} (~0), RMS {rms:.3} (~1; rare big branches make it heavy-tailed)"));
    }
    let pits: Vec<f64> = outs.iter().filter_map(|o| o.pit).collect();
    let tenths: Vec<String> = (0..10).map(|k| format!("{:.0}", pits.iter().filter(|&&p| p >= k as f64 / 10.0 && p < (k + 1) as f64 / 10.0).count() as f64 / pits.len().max(1) as f64 * 100.0)).collect();
    let worst = outs.iter().map(|o| o.like_diff).fold(0f64, f64::max);
    log(format!("3. opponent's exact range vs its likelihood hand by hand through decide_mode: largest difference {worst:.2e} (should be ~0)"));
    log(format!("   opponent's real hand in its exact range, % of {} spots per tenth (10 each): {}", pits.len(), tenths.join(" ")));
    for (name, river) in [("river", true), ("turn", false)] {
        let mut ms: Vec<f64> = outs.iter().filter(|o| o.river == river).map(|o| o.ms).collect();
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if !ms.is_empty() {
            log(format!(
                "4. scoring one {name} decision (every option): median {:.1} ms, 90th percentile {:.1} ms, max {:.1} ms ({} spots, threads busy)",
                ms[ms.len() / 2],
                ms[ms.len() * 9 / 10],
                ms[ms.len() - 1],
                ms.len()
            ));
        }
    }
}
