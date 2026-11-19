//! `hu turncheck`: how much better live solves play than the table, decision
//! by decision on real spots, for several live settings on the same spots
//! (paired comparisons, far fewer hands than matches need).
//!
//!   hu turncheck --run v1c --opp-run v1c,v1a [--stack 200] [--spots 500]
//!       [--street turn|river] [--play off|turn] [--seed 11]
//!       [--configs "default;iters=600;hedges=off"] [--save FILE]
//!
//! Each spot is the bot's first decision on `--street` (default the turn,
//! where live turn play first differs from live river play), the bot in
//! either seat, in a hand it played until then by its table (`--play off`)
//! or solving the turn live (`--play turn`: the river spots of turn mode).
//! A setting's advantage there is the scored term of `hu eval`:
//! Σ_a (σ_live(a) − σ_table(a)) · Q(a), Q being the bot's expected result if
//! both players play their tables from there, against the opponent's exact
//! range (`hu_cfr::score::advantage`). Its mean over spots is what the
//! setting's decisions there gain over the table's, the table playing on.
//!
//! Settings (`;` between settings, `,` within one): `default`, `mode=turn|
//! river` (the live mode the decision is solved in; default turn),
//! `iters=N` (turn iterations), `river_iters=N`, `menu=lean|medium|full`
//! (the turn's), `river=lean|raises|full` (the river inside turn solves),
//! `river_opp=table|solve` (turn mode's opponent range at the river),
//! `safe=off|table|best`,
//! `safe_weight=W|resolve`, `hedges=off` (trust the opponent model:
//! opp_floor, opp_mix and opp_line 0, opp_temper 1, opp_sizes off), and
//! `opp_floor=`, `opp_mix=`, `opp_line=`, `opp_temper=`, `opp_sizes=on|off`.

use std::io::Write;
use std::time::Instant;

use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use hu_cfr::engine::{view_of, Engine};
use hu_cfr::live::{LiveCfg, LiveMode, SafeMode};
use hu_cfr::menu::LiveMenu;
use hu_cfr::score::advantage;
use hu_core::allin_ev::is_runout;
use hu_core::rules::{Phase, State, Street};

use crate::play::{live_flags, load_engine, load_rival};
use crate::util::{flag, log, num};

struct Spot {
    s: State,
    init: State,
    seed: u64,
    bot: usize,
    /// Hands dealt to find this spot (spots per hand dealt = how often the
    /// bot makes a first turn decision).
    hands: u32,
}

/// The bot's first decision on `street`, one hand per index, the bot in
/// seat `i % 2` playing the streets before as `play` says.
#[allow(clippy::too_many_arguments)]
fn spots(me: &Engine, opp: &Engine, n: usize, stack: i32, seed: u64, street: Street, play: LiveMode) -> Vec<Spot> {
    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(seed ^ (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let bot = i % 2;
            let mut hands = 0;
            loop {
                hands += 1;
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
                            if seat == bot && s.street == street {
                                return Spot { s, init, seed: hseed, bot, hands };
                            }
                            let (e, m) = if seat == bot { (me, play) } else { (opp, LiveMode::Off) };
                            let d = e.decide_mode(&view_of(&s, seat, [stack; 2], hseed), &mut rng, m).expect("decide");
                            s.apply(d.action);
                        }
                        Phase::StreetComplete => {
                            if is_runout(&s) || s.street >= street {
                                break;
                            }
                            s.advance_street();
                        }
                        _ => break,
                    }
                }
            }
        })
        .collect()
}

/// `base` with one setting's changes, and the mode to solve in.
fn setting(base: &LiveCfg, spec: &str) -> (LiveCfg, LiveMode) {
    let mut c = base.clone();
    let mut mode = LiveMode::Turn;
    for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty() && *p != "default") {
        let (k, v) = part.split_once('=').unwrap_or_else(|| panic!("setting {part:?}: expected key=value"));
        let f = |v: &str| v.parse::<f32>().unwrap_or_else(|_| panic!("setting {part:?}: not a number"));
        let on = |v: &str| match v {
            "on" => true,
            "off" => false,
            _ => panic!("setting {part:?}: on|off"),
        };
        match k {
            "mode" => mode = LiveMode::parse(v).filter(|m| *m != LiveMode::Off).unwrap_or_else(|| panic!("mode=turn|river, not {v}")),
            "iters" => c.turn_iters = v.parse().expect("iters=N"),
            "river_iters" => c.river_iters = v.parse().expect("river_iters=N"),
            "river_opp" => {
                c.river_opp_table = match v {
                    "table" => true,
                    "solve" => false,
                    _ => panic!("river_opp=table|solve, not {v}"),
                }
            }
            "menu" => c.turn_menu = LiveMenu::turn_by_name(v).unwrap_or_else(|| panic!("menu=lean|medium|full, not {v}")),
            "river" => c.river_in_turn = LiveMenu::river_in_turn_by_name(v).unwrap_or_else(|| panic!("river=lean|raises|full, not {v}")),
            "safe" => c.safe = SafeMode::parse(v).unwrap_or_else(|| panic!("safe=off|table|best, not {v}")),
            "safe_weight" => c.safe_weight = if v == "resolve" { None } else { Some(f(v)) },
            "hedges" => {
                assert_eq!(v, "off", "hedges=off");
                c.opp_floor = 0.0;
                c.opp_mix = 0.0;
                c.opp_line = 0.0;
                c.opp_temper = 1.0;
                c.opp_sizes = false;
            }
            "opp_floor" => c.opp_floor = f(v),
            "opp_mix" => c.opp_mix = f(v),
            "opp_line" => c.opp_line = f(v),
            "opp_temper" => c.opp_temper = f(v),
            "opp_sizes" => c.opp_sizes = on(v),
            _ => panic!("unknown setting {k:?}"),
        }
    }
    (c, mode)
}

/// Mean and 95% half-width.
fn mean_ci(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (m, 1.96 * (var / n).sqrt())
}

/// Per spot and setting: advantage (range, real hand), exploitability (%
/// of the pot), solve ms.
type Row = (f64, f64, f64, u64);

pub fn turncheck_cmd(args: &[String]) {
    let run = flag(args, "--run").expect("--run NAME");
    let opp_runs: Vec<String> = flag(args, "--opp-run").unwrap_or_else(|| run.clone()).split(',').map(String::from).collect();
    let n: usize = num(args, "--spots", 500);
    let stack: i32 = num(args, "--stack", 200);
    let seed: u64 = num(args, "--seed", 11);
    let specs: Vec<String> = flag(args, "--configs").unwrap_or_else(|| "default".into()).split(';').map(|s| s.trim().to_string()).collect();
    let street = match flag(args, "--street").as_deref() {
        None | Some("turn") => Street::Turn,
        Some("river") => Street::River,
        Some(x) => panic!("--street turn|river, not {x}"),
    };
    let play = flag(args, "--play").map(|m| LiveMode::parse(&m).unwrap_or_else(|| panic!("--play off|river|turn, not {m}"))).unwrap_or(LiveMode::Off);
    let mut me = load_engine(&run);
    live_flags(args, &mut me);
    me.live_cfg.parallel = false;
    me.live = LiveMode::Off;
    let base = me.live_cfg.clone();
    // The opponents always get their own engines: the bot's settings change
    // between passes.
    let opps: Vec<Engine> = opp_runs.iter().map(|r| load_rival(r)).collect();

    let t0 = Instant::now();
    let all: Vec<Vec<Spot>> = opps.iter().enumerate().map(|(j, o)| spots(&me, o, n, stack, seed ^ ((j as u64) << 32), street, play)).collect();
    log(format!("{} spots per opponent in {:.1}s", n, t0.elapsed().as_secs_f64()));

    // rows[opponent][setting][spot]
    let mut rows: Vec<Vec<Vec<Row>>> = vec![Vec::new(); opps.len()];
    for (ci, spec) in specs.iter().enumerate() {
        let (cfg, mode) = setting(&base, spec);
        me.live_cfg = cfg;
        let tag = 0x7C00_0000_0000_0000u64 ^ ((ci as u64 + 1) << 40);
        let t1 = Instant::now();
        for (j, o) in opps.iter().enumerate() {
            let me = &me;
            let r: Vec<Row> = all[j]
                .par_iter()
                .map(|sp| {
                    let v = view_of(&sp.s, sp.bot, [stack; 2], sp.seed);
                    let (dist, x, ms) = hu_cfr::live::solve_now(me, &v, mode, tag).unwrap_or_else(|e| panic!("solve: {e}"));
                    let a = advantage(me, o, &sp.s, &sp.init, sp.bot, [stack; 2], sp.seed, &dist);
                    (a.g, a.full, x, ms)
                })
                .collect();
            rows[j].push(r);
        }
        log(format!("setting {ci} ({spec}) done in {:.0}s", t1.elapsed().as_secs_f64()));
    }

    // Report: bb/100 hands = (chips per decision / 2) × decisions per hand × 100.
    let per_hand: Vec<f64> = all.iter().map(|sp| sp.len() as f64 / sp.iter().map(|x| x.hands as f64).sum::<f64>()).collect();
    for (j, name) in opp_runs.iter().enumerate() {
        log(format!("vs {name}: the bot makes a first {} decision in {:.1}% of hands", street_name(street), per_hand[j] * 100.0));
    }
    let scale = |j: usize| per_hand[j] / 2.0 * 100.0;
    for (ci, spec) in specs.iter().enumerate() {
        let mut line = format!("{ci} {spec:<32}");
        let mut pooled = Vec::new();
        for j in 0..opps.len() {
            let g: Vec<f64> = rows[j][ci].iter().map(|r| r.0 * scale(j)).collect();
            let (m, c) = mean_ci(&g);
            line += &format!("  {}: {m:+6.2} ± {c:4.2}", opp_runs[j]);
            pooled.extend(g);
        }
        let (m, c) = mean_ci(&pooled);
        let x: Vec<f64> = rows.iter().flat_map(|r| r[ci].iter().map(|r| r.2)).collect();
        let ms: Vec<f64> = rows.iter().flat_map(|r| r[ci].iter().map(|r| r.3 as f64)).collect();
        line += &format!("  pooled {m:+6.2} ± {c:4.2} bb/100 | exploitability {:.2}% pot, solve {:.0} ms", mean_ci(&x).0, mean_ci(&ms).0);
        if ci > 0 {
            let d: Vec<f64> = (0..opps.len()).flat_map(|j| rows[j][ci].iter().zip(&rows[j][0]).map(move |(a, b)| (a.0 - b.0) * scale(j))).collect();
            let (dm, dc) = mean_ci(&d);
            line += &format!(" | vs setting 0: {dm:+6.2} ± {dc:4.2}");
        }
        log(line);
    }
    log(format!("(advantage of the bot's first {} decisions over the table's, the table playing on, in bb/100 hands, 95%; real-hand versions in --save)", street_name(street)));

    if let Some(path) = flag(args, "--save") {
        let mut f = std::fs::File::create(&path).expect("create --save file");
        writeln!(f, "opp\tspot\tsetting\tadv\tadv_full\texpl_pct\tms\tpot\tstreet_actions").unwrap();
        for (j, name) in opp_runs.iter().enumerate() {
            for (ci, r) in rows[j].iter().enumerate() {
                for (i, x) in r.iter().enumerate() {
                    let s = &all[j][i].s;
                    let pot = s.pot + s.players.iter().map(|p| p.bet).sum::<i32>();
                    let k = s.actions.iter().filter(|a| a.street == street).count();
                    writeln!(f, "{name}\t{i}\t{ci}\t{}\t{}\t{}\t{}\t{pot}\t{k}", x.0, x.1, x.2, x.3).unwrap();
                }
            }
        }
        log(format!("per-spot rows in {path}"));
    }
}

fn street_name(s: Street) -> &'static str {
    if s == Street::River {
        "river"
    } else {
        "turn"
    }
}
