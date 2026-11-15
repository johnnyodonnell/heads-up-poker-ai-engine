//! Encoder + action-space parity: replay the TS traces and, at every decision,
//! require the Rust encoding, action space and band-position -> raise-to
//! mapping to match `encode.ts` / `actions.ts`.

mod common;

use hu_core::actions::{action_space, raise_to_for};
use hu_core::cards::parse;
use hu_core::encode::{encode, INPUT_SIZE};
use hu_core::rules::{Action, ActionType, State};
use serde_json::Value;

const BAND_US: [f64; 7] = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];

fn parse_kind(s: &str) -> ActionType {
    match s {
        "fold" => ActionType::Fold,
        "check" => ActionType::Check,
        "call" => ActionType::Call,
        "bet" => ActionType::Bet,
        "raise" => ActionType::Raise,
        "allin" => ActionType::AllIn,
        other => panic!("bad action type {other}"),
    }
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

#[test]
fn encodings_match_typescript() {
    let fx = common::fixture("rules_traces.json");
    let (mut checked, mut sized) = (0usize, 0usize);
    for (mi, m) in fx["matches"].as_array().unwrap().iter().enumerate() {
        let st = &m["stacks"];
        let mut s = State::new_table([st[0].as_i64().unwrap() as i32, st[1].as_i64().unwrap() as i32]);
        for (ei, ev) in m["events"].as_array().unwrap().iter().enumerate() {
            match ev["kind"].as_str().unwrap() {
                "start" => {
                    let deck: Vec<u8> = ev["deck"]
                        .as_array()
                        .map(|d| d.iter().map(|c| parse(c.as_str().unwrap()).unwrap()).collect())
                        .unwrap_or_default();
                    s.start_hand(deck);
                }
                "advance" => s.advance_street(),
                "act" => {
                    let e = &ev["enc"];
                    let at = format!("match {mi} event {ei}");
                    let (x, sp) = encode(&s).expect("encode");
                    let mut want = vec![0f32; INPUT_SIZE];
                    for pair in e["enc"].as_array().unwrap() {
                        want[pair[0].as_u64().unwrap() as usize] = f(&pair[1]) as f32;
                    }
                    for i in 0..INPUT_SIZE {
                        assert!(
                            (x[i] - want[i]).abs() <= 1e-5 * (1.0 + want[i].abs()),
                            "{at}: x[{i}] = {} vs TS {}",
                            x[i],
                            want[i]
                        );
                    }
                    let w = &e["space"];
                    let mask: Vec<f32> = w["mask"].as_array().unwrap().iter().map(|v| f(v) as f32).collect();
                    assert_eq!(sp.mask.to_vec(), mask, "{at}: mask");
                    assert_eq!(sp.pot_after_call as f64, f(&w["potAfterCall"]), "{at}");
                    assert_eq!(sp.min_raise_to as f64, f(&w["minRaiseTo"]), "{at}");
                    assert_eq!(sp.max_raise_to as f64, f(&w["maxRaiseTo"]), "{at}");
                    assert!((sp.s_lo - f(&w["sLo"])).abs() < 1e-9, "{at}: sLo");
                    assert!((sp.s_hi - f(&w["sHi"])).abs() < 1e-9, "{at}: sHi");
                    let tos: Vec<i32> = w["raiseTo"].as_array().unwrap().iter().map(|v| f(v) as i32).collect();
                    if sp.mask[2] > 0.0 {
                        let ours: Vec<i32> = BAND_US.iter().map(|&u| raise_to_for(&sp, u)).collect();
                        assert_eq!(ours, tos, "{at}: raise-to by band position");
                        assert_eq!(ours[0], sp.min_raise_to, "{at}: u=0 is the min-raise");
                        assert_eq!(ours[6], sp.max_raise_to, "{at}: u=1 is all-in minus one");
                        sized += 1;
                    } else {
                        assert!(tos.is_empty(), "{at}");
                    }
                    assert!(action_space(&s).is_some());
                    checked += 1;
                    let a = &ev["action"];
                    let kind = parse_kind(a["type"].as_str().unwrap());
                    let amount = a.get("amount").and_then(Value::as_i64).map(|v| v as i32);
                    s.apply(Action { kind, amount });
                }
                _ => {}
            }
        }
    }
    println!("checked {checked} encodings ({sized} with a sized-raise band)");
    assert!(checked > 5000, "only {checked} encodings checked");
}
