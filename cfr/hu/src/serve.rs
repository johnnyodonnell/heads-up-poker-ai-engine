//! `hu serve`: the engine as an HTTP service, plus the built web app.
//!
//!   hu serve --run NAME [--port 8790] [--web DIR]
//!
//!   POST /api/act      the bot's move for a hand (see ActRequest)
//!   POST /api/hand     a finished hand from the app, for the hand log
//!   GET  /api/health   what's loaded
//!   GET  /*            the built web app ($HU_HOME/web by default)
//!
//! Logs, in $HU_HOME/logs (one JSON object per line):
//! - `decisions.jsonl`: every decision, with the app's full request (the
//!   hand so far, the human's real bet sizes included) and the engine's
//!   debug output (how the real betting was translated, the options and
//!   their probabilities, any fallback, live-solve stats).
//! - `hands.jsonl`: every finished hand as the app saw it: both players'
//!   cards, the board, every action, the result.

use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tower_http::services::ServeDir;

use hu_cfr::engine::{cards_str, Debug, Engine, HistAction, View};
use hu_core::cards::parse;
use hu_core::rules::ActionType;

use crate::play::{live_flags, load_engine};
use crate::util::{flag, hu_home, log, num};

/// The bot's view of the hand, as the web app sends it.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ActRequest {
    /// Unique per hand (keeps translation stable across the bot's decisions).
    hand_id: String,
    bot: usize,
    button: usize,
    /// Stacks at the start of the hand, before the blinds.
    stacks: [i32; 2],
    hole: Vec<String>,
    board: Vec<String>,
    /// The hand's betting actions so far (the app's HandAction records; extra
    /// fields are ignored).
    actions: Vec<ReqAction>,
}

#[derive(Deserialize)]
struct ReqAction {
    seat: usize,
    #[serde(rename = "type")]
    kind: String,
    to: i32,
}

#[derive(Serialize)]
struct RespAction {
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    amount: Option<i32>,
}

#[derive(Serialize)]
struct ActResponse {
    action: RespAction,
    debug: Debug,
}

fn fnv1a(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3))
}

fn to_view(r: &ActRequest) -> Result<View, String> {
    let card = |s: &String| parse(s).ok_or_else(|| format!("bad card {s:?}"));
    if r.hole.len() != 2 {
        return Err("hole must have two cards".into());
    }
    let hole = [card(&r.hole[0])?, card(&r.hole[1])?];
    let board = r.board.iter().map(card).collect::<Result<Vec<_>, _>>()?;
    let actions = r
        .actions
        .iter()
        .map(|a| {
            let kind = match a.kind.as_str() {
                "fold" => ActionType::Fold,
                "check" => ActionType::Check,
                "call" => ActionType::Call,
                "bet" => ActionType::Bet,
                "raise" => ActionType::Raise,
                other => return Err(format!("bad action type {other:?}")),
            };
            Ok(HistAction { seat: a.seat, kind, to: a.to })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(View { bot: r.bot, button: r.button, stacks: r.stacks, hole, board, actions, seed: fnv1a(&r.hand_id) })
}

type Err = (StatusCode, Json<Value>);

fn bad(msg: String) -> Err {
    log(format!("400 {msg}"));
    (StatusCode::BAD_REQUEST, Json(json!({ "error": msg })))
}

/// What the handlers share: the engine, what it serves, and the JSON logs.
struct App {
    engine: Arc<Engine>,
    run: String,
    live: &'static str,
    decisions: Mutex<std::fs::File>,
    hands: Mutex<std::fs::File>,
}

impl App {
    /// Who served a logged record: the time (unix seconds), run and live mode.
    fn stamp(&self) -> Value {
        let t = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
        json!({ "t": (t * 1000.0).round() / 1000.0, "run": self.run, "live": self.live })
    }
}

fn open_log(name: &str) -> std::fs::File {
    let dir = hu_home().join("logs");
    std::fs::create_dir_all(&dir).expect("logs dir");
    let path = dir.join(name);
    std::fs::OpenOptions::new().create(true).append(true).open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Append one JSON object as a line (a failed write is logged, not fatal).
fn append(file: &Mutex<std::fs::File>, v: &Value) {
    let mut line = v.to_string();
    line.push('\n');
    let mut f = file.lock().unwrap_or_else(|p| p.into_inner());
    if let Err(e) = f.write_all(line.as_bytes()) {
        log(format!("log write failed: {e}"));
    }
}

async fn act(State(app): State<Arc<App>>, Json(raw): Json<Value>) -> Result<Json<ActResponse>, Err> {
    let req: ActRequest = serde_json::from_value(raw.clone()).map_err(|e| bad(format!("bad request: {e}")))?;
    let view = to_view(&req).map_err(bad)?;
    let engine = app.engine.clone();
    let t0 = Instant::now();
    // Live solves take seconds: keep them off the async workers.
    let (d, view) = tokio::task::spawn_blocking(move || {
        let mut rng = SmallRng::from_entropy();
        engine.decide(&view, &mut rng).map(|d| (d, view))
    })
    .await
    .map_err(|e| bad(format!("engine task failed: {e}")))?
    .map_err(bad)?;
    let ms = t0.elapsed().as_millis() as u64;
    let (kind, amount) = match d.action.kind {
        ActionType::Fold => ("fold", None),
        ActionType::Check => ("check", None),
        ActionType::Call => ("call", None),
        ActionType::Bet => ("bet", d.action.amount),
        ActionType::Raise => ("raise", d.action.amount),
        ActionType::AllIn => ("allin", None),
    };
    append(
        &app.decisions,
        &json!({ "server": app.stamp(), "ms": ms, "request": raw, "action": { "type": kind, "amount": amount }, "debug": d.debug }),
    );
    log(format!(
        "hand {} {} [{}|{}] eff {}bb (strategy {}bb) bucket {:?}: {}{}",
        req.hand_id,
        d.debug.street,
        cards_str(&view.hole),
        cards_str(&view.board),
        d.debug.effective_bb,
        d.debug.strategy_bb,
        d.debug.bucket,
        d.debug.chosen,
        d.debug.fallback.as_ref().map(|f| format!("  [fallback: {f}]")).unwrap_or_default()
    ));
    Ok(Json(ActResponse { action: RespAction { kind, amount }, debug: d.debug }))
}

/// A finished hand from the app (its own record, see `logHand` in
/// src/engine/remote.ts), appended to hands.jsonl with the server's stamp.
async fn hand(State(app): State<Arc<App>>, Json(mut raw): Json<Value>) -> Result<StatusCode, Err> {
    let Some(obj) = raw.as_object_mut() else {
        return Err(bad("a hand record must be a JSON object".into()));
    };
    let id = obj.get("handId").and_then(Value::as_str).unwrap_or("?").to_string();
    obj.insert("server".into(), app.stamp());
    append(&app.hands, &raw);
    log(format!("hand {id} logged"));
    Ok(StatusCode::NO_CONTENT)
}

async fn health(State(app): State<Arc<App>>) -> Json<Value> {
    let engine = &app.engine;
    Json(json!({
        "abstraction": engine.abs.meta.name,
        "live": engine.live.name(),
        "riverIters": engine.live_cfg.river_iters,
        "turnIters": engine.live_cfg.turn_iters,
        "safe": engine.live_cfg.safe.name(),
        "fallback": engine.fallback.name(),
        "buckets": engine.abs.buckets(),
        "strategies": engine.strategies.iter().map(|s| json!({
            "stackBb": s.meta.stack / 2,
            "menu": s.meta.menu,
            "iterations": s.meta.iterations,
        })).collect::<Vec<_>>(),
    }))
}

pub fn serve_cmd(args: &[String]) {
    let run = flag(args, "--run").expect("--run NAME");
    let port: u16 = num(args, "--port", 8790);
    let web = flag(args, "--web").map(Into::into).unwrap_or_else(|| hu_home().join("web"));
    let mut e = load_engine(&run);
    live_flags(args, &mut e);
    let live = e.live.name();
    let state = Arc::new(App {
        engine: Arc::new(e),
        run: run.clone(),
        live,
        decisions: Mutex::new(open_log("decisions.jsonl")),
        hands: Mutex::new(open_log("hands.jsonl")),
    });
    let app = Router::new()
        .route("/api/act", post(act))
        .route("/api/hand", post(hand))
        .route("/api/health", get(health))
        .with_state(state)
        .fallback_service(ServeDir::new(&web));
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().expect("tokio runtime");
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await.expect("bind");
        log(format!("serving run {run} (live solving: {live}) on port {port} (web app from {})", web.display()));
        axum::serve(listener, app).await.expect("server");
    });
}
