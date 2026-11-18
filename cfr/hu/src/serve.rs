//! `hu serve`: the engine as an HTTP service, plus the built web app.
//!
//!   hu serve --run NAME [--port 8790] [--web DIR]
//!
//!   POST /api/act      the bot's move for a hand (see ActRequest)
//!   GET  /api/health   what's loaded
//!   GET  /*            the built web app ($HU_HOME/web by default)

use std::sync::Arc;

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

async fn act(State(engine): State<Arc<Engine>>, Json(req): Json<ActRequest>) -> Result<Json<ActResponse>, Err> {
    let view = to_view(&req).map_err(bad)?;
    // Live solves take seconds: keep them off the async workers.
    let (d, view) = tokio::task::spawn_blocking(move || {
        let mut rng = SmallRng::from_entropy();
        engine.decide(&view, &mut rng).map(|d| (d, view))
    })
    .await
    .map_err(|e| bad(format!("engine task failed: {e}")))?
    .map_err(bad)?;
    let (kind, amount) = match d.action.kind {
        ActionType::Fold => ("fold", None),
        ActionType::Check => ("check", None),
        ActionType::Call => ("call", None),
        ActionType::Bet => ("bet", d.action.amount),
        ActionType::Raise => ("raise", d.action.amount),
        ActionType::AllIn => ("allin", None),
    };
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

async fn health(State(engine): State<Arc<Engine>>) -> Json<Value> {
    Json(json!({
        "abstraction": engine.abs.meta.name,
        "live": engine.live.name(),
        "riverIters": engine.live_cfg.river_iters,
        "turnIters": engine.live_cfg.turn_iters,
        "safe": engine.live_cfg.safe.name(),
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
    let engine = Arc::new(e);
    let app = Router::new()
        .route("/api/act", post(act))
        .route("/api/health", get(health))
        .with_state(engine)
        .fallback_service(ServeDir::new(&web));
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().expect("tokio runtime");
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await.expect("bind");
        log(format!("serving run {run} (live solving: {live}) on port {port} (web app from {})", web.display()));
        axum::serve(listener, app).await.expect("server");
    });
}
