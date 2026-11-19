// The bot's engine: a precomputed CFR strategy served from the box
// (cfr/, `hu serve`). Each decision is one POST /api/act with the bot's view
// of the hand. There is deliberately no fallback: if the engine can't be
// reached, bestMove rejects and the table shows the error.
//
//   await bestMove(state)       // -> an Action for state.toAct
//   onDecision(listener)        // the debug info of every decision
//   thinkingMs(state)           // how long the thinking bar runs for this decision

import type { Action, EngineState } from "./game";

export interface DecisionOption {
  label: string; // the abstract action in the strategy
  action: string; // the real action it maps to
  prob: number;
}

// Set when the decision was solved live on the box (turn/river).
export interface LiveDebug {
  street: string;
  solves: number;
  ms: number;
  iterations: number;
  decision_nodes: number;
  exploitability_pct: number;
  bot_range: number;
  opp_range: number;
  // Safe solve: every hand you could hold was held to what it gets against the table.
  safe?: boolean;
}

export interface DecisionDebug {
  effective_bb: number;
  strategy_bb: number;
  street: string;
  bucket: number | null;
  bucket_label: string | null;
  node: number | null;
  path: string[];
  options: DecisionOption[];
  chosen: string;
  fallback: string | null;
  live?: LiveDebug | null;
}

export interface BotDecision {
  hand: number;
  action: Action;
  debug: DecisionDebug;
}

export class EngineError extends Error {}

// One id per page load, so hand ids are unique across matches.
const matchId = Math.random().toString(36).slice(2, 10);

type Listener = (d: BotDecision) => void;
let listeners: Listener[] = [];

export function onDecision(listener: Listener): () => void {
  listeners.push(listener);
  return () => {
    listeners = listeners.filter((l) => l !== listener);
  };
}

// Which streets the box solves live ("off", "river" or "turn"), from
// /api/health at page load. Until it answers (or if it can't), no bar.
let liveMode = "off";
void fetch("/api/health")
  .then((r) => (r.ok ? r.json() : null))
  .then((h: { live?: string } | null) => {
    if (h?.live) liveMode = h.live;
  })
  .catch(() => {});

// How long the bot's thinking bar runs for the decision in `state`: twice
// the typical end-to-end time on the box (measured 2026-11-19, bot vs bot,
// 60 hands per mode: a live river decision ~0.55 s, ~0.6 s under live turn;
// a live turn decision ~2.3 s). Table lookups take ~10 ms, so 0 (no bar).
export function thinkingMs(state: EngineState): number {
  if (state.street === "river" && liveMode !== "off") return 1200;
  if (state.street === "turn" && liveMode === "turn") return 4500;
  return 0;
}

export async function bestMove(state: EngineState): Promise<Action> {
  const bot = state.toAct;
  const body = {
    handId: `${matchId}-${state.handNumber}`,
    bot,
    button: state.button,
    // Stacks at the start of the hand, before the blinds.
    stacks: state.players.map((p) => p.stack + p.commit),
    hole: state.players[bot].holeCards.map((c) => c.id),
    board: state.community.map((c) => c.id),
    actions: state.actions,
  };
  let res: Response;
  try {
    res = await fetch("/api/act", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    throw new EngineError(`Can't reach the engine (${String(err)}).`);
  }
  if (!res.ok) {
    let detail = await res.text();
    try {
      detail = JSON.parse(detail).error ?? detail;
    } catch {
      // not JSON: keep the text
    }
    throw new EngineError(`The engine returned ${res.status}: ${detail}`);
  }
  const data = (await res.json()) as {
    action: { type: Action["type"]; amount?: number };
    debug: DecisionDebug;
  };
  const action: Action =
    data.action.amount === undefined
      ? { type: data.action.type }
      : { type: data.action.type, amount: data.action.amount };
  for (const l of listeners) l({ hand: state.handNumber, action, debug: data.debug });
  return action;
}
