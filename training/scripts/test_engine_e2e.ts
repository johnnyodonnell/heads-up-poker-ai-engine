// End-to-end check of the browser bot without a browser: the real exported
// ONNX through onnxruntime-web (wasm, in Node), the real encoder, sampler and
// game.ts. The neural bot plays seat 1 (the app's bot seat) against the random
// bot in seat 0 — whose arbitrary raise sizes stand in for a human's slider —
// over freeze-out matches from random starting stacks. Every bot action must
// be legal, bet/raise sizes inside the legal non-all-in band, chips
// conserved, and inference must never fall back to random.
//
//   npx tsx training/scripts/test_engine_e2e.ts [model.onnx] [hands=3000]

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import * as ort from "onnxruntime-web";

import { advanceStreet, applyAction, createTable, legalActions, startHand } from "../../src/engine/game.ts";
import type { EngineState } from "../../src/engine/game.ts";
import { loadModel } from "../../src/engine/net.ts";
import { bestMove } from "../../src/engine/neural.ts";
import { bestMove as randomMove } from "../../src/engine/random.ts";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "../..");
ort.env.wasm.wasmPaths = resolve(root, "node_modules/onnxruntime-web/dist") + "/";
ort.env.wasm.numThreads = 1;

const modelPath = process.argv[2] ?? resolve(root, "public/models/current.onnx");
const targetHands = Number(process.argv[3] ?? 3000);
const BOT = 1;
const TOTAL = 400;

// neural.ts falls back to random play on inference errors; treat that as failure.
const origError = console.error;
console.error = (...args: unknown[]) => {
  origError(...args);
  throw new Error("console.error during e2e: " + String(args[0]));
};

await loadModel(new Uint8Array(readFileSync(modelPath)));

const counts: Record<string, number> = {};
const potFracs: number[] = [];
let hands = 0;
let decisions = 0;
let botWins = 0;
let matches = 0;
const t0 = Date.now();

function check(cond: boolean, msg: string): void {
  if (!cond) throw new Error(msg);
}

function newMatch(): EngineState {
  const t = createTable();
  const a = Math.random() < 0.5 ? 200 : 1 + Math.floor(Math.random() * (TOTAL - 1));
  t.players[0].stack = a;
  t.players[1].stack = TOTAL - a;
  return startHand(t);
}

while (hands < targetHands) {
  let s = newMatch();
  while (s.phase !== "match-over" && hands < targetHands) {
    if (s.phase === "hand-over") {
      hands++;
      s = startHand(s);
      continue;
    }
    if (s.phase === "street-complete") {
      s = advanceStreet(s);
      continue;
    }
    if (s.toAct !== BOT) {
      s = applyAction(s, randomMove(s));
      continue;
    }
    const legal = legalActions(s)!;
    const a = await bestMove(s);
    decisions++;
    counts[a.type] = (counts[a.type] ?? 0) + 1;
    const raiseOk = legal.actions.includes("bet") || legal.actions.includes("raise");
    switch (a.type) {
      case "fold":
        check(legal.actions.includes("fold"), "illegal fold");
        break;
      case "check":
        check(legal.canCheck, "illegal check");
        break;
      case "call":
        check(legal.actions.includes("call"), "illegal call");
        break;
      case "bet":
      case "raise": {
        check(legal.actions.includes(a.type), `illegal ${a.type}`);
        const amt = a.amount!;
        check(
          Number.isInteger(amt) && amt >= legal.minRaiseTo && amt < legal.maxRaiseTo,
          `raise ${amt} outside [${legal.minRaiseTo}, ${legal.maxRaiseTo})`,
        );
        const pot = s.pot + s.players.reduce((n, p) => n + p.bet, 0);
        const toCall = s.currentBet - s.players[s.toAct].bet;
        potFracs.push((amt - s.currentBet) / (pot + toCall));
        break;
      }
      case "allin":
        check(raiseOk, "illegal all-in");
        break;
    }
    s = applyAction(s, a);
    const total = s.players.reduce((n, p) => n + p.stack + p.bet, 0) + s.pot;
    check(total === TOTAL, `chips not conserved: ${total}`);
  }
  if (s.phase === "match-over") {
    hands++;
    matches++;
    if (s.players[BOT].stack > 0) botWins++;
  }
}

potFracs.sort((a, b) => a - b);
const q = (p: number) => potFracs[Math.floor(p * (potFracs.length - 1))]?.toFixed(2);
const ms = (Date.now() - t0) / decisions;
console.log(`${hands} hands, ${decisions} bot decisions (${ms.toFixed(2)} ms/decision)`);
console.log("bot actions:", counts);
console.log(`bot raise size (pot fraction) q10/q50/q90: ${q(0.1)} / ${q(0.5)} / ${q(0.9)}`);
console.log(`bot won ${botWins}/${matches} completed matches vs random`);
console.log("E2E OK");
