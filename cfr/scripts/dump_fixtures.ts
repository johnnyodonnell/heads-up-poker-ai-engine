// Dump parity fixtures from the authoritative TS engine (src/engine/*.ts) for
// the Rust port in cfr/hu_core:
//
//   cfr/fixtures/rules_traces.json   full random freeze-out matches from
//       random starting stacks (including 1–3 chip stacks, so blind all-ins
//       happen), step by step: every startHand (with the deck it shuffled),
//       every action (with the legal-action set it was chosen from), every
//       advanceStreet, the state after each step, and each hand's results +
//       action history.
//   cfr/fixtures/eval_hands.json     random 7-card hands with their
//       evaluate7 score arrays (for ordering parity).
//
//   npx tsx cfr/scripts/dump_fixtures.ts [matches=400] [evalHands=20000]
//
// Then cfr/sync_to_box.sh and, on the box, cargo test --release -p hu_core.

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { createDeck, shuffle } from "../../src/engine/cards.ts";
import { evaluate7 } from "../../src/engine/evaluator.ts";
import {
  advanceStreet,
  applyAction,
  createTable,
  legalActions,
  startHand,
  STARTING_STACK,
} from "../../src/engine/game.ts";
import type { EngineState } from "../../src/engine/game.ts";
import { bestMove } from "../../src/engine/random.ts";

const here = dirname(fileURLToPath(import.meta.url));
const outDir = resolve(here, "../fixtures");
const TOTAL = 2 * STARTING_STACK;

function summary(s: EngineState) {
  return {
    phase: s.phase,
    street: s.street,
    toAct: s.toAct,
    pot: s.pot,
    currentBet: s.currentBet,
    minRaise: s.minRaise,
    button: s.button,
    sbSeat: s.sbSeat,
    bbSeat: s.bbSeat,
    community: s.community.map((c) => c.id),
    // [stack, bet, commit, status, hasActed, actedAt, holeCards]
    players: s.players.map((p) => [
      p.stack,
      p.bet,
      p.commit,
      p.status,
      p.hasActed,
      p.actedAt,
      p.holeCards.map((c) => c.id),
    ]),
  };
}

// The full shuffled deck startHand dealt from: it pops hole cards off the end
// (two per seat, seat order), so put them back in reverse.
function fullDeck(s: EngineState): string[] {
  const dealt: string[] = [];
  for (const p of s.players) for (const c of p.holeCards) dealt.push(c.id);
  return [...s.deck.map((c) => c.id), ...dealt.reverse()];
}

function results(s: EngineState) {
  return {
    pots: s.results!.pots.map((p) => ({ amount: p.amount, winners: p.winners })),
    stacks: s.players.map((p) => p.stack),
    history: s.actions,
  };
}

// Starting stacks: half the matches from 200/200, the rest from a random
// split, a quarter of those with one player down to 1–3 chips.
function startStacks(): [number, number] {
  const r = Math.random();
  if (r < 0.5) return [STARTING_STACK, STARTING_STACK];
  const a =
    r < 0.625
      ? 1 + Math.floor(Math.random() * 3)
      : 1 + Math.floor(Math.random() * (TOTAL - 1));
  return Math.random() < 0.5 ? [a, TOTAL - a] : [TOTAL - a, a];
}

function recordMatch() {
  const events: unknown[] = [];
  const stacks = startStacks();
  const t = createTable();
  t.players.forEach((p, i) => (p.stack = stacks[i]));
  let s = startHand(t);
  events.push({ kind: "start", deck: fullDeck(s), after: summary(s) });
  for (let guard = 0; ; guard++) {
    if (guard > 200000) throw new Error("runaway match");
    if (s.phase === "match-over") break;
    if (s.phase === "hand-over") {
      s = startHand(s);
      events.push({
        kind: "start",
        deck: s.phase === "match-over" ? null : fullDeck(s),
        after: summary(s),
      });
      continue;
    }
    if (s.phase === "street-complete") {
      s = advanceStreet(s);
      events.push({ kind: "advance", after: summary(s) });
    } else {
      const legal = legalActions(s)!;
      const action = bestMove(s);
      s = applyAction(s, action);
      events.push({ kind: "act", action, legal, after: summary(s) });
    }
    if (s.phase === "hand-over" || s.phase === "match-over") {
      events.push({ kind: "end", results: results(s) });
    }
  }
  return { stacks, events };
}

const nMatches = Number(process.argv[2] ?? 400);
const nEval = Number(process.argv[3] ?? 20000);

mkdirSync(outDir, { recursive: true });

const matches = [];
for (let i = 0; i < nMatches; i++) matches.push(recordMatch());
writeFileSync(resolve(outDir, "rules_traces.json"), JSON.stringify({ matches }));

const hands = [];
for (let i = 0; i < nEval; i++) {
  const cards = shuffle(createDeck()).slice(0, 7);
  hands.push({ cards: cards.map((c) => c.id), score: evaluate7(cards) });
}
writeFileSync(resolve(outDir, "eval_hands.json"), JSON.stringify({ hands }));

const nEvents = matches.reduce((n, m) => n + m.events.length, 0);
console.log(`wrote ${nMatches} matches (${nEvents} events), ${nEval} eval hands to ${outDir}`);
