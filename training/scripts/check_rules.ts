// Scenario checks for game.ts rule details the AI engine (and its Rust port in
// training/hu_core) depends on:
//   1. a player is only asked to act when there's a real decision — no action
//      when a blind all-in is already covered, and no raise option against an
//      all-in opponent;
//   2. regressions: normal betting lines, short all-ins, uncalled bets, split
//      pots, and the structured per-hand action history.
//
//   npx tsx training/scripts/check_rules.ts

import assert from "node:assert/strict";

import { cardId } from "../../src/engine/cards.ts";
import type { Card, Suit } from "../../src/engine/cards.ts";
import {
  advanceStreet,
  applyAction,
  createTable,
  legalActions,
  startHand,
} from "../../src/engine/game.ts";
import type { Action, EngineState } from "../../src/engine/game.ts";
import { toGameState } from "../../src/engine/adapt.ts";

function card(rank: number, suit: Suit): Card {
  return { rank, suit, id: cardId(rank, suit) };
}

// Deal a hand with the button (= small blind) on `button` and the given stacks.
function deal(button: number, stacks: [number, number]): EngineState {
  const t = createTable();
  t.button = (button + 1) % 2; // startHand moves the button one seat on
  t.players.forEach((p, i) => (p.stack = stacks[i]));
  const s = startHand(t);
  assert.equal(s.button, button);
  return s;
}

// Rig the hole cards and the board (the deck pops from its end: flop, turn,
// river in that order).
function rig(s: EngineState, hole: [Card[], Card[]], board: Card[]): void {
  s.players[0].holeCards = hole[0];
  s.players[1].holeCards = hole[1];
  s.deck = board.slice().reverse();
}

function act(s: EngineState, seat: number, a: Action): EngineState {
  assert.equal(s.phase, "betting");
  assert.equal(s.toAct, seat, `expected seat ${seat} to act, got ${s.toAct}`);
  return applyAction(s, a);
}

// Run the hand out with no more decisions (asserts none are asked for).
function runOut(s: EngineState): EngineState {
  while (s.phase === "street-complete") s = advanceStreet(s);
  assert.ok(
    s.phase === "hand-over" || s.phase === "match-over",
    `run-out stopped in phase ${s.phase}`,
  );
  return s;
}

const actions = (s: EngineState) => legalActions(s)!.actions;

// SB wins with aces; BB holds 7-2 on a board that pairs neither.
const SB_WINS: [Card[], Card[]] = [
  [card(14, "s"), card(14, "h")],
  [card(7, "d"), card(2, "c")],
];
const BB_WINS: [Card[], Card[]] = [SB_WINS[1], SB_WINS[0]];
const BOARD = [card(13, "c"), card(12, "d"), card(11, "h"), card(5, "s"), card(3, "c")];

// --- 1a. SB all-in for 1 chip posting the blind: no decision ----------------
{
  let s = deal(0, [1, 399]);
  assert.equal(s.phase, "street-complete", "BB covers the SB's all-in: no action");
  assert.equal(s.toAct, -1);
  assert.equal(s.currentBet, 2);
  rig(s, SB_WINS, BOARD);
  s = runOut(s);
  // The BB's uncalled extra chip comes back; the SB wins 1 + 1.
  assert.deepEqual(s.players.map((p) => p.stack), [2, 398]);
  console.log("ok  SB all-in from the blind: no decision, correct payout");
}

// --- 1b. BB all-in for 1 chip posting the blind: no decision ----------------
{
  let s = deal(0, [399, 1]);
  assert.equal(s.currentBet, 1, "the bet to match is what the BB actually posted");
  assert.equal(s.phase, "street-complete", "the SB's blind already matches it");
  assert.equal(s.toAct, -1);
  rig(s, BB_WINS, BOARD);
  s = runOut(s);
  assert.deepEqual(s.players.map((p) => p.stack), [398, 2]);
  console.log("ok  BB all-in for 1: no decision, correct payout");
}

// --- 1c. BB all-in for exactly 2: SB may only fold or call ------------------
{
  let s = deal(0, [398, 2]);
  assert.equal(s.toAct, 0);
  assert.deepEqual(actions(s), ["fold", "call"]);
  // A raise attempt is downgraded to a call.
  s = applyAction(s, { type: "raise", amount: 20 });
  assert.equal(s.players[0].bet, 2);
  assert.equal(s.phase, "street-complete");
  console.log("ok  BB all-in for a full blind: SB gets fold/call only");
}

// --- 1d. SB calls all-in from the blind: BB gets no (pointless) option ------
{
  let s = deal(0, [2, 398]);
  assert.deepEqual(actions(s), ["fold", "call"], "SB can't raise: one chip behind");
  s = act(s, 0, { type: "call" });
  assert.equal(s.players[0].status, "allin");
  assert.equal(s.phase, "street-complete", "BB faces nothing against an all-in");
  console.log("ok  no BB option after an all-in limp");
}

// --- 1e. Facing a preflop shove: fold or call only --------------------------
{
  let s = deal(0, [50, 350]);
  s = act(s, 0, { type: "allin" });
  assert.equal(s.toAct, 1);
  assert.deepEqual(actions(s), ["fold", "call"]);
  // An all-in (or raise) attempt against an all-in is just a call.
  s = applyAction(s, { type: "allin" });
  assert.deepEqual(s.players.map((p) => p.bet), [50, 50]);
  assert.equal(s.phase, "street-complete");
  console.log("ok  facing a preflop shove: fold/call only");
}

// --- 1f. The human facing a shove sees no raise slider ----------------------
{
  let s = deal(1, [350, 50]); // the bot (seat 1) is the button / SB
  s = act(s, 1, { type: "allin" });
  const g = toGameState(s);
  assert.ok(g.humanCanAct);
  assert.equal(g.minRaiseAmount, 0);
  assert.equal(g.maxRaiseAmount, 0);
  console.log("ok  human facing a shove: no raise slider");
}

// --- 1g. Facing a shove on the flop: fold or call only ----------------------
{
  let s = deal(0, [100, 300]);
  s = act(s, 0, { type: "call" });
  s = act(s, 1, { type: "check" });
  s = advanceStreet(s); // flop: the BB acts first
  s = act(s, 1, { type: "bet", amount: 10 });
  s = act(s, 0, { type: "allin" }); // to 98
  assert.equal(s.toAct, 1);
  assert.deepEqual(actions(s), ["fold", "call"]);
  console.log("ok  facing a flop shove: fold/call only");
}

// --- 2a. Regression: normal open / 3-bet / call lines -----------------------
{
  let s = deal(0, [200, 200]);
  assert.equal(s.toAct, 0, "the SB (button) acts first preflop");
  assert.deepEqual(actions(s), ["fold", "call", "raise"]);
  s = act(s, 0, { type: "raise", amount: 5 });
  assert.deepEqual(actions(s), ["fold", "call", "raise"]);
  s = act(s, 1, { type: "raise", amount: 15 });
  s = act(s, 0, { type: "call" });
  assert.equal(s.phase, "street-complete");
  s = advanceStreet(s);
  assert.equal(s.toAct, 1, "the BB acts first after the flop");
  assert.deepEqual(actions(s), ["check", "bet"]);
  s = act(s, 1, { type: "check" });
  s = act(s, 0, { type: "bet", amount: 20 });
  s = act(s, 1, { type: "call" });
  assert.equal(s.phase, "street-complete");
  assert.equal(s.street, "flop");
  console.log("ok  open / 3-bet / call / c-bet line");
}

// --- 2b. Regression: the BB keeps its option after an SB limp ---------------
{
  let s = deal(0, [200, 200]);
  s = act(s, 0, { type: "call" });
  assert.equal(s.toAct, 1);
  assert.deepEqual(actions(s), ["check", "raise"]);
  s = act(s, 1, { type: "check" });
  assert.equal(s.phase, "street-complete");
  console.log("ok  BB option after a limp");
}

// --- 2c. Regression: a short all-in raise ----------------------------------
{
  let s = deal(0, [200, 14]);
  s = act(s, 0, { type: "raise", amount: 10 }); // min raise now 8
  s = act(s, 1, { type: "allin" }); // to 14: a raise of 4, short of 8
  assert.deepEqual(actions(s), ["fold", "call"]);
  s = act(s, 0, { type: "call" });
  assert.equal(s.phase, "street-complete");
  console.log("ok  short all-in raise: fold/call only");
}

// --- 2d. Regression: uncalled bets are returned ----------------------------
{
  let s = deal(0, [200, 200]);
  s = act(s, 0, { type: "raise", amount: 6 });
  s = act(s, 1, { type: "fold" });
  s = runOut(s);
  assert.deepEqual(s.players.map((p) => p.stack), [202, 198]);

  // An over-shove called all-in for less: the excess comes back.
  s = deal(0, [300, 100]);
  rig(s, BB_WINS, BOARD);
  s = act(s, 0, { type: "allin" });
  s = act(s, 1, { type: "call" });
  s = runOut(s);
  assert.deepEqual(s.players.map((p) => p.stack), [200, 200]);
  console.log("ok  uncalled bets returned");
}

// --- 2e. Regression: a split pot --------------------------------------------
// Heads-up, the uncalled excess is always returned, so both commitments are
// equal and a split is always even (no odd chip).
{
  let s = deal(0, [200, 200]);
  rig(
    s,
    [
      [card(2, "c"), card(3, "d")],
      [card(2, "d"), card(3, "c")],
    ],
    [card(14, "s"), card(13, "s"), card(12, "s"), card(11, "s"), card(9, "h")],
  );
  s = act(s, 0, { type: "call" });
  s = act(s, 1, { type: "check" });
  while (s.phase !== "hand-over") {
    s = s.phase === "street-complete" ? advanceStreet(s) : applyAction(s, { type: "check" });
  }
  assert.deepEqual(s.players.map((p) => p.stack), [200, 200]);
  assert.equal(s.results!.pots[0].winners.length, 2);
  console.log("ok  split pot");
}

// --- 2f. Regression: structured action history ------------------------------
{
  let s = deal(0, [200, 14]);
  s = act(s, 0, { type: "raise", amount: 10 });
  s = act(s, 1, { type: "allin" });
  s = act(s, 0, { type: "call" });
  assert.deepEqual(s.actions, [
    { seat: 0, street: "preflop", type: "raise", allin: false, to: 10, added: 9,
      potBefore: 3, currentBetBefore: 2, stackBefore: 199 },
    { seat: 1, street: "preflop", type: "raise", allin: true, to: 14, added: 12,
      potBefore: 12, currentBetBefore: 10, stackBefore: 12 },
    { seat: 0, street: "preflop", type: "call", allin: false, to: 14, added: 4,
      potBefore: 24, currentBetBefore: 14, stackBefore: 190 },
  ]);
  // (Use a hand where both players keep chips: here the BB is all-in.)
  let t = deal(0, [200, 200]);
  t = act(t, 0, { type: "raise", amount: 6 });
  t = act(t, 1, { type: "fold" });
  t = runOut(t);
  assert.equal(t.actions.length, 2);
  assert.deepEqual(startHand(t).actions, [], "history resets each hand");
  console.log("ok  action history");
}

console.log("all rule checks passed");
