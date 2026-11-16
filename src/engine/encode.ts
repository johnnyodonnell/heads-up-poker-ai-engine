// Canonical NN input encoding for the heads-up poker engine. It must match
// the encoding the model was trained with.
//
// Everything is from the point of view of the player to act ("the mover"):
// the other seat is "the opponent", and suits are relabeled by order of first
// appearance (hole cards, then flop, turn, river; ties by rank, then real
// suit) so suit-isomorphic spots encode identically.
//
// One flat Float32Array, sliced inside the network:
//   [ cards (7) | history (HIST_TOKENS x TOKEN_FEATS) | static (STATIC_SIZE) ]
//
// cards:   canonical card indices (canonSuit * 13 + rank - 2) for hole 1-2,
//          flop 1-3, turn, river; NO_CARD for cards not dealt yet. The network
//          turns these into one-hot suit x rank planes.
// history: one token per betting action this hand, most recent first; slots
//          past the end are all-zero (valid = 0). Older actions beyond
//          HIST_TOKENS are dropped.
// static:  table state (see `encodeStatic`).

import { BIG_BLIND, SEATS } from "./game";
import type { EngineState, HandAction, Street } from "./game";
import type { Card } from "./cards";
import { SUITS } from "./cards";
import { actionSpace } from "./actions";
import type { ActionSpace } from "./actions";

export const CARD_SLOTS = 7;
export const NO_CARD = 52;
export const HIST_TOKENS = 32;
// street(4) | actor: me, opponent (2) | fold check call bet raise (5) | all-in |
// size as log pot-fraction | call as pot fraction | lbb added | lbb pot before
// | lbb stack before | valid
export const TOKEN_FEATS = 18;
// status: out active allin folded (4) | button | lbb stack | lbb bet |
// lbb commit | stack / 200 | has acted
export const SEAT_FEATS = 10;
// street(4) | seats (me, opponent) | pot (4) | effective stack (2) |
// raise band (4) | legality (5)
export const STATIC_SIZE = 4 + SEATS * SEAT_FEATS + 4 + 2 + 4 + 5; // 39
export const HIST_OFFSET = CARD_SLOTS;
export const STATIC_OFFSET = CARD_SLOTS + HIST_TOKENS * TOKEN_FEATS;
export const INPUT_SIZE = STATIC_OFFSET + STATIC_SIZE; // 622

const STREETS: Street[] = ["preflop", "flop", "turn", "river"];
const TYPES = ["fold", "check", "call", "bet", "raise"] as const;

const lbb = (chips: number): number => Math.log1p(chips / BIG_BLIND);

// Cards in scan order: hole (rank desc, suit), flop (rank desc, suit), turn, river.
function orderedGroup(cards: Card[]): Card[] {
  return cards
    .slice()
    .sort((a, b) => b.rank - a.rank || SUITS.indexOf(a.suit) - SUITS.indexOf(b.suit));
}

// Canonical card indices for the mover's hole cards + the board.
export function canonicalCards(hole: Card[], community: Card[]): number[] {
  const groups = [
    orderedGroup(hole),
    orderedGroup(community.slice(0, 3)),
    community.slice(3, 4),
    community.slice(4, 5),
  ];
  const suitMap = new Map<string, number>();
  const out: number[] = [];
  for (const g of groups) {
    for (const c of g) {
      if (!suitMap.has(c.suit)) suitMap.set(c.suit, suitMap.size);
      out.push(suitMap.get(c.suit)! * 13 + (c.rank - 2));
    }
  }
  // hole(2) flop(3) turn(1) river(1), padded with NO_CARD
  const slots: number[] = [];
  let k = 0;
  for (const [g, n] of [[groups[0], 2], [groups[1], 3], [groups[2], 1], [groups[3], 1]] as const) {
    for (let j = 0; j < n; j++) slots.push(j < g.length ? out[k++] : NO_CARD);
  }
  return slots;
}

function encodeToken(x: Float32Array, off: number, a: HandAction, mover: number): void {
  x[off + STREETS.indexOf(a.street)] = 1;
  x[off + 4 + (a.seat === mover ? 0 : 1)] = 1;
  x[off + 6 + TYPES.indexOf(a.type)] = 1;
  x[off + 11] = a.allin ? 1 : 0;
  const betBefore = a.to - a.added;
  if (a.type === "bet" || a.type === "raise") {
    const potAfterCall = a.potBefore + (a.currentBetBefore - betBefore);
    x[off + 12] = Math.log((a.to - a.currentBetBefore) / potAfterCall);
  }
  if (a.type === "call") x[off + 13] = a.added / a.potBefore;
  x[off + 14] = lbb(a.added);
  x[off + 15] = lbb(a.potBefore);
  x[off + 16] = lbb(a.stackBefore);
  x[off + 17] = 1;
}

function encodeStatic(x: Float32Array, off: number, s: EngineState, space: ActionSpace): void {
  const mover = s.toAct;
  const me = s.players[mover];
  const opp = s.players[1 - mover];
  let o = off;
  x[o + STREETS.indexOf(s.street)] = 1;
  o += 4;
  for (const p of [me, opp]) {
    const st = ["out", "active", "allin", "folded"].indexOf(p.status);
    x[o + st] = 1;
    x[o + 4] = p.id === s.button ? 1 : 0;
    x[o + 5] = lbb(p.stack);
    x[o + 6] = lbb(p.bet);
    x[o + 7] = lbb(p.commit);
    x[o + 8] = p.stack / 200;
    x[o + 9] = p.hasActed ? 1 : 0;
    o += SEAT_FEATS;
  }
  const potTotal = s.pot + s.players.reduce((n, q) => n + q.bet, 0);
  const toCall = space.legal.toCall;
  x[o] = lbb(potTotal);
  x[o + 1] = lbb(toCall);
  x[o + 2] = toCall / (potTotal + toCall);
  x[o + 3] = Math.log1p(me.stack / potTotal);
  o += 4;
  const eff = Math.min(me.stack, opp.stack);
  x[o] = lbb(eff);
  x[o + 1] = Math.log1p(eff / potTotal);
  o += 2;
  x[o] = space.sLo;
  x[o + 1] = space.sHi;
  x[o + 2] = lbb(space.minRaiseTo);
  x[o + 3] = lbb(space.maxRaiseTo);
  o += 4;
  x[o] = space.mask[0];
  x[o + 1] = space.legal.canCheck ? 1 : 0;
  x[o + 2] = space.mask[2];
  x[o + 3] = space.mask[3];
  x[o + 4] = space.legal.canCheck ? 0 : 1;
}

// Encode the state for the player to act. Returns null if nobody is to act.
export function encode(s: EngineState): { x: Float32Array; space: ActionSpace } | null {
  const space = actionSpace(s);
  if (!space) return null;
  const mover = s.toAct;
  const x = new Float32Array(INPUT_SIZE);
  const cards = canonicalCards(s.players[mover].holeCards, s.community);
  for (let k = 0; k < CARD_SLOTS; k++) x[k] = cards[k];
  const acts = s.actions;
  for (let t = 0; t < Math.min(HIST_TOKENS, acts.length); t++) {
    encodeToken(x, HIST_OFFSET + t * TOKEN_FEATS, acts[acts.length - 1 - t], mover);
  }
  encodeStatic(x, STATIC_OFFSET, s, space);
  return { x, space };
}
