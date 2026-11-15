// The neural engine's action space, shared (by parity tests) with the Rust
// trainer (training/hu_core/src/actions.rs).
//
// The policy picks one of four action types — fold, check/call, bet/raise,
// all-in — and, for a bet/raise, a continuous size. Sizes live on a log scale
// of the pot:
//
//   s = ln((raiseTo - currentBet) / potAfterCall)
//
// the raise increment as a fraction of the pot after calling ("half pot" is
// ln 0.5 in every spot). The legal non-all-in raises (min-raise up to one chip
// short of all-in) form the band [sLo, sHi]. The network outputs a Beta(a, b)
// over the band's unit interval, u in [0, 1] -> s = sLo + u * (sHi - sLo); a
// Beta (with a, b >= 1) keeps density across the whole band, so the size can't
// get stuck piled up at the min-raise. All-in is its own action type.

import { legalActions } from "./game";
import type { Action, EngineState, LegalActions } from "./game";

export const FOLD = 0;
export const CALL = 1; // check or call
export const RAISE = 2; // bet or raise, sized by u
export const ALLIN = 3;
export const NUM_TYPES = 4;

export interface ActionSpace {
  legal: LegalActions;
  mask: [number, number, number, number]; // 1 = allowed, per action type
  currentBet: number;
  potAfterCall: number; // gathered pot + street bets + the mover's call
  minRaiseTo: number; // smallest legal (non-all-in) raise-to, 0 if none
  maxRaiseTo: number; // largest legal non-all-in raise-to (all-in minus one)
  sLo: number; // s of minRaiseTo (0 if no sized raise is allowed)
  sHi: number; // s of maxRaiseTo
}

export function actionSpace(state: EngineState): ActionSpace | null {
  const legal = legalActions(state);
  if (!legal) return null;
  const p = state.players[state.toAct];
  const potTotal = state.pot + state.players.reduce((n, q) => n + q.bet, 0);
  const potAfterCall = potTotal + (state.currentBet - p.bet);
  const canRaise = legal.actions.includes("bet") || legal.actions.includes("raise");
  // A non-all-in raise needs room between the min-raise and the shove.
  const sized = canRaise && legal.minRaiseTo < legal.maxRaiseTo;
  const minTo = legal.minRaiseTo;
  const maxTo = legal.maxRaiseTo - 1;
  const s = (to: number): number => Math.log((to - state.currentBet) / potAfterCall);
  return {
    legal,
    mask: [legal.actions.includes("fold") ? 1 : 0, 1, sized ? 1 : 0, canRaise ? 1 : 0],
    currentBet: state.currentBet,
    potAfterCall,
    minRaiseTo: sized ? minTo : 0,
    maxRaiseTo: sized ? maxTo : 0,
    sLo: sized ? s(minTo) : 0,
    sHi: sized ? s(maxTo) : 0,
  };
}

// The raise-to amount for band position u in [0, 1] (rounded to a chip,
// clamped to the legal band).
export function raiseToFor(space: ActionSpace, u: number): number {
  const s = space.sLo + u * (space.sHi - space.sLo);
  const to = Math.round(space.currentBet + Math.exp(s) * space.potAfterCall);
  return Math.min(Math.max(to, space.minRaiseTo), space.maxRaiseTo);
}

// The engine action for an action type (+ band position u for RAISE).
export function toEngineAction(space: ActionSpace, type: number, u = 0): Action {
  switch (type) {
    case FOLD:
      return { type: "fold" };
    case CALL:
      return { type: space.legal.canCheck ? "check" : "call" };
    case RAISE:
      return {
        type: space.currentBet === 0 ? "bet" : "raise",
        amount: raiseToFor(space, u),
      };
    default:
      return { type: "allin" };
  }
}

// --- Beta sampling -----------------------------------------------------------

// Standard normal draw (Box-Muller).
function gaussian(rand: () => number): number {
  let u = 0;
  while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

// Gamma(shape, 1) for shape >= 1 (Marsaglia & Tsang).
function sampleGamma(shape: number, rand: () => number): number {
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number;
    let v: number;
    do {
      x = gaussian(rand);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rand();
    if (u < 1 - 0.0331 * x ** 4) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

// Beta(a, b) draw for a, b >= 1, kept strictly inside (0, 1).
export function sampleBeta(a: number, b: number, rand: () => number = Math.random): number {
  const x = sampleGamma(a, rand);
  const y = sampleGamma(b, rand);
  const u = x / (x + y);
  return Math.min(Math.max(u, 1e-6), 1 - 1e-6);
}
