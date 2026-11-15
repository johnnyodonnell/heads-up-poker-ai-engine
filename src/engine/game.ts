// Pure No-Limit Texas Hold'em rules for a heads-up (two-seat) table. No
// React, no DOM.
//
// The whole game is a plain immutable-ish state object; every function here
// takes a state and returns a new one (we shallow-clone what we touch). The UI
// (via the useGameState hook + adapter) owns orchestration — whose turn it is,
// running out streets — and calls into these pure transitions:
//
//   createTable()           -> seated table, no hand dealt (phase 'waiting')
//   createGame()            -> table with the first hand already dealt
//   startHand(state)        -> deal the next hand
//   legalActions(state)     -> what the player to act may do
//   applyAction(state, act) -> apply one player's action
//   advanceStreet(state)    -> deal the next street / run out / go to showdown
//
// Heads-up rules: the button posts the small blind and acts first pre-flop;
// the big blind acts first on every later street.
//
// Chip model: blinds 1/2, both players start at 200 (100bb). When one player
// has all the chips the match is over.

import { createDeck, shuffle } from "./cards";
import type { Card } from "./cards";
import { evaluate7, compareScores, categoryName } from "./evaluator";
import type { Score } from "./evaluator";

export const SEATS = 2;
export const SMALL_BLIND = 1;
export const BIG_BLIND = 2;
export const STARTING_STACK = 200;

export const PLAYER_NAMES = ["You", "Bot"];

// --- types ------------------------------------------------------------------

// Player.status:
//   'active' — in the hand, has chips, can still act
//   'allin'  — in the hand, no chips left to act with
//   'folded' — out of this hand
//   'out'    — busted, sitting out until a new game
export type EngineStatus = "active" | "allin" | "folded" | "out";
export type Street = "preflop" | "flop" | "turn" | "river";
// phase drives the orchestration layer:
//   'waiting'         — table seated, no hand dealt yet (Deal button)
//   'betting'         — waiting on state.toAct to act
//   'street-complete' — betting closed; caller runs advanceStreet
//   'hand-over'       — hand decided (fold-out or showdown); await next hand
//   'match-over'      — one player holds every chip
export type EnginePhase =
  | "waiting"
  | "betting"
  | "street-complete"
  | "hand-over"
  | "match-over";

export interface EnginePlayer {
  id: number;
  name: string;
  isHuman: boolean;
  stack: number;
  holeCards: Card[];
  status: EngineStatus;
  bet: number; // chips committed on the current street
  commit: number; // chips committed across the whole hand (for side pots)
  hasActed: boolean;
  // The street's bet level (currentBet) right after this player last acted.
  // Only meaningful while hasActed is true: a player who has acted may raise
  // again only once the bet has grown by a full raise since then (a short
  // all-in alone doesn't reopen the betting for them).
  actedAt: number;
  // The player's most recent action on the current street (drives the seat
  // label: CALL / CHECK / FOLD / BET / RAISE / ALL-IN). Cleared each street.
  lastAction: ActionType | null;
}

export type ActionType = "fold" | "check" | "call" | "bet" | "raise" | "allin";
export interface Action {
  type: ActionType;
  amount?: number;
}

// One betting action in the current hand, in order (blind posts excluded).
// Structured so an AI encoder can read the hand's betting history.
export interface HandAction {
  seat: number;
  street: Street;
  // The concrete action taken ('allin' is normalized to call/bet/raise; see
  // the allin flag).
  type: "fold" | "check" | "call" | "bet" | "raise";
  allin: boolean; // the action put the player all-in
  to: number; // the player's street bet after the action
  added: number; // chips the action put in
  potBefore: number; // gathered pot + every street bet, before the action
  currentBetBefore: number; // street bet level to match, before the action
  stackBefore: number; // the actor's stack before the action
}

export interface LegalActions {
  actions: ActionType[];
  toCall: number;
  callTo: number;
  minRaiseTo: number;
  maxRaiseTo: number;
  canCheck: boolean;
}

export interface PotLayer {
  amount: number;
  eligible: number[];
}
export interface PotResult {
  amount: number;
  winners: number[];
  score: Score;
}
export interface WinnerRecord {
  seat: number;
  amountWon: number;
  handDescription: string;
}
export interface Results {
  pots: PotResult[];
  winners: WinnerRecord[];
  // At showdown the still-in players' cards are turned face up; a hand won by
  // everyone else folding stays concealed.
  reveal: boolean;
}

export interface EngineState {
  players: EnginePlayer[];
  button: number;
  sbSeat: number;
  bbSeat: number;
  handNumber: number;
  street: Street;
  community: Card[];
  deck: Card[];
  pot: number; // chips already gathered from earlier streets
  currentBet: number; // highest per-street bet to match
  minRaise: number; // smallest legal raise increment
  toAct: number;
  phase: EnginePhase;
  results: Results | null;
  log: string[];
  actions: HandAction[]; // this hand's betting actions, oldest first
}

// --- table / hand setup -----------------------------------------------------

function createPlayers(): EnginePlayer[] {
  return PLAYER_NAMES.map((name, i) => ({
    id: i,
    name,
    isHuman: i === 0,
    stack: STARTING_STACK,
    holeCards: [],
    status: "active",
    bet: 0,
    commit: 0,
    hasActed: false,
    actedAt: 0,
    lastAction: null,
  }));
}

// A seated table with no hand dealt — the Deal button starts hand 1.
export function createTable(): EngineState {
  return {
    players: createPlayers(),
    button: -1, // startHand advances it, so the first hand's button is seat 0
    sbSeat: -1,
    bbSeat: -1,
    handNumber: 0,
    street: "preflop",
    community: [],
    deck: [],
    pot: 0,
    currentBet: 0,
    minRaise: BIG_BLIND,
    toAct: -1,
    phase: "waiting",
    results: null,
    log: [],
    actions: [],
  };
}

// A table with the first hand already dealt (used by tests/headless play).
export function createGame(): EngineState {
  return startHand(createTable());
}

// --- seat-order helpers -----------------------------------------------------

function contenders(players: EnginePlayer[]): EnginePlayer[] {
  return players.filter((p) => p.stack > 0);
}

// Next seat index (clockwise) after `from` whose player passes `pred`.
// Returns -1 if none.
function nextSeat(
  players: EnginePlayer[],
  from: number,
  pred: (p: EnginePlayer) => boolean,
): number {
  for (let step = 1; step <= SEATS; step++) {
    const i = (from + step) % SEATS;
    if (pred(players[i])) return i;
  }
  return -1;
}

export function startHand(prev: EngineState): EngineState {
  const players: EnginePlayer[] = prev.players.map((p) => ({
    ...p,
    holeCards: [],
    bet: 0,
    commit: 0,
    hasActed: false,
    actedAt: 0,
    lastAction: null,
    // Anyone with chips comes back in; the busted sit out.
    status: p.stack > 0 ? "active" : "out",
  }));

  if (contenders(players).length < 2) {
    return { ...prev, players, phase: "match-over" };
  }

  // Heads-up: the button posts the small blind, the other seat the big blind.
  const button = nextSeat(players, prev.button, (p) => p.stack > 0);
  const sbSeat = button;
  const bbSeat = nextSeat(players, sbSeat, (p) => p.stack > 0);

  postBlind(players[sbSeat], SMALL_BLIND);
  postBlind(players[bbSeat], BIG_BLIND);

  // Deal two hole cards to each player. Deal order is cosmetic in a shuffled
  // deck, so we simply go seat by seat.
  const deck = shuffle(createDeck());
  for (const p of players) {
    if (p.status !== "out") {
      p.holeCards = [deck.pop()!, deck.pop()!];
    }
  }

  // The bet to match is the largest blind actually posted: a big blind who is
  // all-in for less than a full blind doesn't leave a phantom amount to call.
  const currentBet = Math.max(...players.map((p) => p.bet));

  // Pre-flop the small blind (the button) acts first — if anyone has a real
  // decision at all (a blind all-in that's already covered leaves none).
  const firstToAct = nextSeat(players, bbSeat, (p) =>
    owesAction(players, currentBet, p),
  );

  const log = [`--- Hand #${prev.handNumber + 1} ---`];
  log.push(`${players[sbSeat].name} posts SB ${SMALL_BLIND}`);
  log.push(`${players[bbSeat].name} posts BB ${BIG_BLIND}`);

  return {
    ...prev,
    players,
    button,
    sbSeat,
    bbSeat,
    handNumber: prev.handNumber + 1,
    street: "preflop",
    community: [],
    deck,
    pot: 0,
    currentBet,
    minRaise: BIG_BLIND,
    toAct: firstToAct,
    // Nobody has a decision (a blind all-in already covered): straight to the
    // run-out.
    phase: firstToAct === -1 ? "street-complete" : "betting",
    results: null,
    log,
    actions: [],
  };
}

// Post a blind, capping at the player's stack (short stacks post all-in).
function postBlind(player: EnginePlayer, amount: number): void {
  const posted = Math.min(amount, player.stack);
  player.stack -= posted;
  player.bet = posted;
  player.commit = posted;
  if (player.stack === 0) player.status = "allin";
}

// --- legal actions ----------------------------------------------------------

// Describes what the player to act may legally do. Bet/raise amounts are
// expressed as a *raise-to* total (the player's `bet` after the action), which
// keeps the caller's arithmetic simple.
export function legalActions(state: EngineState): LegalActions | null {
  const p = state.players[state.toAct];
  if (!p || p.status !== "active") return null;

  const toCall = state.currentBet - p.bet;
  const maxTo = p.bet + p.stack; // shove: commit the whole stack

  const actions: ActionType[] = [];

  // Fold is offered only when there's something to call, matching how a real
  // table plays (folding when checking is free is pointless).
  if (toCall > 0) actions.push("fold");

  if (toCall <= 0) actions.push("check");
  else actions.push("call");

  // Opening bet vs. raising an existing bet.
  const opening = state.currentBet === 0;
  // Smallest legal raise-to; capped by an all-in that can't reach it.
  const minTo = Math.min(
    opening ? BIG_BLIND : state.currentBet + state.minRaise,
    maxTo,
  );
  // A raise/bet is only possible if the player can put more in than a call,
  // the betting is still open to them, and the opponent can still respond
  // (raising an all-in player would only have the excess returned).
  if (
    maxTo > state.currentBet &&
    hasRaiseRights(state, p) &&
    opponentCanAct(state.players, p)
  ) {
    actions.push(opening ? "bet" : "raise");
  }

  return {
    actions,
    toCall: Math.min(toCall, p.stack),
    callTo: Math.min(state.currentBet, maxTo),
    minRaiseTo: minTo,
    maxRaiseTo: maxTo,
    canCheck: toCall <= 0,
  };
}

// Whether `p` may still bet/raise on this street. Someone who hasn't acted yet
// always may. Someone who has acted may raise again only once the bet has
// grown by at least a full raise since they acted: an all-in short of a full
// raise doesn't reopen the betting for them, though several short all-ins
// that add up to a full raise do. (A full raise also resets everyone else's
// hasActed, so they're covered by the first case.)
function hasRaiseRights(state: EngineState, p: EnginePlayer): boolean {
  return !p.hasActed || state.currentBet - p.actedAt >= state.minRaise;
}

// --- applying an action -----------------------------------------------------

// For bet/raise, `action.amount` is the raise-to total. 'allin' is shorthand
// for a bet/raise/call to the player's maximum.
export function applyAction(state: EngineState, action: Action): EngineState {
  const players = state.players.map((p) => ({ ...p }));
  const i = state.toAct;
  const p = players[i];
  const legal = legalActions(state);
  if (!legal) return state;

  let currentBet = state.currentBet;
  let minRaise = state.minRaise;
  const log = state.log.slice();

  const maxTo = p.bet + p.stack;
  const commit = (to: number): void => {
    const delta = to - p.bet;
    p.stack -= delta;
    p.commit += delta;
    p.bet = to;
    if (p.stack === 0) p.status = "allin";
  };

  const stackBefore = p.stack;
  const betBefore = p.bet;
  const potBefore = state.pot + state.players.reduce((s, q) => s + q.bet, 0);

  let type: ActionType = action.type;
  let amount = action.amount;
  // Normalize an 'allin' into the concrete action it represents.
  if (type === "allin") {
    amount = maxTo;
    if (maxTo <= currentBet) type = currentBet - p.bet > 0 ? "call" : "check";
    else type = currentBet === 0 ? "bet" : "raise";
  }
  // A bet/raise (or all-in for more) when the betting isn't open to this
  // player is just a call.
  const canRaise =
    legal.actions.includes("bet") || legal.actions.includes("raise");
  if ((type === "bet" || type === "raise") && !canRaise) {
    type = currentBet - p.bet > 0 ? "call" : "check";
  }

  switch (type) {
    case "fold":
      p.status = "folded";
      p.hasActed = true;
      log.push(`${p.name} folds`);
      break;

    case "check":
      p.hasActed = true;
      log.push(`${p.name} checks`);
      break;

    case "call": {
      const to = Math.min(currentBet, maxTo);
      const put = to - p.bet;
      commit(to);
      p.hasActed = true;
      log.push(`${p.name} calls ${put}${p.status === "allin" ? " (all in)" : ""}`);
      break;
    }

    case "bet":
    case "raise": {
      let to = amount ?? legal.minRaiseTo;
      // Clamp to the legal band. An all-in that can't reach a full raise is
      // still allowed (it just may not reopen the betting — see below).
      to = Math.max(Math.min(to, maxTo), Math.min(legal.minRaiseTo, maxTo));
      const raiseSize = to - currentBet;
      commit(to);
      p.hasActed = true;
      // A raise of at least a full min-raise reopens the action: everyone who
      // already acted must respond again. A short all-in does not.
      if (raiseSize >= minRaise) {
        minRaise = raiseSize;
        for (const other of players) {
          if (other.id !== p.id && other.status === "active")
            other.hasActed = false;
        }
      }
      currentBet = Math.max(currentBet, to);
      const verb = type === "bet" ? `bets ${to}` : `raises to ${to}`;
      log.push(`${p.name} ${verb}${p.status === "allin" ? " (all in)" : ""}`);
      break;
    }

    default:
      return state;
  }

  p.lastAction = p.status === "allin" ? "allin" : type;
  p.actedAt = currentBet;

  const actions = state.actions.slice();
  actions.push({
    seat: i,
    street: state.street,
    type,
    allin: p.status === "allin",
    to: p.bet,
    added: p.bet - betBefore,
    potBefore,
    currentBetBefore: state.currentBet,
    stackBefore,
  });

  const next: EngineState = {
    ...state,
    players,
    currentBet,
    minRaise,
    log,
    actions,
  };

  // Is the betting round finished?
  if (bettingClosed(next)) {
    return { ...next, phase: "street-complete" };
  }
  next.toAct = nextActor(next, i);
  next.phase = "betting";
  return next;
}

// Whether some other player could still respond to `p` (is in the hand with
// chips behind). Heads-up: the opponent is active, not all-in or folded.
function opponentCanAct(players: EnginePlayer[], p: EnginePlayer): boolean {
  return players.some((q) => q.id !== p.id && q.status === "active");
}

// Whether `p` has a real decision to make: they can act, and either owe chips
// to match the bet, or haven't acted yet while the opponent can still respond.
// (A player facing nothing, against an all-in opponent, has nothing to decide.)
function owesAction(
  players: EnginePlayer[],
  currentBet: number,
  p: EnginePlayer,
): boolean {
  if (p.status !== "active") return false;
  return p.bet < currentBet || (!p.hasActed && opponentCanAct(players, p));
}

// The betting round closes when no player still owes an action. Also closes
// the instant only one non-folded player remains.
function bettingClosed(state: EngineState): boolean {
  const inHand = state.players.filter(
    (p) => p.status === "active" || p.status === "allin",
  );
  if (inHand.length <= 1) return true;
  return !state.players.some((p) =>
    owesAction(state.players, state.currentBet, p),
  );
}

// Next seat after `from` that still owes an action.
function nextActor(state: EngineState, from: number): number {
  return nextSeat(state.players, from, (p) =>
    owesAction(state.players, state.currentBet, p),
  );
}

// --- moving between streets -------------------------------------------------

const NEXT_STREET: Record<Street, Street | undefined> = {
  preflop: "flop",
  flop: "turn",
  turn: "river",
  river: undefined,
};
const DEAL_COUNT: Record<Street, number> = {
  preflop: 0,
  flop: 3,
  turn: 1,
  river: 1,
};

// Called once the current street's betting is complete (phase
// 'street-complete'). Gathers bets into the pot, then either ends the hand or
// deals the next street. When two-plus players are all-in with nothing left to
// decide, it deals ONE street at a time and stays in 'street-complete' so the
// run-out can be revealed street by street.
export function advanceStreet(state: EngineState): EngineState {
  const players = state.players.map((p) => ({ ...p }));
  returnUncalled(players);
  // Sweep this street's bets into the central pot.
  let pot = state.pot;
  for (const p of players) {
    pot += p.bet;
    p.bet = 0;
    p.hasActed = false;
  }

  const notFolded = players.filter(
    (p) => p.status !== "folded" && p.status !== "out",
  );

  // Everyone folded to one player: they win without a showdown.
  if (notFolded.length === 1) {
    return settle({ ...state, players, pot }, false);
  }

  // River already dealt and its betting done -> showdown.
  if (state.street === "river") {
    return settle({ ...state, players, pot }, true);
  }

  // Deal the next street.
  const deck = state.deck.slice();
  const street = NEXT_STREET[state.street]!;
  // New street, fresh seat labels (a fold stays visible via the folded status).
  for (const p of players) p.lastAction = null;
  const dealt: Card[] = [];
  for (let k = 0; k < DEAL_COUNT[street]; k++) dealt.push(deck.pop()!);
  const community = [...state.community, ...dealt];

  const log = state.log.slice();
  log.push(
    `${street[0].toUpperCase() + street.slice(1)}: ${dealt
      .map((c) => c.id)
      .join(" ")}`,
  );

  const canAct = players.filter((p) => p.status === "active");
  const base: EngineState = {
    ...state,
    players,
    pot,
    deck,
    community,
    street,
    currentBet: 0,
    minRaise: BIG_BLIND,
    log,
  };

  // Fewer than two players can still act -> no betting; keep running out.
  if (canAct.length < 2) {
    return { ...base, toAct: -1, phase: "street-complete" };
  }

  return {
    ...base,
    toAct: nextSeat(players, state.button, (p) => p.status === "active"),
    phase: "betting",
  };
}

// An uncalled bet goes back to its owner before the street's bets are swept:
// the part of the biggest bet that nobody else matched (because they folded
// or were all-in for less) was never at risk.
function returnUncalled(players: EnginePlayer[]): void {
  const bets = players.map((p) => p.bet).sort((a, b) => b - a);
  const excess = bets[0] - (bets[1] ?? 0);
  if (excess <= 0) return;
  const p = players.find((q) => q.bet === bets[0])!;
  p.bet -= excess;
  p.commit -= excess;
  p.stack += excess;
  if (p.status === "allin") p.status = "active";
}

// --- showdown / settling the pot -------------------------------------------

// Build main + side pots from each player's total commitment. Returns
// [{ amount, eligible: [ids] }], smallest layer first.
export function buildPots(players: EnginePlayer[]): PotLayer[] {
  const contrib = players
    .filter((p) => p.commit > 0)
    .map((p) => ({ id: p.id, amt: p.commit, folded: p.status === "folded" }));

  const pots: (PotLayer & { key: string })[] = [];
  while (true) {
    const positive = contrib.filter((c) => c.amt > 0);
    if (positive.length === 0) break;
    const layer = Math.min(...positive.map((c) => c.amt));
    let amount = 0;
    for (const c of positive) {
      amount += layer;
      c.amt -= layer;
    }
    const eligible = positive.filter((c) => !c.folded).map((c) => c.id);
    // Merge into the previous pot if the eligible set is identical.
    const key = eligible.join(",");
    const last = pots[pots.length - 1];
    if (last && last.key === key) last.amount += amount;
    else pots.push({ amount, eligible, key });
  }
  return pots.map(({ amount, eligible }) => ({ amount, eligible }));
}

// Decide the hand, distribute chips, and set results/phase.
function settle(state: EngineState, reveal: boolean): EngineState {
  const players = state.players.map((p) => ({ ...p }));
  const pots = buildPots(players);

  // Best hand score per still-in player.
  const scores = new Map<number, Score>();
  for (const p of players) {
    if (p.status === "folded" || p.status === "out") continue;
    scores.set(p.id, evaluate7([...p.holeCards, ...state.community]));
  }

  const potResults: PotResult[] = [];
  const awarded = new Map<number, number>(); // seat -> chips won
  for (const pot of pots) {
    const eligible = pot.eligible.filter((id) => scores.has(id));
    if (eligible.length === 0) continue;
    let best: Score | null = null;
    let winners: number[] = [];
    for (const id of eligible) {
      const s = scores.get(id)!;
      const cmp = best ? compareScores(s, best) : 1;
      if (cmp > 0) {
        best = s;
        winners = [id];
      } else if (cmp === 0) {
        winners.push(id);
      }
    }
    // Split, awarding odd chips to the earliest seats left of the button.
    const leftOfButton = (id: number): number =>
      (id - state.button - 1 + SEATS) % SEATS;
    winners.sort((a, b) => leftOfButton(a) - leftOfButton(b));
    const share = Math.floor(pot.amount / winners.length);
    let remainder = pot.amount - share * winners.length;
    for (const id of winners) {
      let amt = share;
      if (remainder > 0) {
        amt += 1;
        remainder -= 1;
      }
      players[id].stack += amt;
      awarded.set(id, (awarded.get(id) ?? 0) + amt);
    }
    potResults.push({ amount: pot.amount, winners, score: best! });
  }

  const winners: WinnerRecord[] = [...awarded.entries()].map(
    ([seat, amountWon]) => ({
      seat,
      amountWon,
      handDescription: reveal
        ? categoryName(scores.get(seat)!)
        : "opponent folded",
    }),
  );

  const log = state.log.slice();
  for (const w of winners) {
    log.push(`${players[w.seat].name} wins ${w.amountWon}`);
  }

  // The match is decided once one player holds every chip.
  const matchOver = contenders(players).length < 2;
  return {
    ...state,
    players,
    pot: 0,
    toAct: -1,
    phase: matchOver ? "match-over" : "hand-over",
    results: { pots: potResults, winners, reveal },
    log,
  };
}

// The overall match winner (valid once phase === 'match-over').
export function matchWinner(state: EngineState): EnginePlayer | null {
  return state.players.find((p) => p.stack > 0) ?? null;
}
