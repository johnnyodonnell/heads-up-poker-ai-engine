# Heads-Up Poker — AI Engine

A heads-up (one-on-one) **No-Limit Texas Hold'em** game in the browser
(React 19 + TypeScript 7 + Vite 8) with a swappable AI engine. You take the
bottom seat; the bot sits across from you.

For now the bot is a **random engine** (`engine/random.ts`): it picks a
uniform-random legal action and, for a bet or raise, a uniform-random legal
size. It is a placeholder until a trained model replaces it: a neural engine
(`engine/neural.ts`) trained by self-play is in progress — see
[training/README.md](training/README.md) — and takes over once a trained
snapshot is pulled into `public/models/current.onnx`.

The engine module exposes a single `bestMove(state)` function, so the engine can
be swapped without touching the UI. This is the same pattern as the sibling
`6-max-poker-ai-engine`, which this project is based on (rules engine, table
UI, and adapter are ported from it and cut down to two seats).

The table is modelled on the CoinPoker client: a dark room, an oval green felt,
name and stack plates with action labels, card backs for the bot, chips in
front of each bettor, a pot label above the board, and action buttons at the
bottom right. Click **Deal** to start each hand; the bot's moves and street
run-outs resolve without artificial delay.

## Running locally

```sh
npm install
npm run dev        # or: ./run-local.sh
```

Then open the printed URL. `npm run build` type-checks and produces a production
build in `dist/`. Desktop browsers only — the table is a fixed-size stage that
scales to fit the window.

## Rules / setup

- **Table**: 2 seats (You, Bot), blinds **1 / 2**, both start with **200**
  (100 bb).
- **Play**: standard NLHE — pre-flop, flop, turn, river, showdown. Fold, check,
  call, bet, raise, all-in; min-raise rules enforced.
- **Heads-up blinds**: the button posts the small blind and acts first
  pre-flop; the big blind acts first on the flop, turn and river. The button
  alternates every hand.
- **Uncalled bets** are returned to the bettor (so there are never side pots).
- **Freeze-out**: the match ends when one player holds every chip — hit
  **New Game** to start over.
- **Short all-ins**: an all-in raise smaller than a full raise doesn't reopen
  the betting to a player who already acted (they may only call or fold).
- **Only real decisions**: you can't raise against a player who is already
  all-in (only call or fold), and a blind posted all-in that's already covered
  goes straight to the run-out.

## Training

The neural bot is trained by search-free self-play REINFORCE, all in Rust
(`training/`), on a GPU box; see [training/README.md](training/README.md).

## Project layout

```
src/
  main.tsx              entry — mounts <App>
  App.tsx               wires the useGameState hook to <GameTable>
  App.css               global reset + fonts
  types.ts              UI-facing GameState/Player/Card/BetAction contract
  money.ts              chip -> "$" formatting
  hooks/
    useGameState.ts     holds engine state; runs the bot + streets to the next
                        human decision
  engine/
    cards.ts            52-card deck, shuffle, card formatting
    evaluator.ts        best-5-of-7 hand ranking + comparison
    game.ts             pure heads-up NLHE rules — no React, no side effects
    random.ts           the AI — bestMove(state): a random legal action
    neural.ts           the trained AI (not wired in yet) — encode, run the
                        net, sample a move
    net.ts              onnxruntime-web session for public/models/current.onnx
    encode.ts           the network's input encoding (Rust mirror in training/)
    actions.ts          the action space: four types + a continuous raise size
    adapt.ts            maps engine state <-> the UI's GameState / BetAction
  components/
    layout.ts           stage size and every seat/bet/dealer/pot position
    GameTable           scaled stage; composes everything below
    PokerTable          rail + felt
    PlayerSeat          plate, hole cards, action label
    Card, CommunityCards, PotDisplay, ChipStack, DealerButton,
    HandResult, ActionPanel
```

`GameTable` accepts a `faceUp` prop (default `false`) that shows the bot's hole
cards; nothing sets it yet.

## The engine contract

`hooks/useGameState.ts` imports `bestMove` from exactly one engine file. To plug
in a different engine, change that one line:

```ts
import { bestMove } from "../engine/random";   // current
// import { bestMove } from "../engine/neural";   // once a model is pulled in
```

```ts
await bestMove(state)   // state: EngineState (see engine/game.ts)
                        // -> an Action { type, amount? } for state.toAct
```

The hook `await`s each move, so both synchronous engines (like `random.ts`) and
async ones (e.g. a neural net running in the browser) work.

Engine actions are `{ type: 'fold' | 'check' | 'call' | 'bet' | 'raise' | 'allin' }`,
where `amount` (for bet/raise) is the *raise-to* total. `legalActions(state)`
returns exactly what's permitted for the player to act, including the legal
bet/raise band — a bot need only choose among those. `state.actions` holds the
hand's betting history in order, for engines that want it.
