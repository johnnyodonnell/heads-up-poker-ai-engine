# Heads-Up Poker — AI Engine

A heads-up (one-on-one) **No-Limit Texas Hold'em** game in the browser
(React 19 + TypeScript 7 + Vite 8) with a swappable AI engine. You take the
bottom seat; the bot sits across from you.

The bot is a **CFR engine**: a precomputed counterfactual-regret-minimization
strategy for heads-up no-limit, trained and served by the `asus-nvidia` box
(see [cfr/README.md](cfr/README.md)). The browser asks the engine for each of
the bot's moves (`engine/remote.ts` → `POST /api/act`); there is no fallback,
so if the engine can't be reached the table says so and offers a retry.

The engine module exposes a single `bestMove(state)` function, so the engine can
be swapped without touching the UI. This is the same pattern as the sibling
`6-max-poker-ai-engine`, which this project is based on (rules engine, table
UI, and adapter are ported from it and cut down to two seats).

The table is modelled on the CoinPoker client: a dark room, an oval green felt,
name and stack plates with action labels, card backs for the bot, chips in
front of each bettor, a pot label above the board, and action buttons at the
bottom right. Click **Deal** to start each hand; the bot's moves and street
run-outs resolve without artificial delay.

## Playing

The box serves the built app along with the engine:
**http://192.168.0.185:8790**. Desktop browsers only — the table is a
fixed-size stage that scales to fit the window.

The **Debug** toggle (top left) shows the bot's cards face up and, for each of
its decisions this hand, the strategy it used (stack depth), its hand group,
the options it was choosing between with their probabilities, and its pick.

## How strong is it?

The current strategy (v1d, served since 2026-11-22: a wider betting menu,
trained for 68 hours) beats all seven scripted test bots by 44 to 300
bb/100 and is ahead of v1c, the one before it, at every stack depth head
to head. Against an exploiter that knows its strategy and probes for weak
spots (local best response), v1d's table loses 146 ± 18 bb/100, against
208 ± 21 for v1c's; with the turn and river solved live, as served, v1d
loses 17 ± 37 and v1c 39 ± 36 (on the same deals; the gap is within the
noise). Against
[Slumbot](https://slumbot.com), a strong public research bot, v1c was
about break-even: **+8.6 ± 37.1 bb/100 over 10,000 hands**, played at
Slumbot's 200bb stacks, deeper than it was trained for.
Details and every number are in [Results](cfr/README.md#results) and
[Exploitability](cfr/README.md#exploitability-local-best-response).

## The trained strategy

The strategy the bot plays is too big for git (about 11.5 GB on disk with
its card abstraction), so it is saved in a GitHub Release instead:
[**`strategy-v1d`**](https://github.com/johnnyodonnell/heads-up-poker-ai-engine/releases/tag/strategy-v1d)
(served since 2026-11-22; the previous strategy,
[`strategy-v1c`](https://github.com/johnnyodonnell/heads-up-poker-ai-engine/releases/tag/strategy-v1c),
is kept too). To put it back on the box, for example after a disk loss,
without retraining (~68 h):

```sh
cfr/box.sh ./release_restore.sh strategy-v1d
cfr/box.sh 'echo v1d > $HU_HOME/current_run && pm2 restart hu-engine'
```

The restore takes a couple of minutes and checks every file against the
release's SHA-256 sums. See [Saved runs](cfr/README.md#saved-runs) for what
the release holds, restoring onto a fresh box, and saving a new run.

## Developing

```sh
npm install
npm run dev        # or: ./run-local.sh
```

The dev server forwards `/api` to the engine on the box
(`HU_ENGINE=http://host:port` to point elsewhere). `npm run build` type-checks
and produces a production build in `dist/`; `cfr/deploy_web.sh` builds and
copies it to the box.

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
    remote.ts           the AI — bestMove(state): the CFR engine on the box
    random.ts           a random legal action (the original placeholder bot)
    adapt.ts            maps engine state <-> the UI's GameState / BetAction
  components/
    layout.ts           stage size and every seat/bet/dealer/pot position
    GameTable           scaled stage; composes everything below
    PokerTable          rail + felt
    PlayerSeat          plate, hole cards, action label
    Card, CommunityCards, PotDisplay, ChipStack, DealerButton,
    HandResult, ActionPanel
    DebugPanel          the Debug toggle and the bot's decisions this hand
cfr/                    the engine: Rust, trained and served on the box
```

`GameTable`'s `faceUp` prop shows the bot's hole cards; the Debug toggle sets it.

## The engine contract

`hooks/useGameState.ts` imports `bestMove` from exactly one engine file. To plug
in a different engine, change that one line:

```ts
import { bestMove } from "../engine/remote";   // current: the CFR engine
// import { bestMove } from "../engine/random";   // the random placeholder
```

```ts
await bestMove(state)   // state: EngineState (see engine/game.ts)
                        // -> an Action { type, amount? } for state.toAct
```

The hook `await`s each move, so both synchronous engines (like `random.ts`) and
async ones (like the remote engine) work; a rejected move stops the table with
the error and a Retry button.

Engine actions are `{ type: 'fold' | 'check' | 'call' | 'bet' | 'raise' | 'allin' }`,
where `amount` (for bet/raise) is the *raise-to* total. `legalActions(state)`
returns exactly what's permitted for the player to act, including the legal
bet/raise band — a bot need only choose among those. `state.actions` holds the
hand's betting history in order, for engines that want it.
