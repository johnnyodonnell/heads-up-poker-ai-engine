# CFR engine

The bot's engine: a precomputed CFR strategy for heads-up no-limit hold'em,
trained and served on the box (`asus-nvidia`). Everything here is Rust and is
built and run on the box; this machine only edits code, dumps the TS parity
fixtures and builds the web app.

How it plays: for each decision the engine replays the hand through the rules
port, picks the strategy trained nearest the hand's effective stack, maps the
real betting onto its abstract betting tree (off-menu bet sizes via the
pseudo-harmonic mapping), looks up the bot's hand group, and samples from the
strategy at that spot, turning the chosen pot fraction back into legal chips.

## Layout

```
hu_core/          the game: a port of src/engine/game.ts (rules), cards, a 7-card
                  hand evaluator, equity and all-in EV
hu_cfr/           the engine:
  abs/              card abstraction (hand groups): dense board-major tables,
                    exact per-board equities, k-means with earth mover's distance
  tree.rs, menu.rs  the abstract betting tree, built from the rules port
  mccfr.rs          external-sampling MCCFR (Pluribus-style schedule)
  strategy.rs       exported strategies (1 byte per action)
  engine.rs         play time: replay, translation, action mapping, the
                    fallback where the tree can't follow the hand
  eval.rs, bots.rs  matches against scripted bots (duplicate, paired deals)
  lbr.rs            local best response: an exploiter that knows the bot's
                    strategy, for a lower bound on its exploitability
  score.rs          advantage scoring: live solving's edge over the table,
                    scored decision by decision
  live/             live solving: subgame tree, vectorized DCFR, showdowns,
                    safe re-solving against the table's values, the GPU path
  cuda/river.cu     the GPU kernel for turn solves' river subtrees
  pushfold.rs, toy.rs, br.rs   solver validation games and exact best response
hu/               the `hu` command line (abstraction, train, export, eval, compare,
                  probe, livebench, rangecheck, scorecheck, turncheck, lbr, serve,
                  tree, pushfold)
scripts/          TS run with tsx: rule scenario checks, parity fixture dumper
fixtures/         parity fixtures dumped from the TS engine (gitignored)
box_env.sh        environment for the box (sourced by everything below)
box.sh            run a command on the box inside repo/cfr
box_bg.sh         start a long job on the box, detached, logging to $HU_HOME/logs
sync_to_box.sh    rsync this working tree to the box
train_all.sh      train a run's stack depths one after another
serve.sh, pm2.config.cjs   the engine service under pm2
deploy_web.sh     build the web app and copy it to the box
panel.sh, panel_report.sh   the live-solving test panel and its paired results
```

## The box

Everything for this project on the box lives under
`~/Workspace/heads-up-poker-ai-engine/` (`$HU_HOME`): `repo/` (this repo),
`cargo-home/` and `tmp/` (Cargo's cache and temp files), `data/` (hand-group
tables), `runs/` (training checkpoints), `strategies/` (exported strategies),
`logs/`, and `web/` (the built app). Only pm2's own bookkeeping is in `~/.pm2`.

```sh
cfr/sync_to_box.sh                          # copy the working tree to the box
cfr/box.sh cargo build --release            # build there
```

## Pipeline

```sh
# 1. hand groups: [169 preflop classes, flop, turn, river] buckets
cfr/box_bg.sh abs-a2k ./target/release/hu abstraction --name a2k --buckets 169,2000,2000,1000
# 2. train every stack depth of a run (iterations given for 100bb)
cfr/box_bg.sh train-v1b ./train_all.sh v1b a2k 400000000
cfr/box.sh tail -n 3 '$HU_HOME/logs/train-v1b.log'
# 3. check it
cfr/box.sh ./target/release/hu eval --run v1b                 # vs the scripted bots
cfr/box.sh ./target/release/hu probe --run v1b                # preflop charts
# 4. serve it
cfr/box.sh 'echo v1b > $HU_HOME/current_run && pm2 restart hu-engine'
```

The service (port 8790) is started once with
`pm2 start cfr/pm2.config.cjs && pm2 save` from `repo/`; `cfr/deploy_web.sh`
updates the web app it serves. `GET /api/health` shows what's loaded.

Logs of play in the app, in `$HU_HOME/logs/`:

| File | One line per | Holds |
|---|---|---|
| `engine.log` | decision (text) | the bot's cards, board, choice, any fallback |
| `decisions.jsonl` | decision | the app's request (the hand so far at its real sizes) and the engine's debug output: how the real betting was translated, the options and probabilities, any fallback, live-solve stats |
| `hands.jsonl` | finished hand | both players' cards and stacks, the board, every action, the result (the app sends it with `POST /api/hand` when the hand ends) |

Hand ids are `PAGELOAD-GAME-HAND` (a random id per page load, the game's
count since then, the hand number).

## Results

All in bb/100 from the engine's side, ± a 95% interval.

### Runs

| Run | Abstraction (flop/turn/river groups) | Iterations at 100bb | Training time | Served |
|---|---|---|---|---|
| v1a (checkpoint A) | a200: 200/200/200 | 300M, 100bb only | 12 min | 2026-11-16 |
| v1b (checkpoint B) | a2k: 2000/2000/1000 | 2B | 3.6 h | 2026-11-16 |
| v1c (checkpoint C) | a10k: 10000/5000/2000 | 5B | 9.3 h | since 2026-11-17 |
| v1d (menu `wide-river`) | a10k | 23.1B (scaled to the tree: `MENU=wide-river SCALE=tree train_all.sh v1d a10k 5e9`) | training since 2026-11-19 20:26, estimated ~66 h | |

### Against the scripted bots

`hu eval --run RUN --stacks sampled`: 40,000 hands per bot (20,000 deals
played from both seats), stack depths sampled the way the app's freeze-out
reaches them. Intervals are ±5 to ±24.

| Bot | v1a | v1b | v1c |
|---|---|---|---|
| random (the app's original placeholder bot) | +135 | +150 | +145 |
| station (always checks or calls) | +180 | +198 | +203 |
| maniac (pot-size bet or raise whenever allowed) | +138 | +222 | +215 |
| nit (premium hands only; continues with top pair or better) | +44 | +45 | +41 |
| shover (all-in preflop with any two cards) | +96 | +112 | +91 |
| minbet (min-bets and min-raises) | +315 | +369 | +370 |
| overbet (bets or raises 3× pot) | −8 | +45 | +39 |

### Head to head

`hu eval --run v1c --vs run:v1b,run:v1a --deals 200000`: 400,000 hands.

| Match | Sampled stacks | Full 100bb stacks |
|---|---|---|
| v1c vs v1b | +1.0 ± 2.6 | +3.9 ± 3.0 |
| v1c vs v1a | +10.9 ± 2.6 | +7.8 ± 3.1 |

### Against Slumbot

2026-11-17, through Slumbot's public API (the `hu slumbot` client has since
been removed: its rate limits made long runs unreliable, and testing now uses
only bots built here). [Slumbot](https://slumbot.com) is a strong public
research bot. It plays
200bb stacks, deeper than v1c's deepest strategy (100bb), so v1c plays it
with its 100bb strategy.

| Measure | v1c, 10,000 hands |
|---|---|
| Won | **+8.6 ± 37.1** |
| Won, minus what Slumbot wins with the same cards in the same seat | **+18.2 ± 32.5** |
| On the button / in the big blind | +53.1 ± 52.5 / −35.9 ± 52.5 |
| The 64 hands that went to full 200bb stacks | −2,000bb net (about −20 bb/100 overall) |
| Fallback decisions | 53 of 28,655 (0.2%) |

That is break-even with Slumbot within the noise. The clearest leak is in
hands played for the full 200bb: the 100bb strategy's all-in is half the
size of the real one, so the bot gets its whole stack in too easily. A
strategy trained at 200bb (a new deepest depth) would target that; it does
not affect the app, whose matches start at 100bb. The hand logs are kept in
`$HU_HOME/logs/slumbot-*.jsonl`.

## Live solving (v2)

With `--live river` or `--live turn` (on `hu eval`, `lbr`, `serve`,
`livebench`), the engine solves turn and river decisions live instead of
looking them up; preflop and the flop stay on the table. Off by default in
the CLI. The app is served with live turn since 2026-11-19 (the turn and the
river solved live, about 1.4 s per turn decision on the box): it matched
live river against our fixed opponents (test round 2, below) but is far
harder to exploit (local best response wins 94 ± 35 bb/100 less against it;
"Exploitability", below). It was served with live river earlier that day.
`$HU_HOME/live_mode` holds the mode (`serve.sh` reads it), and
`GET /api/health` shows it:

```sh
cfr/box.sh 'echo turn > $HU_HOME/live_mode && pm2 restart hu-engine'
```

How a live decision is made (`hu_cfr/src/live/`):

- **The subgame** starts at the beginning of the current street with the
  real pot and stacks. The street's real actions so far are in the tree at
  their real sizes (off-menu sizes are added), so nothing is translated, and
  every one of the 1,326 hands is exact (no hand groups). A turn solve deals
  all 48 river cards. The bot's own earlier actions on the street are frozen
  to the strategy it actually played.
- **The solver** is Discounted CFR (α 1.5, β 0, γ 2), full-width and
  vectorized over all hands of both players, with exact card removal. River
  solves run 400 iterations (about 0.5 s), turn solves 150 (about 1 s with
  the GPU, below; 2 s on the box's 20 CPU cores). Solves are deterministic,
  and cached per hand so later decisions reuse earlier ones.
- **Ranges.** The bot's range is exact (its own strategy). The opponent's is
  estimated from the table: how likely each hand was to play the hand this
  way if the opponent played like our table.
- **Safe re-solving** (default, `--safe best`). Trusting that estimate
  failed against Slumbot (below): an opponent that doesn't play like our
  table often holds hands the estimate rules out, and a re-solve is only an
  equilibrium of the game it was given, so it can be badly beaten by those
  hands. Safe solving gives every opponent hand the choice between playing
  the solve and what it would get against the table from the start of the
  street (`live/safe.rs`: the table's subtree walked over all hands against
  the bot's exact range, the opponent best-responding). The solver learns
  that choice per hand along with its own strategy, so no hand the opponent
  could hold does better than it would against the table. The opponent's
  root range is the estimate plus twice the hands that would beat their
  table value (`--safe-weight 2`): pure Resolve (`--safe-weight resolve`)
  stops learning when almost no hand wants to enter. `--safe off` is the
  unsafe solver.

Checking it:

```sh
cargo test --release -p hu_cfr live         # solver unit tests (below)
hu livebench --run v1c --street turn --spots 20 --iters 150,300   # time and convergence on real spots
hu livebench --run v1c --street river --spots 40 --iters 400 --check-safe   # safe vs unsafe solves
hu livebench --run v1c --street turn --spots 10 --iters 150 --compare-gpu   # GPU vs CPU solves (identical)
hu rangecheck --run v1c --opp-run v1a       # how well the table's range estimates fit another run's play
hu scorecheck --run v1c --opp-run v1a --spots 3000   # checks of advantage scoring on real spots
hu scorecheck --run v1c --opp-run v1a --stack 200 --spots 300 --play turn   # ... on spots of live turn play
hu scorecheck --run v1c --opp-run v1a --stack 200 --spots 40 --live turn    # the scored identity, exactly
hu turncheck --run v1c --opp-run v1c,v1a --stack 200 --spots 300 --configs "default;iters=600"   # live vs table per decision, per setting
hu compare A.txt B.txt                      # paired difference of two saved matches
./panel.sh; ./panel_report.sh               # the test panel (round 2, below) and its results
```

The unit tests check the solver against a polarized river toy with a known
equilibrium, showdown and fold payoffs against brute force, convergence and
determinism of river and turn solves, off-menu actions, and that safe
solving holds hands an estimate left out to their alternatives while unsafe
solving doesn't (0.34% vs 12.9% of the pot).

### The GPU

Turn solves run their river half on the box's GPU (`--gpu on`, the default;
`hu_cfr/cuda/river.cu`, compiled by `hu_cfr/build.rs` with the `gpu`
feature, on by default in `hu`; `--no-default-features` builds without
CUDA). A turn solve's traversal walks a river subtree for every (turn line,
river card) pair, about 900 of them for a typical spot; on the GPU each is
one thread block, all at once, with their regrets kept in GPU memory for
the whole solve, while the turn's own nodes stay on the CPU.

The kernel does the CPU's arithmetic operation for operation (no fused
multiply-adds, sums in the same order), so GPU and CPU solves are
bit-identical: `hu livebench --street turn --compare-gpu` solves spots both
ways and reports the largest strategy difference, which is 0. To make that
possible the CPU's long sums at showdowns and folds now run in a fixed
chunked order that the GPU can take in parallel (`showdown::chunked_prefix`);
this changed the solver's results only in the last bits.

On real 200bb turn spots a solve takes 1.0 s with the GPU against 2.0 s on
all 20 CPU cores (same strategies, same exploitability). In a batch
evaluation, 400 live-turn hands against v1c took:

| Turn solves | Time | Turn solves on the GPU |
|---|---|---|
| CPU only | 293 s | 0 of 197 |
| GPU, 2 at a time | 161 s | 111 |
| GPU, 4 at a time (default) | **124 s** (2.4× faster) | 132 |
| GPU, all of them | 151 s | 197 |

with the same result deal by deal in all four.

The box's CPU and GPU share one memory (LPDDR5x, about 273 GB/s), and turn
solves are limited by memory traffic on both, so the GPU's edge is bounded.
`--gpu-slots N` (default 4) caps the turn solves on the GPU at once; a solve
that finds the GPU busy runs on the CPU, with the same result, so a batch
evaluation uses both.

### Why safe, and the Slumbot runs

The first version was unsafe: it trusted the estimated opponent range.
Against Slumbot (10,000 hands each, minus what Slumbot wins with the same
cards, ± about 31) it did no better than the table: table +18.2, live river
+17.5, live turn −13.4. Slumbot shows its cards after every hand, so a
diagnostic (since removed with the Slumbot client) could check the
estimates against the hands it really held: they were badly overconfident, predicting those hands worse than
"any hand" would (−2.8 nats per hand at the start of the river). The
estimates are now hedged and fit to those logs (`LiveCfg`: +0.33 nats per
hand, calibrated), and safe re-solving makes the solve hold up when the
estimate is still wrong. Slumbot itself is too noisy to measure a few
bb/100 in a day of play, and its rate limits made long runs unreliable, so
it is no longer part of testing: the engine is tested only against bots
built here (the earlier tables and the scripted bots).

### The test panel (round 1)

Round 1 played the table, safe live and unsafe live (`--safe off`) against
the earlier tables (v1c itself, v1b, v1a) and the scripted bots, at 200bb
(Slumbot's depth) and 100bb (`panel.sh` now runs round 2, below). Every
match is saved deal by deal, and the deals and the opponent's random draws
depend only on the seed, so two variants are compared on identical deals
(`hu compare`): only their decisions differ. Gates, set before the runs:
safe live no worse than the table against any opponent, better against the
other tables, and still beating every scripted bot by a wide margin.

River, safe live minus the table on the same 50,000 duplicate deals
(100,000 hands), bb/100:

| Opponent | 200bb | 100bb |
|---|---|---|
| v1c (the table itself) | **+7.7 ± 6.0** | **+4.7 ± 3.9** |
| v1b | +7.1 ± 7.8 | +3.8 ± 4.6 |
| v1a | +5.9 ± 7.8 | +0.4 ± 4.4 |

Unsafe live minus the table: +5.6, +2.3, +2.4 at 200bb and +3.8, +3.3,
−0.4 at 100bb; safe was ahead of unsafe in all six.

Against the scripted bots at 200bb (10,000 deals each), the table's result
and safe live minus the table:

| Bot | Table | Safe live − table |
|---|---|---|
| random | +377 | +63 ± 26 |
| nit | +47 | +6 ± 8 |
| maniac | +947 | −26 ± 25 |
| station | +276 | −77 ± 28 |
| minbet | +1,224 | −389 ± 48 |
| shover, overbet | +384, +468 | 0 (the hands end before the river) |

Live still beats every bot by a wide margin, but wins less than the table
from the two that never fold. Traced decision by decision: the table bets
its strongest river hands far bigger (2.2× the pot against station, raises
of ~6× the pot over a min bet), because at 200bb it plays its 100bb strategy
and its all-in is the whole 200bb. That pays against a bot that never folds,
and is the same leak that lost on deep all-ins against Slumbot. The live
solve plays an equilibrium against an opponent who can fold. Offering it a
2½-pot bet changed nothing; winning more from these bots would take
opponent modelling, which this engine doesn't do.

Pooled over the four matches against v1b and v1a (both depths), safe live
river gains +4.3 ± 3.2 bb/100 over the table; over all six, +4.9 ± 2.4.

Turn (`--live turn`: the turn and the river solved live), safe live minus
the table at 200bb. A turn solve deals all 48 river cards and takes ~2 s on
the whole box, so a match plays ~1.6 hands/s and the samples are small:

| Opponent | Deals | Table | Safe live turn − table |
|---|---|---|---|
| v1c | 8,000 | 0 | +2.0 ± 19.6 |
| v1b | 8,000 | −9.7 | +21.1 ± 23.2 |
| v1a | 8,000 | +14.6 | −14.4 ± 24.6 |
| maniac | 2,000 | +833 | −107 ± 96 |
| minbet | 2,000 | +1,180 | −430 ± 127 |

No gain over the table shows at this size (+2.9 ± 13.0 pooled over the
three tables), and the turn adds the river's gap against the bots that
never fold. Live turn is not worth its 2 s per decision yet; live river is.

### Scoring live solving against the table

The panel's paired results are noisy because most of a hand's outcome is
luck that live solving doesn't change: the cards still to come, the hand
the opponent happens to hold, which of its mixed actions each player
happened to pick. Advantage scoring (`hu_cfr/src/score.rs`) removes that
luck. Against a table opponent every strategy is known exactly at every
decision: the live solve's, the table's (what the bot would have played
instead), and the opponent's. So each live decision is scored by its
advantage over the table:

    Σ over real actions a of (live probability of a − table probability of a) × Q(a)

where Q(a) is what the bot expects after taking a with both players playing
their tables from there: the real game tree walked exactly (the same
translation, rounding and fallback as in play, every river card, showdowns
and all-ins scored exactly), the opponent's hand weighted by its exact
range (how likely each hand was to play the hand so far as its table
plays). Summed over a hand's live decisions, this has exactly the same
expectation as the bot's result playing live minus its result playing the
table against this opponent (the performance difference lemma, Kakade &
Langford 2002; a form of AIVAT, Burch et al. 2018). No table match is
needed.

`hu eval` prints it for live matches against a table ("scored live -
table") and with `--save` writes it per deal (`.adv`). `.advfull` scores
against the opponent's real hand instead of its range: no range involved,
noisier, and a check on the range. The equity fallback is deterministic
(exact equity on the turn and river) so that every decision's
probabilities are known.

Checks. `hu scorecheck` on 3,000 real spots (v1c against v1a): each Q
equals an independent exact walk through the served decision path and the
rules engine's own payouts (to 1e-14); each range equals the product of the
opponent's decision probabilities (to 3e-16); the opponent's real hand falls
evenly through its range (9–11% of spots in each tenth). On matches at
100bb, 50,000 deals each, scored and paired results agree, also on a big
effect (a deliberately weak live river, one solver iteration):

| Live river − table, bb/100 | vs v1c | vs v1a |
|---|---|---|
| Paired (live match minus table match) | +4.7 ± 3.9 | +0.4 ± 4.4 |
| Scored (opponent's range) | **+2.3 ± 1.4** | **−0.2 ± 1.4** |
| Scored (opponent's real hand) | +1.9 ± 3.1 | −0.5 ± 3.1 |
| One iteration, paired | −67.1 ± 7.6 | −78.5 ± 7.9 |
| One iteration, scored | −71.4 ± 2.8 | −77.3 ± 2.9 |

Deal by deal, paired minus scored is consistent with zero in all eight
comparisons. The scored error bars are 2.7–3.1× narrower: 7–10× fewer deals
for the same precision.

At a larger scale (a 10-iteration live river, five fresh seeds of 400,000
deals against v1c, two million deals): paired −9.11 ± 0.62, scored −8.66 ±
0.22, a difference of −0.45 ± 0.62. One seed alone showed a −2.0 ± 1.4 gap,
which washed out over the others. And `hu scorecheck --live river` checks
the identity itself on real spots, with the exact expected result of
playing live (every later live solve and every opponent reply walked) minus
playing the table: it equals the exact expected scored sum to 1e-14 on
2,400 spots, both seats, at 100bb and 200bb.

The same on live turn play. `hu scorecheck --play turn` takes its spots
from hands where the bot solves the turn live, so they include the states
only live play reaches, after its own off-table bets (300 spots against
v1a, 85 of them river decisions after a live turn): Q equals the exact walk
to 7e-15, the range equals the opponent's likelihood hand by hand to 3e-16,
and the real hand falls evenly through it (7–13% of spots per tenth).
And `hu scorecheck --live turn` checks the identity at the bot's first turn
decision, every river card dealt and every later live solve walked: exact
to 1e-14 on 40 spots against v1a, both seats.

### Test round 2

Set before running (`panel.sh`, `panel_report.sh`; results under
`$HU_HOME/evals/round2`). The question: does live solving plus the table
beat the table alone?

- **Opponents:** v1c (the table itself: its own range estimates fit it,
  the best case for live solving) and v1a (the earlier table whose play
  fits v1c's model worst, as Slumbot's did). v1b, in between, is dropped.
  The scripted bots are a sanity check.
- **Depth:** 100bb, the app's.
- **Measure:** scored live − table (opponent's range), with the paired
  matches as a cross-check. Live river: 50,000 deals per opponent (about
  ±1.4 bb/100 each, ±1 pooled). Live turn: 12,000 deals per opponent to
  start, extended (`TURN_FROM`) if it misses ±3 bb/100; compared with live
  river on the same deals.
- **Gates for live river vs the table** (and the same for live turn vs
  live river): the gain pooled over v1c and v1a has a 95% interval above
  zero; against neither opponent does the 95% interval lie entirely below
  zero; and it still beats every scripted bot (95% interval above zero).

**Live river passes all three.** Live river minus the table, scored, 50,000
deals per opponent (100,000 hands), bb/100:

| Opponent | Scored | Paired (cross-check) |
|---|---|---|
| v1c | **+3.3 ± 1.3** | +3.1 ± 3.7 |
| v1a | **+1.5 ± 1.4** | −0.6 ± 4.5 |
| Pooled | **+2.4 ± 1.0** | |

Both intervals lie above zero, and so does the pooled one: live solving of
the river plus the table beats the table alone. (The validation runs, on
other deals and before the GPU change, gave +2.3 ± 1.4 and −0.2 ± 1.4.)

Against the scripted bots (5,000 deals each), live river still wins big
against all of them, and on the same deals compared with the table:

| Bot | Table | Live river | Live − table |
|---|---|---|---|
| random | +189 | +258 | +70 ± 24 |
| maniac | +278 | +339 | +61 ± 25 |
| nit | +37 | +38 | +1 ± 4 |
| station | +206 | +190 | −16 ± 20 |
| minbet | +465 | +372 | −92 ± 31 |
| shover, overbet | +136, +63 | the same | 0 (hands end before the river) |

As in round 1, it gives up some of the table's profit against the bot that
never folds (minbet), and wins more from the ones that bet with anything.

**Live turn does not pass.** Live turn solves the turn as well as the
river. On the same deals as live river, 16,000 per opponent (32,000 hands
each), scored, bb/100:

| Opponent | Live turn − table | Live turn − live river |
|---|---|---|
| v1c | +2.3 ± 2.5 | −1.1 ± 2.6 |
| v1a | +1.4 ± 2.8 | −1.8 ± 2.9 |
| Pooled | +1.9 ± 1.9 | **−1.5 ± 1.9** |

Solving the turn too gains nothing measurable over solving the river alone:
the pooled interval against live river doesn't lie above zero, so the first
gate fails. The other two pass: against neither opponent is it clearly
worse than live river, and it beats the three scripted bots it played
(1,000 deals each, the ones where the river panel showed a gap): station
+160 ± 54, maniac +387 ± 135, minbet +381 ± 116. The first 12,000 deals per
opponent missed ±3 bb/100, so the turn was extended by 4,000
(`TURN_FROM=12000`), as set.

The cross-checks lean the other way, but don't clear zero either: live turn
− live river is +2.9 ± 4.4 scored on the opponent's real hand and +4.7 ±
6.5 in chips (pooled, same deals). After the first 12,000 deals the two
scored versions differed by +8 ± 5 bb/100 (about three standard errors;
on the river they agree), so the scoring was checked again on the spots only
live turn play reaches (see "Scoring live solving against the table"): it is
exact there. On the next 4,000 deals per opponent the difference went the
other way, and over all 16,000 it is +4.7 ± 4.4. It is the real-hand
version's luck with the opponent's hands, which the range version doesn't
have.

So live river is the candidate to serve. Live turn costs a turn solve per
turn decision (about a second on the GPU) for no measurable gain.

### Why live turn adds nothing here

Checked one cause at a time with `hu turncheck`. It scores the bot's live
decision at real spots against the table's, with the same exact score as
`hu eval` (the table playing on from there), for several solver settings on
the same spots. The comparisons are paired, so hundreds of spots do what
tens of thousands of hands do in a match.

| The bot's first turn decision, bb/100 against the default solve (600 spots) | Change |
|---|---|
| 4× the iterations (exploitability 1.9% → 0.8% of the pot; 200 spots) | −0.2 ± 1.2 |
| Opponent hedges (fit to Slumbot) off (200 spots) | −0.5 ± 2.1 |
| Unsafe solving (200 spots) | −0.0 ± 1.1 |
| Pot raises in the river inside turn solves | +0.0 ± 0.2 |
| The full river menu inside turn solves (3× slower) | −0.1 ± 0.3 |
| The full turn menu | +0.5 ± 0.9 |

None of them matters. Nor is the turn solver wrong:
`turn_best_responses_match_the_rules_engine` computes each hand's best
response in turn solves (CPU and GPU) and by an independent walk through the
rules engine (every action played, every river card dealt, every hand
settled by `eval7`), and they agree. And the river after a live turn is
played as well as in live river (+2.7 ± 4.3 bb/100 at the bot's first river
decision; taking the opponent's river range from the opponent model instead
of the turn solve changes it by −1.5 ± 3.6).

What differs is how much there is to gain. Against these opponents, a live
solve wins where its exact hands beat the table's hand groups, mostly when
it answers the opponent's action. Per decision, in chips:

| | Acting first | After the opponent acted |
|---|---|---|
| River (800 spots) | −0.13 ± 0.19 | +0.46 ± 0.50 |
| Turn (800 spots) | −0.01 ± 0.10 | +0.12 ± 0.25 |

A river decision is a showdown decision, where 2,000 hand groups blur exact
strength the most: live and table differ by more than a chip at 27% of river
spots and 9% of turn spots. On the turn one card is still to come, and the
table's 5,000 groups, built from equity distributions, play close to the
solve. Answering sizes off the table's menu exactly doesn't change that:
against a v1c that bets 1.35× its sizes after the flop (`run:v1c@1.35`),
which live play answers at their real size and the table translates, live
turn decisions gain −3.1 ± 3.5 bb/100 over the table's (v1c: +0.8 ± 3.4).
Live river, by contrast, does gain from them: against v1c@1.35 it beats the
table by +6.2 ± 2.0 bb/100 (30,000 deals, scored), twice its +3.3 ± 1.3
against v1c. In a full match against v1c@1.35, where the river decisions
after an exact turn count too, live turn − live river on the same 16,000
deals is +1.5 ± 3.2 bb/100 (scored; on the real hand +4.8 ± 7.1, in chips
+1.3 ± 10.4). It leans towards live turn, where against v1c and v1a it
leaned the other way, but isn't significant.

Against v1c and v1a a longer match wouldn't change this: live turn − live
river is −1.5 ± 1.9 bb/100, which rules out a gain above +0.5; finding a gain
of +1 would take about 250,000 deals. Against off-menu sizes, which people
playing the app use all the time, it might: finding a gain of +1.5 reliably
(80% of the time) would take about 150,000 deals, a day of the box. What
matches against fixed bots
can't see is exploitability. Published subgame solving made bots far harder
to exploit (Libratus solved live from the turn on and after every opponent
bet; in Brown & Sandholm 2017, re-solving left a test game 150 mbb/hand
exploitable where action translation left 1,465), and our opponents never
look for weaknesses. Measuring that needs an exploiter, such as a local best
response (as DeepStack was measured), not more hands against fixed bots.

## Exploitability: local best response

The matches above are all against opponents that never look for weaknesses:
scripted bots and our own tables, which share the bot's betting menu. People
do look for them, so the bot is also measured against an exploiter:
**local best response** (LBR; Lisý & Bowling, "Equilibrium Approximation
Quality of Current No-Limit Poker Bots", 2017), in `hu_cfr/src/lbr.rs`.

```sh
hu lbr --run v1c --live river --deals 2000 --stacks full --seed 5      # LBR on the flop, turn and river
hu lbr --run v1c --streets 1234 ...                                     # ... and preflop
hu lbr --run v1c --streets none ...                                     # checks and calls: the station bot
```

LBR knows the bot's strategy. At each of its decisions it holds the bot's
exact range (each of the 1,326 hands weighted by how likely the bot was to
play the hand this way, through the same path `Engine::decide` takes:
translation, the fallback, live solves) and takes the action worth the most,
assuming the hand is then checked and called down unless the bot folds at
once:

    U(call)  = wp·pot − (1 − wp)·asked
    U(bet a) = fp·pot + (1 − fp)·(wp′·(pot + a) − (1 − wp′)·(asked + a))

`wp` is its showdown share against the range (every board card still to
come dealt), `fp` the bot's probability of folding to the bet, `wp′` the
share against the hands that don't fold. It folds when nothing beats
folding, and checks and calls on streets it isn't set to play. It considers
the minimum bet or raise, ⅓, ½, ¾, 1, 1½, 2 and 3 times the pot, and all-in
(on streets the bot solves live, where every answer costs the bot a solve:
the minimum, ½, 1 and 2 pots, all-in).

What it knows is the strategy, not the bot's luck: the bot translates
off-menu sizes at random (seeded per hand), and LBR's model averages over 16
independent draws of that translation (knowing the real draws, it could bet
2× pot exactly when the bot will read it as all-in). On streets solved live
it uses one draw, sampled by how well each explains the bot's play.

Its result is scored two ways: in chips (all-in runouts at their
expectation, as in `hu eval`) and against the bot's exact range at each
showdown or all-in (LBR's expected share against every hand the bot could
hold; same expectation, less luck). Checks: with `--streets none` LBR is
exactly the station bot (the same deals, seats and draws as `hu eval`; every
deal agrees with `hu eval --vs station` to 1e-14), and the two scores agree
(−179 ± 38 in chips, −190 ± 26 range-scored, 3,000 deals).

### What it finds

LBR's win rate, bb/100 on its side (a lower bound on how exploitable the bot
is), range-scored, 2,000 duplicate deals (4,000 hands) at 100bb, seed 5:

| Bot | LBR on the flop, turn and river | LBR on every street |
|---|---|---|
| table only (`--live off`) | +191 ± 41 (chips +169 ± 47) | +208 ± 21 (10,000 deals, seed 6) |
| live river | +134 ± 37 (chips +106 ± 46) | |
| live turn (served since 2026-11-19) | **+39 ± 36** (chips +51 ± 45) | |

On the same deals, paired: live river − table −58 ± 29, live turn − live
river −94 ± 35 (range-scored; LBR's side, so negative is better for the
bot). Live turn is what the fixed-opponent tests couldn't see: against them
it played no better than live river ("Why live turn adds nothing here"), but
an opponent that probes the turn finds almost nothing there. The run took
2.6 h (14,075 turn solves: every bet LBR considers on the turn costs the bot
a solve).

For scale: folding every hand loses 75 bb/100. Lisý & Bowling's LBR beat
the abstraction-based competition bots it tested (Hyperborean 2013 and
2014, Slumbot 2016, Act1 2016) by 3,300 to 5,060 mbb/hand (330 to 506
bb/100).

Where LBR makes its money against the table (how often it chose each
action on that street; in parentheses, how often the bot then folded; LBR's
model predicted these within a few points):

| Street | LBR's main actions |
|---|---|
| Flop | min-raise of the bot's bet 23% of its decisions (bot folds 56%), bet 3× pot 17% (74%), min-bet 17% (28%), bet 2× pot 10% (66%), all-in 5% (84%) |
| Turn | min-bet 36% (21%), min-raise 13% (58%), bets of 2–3× pot 12% (61–69%), all-in 5% (82%) |
| River | min-bet 32% (26%), all-in 10% (75%), min-raise 8% (66%) |
| Preflop (with `--streets 1234`) | raise 3× the pot 51% (bot folds 69%), 2× 20% (61%), 1½× 13% (55%) |

The holes are where its sizes leave the bot's menu: a min-raise is read as a
pot-size raise (the menu's only raise), a min-bet as a third of the pot, and
anything above the pot partly as all-in. Live solving answers real sizes
with exact hands, and LBR finds little to bet into there: against live turn
it checks or folds on 63% of its turn decisions and its turn min-bets get 7%
folds (against the table: 21%). What's left is mostly the flop, still played
from the table.

### The fallback

Where the abstract tree can't follow the real hand, the table has no
strategy and a heuristic plays (the debug output says why):

- a raise past the menu's three per street (a 5-bet preflop, a fourth raise
  after the flop), which the translation can only map to a call;
- the rest of a hand whose abstract version is over: a large bet translated
  as all-in and called, or a strategy trained shallower than the real stacks.

The default heuristic (`--fallback equity`) calls when its equity against a
*random* hand is at least 5% above the pot odds. Against a 5-bet, where the
price is low and almost any hand beats a random one, it continues with
90–100% of the bot's 4-bet range, bluffs included.

`--fallback range` (`engine::range_rule`) plays on equity against the
opponent's *estimated range* instead: the table's estimate along the
translated history up to where the hand left the tree, with the hedges live
solving uses and some "any hand" mixed in. Facing a bet, it continues with
the hands whose equity covers the price (in a shallow spot, the price of
getting all the chips in; deeper, the pot odds plus 5%) and jams the
strongest; a guard keeps it from folding more of its range than a pure bluff
of the size needs to break even. A raise past the raise cap, which the table
has no model of, is read as a raise for value (the opponent's hands ahead of
the bot's range; `--fallback-value` sets the share read that way). Against
5-bets after the bot opens to 2bb, is 3-bet to 6bb and 4-bets to 15bb (the
share of the bot's 4-bet range; `hu` audit of the served table):

| The 5-bet | Bluff breaks even at | `equity`: fold / call / jam | `range`: fold / call / jam |
|---|---|---|---|
| to 30bb (2×) | 53% folds | 0 / 92 / 8% | 6 / 89 / 5% |
| to 37.5bb (2½×) | 60% | 0 / 92 / 8% | 60 / 26 / 14% |
| to 45bb (3×) | 65% | 1 / 92 / 8% | 65 / 21 / 14% |
| all-in | 82% | 10 / 90 / 0% | 68 / 32 / 0% |

That folds the 4-bet bluffs, but it isn't the default, because it doesn't
pay. Against opponents that raise with anything, reading their raises as
value costs a lot, and the exploiter, which 5-bets only when it pays, doesn't
mind either way. The range fallback minus the default, on the same deals
(20,000 deals per bot, 10,000 for LBR; bb/100, the bot's side):

| Opponent | Value share 1 | Value share ½ | Value share 0 | Value share 1, preflop only |
|---|---|---|---|---|
| maniac (pot-size raises with any hand) | −98.8 ± 12.0 | −31.4 ± 9.2 | +2.9 ± 7.7 | −95.5 ± 11.8 |
| minbet | −16.6 ± 6.8 | −11.0 ± 6.2 | −8.6 ± 6.2 | −8.6 ± 6.1 |
| overbet | −16.5 ± 5.0 | −0.4 ± 3.9 | −0.4 ± 3.9 | −16.5 ± 5.0 |
| v1a, v1c, station, shover | 0 (no fallbacks) | | | |
| LBR on every street (range-scored) | −6.1 ± 13.1 | | | |

The cost is exactly in folding to 5-bets (preflop only costs as much), and
without the value reading the range rule plays like the default. The
structural fix is in the menu instead: with all-in offered past the raise
cap, a 5-bet jam is in the tree, and the bot both learns to 4-bet less
lightly and answers 5-bets from its training (the `cap` menu, `hu tree --menu cap`).

## A v2 menu: candidates and training time

LBR's edge against the served bot comes from the lean menu: preflop and on
the flop (the table still plays them), its min-raises are read as pot-size
raises, its overbets partly as all-ins, and the raise cap leaves no 5-bet.
`Menu::candidate` defines menus that close those gaps, for a retrain
(`hu tree --menu NAME` sizes them; `hu train --menu NAME` trains them):

| Menu | Adds to lean |
|---|---|
| `cap` | all-in stays an option past the raise cap (a 4-bet can be 5-bet all-in; a raise beyond the cap lands on a node) |
| `wide-flop` | `cap`, plus a 6bb open and a 2× 3-bet preflop, and on the flop bets of 1.5× and 2.5× the pot and half-pot raises; turn and river lean |
| `wide` | `wide-flop`'s sizes on the turn too |
| `wide-river` | `wide` on the river too |

With live turn served, the table's turn and river play no hands: they shape
how the flop is trained and give the live solves their safe values. So
`wide-flop` targets what's left to exploit at the least cost.

Entries (regret and strategy tables, 16 bytes each) with the a10k hand
groups:

| Depth | lean (v1c) | cap | wide-flop | wide | wide-river |
|---|---|---|---|---|---|
| 100bb | 352M (5.2 GiB) | 359M | 849M (12.7 GiB) | 1,322M (19.7 GiB) | 1,624M (24.2 GiB) |
| 70bb | 258M | 262M | 563M | 815M | 960M |
| 50bb | 180M | 182M | 357M | 496M | 570M |
| 35bb | 120M | 121M | 213M | 276M | 307M |
| 25bb | 79M | 79M | 131M | 165M | 180M |
| 12bb / 6bb | 28M / 6M | the same | 37M / 7M | 40M / 7M | 41M / 7M |

Training speed at 100bb (7 minutes each on the box, the same schedule, time
per iteration against lean at the same iteration count): `cap` 1.0×,
`wide-flop` 1.4×, `wide` 1.83×, `wide-river` 2.1×.

Estimated training time for every depth (v1c took 9.3 h: 3.8 h at 100bb),
scaling v1c's real times per depth by the slowdown (assumed to shrink at
shallower depths with the tree's growth) and either keeping v1c's iteration
counts or scaling them with the tree (as many updates per information set):

| Menu | 100bb alone, v1c's iterations | Every depth, v1c's iterations | Every depth, iterations scaled to the tree |
|---|---|---|---|
| lean (v1c, measured) | 3.8 h | 9.3 h | |
| `cap` | 3.8 h | **9.3 h** | 9.4 h |
| `wide-flop` | 5.3 h | **12.3 h** | 26.8 h |
| `wide` | 6.9 h | **15.3 h** | 49.3 h |
| `wide-river` | 7.9 h | 17.1 h | 66.0 h |

v1c's iteration count was probably more than it needed (v1c against v1b,
which had 2.5× fewer iterations and coarser hand groups: +1.0 ± 2.6 bb/100),
so the first column is the realistic one; the second is the cautious bound.
Memory (one depth at a time) and disk (about 4× v1c's 16 GB of checkpoints
for `wide`) fit on the box.

## Saved runs

Trained runs are too big for git (v1c is 7.2 GB of card abstraction, 1 GB
of strategy and 16 GB of solver checkpoints), so each one is saved to a
GitHub Release tagged `strategy-RUN`, created at the commit whose code reads
its files.

| Release | Run | Abstraction | Commit | Notes |
|---|---|---|---|---|
| [`strategy-v1c`](https://github.com/johnnyodonnell/heads-up-poker-ai-engine/releases/tag/strategy-v1c) | v1c (checkpoint C) | a10k: 169/10000/5000/2000 | 01ed9ef | served since 2026-11-17 |

### What a release holds

| Asset | Restores to | Size (v1c) | Needed for |
|---|---|---|---|
| `abstraction-ABS.tar.zst` | `data/ABS/` | 0.78 GB | serving: maps cards to hand groups |
| `strategy-RUN.tar.zst` | `strategies/RUN/` | 0.59 GB | serving: the lookup table, every stack depth |
| `checkpoint-RUN-sN.zst.partNN` | `runs/RUN/sN/solver.ckpt` | ~13 GB in all | only to resume training a depth |
| `SHA256SUMS` | | | checksums of every original file |
| `release.env` | | | the run and abstraction names, for the restore script |

The abstraction must be saved with the strategy: the strategy is indexed by
its hand groups, and rebuilding it (multithreaded k-means) isn't guaranteed to
come out identical. The strategy itself could be re-exported from the
checkpoints (`hu export`), but it is saved directly so serving needs only
the two small assets.

### Restoring

`release_restore.sh` downloads with `gh` (logged in), unpacks with `zstd`, and
verifies every file against `SHA256SUMS`. It never overwrites an existing
run or abstraction directory. It needs ~8.3 GB free, plus 16 GB with
`--checkpoints`.

```sh
# on the box (into $HU_HOME); add --checkpoints to resume training too
cfr/box.sh ./release_restore.sh strategy-v1c
cfr/box.sh 'echo v1c > $HU_HOME/current_run && pm2 restart hu-engine'
```

On a fresh box (with Rust, Node + pm2, `gh` logged in, and `zstd`), first put
the code in place and build it, then restore, then start the service once:

```sh
cfr/sync_to_box.sh
cfr/box.sh cargo build --release -p hu
cfr/box.sh ./release_restore.sh strategy-v1c
cfr/box.sh 'echo v1c > $HU_HOME/current_run && pm2 start pm2.config.cjs && pm2 save'
cfr/deploy_web.sh
```

The script also works on any other machine with `gh` and `zstd`, with
`HU_HOME` pointing at where the files should go:
`HU_HOME=/some/dir cfr/release_restore.sh strategy-v1c`. If the file formats
change in later code, build the release's commit to read it.

### Saving a new run

```sh
# on the box: RUN, its abstraction, and the (pushed) commit whose code reads them
cfr/box_bg.sh release-v1d ./release_save.sh v1d a10k <commit-sha>
cfr/box.sh tail -f '$HU_HOME/logs/release-v1d.log'
```

The abstraction and strategy go up first, then the checkpoints one depth at a
time, compressed and split under GitHub's 2 GiB asset limit (the box
uploads at roughly 4 MB/s, so v1c's checkpoints take about an hour). Pass
`--no-checkpoints` to skip them. Add the release to the table above.

## Parity with the TS game

`game.ts` and `evaluator.ts` are the source of truth. The Rust port must
reproduce random freeze-out matches recorded from the TS engine step by step,
and order random 7-card hands exactly as the TS evaluator does:

```sh
npx tsx cfr/scripts/check_rules.ts        # TS rule scenarios
npx tsx cfr/scripts/dump_fixtures.ts      # -> cfr/fixtures/
cfr/sync_to_box.sh
cfr/box.sh cargo test --release
```

## Solver checks

- `cargo test --release -p hu_cfr`: MCCFR reaches the known equilibria of Kuhn
  poker (value -1/18) and Leduc hold'em (exact exploitability via best
  response), single- and multi-threaded; exact per-board equities match brute
  force; tree, table-index and checkpoint tests.
- `hu pushfold --stack 20`: heads-up push/fold at 10bb, MCCFR on real dealt
  cards against an exact full-width solve (58.4% shove / 37.3% call; MCCFR
  lands within 0.5 mbb/hand of exploitability in seconds).
- `hu tree --stack 200 --buckets 169,2000,2000,1000`: size of a betting tree.
