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
  engine.rs         play time: replay, translation, action mapping
  eval.rs, bots.rs  matches against scripted bots (duplicate, paired deals)
  live/             live solving: subgame tree, vectorized DCFR, showdowns,
                    safe re-solving against the table's values
  pushfold.rs, toy.rs, br.rs   solver validation games and exact best response
hu/               the `hu` command line (abstraction, train, export, eval, compare,
                  probe, livebench, serve, slumbot, slumbot-diag, tree, pushfold)
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
cfr/box_bg.sh slumbot-v1b ./target/release/hu slumbot --run v1b --hands 10000 --sessions 2   # vs Slumbot (200bb)
# 4. serve it
cfr/box.sh 'echo v1b > $HU_HOME/current_run && pm2 restart hu-engine'
```

The service (port 8790) is started once with
`pm2 start cfr/pm2.config.cjs && pm2 save` from `repo/`; `cfr/deploy_web.sh`
updates the web app it serves. `GET /api/health` shows what's loaded.

## Results

All in bb/100 from the engine's side, ± a 95% interval.

### Runs

| Run | Abstraction (flop/turn/river groups) | Iterations at 100bb | Training time | Served |
|---|---|---|---|---|
| v1a (checkpoint A) | a200: 200/200/200 | 300M, 100bb only | 12 min | 2026-11-16 |
| v1b (checkpoint B) | a2k: 2000/2000/1000 | 2B | 3.6 h | 2026-11-16 |
| v1c (checkpoint C) | a10k: 10000/5000/2000 | 5B | 9.3 h | since 2026-11-17 |

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

`hu slumbot --run v1c --hands 10000 --sessions 2` (2026-11-17, 91 min).
[Slumbot](https://slumbot.com) is a strong public research bot. It plays
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
not affect the app, whose matches start at 100bb. Every hand is logged to
`$HU_HOME/logs/slumbot-v1c.jsonl`.

## Live solving (v2)

With `--live river` or `--live turn` (on `hu eval`, `serve`, `slumbot`), the
engine solves turn and river decisions live instead of looking them up;
preflop and the flop stay on the table. Off by default, and not served yet.

How a live decision is made (`hu_cfr/src/live/`):

- **The subgame** starts at the beginning of the current street with the
  real pot and stacks. The street's real actions so far are in the tree at
  their real sizes (off-menu sizes are added), so nothing is translated, and
  every one of the 1,326 hands is exact (no hand groups). A turn solve deals
  all 48 river cards. The bot's own earlier actions on the street are frozen
  to the strategy it actually played.
- **The solver** is Discounted CFR (α 1.5, β 0, γ 2), full-width and
  vectorized over all hands of both players, with exact card removal. River
  solves run 400 iterations (about 0.5 s), turn solves 150 (about 1.9 s on
  the box, river cards in parallel). Solves are deterministic, and cached per
  hand so later decisions reuse earlier ones.
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
hu slumbot-diag --run v1c --live river --log $HU_HOME/logs/slumbot-v1c-live-river.jsonl
                                            # opponent ranges vs the hands Slumbot really held
hu compare A.txt B.txt                      # paired difference of two saved matches
./panel.sh; ./panel_report.sh               # the test panel (below) and its paired results
```

The unit tests check the solver against a polarized river toy with a known
equilibrium, showdown and fold payoffs against brute force, convergence and
determinism of river and turn solves, off-menu actions, and that safe
solving holds hands an estimate left out to their alternatives while unsafe
solving doesn't (0.34% vs 12.9% of the pot).

### Why safe, and the Slumbot runs

The first version was unsafe: it trusted the estimated opponent range.
Against Slumbot (10,000 hands each, minus what Slumbot wins with the same
cards, ± about 31) it did no better than the table: table +18.2, live river
+17.5, live turn −13.4. Slumbot shows its cards after every hand, so
`hu slumbot-diag` could check the estimates against the hands it really
held: they were badly overconfident, predicting those hands worse than
"any hand" would (−2.8 nats per hand at the start of the river). The
estimates are now hedged and fit to those logs (`LiveCfg`: +0.33 nats per
hand, calibrated), and safe re-solving makes the solve hold up when the
estimate is still wrong. Slumbot itself is too noisy to measure a few
bb/100 in a day of play, and its rate limits made long runs unreliable, so
it is no longer part of testing: the engine is tested only against bots
built here (the earlier tables and the scripted bots).

### The test panel

`panel.sh` plays the table, safe live and unsafe live (`--safe off`) against
the earlier tables (v1c itself, v1b, v1a) and the scripted bots, at 200bb
(Slumbot's depth) and 100bb. Every match is saved deal by deal, and the
deals and the opponent's random draws depend only on the seed, so two
variants are compared on identical deals (`hu compare`): only their
decisions differ. Gates, set before the runs: safe live no worse than the
table against any opponent, better against the other tables, and still
beating every scripted bot by a wide margin.

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
