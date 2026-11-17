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
  eval.rs, bots.rs  matches against scripted bots
  pushfold.rs, toy.rs, br.rs   solver validation games and exact best response
hu/               the `hu` command line (abstraction, train, export, eval, probe,
                  serve, tree, pushfold)
scripts/          TS run with tsx: rule scenario checks, parity fixture dumper
fixtures/         parity fixtures dumped from the TS engine (gitignored)
box_env.sh        environment for the box (sourced by everything below)
box.sh            run a command on the box inside repo/cfr
box_bg.sh         start a long job on the box, detached, logging to $HU_HOME/logs
sync_to_box.sh    rsync this working tree to the box
train_all.sh      train a run's stack depths one after another
serve.sh, pm2.config.cjs   the engine service under pm2
deploy_web.sh     build the web app and copy it to the box
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
