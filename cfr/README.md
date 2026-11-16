# CFR engine

The bot's engine: a precomputed CFR strategy for heads-up no-limit hold'em,
trained and served on the box (`asus-nvidia`). Everything here is Rust and is
built and run on the box; this machine only edits code and dumps the TS parity
fixtures.

## Layout

```
hu_core/          the game: a port of src/engine/game.ts (rules), cards, a 7-card
                  hand evaluator, equity and all-in EV
hu_cfr/           the solver: abstract betting trees + bet-size menus, 169 preflop
                  classes, external-sampling MCCFR, exact best response for
                  small games, push/fold and Kuhn/Leduc validation games
hu/               the `hu` command line (built on the box)
scripts/          TS run with tsx: rule scenario checks, parity fixture dumper
fixtures/         parity fixtures dumped from the TS engine (gitignored)
box_env.sh        environment for the box (sourced by box.sh)
box.sh            run a command on the box inside repo/cfr
sync_to_box.sh    rsync this working tree to the box
```

## The box

Everything for this project on the box lives under
`~/Workspace/heads-up-poker-ai-engine/` (see `box_env.sh` for the subfolders);
Cargo's download cache and temp files are kept there too.

```sh
cfr/sync_to_box.sh                         # copy the working tree to the box
cfr/box.sh cargo test --release -p hu_core # run something there
```

## Parity with the TS game

`game.ts` and `evaluator.ts` are the source of truth. The Rust port must
reproduce random freeze-out matches recorded from the TS engine step by step,
and order random 7-card hands exactly as the TS evaluator does:

```sh
npx tsx cfr/scripts/check_rules.ts        # TS rule scenarios
npx tsx cfr/scripts/dump_fixtures.ts      # -> cfr/fixtures/
cfr/sync_to_box.sh
cfr/box.sh cargo test --release -p hu_core
```

## Solver checks

- `cargo test --release -p hu_cfr`: MCCFR reaches the known equilibria of Kuhn
  poker (value -1/18) and Leduc hold'em (exact exploitability via best
  response), single- and multi-threaded; tree and checkpoint tests.
- `hu pushfold --stack 20`: heads-up push/fold at 10bb, MCCFR on real dealt
  cards against an exact full-width solve (58.4% shove / 37.3% call; MCCFR
  lands within 0.5 mbb/hand of exploitability in seconds).
- `hu tree --stack 200 --buckets 169,2000,2000,1000`: size of a betting tree.
