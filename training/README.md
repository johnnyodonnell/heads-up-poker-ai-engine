# Heads-up poker — self-play training

Search-free self-play **REINFORCE** for the browser bot, following
fox-lite-ai-engine's recipe, with all training logic in Rust. Self-play,
training, evaluation, probes and ONNX export all run in one binary, `hu`,
using `tch`/libtorch on the GPU. The browser (`src/engine/neural.ts`) runs the
exported ONNX with onnxruntime-web: one forward pass per decision, sampling
from the policy. The agreed plan is in [PLAN.md](PLAN.md).

## Layout

```
hu_core/           pure Rust (no torch): rules (port of src/engine/game.ts), 7-card
                   evaluator, action space + NN encoder (mirrors actions.ts /
                   encode.ts), all-in EV, equity, training-hand sampler
trainer/           the `hu` binary (tch): network + ONNX writer, self-play,
                   REINFORCE step, daemon loop, evaluator, scripted bots, probes
scripts/           TS run with tsx: rule checks, fixture dumper, ONNX parity,
                   browser-engine e2e
sync_to_box.sh     rsync this working tree to the box
run_daemon.sh      start/resume training on the box
promote.sh         export a snapshot to public/models/current.onnx and verify it
```

## The model

- **Actions:** four types (fold, check/call, bet/raise, all-in). A bet/raise
  is sized by a position u in [0, 1] drawn from a Beta(a, b) (a, b >= 1) over
  the legal non-all-in raises, on a log scale of the pot after calling
  (`src/engine/actions.ts`).
- **Input** (`src/engine/encode.ts`, 622 floats): the mover's hole cards and
  the board (suit-canonical), up to 32 action-history tokens, and table state,
  all from the mover's point of view.
- **Net (~3.2M params, `trainer/src/net.rs`):** one-hot suit × rank card
  planes; attention over the history (mean + max + learned-query pooling); a
  512-wide residual trunk; heads for the type logits, the Beta (a, b) and a
  training-only value.
- **Training (`train.rs`):** both seats play the current policy, sampling at
  temperature 1. Each hand starts from sampled stacks (40% 200/200, otherwise
  a Beta(1.5, 1.5) split of the 400 chips; random button). A seat's chip
  result ÷ 200 — with the board's run-out replaced by its expected value when
  no decisions are left — is z for all its decisions. One on-policy AdamW
  step per cohort (32,768 hands): advantage z − V (own value head), value
  MSE, entropy bonuses on the type and the Beta.

## Box setup (asus-nvidia)

Everything lives under `~/Workspace/heads-up-poker-ai-engine/`:

```
repo/    this repository (synced with training/sync_to_box.sh)
deps/    venv with torch==2.13.0 (cu130): only a source of libtorch for tch 0.26
runs/    training runs (checkpoints, snapshots, pool.json, logs)
```

```sh
cd ~/Workspace/heads-up-poker-ai-engine
~/.local/bin/uv venv deps/venv --python 3.12
~/.local/bin/uv pip install --python deps/venv/bin/python \
    --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0
source ~/.cargo/env
cd repo/training && cargo build --release      # LIBTORCH comes from .cargo/config.toml
./target/release/hu gpu-check
```

No Python runs at build or train time; the binary finds libtorch through its
rpath.

## Train

```sh
training/sync_to_box.sh                         # here
RUN=run1 SNAPSHOT_EVERY=30m training/run_daemon.sh   # on the box (from repo/)
tail -f ~/Workspace/heads-up-poker-ai-engine/runs/run1/train.log
```

`run_daemon.sh` resumes from `runs/$RUN/latest.*` if present.
`SNAPSHOT_EVERY` defaults to 4h. To change the cadence, stop the process
(`pkill -f 'hu train'`) and restart with the new value: the already-scheduled
snapshot keeps its time, and later ones use the new interval.

Each cohort logs one JSON line: timings, losses, the type and Beta entropies,
the Beta's mean (a, b), grad norm, `logp_gap` (sampler vs trainer log-prob —
must stay ~0: the on-policy check), approximate KL, and HUD stats (VPIP/PFR by
position, 3-bet, c-bet, fold to c-bet, WTSD, the action mix, bet-size
quantiles as pot fractions). The HUD is the early warning for nit/maniac
swings and for bet sizes collapsing to min-raises.

About 8 s per cohort on the GB10 (~2 s self-play at ~15k hands/s, ~6 s for the
update): roughly 15M hands per hour.

## Evaluate

Every snapshot is probed (`[probe]` in the log) and evaluated (`[eval]`).
A match is duplicate heads-up: each deal (sampled stacks, button, deck) is
played twice with the seats swapped, all-in-EV adjusted, both sides sampling
as in the browser; 30k deal pairs per opponent, bb/100 ± 95% CI. Each
snapshot plays the previous snapshot, the top-rated and spread-anchor
snapshots — a least-squares rating ladder in `runs/<run>/pool.json`, first
snapshot = 0 — plus the random bot and a scripted panel (calling station,
maniac, nit, shover), reported separately as `vs_random` / `vs_panel`.

```sh
hu match --a <snap>.safetensors --b random|station|maniac|nit|shover|<snap> [--deals 30000]
hu probe --weights <snap>.safetensors
hu best --run-dir <run>              # top-rated snapshot
```

## Pull a snapshot into the app (only when asked)

```sh
training/promote.sh [<snapshot path on the box>]   # default: top-rated of run1
```

It exports the ONNX on the box, copies it to `public/models/current.onnx`, and
runs the ONNX parity check and the browser-engine e2e. The first time, switch
`src/hooks/useGameState.ts` to `import { bestMove } from "../engine/neural";`.
Commit and push separately.

## Parity checks

```sh
# here (Node):
npx tsx training/scripts/check_rules.ts          # game.ts rule scenarios
npx tsx training/scripts/dump_fixtures.ts        # -> training/fixtures/ (gitignored)
training/sync_to_box.sh
# on the box:
cargo test --release -p hu_core                  # rules, evaluator, encoder parity
hu export --init 7 --out fixtures/parity_model.onnx --fixture fixtures/parity_fx.json
# copy both back to training/fixtures/, then here:
npx tsx training/scripts/test_onnx_parity.ts     # onnxruntime-web vs tch
npx tsx training/scripts/test_engine_e2e.ts training/fixtures/parity_model.onnx
```
