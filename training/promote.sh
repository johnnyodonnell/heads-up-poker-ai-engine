#!/usr/bin/env bash
# Pull a snapshot into the app (only when asked): export it to the browser
# model (public/models/current.onnx), then verify it — ONNX-vs-tch parity and
# the browser-engine e2e test. Does NOT commit; commit/push separately.
#
#   training/promote.sh                 # top-rated snapshot of run1
#   RUN=run2 training/promote.sh        # ... of another run
#   training/promote.sh <snapshot path on the box>
#
# The first time, also switch src/hooks/useGameState.ts to
# `import { bestMove } from "../engine/neural";`.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${RUN:-run1}"
W='~/Workspace/heads-up-poker-ai-engine'
BIN="$W/repo/training/target/release/hu"

SNAP="${1:-$(ssh asus-nvidia "$BIN best --run-dir $W/runs/$RUN")}"
echo "promoting $SNAP"
ssh asus-nvidia "mkdir -p $W/runs/$RUN/exports && $BIN export --weights $SNAP \
  --out $W/runs/$RUN/exports/current.onnx --fixture $W/runs/$RUN/exports/fx.json"
mkdir -p public/models training/fixtures
scp -q "asus-nvidia:$W/runs/$RUN/exports/current.onnx" public/models/current.onnx
scp -q "asus-nvidia:$W/runs/$RUN/exports/fx.json" training/fixtures/promote_fx.json

npx tsx training/scripts/test_onnx_parity.ts public/models/current.onnx training/fixtures/promote_fx.json
npx tsx training/scripts/test_engine_e2e.ts public/models/current.onnx 3000
echo "promoted $SNAP -> public/models/current.onnx (not committed)"
