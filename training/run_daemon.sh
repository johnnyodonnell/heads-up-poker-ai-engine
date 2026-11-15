#!/usr/bin/env bash
# Start (or resume) the self-play training daemon on the box, detached.
#
#   RUN=run1 SNAPSHOT_EVERY=30m training/run_daemon.sh [extra `hu train` flags...]
#
# Output goes to ~/Workspace/heads-up-poker-ai-engine/runs/$RUN/ (latest.*,
# snapshots/, pool.json, train.log). Re-running resumes from latest.*.
# SNAPSHOT_EVERY defaults to the trainer's 4h.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
W="$(cd "$DIR/../.." && pwd)" # ~/Workspace/heads-up-poker-ai-engine
RUN="${RUN:-run1}"
OUT="$W/runs/$RUN"
mkdir -p "$OUT"
cd "$DIR"
nohup ./target/release/hu train --run-dir "$OUT" --snapshot-every "${SNAPSHOT_EVERY:-4h}" "$@" \
  >>"$OUT/train.log" 2>&1 &
echo "started pid $! -> $OUT/train.log"
