#!/usr/bin/env bash
# Train every stack depth of a run, one after another (on the box).
#
#   cfr/train_all.sh RUN ABSTRACTION ITERS_AT_100BB [STACKS_IN_CHIPS...]
#   MENU=wide-river SCALE=tree cfr/train_all.sh RUN ...
#
# Default depths (chips; blinds 1/2): 200 140 100 70 50 24 12, i.e.
# 100/70/50/35/25/12/6bb. Shallower trees are smaller, so each depth gets
# ITERS × max(0.15, stack/200) iterations. Strategies are exported to
# $HU_HOME/strategies/RUN/s<stack>/ as each depth finishes.
#
# MENU picks the betting menu (default lean; see hu_cfr/src/menu.rs).
# SCALE=tree multiplies each depth's iterations by its tree's size against
# lean's at that depth (`hu train --scale-iters`).
set -euo pipefail
cd "$(dirname "$0")"
source ./box_env.sh
RUN=$1
ABS=$2
ITERS=$3
shift 3
STACKS="${*:-200 140 100 70 50 24 12}"
MENU="${MENU:-lean}"
SCALE_FLAG=""
if [ "${SCALE:-}" = "tree" ]; then SCALE_FLAG="--scale-iters"; fi
for S in $STACKS; do
  N=$(awk -v i="$ITERS" -v s="$S" 'BEGIN { f = s / 200; if (f < 0.15) f = 0.15; printf "%d", i * f }')
  echo "=== $RUN: stack $S, menu $MENU, $N iterations${SCALE_FLAG:+ (scaled to the tree)}"
  ./target/release/hu train --run "$RUN" --abs "$ABS" --stack "$S" --iters "$N" --menu "$MENU" $SCALE_FLAG
done
echo "=== $RUN: all depths done"
