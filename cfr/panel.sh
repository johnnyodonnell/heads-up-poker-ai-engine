#!/usr/bin/env bash
# The live-solving test panel (internal opponents only): table-only, unsafe
# and safe live solving against the earlier tables and the probe bots, at
# 200bb and 100bb, every match saved per deal ($HU_HOME/evals/panel) so
# variants are compared on identical deals (`hu compare`).
#
#   cfr/box.sh 'nohup ./panel.sh > $HU_HOME/logs/panel.log 2>&1 &'
#
# PANEL_PHASES (default "river probes turn") picks the parts to run.
set -uo pipefail
cd "$(dirname "$0")"
source ./box_env.sh
S=$HU_HOME/evals/panel
HU=./target/release/hu
RUNS="run:v1c,run:v1b,run:v1a"
PROBES="station,maniac,nit,shover,minbet,overbet,random"
ev() { # tag stacks deals vs [flags...]
  local tag=$1 stacks=$2 deals=$3 vs=$4; shift 4
  $HU eval --run v1c --vs "$vs" --deals "$deals" --stacks "$stacks" --save "$S" --tag "$tag" "$@" 2>&1 | grep " vs "
}
for phase in ${PANEL_PHASES:-river probes turn}; do
  case $phase in
    river)
      for st in 200bb full; do
        ev table "$st" 50000 "$RUNS"
        ev river-safe "$st" 50000 "$RUNS" --live river
        ev river-unsafe "$st" 50000 "$RUNS" --live river --safe off
      done ;;
    probes)
      for st in 200bb full; do
        ev table "$st" 10000 "$PROBES"
        ev river-safe "$st" 10000 "$PROBES" --live river
        ev river-unsafe "$st" 10000 "$PROBES" --live river --safe off
      done ;;
    turn)
      ev table 200bb 8000 "$RUNS"
      ev turn-safe 200bb 8000 "$RUNS" --live turn
      ev turn-unsafe 200bb 8000 run:v1a --live turn --safe off
      ev table full 6000 run:v1b,run:v1a
      ev turn-safe full 6000 run:v1b,run:v1a --live turn
      ev table 200bb 2000 maniac,minbet
      ev turn-safe 200bb 2000 maniac,minbet --live turn ;;
  esac
done
echo "=== panel done"
