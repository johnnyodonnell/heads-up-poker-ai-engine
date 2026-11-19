#!/usr/bin/env bash
# The live-solving test panel, round 2 (design and gates: README, "Test
# round 2"). Internal opponents only (v1c, v1a), 100bb, seed 2, every match
# saved deal by deal under $HU_HOME/evals/round2. Live matches against a
# table are scored against the table (`hu_cfr::score`, the .adv files): the
# main measure. The table matches give the paired cross-check.
#
#   cfr/box_bg.sh panel ./panel.sh          # then: cfr/box.sh ./panel_report.sh
#
# PANEL_PHASES (default "river probes turn") picks the parts. The turn
# phase plays TURN_DEALS deals from TURN_FROM (to extend it later without
# replaying deals).
set -uo pipefail
cd "$(dirname "$0")"
source ./box_env.sh
S=$HU_HOME/evals/round2
HU=./target/release/hu
RUNS="run:v1c,run:v1a"
PROBES="station,maniac,nit,shover,minbet,overbet,random"
ev() { # tag deals vs [flags...]
  local tag=$1 deals=$2 vs=$3; shift 3
  $HU eval --run v1c --vs "$vs" --deals "$deals" --stacks full --seed 2 --save "$S" --tag "$tag" "$@" 2>&1 | grep " vs "
}
for phase in ${PANEL_PHASES:-river probes turn}; do
  case $phase in
    river)
      ev table 50000 "$RUNS"
      ev river 50000 "$RUNS" --live river ;;
    probes)
      ev table 5000 "$PROBES"
      ev river 5000 "$PROBES" --live river
      # The turn only where the river panel showed a gap.
      ev turn 1000 station,maniac,minbet --live turn ;;
    turn)
      ev turn "${TURN_DEALS:-12000}" "$RUNS" --live turn --from "${TURN_FROM:-0}" ;;
  esac
done
echo "=== panel done"
