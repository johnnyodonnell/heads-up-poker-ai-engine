#!/usr/bin/env bash
# Summarize cfr/panel.sh: for each opponent, stack depth and deal count, the
# table-only result and each live variant minus the table on the same deals.
#
#   cfr/box.sh ./panel_report.sh
set -uo pipefail
cd "$(dirname "$0")"
source ./box_env.sh
S=$HU_HOME/evals/panel
for base in "$S"/table--vs-*.txt; do
  rest=${base#"$S"/table--}
  others=()
  for tag in river-unsafe river-safe turn-unsafe turn-safe; do
    [ -f "$S/$tag--$rest" ] && others+=("$S/$tag--$rest")
  done
  [ ${#others[@]} -gt 0 ] || continue
  echo "== ${rest%.txt}"
  ./target/release/hu compare "$base" "${others[@]}" | sed "s|$S/||"
done
