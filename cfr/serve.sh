#!/usr/bin/env bash
# Start the engine service on the box (pm2 runs this; see pm2.config.cjs).
# Serves the run named in $HU_HOME/current_run (override with HU_RUN) and the
# built web app in $HU_HOME/web, on port 8790 (override with HU_PORT).
#
# Live solving per $HU_HOME/live_mode (off, river or turn; override with
# HU_LIVE); default off.
#
# Switch the served run:  echo RUN > $HU_HOME/current_run && pm2 restart hu-engine
# Switch live solving:    echo river > $HU_HOME/live_mode && pm2 restart hu-engine
set -euo pipefail
cd "$(dirname "$0")"
source ./box_env.sh
RUN="${HU_RUN:-$(cat "$HU_HOME/current_run")}"
LIVE="${HU_LIVE:-$(cat "$HU_HOME/live_mode" 2>/dev/null || echo off)}"
exec ./target/release/hu serve --run "$RUN" --live "$LIVE" --port "${HU_PORT:-8790}"
