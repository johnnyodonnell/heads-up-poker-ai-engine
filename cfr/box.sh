#!/usr/bin/env bash
# Run a command on the box inside repo/cfr, with the project environment
# (cfr/box_env.sh) loaded. Sync first with cfr/sync_to_box.sh.
#
#   cfr/box.sh cargo test --release -p hu_core
#   cfr/box.sh ./target/release/hu --help
set -euo pipefail
if [ $# -eq 0 ]; then
  echo "usage: cfr/box.sh <command...>" >&2
  exit 2
fi
ssh asus-nvidia "cd Workspace/heads-up-poker-ai-engine/repo/cfr && source ./box_env.sh && $*"
