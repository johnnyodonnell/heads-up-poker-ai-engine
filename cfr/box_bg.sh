#!/usr/bin/env bash
# Start a long-running command on the box, fully detached, inside repo/cfr
# with the project environment; output goes to $HU_HOME/logs/NAME.log.
#
#   cfr/box_bg.sh abs-a2k ./target/release/hu abstraction --name a2k ...
#   cfr/box.sh tail -f '$HU_HOME/logs/abs-a2k.log'
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "usage: cfr/box_bg.sh NAME <command...>" >&2
  exit 2
fi
NAME=$1
shift
ssh -n asus-nvidia "cd Workspace/heads-up-poker-ai-engine/repo/cfr && source ./box_env.sh && \
  setsid -f bash -c 'exec $* > \$HU_HOME/logs/$NAME.log 2>&1 < /dev/null' && echo started $NAME"
