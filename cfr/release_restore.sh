#!/usr/bin/env bash
# Restore a trained run saved by cfr/release_save.sh into $HU_HOME.
#
#   cfr/release_restore.sh TAG [--checkpoints]
#   e.g. cfr/release_restore.sh strategy-v1c
#
# Downloads the card abstraction and the strategy (and, with --checkpoints,
# the solver checkpoints), unpacks them into $HU_HOME/{data,strategies,runs},
# and verifies every file against the release's SHA256SUMS. Existing
# directories are never overwritten. Set HU_HOME to restore somewhere else.
# Afterwards: echo RUN > $HU_HOME/current_run && pm2 restart hu-engine
set -euo pipefail
TAG=$1
CKPTS=0
[ "${2:-}" = "--checkpoints" ] && CKPTS=1
REPO=${HU_REPO:-johnnyodonnell/heads-up-poker-ai-engine}
HU_HOME=${HU_HOME:-$HOME/Workspace/heads-up-poker-ai-engine}
DL=$HU_HOME/tmp/restore-$TAG
log() { echo "[$(date +%T)] $*"; }

mkdir -p "$DL"
log "downloading $TAG metadata"
gh release download "$TAG" --repo "$REPO" -D "$DL" --clobber -p release.env -p SHA256SUMS
# shellcheck disable=SC1091
source "$DL/release.env"
for d in "data/$ABS" "strategies/$RUN"; do
  [ -e "$HU_HOME/$d" ] && { echo "$HU_HOME/$d already exists; not overwriting" >&2; exit 1; }
done
if [ $CKPTS = 1 ] && [ -e "$HU_HOME/runs/$RUN" ]; then
  echo "$HU_HOME/runs/$RUN already exists; not overwriting" >&2
  exit 1
fi

mkdir -p "$HU_HOME/data" "$HU_HOME/strategies" "$HU_HOME/runs"
for a in "abstraction-$ABS" "strategy-$RUN"; do
  log "downloading and unpacking $a"
  # One file, or .partNN pieces when it was over GitHub's asset limit.
  gh release download "$TAG" --repo "$REPO" -D "$DL" --clobber -p "$a.tar.zst*"
  cat "$DL/$a.tar.zst"* | zstd -q -d -c | tar -xf - -C "$HU_HOME"
  rm -f "$DL/$a.tar.zst"*
done

if [ $CKPTS = 1 ]; then
  for s in $(grep -o "runs/$RUN/s[0-9]*/" "$DL/SHA256SUMS" | cut -d/ -f3); do
    log "downloading and unpacking checkpoint $s"
    gh release download "$TAG" --repo "$REPO" -D "$DL" --clobber -p "checkpoint-$RUN-$s.zst.part*"
    mkdir -p "$HU_HOME/runs/$RUN/$s"
    cat "$DL/checkpoint-$RUN-$s.zst.part"* | zstd -q -d -o "$HU_HOME/runs/$RUN/$s/solver.ckpt"
    rm -f "$DL/checkpoint-$RUN-$s.zst.part"*
  done
fi

log "verifying checksums"
WANT="^[0-9a-f]+  (data/$ABS/|strategies/$RUN/"
[ $CKPTS = 1 ] && WANT="$WANT|runs/$RUN/"
grep -E "$WANT)" "$DL/SHA256SUMS" > "$DL/want.sha256"
(cd "$HU_HOME" && sha256sum --quiet -c "$DL/want.sha256")
rm -rf "$DL"
log "restored $RUN ($ABS) into $HU_HOME"
log "serve it with: echo $RUN > \$HU_HOME/current_run && pm2 restart hu-engine"
