#!/usr/bin/env bash
# Save a trained run to a GitHub Release (on the box), so it can be restored
# with cfr/release_restore.sh without retraining.
#
#   cfr/release_save.sh RUN ABSTRACTION COMMIT [--no-checkpoints]
#   e.g. cfr/release_save.sh v1c a10k c84c400
#
# Release "strategy-RUN" at COMMIT (the code that reads these files), with:
#   abstraction-ABS.tar.zst        data/ABS/          card abstraction tables
#   strategy-RUN.tar.zst           strategies/RUN/    what the engine serves
#   checkpoint-RUN-sN.zst.partNN   runs/RUN/sN/       solver state, to resume training
#   SHA256SUMS                     of every original file
#   release.env                    RUN= and ABS= for the restore script
# Assets go up in that order; each is kept under GitHub's 2 GiB asset limit.
set -euo pipefail
cd "$(dirname "$0")"
source ./box_env.sh
RUN=$1
ABS=$2
COMMIT=$3
CKPTS=1
[ "${4:-}" = "--no-checkpoints" ] && CKPTS=0
REPO=${HU_REPO:-johnnyodonnell/heads-up-poker-ai-engine}
TAG=strategy-$RUN
OUT=$HU_HOME/tmp/release-$TAG
PART=1900M

cd "$HU_HOME"
for d in "data/$ABS" "strategies/$RUN"; do
  [ -d "$d" ] || { echo "missing $HU_HOME/$d" >&2; exit 1; }
done
rm -rf "$OUT"
mkdir -p "$OUT"
log() { echo "[$(date +%T)] $*"; }

log "checksums"
FILES="data/$ABS/* strategies/$RUN/*/*"
[ $CKPTS = 1 ] && FILES="$FILES runs/$RUN/*/solver.ckpt"
# shellcheck disable=SC2086
sha256sum $FILES > "$OUT/SHA256SUMS"
printf 'RUN=%s\nABS=%s\n' "$RUN" "$ABS" > "$OUT/release.env"

log "compressing abstraction and strategy"
tar -cf - "data/$ABS" | zstd -q -T0 -10 -o "$OUT/abstraction-$ABS.tar.zst"
tar -cf - "strategies/$RUN" | zstd -q -T0 -10 -o "$OUT/strategy-$RUN.tar.zst"
for f in "$OUT"/*.tar.zst; do
  [ "$(stat -c%s "$f")" -lt 2000000000 ] || { echo "$f is over 2 GB; split it" >&2; exit 1; }
done

NOTES=$OUT/notes.md
cat > "$NOTES" <<EOF
Trained strategy **$RUN** (card abstraction **$ABS**), built from commit $COMMIT.

Restore it on the box (or any machine with \`gh\`, \`zstd\` and this repo):

    cfr/release_restore.sh $TAG                 # abstraction + strategy (what serving needs)
    cfr/release_restore.sh $TAG --checkpoints   # also the solver checkpoints, to keep training

then serve it: \`echo $RUN > \$HU_HOME/current_run && pm2 restart hu-engine\`.

| Asset | Restores to |
|---|---|
| abstraction-$ABS.tar.zst | \`data/$ABS/\` |
| strategy-$RUN.tar.zst | \`strategies/$RUN/\` |
| checkpoint-$RUN-sN.zst.partNN | \`runs/$RUN/sN/solver.ckpt\` (concatenate the parts, then zstd -d) |
| SHA256SUMS | checksums of the original files |
EOF

if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  log "release $TAG exists; uploading into it"
else
  gh release create "$TAG" --repo "$REPO" --target "$COMMIT" --title "Strategy $RUN ($ABS)" --notes-file "$NOTES"
fi
log "uploading abstraction and strategy"
gh release upload "$TAG" --repo "$REPO" --clobber "$OUT/release.env" "$OUT/SHA256SUMS" "$OUT/abstraction-$ABS.tar.zst" "$OUT/strategy-$RUN.tar.zst"
rm -f "$OUT"/*.tar.zst

if [ $CKPTS = 1 ]; then
  for c in runs/"$RUN"/s*/solver.ckpt; do
    s=$(basename "$(dirname "$c")")
    log "checkpoint $s: compressing"
    zstd -q -T0 -3 -c "$c" | split -b $PART -d -a 2 - "$OUT/checkpoint-$RUN-$s.zst.part"
    log "checkpoint $s: uploading $(ls "$OUT"/checkpoint-* | wc -l) part(s)"
    gh release upload "$TAG" --repo "$REPO" --clobber "$OUT"/checkpoint-*
    # Keep local digests to check against GitHub's afterwards.
    (cd "$OUT" && sha256sum checkpoint-* >> parts.sha256)
    rm -f "$OUT"/checkpoint-*
  done
fi
log "done: https://github.com/$REPO/releases/tag/$TAG"
