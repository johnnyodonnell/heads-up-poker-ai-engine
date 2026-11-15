#!/usr/bin/env bash
# Copy this working tree (including uncommitted changes) to the training box:
#   asus-nvidia:~/Workspace/heads-up-poker-ai-engine/repo/
#
#   training/sync_to_box.sh
#
# Build output, node_modules and .git are left alone, so the box keeps its
# compiled crates between syncs. Everything for this project on the box lives
# under ~/Workspace/heads-up-poker-ai-engine/ (repo/, deps/, runs/).
set -euo pipefail
cd "$(dirname "$0")/.."
rsync -a --delete \
  --exclude .git/ \
  --exclude node_modules/ \
  --exclude dist/ \
  --exclude .playwright-mcp/ \
  --exclude 'training/target/' \
  --exclude '*.tsbuildinfo' \
  ./ asus-nvidia:Workspace/heads-up-poker-ai-engine/repo/
