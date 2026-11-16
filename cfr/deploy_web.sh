#!/usr/bin/env bash
# Build the web app here and copy it to the box, where the engine service
# serves it: http://192.168.0.185:8790
set -euo pipefail
cd "$(dirname "$0")/.."
npm run build
rsync -a --delete dist/ asus-nvidia:Workspace/heads-up-poker-ai-engine/web/
echo "deployed to http://192.168.0.185:8790"
