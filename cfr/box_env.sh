# Source this on asus-nvidia before building or running anything.
#
# Everything this project puts on the box lives under one folder, $HU_HOME:
#
#   repo/          this repository (synced by cfr/sync_to_box.sh)
#   cargo-home/    Cargo's download cache (CARGO_HOME), kept out of ~/.cargo
#   tmp/           scratch space (TMPDIR)
#   data/          hand-group (abstraction) tables
#   runs/          training runs: checkpoints and logs
#   strategies/    exported strategies the engine serves
#   logs/          engine service logs
#   web/           the built web app the engine serves
#
# The Rust toolchain already installed under ~/.rustup is used as-is.

export HU_HOME="$HOME/Workspace/heads-up-poker-ai-engine"
export CARGO_HOME="$HU_HOME/cargo-home"
export TMPDIR="$HU_HOME/tmp"
export PATH="$HOME/.cargo/bin:$PATH"
mkdir -p "$CARGO_HOME" "$TMPDIR" "$HU_HOME/data" "$HU_HOME/runs" \
  "$HU_HOME/strategies" "$HU_HOME/logs" "$HU_HOME/web"
