# Heads-up poker bot: self-play training plan

## Context

The heads-up No-Limit Hold'em web app currently plays against a placeholder **random** bot (`src/engine/random.ts`). This plan replaces it with a neural policy trained by **self-play**. It follows the recipe in `~/Code/fox-lite-ai-engine/training`, whose search-free REINFORCE approach beat both its ISMCTS and AWR attempts. All training logic runs in Rust, and the model runs in the browser through onnxruntime-web.

`~/Code/6-max-poker-ai-engine` is used only as a source of code to adapt: its Rust rules core, its ONNX writer, and especially its **run1**, a Rust REINFORCE trainer with continuous bet sizing. Its later PPO design is not the model for this project.

The end result:
- A training process on `asus-nvidia` (GB10) that runs indefinitely, saves a snapshot on a timer, and evaluates each snapshot automatically.
- A browser engine that can load any snapshot.

A snapshot goes into the app only when the user says so.

## Agreed decisions

| Topic | Decision |
|---|---|
| Learning method | fox-lite REINFORCE: one on-policy optimizer step per cohort (a cohort is one batch of self-play hands). The advantage is z − V, using the policy net's own value head. Entropy bonus. Temperature 1 (no annealing). |
| Actions | Action type is one of fold, check/call, bet/raise or all-in. A bet/raise size is continuous: u ~ Beta(α, β) with α, β ≥ 1, over the legal band from the min-raise to one chip below all-in, measured on a log pot-fraction scale. |
| Training episodes | Single hands from sampled stacks: 40% start at 200/200, otherwise the 400 chips are split with a Beta(1.5, 1.5) share. Random button. |
| Reward | Chips won or lost ÷ 200. When an all-in hand runs out the board with no decisions left, the run-out is replaced by its expected value. |
| Code | All Rust: tch 0.26 with libtorch 2.13 taken from the PyTorch cu130 wheel. No Python code runs. The TS engine stays the source of truth, and the Rust port is parity-tested against it. |
| Rules | Fix first: a player is only asked to act when there's a real decision. |
| Browser | The bot samples from the policy and falls back to random if the model fails. |
| Evaluation | Duplicate format (each deal played twice with seats swapped) with all-in expected value, reported as bb/100 ± CI. The rating ladder is fit from snapshot-vs-snapshot results only. Results against random and a scripted opponent panel are reported separately. |
| Snapshots | Default every 4h. run1 starts at 30m and has no time limit. The cadence changes when the user asks. |
| Promotion | Never automatic. A snapshot is pulled into the app, and committed and pushed, only when the user says. |
| Git | Commit and push to `main` after each phase. No review gate. |
| Box | Everything lives under `~/Workspace/heads-up-poker-ai-engine/` on asus-nvidia (tool caches in the home directory are OK). The project has full GPU priority. Root is available via `ssh asus-nvidia-root` if needed. |

## Layout

Repo (new unless marked):
```
training/
  PLAN.md               this plan
  README.md             setup / train / evaluate / promote / parity checks
  Cargo.toml            workspace: hu_core, trainer
  .cargo/config.toml    LIBTORCH -> ../../deps/venv/lib/python3.12/site-packages/torch (relative)
  hu_core/              pure Rust: cards, hand_eval, rules, actions, encode, allin_ev, table (+ tests/)
  trainer/              tch binary `hu`: net, onnx, selfplay, train, run, eval, probe, hud, bots, main
  scripts/              TS run via `npx tsx`: check_rules, dump_fixtures, test_onnx_parity, test_engine_e2e
  sync_to_box.sh  run_daemon.sh  promote.sh
src/engine/game.ts      (modified) rules fix
src/engine/actions.ts   action space + Beta sampling (source of truth)
src/engine/encode.ts    network input encoding (source of truth)
src/engine/net.ts       onnxruntime-web session
src/engine/neural.ts    bestMove: encode -> forward -> sample
vite.config.ts          (modified) serve onnxruntime-web's wasm under /onnx-wasm/
package.json            (modified) + onnxruntime-web, + tsx (dev)
.gitignore              (modified) + training/target/, training/fixtures/
```

Box: `~/Workspace/heads-up-poker-ai-engine/`
- `repo/`: an rsync copy of this working tree.
- `deps/venv/`: libtorch only.
- `runs/<run>/`: `latest.*`, `snapshots/`, `pool.json`, `train.log`.

## Reference code to adapt

| Need | Source |
|---|---|
| Rules, 7-card evaluator, all-in expected value, card helpers | `~/Code/6-max-poker-ai-engine/training/poker_core/src/{rules,hand_eval,allin_ev,cards,table}.rs` (`State::start_hand(deck)`, `legal`, `apply`, `advance_street`, `eval7`, `is_runout`, `expected_stacks`) |
| Parity tests | `~/Code/6-max-poker-ai-engine/training/poker_core/tests/{rules_parity,eval_parity,encode_parity}.rs`, `.../training/scripts/{dump_fixtures,check_rules}.ts` |
| Continuous-size band math (`sLo`/`sHi`, `raiseToFor`) | `git -C ~/Code/6-max-poker-ai-engine show ec5b9d2:src/engine/actions.ts` (swap its truncated normal for a Beta) |
| Encoder (canonical suits, action tokens, static features) | `~/Code/6-max-poker-ai-engine/src/engine/encode.ts` + `training/poker_core/src/encode.rs` (cut from 6 seats to 2) |
| REINFORCE step, checkpointable AdamW, self-play loop, daemon | `git -C ~/Code/6-max-poker-ai-engine show bf7204b:training/trainer/src/{train,selfplay,run,hud}.rs` |
| Network blocks (`fwd` + ONNX `emit`), ONNX writer | `~/Code/6-max-poker-ai-engine/training/trainer/src/{net,onnx}.rs`; history pooling design from `~/Code/fox-lite-ai-engine/training/net.py` |
| Opponent selection + `pool.json` | `~/Code/fox-lite-ai-engine/training/selfplay_rs/src/bin/evaluate_rs.rs` (`select_anchors`) |
| Loop / snapshot / resume behavior | `~/Code/fox-lite-ai-engine/training/orchestrator.py` |
| Probes | `~/Code/6-max-poker-ai-engine/training/trainer/src/probe.rs` |
| Build, sync, run, promote scripts | `~/Code/6-max-poker-ai-engine/training/{.cargo/config.toml,trainer/build.rs,sync_to_box.sh,run_daemon.sh,promote.sh}` |
| Browser ONNX plumbing | `~/Code/6-max-poker-ai-engine/{vite.config.ts,src/engine/net.ts,src/engine/neural.ts}`, `training/scripts/{test_onnx_parity,test_engine_e2e}.ts` |

## Phases

Each phase ends at its gate, then a commit and push to `main`.

### 0. Plan and rules fix (TS)
- Commit this plan as `training/PLAN.md`.
- `src/engine/game.ts`:
  - Add `owesAction(p)`: true when `p` is active and either `p.bet < currentBet`, or `p` hasn't acted and the opponent is still active.
  - Use it for the first actor in `startHand` (game.ts:248, today `status === "active"`), in `nextActor` (:483) and in `bettingClosed` (:469).
  - `startHand` sets `currentBet` to the largest blind actually posted (today it's the `BIG_BLIND` constant, :265).
  - `legalActions` (:290) offers bet/raise only while the opponent is still active.
  - No other changes are needed: `applyAction` already turns an illegal bet/raise/all-in into a call or check (:377). In the UI, `adapt.ts` already hides the slider when raising isn't legal, and `useGameState.advance` already runs out `street-complete` hands.
- `training/scripts/check_rules.ts`, modeled on 6-max's scenario style. Scenarios:
  - SB all-in for 1 chip → no decision; the hand runs out with the correct payout.
  - BB all-in for 1 chip → no decision.
  - BB all-in for exactly 2 chips → SB gets fold/call only.
  - Facing a shove, preflop or on the flop → fold/call only.
  - Regressions: normal open, 3-bet and call lines; a short all-in doesn't reopen the betting; an uncalled bet is returned; the odd chip in a split pot goes to the right player.
- Add `tsx` as a dev dependency.
- **Gate:** `npx tsx training/scripts/check_rules.ts` and `npm run build` both pass.

### 1. Box setup
- Create the box directory.
- `uv venv deps/venv --python 3.12`, then `uv pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0`. This is the libtorch source only.
- `training/sync_to_box.sh`: `rsync -a --delete`, excluding `.git/ node_modules/ dist/ training/target/`.
- Workspace `Cargo.toml`, `.cargo/config.toml` (relative `LIBTORCH`), and `trainer/build.rs`, which sets an rpath to libtorch and force-links `torch_cuda` so CUDA is visible.
- **Gate:** `hu gpu-check` reports CUDA and runs a matmul on the GPU.

### 2. Rules core and parity
- `hu_core`: port 6-max's `poker_core` rules, evaluator, cards and all-in EV to two seats, matching heads-up `game.ts`:
  - The button posts the SB.
  - `returnUncalled`.
  - The Phase 0 fixes.
  - The match is over when one player holds every chip.
  - `start_hand(deck)` takes the deck as an argument so TS deals replay exactly.
- `table.rs`: the training-hand sampler (stacks as described above, random button).
- `scripts/dump_fixtures.ts`:
  - Random-stack matches, including 1–2 chip stacks so blind all-ins happen. Each step records the deck, the legal set and the state after the step.
  - Random 7-card hands with their `evaluate7` scores.
- `hu_core/tests/{rules_parity,eval_parity}.rs`.
- **Gate:** `cargo test --release -p hu_core` matches TS over thousands of hands.

### 3. Action space and encoder (TS first, Rust mirror)
- `src/engine/actions.ts`:
  - Type mask from `legalActions`. Bet/raise needs room for a raise that isn't all-in (`minRaiseTo < maxRaiseTo`).
  - Band: s = ln((raiseTo − currentBet) / potAfterCall) over [s(minRaiseTo), s(allIn − 1)].
  - Size: s = sLo + u·(sHi − sLo). raiseTo = round(currentBet + e^s · potAfterCall), clamped to the band.
  - A Beta sampler (two Gamma draws, Marsaglia–Tsang) and `toEngineAction`.
- `src/engine/encode.ts` (from the mover's point of view, suits relabeled canonically):
  - 7 card indices (hole, flop, turn, river; padded with NO_CARD). The network turns them into one-hot suit×rank planes.
  - Up to 32 action tokens, most recent first: street, actor (me or opponent), action type, all-in flag, size as log pot-fraction, call fraction, log-bb chip amounts, valid bit.
  - Static features: street, position, stacks/bets/commits, pot, amount to call, pot odds, stack-to-pot ratio, the size band, and the type mask.
- `hu_core/src/{actions,encode}.rs` mirror both; Beta sampling uses `rand_distr::Beta`. The fixtures are extended with the encodings, masks and bands.
- **Gate:** bit-exact f32 parity (`encode_parity.rs`).

### 4. Network and ONNX export
- `trainer/src/net.rs`. The network is defined once, and each block has a `fwd` (tch) and an `emit` (ONNX):
  - Cards → one-hot suit×rank planes (hole, flop, turn, river, board, all cards) plus rank and suit counts → Linear → GELU.
  - Action tokens → projection + position → 2 pre-LN attention layers (d_model 128, 4 heads) → masked mean + masked max + 4 learned-query readouts.
  - Concatenated with the static features → Linear(512) → 4 residual blocks.
  - Heads:
    - Type logits [4]. The initial bias makes all-in rare, as in 6-max run1.
    - Size Beta (α, β) = 1 + softplus(·), initialized toward the low end of the band (roughly half-pot to pot).
    - Value: tanh, training only.
  - Roughly 2–3M parameters.
- `trainer/src/onnx.rs`: port of 6-max's minimal ONNX writer. The export takes `input [batch, INPUT_SIZE]` and outputs `type_logits [batch, 4]` and `size_ab [batch, 2]`. The value head is left out.
- `hu export --weights W --out M.onnx --fixture fx.json` and `scripts/test_onnx_parity.ts` (onnxruntime-web in Node).
- **Gate:** onnxruntime-web vs tch, max |Δ| < 1e-4.

### 5. Self-play
- `trainer/src/selfplay.rs`:
  - 4,096 hands in flight. Each step: rayon CPU advance + encode → one batched GPU forward → rayon sample + apply. (fox-lite's multi-threaded pipeline is only ported if profiling shows self-play is the bottleneck.)
  - Sampling: the type from the masked softmax. For bet/raise, u ~ Beta(α, β), clamped to [1e-6, 1 − 1e-6].
  - Each row stores: encoding, type mask, band, type, u, behavior log-prob, z.
  - A cohort is a fixed number of hands, all played to completion, as in fox-lite (nothing discarded).
  - z per seat = (end stack − start stack) / 200, using `expected_stacks` once `is_runout` holds: exact when ≤2 cards are to come, sampled preflop.
- `trainer/src/hud.rs` (per cohort): VPIP and PFR by position, 3-bet, c-bet, fold to c-bet, WTSD, the action mix, and bet-size quantiles. These are the early warning for swings between nit and maniac play and for collapse to min-raises.
- `hu selfplay --weights W --hands N` runs one cohort on its own, for profiling.
- **Gate:**
  - Every sampled action is legal.
  - z ∈ [−1, 1], and the two seats' z values sum to 0 for every hand.
  - Throughput is logged.

### 6. Trainer and daemon
- `trainer/src/train.rs` (6-max run1's `train.rs` with the truncated normal swapped for a Beta):
  - log π = log_softmax(masked type logits)[type] + [bet/raise] · log Beta(u; α, β)
  - loss = −mean(stopgrad(z − V) · log π) + c_v · MSE(V, z) − c_t · H(type) − c_s · H(Beta)
  - One AdamW step per cohort, with gradients accumulated over micro-batches. The AdamW moments are saved in checkpoints.
  - Logged: loss terms, both entropies, mean α and β, grad norm, `logp_gap` (sampler vs trainer log-prob, ≈0 means on-policy), and approximate KL.
- `trainer/src/run.rs` (`hu train`) runs fox-lite's strict loop in one process:
  1. Self-play one cohort.
  2. Take one training step.
  3. Save `latest.*` atomically.
  4. If a snapshot is due, write `snapshots/snap_hHHHHH_cCCCCCCC.safetensors`, then run eval and probes.
  5. Log one JSON line.
  - Flags: `--snapshot-every` (default `4h`), `--max-cohorts`, `--seed`.
- `training/run_daemon.sh`: starts or resumes with nohup, logging to `runs/$RUN/train.log`.
- **Gate:** a smoke run (`--max-cohorts 50 --snapshot-every 2m`) shows finite losses, `logp_gap` ≈ 0, and bb/100 vs random going up. Killing and restarting resumes cleanly.

### 7. Evaluation and probes
- `trainer/src/eval.rs`:
  - A match is D deals, each played twice with seats swapped.
  - All-in expected value is applied, and both sides sample their moves.
  - Forward passes are batched across deals.
  - The result is the candidate's bb/100 ± 95% CI.
- Automatically, for each snapshot:
  - Play the top 2 rated snapshots, 3 anchor snapshots spread across the rating range, and the previous snapshot (fox-lite's `select_anchors`).
  - Refit least-squares ratings over all snapshot-vs-snapshot results, with the first snapshot fixed at 0, and store them in `pool.json`.
- Reported separately:
  - `random`, a Rust port of `random.ts` (fox-lite's collapse alarm).
  - The scripted panel in `trainer/src/bots.rs`: calling station, maniac, nit and shover.
- `trainer/src/probe.rs`:
  - Preflop fold and raise rates vs equity by hand class (rank correlation).
  - AA never folds preflop.
  - Flop fold rates for sets, top pair and air.
  - The bet-size distribution.
- Subcommands: `hu eval`, `hu match --a X --b Y --deals N`, `hu probe`, `hu best`.
- **Gate:**
  - Random vs random, and a snapshot vs itself, both come out at 0 within the CI.
  - Evaluating one snapshot takes a few minutes at most, with about 10k duplicate deal pairs per opponent.

### 8. Launch run1
- Sync, build, then `RUN=run1 SNAPSHOT_EVERY=30m training/run_daemon.sh`. No time limit.
- Stay with it through the first snapshot's eval and probes to confirm the whole loop works.

### 9. Browser engine and docs (while run1 trains)
- Add `onnxruntime-web`. `vite.config.ts` gets 6-max's `/onnx-wasm/` plugin.
- `src/engine/net.ts`: lazy session, WebGPU with WASM fallback, retry after a failure.
- `src/engine/neural.ts`: encode → forward → sample the type → for bet/raise, sample u ~ Beta → `toEngineAction`. Uses `random.ts` on any failure. Also provides `preload()`.
- `scripts/test_engine_e2e.ts`: thousands of hands in Node with an exported snapshot running through `neural.ts` against random, checking that every bot action is legal.
- `useGameState.ts:11` stays on `../engine/random`, and no `public/models/current.onnx` is committed.
- `training/README.md` and a training section in the root `README.md`.
- **Gate:** `npm run build` passes, and the e2e test passes.
- Then hand off: how to watch the run, change the cadence, and pull in a snapshot.

## Training defaults (run1 starting points)

| Setting | Value |
|---|---|
| Hands per cohort / in flight | 32,768 / 4,096 |
| Optimizer | AdamW: lr 3e-4 (100-step warmup), weight decay 1e-4, grad clip 1.0 |
| Loss weights | value 1.0 (MSE), type entropy 0.01, size entropy 0.001 |
| Sampling temperature | 1 |
| Snapshots | `--snapshot-every` default `4h`; run1 uses `30m` |
| Eval | top 2 + 3 spread anchors + previous snapshot; about 10k duplicate deal pairs each |

## Operations
- **Sync and build:** run `training/sync_to_box.sh`, then on the box `cd ~/Workspace/heads-up-poker-ai-engine/repo/training && cargo build --release`.
- **Start or resume:** `RUN=run1 SNAPSHOT_EVERY=30m training/run_daemon.sh`.
- **Watch:** `tail -f runs/run1/train.log` (one JSON line per cohort, plus `[snapshot]`, `[eval]` and `[probe]` lines) and `runs/run1/pool.json`.
- **Change cadence (when asked):** stop the process and restart with the new `SNAPSHOT_EVERY`. It resumes from `latest.*`. The already-scheduled snapshot keeps its time, and later ones use the new interval.
- **Pull a snapshot in (when asked):** `training/promote.sh <snapshot>` exports on the box, copies to `public/models/current.onnx`, and runs ONNX parity + e2e. The first time, `useGameState.ts:11` also switches to `../engine/neural`. Commit and push only when told.

## Verification
1. **Rules:** `npx tsx training/scripts/check_rules.ts` and `npm run build`. Then `npm run dev` and play a few hands, including facing a shove: the raise option should be gone.
2. **Port parity, on the box:** `cargo test --release -p hu_core` (rules, evaluator and encoder vs the TS fixtures).
3. **Model path:** `hu export` → `npx tsx training/scripts/test_onnx_parity.ts` (onnxruntime-web vs tch) → `npx tsx training/scripts/test_engine_e2e.ts` (legal play through `neural.ts`).
4. **Training:**
   - Smoke run plus a resume test.
   - On run1's first snapshot: an `[eval]` line, an updated `pool.json`, sensible `[probe]` output, and plausible HUD stats.
5. **When a snapshot is pulled in:** the `promote.sh` checks pass, and in `npm run dev` the bot makes sampled, legal moves with no console errors.

## If run1 trains weak (levers, in order)
0. fox-lite's own levers: larger cohorts, adjusted entropy coefficients.
1. A value baseline that also sees the opponent's hole cards and equity (training only; the exported model doesn't change).
2. An anchor pull: a KL term toward a past snapshot that beat its predecessor (still REINFORCE).
3. PPO (as in 6-max run2).
