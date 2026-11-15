// The neural AI engine — an async drop-in for random.ts.
//
//   await bestMove(state)  // -> an Action for state.toAct (one forward pass, no search)
//   preload()              // optional: warm the ONNX session before the first move
//
// Encode the acting bot's view, run the net, and *sample* its move: an action
// type from the masked softmax (fold, check/call, bet/raise, all-in), then for
// a bet/raise a position in the legal raise band from the net's Beta. Sampling
// (not argmax) keeps the mixed strategy the bot was trained to play.

import { encode } from "./encode";
import { NUM_TYPES, RAISE, sampleBeta, toEngineAction } from "./actions";
import type { Action, EngineState } from "./game";
import { evaluate, loadModel } from "./net";
import { bestMove as randomMove } from "./random";

export async function bestMove(state: EngineState): Promise<Action> {
  const e = encode(state);
  if (!e) return { type: "check" };
  const { x, space } = e;

  let out;
  try {
    out = await evaluate(x);
  } catch (err) {
    // Keep the game playable if the model can't load.
    console.error("HuNet: inference failed, playing randomly", err);
    return randomMove(state);
  }

  // Masked softmax over the legal action types.
  const legal = [...Array(NUM_TYPES).keys()].filter((t) => space.mask[t] > 0);
  const mx = Math.max(...legal.map((t) => out.typeLogits[t]));
  const w = legal.map((t) => Math.exp(out.typeLogits[t] - mx));
  const total = w.reduce((s, v) => s + v, 0);
  let r = Math.random() * total;
  let type = legal[legal.length - 1];
  for (let k = 0; k < legal.length; k++) {
    r -= w[k];
    if (r < 0) {
      type = legal[k];
      break;
    }
  }
  const u = type === RAISE ? sampleBeta(out.sizeAB[0], out.sizeAB[1]) : 0;
  return toEngineAction(space, type, u);
}

export function preload(): Promise<unknown> {
  return loadModel();
}
