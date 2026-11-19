import { useState, useCallback, useEffect, useRef } from "react";
import type { BetAction, GameState } from "../types";
import {
  createTable,
  startHand,
  applyAction,
  advanceStreet,
} from "../engine/game";
import type { EngineState } from "../engine/game";
import { loadTable, saveTable } from "../engine/saved";
// The bot's engine. Swap this one import to plug in a different engine.
import { bestMove, onDecision, thinkingMs } from "../engine/remote";
import type { BotDecision } from "../engine/remote";
import { toGameState, fromBetAction } from "../engine/adapt";

// Thrown when the bot can't move; carries the state reached so far (e.g. a
// street dealt before the bot's decision) so the table shows where it stopped.
class AdvanceError extends Error {
  constructor(
    message: string,
    public state: EngineState,
  ) {
    super(message);
  }
}

// The bot's decision being worked out: its seat, how long the thinking bar
// runs (ms, 0 = no bar) and an id per decision (restarts the bar).
export interface Thinking {
  seat: number;
  ms: number;
  id: number;
}

// Run the engine forward until a human decision is required or the hand/match
// ends. The bot's move is awaited (it comes from the engine service). Before
// each bot decision `onThink` gets the state it's deciding in (so the table
// shows a newly dealt card while the bot thinks), and null once it has moved;
// otherwise there is no pacing. Bounded by a generous step cap purely as a
// runaway-loop backstop.
async function advance(
  state: EngineState,
  onThink: (s: EngineState | null) => void,
): Promise<EngineState> {
  let s = state;
  for (let step = 0; step < 10000; step++) {
    if (s.phase === "street-complete") {
      s = advanceStreet(s);
      continue;
    }
    if (s.phase === "betting") {
      const actor = s.players[s.toAct];
      if (actor && !actor.isHuman) {
        let action;
        onThink(s);
        try {
          action = await bestMove(s);
        } catch (err) {
          throw new AdvanceError(err instanceof Error ? err.message : String(err), s);
        } finally {
          onThink(null);
        }
        s = applyAction(s, action);
        continue;
      }
    }
    break; // human to act, or hand-over / match-over / waiting
  }
  return s;
}

export function useGameState(): {
  gameState: GameState;
  engineState: EngineState;
  dealHand: () => void;
  performAction: (action: BetAction) => void;
  newGame: () => void;
  retry: () => void;
  gameOver: boolean;
  isProcessingBots: boolean;
  thinking: Thinking | null;
  engineError: string | null;
  decisions: BotDecision[];
} {
  const [engine, setEngine] = useState<EngineState>(loadTable);
  const [busy, setBusy] = useState(false);
  const [engineError, setEngineError] = useState<string | null>(null);
  const [decisions, setDecisions] = useState<BotDecision[]>([]);
  const [thinking, setThinking] = useState<Thinking | null>(null);
  const thinkId = useRef(0);
  const engineRef = useRef(engine);
  // Bumped by every new run (and New Game): a bot run that finishes after
  // it has been superseded is discarded.
  const generation = useRef(0);

  // The bot's decisions this hand, for the debug view.
  useEffect(
    () =>
      onDecision((d) =>
        setDecisions((prev) => (prev.length && prev[0].hand !== d.hand ? [d] : [...prev, d])),
      ),
    [],
  );

  const set = useCallback((s: EngineState) => {
    engineRef.current = s;
    setEngine(s);
    saveTable(s);
  }, []);

  // Show `start`, then let the bot play until the human is up again.
  const run = useCallback(
    async (start: EngineState) => {
      const my = ++generation.current;
      set(start);
      setBusy(true);
      setEngineError(null);
      const onThink = (s: EngineState | null) => {
        if (generation.current !== my) return;
        if (!s) {
          setThinking(null);
          return;
        }
        if (s !== engineRef.current) set(s);
        setThinking({ seat: s.toAct, ms: thinkingMs(s), id: ++thinkId.current });
      };
      try {
        const end = await advance(start, onThink);
        if (generation.current !== my) return;
        set(end);
      } catch (err) {
        if (generation.current !== my) return;
        if (err instanceof AdvanceError) {
          set(err.state);
          setEngineError(err.message);
        } else {
          setEngineError(String(err));
        }
      }
      setBusy(false);
    },
    [set],
  );

  const dealHand = useCallback(() => {
    const s = engineRef.current;
    if (busy || engineError || (s.phase !== "waiting" && s.phase !== "hand-over")) return;
    setDecisions([]);
    void run(startHand(s));
  }, [busy, engineError, run]);

  const performAction = useCallback(
    (action: BetAction) => {
      const s = engineRef.current;
      if (busy || engineError || s.phase !== "betting" || !s.players[s.toAct]?.isHuman) return;
      void run(applyAction(s, fromBetAction(action, s)));
    },
    [busy, engineError, run],
  );

  // Ask the engine again from where the bot was stuck.
  const retry = useCallback(() => {
    if (busy) return;
    void run(engineRef.current);
  }, [busy, run]);

  const newGame = useCallback(() => {
    generation.current++;
    set(createTable());
    setBusy(false);
    setThinking(null);
    setEngineError(null);
    setDecisions([]);
  }, [set]);

  return {
    gameState: toGameState(engine),
    engineState: engine,
    dealHand,
    performAction,
    newGame,
    retry,
    gameOver: engine.phase === "match-over",
    isProcessingBots: busy,
    thinking,
    engineError,
    decisions,
  };
}
