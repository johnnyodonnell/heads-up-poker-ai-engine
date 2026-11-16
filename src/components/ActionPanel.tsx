import { dollars } from "../money";
import { useState, useEffect } from "react";
import type { Phase, BetAction } from "../types";
import "./ActionPanel.css";

interface ActionPanelProps {
  phase: Phase;
  humanCanAct: boolean;
  canCheck: boolean;
  callAmount: number;
  minRaiseAmount: number;
  maxRaiseAmount: number;
  onAction: (action: BetAction) => void;
  onDeal: () => void;
  gameOver: boolean;
  onNewGame: () => void;
  isProcessingBots: boolean;
}

// Chip denominations, smallest first (one chip unit is one dollar).
const DENOMINATIONS = [1, 5, 25, 100] as const;

function Chip({
  value,
  onClick,
  disabled = false,
  label,
}: {
  value: number;
  onClick: () => void;
  disabled?: boolean;
  label: string;
}) {
  return (
    <button
      type="button"
      className={`bet-chip bet-chip-${value}`}
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={label}
    >
      <span className="bet-chip-face">{value}</span>
    </button>
  );
}

export default function ActionPanel({
  phase,
  humanCanAct,
  canCheck,
  callAmount,
  minRaiseAmount,
  maxRaiseAmount,
  onAction,
  onDeal,
  gameOver,
  onNewGame,
  isProcessingBots,
}: ActionPanelProps) {
  // The "raise to" / "bet" total, kept between the minimum raise and all-in.
  const [raiseAmount, setRaiseAmount] = useState(minRaiseAmount);

  // Start every decision at the minimum raise.
  useEffect(() => {
    setRaiseAmount(minRaiseAmount);
  }, [phase, humanCanAct, isProcessingBots, minRaiseAmount, maxRaiseAmount]);

  if (gameOver) {
    return (
      <div className="action-panel">
        <button className="btn btn-deal" onClick={onNewGame}>
          New Game
        </button>
      </div>
    );
  }

  if (phase === "waiting" || phase === "hand_complete") {
    return (
      <div className="action-panel">
        <button className="btn btn-deal" onClick={onDeal}>
          Deal
        </button>
      </div>
    );
  }

  if (!humanCanAct || isProcessingBots) {
    return null;
  }

  const canRaise = minRaiseAmount > 0 && maxRaiseAmount > 0;
  // With a choice of sizes, chips step the bet up and down; otherwise the
  // only raise is all-in.
  const showChips = canRaise && minRaiseAmount < maxRaiseAmount;
  // Nothing to call after the flop means this is an opening bet, not a raise.
  const raiseVerb = canCheck && phase !== "preflop" ? "Bet" : "Raise to";
  const allIn = raiseAmount >= maxRaiseAmount;

  // A chip that would take the bet past all-in, or below the minimum raise,
  // can't be used (and is dimmed).
  const canAdd = (value: number) => raiseAmount + value <= maxRaiseAmount;
  const canRemove = (value: number) => raiseAmount - value >= minRaiseAmount;
  const addChip = (value: number) => setRaiseAmount((r) => r + value);
  const removeChip = (value: number) => setRaiseAmount((r) => r - value);

  return (
    <div className="action-panel">
      {showChips && (
        <>
          <div className="chip-row" aria-label="Take chips off the bet">
            <span className="chip-row-sign">−</span>
            {DENOMINATIONS.map((value) => (
              <Chip
                key={value}
                value={value}
                onClick={() => removeChip(value)}
                disabled={!canRemove(value)}
                label={`Remove ${dollars(value)}`}
              />
            ))}
          </div>
          <div className="chip-row" aria-label="Add chips to the bet">
            <span className="chip-row-sign">+</span>
            {DENOMINATIONS.map((value) => (
              <Chip
                key={value}
                value={value}
                onClick={() => addChip(value)}
                disabled={!canAdd(value)}
                label={`Add ${dollars(value)}`}
              />
            ))}
          </div>
        </>
      )}
      <div className="action-buttons">
        {canCheck ? (
          <button className="btn btn-check" onClick={() => onAction({ type: "check" })}>
            Check
          </button>
        ) : (
          <>
            <button className="btn btn-fold" onClick={() => onAction({ type: "fold" })}>
              Fold
            </button>
            <button className="btn btn-call" onClick={() => onAction({ type: "call" })}>
              Call {dollars(callAmount)}
            </button>
          </>
        )}

        {canRaise &&
          (showChips ? (
            <button
              className="btn btn-raise"
              onClick={() => onAction({ type: "raise", amount: raiseAmount })}
            >
              {allIn ? "All-In" : raiseVerb} {dollars(raiseAmount)}
            </button>
          ) : (
            <button
              className="btn btn-raise"
              onClick={() => onAction({ type: "raise", amount: maxRaiseAmount })}
            >
              All-In {dollars(maxRaiseAmount)}
            </button>
          ))}
      </div>
    </div>
  );
}
