import type { BotDecision } from "../engine/remote";
import "./DebugPanel.css";

interface DebugPanelProps {
  enabled: boolean;
  onToggle: () => void;
  decisions: BotDecision[];
}

const pct = (p: number) => `${Math.round(p * 100)}%`;

// The debug view: a toggle, and (when on) the bot's decisions this hand —
// which strategy it used, its hand group, the options it chose between with
// their probabilities, and what it picked. The bot's cards are shown face up
// on the table while it's on (see App).
export default function DebugPanel({ enabled, onToggle, decisions }: DebugPanelProps) {
  return (
    <div className="debug">
      <button className={`debug-toggle ${enabled ? "on" : ""}`} onClick={onToggle}>
        Debug: {enabled ? "on" : "off"}
      </button>
      {enabled && (
        <div className="debug-panel">
          <div className="debug-title">
            Bot's decisions{decisions.length ? ` — hand #${decisions[0].hand}` : ""}
          </div>
          {decisions.length === 0 && <div className="debug-empty">None yet this hand.</div>}
          {decisions.map((d, i) => (
            <div key={i} className="debug-decision">
              <div className="debug-head">
                <span className="debug-street">{d.debug.street}</span>
                {d.debug.bucket_label && <span> · {d.debug.bucket_label}</span>}
                <span className="debug-dim">
                  {" "}
                  · {d.debug.effective_bb}bb deep, {d.debug.strategy_bb}bb strategy
                </span>
              </div>
              {d.debug.path.length > 0 && (
                <div className="debug-path" title="The betting so far, as the strategy saw it">
                  {d.debug.path.join(" → ")}
                </div>
              )}
              {d.debug.fallback ? (
                <div className="debug-fallback">Off the strategy's map ({d.debug.fallback}); used a simple equity rule.</div>
              ) : (
                <table className="debug-options">
                  <tbody>
                    {d.debug.options.map((o, j) => (
                      <tr key={j} className={o.prob >= 0.005 ? "" : "debug-zero"}>
                        <td className="debug-opt">{o.action}</td>
                        <td className="debug-bar">
                          <div style={{ width: pct(o.prob) }} />
                        </td>
                        <td className="debug-prob">{pct(o.prob)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              <div className="debug-chosen">→ {d.debug.chosen}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
