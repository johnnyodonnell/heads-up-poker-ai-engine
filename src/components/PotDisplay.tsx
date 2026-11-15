import { dollars } from "../money";
import ChipStack from "./ChipStack";
import { POT_LABEL_POS, COLLECTED_POT_POS, at } from "./layout";
import "./PotDisplay.css";

interface PotDisplayProps {
  // Chips gathered from earlier streets (shown as a stack under the board).
  collected: number;
  // Everything in the middle including this street's bets ("Pot N" label).
  liveTotal: number;
}

export default function PotDisplay({ collected, liveTotal }: PotDisplayProps) {
  if (liveTotal === 0) return null;

  return (
    <>
      <div className="pot-label" style={at(POT_LABEL_POS)}>
        <span className="pot-word">Pot</span> {dollars(liveTotal)}
      </div>
      {collected > 0 && <ChipStack amount={collected} pos={COLLECTED_POT_POS} />}
    </>
  );
}
