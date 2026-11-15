// Fixed-size "stage" geometry, modelled on the CoinPoker table. The whole
// stage is laid out in these design pixels and then scaled to fit the window
// (see GameTable), so every position here is absolute.
//
// Heads-up: the human (seat 0) sits bottom-center, the bot (seat 1)
// top-center.

export const STAGE_WIDTH = 1340;
export const STAGE_HEIGHT = 940;

export interface Point {
  x: number;
  y: number;
}

// Center of each seat's name/stack plate.
export const SEAT_POS: Point[] = [
  { x: 670, y: 845 },
  { x: 670, y: 172 },
];

// Where each seat's current-street bet sits on the felt.
export const BET_POS: Point[] = [
  { x: 670, y: 628 },
  { x: 670, y: 245 },
];

// Where the dealer button sits when that seat has the button.
export const DEALER_POS: Point[] = [
  { x: 545, y: 700 },
  { x: 808, y: 236 },
];

export const POT_LABEL_POS: Point = { x: 670, y: 318 };
export const BOARD_POS: Point = { x: 670, y: 434 };
export const COLLECTED_POT_POS: Point = { x: 670, y: 552 };
export const RESULT_POS: Point = { x: 670, y: 612 };

export function at(p: Point): React.CSSProperties {
  return { left: p.x, top: p.y };
}
