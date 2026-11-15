//! Heads-up No-Limit Hold'em rules: a faithful port of `src/engine/game.ts`.
//!
//! Same chip model (blinds 1/2, integer chips), same seat order, same phases,
//! same legal-action logic (short-all-in raise rights; a player is only asked
//! to act when there's a real decision), uncalled bets returned before the
//! bets are swept, same odd-chip rule. Where TS returns a fresh state, this
//! mutates in place. Parity is checked against traces dumped from the TS
//! engine.
//!
//! Heads-up: the button posts the small blind and acts first pre-flop; the
//! big blind acts first on every later street.

use crate::cards::Card;
use crate::hand_eval::eval7;

pub const SEATS: usize = 2;
pub const SMALL_BLIND: i32 = 1;
pub const BIG_BLIND: i32 = 2;
pub const STARTING_STACK: i32 = 200;
/// Chips in play: the match is a freeze-out between two 200-chip stacks.
pub const TOTAL_CHIPS: i32 = SEATS as i32 * STARTING_STACK;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Active,
    AllIn,
    Folded,
    Out,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Street {
    Preflop,
    Flop,
    Turn,
    River,
}

impl Street {
    pub fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Waiting,
    Betting,
    StreetComplete,
    HandOver,
    MatchOver,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionType {
    Fold,
    Check,
    Call,
    Bet,
    Raise,
    AllIn,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Action {
    pub kind: ActionType,
    /// Raise-to total for bet/raise (None = the minimum).
    pub amount: Option<i32>,
}

impl Action {
    pub fn new(kind: ActionType) -> Self {
        Action { kind, amount: None }
    }
    pub fn to(kind: ActionType, amount: i32) -> Self {
        Action { kind, amount: Some(amount) }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Player {
    pub stack: i32,
    pub hole: Vec<Card>,
    pub status: Status,
    pub bet: i32,
    pub commit: i32,
    pub has_acted: bool,
    pub acted_at: i32,
    pub last_action: Option<ActionType>,
}

/// One betting action this hand (TS `HandAction`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HandAction {
    pub seat: usize,
    pub street: Street,
    /// fold / check / call / bet / raise (never AllIn — see `allin`).
    pub kind: ActionType,
    pub allin: bool,
    pub to: i32,
    pub added: i32,
    pub pot_before: i32,
    pub current_bet_before: i32,
    pub stack_before: i32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Legal {
    pub can_fold: bool,
    pub can_check: bool,
    pub can_call: bool,
    /// bet (opening) or raise is offered.
    pub can_raise: bool,
    pub opening: bool,
    pub to_call: i32,
    pub call_to: i32,
    pub min_raise_to: i32,
    pub max_raise_to: i32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PotResult {
    pub amount: i32,
    pub winners: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PotLayer {
    pub amount: i32,
    pub eligible: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct State {
    pub players: Vec<Player>,
    pub button: i32,
    pub sb_seat: i32,
    pub bb_seat: i32,
    pub hand_number: u32,
    pub street: Street,
    pub community: Vec<Card>,
    pub deck: Vec<Card>,
    pub pot: i32,
    pub current_bet: i32,
    pub min_raise: i32,
    pub to_act: i32,
    pub phase: Phase,
    pub results: Option<Vec<PotResult>>,
    pub actions: Vec<HandAction>,
}

fn next_seat(players: &[Player], from: i32, pred: impl Fn(&Player) -> bool) -> i32 {
    for step in 1..=SEATS as i32 {
        let i = (from + step).rem_euclid(SEATS as i32);
        if pred(&players[i as usize]) {
            return i;
        }
    }
    -1
}

/// Whether some other player could still respond to seat `i` (is in the hand
/// with chips behind). TS `opponentCanAct`.
fn opponent_can_act(players: &[Player], i: usize) -> bool {
    players.iter().enumerate().any(|(j, q)| j != i && q.status == Status::Active)
}

/// Whether seat `i` has a real decision: it can act, and either owes chips to
/// match the bet, or hasn't acted while the opponent can still respond.
/// TS `owesAction`.
fn owes_action(players: &[Player], current_bet: i32, i: usize) -> bool {
    let p = &players[i];
    p.status == Status::Active
        && (p.bet < current_bet || (!p.has_acted && opponent_can_act(players, i)))
}

/// The first seat after `from` (clockwise) that owes an action, or -1.
fn next_actor(players: &[Player], current_bet: i32, from: i32) -> i32 {
    for step in 1..=SEATS as i32 {
        let i = (from + step).rem_euclid(SEATS as i32);
        if owes_action(players, current_bet, i as usize) {
            return i;
        }
    }
    -1
}

impl State {
    /// A seated table, no hand dealt (TS `createTable`, with given stacks).
    pub fn new_table(stacks: [i32; SEATS]) -> State {
        State {
            players: stacks
                .iter()
                .map(|&stack| Player {
                    stack,
                    hole: Vec::new(),
                    status: Status::Active,
                    bet: 0,
                    commit: 0,
                    has_acted: false,
                    acted_at: 0,
                    last_action: None,
                })
                .collect(),
            button: -1,
            sb_seat: -1,
            bb_seat: -1,
            hand_number: 0,
            street: Street::Preflop,
            community: Vec::new(),
            deck: Vec::new(),
            pot: 0,
            current_bet: 0,
            min_raise: BIG_BLIND,
            to_act: -1,
            phase: Phase::Waiting,
            results: None,
            actions: Vec::new(),
        }
    }

    /// Deal the next hand from `deck` (a full shuffled deck; cards are popped
    /// off the end, as in TS). TS `startHand`.
    pub fn start_hand(&mut self, deck: Vec<Card>) {
        for p in self.players.iter_mut() {
            p.hole.clear();
            p.bet = 0;
            p.commit = 0;
            p.has_acted = false;
            p.acted_at = 0;
            p.last_action = None;
            p.status = if p.stack > 0 { Status::Active } else { Status::Out };
        }
        if self.players.iter().filter(|p| p.stack > 0).count() < 2 {
            self.phase = Phase::MatchOver;
            return;
        }

        let button = next_seat(&self.players, self.button, |p| p.stack > 0);
        let sb = button;
        let bb = next_seat(&self.players, sb, |p| p.stack > 0);
        post_blind(&mut self.players[sb as usize], SMALL_BLIND);
        post_blind(&mut self.players[bb as usize], BIG_BLIND);

        let mut deck = deck;
        for p in self.players.iter_mut() {
            if p.status != Status::Out {
                let a = deck.pop().expect("deck");
                let b = deck.pop().expect("deck");
                p.hole = vec![a, b];
            }
        }
        // The bet to match is the largest blind actually posted.
        let current_bet = self.players.iter().map(|p| p.bet).max().unwrap_or(0);
        let first = next_actor(&self.players, current_bet, bb);

        self.button = button;
        self.sb_seat = sb;
        self.bb_seat = bb;
        self.hand_number += 1;
        self.street = Street::Preflop;
        self.community.clear();
        self.deck = deck;
        self.pot = 0;
        self.current_bet = current_bet;
        self.min_raise = BIG_BLIND;
        self.to_act = first;
        self.phase = if first == -1 { Phase::StreetComplete } else { Phase::Betting };
        self.results = None;
        self.actions.clear();
    }

    fn has_raise_rights(&self, p: &Player) -> bool {
        !p.has_acted || self.current_bet - p.acted_at >= self.min_raise
    }

    /// What the player to act may do (TS `legalActions`).
    pub fn legal(&self) -> Option<Legal> {
        if self.to_act < 0 {
            return None;
        }
        let i = self.to_act as usize;
        let p = &self.players[i];
        if p.status != Status::Active {
            return None;
        }
        let to_call = self.current_bet - p.bet;
        let max_to = p.bet + p.stack;
        let opening = self.current_bet == 0;
        let min_to = (if opening { BIG_BLIND } else { self.current_bet + self.min_raise }).min(max_to);
        Some(Legal {
            can_fold: to_call > 0,
            can_check: to_call <= 0,
            can_call: to_call > 0,
            can_raise: max_to > self.current_bet
                && self.has_raise_rights(p)
                && opponent_can_act(&self.players, i),
            opening,
            to_call: to_call.min(p.stack),
            call_to: self.current_bet.min(max_to),
            min_raise_to: min_to,
            max_raise_to: max_to,
        })
    }

    /// Apply the action of the player to act (TS `applyAction`).
    pub fn apply(&mut self, action: Action) {
        let Some(legal) = self.legal() else { return };
        let i = self.to_act as usize;
        let mut current_bet = self.current_bet;
        let mut min_raise = self.min_raise;
        let pot_before = self.pot + self.players.iter().map(|q| q.bet).sum::<i32>();
        let (stack_before, bet_before, max_to) = {
            let p = &self.players[i];
            (p.stack, p.bet, p.bet + p.stack)
        };

        let mut kind = action.kind;
        let mut amount = action.amount;
        if kind == ActionType::AllIn {
            amount = Some(max_to);
            kind = if max_to <= current_bet {
                if current_bet - bet_before > 0 { ActionType::Call } else { ActionType::Check }
            } else if current_bet == 0 {
                ActionType::Bet
            } else {
                ActionType::Raise
            };
        }
        if matches!(kind, ActionType::Bet | ActionType::Raise) && !legal.can_raise {
            kind = if current_bet - bet_before > 0 { ActionType::Call } else { ActionType::Check };
        }

        let commit = |p: &mut Player, to: i32| {
            let delta = to - p.bet;
            p.stack -= delta;
            p.commit += delta;
            p.bet = to;
            if p.stack == 0 {
                p.status = Status::AllIn;
            }
        };

        match kind {
            ActionType::Fold => {
                let p = &mut self.players[i];
                p.status = Status::Folded;
                p.has_acted = true;
            }
            ActionType::Check => {
                self.players[i].has_acted = true;
            }
            ActionType::Call => {
                let p = &mut self.players[i];
                commit(p, current_bet.min(max_to));
                p.has_acted = true;
            }
            ActionType::Bet | ActionType::Raise => {
                let mut to = amount.unwrap_or(legal.min_raise_to);
                to = to.min(max_to).max(legal.min_raise_to.min(max_to));
                let raise_size = to - current_bet;
                {
                    let p = &mut self.players[i];
                    commit(p, to);
                    p.has_acted = true;
                }
                if raise_size >= min_raise {
                    min_raise = raise_size;
                    for (j, other) in self.players.iter_mut().enumerate() {
                        if j != i && other.status == Status::Active {
                            other.has_acted = false;
                        }
                    }
                }
                current_bet = current_bet.max(to);
            }
            ActionType::AllIn => unreachable!(),
        }

        let p = &mut self.players[i];
        p.last_action = Some(if p.status == Status::AllIn { ActionType::AllIn } else { kind });
        p.acted_at = current_bet;
        self.actions.push(HandAction {
            seat: i,
            street: self.street,
            kind,
            allin: p.status == Status::AllIn,
            to: p.bet,
            added: p.bet - bet_before,
            pot_before,
            current_bet_before: self.current_bet,
            stack_before,
        });
        self.current_bet = current_bet;
        self.min_raise = min_raise;

        if self.betting_closed() {
            self.phase = Phase::StreetComplete;
            return;
        }
        self.to_act = next_actor(&self.players, self.current_bet, i as i32);
        self.phase = Phase::Betting;
    }

    fn betting_closed(&self) -> bool {
        let in_hand = self
            .players
            .iter()
            .filter(|p| matches!(p.status, Status::Active | Status::AllIn))
            .count();
        if in_hand <= 1 {
            return true;
        }
        !(0..SEATS).any(|i| owes_action(&self.players, self.current_bet, i))
    }

    /// Return the uncalled part of the biggest bet to its owner, then gather
    /// bets and deal the next street, run out, or settle (TS `advanceStreet`).
    pub fn advance_street(&mut self) {
        return_uncalled(&mut self.players);
        for p in self.players.iter_mut() {
            self.pot += p.bet;
            p.bet = 0;
            p.has_acted = false;
        }
        let not_folded = self
            .players
            .iter()
            .filter(|p| p.status != Status::Folded && p.status != Status::Out)
            .count();
        if not_folded == 1 {
            self.settle(false);
            return;
        }
        if self.street == Street::River {
            self.settle(true);
            return;
        }
        let (street, n) = match self.street {
            Street::Preflop => (Street::Flop, 3),
            Street::Flop => (Street::Turn, 1),
            Street::Turn => (Street::River, 1),
            Street::River => unreachable!(),
        };
        for p in self.players.iter_mut() {
            p.last_action = None;
        }
        for _ in 0..n {
            let c = self.deck.pop().expect("deck");
            self.community.push(c);
        }
        self.street = street;
        self.current_bet = 0;
        self.min_raise = BIG_BLIND;
        let can_act = self.players.iter().filter(|p| p.status == Status::Active).count();
        if can_act < 2 {
            self.to_act = -1;
            self.phase = Phase::StreetComplete;
        } else {
            self.to_act = next_seat(&self.players, self.button, |p| p.status == Status::Active);
            self.phase = Phase::Betting;
        }
    }

    /// Pots from each player's whole-hand commitment (TS `buildPots`).
    pub fn build_pots(&self) -> Vec<PotLayer> {
        build_pots(&self.players)
    }

    fn settle(&mut self, reveal: bool) {
        let pots = build_pots(&self.players);
        // Strengths for the still-in players (only compared at showdown,
        // where the board is complete).
        let mut strength = [None::<u32>; SEATS];
        for (i, p) in self.players.iter().enumerate() {
            if p.status == Status::Folded || p.status == Status::Out {
                continue;
            }
            strength[i] = Some(if reveal {
                let mut cards = p.hole.clone();
                cards.extend_from_slice(&self.community);
                eval7(&cards)
            } else {
                0
            });
        }
        let (results, won) = award_pots(&pots, &strength, self.button);
        for (p, w) in self.players.iter_mut().zip(won) {
            p.stack += w;
        }
        let match_over = self.players.iter().filter(|p| p.stack > 0).count() < 2;
        self.pot = 0;
        self.to_act = -1;
        self.phase = if match_over { Phase::MatchOver } else { Phase::HandOver };
        self.results = Some(results);
    }

    /// Whether the hand is decided (no more decisions or cards).
    pub fn hand_done(&self) -> bool {
        matches!(self.phase, Phase::HandOver | Phase::MatchOver)
    }
}

fn post_blind(p: &mut Player, amount: i32) {
    let posted = amount.min(p.stack);
    p.stack -= posted;
    p.bet = posted;
    p.commit = posted;
    if p.stack == 0 {
        p.status = Status::AllIn;
    }
}

/// The part of the biggest bet nobody matched goes back to its owner before
/// the street's bets are swept (TS `returnUncalled`).
fn return_uncalled(players: &mut [Player]) {
    let mut bets: Vec<i32> = players.iter().map(|p| p.bet).collect();
    bets.sort_unstable_by(|a, b| b.cmp(a));
    let excess = bets[0] - bets.get(1).copied().unwrap_or(0);
    if excess <= 0 {
        return;
    }
    let p = players.iter_mut().find(|q| q.bet == bets[0]).expect("top bettor");
    p.bet -= excess;
    p.commit -= excess;
    p.stack += excess;
    if p.status == Status::AllIn {
        p.status = Status::Active;
    }
}

/// Award each pot to its strongest eligible hand(s) (`strength[i]` is None for
/// players out of the hand). Splits give odd chips to the winners in seat
/// order starting left of the button. Returns the per-pot results and the
/// chips won by each seat.
pub fn award_pots(
    pots: &[PotLayer],
    strength: &[Option<u32>; SEATS],
    button: i32,
) -> (Vec<PotResult>, [i32; SEATS]) {
    let mut won = [0i32; SEATS];
    let mut results = Vec::with_capacity(pots.len());
    for pot in pots {
        let best = pot.eligible.iter().filter_map(|&i| strength[i]).max();
        let Some(best) = best else { continue };
        let mut winners: Vec<usize> =
            pot.eligible.iter().copied().filter(|&i| strength[i] == Some(best)).collect();
        winners.sort_by_key(|&i| (i as i32 - button - 1).rem_euclid(SEATS as i32));
        let share = pot.amount / winners.len() as i32;
        let mut remainder = pot.amount - share * winners.len() as i32;
        for &w in &winners {
            let mut amt = share;
            if remainder > 0 {
                amt += 1;
                remainder -= 1;
            }
            won[w] += amt;
        }
        results.push(PotResult { amount: pot.amount, winners });
    }
    (results, won)
}

pub fn build_pots(players: &[Player]) -> Vec<PotLayer> {
    let mut contrib: Vec<(usize, i32, bool)> = players
        .iter()
        .enumerate()
        .filter(|(_, p)| p.commit > 0)
        .map(|(i, p)| (i, p.commit, p.status == Status::Folded))
        .collect();
    let mut pots: Vec<PotLayer> = Vec::new();
    loop {
        let layer = contrib.iter().filter(|c| c.1 > 0).map(|c| c.1).min();
        let Some(layer) = layer else { break };
        let mut amount = 0;
        let mut eligible = Vec::new();
        for c in contrib.iter_mut().filter(|c| c.1 > 0) {
            amount += layer;
            c.1 -= layer;
            if !c.2 {
                eligible.push(c.0);
            }
        }
        match pots.last_mut() {
            Some(last) if last.eligible == eligible => last.amount += amount,
            _ => pots.push(PotLayer { amount, eligible }),
        }
    }
    pots
}
