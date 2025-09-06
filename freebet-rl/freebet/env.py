import random
from dataclasses import dataclass
from collections import Counter
from typing import List, Tuple, Dict

import numpy as np

from .cards import hand_value, is_blackjack, is_pair
from .shoe import Shoe


@dataclass
class HandRecord:
    cards: List[int]
    first_action: bool = True
    # Betting profile
    base_paid_on_loss: int = 1   # 1 = payant, 0 = gratuit (ex: main issue d’un free split)
    base_win_amount: int = 1     # 1 = mise de base

    # Flags
    doubled: bool = False
    double_free: bool = False
    came_from_free_split: bool = False
    split_aces: bool = False

    transitions: List[Tuple[Tuple[int, int, int, int, int], int]] = None

    def __post_init__(self):
        if self.transitions is None:
            self.transitions = []

    def bet_profile(self) -> Tuple[int, float, bool]:
        """
        Retourne (mise_perdue, mise_gagnée, is_blackjack_bonus)
        - mise_perdue : montant perdu si la main perd
        - mise_gagnée : montant gagné si la main gagne
        - is_blackjack_bonus : True si blackjack naturel
        """
        total, soft = hand_value(self.cards)
        bj = (len(self.cards) == 2 and total == 21)  # Blackjack naturel

        # Valeurs par défaut
        paid_on_loss = self.base_paid_on_loss
        win_amount = self.base_win_amount

        # Double
        if self.doubled:
            if self.double_free:
                # Double gratuit → la mise ne double pas en cas de perte
                paid_on_loss = self.base_paid_on_loss
                win_amount = self.base_win_amount + 1  # gain total = 2
            else:
                # Double payant → mise doublée
                paid_on_loss = self.base_paid_on_loss + 1  # perte totale = 2
                win_amount = self.base_win_amount + 1      # gain total = 2

        # Blackjack naturel → payé 3:2
        if bj:
            return paid_on_loss, 1.5, True

        return paid_on_loss, float(win_amount), False


class FreeBetEnv:
    ACTIONS = ['H', 'S', 'D', 'P']
    ACT_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}

    def __init__(self, num_decks=8, penetration=0.5, dealer_hits_soft_17=True, seed=None):
        self.shoe = Shoe(num_decks=num_decks, penetration=penetration, seed=seed)
        self.dealer_hits_soft_17 = dealer_hits_soft_17
        self.rng = random.Random(seed)

    # ----- Deal & Dealer play -----
    def initial_deal(self) -> Tuple[List[int], List[int]]:
        player = [self.shoe.draw(), self.shoe.draw()]
        dealer = [self.shoe.draw(), self.shoe.draw()]  # dealer[1] = upcard
        return player, dealer

    def dealer_play(self, dealer_cards: List[int]) -> List[int]:
        if is_blackjack(dealer_cards):
            return dealer_cards
        while True:
            total, soft = hand_value(dealer_cards)
            if total >= 17:
                if total > 17:
                    break
                if soft and self.dealer_hits_soft_17:
                    dealer_cards.append(self.shoe.draw())
                    continue
                else:
                    break
            else:
                dealer_cards.append(self.shoe.draw())
        return dealer_cards

    # ----- State & Actions -----
    def state_key(self, cards: List[int], dealer_up: int, first_action: bool) -> Tuple[int, int, int, int, int]:
        total, soft = hand_value(cards)
        pair, pr = is_pair(cards)
        pt = min(max(total, 4), 21)
        is_soft_int = 1 if soft else 0
        pair_rank = pr if pair else 0
        up = dealer_up
        fa = 1 if first_action else 0
        return (pt, is_soft_int, pair_rank, up, fa)

    def available_actions(self, hand: HandRecord, dealer_up: int) -> List[str]:
        acts = ['H', 'S']
        if hand.first_action:
            acts.append('D')
            pair, pr = is_pair(hand.cards)
            if pair:
                acts.append('P')
        if hand.split_aces and len(hand.cards) >= 2:
            return ['S']
        return acts

    def choose_action(self, q: Dict[Tuple[int, int, int, int, int], np.ndarray], state, avail, eps: float):
        if self.rng.random() < eps:
            return self.rng.choice(avail)
        qvals = q[state]
        mask = np.full(len(self.ACTIONS), -1e9, dtype=float)
        for a in avail:
            mask[self.ACT_TO_IDX[a]] = 0.0
        masked = qvals + mask
        best_idx = int(np.argmax(masked))
        return self.ACTIONS[best_idx]

    # ----- One round with RL transitions -----
    def play_round(self, qtab, eps: float, tc: int = 0):
        player_cards, dealer_cards = self.initial_deal()
        dealer_up = dealer_cards[1]

        dealer_has_bj = is_blackjack(dealer_cards)
        player_has_bj = is_blackjack(player_cards)
        transitions = []
        per_hand_rewards = []

        if dealer_has_bj:
            if player_has_bj:
                per_hand_rewards.append(0.0)
                return transitions, per_hand_rewards, {"push": 1}
            else:
                per_hand_rewards.append(-1.0)
                return transitions, per_hand_rewards, {"loss": 1}

        if player_has_bj:
            per_hand_rewards.append(1.5)
            return transitions, per_hand_rewards, {"win": 1}

        hands = [HandRecord(cards=player_cards.copy())]
        round_transitions = []
        i = 0
        while i < len(hands):
            hand = hands[i]
            round_transitions.append([])

            if hand.split_aces and len(hand.cards) == 1:
                hand.cards.append(self.shoe.draw())
                hand.first_action = False
                i += 1
                continue

            while True:
                avail = self.available_actions(hand, dealer_up)
                state = self.state_key(hand.cards, dealer_up, hand.first_action)

                # ✅ Avant : _ = q[state]
                _ = qtab.q_by_tc[tc][state]  # initialise les qvals si absent

                act = qtab.choose_action(state, avail, eps, tc)
                a_idx = self.ACT_TO_IDX[act]
                round_transitions[i].append((state, a_idx))

                if act == 'H':
                    hand.cards.append(self.shoe.draw())
                    hand.first_action = False
                    total, _ = hand_value(hand.cards)
                    if total > 21:
                        pol, _, _ = hand.bet_profile()
                        per_hand_rewards.append(-float(pol))
                        break
                    else:
                        continue

                elif act == 'S':
                    break

                elif act == 'D':
                    total, _ = hand_value(hand.cards)
                    hand.doubled = True
                    hand.double_free = (total in (9, 10, 11))
                    hand.cards.append(self.shoe.draw())
                    hand.first_action = False
                    t, _ = hand_value(hand.cards)
                    pol, _, _ = hand.bet_profile()
                    if t > 21:
                        per_hand_rewards.append(-float(pol))
                    break

                elif act == 'P':
                    pair, pr = is_pair(hand.cards)
                    if not (hand.first_action and pair):
                        break
                    card_rank = pr
                    left = HandRecord(
                        cards=[hand.cards[0]],
                        first_action=True,
                        came_from_free_split=(card_rank in (2,3,4,5,6,7,8,9,11)),
                        split_aces=(card_rank == 11)
                    )
                    right = HandRecord(
                        cards=[hand.cards[1]],
                        first_action=True,
                        came_from_free_split=(card_rank in (2,3,4,5,6,7,8,9,11)),
                        split_aces=(card_rank == 11)
                    )
                    left.base_paid_on_loss = 1
                    left.base_win_amount = 1
                    right.base_paid_on_loss = 0 if right.came_from_free_split else 1
                    right.base_win_amount = 1
                    hands[i] = left
                    hands.insert(i + 1, right)
                    break

            i += 1

        any_pending = any(hand_value(h.cards)[0] <= 21 for h in hands)
        if any_pending:
            dealer_cards = self.dealer_play(dealer_cards)
        dealer_total, _ = hand_value(dealer_cards)

        outcomes = Counter()
        hidx = 0
        for hand in hands:
            if hidx < len(per_hand_rewards):
                hidx += 1
                continue
            pol, win_amt, bj_bonus = hand.bet_profile()
            pt, _ = hand_value(hand.cards)

            if pt > 21:
                per_hand_rewards.append(-float(pol))
                outcomes['loss'] += 1
                continue

            if dealer_total == 22:
                per_hand_rewards.append(0.0)
                outcomes['push'] += 1
                continue

            if dealer_total > 21:
                per_hand_rewards.append(float(win_amt))
                outcomes['win'] += 1
            else:
                if pt > dealer_total:
                    per_hand_rewards.append(float(win_amt))
                    outcomes['win'] += 1
                elif pt < dealer_total:
                    per_hand_rewards.append(-float(pol))
                    outcomes['loss'] += 1
                else:
                    per_hand_rewards.append(0.0)
                    outcomes['push'] += 1
            hidx += 1

        flat_transitions = []
        for tl in round_transitions:
            flat_transitions.extend(tl)
        return flat_transitions, per_hand_rewards, outcomes
