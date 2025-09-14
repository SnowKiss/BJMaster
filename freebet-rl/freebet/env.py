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
    base_paid_on_loss: int = 1   # 1 = payant
    base_win_amount: int = 1     # 1 = mise de base

    # Flags
    doubled: bool = False
    split_aces: bool = False
    from_split: bool = False      # <-- empêche resplit et (si no DAS) empêche double

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

        paid_on_loss = self.base_paid_on_loss
        win_amount = self.base_win_amount

        # Double (toujours payant en classique)
        if self.doubled:
            paid_on_loss = self.base_paid_on_loss + 1  # perte totale = 2
            win_amount = self.base_win_amount + 1      # gain total = 2

        # Blackjack naturel → payé 3:2
        if bj:
            return paid_on_loss, 1.5, True

        return paid_on_loss, float(win_amount), False


class FreeBetEnv:  # 👈 même nom conservé
    ACTIONS = ['H', 'S', 'D', 'P']
    ACT_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}

    def __init__(self, num_decks=8, penetration=0.5, dealer_hits_soft_17=False, seed=None,
                 allow_das: bool = False):
        """
        allow_das=False  -> 'no DAS' (double après split interdit)  [règle que tu utilises]
        """
        self.shoe = Shoe(num_decks=num_decks, penetration=penetration, seed=seed)
        self.dealer_hits_soft_17 = dealer_hits_soft_17
        self.allow_das = bool(allow_das)
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
        # Interdire Stand / Double tant qu'on n'a pas 2 cartes
        acts = ['H']

        # Split As : une carte tirée puis stand forcé (géré après split)
        if hand.split_aces and len(hand.cards) >= 2:
            return ['S']

        if len(hand.cards) >= 2:
            acts.append('S')

            # Double uniquement sur la 1ère décision et main à 2 cartes.
            # No DAS : si la main vient d'un split, on bloque D (sauf si allow_das=True).
            if hand.first_action and (self.allow_das or not hand.from_split):
                acts.append('D')

            # Split uniquement à la 1ère décision, 2 cartes, et jamais après un split (no resplit)
            pair, _ = is_pair(hand.cards)
            if hand.first_action and pair and not hand.from_split:
                acts.append('P')

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
        outcomes = Counter()

        # ----- Blackjack checks (peek effectif) -----
        if dealer_has_bj:
            if player_has_bj:
                per_hand_rewards.append(0.0)
                outcomes["push"] += 1
                return transitions, per_hand_rewards, outcomes
            else:
                per_hand_rewards.append(-1.0)
                outcomes["loss"] += 1
                return transitions, per_hand_rewards, outcomes

        if player_has_bj:
            per_hand_rewards.append(1.5)
            outcomes["win"] += 1
            return transitions, per_hand_rewards, outcomes

        # ----- Joueur -----
        hands = [HandRecord(cards=player_cards.copy())]
        round_transitions = []
        i = 0
        while i < len(hands):
            hand = hands[i]
            round_transitions.append([])

            while True:
                avail = self.available_actions(hand, dealer_up)
                state = self.state_key(hand.cards, dealer_up, hand.first_action)

                _ = qtab.q_by_tc[tc][state]  # init qvals si absent

                act = qtab.choose_action(state, avail, eps, tc)
                a_idx = self.ACT_TO_IDX[act]
                round_transitions[i].append((state, a_idx))

                if act == "H":
                    hand.cards.append(self.shoe.draw())
                    # après un hit, ce n'est plus la première décision
                    hand.first_action = False

                    # Règle optionnelle : 6-card Charlie (si tu veux la garder)
                    if len(hand.cards) >= 6:
                        total, _ = hand_value(hand.cards)
                        if total <= 21:
                            pol, win_amt, _ = hand.bet_profile()
                            per_hand_rewards.append(float(win_amt))
                            outcomes["win"] += 1
                            break

                    total, _ = hand_value(hand.cards)
                    if total > 21:
                        pol, _, _ = hand.bet_profile()
                        per_hand_rewards.append(-float(pol))
                        outcomes["loss"] += 1
                        break
                    else:
                        continue

                elif act == "S":
                    break

                elif act == "D":
                    # double seulement permis par available_actions (2 cartes, 1ère décision, DAS respecté)
                    hand.doubled = True
                    hand.cards.append(self.shoe.draw())
                    hand.first_action = False
                    t, _ = hand_value(hand.cards)
                    pol, _, _ = hand.bet_profile()
                    if t > 21:
                        per_hand_rewards.append(-float(pol))
                        outcomes["loss"] += 1
                    break

                elif act == "P":
                    # split seulement permis par available_actions (2 cartes, 1ère décision, pas de resplit)
                    pair, pr = is_pair(hand.cards)
                    if not (hand.first_action and pair and len(hand.cards) == 2 and not hand.from_split):
                        break

                    # Créer deux mains issues du split
                    card_rank = pr
                    left = HandRecord(
                        cards=[hand.cards[0]],
                        first_action=True,
                        split_aces=(card_rank == 11),
                        from_split=True,
                    )
                    right = HandRecord(
                        cards=[hand.cards[1]],
                        first_action=True,
                        split_aces=(card_rank == 11),
                        from_split=True,
                    )
                    left.base_paid_on_loss = 1
                    left.base_win_amount = 1
                    right.base_paid_on_loss = 1
                    right.base_win_amount = 1

                    # 👉 Distribuer immédiatement une carte à chaque main
                    left.cards.append(self.shoe.draw())
                    right.cards.append(self.shoe.draw())

                    if card_rank == 11:
                        # Split des As : une carte chacun, stand forcé (pas de hit/double)
                        left.first_action = False
                        right.first_action = False

                    # Remplacer la main courante par les deux nouvelles
                    hands[i] = left
                    hands.insert(i + 1, right)
                    break

            i += 1

        # ----- Dealer -----
        any_pending = any(hand_value(h.cards)[0] <= 21 for h in hands)
        if any_pending:
            dealer_cards = self.dealer_play(dealer_cards)
        dealer_total, _ = hand_value(dealer_cards)

        # ----- Résolution -----
        hidx = 0
        for hand in hands:
            if hidx < len(per_hand_rewards):
                hidx += 1
                continue
            pol, win_amt, bj_bonus = hand.bet_profile()
            pt, _ = hand_value(hand.cards)

            if pt > 21:
                per_hand_rewards.append(-float(pol))
                outcomes["loss"] += 1
                continue

            if dealer_total > 21:
                per_hand_rewards.append(float(win_amt))
                outcomes["win"] += 1
            else:
                if pt > dealer_total:
                    per_hand_rewards.append(float(win_amt))
                    outcomes["win"] += 1
                elif pt < dealer_total:
                    per_hand_rewards.append(-float(pol))
                    outcomes["loss"] += 1
                else:
                    per_hand_rewards.append(0.0)
                    outcomes["push"] += 1
            hidx += 1

        # Flatten transitions
        flat_transitions = []
        for tl in round_transitions:
            flat_transitions.extend(tl)
        return flat_transitions, per_hand_rewards, outcomes
