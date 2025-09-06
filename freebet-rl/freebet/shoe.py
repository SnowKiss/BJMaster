import random
from typing import Optional


class Shoe:
    def __init__(self, num_decks: int = 8, penetration: float = 0.5, seed: Optional[int] = None):
        self.num_decks = num_decks
        self.penetration = penetration
        self.rng = random.Random(seed)
        self.sessions_played = 0
        self.running_count = 0
        self._new_shoe()

    def _card_value_for_count(self, card: int) -> int:
        """Hi-Lo system: 2–6 = +1, 7–9 = 0, 10/A = −1."""
        if 2 <= card <= 6:
            return 1
        elif card in (10, 11):  # 10, J, Q, K, A
            return -1
        return 0

    def _new_shoe(self):
        # Build a multi-deck shoe
        single_deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4
        self.cards = []
        for _ in range(self.num_decks):
            self.cards.extend(single_deck)
        self.rng.shuffle(self.cards)
        self.initial_size = len(self.cards)
        self.dealt = 0
        self.sessions_played += 1
        self.running_count = 0

    def draw(self) -> int:
        if len(self.cards) == 0 or self.cards_left() / self.initial_size <= (1 - self.penetration):
            self._new_shoe()
        card = self.cards.pop()
        self.dealt += 1
        self.running_count += self._card_value_for_count(card)
        return card

    def true_count(self) -> int:
        decks_remaining = max(1, self.cards_left() / 52)
        return int(round(self.running_count / decks_remaining))

    def cards_left(self) -> int:
        return len(self.cards)
