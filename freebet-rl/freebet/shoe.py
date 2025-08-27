import random
from typing import Optional


class Shoe:
    def __init__(self, num_decks: int = 8, penetration: float = 0.5, seed: Optional[int] = None):
        self.num_decks = num_decks
        self.penetration = penetration
        self.rng = random.Random(seed)
        self.sessions_played = 0
        self._new_shoe()

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

    def draw(self) -> int:
        # Reshuffle at penetration threshold
        if len(self.cards) == 0 or self.cards_left() / self.initial_size <= (1 - self.penetration):
            self._new_shoe()
        self.dealt += 1
        return self.cards.pop()

    def cards_left(self) -> int:
        return len(self.cards)
