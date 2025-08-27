from typing import List, Tuple

# Ranks: 2..10, 10 (J/Q/K), 11 (Ace)
RANKS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]
RANK_TO_STR = {**{i: str(i) for i in range(2, 11)}, 11: 'A'}


def hand_value(cards: List[int]) -> Tuple[int, bool]:
    total = sum(cards)
    aces = cards.count(11)
    # Reduce aces as needed
    reductions = 0
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
        reductions += 1
    # Soft if at least one ace still counted as 11 in the final total
    is_soft = (cards.count(11) - reductions) > 0 and total <= 21
    return total, is_soft


def is_blackjack(cards: List[int]) -> bool:
    return len(cards) == 2 and set(cards) == {10, 11}


def is_pair(cards: List[int]) -> Tuple[bool, int]:
    if len(cards) == 2 and cards[0] == cards[1]:
        rank = cards[0]
        if rank == 11:
            return True, 11
        if rank == 10:
            return True, 10
        return True, rank
    return False, 0
