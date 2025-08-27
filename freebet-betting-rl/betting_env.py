import random
import sys, os
sys.path.append(os.path.abspath("../freebet-rl"))

from freebet.env import FreeBetEnv

class BettingEnv:
    def __init__(self, basic_strategy, bankroll=100, max_steps=50):
        self.basic_strategy = basic_strategy
        self.starting_bankroll = bankroll
        self.bankroll = bankroll
        self.max_steps = max_steps
        self.steps = 0

        self.prev_bet = 0
        self.prev_gain = 0
        self.winstreak = 0
        self.loosestreak = 0

        self.stopped = False
        self.bankrupt = False
        self.sessions_played = 0
        self.stops = 0
        self.bankruptcies = 0

        # Env blackjack (simule la main avec basic strategy)
        self.game = FreeBetEnv(num_decks=8, penetration=0.5, dealer_hits_soft_17=True)

    def reset(self):
        self.sessions_played += 1
        self.bankroll = self.starting_bankroll
        self.steps = 0
        self.prev_bet = 0
        self.prev_gain = 0
        self.winstreak = 0
        self.loosestreak = 0
        self.stopped = False
        self.bankrupt = False
        return self._get_state()

    def _get_state(self):
        return [
            self.bankroll / 500.0,     # normalisation
            self.prev_bet / 5.0,
            self.prev_gain / 5.0,
            min(self.winstreak, 10) / 10.0,
            min(self.loosestreak, 10) / 10.0
        ]

    def step(self, action):
        """action ∈ {0 = Stop, 1..5 = mise réelle}"""
        if action == 0:  # STOP
            self.steps += 1
            reward = -0.1   # 🔥 petite pénalité pour ne pas jouer
            done = self.steps >= self.max_steps
            return self._get_state(), reward, done

        bet_size = action  # ⚠️ attention : ici action est déjà 1..5
        self.steps += 1

        # Simule une main avec la basic strategy
        outcome = self.simulate_hand(bet_size)

        # Update bankroll et streaks
        if outcome > 0:
            self.bankroll += outcome
            self.winstreak += 1
            self.loosestreak = 0
        elif outcome < 0:
            self.bankroll += outcome
            self.loosestreak += 1
            self.winstreak = 0

        self.prev_bet = bet_size
        self.prev_gain = outcome

        reward = outcome
        done = False

        if self.bankroll <= 0:
            reward -= 1000  # énorme pénalité si ruine
            done = True
        elif self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done

    def simulate_hand(self, bet_size):
        """Simule une main appliquant basic strategy sur FreeBetEnv"""
        player_cards, dealer_cards = self.game.initial_deal()
        dealer_up = dealer_cards[1]

        # Basic strategy
        while True:
            total, soft = self.game.state_key(player_cards, dealer_up, True)[:2]
            pair_rank = 0
            action = self.basic_strategy.get_action(total, soft, pair_rank, dealer_up)

            if action == "H":
                player_cards.append(self.game.shoe.draw())
                t, _ = self.game.state_key(player_cards, dealer_up, False)[:2]
                if t > 21:
                    return -bet_size
            elif action == "S":
                break
            elif action == "D":
                bet_size *= 2
                player_cards.append(self.game.shoe.draw())
                break
            else:
                break

        dealer_cards = self.game.dealer_play(dealer_cards)
        dealer_total, _ = self.game.state_key(dealer_cards, dealer_up, False)[:2]
        player_total, _ = self.game.state_key(player_cards, dealer_up, False)[:2]

        if player_total > 21:
            return -bet_size
        if dealer_total > 21 or player_total > dealer_total:
            return +bet_size
        elif player_total < dealer_total:
            return -bet_size
        else:
            return 0  # push
        
        
    def run_session(self, policy, max_steps=200, target=1.03):
        """Joue une session entière et retourne (reward, steps, logs)."""
        state = self.reset()
        logs = []

        for step in range(max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            probs = policy(state_tensor)
            m = torch.distributions.Categorical(probs)
            action = m.sample().item()

            next_state, _, done = self.step(action)

            logs.append({
                "bankroll": self.bankroll,
                "bet": self.prev_bet,
                "gain": self.prev_gain,
                "winstreak": self.winstreak,
                "loosestreak": self.loosestreak,
            })

            state = next_state
            if done:
                break

        # --- Objectif de session ---
        if self.bankroll >= self.starting_bankroll * target:
            reward = 1.0   # succès
        elif self.bankroll <= 0:
            reward = -1.0  # ruine
        else:
            reward = 0.0   # ni l’un ni l’autre

        return reward, step + 1, logs
