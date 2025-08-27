from collections import defaultdict
import numpy as np

class QTable:
    def __init__(self, alpha=0.05, gamma=1.0, epsilon=0.2, epsilon_min=0.01, epsilon_decay=0.999999):
        """
        Q-learning tabulaire avec epsilon décroissant et alpha adaptatif.

        - alpha        : taux d'apprentissage de base
        - gamma        : discount factor
        - epsilon      : exploration initiale
        - epsilon_min  : exploration minimale
        - epsilon_decay: facteur de décroissance de epsilon
        """
        self.q = defaultdict(lambda: np.zeros(4))  # H, S, D, P
        self.visits = defaultdict(lambda: np.zeros(4, dtype=int))  # compteur de visites
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episodes = 0  # compteur global

    def choose_action(self, state, valid_actions):
        """Choisit une action selon epsilon-greedy."""
        self.episodes += 1
        # décroissance epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_actions)  # exploration
        else:
            qvals = self.q[state]
            return max(valid_actions, key=lambda a: qvals[a])  # exploitation

    def update(self, state, action, reward, next_state=None, done=True):
        """Met à jour la Q-table (TD-learning)."""
        self.visits[state][action] += 1

        # alpha adaptatif
        visits_sa = self.visits[state][action]
        alpha = 1.0 / visits_sa if visits_sa > 0 else self.alpha

        q_old = self.q[state][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q[next_state])

        self.q[state][action] = (1 - alpha) * q_old + alpha * target

    def update_episode(self, transitions, G, alpha=None):
        """
        Version Monte Carlo (comme tu avais avant).
        transitions : [(state, action), ...]
        G : return final
        """
        for (state, action) in transitions:
            self.visits[state][action] += 1
            visits_sa = self.visits[state][action]
            a = 1.0 / visits_sa if visits_sa > 0 else (alpha or self.alpha)
            self.q[state][action] = (1 - a) * self.q[state][action] + a * G
