from collections import defaultdict
from freebet.env import FreeBetEnv
import numpy as np

class QTable:
    def __init__(self, alpha=0.05, gamma=1.0, epsilon=0.2, epsilon_min=0.01, epsilon_decay=0.999999, tc_range=(-10, 10)):
        """
        Q-learning tabulaire avec epsilon décroissant et gestion multi-TC (True Count).
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episodes = 0

        # Une Q-table et un compteur de visites par True Count
        self.q_by_tc = {tc: defaultdict(lambda: np.zeros(4)) for tc in range(tc_range[0], tc_range[1] + 1)}
        self.visits_by_tc = {tc: defaultdict(lambda: np.zeros(4, dtype=int)) for tc in range(tc_range[0], tc_range[1] + 1)}

    def choose_action(self, state, valid_actions, eps, tc=0):
        """Choisit une action epsilon-greedy en fonction du TC"""
        self.episodes += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        qvals = self.q_by_tc[tc][state]

        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_actions)  # exploration
        else:
            # exploitation : on prend l'action avec le plus haut Q
            return max(valid_actions, key=lambda a: qvals[FreeBetEnv.ACT_TO_IDX[a]])


    def update(self, state, action, reward, next_state=None, done=True, tc: int = 0):
        """Mise à jour TD-learning spécifique au TC."""
        self.visits_by_tc[tc][state][action] += 1

        visits_sa = self.visits_by_tc[tc][state][action]
        alpha = 1.0 / visits_sa if visits_sa > 0 else self.alpha

        q_old = self.q_by_tc[tc][state][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_by_tc[tc][next_state])

        self.q_by_tc[tc][state][action] = (1 - alpha) * q_old + alpha * target

    def update_episode(self, transitions, G, alpha=None, tc=0):
        """
        Version Monte Carlo (par TC).
        transitions : [(state, action), ...]
        G : return final
        """
        for (state, action) in transitions:
            self.visits_by_tc[tc][state][action] += 1
            visits_sa = self.visits_by_tc[tc][state][action]
            a = 1.0 / visits_sa if visits_sa > 0 else (alpha or self.alpha)
            self.q_by_tc[tc][state][action] = (1 - a) * self.q_by_tc[tc][state][action] + a * G
