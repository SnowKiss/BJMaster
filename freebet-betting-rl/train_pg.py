import torch
import streamlit as st

def select_action(policy, state):
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    probs = torch.softmax(policy(state_t), dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob


def train_one_episode(env, policy, optimizer, gamma=0.99, target_gain=1.03):
    """Joue une session complète et entraîne la policy.
       Objectif : atteindre bankroll * target_gain = succès,
       tomber à 0 = échec, sinon neutre.
    """
    log_probs = []
    rewards = []
    logs = []

    init_bankroll = env.starting_bankroll
    state = env.reset()
    done = False

    total_reward = 0
    steps = 0
    outcome = "neutral"

    while not done:
        # --- forward policy ---
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = policy(state_tensor)
        m = torch.distributions.Categorical(probs)
        action = m.sample()

        log_prob = m.log_prob(action)
        log_probs.append(log_prob)

        bet_size = action.item()  # 0 = stop, 1..5 = mise
        next_state, reward, done = env.step(bet_size)

        rewards.append(reward)
        total_reward += reward
        steps += 1

        logs.append({
            "bankroll": env.bankroll,
            "bet": bet_size,
            "prev_bet": env.prev_bet,
            "prev_gain": env.prev_gain,
            "winstreak": env.winstreak,
            "loosestreak": env.loosestreak,
        })

        state = next_state

        # --- conditions de stop ---
        if env.bankroll >= init_bankroll * target_gain:
            outcome = "success"
            rewards[-1] += 10.0  # gros bonus
            done = True
        elif env.bankroll <= 0:
            outcome = "fail"
            rewards[-1] -= 10.0  # grosse pénalité
            done = True

    # --- Update policy ---
    discounted = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted.insert(0, R)
    discounted = torch.tensor(discounted, dtype=torch.float32)

    if discounted.std() > 1e-9:  # éviter NaN
        discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-9)

    loss = []
    for log_prob, Gt in zip(log_probs, discounted):
        loss.append(-log_prob * Gt)
    loss = torch.stack(loss).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return total_reward, steps, outcome, logs