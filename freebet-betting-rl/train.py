from betting_env import BettingEnv
from basic_strategy import BasicStrategy
from agent import QAgent

bs = BasicStrategy("data/hard_totals.csv", "data/soft_totals.csv", "data/pairs.csv")
env = BettingEnv(bs)

agent = QAgent(actions=[1, 2, 5, 10, 20, 50])

episodes = 10000
for ep in range(episodes):
    state = env.reset()
    done = False
    while not done:
        key = agent.get_state_key(state)
        action = agent.act(key)
        next_state, reward, done = env.step(action)
        next_key = agent.get_state_key(next_state)
        agent.update(key, action, reward, next_key)
        state = next_state

print("Training done ✅")
