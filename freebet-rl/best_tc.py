import pickle
import numpy as np

# Charger la Q-table
with open("qtable.pkl", "rb") as f:
    data = pickle.load(f)

q_by_tc = data["q_by_tc"]

summary = []
for tc, inner in q_by_tc.items():
    evs = []
    for state, qvals in inner.items():
        arr = np.array(qvals, dtype=float)
        evs.extend(arr)
    avg_ev = np.mean(evs) if evs else 0.0
    summary.append((tc, avg_ev))

summary_filtered = [s for s in summary if -5 <= s[0] <= 5]
summary_sorted = sorted(summary_filtered, key=lambda x: x[1], reverse=True)

print("=== TC les plus favorables (–5 à +5) ===")
for tc, ev in summary_sorted[:5]:
    print(f"TC {tc:+d} → EV moyenne {ev:.4f}")

print("\n=== TC les plus défavorables (–5 à +5) ===")
for tc, ev in summary_sorted[-5:]:
    print(f"TC {tc:+d} → EV moyenne {ev:.4f}")

