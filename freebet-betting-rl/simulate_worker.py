import os
import pandas as pd
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.append(r"C:\Project\BJMaster\BJMaster\freebet-rl")
from freebet.env import FreeBetEnv
from basic_strategy import BasicStrategy

# ---------------- CONFIG ----------------
DATA_DIR = "data"
RESULTS_FILE = os.path.join(os.getcwd(), "stoploss_results.csv")
MAX_STEPS = 200
TARGET_GAIN = 1.03
STOP_LOSSES = [0.95, 0.90, 0.80, 0.70]
N_SIMULATIONS = 100000   # runs par bankroll
MAX_BANKROLL = 10000
N_WORKERS = os.cpu_count() or 4

# Charger la Basic Strategy une seule fois
bs = BasicStrategy(
    hard_csv=os.path.join(DATA_DIR, "hard_totals.csv"),
    soft_csv=os.path.join(DATA_DIR, "soft_totals.csv"),
    pairs_csv=os.path.join(DATA_DIR, "pairs.csv"),
)

# ---------------- SIMULATION ----------------
def simulate_hand(game, player_cards, dealer_cards):
    """Simule une main avec mise unitaire (±1 ou ±2 en cas de double)."""
    dealer_up = dealer_cards[1]
    while True:
        total, soft = game.state_key(player_cards, dealer_up, True)[:2]
        action = bs.get_action(total, soft, 0, dealer_up)
        if action == "H":
            player_cards.append(game.shoe.draw())
            if game.state_key(player_cards, dealer_up, False)[0] > 21:
                return -1
        elif action == "S":
            break
        elif action == "D":
            player_cards.append(game.shoe.draw())
            return -2 if game.state_key(player_cards, dealer_up, False)[0] > 21 else +2
        else:
            break
    dealer_cards = game.dealer_play(dealer_cards)
    dealer_total, _ = game.state_key(dealer_cards, dealer_up, False)[:2]
    player_total, _ = game.state_key(player_cards, dealer_up, False)[:2]
    if player_total > 21:
        return -1
    if dealer_total > 21 or player_total > dealer_total:
        return +1
    elif player_total < dealer_total:
        return -1
    else:
        return 0

def simulate_chunk(n_runs, max_steps=MAX_STEPS):
    """Simule un chunk de runs et retourne les séquences d’outcomes."""
    game = FreeBetEnv(num_decks=8, penetration=0.5, dealer_hits_soft_17=True)
    runs = []
    for _ in range(n_runs):
        outcomes = []
        for _ in range(max_steps):
            p, d = game.initial_deal()
            outcomes.append(simulate_hand(game, p, d))
        runs.append(outcomes)
    return runs

def simulate_runs_parallel(bankroll, n_runs=N_SIMULATIONS, max_steps=MAX_STEPS):
    """Split runs en chunks et simule en parallèle."""
    chunk_size = n_runs // N_WORKERS
    chunks = [chunk_size] * N_WORKERS
    chunks[-1] += n_runs % N_WORKERS  # ajuster le dernier

    runs = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(simulate_chunk, c, max_steps) for c in chunks if c > 0]
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"Simulating {n_runs} runs", ncols=80):
            runs.extend(f.result())
    return runs

def evaluate_strategy(args):
    runs, bankroll, bet_size, stop_loss, mode = args
    stop_win = int(bankroll * TARGET_GAIN)
    stop_fail = bankroll * stop_loss

    results = {"success": 0, "fail": 0, "neutral": 0}
    for seq in runs:
        br = bankroll
        for outcome in seq:
            br += outcome * bet_size
            if br >= stop_win:
                results["success"] += 1
                break
            if br <= stop_fail:
                results["fail"] += 1
                break
        else:
            results["neutral"] += 1

    n = len(runs)
    success_rate = results["success"] / n
    fail_rate = results["fail"] / n
    neutral_rate = results["neutral"] / n
    score = success_rate - fail_rate

    return {
        "bankroll": bankroll,
        "bet": bet_size,
        "stop_loss": stop_loss,
        "mode": mode,
        "success_rate": success_rate,
        "fail_rate": fail_rate,
        "neutral_rate": neutral_rate,
        "score": score,
    }

def simulate_bankroll(bankroll):
    print(f"\n▶️ Bankroll={bankroll} → {N_SIMULATIONS} runs", flush=True)
    runs = simulate_runs_parallel(bankroll)

    combos = []
    for bet in range(1, min(5, bankroll) + 1):  # flat bets
        for sl in STOP_LOSSES:
            combos.append((runs, bankroll, bet, sl, "flat"))
    for pct in range(1, 11):  # percent bets
        for sl in STOP_LOSSES:
            bet_val = max(1, int(bankroll * (pct / 100)))
            combos.append((runs, bankroll, bet_val, sl, "percent"))

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        results = list(executor.map(evaluate_strategy, combos))

    df_results = pd.DataFrame(results)
    top_flat = df_results[df_results["mode"] == "flat"].nlargest(5, "score")
    top_percent = df_results[df_results["mode"] == "percent"].nlargest(5, "score")
    return pd.concat([top_flat, top_percent])

# ---------------- MAIN ----------------
if __name__ == "__main__":
    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)
        max_done = df["bankroll"].max()
    else:
        max_done = 0

    for bankroll in range(max_done + 1, MAX_BANKROLL + 1):
        final_df = simulate_bankroll(bankroll)
        header = not os.path.exists(RESULTS_FILE)
        final_df.to_csv(RESULTS_FILE, mode="a", index=False, header=header)
        print(f"✅ Bankroll {bankroll} done → {len(final_df)} stratégies sauvées", flush=True)
