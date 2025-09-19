# train.py
import argparse
import os
import pickle
import signal
import time
from collections import Counter, defaultdict

import numpy as np

from freebet.env import FreeBetEnv
from freebet.rl.qtable import QTable

SAVE_PATH_DEFAULT = "qtable.pkl"

# --------- Q-Table save/load (compatible Streamlit app) ----------
def save_qtable(qtab: QTable, path: str) -> None:
    data = {
        "q_by_tc": {tc: dict(qtab.q_by_tc[tc]) for tc in qtab.q_by_tc},
        "visits_by_tc": {tc: dict(qtab.visits_by_tc[tc]) for tc in qtab.visits_by_tc},
        "episodes": qtab.episodes,
        "epsilon": qtab.epsilon,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_qtable(path: str) -> QTable:
    qtab = QTable()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return qtab
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Rebuild structures
        from collections import defaultdict as DD
        qtab.q_by_tc = DD(lambda: DD(lambda: np.zeros(4)))
        for tc, qdict in data.get("q_by_tc", {}).items():
            qtab.q_by_tc[tc] = DD(lambda: np.zeros(4), qdict)
        qtab.visits_by_tc = DD(lambda: DD(lambda: np.zeros(4, dtype=int)))
        for tc, vdict in data.get("visits_by_tc", {}).items():
            qtab.visits_by_tc[tc] = DD(lambda: np.zeros(4, dtype=int), vdict)
        qtab.episodes = int(data.get("episodes", 0))
        qtab.epsilon = float(data.get("epsilon", qtab.epsilon))
    except Exception:
        pass
    return qtab

# --------- Helpers ----------
def is_win(outcomes) -> bool:
    if isinstance(outcomes, str):
        return outcomes == "win"
    if isinstance(outcomes, dict):
        return bool(outcomes.get("win"))
    return False

def is_push(outcomes) -> bool:
    if isinstance(outcomes, str):
        return outcomes == "push"
    if isinstance(outcomes, dict):
        return bool(outcomes.get("push"))
    return False

def train_loop(env: FreeBetEnv, qtab: QTable, *,
               alpha: float, epsilon: float,
               episodes_target: int, episodes_per_tick: int,
               autosave_secs: float, save_path: str,
               log_every: float) -> None:
    stats = Counter()
    returns_sum = 0.0
    returns_by_tc = defaultdict(float)
    episodes_by_tc = defaultdict(int)

    # timers
    t0 = time.perf_counter()
    t_last = t0
    last_save_t = t0
    episodes_at_last_save = qtab.episodes
    last_eps = qtab.episodes

    stop_flag = {"stop": False}
    def handle_sigint(sig, frame):
        print("\n[!] Ctrl+C reçu → sauvegarde et sortie propre…")
        stop_flag["stop"] = True
    signal.signal(signal.SIGINT, handle_sigint)

    while (episodes_target <= 0) or (qtab.episodes < episodes_target):
        if stop_flag["stop"]:
            break

        # --- un gros tick ---
        ep_this_tick = episodes_per_tick
        for _ in range(ep_this_tick):
            tc = max(-5, min(5, env.shoe.true_count()))
            trans, rewards, outcomes = env.play_round(qtab, epsilon, tc)

            G = 0.0
            for r in rewards:
                G += r

            if trans:
                qtab.update_episode(trans, G, alpha, tc)

            qtab.episodes += 1
            returns_sum += G
            episodes_by_tc[tc] += 1
            returns_by_tc[tc] += G
            if is_win(outcomes):
                stats["win"] += 1
            elif is_push(outcomes):
                stats["push"] += 1
            else:
                stats["loss"] += 1

            if stop_flag["stop"]:
                break

        # --- logging / autosave ---
        now = time.perf_counter()
        if (now - t_last) >= log_every:
            dt = now - t_last
            eps_delta = qtab.episodes - last_eps
            eps_per_s = eps_delta / max(1e-9, dt)
            total = sum(stats.values()) or 1
            winp = 100.0 * stats.get("win", 0) / total
            pushp = 100.0 * stats.get("push", 0) / total
            lossp = 100.0 * stats.get("loss", 0) / total
            ev = returns_sum / max(1, qtab.episodes)
            print(
                f"[{qtab.episodes:,} eps] "
                f"⚡ {eps_per_s:,.0f} eps/s | "
                f"EV/round={ev:+.4f} | W/P/L: {winp:.1f}% / {pushp:.1f}% / {lossp:.1f}%"
            )
            last_eps = qtab.episodes
            t_last = now

        if (now - last_save_t) >= autosave_secs and qtab.episodes != episodes_at_last_save:
            save_qtable(qtab, save_path)
            last_save_t = now
            episodes_at_last_save = qtab.episodes
            print(f"💾 Autosave: {save_path} @ {qtab.episodes:,} eps")

    # final save
    save_qtable(qtab, save_path)
    print(f"✅ Fin. Q-table sauvegardée dans {save_path} @ {qtab.episodes:,} eps")

def main():
    ap = argparse.ArgumentParser(description="Free Bet Blackjack RL - trainer (no UI)")
    ap.add_argument("--save-path", default=SAVE_PATH_DEFAULT, help="Chemin du qtable.pkl")
    ap.add_argument("--resume", action="store_true", help="Charger le Q-table s'il existe")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--epsilon", type=float, default=0.20)
    ap.add_argument("--gamma", type=float, default=1.0)  # si utilisé dans QTable
    ap.add_argument("--episodes-per-tick", type=int, default=100_000)
    ap.add_argument("--episodes", type=int, default=5_000_000, help="Nombre total d'épisodes (0 = infini)")
    ap.add_argument("--autosave-secs", type=float, default=60.0)
    ap.add_argument("--log-every", type=float, default=2.0, help="Affichage des stats toutes les X secondes")
    ap.add_argument("--num-decks", type=int, default=8)
    ap.add_argument("--penetration", type=float, default=0.5)
    ap.add_argument("--h17", action="store_true", help="Dealer hits soft 17 (sinon S17)")
    args = ap.parse_args()

    # Env + Qtable
    env = FreeBetEnv(
        num_decks=args.num_decks,
        penetration=args.penetration,
        dealer_hits_soft_17=bool(args.h17),
    )
    if args.resume and os.path.exists(args.save_path):
        print(f"↻ Reprise depuis {args.save_path}")
        qtab = load_qtable(args.save_path)
    else:
        qtab = QTable()

    # Optionnel: stocker epsilon dans l'objet si tu l'utilises ailleurs
    qtab.epsilon = args.epsilon

    print("🚀 Training lancé "
          f"(alpha={args.alpha}, epsilon={args.epsilon}, tick={args.episodes_per_tick:,}, "
          f"autosave={int(args.autosave_secs)}s, H17={bool(args.h17)})")
    train_loop(
        env, qtab,
        alpha=args.alpha, epsilon=args.epsilon,
        episodes_target=args.episodes,
        episodes_per_tick=args.episodes_per_tick,
        autosave_secs=args.autosave_secs,
        save_path=args.save_path,
        log_every=args.log_every,
    )

if __name__ == "__main__":
    main()
