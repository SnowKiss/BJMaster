# eval_ev.py
import argparse
import os
import pickle
import sys
import time
import multiprocessing as mp
from collections import defaultdict, Counter
from typing import Dict, Any, Tuple

import numpy as np

from freebet.env import FreeBetEnv
from freebet.rl.qtable import QTable

# --------------------- QTable load helpers (tolère dict ou objet picklé) ---------------------
def _dict_to_qtable(d: dict) -> QTable:
    qtab = QTable()
    # numpy arrays garantis (pour .sum(), argmax, etc.)
    qtab.q_by_tc = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))
    for tc, dd in d.get("q_by_tc", {}).items():
        qtab.q_by_tc[tc] = defaultdict(lambda: np.zeros(4), {s: np.asarray(v, dtype=float) for s, v in dd.items()})
    qtab.visits_by_tc = defaultdict(lambda: defaultdict(lambda: np.zeros(4, dtype=int)))
    for tc, dd in d.get("visits_by_tc", {}).items():
        qtab.visits_by_tc[tc] = defaultdict(lambda: np.zeros(4, dtype=int), {s: np.asarray(v, dtype=int) for s, v in dd.items()})
    qtab.episodes = int(d.get("episodes", 0))
    qtab.epsilon = float(d.get("epsilon", getattr(qtab, "epsilon", 0.0)))
    return qtab

def load_qtable(path: str) -> QTable:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(f"Q-table introuvable: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return _dict_to_qtable(data)
    elif isinstance(data, QTable):
        # normalise tout de même en ndarrays
        data.q_by_tc = defaultdict(lambda: defaultdict(lambda: np.zeros(4)), {
            tc: defaultdict(lambda: np.zeros(4), {s: np.asarray(v, dtype=float) for s, v in dd.items()})
            for tc, dd in data.q_by_tc.items()
        })
        data.visits_by_tc = defaultdict(lambda: defaultdict(lambda: np.zeros(4, dtype=int)), {
            tc: defaultdict(lambda: np.zeros(4, dtype=int), {s: np.asarray(v, dtype=int) for s, v in dd.items()})
            for tc, dd in data.visits_by_tc.items()
        })
        return data
    else:
        raise ValueError("Format de qtable.pkl inattendu (ni dict ni QTable).")

# --------------------- Outcome helpers ---------------------
def _is_win(outcomes) -> bool:
    if isinstance(outcomes, str):
        return outcomes == "win"
    if isinstance(outcomes, dict):
        return bool(outcomes.get("win"))
    return False

def _is_push(outcomes) -> bool:
    if isinstance(outcomes, str):
        return outcomes == "push"
    if isinstance(outcomes, dict):
        return bool(outcomes.get("push"))
    return False

# --------------------- Mono-process evaluation ---------------------
def evaluate_single(qtab: QTable, *, rounds: int, num_decks: int, penetration: float, h17: bool
                   ) -> Tuple[float, Counter, Dict[int, float], Dict[int, int], int]:
    env = FreeBetEnv(num_decks=num_decks, penetration=penetration, dealer_hits_soft_17=h17)
    eps = 0.0  # greedy: PAS d'exploration
    total_return = 0.0
    stats = Counter()
    returns_by_tc = defaultdict(float)
    episodes_by_tc = defaultdict(int)

    t0 = time.perf_counter()
    for _ in range(rounds):
        tc = max(-5, min(5, env.shoe.true_count()))
        trans, rewards, outcomes = env.play_round(qtab, eps, tc)
        g = float(np.sum(rewards))  # pas d'update_episode ici !
        total_return += g
        returns_by_tc[tc] += g
        episodes_by_tc[tc] += 1
        if _is_win(outcomes):
            stats["win"] += 1
        elif _is_push(outcomes):
            stats["push"] += 1
        else:
            stats["loss"] += 1
    dt = time.perf_counter() - t0
    eps_per_s = int(rounds / max(1e-9, dt))
    ev = total_return / max(1, rounds)
    return ev, stats, returns_by_tc, episodes_by_tc, eps_per_s

# --------------------- Multi-process evaluation ---------------------
# On broadcast un "dict qtable" compact à chaque worker via Pool initializer
GLOBAL_QTAB_DICT = None
GLOBAL_CFG = None

def _pool_init(qtab_dict: dict, cfg: dict):
    global GLOBAL_QTAB_DICT, GLOBAL_CFG
    GLOBAL_QTAB_DICT = qtab_dict
    GLOBAL_CFG = cfg

def _qtable_to_dict(q: QTable) -> dict:
    return {
        "q_by_tc": {tc: dict(q.q_by_tc[tc]) for tc in q.q_by_tc},
        "visits_by_tc": {tc: dict(q.visits_by_tc[tc]) for tc in q.visits_by_tc},
        "episodes": int(getattr(q, "episodes", 0)),
        "epsilon": float(getattr(q, "epsilon", 0.0)),
    }

def _eval_chunk(n_rounds: int, seed: int = 0):
    # Chaque worker recrée son env, reconstruit QTable localement (read-only)
    qtab = _dict_to_qtable(GLOBAL_QTAB_DICT)
    env = FreeBetEnv(
        num_decks=int(GLOBAL_CFG["num_decks"]),
        penetration=float(GLOBAL_CFG["penetration"]),
        dealer_hits_soft_17=bool(GLOBAL_CFG["h17"]),
    )
    eps = 0.0
    total_return = 0.0
    stats = Counter()
    returns_by_tc = defaultdict(float)
    episodes_by_tc = defaultdict(int)

    for _ in range(n_rounds):
        tc = max(-5, min(5, env.shoe.true_count()))
        trans, rewards, outcomes = env.play_round(qtab, eps, tc)
        g = float(np.sum(rewards))
        total_return += g
        returns_by_tc[tc] += g
        episodes_by_tc[tc] += 1
        if _is_win(outcomes):
            stats["win"] += 1
        elif _is_push(outcomes):
            stats["push"] += 1
        else:
            stats["loss"] += 1

    # On renvoie des objets merge-ables
    return total_return, dict(stats), dict(returns_by_tc), dict(episodes_by_tc)

def evaluate_mp(qtab: QTable, *, rounds: int, workers: int, num_decks: int, penetration: float, h17: bool,
                log_every: float) -> Tuple[float, Counter, Dict[int, float], Dict[int, int], int]:
    # Découpage des épisodes par worker
    if workers <= 1 or rounds < workers * 10_000:
        return evaluate_single(qtab, rounds=rounds, num_decks=num_decks, penetration=penetration, h17=h17)

    ctx = mp.get_context("spawn") if sys.platform.startswith("win") else mp.get_context()
    pool = ctx.Pool(
        processes=workers,
        initializer=_pool_init,
        initargs=(_qtable_to_dict(qtab), dict(num_decks=num_decks, penetration=penetration, h17=h17)),
    )

    per_worker = rounds // workers
    remainder = rounds % workers
    chunks = [per_worker + (1 if i < remainder else 0) for i in range(workers)]
    # On peut mapper par paquets pour avoir des logs périodiques
    t0 = time.perf_counter()
    total_return = 0.0
    stats = Counter()
    returns_by_tc = defaultdict(float)
    episodes_by_tc = defaultdict(int)

    # map imap_unordered pour récupérer au fil de l’eau
    next_log = t0 + log_every
    done = 0
    for total, st, rbtc, ebtc in pool.imap_unordered(_eval_chunk, chunks):
        total_return += total
        stats.update(st)
        for k, v in rbtc.items():
            returns_by_tc[k] += v
        for k, v in ebtc.items():
            episodes_by_tc[k] += v
        done += 1

        now = time.perf_counter()
        if now >= next_log:
            processed = sum(chunks[:done])
            dt = now - t0
            eps_per_s = int(processed / max(1e-9, dt))
            ev_tmp = total_return / max(1, processed)
            print(f"[eval] {processed:,}/{rounds:,}  ⚡ {eps_per_s:,.0f} eps/s  EV≈{ev_tmp:+.4f}")
            next_log = now + log_every

    pool.close()
    pool.join()

    dt = time.perf_counter() - t0
    eps_per_s = int(rounds / max(1e-9, dt))
    ev = total_return / max(1, rounds)
    return ev, stats, returns_by_tc, episodes_by_tc, eps_per_s

# --------------------- Pretty print ---------------------
def print_report(ev: float, stats: Counter, returns_by_tc: Dict[int, float], episodes_by_tc: Dict[int, int], eps_per_s: int):
    total = max(1, sum(stats.values()))
    winp = 100.0 * stats.get("win", 0) / total
    pushp = 100.0 * stats.get("push", 0) / total
    lossp = 100.0 * stats.get("loss", 0) / total
    print(f"\nEV/round (greedy) = {ev:+.4f}   |   W/P/L = {winp:.1f}% / {pushp:.1f}% / {lossp:.1f}%   |   speed ≈ {eps_per_s:,} eps/s")

    # EV par TC
    tcs = sorted(episodes_by_tc.keys())
    if tcs:
        print("\nEV par True Count (TC):")
        print("  TC   Episodes        EV/round")
        for tc in tcs:
            n = episodes_by_tc[tc]
            if n <= 0:
                continue
            ev_tc = returns_by_tc[tc] / n
            print(f" {tc:>3}  {n:>10,}    {ev_tc:+.4f}")
    else:
        print("\n(pas de stats par TC disponibles)")

# --------------------- CLI ---------------------
def main():
    ap = argparse.ArgumentParser(description="Évaluation greedy (ε=0) de la Q-Table, sans apprentissage.")
    ap.add_argument("--save-path", default="qtable.pkl", help="Chemin du qtable.pkl")
    ap.add_argument("--rounds", type=int, default=1_000_000, help="Nombre de mains à simuler")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1), help="Workers pour évaluation parallèle (0/1 = mono)")
    ap.add_argument("--log-every", type=float, default=3.0, help="Logs périodiques (MP)")
    ap.add_argument("--num-decks", type=int, default=8)
    ap.add_argument("--penetration", type=float, default=0.5)
    ap.add_argument("--h17", action="store_true", help="Dealer hits soft 17 (sinon S17)")
    args = ap.parse_args()

    # Windows: spawn
    if sys.platform.startswith("win"):
        mp.set_start_method("spawn", force=True)

    qtab = load_qtable(args.save_path)
    qtab.epsilon = 0.0  # sécurité

    print(f"🚀 Évaluation greedy: rounds={args.rounds:,}, workers={args.workers}, H17={bool(args.h17)}")
    if args.workers <= 1:
        ev, stats, rbtc, ebtc, epsps = evaluate_single(
            qtab, rounds=args.rounds, num_decks=args.num_decks, penetration=args.penetration, h17=bool(args.h17)
        )
    else:
        ev, stats, rbtc, ebtc, epsps = evaluate_mp(
            qtab, rounds=args.rounds, workers=args.workers, num_decks=args.num_decks,
            penetration=args.penetration, h17=bool(args.h17), log_every=args.log_every
        )

    print_report(ev, stats, rbtc, ebtc, epsps)

if __name__ == "__main__":
    main()
