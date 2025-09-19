# train_mp.py
import argparse
import os
import pickle
import signal
import time
import sys
import multiprocessing as mp
from collections import defaultdict
from typing import Dict, Any

import numpy as np

from freebet.env import FreeBetEnv
from freebet.rl.qtable import QTable

SAVE_PATH_DEFAULT = "qtable.pkl"


# --------------------- Utils: QTable <-> dict ---------------------
def qtable_to_dict(qtab: QTable) -> dict:
    return {
        "q_by_tc": {tc: dict(qtab.q_by_tc[tc]) for tc in qtab.q_by_tc},
        "visits_by_tc": {tc: dict(qtab.visits_by_tc[tc]) for tc in qtab.visits_by_tc},
        "episodes": int(qtab.episodes),
        "epsilon": float(getattr(qtab, "epsilon", 0.0)),
    }


def dict_to_qtable(d: dict) -> QTable:
    qtab = QTable()
    qtab.q_by_tc = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))
    for tc, dd in d.get("q_by_tc", {}).items():
        qtab.q_by_tc[tc] = defaultdict(lambda: np.zeros(4), {s: np.array(v, dtype=float) for s, v in dd.items()})
    qtab.visits_by_tc = defaultdict(lambda: defaultdict(lambda: np.zeros(4, dtype=int)))
    for tc, dd in d.get("visits_by_tc", {}).items():
        qtab.visits_by_tc[tc] = defaultdict(lambda: np.zeros(4, dtype=int), {s: np.array(v, dtype=int) for s, v in dd.items()})
    qtab.episodes = int(d.get("episodes", 0))
    qtab.epsilon = float(d.get("epsilon", getattr(qtab, "epsilon", 0.0)))
    return qtab


def save_qtable(qtab: QTable, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(qtable_to_dict(qtab), f)


def load_qtable(path: str) -> QTable:
    qtab = QTable()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return qtab
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return dict_to_qtable(data)
    except Exception:
        return qtab


# --------------------- CLI helper: parse --tc-only ---------------------
def parse_tc_only(spec: str):
    """
    Parse a CSV or range spec into a sorted list of TC integers clipped to [-5, 5].
      - "" -> None (means 'all TC')
      - "4,5" -> [4, 5]
      - "4:5" -> [4, 5]
      - "-2:3,5" -> [-2, -1, 0, 1, 2, 3, 5]
    """
    if not spec:
        return None
    vals = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            a, b = part.split(":")
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            for v in range(a, b + 1):
                vals.add(max(-5, min(5, v)))
        else:
            v = int(part)
            vals.add(max(-5, min(5, v)))
    return sorted(vals)


# --------------------- Outcome helpers ---------------------
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


# --------------------- Worker ---------------------
def snapshot_visits(visits_by_tc) -> Dict[int, Dict[Any, np.ndarray]]:
    out = {}
    for tc, dd in visits_by_tc.items():
        out_tc = {}
        for s, v in dd.items():
            out_tc[s] = v.copy()
        out[tc] = out_tc
    return out


def visits_delta(curr, prev) -> Dict[int, Dict[Any, np.ndarray]]:
    out = {}
    for tc, dd in curr.items():
        for s, v_now in dd.items():
            v_prev = prev.get(tc, {}).get(s)
            if v_prev is None:
                delta = v_now
            else:
                delta = v_now - v_prev
            if np.any(delta > 0):
                out.setdefault(tc, {})[s] = delta.astype(int, copy=False)
    return out


def extract_q_for_states(q_by_tc, keys_dict) -> Dict[int, Dict[Any, np.ndarray]]:
    out = {}
    for tc, dd in keys_dict.items():
        for s in dd.keys():
            q = q_by_tc.get(tc, {}).get(s)
            if q is not None:
                out.setdefault(tc, {})[s] = q
    return out


def worker_proc(wid: int, cfg: dict, out_q: mp.Queue, stop_ev: mp.Event):
    # init env & qtab
    env = FreeBetEnv(
        num_decks=cfg["num_decks"],
        penetration=cfg["penetration"],
        dealer_hits_soft_17=cfg["h17"],
    )
    if cfg["resume_dict"] is not None:
        qtab = dict_to_qtable(cfg["resume_dict"])
    else:
        qtab = QTable()
    qtab.epsilon = cfg["epsilon"]

    alpha = cfg["alpha"]
    report_eps = cfg["report_eps"]
    allowed_tcs = set(cfg["tc_only"]) if cfg.get("tc_only") else None  # None => train on all TCs

    # stats (count only trained episodes)
    wins = pushes = losses = 0
    returns_sum = 0.0
    episodes_since_last = 0

    last_visits_snapshot = snapshot_visits(qtab.visits_by_tc)

    while not stop_ev.is_set():
        # run a small batch of hands (we always play to advance the shoe/TC)
        for _ in range(report_eps):
            if stop_ev.is_set():
                break
            tc = max(-5, min(5, int(env.shoe.true_count())))
            trans, rewards, outcomes = env.play_round(qtab, qtab.epsilon, tc)
            G = float(sum(rewards)) if rewards else 0.0

            # Train only if TC in filter (or no filter)
            train_this = (allowed_tcs is None) or (tc in allowed_tcs)

            if train_this and trans:
                qtab.update_episode(trans, G, alpha, tc)

                # Count stats/episodes only for trained rounds
                episodes_since_last += 1
                returns_sum += G
                if is_win(outcomes):
                    wins += 1
                elif is_push(outcomes):
                    pushes += 1
                else:
                    losses += 1

        # Build delta of visits (since last snapshot)
        curr_visits = qtab.visits_by_tc
        v_add = visits_delta(curr_visits, last_visits_snapshot)

        # Only flush if we actually trained something
        if episodes_since_last > 0 and v_add:
            q_partial = extract_q_for_states(qtab.q_by_tc, v_add)
            payload = {
                "type": "update",
                "worker": wid,
                "episodes": episodes_since_last,
                "returns_sum": returns_sum,
                "wins": wins,
                "push": pushes,
                "loss": losses,
                "visits_add": v_add,   # dict[int] -> dict[state] -> np.ndarray(int, shape=(4,))
                "q_by_tc": q_partial,  # dict[int] -> dict[state] -> np.ndarray(float, shape=(4,))
            }
            out_q.put_nowait(payload)

            # reset local accumulators and snapshot
            wins = pushes = losses = 0
            returns_sum = 0.0
            episodes_since_last = 0
            last_visits_snapshot = snapshot_visits(curr_visits)

    # final flush (if anything left)
    curr_visits = qtab.visits_by_tc
    v_add = visits_delta(curr_visits, last_visits_snapshot)
    if episodes_since_last > 0 and v_add:
        q_partial = extract_q_for_states(qtab.q_by_tc, v_add)
        payload = {
            "type": "update",
            "worker": wid,
            "episodes": episodes_since_last,
            "returns_sum": returns_sum,
            "wins": wins,
            "push": pushes,
            "loss": losses,
            "visits_add": v_add,
            "q_by_tc": q_partial,
        }
        out_q.put_nowait(payload)

    out_q.put({"type": "done", "worker": wid})


# --------------------- Master merge ---------------------
def merge_update_into_master(master: QTable, upd: dict):
    # For each (tc, state), update visits and do per-action weighted average of Q
    for tc, dd in upd["visits_add"].items():
        for s, v_add_list in dd.items():
            v_add = np.array(v_add_list, dtype=int)
            if v_add.sum() == 0:
                continue
            q_add = np.array(upd["q_by_tc"][tc][s], dtype=float)

            q_m = master.q_by_tc[tc][s]
            v_m = master.visits_by_tc[tc][s]

            tot = v_m + v_add
            # per-action weighted mean (skip zeros)
            for a in range(4):
                if tot[a] > 0:
                    q_m[a] = (q_m[a] * v_m[a] + q_add[a] * v_add[a]) / tot[a]
            master.q_by_tc[tc][s] = q_m
            master.visits_by_tc[tc][s] = tot


def master_loop(args):
    # load / init
    master = load_qtable(args.save_path) if (args.resume and os.path.exists(args.save_path)) else QTable()
    master.epsilon = args.epsilon

    out_q: mp.Queue = mp.Queue(maxsize=args.workers * 4)
    stop_ev = mp.Event()

    resume_dict = qtable_to_dict(master) if args.resume and master.episodes > 0 else None

    # parse tc-only filter once here, pass to workers
    tc_only = parse_tc_only(args.tc_only)

    # start workers
    workers = []
    for wid in range(args.workers):
        cfg = dict(
            num_decks=args.num_decks,
            penetration=args.penetration,
            h17=bool(args.h17),
            alpha=args.alpha,
            epsilon=args.epsilon,
            report_eps=args.report_eps,
            resume_dict=resume_dict,
            tc_only=tc_only,  # << NEW
        )
        p = mp.Process(target=worker_proc, args=(wid, cfg, out_q, stop_ev), daemon=True)
        p.start()
        workers.append(p)

    # stats & timers
    total_target = args.episodes
    total_eps = int(master.episodes)  # counts trained episodes only
    wins = pushes = losses = 0
    returns_sum = 0.0

    t0 = time.perf_counter()
    t_last = t0
    last_save_t = t0
    eps_at_last_save = total_eps

    # ctrl+c → stop workers cleanly
    def handle_sigint(sig, frame):
        print("\n[!] Ctrl+C → arrêt des workers…")
        stop_ev.set()

    signal.signal(signal.SIGINT, handle_sigint)

    done_workers = 0

    while True:
        # stop condition
        if total_target > 0 and total_eps >= total_target:
            stop_ev.set()

        # try receive
        try:
            msg = out_q.get(timeout=0.5)
        except Exception:
            msg = None

        if msg:
            if msg["type"] == "update":
                # merge Q + visits
                merge_update_into_master(master, msg)
                # global counters (trained episodes only)
                total_eps += int(msg["episodes"])
                returns_sum += float(msg["returns_sum"])
                wins += int(msg["wins"])
                pushes += int(msg["push"])
                losses += int(msg["loss"])
                master.episodes = total_eps  # keep in sync

            elif msg["type"] == "done":
                done_workers += 1

        # logs
        now = time.perf_counter()
        if (now - t_last) >= args.log_every:
            dt = now - t_last
            eps_delta = total_eps - getattr(master, "_last_log_eps", total_eps)
            eps_per_s = eps_delta / max(1e-9, dt)
            total = max(1, wins + pushes + losses)
            winp = 100.0 * wins / total
            pushp = 100.0 * pushes / total
            lossp = 100.0 * losses / total
            ev = returns_sum / max(1, total_eps if total_eps > 0 else 1)
            print(
                f"[{total_eps:,} eps] 👷×{args.workers}  ⚡ {eps_per_s:,.0f} eps/s | "
                f"EV/round={ev:+.4f} | W/P/L: {winp:.1f}%/{pushp:.1f}%/{lossp:.1f}%"
            )
            master._last_log_eps = total_eps
            t_last = now

        # autosave
        if (now - last_save_t) >= args.autosave_secs and total_eps != eps_at_last_save:
            save_qtable(master, args.save_path)
            last_save_t = now
            eps_at_last_save = total_eps
            print(f"💾 Autosave: {args.save_path} @ {total_eps:,} eps")

        # exit if all workers done and no target
        if stop_ev.is_set() and done_workers >= len(workers):
            break

    # join and final save
    for p in workers:
        if p.is_alive():
            p.join(timeout=1.0)
    save_qtable(master, args.save_path)
    print(f"✅ Fin. Q-table sauvegardée dans {args.save_path} @ {total_eps:,} eps")


# --------------------- CLI ---------------------
def main():
    ap = argparse.ArgumentParser(description="Free Bet Blackjack RL - multi-process trainer")
    ap.add_argument("--save-path", default=SAVE_PATH_DEFAULT)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--epsilon", type=float, default=0.20)
    ap.add_argument("--gamma", type=float, default=1.0, help="Si utilisé dans QTable")
    ap.add_argument("--episodes", type=int, default=5_000_000, help="0 = infini (compte les épisodes entraînés)")
    ap.add_argument("--report-eps", type=int, default=50_000, help="Mains jouées entre deux rapports worker→maître")
    ap.add_argument("--autosave-secs", type=float, default=60.0)
    ap.add_argument("--log-every", type=float, default=2.0)
    ap.add_argument("--num-decks", type=int, default=8)
    ap.add_argument("--penetration", type=float, default=0.5)
    ap.add_argument("--h17", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))

    # NEW: filter training to specific true counts
    ap.add_argument(
        "--tc-only",
        type=str,
        default="",
        help="Liste de TC à entraîner (CSV ou range a:b). Ex: '4,5' ou '4:5'. Vide = tous."
    )

    args = ap.parse_args()

    # Windows: spawn
    if sys.platform.startswith("win"):
        mp.set_start_method("spawn", force=True)

    print(
        f"🚀 MP training: workers={args.workers}, report_eps={args.report_eps:,}, "
        f"autosave={int(args.autosave_secs)}s, tc_only={parse_tc_only(args.tc_only)}"
    )
    master_loop(args)


if __name__ == "__main__":
    main()
