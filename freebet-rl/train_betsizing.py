# train_betsizing.py — RL bet sizing (loop + checkpoint + allow-zero)
# -----------------------------------------------------------
# État    : True Count (TC) borné [-5..+5]
# Action  : fraction de bankroll f ∈ actions (ex: 0.5%, 1%, ..., 20%)
# Reward  : log(1 + f * g) avec g = retour unitaire par main (mises=1 dans l'env)
# Jeu     : les décisions H/S/D/Split viennent de la Q-Table (greedy)
# Loop    : --loop avec époques; Ctrl+C pour stopper, checkpoint NPZ pour reprendre
# -----------------------------------------------------------

import os
import sys
import json
import math
import time
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from freebet.env import FreeBetEnv
from freebet.rl.qtable import QTable

TC_MIN, TC_MAX = -5, 5


def clamp_tc(tc: int) -> int:
    return max(TC_MIN, min(TC_MAX, tc))


# ---------- Chargement Q-Table (jeu greedy, pas d'apprentissage ici) ----------
def load_qtable(path: str) -> QTable:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"[ERR] Q-Table introuvable ou vide: {path}", flush=True)
        sys.exit(1)
    import pickle
    qtab = QTable()
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        qtab.q_by_tc = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))
        for tc, qdict in data.get("q_by_tc", {}).items():
            qtab.q_by_tc[tc] = defaultdict(
                lambda: np.zeros(4),
                {s: np.asarray(v, dtype=float) for s, v in qdict.items()},
            )
        qtab.visits_by_tc = defaultdict(lambda: defaultdict(lambda: np.zeros(4, dtype=int)))
        for tc, vdict in data.get("visits_by_tc", {}).items():
            qtab.visits_by_tc[tc] = defaultdict(
                lambda: np.zeros(4, dtype=int),
                {s: np.asarray(v, dtype=int) for s, v in vdict.items()},
            )
        qtab.episodes = int(data.get("episodes", 0))
        qtab.epsilon = 0.0  # jeu 100% greedy
        return qtab
    except Exception as e:
        print(f"[ERR] Échec chargement Q-Table: {e}", flush=True)
        sys.exit(1)


# ---------- Simulation bankroll-aware pour évaluer une policy f(TC) ----------
def simulate_with_fraction_policy(
    env: FreeBetEnv,
    qtab: QTable,
    frac_by_tc: Dict[int, float],
    bankroll_eur: float,
    unit_value_eur: float,
    table_min_eur: float,
    table_max_eur: Optional[float],
    rounds: int,
    burn_when_zero: int = 0,
    min_tc_to_play: int = 0,
) -> Dict[str, float]:
    br = float(bankroll_eur)
    traj = [br]
    rounds_win = rounds_push = rounds_loss = 0
    episodes_by_tc: Dict[int, int] = defaultdict(int)

    for _ in range(int(rounds)):
        tc = clamp_tc(env.shoe.true_count())
        f = float(frac_by_tc.get(tc, 0.0)) if tc >= min_tc_to_play else 0.0

        # Pas de mise -> brûle quelques cartes puis continue
        if f <= 0.0 or br < table_min_eur:
            for __ in range(int(burn_when_zero)):
                env.shoe.draw()
            continue

        # Mise € selon fraction, bornée table
        bet_eur = max(table_min_eur, f * br)
        if table_max_eur is not None:
            bet_eur = min(bet_eur, table_max_eur)

        # conversion en unités (arrondi à l'unité)
        units = max(1, int(round(bet_eur / unit_value_eur)))
        stake_eur = units * unit_value_eur

        # Joue la main (greedy)
        _, rewards, outcomes = env.play_round(qtab, 0.0, tc)
        g = float(np.sum(rewards))  # retour/unité
        delta = g * stake_eur

        br += delta
        traj.append(br)

        episodes_by_tc[tc] += 1
        if outcomes.get("win", 0) > 0:
            rounds_win += 1
        elif outcomes.get("push", 0) > 0:
            rounds_push += 1
        else:
            rounds_loss += 1

    total_rounds = max(1, rounds_win + rounds_push + rounds_loss)
    ev_per_round_eur = (br - bankroll_eur) / total_rounds
    max_drawdown = float(np.max(traj) - np.min(traj)) if traj else 0.0
    return {
        "final_bankroll_eur": br,
        "pnl_eur": br - bankroll_eur,
        "ev_per_round_eur": ev_per_round_eur,
        "rounds_win": rounds_win,
        "rounds_push": rounds_push,
        "rounds_loss": rounds_loss,
        "episodes_by_tc": dict(episodes_by_tc),
        "trajectory_len": len(traj),
        "max_drawdown_eur": max_drawdown,
        "shoe_sessions": env.shoe.sessions_played,
    }


def greedy_policy_from_Q(Q: np.ndarray, action_fracs: List[float]) -> Dict[int, float]:
    return {tc: float(action_fracs[int(np.argmax(Q[tc - TC_MIN]))]) for tc in range(TC_MIN, TC_MAX + 1)}


def format_policy_short(frac_by_tc: Dict[int, float], min_tc_to_play: int = 0) -> str:
    parts: List[str] = []
    for tc in range(TC_MIN, TC_MAX + 1):
        f = 0.0 if tc < min_tc_to_play else float(frac_by_tc.get(tc, 0.0))
        if f > 0:
            parts.append(f"{tc}:{int(round(100 * f))}%")
    return " ".join(parts) if parts else "all≤0%"


# ---------- Entraînement RL (par bloc), avec reprise d'état ----------
def train_rl_bet_policy(
    env: FreeBetEnv,
    qtab: QTable,
    n_rounds: int,
    action_fracs: List[float],
    gamma: float = 0.99,
    alpha: float = 0.10,
    eps_start: float = 0.10,
    eps_end: float = 0.01,
    burn_when_zero: int = 8,
    min_tc_to_play: int = 0,
    eval_every: int = 50_000,
    eval_rounds: int = 100_000,
    bankroll_eur: float = 200.0,
    unit_value_eur: float = 1.0,
    table_min_eur: float = 1.0,
    table_max_eur: Optional[float] = None,
    seed: Optional[int] = None,
    policy_out: Optional[str] = None,
    env_factory=None,
    Q_init: Optional[np.ndarray] = None,
    eps_init: Optional[float] = None,
    ewma_init: Optional[float] = None,
) -> Tuple[Dict[int, float], np.ndarray, float, Optional[float]]:
    if env_factory is None:
        raise ValueError("env_factory is required.")

    rng = np.random.default_rng(seed)
    S = TC_MAX - TC_MIN + 1
    A = len(action_fracs)
    Q = Q_init.copy() if Q_init is not None else np.zeros((S, A), dtype=float)

    def s_idx(tc: int) -> int:
        return clamp_tc(tc) - TC_MIN

    # epsilon continu entre blocs
    eps = float(eps_start) if eps_init is None else max(eps_end, min(eps_start, float(eps_init)))
    eps_decay = (eps_end / eps_start) ** (1.0 / max(1, n_rounds))

    played = 0
    ewma_reward = ewma_init
    t0 = time.time()

    print("step\tplayed\teps\tavg_logR\tlast_g\tpolicy_snapshot", flush=True)

    for step in range(1, n_rounds + 1):
        tc = clamp_tc(env.shoe.true_count())
        s = s_idx(tc)

        # TC en-dessous du seuil : on brûle et on passe
        if tc < min_tc_to_play:
            for __ in range(int(burn_when_zero)):
                env.shoe.draw()
            if step % eval_every == 0:
                frac_by_tc = greedy_policy_from_Q(Q, action_fracs)
                eval_res = simulate_with_fraction_policy(
                    env=env_factory(seed),
                    qtab=qtab,
                    frac_by_tc=frac_by_tc,
                    bankroll_eur=bankroll_eur,
                    unit_value_eur=unit_value_eur,
                    table_min_eur=table_min_eur,
                    table_max_eur=table_max_eur,
                    rounds=eval_rounds,
                    burn_when_zero=burn_when_zero,
                    min_tc_to_play=min_tc_to_play,
                )
                pol_str = format_policy_short(frac_by_tc, min_tc_to_play)
                print(
                    f"[EVAL] step={step:,}\teps={eps:.3f}\tplayed={played:,}\t"
                    f"EV€/hand={eval_res['ev_per_round_eur']:+.4f}\t"
                    f"PnL€={eval_res['pnl_eur']:+.2f}\tBRf={eval_res['final_bankroll_eur']:.2f}\t"
                    f"DDmax€={eval_res['max_drawdown_eur']:.2f}\tshoes={eval_res['shoe_sessions']}\t"
                    f"policy={pol_str}",
                    flush=True,
                )
            eps = max(eps_end, eps * eps_decay)
            continue

        # ε-greedy sur la fraction
        if rng.random() < eps:
            a = int(rng.integers(A))
        else:
            a = int(np.argmax(Q[s]))
        f = float(action_fracs[a])

        # Joue une main (jeu greedy via Q-Table)
        _, rewards, _ = env.play_round(qtab, 0.0, tc)
        g = float(np.sum(rewards))
        played += 1

        # Reward = log growth bornée
        grow = 1.0 + f * g
        r = math.log(grow) if grow > 1e-12 else math.log(1e-12)
        ewma_reward = r if ewma_reward is None else 0.99 * ewma_reward + 0.01 * r

        # Update Q
        s2 = s_idx(clamp_tc(env.shoe.true_count()))
        Q[s, a] += alpha * (r + gamma * float(np.max(Q[s2])) - Q[s, a])

        # Logs fréquents
        if step % max(1, (eval_every // 10)) == 0:
            pol_str = format_policy_short(greedy_policy_from_Q(Q, action_fracs), min_tc_to_play)
            print(f"{step}\t{played}\t{eps:.3f}\t{(ewma_reward or 0):+.5f}\t{g:+.3f}\t{pol_str}", flush=True)

        # Évaluation périodique
        if step % eval_every == 0:
            frac_by_tc = greedy_policy_from_Q(Q, action_fracs)
            eval_env = env_factory(seed)
            eval_res = simulate_with_fraction_policy(
                env=eval_env,
                qtab=qtab,
                frac_by_tc=frac_by_tc,
                bankroll_eur=bankroll_eur,
                unit_value_eur=unit_value_eur,
                table_min_eur=table_min_eur,
                table_max_eur=table_max_eur,
                rounds=eval_rounds,
                burn_when_zero=burn_when_zero,
                min_tc_to_play=min_tc_to_play,
            )
            pol_str = format_policy_short(frac_by_tc, min_tc_to_play)
            print(
                f"[EVAL] step={step:,}\teps={eps:.3f}\tplayed={played:,}\t"
                f"EV€/hand={eval_res['ev_per_round_eur']:+.4f}\tPnL€={eval_res['pnl_eur']:+.2f}\t"
                f"BRf={eval_res['final_bankroll_eur']:.2f}\tDDmax€={eval_res['max_drawdown_eur']:.2f}\t"
                f"shoes={eval_res['shoe_sessions']}\tpolicy={pol_str}",
                flush=True,
            )

        eps = max(eps_end, eps * eps_decay)

    dt = time.time() - t0
    frac_by_tc = greedy_policy_from_Q(Q, action_fracs)
    if policy_out:
        with open(policy_out, "w", encoding="utf-8") as f:
            json.dump(frac_by_tc, f, ensure_ascii=False, indent=2)
        print(f"[OK] Policy sauvegardée → {policy_out}", flush=True)

    print(f"[DONE-BLOCK] rounds={n_rounds:,}, played={played:,}, time={dt:.1f}s", flush=True)
    print(f"[POLICY] {format_policy_short(frac_by_tc, min_tc_to_play)}", flush=True)
    return frac_by_tc, Q, eps, ewma_reward


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="RL bet sizing (loop, checkpoint, allow-zero).")
    # Environnement de jeu
    p.add_argument("--decks", type=int, default=8)
    p.add_argument("--penetration", type=float, default=0.5)
    p.add_argument("--h17", action="store_true", help="Dealer H17 (défaut: S17)")
    p.add_argument("--min-tc", type=int, default=0, help="Seuil TC minimal pour jouer")
    p.add_argument("--burn-when-zero", type=int, default=8, help="Cartes brûlées quand pas de mise")

    # I/O
    p.add_argument("--qtable-path", type=str, default="qtable.pkl")
    p.add_argument("--policy-out", type=str, default="policy_betsizing.json")
    p.add_argument("--checkpoint", type=str, default="betsizing_ckpt.npz",
                   help="Fichier NPZ pour sauver/reprendre Q/eps/ewma/actions")

    # RL
    p.add_argument("--rounds", type=int, default=1_000_000)  # si --loop absent
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--eps-start", type=float, default=0.10)
    p.add_argument("--eps-end", type=float, default=0.01)
    p.add_argument("--actions", type=str,
                   default="0.005,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20",
                   help="Fractions de bankroll séparées par des virgules")
    p.add_argument("--allow-zero", action="store_true",
                   help="Autoriser 0%% comme action possible (par défaut: interdit)")

    # Évaluation périodique
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--eval-rounds", type=int, default=100_000)

    # Contexte bankroll
    p.add_argument("--bankroll-eur", type=float, default=200.0)
    p.add_argument("--unit-eur", type=float, default=1.0)
    p.add_argument("--table-min-eur", type=float, default=1.0)
    p.add_argument("--table-max-eur", type=float, default=0.0, help="0 = illimité")

    # Divers
    p.add_argument("--seed", type=int, default=0)

    # Mode boucle
    p.add_argument("--loop", action="store_true", help="Entraîner par époques jusqu'au Ctrl+C")
    p.add_argument("--epoch-rounds", type=int, default=200_000)
    p.add_argument("--autosave-every", type=int, default=1, help="Autosave policy toutes N époques")
    return p.parse_args()


def maybe_load_checkpoint(path: str, action_fracs: List[float], S: int) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float], List[float]]:
    if not path or not os.path.exists(path):
        return None, None, None, action_fracs
    try:
        data = np.load(path, allow_pickle=True)
        Q = data["Q"]
        eps = float(data["eps"])
        ewma = float(data["ewma"])
        actions_ckpt = data["actions"].tolist()
        if not isinstance(actions_ckpt, list):
            actions_ckpt = list(actions_ckpt)

        # Vérif compat : nb d'actions et shape de Q
        if len(actions_ckpt) == len(action_fracs) and Q.shape == (S, len(action_fracs)):
            print(f"[CKPT] Reprise depuis {path} (actions={actions_ckpt})", flush=True)
            return Q, eps, ewma, actions_ckpt
        else:
            print(f"[CKPT] Incompatibilité détectée (ckpt actions/shape ≠ args). Ignore le checkpoint.", flush=True)
            return None, None, None, action_fracs
    except Exception as e:
        print(f"[CKPT] Impossible de charger {path}: {e}", flush=True)
        return None, None, None, action_fracs


def save_checkpoint(path: str, Q: np.ndarray, eps: float, ewma: Optional[float], action_fracs: List[float]) -> None:
    try:
        np.savez(path, Q=Q, eps=np.array(eps), ewma=np.array(ewma if ewma is not None else np.nan),
                 actions=np.array(action_fracs, dtype=float))
        print(f"[CKPT] Sauvegardé → {path}", flush=True)
    except Exception as e:
        print(f"[CKPT] Échec sauvegarde {path}: {e}", flush=True)


def main():
    args = parse_args()

    # Parse & valider actions
    action_fracs = sorted(set(float(x) for x in args.actions.split(",") if x.strip() != ""))
    if any(f < 0 for f in action_fracs):
        print("[ERR] Fractions de bankroll doivent être ≥ 0", flush=True)
        sys.exit(1)

    if args.allow_zero:
        if 0.0 not in action_fracs:
            action_fracs = [0.0] + action_fracs
    else:
        # On retire 0% -> pas de wonging via sizing (tu gardes --min-tc pour TC<0)
        action_fracs = [f for f in action_fracs if f > 0.0]
        if not action_fracs:
            print("[ERR] --actions doit contenir au moins une fraction > 0 (ou utilise --allow-zero)", flush=True)
            sys.exit(1)

    qtab = load_qtable(args.qtable_path)

    # Factory pour recréer un env identique pour les évaluations
    def make_env(seed=None):
        return FreeBetEnv(
            num_decks=args.decks,
            penetration=args.penetration,
            dealer_hits_soft_17=bool(args.h17),
            seed=seed,
        )

    seed = None if args.seed == 0 else int(args.seed)
    env = make_env(seed)

    table_max = None if args.table_max_eur <= 0 else float(args.table_max_eur)

    # Reprise éventuelle depuis checkpoint
    S = TC_MAX - TC_MIN + 1
    Q_ckpt, eps_ckpt, ewma_ckpt, action_fracs = maybe_load_checkpoint(args.checkpoint, action_fracs, S)

    if not args.loop:
        # Run simple
        frac_by_tc, Q, eps_out, ewma_out = train_rl_bet_policy(
            env=env,
            qtab=qtab,
            n_rounds=int(args.rounds),
            action_fracs=action_fracs,
            gamma=float(args.gamma),
            alpha=float(args.alpha),
            eps_start=float(args.eps_start),
            eps_end=float(args.eps_end),
            burn_when_zero=int(args.burn_when_zero),
            min_tc_to_play=int(args.min_tc),
            eval_every=int(args.eval_every),
            eval_rounds=int(args.eval_rounds),
            bankroll_eur=float(args.bankroll_eur),
            unit_value_eur=float(args.unit_eur),
            table_min_eur=float(args.table_min_eur),
            table_max_eur=table_max,
            seed=seed,
            policy_out=args.policy_out,
            env_factory=make_env,
            Q_init=Q_ckpt,
            eps_init=eps_ckpt,
            ewma_init=ewma_ckpt,
        )
        if args.checkpoint:
            save_checkpoint(args.checkpoint, Q, eps_out, ewma_out, action_fracs)
        return

    # Mode LOOP : enchaîne des époques jusqu'à Ctrl+C
    print(f"[LOOP] Entraînement continu — Ctrl+C pour arrêter. Bloc={args.epoch_rounds:,} rounds.", flush=True)
    epoch = 0
    Q = Q_ckpt
    eps_cont = eps_ckpt
    ewma_cont = ewma_ckpt

    try:
        while True:
            epoch += 1
            print(f"\n=== EPOCH {epoch} ===", flush=True)
            frac_by_tc, Q, eps_cont, ewma_cont = train_rl_bet_policy(
                env=env,
                qtab=qtab,
                n_rounds=int(args.epoch_rounds),
                action_fracs=action_fracs,
                gamma=float(args.gamma),
                alpha=float(args.alpha),
                eps_start=float(args.eps_start),
                eps_end=float(args.eps_end),
                burn_when_zero=int(args.burn_when_zero),
                min_tc_to_play=int(args.min_tc),
                eval_every=int(args.eval_every),
                eval_rounds=int(args.eval_rounds),
                bankroll_eur=float(args.bankroll_eur),
                unit_value_eur=float(args.unit_eur),
                table_min_eur=float(args.table_min_eur),
                table_max_eur=table_max,
                seed=seed,
                policy_out=None,  # autosave policy plus bas
                env_factory=make_env,
                Q_init=Q,
                eps_init=eps_cont,
                ewma_init=ewma_cont,
            )

            # Autosave checkpoint à chaque époque
            if args.checkpoint:
                save_checkpoint(args.checkpoint, Q, eps_cont, ewma_cont, action_fracs)

            # Autosave policy selon fréquence
            if args.policy_out and (epoch % max(1, args.autosave_every) == 0):
                try:
                    with open(args.policy_out, "w", encoding="utf-8") as f:
                        json.dump(frac_by_tc, f, ensure_ascii=False, indent=2)
                    print(f"[AUTOSAVE] Policy → {args.policy_out}", flush=True)
                except Exception as e:
                    print(f"[AUTOSAVE] Échec policy: {e}", flush=True)

    except KeyboardInterrupt:
        print("\n[STOP] Interruption utilisateur (Ctrl+C).", flush=True)
        if Q is not None and args.checkpoint:
            save_checkpoint(args.checkpoint, Q, eps_cont if eps_cont is not None else float(args.eps_end), ewma_cont, action_fracs)
        if Q is not None and args.policy_out:
            final_policy = greedy_policy_from_Q(Q, action_fracs)
            try:
                with open(args.policy_out, "w", encoding="utf-8") as f:
                    json.dump(final_policy, f, ensure_ascii=False, indent=2)
                print(f"[STOP] Policy finale → {args.policy_out}", flush=True)
                print(f"[FINAL POLICY] {format_policy_short(final_policy, int(args.min_tc))}", flush=True)
            except Exception as e:
                print(f"[STOP] Échec sauvegarde policy finale: {e}", flush=True)


if __name__ == "__main__":
    main()
