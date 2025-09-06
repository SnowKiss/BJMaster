import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- adapte ce chemin si besoin ---
sys.path.append(r"C:\Project\BJMaster\BJMaster\freebet-rl")

from freebet.env import FreeBetEnv
from freebet.cards import hand_value, is_blackjack, is_pair
from basic_strategy import BasicStrategy

# ===================== CONFIG =====================
HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "data")
RESULTS_FILE = os.path.join(os.getcwd(), "stoploss_results.csv")

DEFAULT_MAX_STEPS = 200
DEFAULT_TARGET_GAIN = 1.03
DEFAULT_STOP_LOSSES = [0.95, 0.90, 0.80, 0.70]
DEFAULT_N_SIMULATIONS = 100_000
DEFAULT_BANKROLL = 100
N_WORKERS = max(1, os.cpu_count() or 4)

DTYPE = np.float32
ACTIONS = ['H', 'S', 'D', 'P']
ACT2IDX = {a: i for i, a in enumerate(ACTIONS)}

# ===================== Basic Strategy =====================
@lru_cache(maxsize=1)
def get_bs(hard_csv: str, soft_csv: str, pairs_csv: str) -> BasicStrategy:
    return BasicStrategy(hard_csv=hard_csv, soft_csv=soft_csv, pairs_csv=pairs_csv)

# ===================== Q-Table loader (état sérialisé) =====================
@st.cache_resource
def load_qstate(path: str) -> dict:
    """
    Lecture de qtable.pkl -> dict pur: {tc: {state_tuple: np.ndarray shape(4)}}
    Pas d'objets ni de lambdas => picklable pour ProcessPoolExecutor (Windows).
    """
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        st.warning("⚠️ qtable.pkl introuvable ou vide → politique Q-Table désactivée.")
        return {}
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        raw = data.get("q_by_tc", {})
        qstate: dict[int, dict] = {}
        for tc, inner in raw.items():
            tci = int(tc)
            d = {}
            for k, v in inner.items():
                d[k] = np.array(v, dtype=np.float32)
            qstate[tci] = d
        return qstate
    except Exception as e:
        st.error(f"❌ Erreur lecture qtable.pkl : {e}")
        return {}

# ===================== Hi-Lo / TC =====================
def hilo_value(card: int) -> int:
    if 2 <= card <= 6:
        return 1
    if 7 <= card <= 9:
        return 0
    return -1

def tc_from_rc_nearest(rc: int, cards_left: int) -> int:
    decks_left = max(1.0, cards_left / 52.0)
    return int(round(rc / decks_left))

# ===================== Free Bet Helpers =====================
def is_ace(card: int) -> bool:
    return card == 11  # adapte si besoin

def can_free_split(cards: list[int]) -> bool:
    return len(cards) == 2 and cards[0] == cards[1] and ((2 <= cards[0] <= 9) or is_ace(cards[0]))

def can_free_double(cards: list[int]) -> bool:
    total, _ = hand_value(cards)
    return len(cards) == 2 and (9 <= total <= 11)

def available_actions_freebet(cards: list[int], first_action: bool, already_split: bool, split_aces_lock: bool) -> list[str]:
    if split_aces_lock:
        return ['S']
    acts = ['H', 'S']
    if first_action:
        if can_free_double(cards):
            acts.append('D')
        if (not already_split) and can_free_split(cards):
            acts.append('P')
    return acts

def normalize_action_basic(action: str, cards: list[int], already_split: bool) -> str:
    if action == "P" and (already_split or not can_free_split(cards)):
        return "S"
    if action == "D" and not can_free_double(cards):
        return "H"
    return action

# ===================== Choix d'action Q-Table =====================
def choose_action_qtable(state, avail: list[str], tc: int, qstate: dict) -> str:
    qvals = None
    inner = qstate.get(tc)
    if inner is not None:
        qvals = inner.get(state)
    if qvals is None:
        return 'S' if 'S' in avail else avail[0]
    mask = np.full(4, -1e9, dtype=np.float32)
    for a in avail:
        mask[ACT2IDX[a]] = 0.0
    idx = int(np.argmax(qvals + mask))
    act = ACTIONS[idx]
    if act not in avail:
        return 'S' if 'S' in avail else avail[0]
    return act

# ===================== Simulation d’UNE main =====================
def simulate_hand_freebet_basic(game: FreeBetEnv, player_cards, dealer_cards, hard_csv, soft_csv, pairs_csv) -> DTYPE:
    """
    Basic Strategy sous règles Free Bet.
    """
    bs = get_bs(hard_csv, soft_csv, pairs_csv)
    dealer_up = dealer_cards[1]
    dealer_bj = is_blackjack(dealer_cards)
    player_bj = is_blackjack(player_cards)

    if dealer_bj and player_bj:
        return DTYPE(0.0)
    if dealer_bj and not player_bj:
        return DTYPE(-1.0)
    if player_bj and not dealer_bj:
        return DTYPE(1.5)

    hands = [{
        "cards": player_cards[:],
        "already_split": False,   # interdit le re-split (simple)
        "split_aces_lock": False, # split As → 1 carte puis stand
        "free_extra": False,      # ✅ main gratuite issue d’un split
        "doubled_free": False,    # double gratuit appliqué
        "is_initial": True,       # pour BJ 3:2 naturel
        "first_action": True,
    }]

    i = 0
    while i < len(hands):
        h = hands[i]
        cards = h["cards"]
        while True:
            if h["split_aces_lock"]:
                break
            total, soft = hand_value(cards)
            pair, pr = is_pair(cards)
            pair_rank = pr if (pair and len(cards) == 2) else 0
            action = bs.get_action(total, soft, pair_rank, dealer_up)
            action = normalize_action_basic(action, cards, h["already_split"])
            if action == "H":
                cards.append(game.shoe.draw())
                h["first_action"] = False
                t, _ = hand_value(cards)
                if t > 21:
                    break
            elif action == "S":
                break
            elif action == "D":
                cards.append(game.shoe.draw())
                h["doubled_free"] = True
                h["first_action"] = False
                break
            elif action == "P":
                v = cards[0]
                c1 = game.shoe.draw()
                cards[:] = [v, c1]
                c2 = game.shoe.draw()
                newh = {
                    "cards": [v, c2],
                    "already_split": True,
                    "split_aces_lock": False,
                    "free_extra": True,      # ✅ seule la main ajoutée est “gratuite”
                    "doubled_free": False,
                    "is_initial": False,
                    "first_action": True,
                }
                hands.append(newh)
                if is_ace(v):
                    h["split_aces_lock"] = True
                    newh["split_aces_lock"] = True
                    break
                h["already_split"] = True
                h["first_action"] = False
            else:
                break
        i += 1

    if any(hand_value(h["cards"])[0] <= 21 for h in hands):
        dealer_cards = game.dealer_play(dealer_cards)
    dealer_total, _ = hand_value(dealer_cards)

    delta = 0.0
    for h in hands:
        cards = h["cards"]
        pt, _ = hand_value(cards)

        # BJ naturel 3:2 (main initiale 2 cartes non issue de split)
        if h["is_initial"] and len(cards) == 2 and pt == 21:
            delta += 1.5
            continue

        if pt > 21:
            if h["doubled_free"]:
                delta -= 1.0
            elif h["free_extra"]:
                delta += 0.0
            else:
                delta -= 1.0
            continue

        if dealer_total == 22:
            continue  # push

        if dealer_total > 21:
            delta += 2.0 if h["doubled_free"] else 1.0
            continue

        if pt > dealer_total:
            delta += 2.0 if h["doubled_free"] else 1.0
        elif pt < dealer_total:
            if h["doubled_free"]:
                delta -= 1.0
            elif h["free_extra"]:
                delta += 0.0
            else:
                delta -= 1.0
        else:
            pass
    return DTYPE(delta)

def simulate_hand_freebet_qtable_collect(game: FreeBetEnv, player_cards, dealer_cards, tc: int, qstate: dict) -> tuple[DTYPE, list[int]]:
    """
    Même règles Free Bet, mais action = argmax Q-Table pour (state, tc).
    Retourne (delta, used_cards_after_initial) pour mettre à jour le RC dans le caller.
    """
    used = []
    dealer_up = dealer_cards[1]
    dealer_bj = is_blackjack(dealer_cards)
    player_bj = is_blackjack(player_cards)

    if dealer_bj and player_bj:
        return DTYPE(0.0), used
    if dealer_bj and not player_bj:
        return DTYPE(-1.0), used
    if player_bj and not dealer_bj:
        return DTYPE(1.5), used

    hands = [{
        "cards": player_cards[:],
        "already_split": False,
        "split_aces_lock": False,
        "free_extra": False,      # ✅ pour les payouts Free Bet
        "doubled_free": False,
        "is_initial": True,
        "first_action": True,
    }]

    i = 0
    while i < len(hands):
        h = hands[i]
        cards = h["cards"]
        while True:
            if h["split_aces_lock"]:
                break
            avail = available_actions_freebet(cards, h["first_action"], h["already_split"], h["split_aces_lock"])
            state = game.state_key(cards, dealer_up, h["first_action"])
            action = choose_action_qtable(state, avail, tc, qstate)
            if action == "H":
                c = game.shoe.draw()
                used.append(c)
                cards.append(c)
                h["first_action"] = False
                t, _ = hand_value(cards)
                if t > 21:
                    break
            elif action == "S":
                break
            elif action == "D":
                c = game.shoe.draw()
                used.append(c)
                cards.append(c)
                h["doubled_free"] = True
                h["first_action"] = False
                break
            elif action == "P":
                v = cards[0]
                c1 = game.shoe.draw(); used.append(c1)
                cards[:] = [v, c1]
                c2 = game.shoe.draw(); used.append(c2)
                newh = {
                    "cards": [v, c2],
                    "already_split": True,
                    "split_aces_lock": False,
                    "free_extra": True,   # ✅ seule la main ajoutée est gratuite
                    "doubled_free": False,
                    "is_initial": False,
                    "first_action": True,
                }
                hands.append(newh)
                if is_ace(v):
                    h["split_aces_lock"] = True
                    newh["split_aces_lock"] = True
                    break
                h["already_split"] = True
                h["first_action"] = False
            else:
                break
        i += 1

    if any(hand_value(h["cards"])[0] <= 21 for h in hands):
        pre = len(dealer_cards)
        dealer_cards = game.dealer_play(dealer_cards)
        if len(dealer_cards) > pre:
            used.extend(dealer_cards[pre:])
    dealer_total, _ = hand_value(dealer_cards)

    delta = 0.0
    for h in hands:
        cards = h["cards"]
        pt, _ = hand_value(cards)
        if h["is_initial"] and len(cards) == 2 and pt == 21:
            delta += 1.5
            continue
        if pt > 21:
            if h["doubled_free"]:
                delta -= 1.0
            elif h["free_extra"]:
                delta += 0.0
            else:
                delta -= 1.0
            continue
        if dealer_total == 22:
            continue
        if dealer_total > 21:
            delta += 2.0 if h["doubled_free"] else 1.0
            continue
        if pt > dealer_total:
            delta += 2.0 if h["doubled_free"] else 1.0
        elif pt < dealer_total:
            if h["doubled_free"]:
                delta -= 1.0
            elif h["free_extra"]:
                delta += 0.0
            else:
                delta -= 1.0
        else:
            pass
    return DTYPE(delta), used

# ===================== Chunks (parallélisés) =====================
def simulate_chunk_basic(n_runs: int, max_steps: int, hard_csv: str, soft_csv: str, pairs_csv: str) -> np.ndarray:
    game = FreeBetEnv(num_decks=8, penetration=0.5, dealer_hits_soft_17=True)
    out = np.zeros((n_runs, max_steps), dtype=DTYPE)
    for i in range(n_runs):
        for t in range(max_steps):
            p, d = game.initial_deal()
            out[i, t] = simulate_hand_freebet_basic(game, p, d, hard_csv, soft_csv, pairs_csv)
    return out

# Q-Table: global injecté côté worker
_G_QSTATE = None
def _init_worker_qstate(qstate: dict):
    global _G_QSTATE
    _G_QSTATE = qstate

def simulate_chunk_qtable(n_runs: int, max_steps: int) -> np.ndarray:
    qstate = _G_QSTATE or {}
    game = FreeBetEnv(num_decks=8, penetration=0.5, dealer_hits_soft_17=True)
    out = np.zeros((n_runs, max_steps), dtype=DTYPE)
    for i in range(n_runs):
        rc = 0  # RC par run ✅
        for t in range(max_steps):
            p, d = game.initial_deal()
            rc += sum(hilo_value(c) for c in (p + d))
            tc = tc_from_rc_nearest(rc, game.shoe.cards_left())
            delta, used = simulate_hand_freebet_qtable_collect(game, p, d, tc, qstate)
            out[i, t] = delta
            if used:
                rc += sum(hilo_value(c) for c in used)
    return out

def simulate_runs_parallel_basic(n_runs: int, max_steps: int, hard_csv: str, soft_csv: str, pairs_csv: str, progress_cb=None) -> np.ndarray:
    if n_runs <= 0:
        return np.zeros((0, max_steps), dtype=DTYPE)
    n = max(1, N_WORKERS)
    base = n_runs // n
    rem = n_runs % n
    sizes = [base + (1 if i < rem else 0) for i in range(n)]
    sizes = [s for s in sizes if s > 0]
    arrays, done = [], 0
    with ProcessPoolExecutor(max_workers=len(sizes)) as ex:
        futs = [ex.submit(simulate_chunk_basic, s, max_steps, hard_csv, soft_csv, pairs_csv) for s in sizes]
        for f in as_completed(futs):
            arrays.append(f.result())
            done += 1
            if progress_cb:
                progress_cb(done, len(sizes))
    return np.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]

def simulate_runs_parallel_qtable(n_runs: int, max_steps: int, qstate: dict, progress_cb=None) -> np.ndarray:
    if n_runs <= 0 or not qstate:
        return np.zeros((0, max_steps), dtype=DTYPE)
    n = max(1, N_WORKERS)
    base = n_runs // n
    rem = n_runs % n
    sizes = [base + (1 if i < rem else 0) for i in range(n)]
    sizes = [s for s in sizes if s > 0]
    arrays, done = [], 0
    with ProcessPoolExecutor(max_workers=len(sizes), initializer=_init_worker_qstate, initargs=(qstate,)) as ex:
        futs = [ex.submit(simulate_chunk_qtable, s, max_steps) for s in sizes]
        for f in as_completed(futs):
            arrays.append(f.result())
            done += 1
            if progress_cb:
                progress_cb(done, len(sizes))
    return np.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]

# ---------- Head-to-Head ----------
_G_QSTATE_H2H = None
def _init_worker_qstate_h2h(qstate: dict):
    global _G_QSTATE_H2H
    _G_QSTATE_H2H = qstate

def simulate_chunk_h2h(n_runs: int, max_steps: int, hard_csv: str, soft_csv: str, pairs_csv: str, seed0: int = 1234567):
    qstate = _G_QSTATE_H2H or {}
    out_basic = np.zeros((n_runs, max_steps), dtype=DTYPE)
    out_q     = np.zeros((n_runs, max_steps), dtype=DTYPE)
    for i in range(n_runs):
        seed = seed0 + i
        gb = FreeBetEnv(num_decks=8, penetration=0.5, dealer_hits_soft_17=True, seed=seed)
        gq = FreeBetEnv(num_decks=8, penetration=0.5, dealer_hits_soft_17=True, seed=seed)
        rc = 0  # RC pour la branche Q-table
        for t in range(max_steps):
            pb, db = gb.initial_deal()
            out_basic[i, t] = simulate_hand_freebet_basic(gb, pb, db, hard_csv, soft_csv, pairs_csv)
            pq, dq = gq.initial_deal()
            rc += sum(hilo_value(c) for c in (pq + dq))
            tc = tc_from_rc_nearest(rc, gq.shoe.cards_left())
            delta_q, used_q = simulate_hand_freebet_qtable_collect(gq, pq, dq, tc, qstate)
            out_q[i, t] = delta_q
            if used_q:
                rc += sum(hilo_value(c) for c in used_q)
    return out_basic, out_q

def simulate_runs_parallel_h2h(n_runs: int, max_steps: int, hard_csv: str, soft_csv: str, pairs_csv: str, qstate: dict, progress_cb=None):
    if n_runs <= 0 or not qstate:
        return np.zeros((0, max_steps), dtype=DTYPE), np.zeros((0, max_steps), dtype=DTYPE)
    n = max(1, N_WORKERS)
    base = n_runs // n
    rem = n_runs % n
    sizes = [base + (1 if i < rem else 0) for i in range(n)]
    sizes = [s for s in sizes if s > 0]
    basic_parts, q_parts, done = [], [], 0
    with ProcessPoolExecutor(max_workers=len(sizes), initializer=_init_worker_qstate_h2h, initargs=(qstate,)) as ex:
        futs = [ex.submit(simulate_chunk_h2h, s, max_steps, hard_csv, soft_csv, pairs_csv) for s in sizes]
        for f in as_completed(futs):
            b, q = f.result()
            basic_parts.append(b)
            q_parts.append(q)
            done += 1
            if progress_cb:
                progress_cb(done, len(sizes))
    runs_basic = np.concatenate(basic_parts, axis=0) if len(basic_parts) > 1 else basic_parts[0]
    runs_q = np.concatenate(q_parts, axis=0) if len(q_parts) > 1 else q_parts[0]
    return runs_basic, runs_q

# ===================== Évaluation (commune) =====================
def evaluate_strategy(runs_cum: np.ndarray, bankroll: int, bet_size: int, stop_loss: float, target_gain: float, mode: str):
    stop_win = int(bankroll * target_gain)
    stop_fail = bankroll * stop_loss
    wins = fails = neutrals = 0
    block = 4096
    n_runs = runs_cum.shape[0]
    for start in range(0, n_runs, block):
        end = min(start + block, n_runs)
        block_cum = runs_cum[start:end]
        banc_paths = bankroll + bet_size * block_cum
        win_mask = (banc_paths >= stop_win)
        fail_mask = (banc_paths <= stop_fail)
        win_idx = [np.argmax(r) if r.any() else None for r in win_mask]
        fail_idx = [np.argmax(r) if r.any() else None for r in fail_mask]
        for w, f in zip(win_idx, fail_idx):
            if w is None and f is None:
                neutrals += 1
            elif w is None:
                fails += 1
            elif f is None:
                wins += 1
            else:
                wins += 1 if w <= f else 0
                fails += 1 if f < w else 0
    n = n_runs
    return {
        "bankroll": bankroll,
        "bet": bet_size,
        "stop_loss": stop_loss,
        "mode": mode,
        "success_rate": wins / n,
        "fail_rate": fails / n,
        "neutral_rate": neutrals / n,
        "score": (wins - fails) / n,
    }

# ===================== UI =====================
st.title("🃏 Free Bet Blackjack — Stop-Loss : Basic vs Q-Table (TC)")

with st.sidebar:
    st.header("⚙️ Paramètres")
    bankroll = st.number_input("💰 Bankroll", min_value=1, value=DEFAULT_BANKROLL, step=1)
    n_sims = st.number_input("🎲 Nombre de runs", min_value=1000, step=1000, value=DEFAULT_N_SIMULATIONS)
    max_steps = st.number_input("🕒 Max mains par run", min_value=10, step=10, value=DEFAULT_MAX_STEPS)
    target_gain = st.number_input("🎯 Target gain (multiplier)", min_value=1.0, step=0.01, value=DEFAULT_TARGET_GAIN, format="%.2f")
    stop_losses = st.multiselect("🛑 Stop-loss (multiplicateurs)", DEFAULT_STOP_LOSSES, default=DEFAULT_STOP_LOSSES)
    st.caption(f"CPU détectés : {os.cpu_count()} • Workers : {N_WORKERS}")

col1, col2 = st.columns(2)
with col1:
    hard_csv = st.text_input("hard_totals.csv", os.path.join(DATA_DIR, "hard_totals.csv"))
    soft_csv = st.text_input("soft_totals.csv", os.path.join(DATA_DIR, "soft_totals.csv"))
with col2:
    pairs_csv = st.text_input("pairs.csv", os.path.join(DATA_DIR, "pairs.csv"))
    qtable_path = st.text_input(
        "qtable.pkl (pour Q-Table/TC)",
        os.path.abspath(os.path.join(HERE, "..", "freebet-rl", "qtable.pkl"))
    )
    results_path = st.text_input("Fichier résultats (.csv)", RESULTS_FILE)

run_btn = st.button("🚀 Lancer la simulation", type="primary")

if run_btn:
    qstate = load_qstate(qtable_path)

    if qstate:
        # === H2H : un seul passage pour Basic + Q-Table ===
        prog_h2h = st.progress(0, text="H2H — Génération des runs appariés (Basic & Q-Table)…")
        def _cb_h2h(done, total):
            prog_h2h.progress(int(done * 100 / total), text=f"H2H — Génération… {done}/{total} chunks")
        runs_basic, runs_q = simulate_runs_parallel_h2h(int(n_sims), int(max_steps), hard_csv, soft_csv, pairs_csv, qstate, progress_cb=_cb_h2h)
        prog_h2h.progress(100, text="H2H — Génération terminée ✅")
        runs_cum_basic = np.cumsum(runs_basic, axis=1, dtype=DTYPE)
        runs_cum_q = np.cumsum(runs_q, axis=1, dtype=DTYPE)

        # combos communs
        combos = []
        for bet in range(1, min(5, bankroll) + 1):
            for sl in stop_losses:
                combos.append(("flat", bet, float(sl)))
        for pct in range(1, 11):
            bet_val = max(1, int(bankroll * (pct / 100)))
            for sl in stop_losses:
                combos.append(("percent", bet_val, float(sl)))

        # Évaluations
        eval_prog_b = st.progress(0, text="BASIC — Évaluation…")
        results_basic = []
        for i, (mode, bet, sl) in enumerate(combos, start=1):
            res = evaluate_strategy(runs_cum_basic, bankroll, bet, sl, target_gain, f"basic-{mode}")
            results_basic.append(res)
            eval_prog_b.progress(int(i * 100 / len(combos)), text=f"BASIC — Évaluation… {i}/{len(combos)}")
        eval_prog_b.progress(100, text="BASIC — Évaluation terminée ✅")
        df_basic = pd.DataFrame(results_basic)

        eval_prog_q = st.progress(0, text="Q-TABLE — Évaluation…")
        results_q = []
        for i, (mode, bet, sl) in enumerate(combos, start=1):
            res = evaluate_strategy(runs_cum_q, bankroll, bet, sl, target_gain, f"qtable-{mode}")
            results_q.append(res)
            eval_prog_q.progress(int(i * 100 / len(combos)), text=f"Q-TABLE — Évaluation… {i}/{len(combos)}")
        eval_prog_q.progress(100, text="Q-TABLE — Évaluation terminée ✅")
        df_q = pd.DataFrame(results_q)

        st.caption("Mode **Head-to-Head** activé : mêmes seeds de sabot par run pour Basic & Q-Table (variance réduite).")

    else:
        # === Fallback : Basic seul (✅ correction : plus d’écrasement de fonction) ===
        prog_basic = st.progress(0, text="BASIC — Génération des runs…")
        def _cb_basic(done, total):
            prog_basic.progress(int(done * 100 / total), text=f"BASIC — Génération… {done}/{total} chunks")
        runs_basic = simulate_runs_parallel_basic(int(n_sims), int(max_steps), hard_csv, soft_csv, pairs_csv, progress_cb=_cb_basic)
        prog_basic.progress(100, text="BASIC — Génération terminée ✅")
        runs_cum_basic = np.cumsum(runs_basic, axis=1, dtype=DTYPE)

        combos = []
        for bet in range(1, min(5, bankroll) + 1):
            for sl in stop_losses:
                combos.append(("flat", bet, float(sl)))
        for pct in range(1, 11):
            bet_val = max(1, int(bankroll * (pct / 100)))
            for sl in stop_losses:
                combos.append(("percent", bet_val, float(sl)))

        eval_prog = st.progress(0, text="BASIC — Évaluation…")
        results_basic = []
        for i, (mode, bet, sl) in enumerate(combos, start=1):
            res = evaluate_strategy(runs_cum_basic, bankroll, bet, sl, target_gain, f"basic-{mode}")
            results_basic.append(res)
            eval_prog.progress(int(i * 100 / len(combos)), text=f"BASIC — Évaluation… {i}/{len(combos)}")
        eval_prog.progress(100, text="BASIC — Évaluation terminée ✅")
        df_basic = pd.DataFrame(results_basic)
        df_q = pd.DataFrame(columns=["bankroll","bet","stop_loss","mode","success_rate","fail_rate","neutral_rate","score"])

    # === AFFICHAGE ===
    st.subheader("🏆 Résultats — Basic Strategy (mises fixes)")
    top_flat_b = df_basic[df_basic["mode"].str.contains("basic-flat")].nlargest(5, "score")
    top_pct_b  = df_basic[df_basic["mode"].str.contains("basic-percent")].nlargest(5, "score")
    st.markdown("**Top Flat (Basic)**")
    st.dataframe(top_flat_b, use_container_width=True)
    st.markdown("**Top Percent (Basic)**")
    st.dataframe(top_pct_b, use_container_width=True)

    if 'df_q' in locals() and not df_q.empty:
        st.subheader("🤖 Résultats — Q-Table (TC) (mises fixes)")
        top_flat_q = df_q[df_q["mode"].str.contains("qtable-flat")].nlargest(5, "score")
        top_pct_q  = df_q[df_q["mode"].str.contains("qtable-percent")].nlargest(5, "score")
        st.markdown("**Top Flat (Q-Table)**")
        st.dataframe(top_flat_q, use_container_width=True)
        st.markdown("**Top Percent (Q-Table)**")
        st.dataframe(top_pct_q, use_container_width=True)

    with st.expander("Voir tous les résultats (Basic & Q-Table)"):
        st.dataframe(
            (pd.concat([df_basic, df_q], ignore_index=True) if 'df_q' in locals() else df_basic)
            .sort_values(["mode", "score"], ascending=[True, False]),
            use_container_width=True
        )

    # Export CSV
    try:
        export_df = pd.concat([df_basic, df_q], ignore_index=True) if 'df_q' in locals() else df_basic
        export_df.to_csv(results_path, index=False)
        st.success(f"Résultats sauvegardés : `{results_path}`")
    except Exception as e:
        st.warning(f"Impossible d’écrire le CSV : {e}")

    st.caption("Politiques comparées : **Basic (tables CSV)** vs **Q-Table guidée par le True Count**. "
               "Règles Free Bet : splits/doubles gratuits, dealer 22 = push, BJ 3:2, split As = 1 carte/hand, pas de re-split.")
