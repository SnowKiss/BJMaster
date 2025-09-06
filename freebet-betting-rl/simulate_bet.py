import os
import sys
import threading
import pickle
import random
import time
import traceback
import numpy as np
import pandas as pd
import streamlit as st
from collections import defaultdict

# ========= Chemins =========
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "freebet-rl"))
sys.path.append(BASE_DIR)

from freebet.env import FreeBetEnv
from freebet.cards import hand_value, is_blackjack

# ========= Constantes =========
HERE = os.path.dirname(__file__)
DTYPE = np.float32
ACTIONS = ['H', 'S', 'D', 'P']
ACT2IDX = {a: i for i, a in enumerate(ACTIONS)}

QSTATE_FILE = os.path.abspath(os.path.join(HERE, "..", "freebet-rl", "qtable.pkl"))

# ========= Contexte Streamlit pour threads =========
try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx as _add_ctx
except Exception:
    def _add_ctx(_t):
        pass

# ========= Utils comptage =========
def hilo_value(card: int) -> int:
    if 2 <= card <= 6:  return 1
    if 7 <= card <= 9:  return 0
    return -1

def tc_from_rc_nearest(rc: int, cards_left: int) -> int:
    decks_left = max(1.0, cards_left / 52.0)
    return int(round(rc / decks_left))

# ========= Helpers Free Bet =========
def is_ace(card: int) -> bool:
    return card == 11

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

# ========= Q-table actions =========
@st.cache_resource
def load_qstate_actions(path: str) -> dict:
    if not path or not os.path.exists(path):
        st.error(f"❌ Q-Table actions introuvable : {path}")
        return {}
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

def choose_action_qtable(state, avail: list[str], tc: int, qstate: dict) -> str:
    inner = qstate.get(tc)
    qvals = inner.get(state) if inner is not None else None
    if qvals is None:
        return 'S' if 'S' in avail else avail[0]
    mask = np.full(4, -1e9, dtype=np.float32)
    for a in avail:
        mask[ACT2IDX[a]] = 0.0
    idx = int(np.argmax(qvals + mask))
    act = ACTIONS[idx]
    return act if act in avail else ('S' if 'S' in avail else avail[0])

# ========= Simulation d’une main =========
def simulate_hand(game: FreeBetEnv, p, d, tc: int, bet: int, qstate_actions: dict):
    dealer_up = d[1]
    dealer_bj = is_blackjack(d)
    player_bj = is_blackjack(p)

    if dealer_bj and player_bj: return 0.0, []
    if dealer_bj and not player_bj: return -bet, []
    if player_bj and not dealer_bj: return 1.5 * bet, []

    hands = [{
        "cards": p[:],
        "already_split": False,
        "split_aces_lock": False,
        "free_extra": False,
        "doubled_free": False,
        "is_initial": True,
        "first_action": True,
    }]
    used = []
    i = 0
    while i < len(hands):
        h = hands[i]
        cards = h["cards"]
        while True:
            if h["split_aces_lock"]: break
            avail = available_actions_freebet(cards, h["first_action"], h["already_split"], h["split_aces_lock"])
            state = game.state_key(cards, dealer_up, h["first_action"])
            action = choose_action_qtable(state, avail, tc, qstate_actions)
            if action == "H":
                c = game.shoe.draw(); used.append(c)
                cards.append(c); h["first_action"] = False
                if hand_value(cards)[0] > 21: break
            elif action == "S":
                break
            elif action == "D":
                c = game.shoe.draw(); used.append(c)
                cards.append(c); h["doubled_free"] = True; h["first_action"] = False
                break
            elif action == "P":
                v = cards[0]
                c1 = game.shoe.draw(); used.append(c1); cards[:] = [v, c1]
                c2 = game.shoe.draw(); used.append(c2)
                newh = {"cards": [v, c2], "already_split": True, "split_aces_lock": False,
                        "free_extra": True, "doubled_free": False, "is_initial": False, "first_action": True}
                hands.append(newh)
                if is_ace(v): h["split_aces_lock"] = True; newh["split_aces_lock"] = True; break
                h["already_split"] = True; h["first_action"] = False
            else: break
        i += 1

    if any(hand_value(h["cards"])[0] <= 21 for h in hands):
        pre = len(d); dealer_cards = game.dealer_play(d)
        if len(dealer_cards) > pre: used.extend(dealer_cards[pre:])
    else: dealer_cards = d

    dealer_total, _ = hand_value(dealer_cards)

    delta = 0.0
    for h in hands:
        pt, _ = hand_value(h["cards"])
        if h["is_initial"] and len(h["cards"]) == 2 and pt == 21: delta += 1.5 * bet; continue
        if pt > 21: delta -= bet; continue
        if dealer_total == 22: continue
        if dealer_total > 21: delta += 2 * bet if h["doubled_free"] else bet; continue
        if pt > dealer_total: delta += 2 * bet if h["doubled_free"] else bet
        elif pt < dealer_total: delta -= bet
    return delta, used

# ========= Training step (mise fixée à 1) =========
def training_step(qstate_actions: dict):
    game = st.session_state.game
    prev_left = st.session_state.cards_left
    current_left = game.shoe.cards_left()
    if current_left > prev_left:
        st.session_state.rc = 0
        st.session_state.shoe_idx += 1
        st.session_state.hand_in_shoe = 0
    st.session_state.cards_left = current_left

    p, d = game.initial_deal()
    st.session_state.rc += sum(hilo_value(c) for c in (p + d))
    st.session_state.hand_in_shoe += 1
    st.session_state.total_hands += 1
    tc = tc_from_rc_nearest(st.session_state.rc, game.shoe.cards_left())

    bet = 1  # ✅ mise fixée
    gain, used = simulate_hand(game, p, d, tc, bet, qstate_actions)
    if used: st.session_state.rc += sum(hilo_value(c) for c in used)

    st.session_state.rewards_sum += gain
    st.session_state.rewards_count += 1
    if "ev_by_tc" not in st.session_state: st.session_state.ev_by_tc = defaultdict(list)
    st.session_state.ev_by_tc[tc].append(gain)

# ========= Thread training =========
def trainer_loop(qstate_actions: dict, steps_per_batch: int, stop_flag_key: str):
    while st.session_state.get(stop_flag_key, False):
        for _ in range(steps_per_batch):
            training_step(qstate_actions)
        time.sleep(0)

# ========= UI =========
st.title("🤖 Free Bet Blackjack — Analyse EV (mise = 1)")

# init session state
defaults = {
    "running": False, "trainer_thread": None, "game": FreeBetEnv(num_decks=8, penetration=0.5, dealer_hits_soft_17=True),
    "rc": 0, "cards_left": None, "shoe_idx": 1, "hand_in_shoe": 0, "total_hands": 0,
    "rewards_sum": 0.0, "rewards_count": 0, "ev_by_tc": defaultdict(list)
}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
if st.session_state.cards_left is None:
    st.session_state.cards_left = st.session_state.game.shoe.cards_left()

steps_per_batch = st.sidebar.slider("Steps par batch", 100, 20000, 5000, 100)
qstate_actions = load_qstate_actions(QSTATE_FILE)

colA, colB, colC = st.columns(3)
start_btn = colA.button("▶️ Start")
pause_btn = colB.button("⏸️ Pause")
reset_btn = colC.button("🧹 Reset")

if start_btn and not st.session_state.running:
    st.session_state.running = True
    st.session_state["run_flag"] = True
    t = threading.Thread(target=trainer_loop, args=(qstate_actions, steps_per_batch, "run_flag"), daemon=True)
    _add_ctx(t)
    st.session_state.trainer_thread = t; t.start()

if pause_btn and st.session_state.running:
    st.session_state["run_flag"] = False
    t = st.session_state.trainer_thread
    if t and t.is_alive(): t.join()
    st.session_state.running = False
    st.session_state.trainer_thread = None

if reset_btn and not st.session_state.running:
    st.session_state.rc = 0; st.session_state.shoe_idx = 1
    st.session_state.hand_in_shoe = 0; st.session_state.total_hands = 0
    st.session_state.rewards_sum = 0.0; st.session_state.rewards_count = 0
    st.session_state.ev_by_tc = defaultdict(list)
    st.session_state.game = FreeBetEnv(num_decks=8, penetration=0.5, dealer_hits_soft_17=True)
    st.session_state.cards_left = st.session_state.game.shoe.cards_left()
    st.info("Session reset.")

# affichages
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("État", "🟢 Running" if st.session_state.running else "⏸ Paused")
col2.metric("Sabot #", st.session_state.shoe_idx)
col3.metric("Main dans sabot", st.session_state.hand_in_shoe)
col4.metric("Total mains", st.session_state.total_hands)
ev_obs = (st.session_state.rewards_sum / max(1, st.session_state.rewards_count))
col5.metric("EV observée / main", f"{ev_obs:.4f}")

# EV par TC
rows = []
for tc in range(-5, 6):
    vals = st.session_state.ev_by_tc.get(tc, [])
    mean_ev = np.mean(vals) if vals else 0.0
    rows.append({"TC": tc, "Mean EV": round(float(mean_ev), 4)})

st.subheader("📊 EV observée par TC (mise=1)")
st.dataframe(pd.DataFrame(rows), use_container_width=True)
