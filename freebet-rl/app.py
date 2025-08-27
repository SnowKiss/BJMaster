import os
import pickle
import streamlit as st
from collections import Counter, defaultdict
import numpy as np

from freebet.env import FreeBetEnv
from freebet.rl.qtable import QTable
from freebet.ui.tables import (
    build_table_hard,
    build_table_soft,
    build_table_pairs,
    style_actions,
)

# ------------------------------ Page config ------------------------------
st.set_page_config(page_title="Free Bet Blackjack RL", layout="wide")

# ------------------------------ Q-Table save/load ------------------------------
SAVE_PATH = "qtable.pkl"

def save_qtable(qtab, path=SAVE_PATH):
    data = {
        "qtable": dict(qtab.q),
        "episodes": st.session_state.episodes,
        "returns_sum": st.session_state.returns_sum,
        "stats": dict(st.session_state.stats),
        "sessions": st.session_state.env.shoe.sessions_played,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_qtable(qtab, path=SAVE_PATH):
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        # restaurer Q-table
        qtab.q = defaultdict(lambda: np.zeros(4), data["qtable"])
        # restaurer métriques
        st.session_state.episodes = data.get("episodes", 0)
        st.session_state.returns_sum = data.get("returns_sum", 0.0)
        st.session_state.stats = Counter(data.get("stats", {}))
        st.session_state.env.shoe.sessions_played = data.get("sessions", 0)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        pass
    

# ------------------------------ Init Session State ------------------------------
if "env" not in st.session_state:
    st.session_state.env = FreeBetEnv(num_decks=8, penetration=0.5, dealer_hits_soft_17=True)

if "qtab" not in st.session_state:
    st.session_state.qtab = QTable()
    # auto-load si fichier existe
    load_qtable(st.session_state.qtab)

if "running" not in st.session_state:
    st.session_state.running = False

if "episodes" not in st.session_state:
    st.session_state.episodes = 0

if "stats" not in st.session_state:
    st.session_state.stats = Counter()

if "returns_sum" not in st.session_state:
    st.session_state.returns_sum = 0.0

if "settings" not in st.session_state:
    st.session_state.settings = {
        "alpha": 0.05,
        "epsilon": 0.2,
        "gamma": 1.0,
        "episodes_per_tick": 500,
        "dealer_hits_soft_17": True,
        "num_decks": 8,
        "penetration": 0.5,
    }

# ------------------------------ Controls ------------------------------
c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 2, 2])

with c1:
    if st.button("▶️ Start / Resume", type="primary"):
        st.session_state.running = True

with c2:
    if st.button("⏸️ Pause"):
        st.session_state.running = False

with c3:
    if st.button("⏭️ Train one tick"):
        env = st.session_state.env
        qtab = st.session_state.qtab
        eps = st.session_state.settings['epsilon']
        alpha = st.session_state.settings['alpha']
        for _ in range(int(st.session_state.settings['episodes_per_tick'])):
            trans, rewards, outcomes = env.play_round(qtab.q, eps)
            G = sum(rewards)
            if trans:
                qtab.update_episode(trans, G, alpha)
            st.session_state.episodes += 1
            st.session_state.returns_sum += G
            st.session_state.stats.update(outcomes)

with c4:
    if st.button("💾 Save Q-Table"):
        save_qtable(st.session_state.qtab)
        st.success("Q-Table sauvegardée !")

with c5:
    if st.button("📂 Load Q-Table"):
        load_qtable(st.session_state.qtab)
        st.success("Q-Table chargée !")

# ------------------------------ Training loop (bloque UI jusqu'à Pause) ------------------------------
if st.session_state.running:
    env = st.session_state.env
    qtab = st.session_state.qtab
    eps = st.session_state.settings['epsilon']
    alpha = st.session_state.settings['alpha']

    while st.session_state.running:
        for _ in range(int(st.session_state.settings['episodes_per_tick'])):
            trans, rewards, outcomes = env.play_round(qtab.q, eps)
            G = sum(rewards)
            if trans:
                qtab.update_episode(trans, G, alpha)
            st.session_state.episodes += 1
            st.session_state.returns_sum += G
            st.session_state.stats.update(outcomes)
        # ⚠️ pas de st.rerun() → l’UI ne bouge pas tant que Pause n’est pas cliqué

# ------------------------------ Metrics ------------------------------
st.subheader("📈 Training Metrics")
colA, colB, colC, colD, colE, colF = st.columns([2, 1, 1, 1, 2, 2])

with colA:
    st.metric("Episodes", f"{st.session_state.episodes:,}")

with colB:
    total_outcomes = sum(st.session_state.stats.values()) or 1
    wins = st.session_state.stats.get('win', 0)
    st.metric("Win %", f"{100 * wins / total_outcomes:.1f} %")

with colC:
    pushes = st.session_state.stats.get('push', 0)
    st.metric("Push %", f"{100 * pushes / total_outcomes:.1f} %")

with colD:
    losses = st.session_state.stats.get('loss', 0)
    st.metric("Loss %", f"{100 * losses / total_outcomes:.1f} %")

with colE:
    avg_return = st.session_state.returns_sum / max(1, st.session_state.episodes)
    st.metric("EV per round", f"{avg_return:.4f}")

with colF:
    st.metric("Sessions (sabots joués)", st.session_state.env.shoe.sessions_played)

st.divider()

# ------------------------------ Strategy Tables ------------------------------
st.subheader("🧠 Learned Best Actions (style basic strategy)")

show_percentages = st.checkbox("Afficher les pourcentages D/H/S/P", value=False)

htab1, htab2, htab3 = st.tabs(["Hard totals", "Soft totals", "Pairs"])

with htab1:
    df_hard = build_table_hard(st.session_state.qtab, st.session_state.env, show_percentages)
    st.dataframe(style_actions(df_hard), use_container_width=True, hide_index=True)

with htab2:
    df_soft = build_table_soft(st.session_state.qtab, st.session_state.env, show_percentages)
    st.dataframe(style_actions(df_soft), use_container_width=True, hide_index=True)

with htab3:
    df_pairs = build_table_pairs(st.session_state.qtab, st.session_state.env, show_percentages)
    st.dataframe(style_actions(df_pairs), use_container_width=True, hide_index=True)

st.caption(
    """
Rappels de règles implémentées :
• Push 22 du croupier (hors blackjack joueur).\
• Double gratuit 9–11 : win = +2, lose = −1, push = 0. Doubles non gratuits : win +2 / lose −2.\
• Splits gratuits sur 2–9 et As : la main "free" perd 0 / gagne +1. Split As = une carte, stand.\
• Un seul split par main (simplification).\
• BJ paie 3:2 et est résolu avant un éventuel push 22.
"""
)
