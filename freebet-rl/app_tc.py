import os
import pickle
import streamlit as st
import numpy as np
from collections import defaultdict

from freebet.env import FreeBetEnv
from freebet.rl.qtable import QTable
from freebet.ui.tables import (
    build_table_hard,
    build_table_soft,
    build_table_pairs,
    style_actions,
)

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Live Strategy by True Count", layout="wide")
SAVE_PATH = "qtable.pkl"
TOTAL_CARDS = 8 * 52  # 8 jeux

# ---------------- HELPER: LOAD QTABLE ----------------
def load_qtable(qtab, path=SAVE_PATH):
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            st.warning("⚠️ Aucun fichier Q-Table trouvé, affichage par défaut.")
            return
        with open(path, "rb") as f:
            data = pickle.load(f)

        qtab.q_by_tc = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))
        for tc, qdict in data.get("q_by_tc", {}).items():
            qtab.q_by_tc[tc] = defaultdict(lambda: np.zeros(4), qdict)

        qtab.visits_by_tc = defaultdict(lambda: defaultdict(lambda: np.zeros(4, dtype=int)))
        for tc, vdict in data.get("visits_by_tc", {}).items():
            qtab.visits_by_tc[tc] = defaultdict(lambda: np.zeros(4, dtype=int), vdict)

        qtab.episodes = data.get("episodes", 0)
        qtab.epsilon = data.get("epsilon", qtab.epsilon)

    except Exception as e:
        st.error(f"❌ Erreur lors du chargement Q-Table: {e}")

# ---------------- INIT SESSION STATE ----------------
if "rc" not in st.session_state:
    st.session_state.rc = 0   # Running Count
if "cards_left" not in st.session_state:
    st.session_state.cards_left = TOTAL_CARDS
if "env" not in st.session_state:
    st.session_state.env = FreeBetEnv()
if "qtab" not in st.session_state:
    st.session_state.qtab = QTable()
    load_qtable(st.session_state.qtab)

# ---------------- LAYOUT ----------------
col_left, col_right = st.columns([1, 5])  # 1/3 gauche, 2/3 droite

with col_left:
    st.title("🎴 True Count Live")

    st.button("+1 (2–6)", use_container_width=True, on_click=lambda: (
        st.session_state.__setitem__("rc", st.session_state.rc + 1),
        st.session_state.__setitem__("cards_left", max(1, st.session_state.cards_left - 1))
    ))

    st.button("0 (7–9)", use_container_width=True, on_click=lambda: (
        st.session_state.__setitem__("cards_left", max(1, st.session_state.cards_left - 1))
    ))

    st.button("-1 (10–A)", use_container_width=True, on_click=lambda: (
        st.session_state.__setitem__("rc", st.session_state.rc - 1),
        st.session_state.__setitem__("cards_left", max(1, st.session_state.cards_left - 1))
    ))

    st.button("♻️ Reshuffle", use_container_width=True, on_click=lambda: (
        st.session_state.__setitem__("rc", 0),
        st.session_state.__setitem__("cards_left", TOTAL_CARDS)
    ))

    # Metrics
    decks_equiv = st.session_state.cards_left / 52
    tc = int(round(st.session_state.rc / max(1, decks_equiv)))

    st.metric("Running Count", st.session_state.rc)
    st.metric("Cartes restantes", st.session_state.cards_left)
    st.metric("True Count (TC)", tc)

    show_ev = st.checkbox("Afficher les EV", value=False)

with col_right:
    st.subheader("📊 Tables de stratégie")

    # --- Hard totals ---
    st.markdown("### Hard Totals")
    df_hard = build_table_hard(st.session_state.qtab, st.session_state.env, tc, show_ev)
    st.dataframe(
        style_actions(df_hard),
        use_container_width=True,
        hide_index=True,
        height=250  # 👈 compact
    )

    # --- Soft totals ---
    st.markdown("### Soft Totals")
    df_soft = build_table_soft(st.session_state.qtab, st.session_state.env, tc, show_ev)
    st.dataframe(
        style_actions(df_soft),
        use_container_width=True,
        hide_index=True,
        height=250
    )

    # --- Pairs ---
    st.markdown("### Pairs")
    df_pairs = build_table_pairs(st.session_state.qtab, st.session_state.env, tc, show_ev)
    st.dataframe(
        style_actions(df_pairs),
        use_container_width=True,
        hide_index=True,
        height=250
    )
