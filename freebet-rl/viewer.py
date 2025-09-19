# viewer.py
import os
import io
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st

from freebet.env import FreeBetEnv
from freebet.rl.qtable import QTable
from freebet.ui.tables import (
    build_table_hard,
    build_table_soft,
    build_table_pairs,
    style_actions,
)

# ------------------------------ Page config ------------------------------
st.set_page_config(page_title="Free Bet Blackjack — Viewer", layout="wide")

DEFAULT_PATH = "qtable.pkl"

# ------------------------------ Loaders ----------------------------------
def _rebuild_qtable_from_dict(data: dict) -> QTable:
    qtab = QTable()
    qtab.q_by_tc = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))
    for tc, qdict in data.get("q_by_tc", {}).items():
        qtab.q_by_tc[tc] = defaultdict(lambda: np.zeros(4), qdict)
    qtab.visits_by_tc = defaultdict(lambda: defaultdict(lambda: np.zeros(4, dtype=int)))
    for tc, vdict in data.get("visits_by_tc", {}).items():
        qtab.visits_by_tc[tc] = defaultdict(lambda: np.zeros(4, dtype=int), vdict)
    qtab.episodes = int(data.get("episodes", 0))
    qtab.epsilon = float(data.get("epsilon", 0.2))
    return qtab

def load_qtable_from_path(path: str) -> QTable | None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return _rebuild_qtable_from_dict(data)
    except Exception as e:
        st.error(f"Erreur de chargement '{path}': {e}")
        return None

def load_qtable_from_filelike(file) -> QTable | None:
    try:
        data = pickle.load(file)
        return _rebuild_qtable_from_dict(data)
    except Exception as e:
        st.error(f"Erreur de chargement du fichier uploadé: {e}")
        return None

def tables_to_csv(df_hard: pd.DataFrame, df_soft: pd.DataFrame, df_pairs: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    buf.write("# HARD\n")
    df_hard.to_csv(buf, index=False)
    buf.write("\n# SOFT\n")
    df_soft.to_csv(buf, index=False)
    buf.write("\n# PAIRS\n")
    df_pairs.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

# ------------------------------ Session init -----------------------------
if "qtab" not in st.session_state:
    q = load_qtable_from_path(DEFAULT_PATH)
    st.session_state.qtab = q if q is not None else QTable()

if "env_opts" not in st.session_state:
    st.session_state.env_opts = {
        "num_decks": 8,
        "penetration": 0.5,
        "dealer_hits_soft_17": False,
    }

# ------------------------------ Sidebar ----------------------------------
st.sidebar.header("📂 Source des données")

file_up = st.sidebar.file_uploader("Importer un qtable.pkl", type=["pkl"])
path = st.sidebar.text_input("…ou chemin du fichier", value=DEFAULT_PATH)
col_sb1, col_sb2 = st.sidebar.columns([1, 1])
with col_sb1:
    if st.button("Charger", key="btn_load_path"):
        q = load_qtable_from_path(path)
        if q is not None:
            st.session_state.qtab = q
            st.success(f"Chargé depuis {path} ({q.episodes:,} épisodes)")
with col_sb2:
    if file_up is not None and st.button("Charger l'upload", key="btn_load_upload"):
        q = load_qtable_from_filelike(file_up)
        if q is not None:
            st.session_state.qtab = q
            st.success(f"Chargé depuis upload ({q.episodes:,} épisodes)")

st.sidebar.header("⚙️ Paramètres d’affichage")
tc_choice = st.sidebar.slider("True Count (TC)", -5, 5, 0)
show_pct = st.sidebar.checkbox("Afficher les % D/H/S/P", value=False)
s17 = st.sidebar.radio("Règle du croupier", ["S17 (stand on soft 17)", "H17 (hits soft 17)"], index=0)
dealer_hits_soft_17 = (s17.startswith("H17"))

# met à jour l'env si nécessaire
if st.session_state.env_opts["dealer_hits_soft_17"] != dealer_hits_soft_17:
    st.session_state.env_opts["dealer_hits_soft_17"] = dealer_hits_soft_17

# ------------------------------ Header -----------------------------------
left, right = st.columns([3, 2])
with left:
    st.title("🖨️ Strategy Viewer (RL)")
    st.caption("Affiche les tables Hard / Soft / Pairs apprises depuis votre Q-table.")
with right:
    st.metric("Episodes appris", f"{st.session_state.qtab.episodes:,}")
    st.metric("Epsilon", f"{getattr(st.session_state.qtab, 'epsilon', 0.0):.2f}")

st.divider()

# ------------------------------ Build env + tables -----------------------
env = FreeBetEnv(
    num_decks=st.session_state.env_opts["num_decks"],
    penetration=st.session_state.env_opts["penetration"],
    dealer_hits_soft_17=st.session_state.env_opts["dealer_hits_soft_17"],
)

TABLE_HEIGHT = 680

tab1, tab2, tab3 = st.tabs(["Hard totals", "Soft totals", "Pairs"])

with tab1:
    df_hard = build_table_hard(st.session_state.qtab, env, tc_choice, show_pct)
    st.dataframe(style_actions(df_hard), width="stretch", height=TABLE_HEIGHT, hide_index=True)

with tab2:
    df_soft = build_table_soft(st.session_state.qtab, env, tc_choice, show_pct)
    st.dataframe(style_actions(df_soft), width="stretch", height=TABLE_HEIGHT, hide_index=True)

with tab3:
    df_pairs = build_table_pairs(st.session_state.qtab, env, tc_choice, show_pct)
    st.dataframe(style_actions(df_pairs), width="stretch", height=TABLE_HEIGHT, hide_index=True)

# ------------------------------ Exports ----------------------------------
st.subheader("📤 Export")
csv_bytes = tables_to_csv(df_hard, df_soft, df_pairs)
st.download_button(
    "Télécharger CSV (3 tables)",
    data=csv_bytes,
    file_name=f"strategy_TC{tc_choice}_{'H17' if dealer_hits_soft_17 else 'S17'}.csv",
    mime="text/csv",
    key="dl_csv",
)

st.caption(
    "Rappels des règles : "
    + ("Dealer **tire** sur soft 17 (H17)." if dealer_hits_soft_17 else "Dealer **s'arrête** sur 17 (S17).")
    + " • Blackjack paie 3:2 • Pas de resplit/redouble après split • 1 seul split • Charlie 6 cartes."
)
