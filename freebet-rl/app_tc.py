import os
import pickle
import streamlit as st
import numpy as np
import pandas as pd
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
DEFAULT_DECKS = 8
SAVE_PATH = "qtable.pkl"

RANKS = ["2","3","4","5","6","7","8","9","10","J","Q","K","A"]
KEYS  = ["2","3","4","5","6","7","8","9","T","J","Q","K","A"]  # interne: "10" -> "T"
HILO = { "2":+1,"3":+1,"4":+1,"5":+1,"6":+1,"7":0,"8":0,"9":0,"T":-1,"J":-1,"Q":-1,"K":-1,"A":-1 }

# ---------------- HELPERS ----------------
def label_to_key(label: str) -> str:
    return "T" if label == "10" else label

def init_state():
    if "decks" not in st.session_state:
        st.session_state.decks = DEFAULT_DECKS
    if "rc" not in st.session_state:
        st.session_state.rc = 0
    if "env" not in st.session_state:
        st.session_state.env = FreeBetEnv()
    if "qtab" not in st.session_state:
        st.session_state.qtab = QTable()
        load_qtable(st.session_state.qtab, SAVE_PATH)
    if "shoe_counts" not in st.session_state:
        st.session_state.shoe_counts = fresh_shoe_counts(st.session_state.decks)
    if "initial_per_rank" not in st.session_state:
        st.session_state.initial_per_rank = 4 * st.session_state.decks
    if "history" not in st.session_state:
        st.session_state.history = []  # {"rank":"T","qty":+/-n,"rc_delta":x}
    if "qty_click" not in st.session_state:
        st.session_state.qty_click = 1  # quantité ajoutée par clic "+"

def fresh_shoe_counts(decks: int) -> dict:
    per_rank = 4 * decks
    return {k: per_rank for k in KEYS}

def cards_left() -> int:
    return int(sum(st.session_state.shoe_counts.values()))

def cards_seen() -> int:
    total = 52 * st.session_state.decks
    return total - cards_left()

def compute_tc(rc: int) -> int:
    decks_equiv = max(cards_left() / 52.0, 1.0)
    return int(round(rc / decks_equiv))

def can_decrease(rank_key: str, qty: int = 1) -> bool:
    return st.session_state.shoe_counts[rank_key] >= qty

def can_increase(rank_key: str, qty: int = 1) -> bool:
    return st.session_state.shoe_counts[rank_key] + qty <= st.session_state.initial_per_rank

def add_card_seen(rank_key: str, qty: int = 1):
    qty = int(qty)
    if qty <= 0:
        return
    if not can_decrease(rank_key, qty):
        st.warning(f"Impossible de retirer {qty} carte(s) {rank_key}, plus assez dans le sabot.")
        return
    st.session_state.shoe_counts[rank_key] -= qty
    delta = HILO[rank_key] * qty
    st.session_state.rc += delta
    st.session_state.history.append({"rank": rank_key, "qty": qty, "rc_delta": delta})

def add_card_back(rank_key: str, qty: int = 1):
    qty = int(qty)
    if qty <= 0:
        return
    if not can_increase(rank_key, qty):
        st.warning(f"Impossible d'ajouter {qty} carte(s) {rank_key} (on dépasserait le total initial).")
        return
    st.session_state.shoe_counts[rank_key] += qty
    delta = -HILO[rank_key] * qty
    st.session_state.rc += delta
    st.session_state.history.append({"rank": rank_key, "qty": -qty, "rc_delta": delta})

def undo_last():
    if not st.session_state.history:
        return
    ev = st.session_state.history.pop()
    rank_key, qty, rc_delta = ev["rank"], ev["qty"], ev["rc_delta"]
    if qty > 0:
        st.session_state.shoe_counts[rank_key] += qty
        st.session_state.rc -= rc_delta
    else:
        st.session_state.shoe_counts[rank_key] -= (-qty)
        st.session_state.rc -= rc_delta

def reshuffle():
    st.session_state.shoe_counts = fresh_shoe_counts(st.session_state.decks)
    st.session_state.initial_per_rank = 4 * st.session_state.decks
    st.session_state.rc = 0
    st.session_state.history.clear()

def load_qtable(qtab: QTable, path: str):
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
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

def probs_dataframe(group_faces: bool) -> pd.DataFrame:
    total = cards_left()
    if total <= 0:
        return pd.DataFrame(columns=["Carte", "Restantes", "Probabilité"])

    counts = st.session_state.shoe_counts

    if group_faces:
        ten_val = counts["T"] + counts["J"] + counts["Q"] + counts["K"]
        rows = []
        for k in ["2","3","4","5","6","7","8","9"]:
            rows.append((" " + k, counts[k], counts[k] / total))
        rows.append((" 10/J/Q/K", ten_val, ten_val / total))
        rows.append((" A", counts["A"], counts["A"] / total))
    else:
        order = ["2","3","4","5","6","7","8","9","T","J","Q","K","A"]
        label_map = {**{str(n):str(n) for n in range(2,10)}, "T":"10"}
        rows = []
        for k in order:
            label = label_map.get(k, k)
            rows.append((f" {label}", counts[k], counts[k] / total))

    df = pd.DataFrame(rows, columns=["Carte", "Restantes", "Probabilité"])
    df["Probabilité (%)"] = (df["Probabilité"] * 100).round(2)
    df = df.drop(columns=["Probabilité"]).sort_values("Probabilité (%)", ascending=False)
    return df

def full_shoe_dataframe(group_faces: bool) -> pd.DataFrame:
    counts = st.session_state.shoe_counts
    left = max(cards_left(), 1)

    if group_faces:
        ten_val = counts["T"] + counts["J"] + counts["Q"] + counts["K"]
        rows = []
        for k in ["2","3","4","5","6","7","8","9"]:
            fallen = st.session_state.initial_per_rank - counts[k]
            rows.append((k, counts[k], fallen, counts[k]/left))
        fallen_10s = (4*st.session_state.decks*4) - ten_val
        rows.append(("10/J/Q/K", ten_val, fallen_10s, ten_val/left))
        fallen_a = st.session_state.initial_per_rank - counts["A"]
        rows.append(("A", counts["A"], fallen_a, counts["A"]/left))
    else:
        label_map = {**{str(n):str(n) for n in range(2,10)}, "T":"10"}
        order = ["2","3","4","5","6","7","8","9","T","J","Q","K","A"]
        rows = []
        for k in order:
            fallen = st.session_state.initial_per_rank - counts[k]
            rows.append((label_map.get(k,k), counts[k], fallen, counts[k]/left))

    df = pd.DataFrame(rows, columns=["Carte", "Restantes", "Tombées", "Probabilité"])
    df["Probabilité (%)"] = (df["Probabilité"] * 100).round(2)
    return df.drop(columns=["Probabilité"])

# ---- UI helper: compute dataframe height to avoid inner scroll ----
def df_height(df, row_px=34, header_px=38, pad_px=18, min_px=240, max_px=1200):
    n = getattr(df, "shape", (0,0))[0]
    h = header_px + n * row_px + pad_px
    return int(min(max(h, min_px), max_px))

# ---------------- INIT ----------------
init_state()

# ---------------- SIDEBAR (vertical, compact) ----------------
with st.sidebar:
    st.title("🎴 True Count Live")

    decks_new = st.number_input("Nombre de jeux", 1, 12, value=st.session_state.decks)
    if decks_new != st.session_state.decks:
        st.session_state.decks = int(decks_new)
        reshuffle()

    st.markdown("#### 🧮 Comptage fin")
    group_faces = st.checkbox("Grouper 10/J/Q/K", value=True)

    st.number_input(
        "Quantité par clic",
        min_value=1, max_value=20, step=1,
        value=st.session_state.qty_click,
        key="qty_click"
    )

    for label in RANKS:
        rank_key = label_to_key(label)
        c_btn, c_txt = st.columns([1,3])
        with c_btn:
            if st.button("+", key=f"plus_{label}", width="stretch"):
                add_card_seen(rank_key, st.session_state.qty_click)
        with c_txt:
            st.caption(f"{label} — restantes: {st.session_state.shoe_counts[rank_key]}")

    with st.expander("Saisie rapide (plusieurs cartes / correction)"):
        colq1, colq2, colq3 = st.columns([2,2,1])
        with colq1:
            sel = st.selectbox("Carte", RANKS, index=8)
        with colq2:
            qty = st.number_input("Quantité", min_value=1, max_value=50, value=1, step=1, key="bulk_qty")
        with colq3:
            if st.button("Ajouter", key="bulk_add", width="stretch"):
                add_card_seen(label_to_key(sel), qty)
            if st.button("Retirer", key="bulk_remove", width="stretch"):
                add_card_back(label_to_key(sel), qty)

    if st.button("↩️ Annuler la dernière action", key="undo", width="stretch"):
        undo_last()
    if st.button("♻️ Reshuffle", key="reshuffle", width="stretch"):
        reshuffle()

# ---------------- MAIN ----------------
total_cards = 52 * st.session_state.decks
pen = cards_seen() / total_cards if total_cards else 0.0
tc = compute_tc(st.session_state.rc)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Running Count", st.session_state.rc)
with c2:
    st.metric("Cartes restantes", cards_left())
with c3:
    st.metric("Sabot", f"{st.session_state.decks}×52 = {52*st.session_state.decks}")
with c4:
    st.metric("True Count (TC)", tc)

st.progress(min(max(pen, 0.0), 1.0), text=f"Pénétration du sabot : {pen:.0%}")

# ====== 👉 Tables de stratégie AVANT les probabilités (et sans scroll interne) ======
st.subheader("📊 Tables de stratégie")
show_ev = st.checkbox("Afficher les EV", value=False)

# Hard
st.markdown("### Hard Totals")
df_hard = build_table_hard(st.session_state.qtab, st.session_state.env, tc, show_ev)
st.dataframe(style_actions(df_hard), hide_index=True, height=df_height(df_hard), width="stretch")

# Soft
st.markdown("### Soft Totals")
df_soft = build_table_soft(st.session_state.qtab, st.session_state.env, tc, show_ev)
st.dataframe(style_actions(df_soft), hide_index=True, height=df_height(df_soft), width="stretch")

# Pairs
st.markdown("### Pairs")
df_pairs = build_table_pairs(st.session_state.qtab, st.session_state.env, tc, show_ev)
st.dataframe(style_actions(df_pairs), hide_index=True, height=df_height(df_pairs), width="stretch")

# ====== Probabilités après ======
top_df = probs_dataframe(group_faces)
st.markdown("#### 🔝 5 cartes les plus probables à sortir")
if not top_df.empty:
    st.dataframe(top_df.head(5), hide_index=True, height=df_height(top_df.head(5), min_px=190), width="stretch")
else:
    st.info("Sabot vide.")

with st.expander("Détails du sabot (toutes les cartes)"):
    full_df = full_shoe_dataframe(group_faces)
    st.dataframe(full_df, hide_index=True, height=df_height(full_df), width="stretch")
