import os
import io
import json
import pickle
import math
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

# ============================== CONFIG ==============================
st.set_page_config(
    page_title="Blackjack – Live Strategy by True Count",
    page_icon="🎴",
    layout="wide",
)

DEFAULT_DECKS = 8
SAVE_PATH = "qtable.pkl"

RANKS = ["2","3","4","5","6","7","8","9","10","J","Q","K","A"]
KEYS  = ["2","3","4","5","6","7","8","9","T","J","Q","K","A"]  # interne: "10" -> "T"
HILO  = {"2":+1,"3":+1,"4":+1,"5":+1,"6":+1,"7":0,"8":0,"9":0,"T":-1,"J":-1,"Q":-1,"K":-1,"A":-1}

# ============================== STYLES ==============================
st.markdown("""
<style>
/* Sidebar compacte */
section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] .stNumberInput {
  margin-bottom: .5rem;
}

/* Badges TC */
.badge { display:inline-block; padding:.2rem .5rem; border-radius:999px; font-size:.85rem; font-weight:600; }
.badge.tcpos { background:#e6ffed; color:#046a38; border:1px solid #c1f2cf; }
.badge.tcneg { background:#ffecec; color:#8a0a0a; border:1px solid #ffd0d0; }
.badge.tczero{ background:#eef2ff; color:#243c5a; border:1px solid #d7e0ff; }

/* Chips des dernières cartes */
.cardstrip { display:flex; gap:.5rem; flex-wrap:wrap; align-items:center; }
.cardchip-wrap { position:relative; }
.cardchip {
  width:48px; height:62px; border-radius:8px; border:2px solid #e5e7eb;
  box-shadow:0 2px 6px rgba(0,0,0,.06);
  display:flex; align-items:center; justify-content:center;
  font-weight:800; font-size:1.1rem; background:#fff;
}
.cardchip.green { border-color:#b7f0c0; background:#f3fff5; color:#046a38; }
.cardchip.gray  { border-color:#e5e7eb; background:#fafafa; color:#374151; }
.cardchip.red   { border-color:#ffbdbd; background:#fff5f5; color:#8a0a0a; }
.cardqty {
  position:absolute; bottom:-6px; right:-6px;
  background:#111827; color:#fff; font-size:.70rem; line-height:1; padding:.15rem .35rem;
  border-radius:999px; border:2px solid #fff; box-shadow:0 1px 3px rgba(0,0,0,.15);
}

/* Petits libellés dans le pad */
.pad-caption { color: var(--text-color-secondary, #9ca3af); font-size: .8rem; }
</style>
""", unsafe_allow_html=True)

# ============================== HELPERS ==============================
def _force_rerun():
    try:
        st.rerun()                   # Streamlit >=1.27
    except Exception:
        try:
            st.experimental_rerun()  # fallback anciens Streamlit
        except Exception:
            pass

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
        st.session_state.qty_click = 1
    if "group_faces" not in st.session_state:
        st.session_state.group_faces = True
    if "min_tc_play" not in st.session_state:
        st.session_state.min_tc_play = 2
    if "max_tc_play" not in st.session_state:
        st.session_state.max_tc_play = 99
    if "qtable_mtime" not in st.session_state:
        st.session_state.qtable_mtime = os.path.getmtime(SAVE_PATH) if os.path.exists(SAVE_PATH) else None

def fresh_shoe_counts(decks: int) -> dict:
    per_rank = 4 * decks
    return {k: per_rank for k in KEYS}

def cards_left() -> int:
    return int(sum(st.session_state.shoe_counts.values()))

def cards_seen() -> int:
    total = 52 * st.session_state.decks
    return total - cards_left()

def compute_tc(rc: int) -> int:
    left = cards_left()
    if left <= 0:
        return 0
    decks_rem = left / 52.0              # pas de clamp à 1.0 !
    tc_float = rc / decks_rem
    return math.trunc(tc_float)          # troncature vers 0

def can_decrease(rank_key: str, qty: int = 1) -> bool:
    return st.session_state.shoe_counts[rank_key] >= qty

def can_increase(rank_key: str, qty: int = 1) -> bool:
    return st.session_state.shoe_counts[rank_key] + qty <= st.session_state.initial_per_rank

def add_card_seen(rank_key: str, qty: int = 1):
    qty = int(qty)
    if qty <= 0:
        return
    if not can_decrease(rank_key, qty):
        st.toast(f"❗️ Plus assez de {rank_key} dans le sabot.", icon="⚠️")
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
        st.toast(f"❗️ Dépasse le total initial pour {rank_key}.", icon="⚠️")
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
        qtab.epsilon = data.get("epsilon", getattr(qtab, "epsilon", 0.0))
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement Q-Table: {e}")

def auto_load_qtable(path: str = SAVE_PATH, show_toast: bool = False):
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return
        mtime = os.path.getmtime(path)
        if st.session_state.get("qtable_mtime") != mtime:
            load_qtable(st.session_state.qtab, path)
            st.session_state.qtable_mtime = mtime
            if show_toast:
                st.toast("Q-Table rechargée automatiquement", icon="🔄")
    except Exception as e:
        st.error(f"❌ Auto-load Q-Table: {e}")

def export_qtable_bytes(qtab: QTable) -> bytes:
    data = {
        "q_by_tc": {tc: dict(qtab.q_by_tc[tc]) for tc in qtab.q_by_tc},
        "visits_by_tc": {tc: dict(qtab.visits_by_tc[tc]) for tc in qtab.visits_by_tc},
        "episodes": int(getattr(qtab, "episodes", 0)),
        "epsilon": float(getattr(qtab, "epsilon", 0.0)),
    }
    bio = io.BytesIO()
    pickle.dump(data, bio)
    return bio.getvalue()

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

def df_height(df, row_px=34, header_px=38, pad_px=18, min_px=240, max_px=1200):
    n = getattr(df, "shape", (0,0))[0]
    h = header_px + n * row_px + pad_px
    return int(min(max(h, min_px), max_px))

def render_last_cards(n: int = 3):
    if not st.session_state.history:
        st.info("Aucune saisie pour l’instant.")
        return
    last_events = [ev for ev in reversed(st.session_state.history) if ev.get("qty", 0) > 0][:n]
    if not last_events:
        st.info("Aucune saisie positive récente."); return
    label_map = {**{str(k): str(k) for k in range(2,10)}, "T": "10"}
    chips = ['<div class="cardstrip">']
    for ev in last_events:
        rank = ev["rank"]; qty = int(ev["qty"])
        label = label_map.get(rank, rank)
        hilo = HILO[rank]; cls = "green" if hilo > 0 else ("red" if hilo < 0 else "gray")
        chips.append(
            f'<div class="cardchip-wrap">'
            f'<div class="cardchip {cls}">{label}</div>'
            f'<div class="cardqty">×{qty}</div>'
            f'</div>'
        )
    chips.append('</div>')
    st.markdown(''.join(chips), unsafe_allow_html=True)

# ---- Pad 2 colonnes (clics traités AVANT affichage + rerun immédiat) ----
def render_card_pad_two_cols():
    label_map = {**{str(n):str(n) for n in range(2,10)}, "T":"10"}
    left_order  = ["2","3","4","5","6","7","8"]
    right_order = ["9","T","J","Q","K","A"]

    def section(order, prefix):
        st.caption(f'<span class="pad-caption">±{st.session_state.qty_click} par clic</span>', unsafe_allow_html=True)
        hdr = st.columns([1.1, 1.2, 1, 1])
        with hdr[0]: st.caption("Carte")
        with hdr[1]: st.caption("Restantes")
        with hdr[2]: st.caption("−")
        with hdr[3]: st.caption("+")

        for k in order:
            cols = st.columns([1.1, 1.2, 1, 1])

            # 1) Boutons en premier → update state → rerun immédiat
            plus  = cols[3].button("+", key=f"{prefix}_plus_{k}",  use_container_width=True)
            minus = cols[2].button("–", key=f"{prefix}_minus_{k}", use_container_width=True)

            if plus:
                add_card_seen(k, st.session_state.qty_click)
                _force_rerun()
            if minus:
                add_card_back(k, st.session_state.qty_click)
                _force_rerun()

            # 2) Affichage après potentielle mise à jour
            with cols[0]:
                st.markdown(f"**{label_map.get(k,k)}**")
            with cols[1]:
                st.write(st.session_state.shoe_counts[k])

    colL, colR = st.columns(2)
    with colL: section(left_order, "L")
    with colR: section(right_order, "R")

# ============================== INIT ==============================
init_state()
auto_load_qtable(SAVE_PATH, show_toast=False)

# ============================== HEADER ==============================
total_cards = 52 * st.session_state.decks
pen = cards_seen() / total_cards if total_cards else 0.0
tc_display = compute_tc(st.session_state.rc)           # TC réel (tronqué)
tc_tables  = max(-5, min(5, tc_display))               # clamp pour indexer la Q-table ([-5,+5])

title_col, play_col, actions_col = st.columns([4,2,2])
with title_col:
    st.title("🎴 Live Strategy by True Count")
    if tc_display > 0:
        st.markdown(f'<span class="badge tcpos">TC = {tc_display:+d}</span>', unsafe_allow_html=True)
    elif tc_display < 0:
        st.markdown(f'<span class="badge tcneg">TC = {tc_display:+d}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge tczero">TC = {tc_display:+d}</span>', unsafe_allow_html=True)

with play_col:
    play = st.session_state.min_tc_play <= tc_display <= st.session_state.max_tc_play
    st.subheader("✅ PLAY (Wong-in)" if play else "⏭️ SKIP (Wong-out)")

with actions_col:
    c1, c2 = st.columns(2)
    with c1:
        if st.button("↩️ Undo", use_container_width=True):
            undo_last()
            _force_rerun()
    with c2:
        if st.button("♻️ Reshuffle", use_container_width=True):
            reshuffle()
            _force_rerun()

m1,m2,m3,m4,m5 = st.columns(5)
with m1: st.metric("Running Count", st.session_state.rc)
with m2: st.metric("True Count", tc_display)
with m3: st.metric("Restantes", cards_left())
with m4: st.metric("Sabot", f"{st.session_state.decks}×52")
with m5: st.metric("Pénétration", f"{pen:.0%}")
st.progress(min(max(pen, 0.0), 1.0), text=f"Pénétration du sabot : {pen:.0%}")

# Dernières cartes (chips)
st.markdown("#### 🃏 Dernières cartes saisies")
render_last_cards(n=3)

# ============================== LAYOUT 2 COLONNES ==============================
left, right = st.columns([1.0, 1.25])

# ---- Colonne gauche : PAD + Probas ----
with left:
    st.subheader("🎯 Saisie rapide des cartes")
    render_card_pad_two_cols()

    # Probas
    st.markdown("##### 🔝 Probabilités immédiates")
    top_df = probs_dataframe(st.session_state.group_faces)
    if not top_df.empty:
        st.dataframe(top_df.head(5), hide_index=True,
                     height=df_height(top_df.head(5), min_px=170),
                     use_container_width=True)
    else:
        st.info("Sabot vide.")

    with st.expander("Détails du sabot (toutes les cartes)"):
        full_df = full_shoe_dataframe(st.session_state.group_faces)
        st.dataframe(full_df, hide_index=True, height=df_height(full_df), use_container_width=True)
        chart_df = pd.DataFrame({
            "Carte": [("10" if k=="T" else k) for k in ["2","3","4","5","6","7","8","9","T","J","Q","K","A"]],
            "Restantes": [st.session_state.shoe_counts[k] for k in ["2","3","4","5","6","7","8","9","T","J","Q","K","A"]]
        }).set_index("Carte")
        st.bar_chart(chart_df, use_container_width=True)

# ---- Colonne droite : Tables stratégie (toujours visibles) ----
with right:
    st.subheader("📊 Tables de stratégie (TC courant)")
    if tc_tables != tc_display:
        st.caption(f"TC utilisé pour les tables : {tc_tables:+d} (clampé à [-5,+5])")
    show_ev = st.checkbox("Afficher les EV", value=False, key="show_ev")
    # Hard
    st.markdown("###### Hard Totals")
    df_hard = build_table_hard(st.session_state.qtab, st.session_state.env, tc_tables, show_ev)
    st.dataframe(style_actions(df_hard), hide_index=True, height=df_height(df_hard, min_px=260), use_container_width=True)
    # Soft
    st.markdown("###### Soft Totals")
    df_soft = build_table_soft(st.session_state.qtab, st.session_state.env, tc_tables, show_ev)
    st.dataframe(style_actions(df_soft), hide_index=True, height=df_height(df_soft, min_px=240), use_container_width=True)
    # Pairs
    st.markdown("###### Pairs")
    df_pairs = build_table_pairs(st.session_state.qtab, st.session_state.env, tc_tables, show_ev)
    st.dataframe(style_actions(df_pairs), hide_index=True, height=df_height(df_pairs, min_px=200), use_container_width=True)

# ============================== SIDEBAR ==============================
with st.sidebar:
    st.header("Paramètres")
    decks_new = st.number_input("Nombre de jeux", 1, 12, value=st.session_state.decks)
    if decks_new != st.session_state.decks:
        st.session_state.decks = int(decks_new)
        reshuffle()
        _force_rerun()

    st.toggle("Grouper 10/J/Q/K", key="group_faces", value=st.session_state.group_faces)

    st.number_input(
        "Quantité par clic",
        min_value=1, max_value=20, step=1,
        value=st.session_state.qty_click, key="qty_click"
    )

    st.markdown("#### Wonging")
    cmin, cmax = st.columns(2)
    with cmin:
        st.number_input("TC min PLAY", value=st.session_state.min_tc_play, step=1, key="min_tc_play")
    with cmax:
        st.number_input("TC max PLAY", value=st.session_state.max_tc_play, step=1, key="max_tc_play")

    st.divider()
    st.markdown("#### Q-Table")
    uploaded = st.file_uploader("Charger Q-Table (.pkl)", type=["pkl"])
    if uploaded:
        with open(SAVE_PATH, "wb") as f:
            f.write(uploaded.getbuffer())
        auto_load_qtable(SAVE_PATH, show_toast=True)
        _force_rerun()

    st.download_button(
        "Exporter Q-Table",
        data=export_qtable_bytes(st.session_state.qtab),
        file_name="qtable_export.pkl",
        mime="application/octet-stream",
        use_container_width=True
    )

    st.divider()
    st.markdown("#### Sauvegarde sabot")
    snapshot = {
        "decks": st.session_state.decks,
        "rc": st.session_state.rc,
        "shoe_counts": st.session_state.shoe_counts,
        "history": st.session_state.history,
    }
    st.download_button(
        "Télécharger sabot.json",
        data=json.dumps(snapshot, indent=2).encode("utf-8"),
        file_name="shoe_snapshot.json",
        mime="application/json",
        use_container_width=True
    )
    snap = st.file_uploader("Restaurer sabot (.json)", type=["json"], key="snap_upl")
    if snap is not None:
        try:
            restored = json.loads(snap.read().decode("utf-8"))
            st.session_state.decks = int(restored.get("decks", st.session_state.decks))
            st.session_state.rc = int(restored.get("rc", 0))
            st.session_state.shoe_counts = {k: int(v) for k, v in restored.get("shoe_counts", {}).items()}
            st.session_state.initial_per_rank = 4 * st.session_state.decks
            st.session_state.history = restored.get("history", [])
            st.toast("Sabot restauré ✅", icon="✅")
            _force_rerun()
        except Exception as e:
            st.error(f"Impossible de restaurer: {e}")
