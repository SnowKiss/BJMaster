# simulate_spread.py
import os
import pickle
import streamlit as st
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

from freebet.env import FreeBetEnv
from freebet.rl.qtable import QTable
from freebet.ui.tables import (
    build_table_hard,
    build_table_soft,
    build_table_pairs,
    style_actions,
)

# ============================== Page config ==============================
st.set_page_config(page_title="Blackjack 3:2 S17 – Simulation par Spread (greedy, sans apprentissage)", layout="wide")

# ============================== Config / Constantes ======================
SAVE_PATH = "qtable.pkl"
TC_MIN, TC_MAX = -5, 5

# 👉 Par défaut, TOUTES les mises = 1 unité (éditable)
DEFAULT_SPREAD = {tc: 1.0 for tc in range(TC_MIN, TC_MAX + 1)}

# ============================== Utils Q-Table ============================
def load_qtable(qtab: QTable, path: str = SAVE_PATH) -> bool:
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            st.warning(f"⚠️ Q-Table introuvable ou vide à '{path}'.")
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)

        # q_by_tc -> ndarray(float)
        qtab.q_by_tc = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))
        for tc, qdict in data.get("q_by_tc", {}).items():
            qtab.q_by_tc[tc] = defaultdict(
                lambda: np.zeros(4),
                {s: np.asarray(v, dtype=float) for s, v in qdict.items()},
            )

        # visits_by_tc -> ndarray(int)
        qtab.visits_by_tc = defaultdict(lambda: defaultdict(lambda: np.zeros(4, dtype=int)))
        for tc, vdict in data.get("visits_by_tc", {}).items():
            qtab.visits_by_tc[tc] = defaultdict(
                lambda: np.zeros(4, dtype=int),
                {s: np.asarray(v, dtype=int) for s, v in vdict.items()},
            )

        qtab.episodes = int(data.get("episodes", 0))
        qtab.epsilon  = float(data.get("epsilon", getattr(qtab, "epsilon", 0.0)))
        return True
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de la Q-Table : {e}")
        return False

def clamp_tc(tc: int) -> int:
    return max(TC_MIN, min(TC_MAX, tc))

# ============================== État Session =============================
if "env" not in st.session_state:
    # S17 par défaut (dealer_hits_soft_17=False)
    st.session_state.env = FreeBetEnv(num_decks=8, penetration=0.5, dealer_hits_soft_17=False)

if "qtab" not in st.session_state:
    st.session_state.qtab = QTable()
    load_qtable(st.session_state.qtab, SAVE_PATH)

if "spread" not in st.session_state:
    st.session_state.spread = DEFAULT_SPREAD.copy()

if "sim_results" not in st.session_state:
    st.session_state.sim_results = None  # gardera le dernier résultat

# ============================== Sidebar: Paramètres ======================
st.sidebar.header("⚙️ Paramètres de simulation")
qtable_path = st.sidebar.text_input("Chemin Q-Table (.pkl)", value=SAVE_PATH)

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    decks = st.number_input("Nombre de decks", min_value=1, max_value=12, value=8, step=1)
with col_sb2:
    penetration = st.number_input("Pénétration", min_value=0.1, max_value=0.99, value=0.5, step=0.05)

# S17 par défaut
h17 = st.sidebar.checkbox("Dealer H17 (tire sur soft 17)", value=False)

n_hands   = st.sidebar.number_input("Nombre de parties à simuler", min_value=100, max_value=1_000_000_000, value=1_000_000, step=10_000)
unit_value = st.sidebar.number_input("Valeur d'une unité (€)", min_value=0.01, max_value=10_000.0, value=1.0, step=0.5)
seed      = st.sidebar.number_input("Seed aléatoire (optionnel)", min_value=0, max_value=2_147_483_647, value=0, step=1)

# Option “wonging” si jamais tu remets des 0 (inutile si toutes les mises = 1)
burn_when_zero = st.sidebar.number_input("Cartes brûlées si mise=0 (approx.)", min_value=0, max_value=40, value=8, step=1)

st.sidebar.divider()
if st.sidebar.button("📂 (Re)charger la Q-Table"):
    ok = load_qtable(st.session_state.qtab, qtable_path)
    if ok:
        st.success("Q-Table chargée avec succès ✅")
    else:
        st.stop()

if st.sidebar.button("🔄 Réinitialiser le shoe"):
    st.session_state.env = FreeBetEnv(
        num_decks=decks,
        penetration=penetration,
        dealer_hits_soft_17=h17,
        seed=int(seed) if seed > 0 else None,
    )
    st.success("Shoe réinitialisé.")

# ============================== Spread Editor ===========================
st.subheader("💰 Spread de mises par True Count (en unités)")
spread_df = pd.DataFrame({
    "TC": list(range(TC_MIN, TC_MAX + 1)),
    "Mise (unités)": [st.session_state.spread.get(tc, 1.0) for tc in range(TC_MIN, TC_MAX + 1)],
})
col_sp1, col_sp2 = st.columns([3, 1])
with col_sp1:
    edited = st.data_editor(
        spread_df,
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        key="spread_editor",
    )
with col_sp2:
    if st.button("↺ Tout remettre à 1"):
        edited["Mise (unités)"] = 1.0

# Mettre à jour le dict spread (par défaut tout = 1.0)
st.session_state.spread = {int(row["TC"]): float(row["Mise (unités)"]) for _, row in edited.iterrows()}
st.caption("Par défaut, toutes les mises = 1 unité. Tu peux éditer si besoin.")

st.divider()

# ============================== Contrôles ================================
cA, cB, _ = st.columns([1, 1, 3])
with cA:
    run = st.button("▶️ Lancer la simulation", type="primary")
with cB:
    clear = st.button("🧹 Effacer les résultats")

if clear:
    st.session_state.sim_results = None

# ============================== Simulation (greedy) ======================
def burn_cards(env: FreeBetEnv, n: int):
    """Fait avancer le shoe sans distribuer au joueur (utile si mise=0)."""
    for _ in range(int(n)):
        env.shoe.draw()

def simulate_spread(
    env: FreeBetEnv,
    qtab: QTable,
    spread: dict,
    n_rounds: int,
    eps: float = 0.0,
    seed: int | None = None,
    burn_when_zero: int = 8
):
    """
    Joue n_rounds coups en suivant la politique greedy (eps=0).
    - EV/EV par TC sont calculés en unités de base (comme eval_ev.py).
    - W/P/L sont comptés par manche (comme eval_ev.py).
    - Le profit pondéré par la mise est calculé A POSTERIORI
      en multipliant le retour-base de chaque TC par la mise de ce TC.
    """
    # agrégats en unités de base
    returns_by_tc_base = defaultdict(float)
    episodes_by_tc     = defaultdict(int)

    rounds_win = rounds_push = rounds_loss = 0

    for _ in range(int(n_rounds)):
        tc = clamp_tc(env.shoe.true_count())

        # cas wonging (si tu remets des 0 plus tard)
        bet_units = float(spread.get(tc, 1.0))
        if bet_units <= 0.0:
            for __ in range(int(burn_when_zero)):
                env.shoe.draw()
            continue

        # même logique que eval_ev.py
        _, rewards, outcomes = env.play_round(qtab, 0.0, tc)
        g = float(np.sum(rewards))

        returns_by_tc_base[tc] += g
        episodes_by_tc[tc]     += 1

        if outcomes.get("win", 0) > 0:
            rounds_win += 1
        elif outcomes.get("push", 0) > 0:
            rounds_push += 1
        else:
            rounds_loss += 1

    # totaux
    total_return_base = sum(returns_by_tc_base.values())

    # profit pondéré par la mise (ici tout = 1 → identique au base)
    profit_by_tc_units = {
        tc: returns_by_tc_base.get(tc, 0.0) * float(spread.get(tc, 1.0))
        for tc in range(TC_MIN, TC_MAX + 1)
    }
    total_profit_units = sum(profit_by_tc_units.values())

    return {
        "total_return_base": total_return_base,
        "returns_by_tc_base": dict(returns_by_tc_base),
        "episodes_by_tc": dict(episodes_by_tc),

        "profit_by_tc_units": profit_by_tc_units,
        "total_profit_units": total_profit_units,

        "rounds_win": rounds_win,
        "rounds_push": rounds_push,
        "rounds_loss": rounds_loss,
        "shoe_sessions": env.shoe.sessions_played,
    }

if run:
    # Recréer un env propre avec les paramètres choisis (seed inclus)
    st.session_state.env = FreeBetEnv(
        num_decks=decks,
        penetration=penetration,
        dealer_hits_soft_17=h17,
        seed=int(seed) if seed > 0 else None,
    )

    # (Re)charger systématiquement la Q-table du chemin choisi
    ok = load_qtable(st.session_state.qtab, qtable_path)
    if not ok:
        st.error("Q-Table introuvable : simulation impossible (sinon stratégie ‘Hit’).")
        st.stop()

    # sécurité : politique 100% greedy
    st.session_state.qtab.epsilon = 0.0

    with st.spinner("Simulation en cours..."):
        res = simulate_spread(
            env=st.session_state.env,
            qtab=st.session_state.qtab,
            spread=st.session_state.spread,
            n_rounds=n_hands,
            eps=0.0,
            seed=int(seed) if seed > 0 else None,
            burn_when_zero=int(burn_when_zero),
        )
        st.session_state.sim_results = res

# ============================== Affichage résultats ======================
if st.session_state.sim_results is not None:
    res = st.session_state.sim_results
    total_hands        = int(sum(res["episodes_by_tc"].values()))
    ev_per_round_units = res["total_return_base"] / max(1, total_hands)

    total_profit_units = res["total_profit_units"]   # = base si mises=1
    total_profit_eur   = total_profit_units * unit_value


    st.subheader("📈 Résultats de simulation")
    m1, m2, m3, m4 = st.columns([1, 1, 1, 1])
    with m1: st.metric("Mains simulées", f"{total_hands:,}")
    with m2: st.metric("Profit total (unités)", f"{total_profit_units:,.2f}")
    with m3: st.metric("Profit total (€)", f"{total_profit_eur:,.2f}")
    with m4: st.metric("EV / main (unités)", f"{ev_per_round_units:+.4f}")

    # W / P / L PAR MANCHE (aligné eval_ev.py)
    total_rounds = max(1, res["rounds_win"] + res["rounds_push"] + res["rounds_loss"])
    cW, cP, cL, cS = st.columns([1, 1, 1, 1])
    with cW: st.metric("Win %",  f"{100*res['rounds_win']/total_rounds:.1f} %")
    with cP: st.metric("Push %", f"{100*res['rounds_push']/total_rounds:.1f} %")
    with cL: st.metric("Loss %", f"{100*res['rounds_loss']/total_rounds:.1f} %")
    with cS: st.metric("Sabots joués", res["shoe_sessions"])

    # Tableau par TC
    rows = []
    for tc in range(TC_MIN, TC_MAX + 1):
        n      = int(res["episodes_by_tc"].get(tc, 0))
        ret_b  = float(res["returns_by_tc_base"].get(tc, 0.0))  # base
        ev_tc  = (ret_b / n) if n > 0 else 0.0
        prof_u = float(res["profit_by_tc_units"].get(tc, 0.0))  # pondéré
        rows.append({
            "TC": tc,
            "Mains": n,
            "Mise (unités)": float(st.session_state.spread.get(tc, 1.0)),
            "EV / main (unités)": round(ev_tc, 4),
            "Profit (unités)": round(prof_u, 2),
            "Profit (€)": round(prof_u * unit_value, 2),
            "% du total": round(100 * n / max(1, total_hands), 2),
        })

    df_tc = pd.DataFrame(rows)
    st.dataframe(df_tc, hide_index=True, width="stretch")

    # Graph EV / TC (unités de base, comme eval_ev.py)
    if df_tc["Mains"].sum() > 0:
        st.line_chart(df_tc.set_index("TC")["EV / main (unités)"])

    # Export CSV
    csv = df_tc.to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Télécharger le détail par TC (CSV)",
        data=csv,
        file_name="simulation_spread_par_TC.csv",
        mime="text/csv",
    )

st.divider()

# ============================== Visualisation stratégie ==================
st.subheader("🧠 Tables de décision (issues de la Q-Table chargée)")
tc_choice = st.slider("True Count (TC)", TC_MIN, TC_MAX, 0)
show_percentages = st.checkbox("Afficher les pourcentages D/H/S/P", value=False)

tab1, tab2, tab3 = st.tabs(["Hard totals", "Soft totals", "Pairs"])
with tab1:
    df_hard = build_table_hard(st.session_state.qtab, st.session_state.env, tc_choice, show_percentages)
    st.dataframe(style_actions(df_hard), width="stretch", hide_index=True)
with tab2:
    df_soft = build_table_soft(st.session_state.qtab, st.session_state.env, tc_choice, show_percentages)
    st.dataframe(style_actions(df_soft), width="stretch", hide_index=True)
with tab3:
    df_pairs = build_table_pairs(st.session_state.qtab, st.session_state.env, tc_choice, show_percentages)
    st.dataframe(style_actions(df_pairs), width="stretch", hide_index=True)

st.caption(
    """
Règles simulées :
• Blackjack paie 3:2  
• Dealer s'arrête sur 17 (S17)  
• Pas de resplit, pas de double après split (no DAS)  
• Pas de surrender
"""
)
