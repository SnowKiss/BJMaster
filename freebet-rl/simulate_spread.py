# simulate_spread_fractional.py
# --- Simulation Blackjack 3:2 S17 (greedy, sans apprentissage) ---
# Variante dynamique : mise = % de la bankroll COURANTE (par TC)
# + exécutions multiples pour moyenne / écart-type / IC95
# + métriques de risque (MDD, VaR, N0) et suggestion Kelly de spread
# + hysteresis de baisse des mises (anti-sawtooth)

import os
import math
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

# ============================== Page config ==============================
st.set_page_config(
    page_title="Blackjack 3:2 S17 – Simulation fractionnelle (% bankroll)",
    layout="wide",
)

# ============================== Config / Constantes ======================
SAVE_PATH = "qtable.pkl"
TC_MIN, TC_MAX = -5, 5

# 👉 Par défaut, un pourcentage unique de 1% pour tous les TC
DEFAULT_FRACTIONS = {tc: 1.0 for tc in range(TC_MIN, TC_MAX + 1)}  # en %

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
        qtab.epsilon = float(data.get("epsilon", getattr(qtab, "epsilon", 0.0)))
        return True
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de la Q-Table : {e}")
        return False


def clamp_tc(tc: float) -> int:
    """Tronque le TC vers 0 puis borne à [-5, +5] pour coller à l'entraînement."""
    return max(TC_MIN, min(TC_MAX, math.trunc(tc)))


# ============================== État Session =============================
if "env" not in st.session_state:
    # S17 par défaut (dealer_hits_soft_17=False)
    st.session_state.env = FreeBetEnv(num_decks=8, penetration=0.5, dealer_hits_soft_17=False)

if "qtab" not in st.session_state:
    st.session_state.qtab = QTable()
    load_qtable(st.session_state.qtab, SAVE_PATH)

if "fractions" not in st.session_state:
    st.session_state.fractions = DEFAULT_FRACTIONS.copy()  # % par TC

if "multi_results" not in st.session_state:
    st.session_state.multi_results = None  # gardera le dernier lot de simulations


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

hands_target_is_played = st.sidebar.checkbox(
    "Interpréter 'Mains par simulation' = mains effectivement jouées",
    value=True,
    help="Si coché, on comptera uniquement les mains où tu mises (wonging ignoré). Sinon, c'est un nombre d'itérations incluant les passages.",
)

n_hands = st.sidebar.number_input(
    "Mains par simulation",
    min_value=100,
    max_value=1_000_000_000,
    value=1_000_000,
    step=10_000,
)

n_sims = st.sidebar.number_input("Nombre de simulations (pour moyenne)", min_value=1, max_value=500, value=10, step=1)

# Argent / sizing
unit_value = st.sidebar.number_input("Valeur d'une unité (€)", min_value=0.01, max_value=10_000.0, value=1.0, step=0.5)
initial_bankroll_eur = st.sidebar.number_input("Bankroll initiale (€)", min_value=1.0, max_value=10_000_000.0, value=174.0, step=1.0)

col_money1, col_money2 = st.sidebar.columns(2)
with col_money1:
    table_min_eur = st.number_input("Mise minimum table (€)", min_value=0.0, max_value=100_000.0, value=1.0, step=0.5)
with col_money2:
    table_max_eur = st.number_input("Mise maximum table (€) (0 = illimité)", min_value=0.0, max_value=10_000_000.0, value=0.0, step=10.0)

col_money3, col_money4 = st.sidebar.columns(2)
with col_money3:
    chip_step_eur = st.number_input("Pas d'arrondi (jeton) (€)", min_value=0.01, max_value=1_000.0, value=1.0, step=0.5)
with col_money4:
    round_floor = st.checkbox("Arrondir vers le bas (floor)", value=True)

force_min_if_frac_pos = st.sidebar.checkbox(
    "Forcer la mise min si %>0", value=True,
    help="Si coché : toute fraction >0 entraîne au moins la mise mini. Sinon, la main peut être passée si la fraction produit < mise mini."
)

stop_when_br_below_min = st.sidebar.checkbox(
    "Stopper la sim si BR < mise mini",
    value=True,
    help="S'arrête dès que la bankroll ne permet plus de miser la mise minimum.",
)

seed = st.sidebar.number_input("Seed aléatoire (base, optionnel)", min_value=0, max_value=2_147_483_647, value=0, step=1)

# Wong-out réaliste
players_at_table = st.sidebar.number_input("Joueurs à table (approx.)", min_value=1, max_value=7, value=1, step=1)
burn_when_zero = st.sidebar.number_input(
    "Cartes brûlées si % = 0 (≈ multiplicateur de rounds, 8 ≈ 1 round)",
    min_value=0, max_value=80, value=8, step=1,
)

st.sidebar.divider()
# Hystérésis de baisse
hyst_enabled = st.sidebar.checkbox(
    "Anti-baisse de mise (hystérésis)", value=True,
    help="La mise n'est pas réduite tant que la nouvelle mise calculée reste au-dessus du seuil (ex: 70%) de la mise actuelle."
)
hyst_down_pct = st.sidebar.slider(
    "Seuil de réduction de mise (%)", min_value=50, max_value=95, value=70, step=1
)

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


# ============================== Fraction Editor ==========================
st.subheader("💰 % de bankroll par True Count (mise fractionnelle, dynamique)")
frac_df = pd.DataFrame(
    {
        "TC": list(range(TC_MIN, TC_MAX + 1)),
        "% de bankroll": [st.session_state.fractions.get(tc, 1.0) for tc in range(TC_MIN, TC_MAX + 1)],
    }
)

col_sp1, col_sp2 = st.columns([3, 1])
with col_sp1:
    edited = st.data_editor(
        frac_df,
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        key="fraction_editor",
    )
with col_sp2:
    if st.button("↺ Tout remettre à 1%"):
        edited["% de bankroll"] = 1.0
    if st.button("Preset: 0/0/0/0/0/0/0/0/0/1/2/3"):
        # TC -5..5 : 0% jusqu'à TC=1, 1% à TC=2, 2% à TC=3, 3% à TC=4/5
        vals = []
        for tc in range(TC_MIN, TC_MAX + 1):
            if tc <= 1:
                vals.append(0.0)
            elif tc == 2:
                vals.append(1.0)
            elif tc == 3:
                vals.append(2.0)
            else:
                vals.append(3.0)
        edited["% de bankroll"] = vals

# Mettre à jour le dict fractions (% → décimal pendant la sim)
st.session_state.fractions = {int(row["TC"]): float(row["% de bankroll"]) for _, row in edited.iterrows()}
st.caption("Saisis un pourcentage par TC (ex: 1 = 1% de la bankroll courante). Mets 0 pour wonger.")

st.divider()

# ============================== Contrôles ================================
cA, cB, _ = st.columns([1, 1, 3])
with cA:
    run = st.button("▶️ Lancer les simulations", type="primary")
with cB:
    clear = st.button("🧹 Effacer les résultats")

if clear:
    st.session_state.multi_results = None


# ============================== Helpers risques / wong-out ===============
def burn_round(env: FreeBetEnv, n_players: int = 1):
    """
    Avance le shoe d'un 'round' approx. sans jouer.
    Base ~ [4..12] cartes, +2 par joueur additionnel.
    """
    base_min, base_max = 4, 12
    extra = max(0, n_players - 1) * 2
    n = np.random.randint(base_min + extra, base_max + extra + 1)
    for _ in range(n):
        env.shoe.draw()

def burn_zero(env: FreeBetEnv, n_players: int, burn_param: int):
    """
    Utilise burn_when_zero comme multiplicateur de rounds (8 ≈ 1 round).
    """
    k = max(1, int(round(max(0, burn_param) / 8.0)))
    for _ in range(k):
        burn_round(env, n_players=n_players)

def max_drawdown(curve):
    peak = curve[0]
    mdd = 0.0
    for x in curve:
        peak = max(peak, x)
        mdd = max(mdd, (peak - x))
    return mdd

def var5(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, 5))

def estimate_N0(ev_per_hand_units: float, unit_returns: np.ndarray) -> int:
    """
    N0 ≈ variance / edge^2 (mains), variance estimée sur les retours unitaires.
    """
    if unit_returns.size < 2 or abs(ev_per_hand_units) < 1e-12:
        return 0
    var = float(np.var(unit_returns, ddof=1))
    return int(max(0, var / (ev_per_hand_units ** 2)))


# ============================== Simulation (greedy) ======================
def simulate_fractional(
    env: FreeBetEnv,
    qtab: QTable,
    frac_by_tc_pct: dict,
    n_rounds: int,
    initial_br_eur: float,
    unit_value_eur: float,
    table_min_eur: float,
    table_max_eur: float,  # 0 = illimité
    chip_step_eur: float,
    round_floor: bool = True,
    force_min_if_frac_pos: bool = True,
    stop_when_br_lt_min: bool = True,
    burn_when_zero_param: int = 8,
    players_at_table: int = 1,
    hands_are_played: bool = False,
    hysteresis_enabled: bool = True,
    hysteresis_down_pct: float = 70.0,
):
    """
    Stratégie greedy (eps=0). 'g' est en unités de base (1u).
    On convertit en € via la mise effective (issue du % de bankroll).
    Hystérésis: on n'abaisse pas la mise tant que bet_target >= seuil% * bet_last.
    """
    returns_by_tc_base = defaultdict(float)
    sumsq_by_tc = defaultdict(float)   # pour variance par TC
    euros_by_tc = defaultdict(float)
    episodes_by_tc = defaultdict(int)

    unit_returns = []  # g par main jouée (unités)

    rounds_win = rounds_push = rounds_loss = 0
    br_eur = float(initial_br_eur)
    br_curve = [br_eur]
    has_table_max = table_max_eur > 0

    last_bet_eur = 0.0  # mémorise la DERNIÈRE mise effective (post arrondi/caps)

    def compute_bet_eur(br_eur: float, frac_pct: float) -> float:
        """Min/Max AVANT et APRÈS arrondi. Option 'floor' pour coller aux jetons réels."""
        raw_bet = br_eur * (frac_pct / 100.0)
        bet = raw_bet

        # Cap min/max avant arrondi
        if force_min_if_frac_pos and frac_pct > 0.0:
            bet = max(bet, table_min_eur)
        if has_table_max:
            bet = min(bet, table_max_eur)

        # Arrondi
        if chip_step_eur > 0:
            if round_floor:
                bet = math.floor(bet / chip_step_eur) * chip_step_eur
            else:
                bet = round(bet / chip_step_eur) * chip_step_eur

        # Re-cap après arrondi (utile si step large)
        if force_min_if_frac_pos and frac_pct > 0.0:
            bet = max(bet, table_min_eur)
        if has_table_max:
            bet = min(bet, table_max_eur)

        return bet

    def apply_hysteresis(target_bet: float, last_bet: float) -> float:
        """
        Autorise toujours les hausses. Empêche les baisses tant que
        target_bet >= (seuil%) * last_bet. Sinon, accepte la baisse.
        """
        if not hysteresis_enabled or last_bet <= 0:
            return target_bet
        if target_bet >= last_bet:
            return target_bet
        threshold = last_bet * (float(hysteresis_down_pct) / 100.0)
        return last_bet if target_bet >= threshold else target_bet

    if hands_are_played:
        hands = 0
        max_iters = int(n_rounds) * 200  # garde-fou si wonging agressif
        iters = 0
        while hands < int(n_rounds) and iters < max_iters:
            iters += 1
            if stop_when_br_lt_min and br_eur < max(1e-9, table_min_eur):
                break

            tc = clamp_tc(env.shoe.true_count())
            frac_pct = float(frac_by_tc_pct.get(tc, 0.0))
            if frac_pct <= 0.0:
                # Wong-out : ne touche PAS à last_bet_eur
                burn_zero(env, players_at_table, burn_when_zero_param)
                continue

            bet_eur_target = compute_bet_eur(br_eur, frac_pct)
            if bet_eur_target <= 0.0:
                burn_zero(env, players_at_table, burn_when_zero_param)
                continue

            # Hystérésis (comparaison sur la mise effective)
            bet_eur_effective = apply_hysteresis(bet_eur_target, last_bet_eur)
            last_bet_eur = bet_eur_effective  # on mémorise la mise réellement utilisée

            bet_units = bet_eur_effective / max(1e-12, unit_value_eur)
            _, rewards, outcomes = env.play_round(qtab, 0.0, tc)
            g_units = float(np.sum(rewards))
            pnl_eur = g_units * bet_units * unit_value_eur
            br_eur += pnl_eur

            # stats
            unit_returns.append(g_units)
            returns_by_tc_base[tc] += g_units
            sumsq_by_tc[tc] += g_units * g_units
            euros_by_tc[tc] += pnl_eur
            episodes_by_tc[tc] += 1
            hands += 1

            if outcomes.get("win", 0) > 0:
                rounds_win += 1
            elif outcomes.get("push", 0) > 0:
                rounds_push += 1
            else:
                rounds_loss += 1

            br_curve.append(br_eur)
    else:
        for _ in range(int(n_rounds)):
            if stop_when_br_lt_min and br_eur < max(1e-9, table_min_eur):
                break

            tc = clamp_tc(env.shoe.true_count())
            frac_pct = float(frac_by_tc_pct.get(tc, 0.0))
            if frac_pct <= 0.0:
                # Wong-out : ne touche PAS à last_bet_eur
                burn_zero(env, players_at_table, burn_when_zero_param)
                continue

            bet_eur_target = compute_bet_eur(br_eur, frac_pct)
            if bet_eur_target <= 0.0:
                burn_zero(env, players_at_table, burn_when_zero_param)
                continue

            # Hystérésis
            bet_eur_effective = apply_hysteresis(bet_eur_target, last_bet_eur)
            last_bet_eur = bet_eur_effective

            bet_units = bet_eur_effective / max(1e-12, unit_value_eur)
            _, rewards, outcomes = env.play_round(qtab, 0.0, tc)
            g_units = float(np.sum(rewards))
            pnl_eur = g_units * bet_units * unit_value_eur
            br_eur += pnl_eur

            # stats
            unit_returns.append(g_units)
            returns_by_tc_base[tc] += g_units
            sumsq_by_tc[tc] += g_units * g_units
            euros_by_tc[tc] += pnl_eur
            episodes_by_tc[tc] += 1

            if outcomes.get("win", 0) > 0:
                rounds_win += 1
            elif outcomes.get("push", 0) > 0:
                rounds_push += 1
            else:
                rounds_loss += 1

            br_curve.append(br_eur)

    total_return_base = sum(returns_by_tc_base.values())
    total_profit_eur = sum(euros_by_tc.values())
    total_hands = int(sum(episodes_by_tc.values()))
    ev_per_round_units = total_return_base / max(1, total_hands)
    ev_per_round_eur = total_profit_eur / max(1, total_hands)

    return {
        "total_return_base": total_return_base,
        "returns_by_tc_base": dict(returns_by_tc_base),
        "sumsq_by_tc": dict(sumsq_by_tc),
        "episodes_by_tc": dict(episodes_by_tc),
        "euros_by_tc": dict(euros_by_tc),
        "total_profit_eur": total_profit_eur,
        "ev_per_round_units": ev_per_round_units,
        "ev_per_round_eur": ev_per_round_eur,
        "rounds_win": rounds_win,
        "rounds_push": rounds_push,
        "rounds_loss": rounds_loss,
        "end_bankroll_eur": br_eur,
        "bankroll_curve": br_curve,
        "unit_returns": unit_returns,
        "shoe_sessions": env.shoe.sessions_played,
        "hands_played": total_hands,
    }



# ============================== Run multi simulations ====================
# Quick header to confirm Q-Table really loaded
qtab_eps = int(getattr(st.session_state.qtab, "episodes", 0))
st.caption(f"Q-Table: episodes chargés = {qtab_eps:,}")

if run:
    # (Re)charger systématiquement la Q-table du chemin choisi
    ok = load_qtable(st.session_state.qtab, qtable_path)
    if not ok:
        st.error("Q-Table introuvable : simulation impossible (sinon stratégie ‘Hit’).")
        st.stop()

    # sécurité : politique 100% greedy
    st.session_state.qtab.epsilon = 0.0

    sim_results = []

    with st.spinner("Simulations en cours..."):
        for i in range(int(n_sims)):
            # Environnement propre à chaque run
            env = FreeBetEnv(
                num_decks=decks,
                penetration=penetration,
                dealer_hits_soft_17=h17,
                seed=(int(seed) + i) if seed > 0 else None,
            )

            res = simulate_fractional(
                env=env,
                qtab=st.session_state.qtab,
                frac_by_tc_pct=st.session_state.fractions,
                n_rounds=int(n_hands),
                initial_br_eur=float(initial_bankroll_eur),
                unit_value_eur=float(unit_value),
                table_min_eur=float(table_min_eur),
                table_max_eur=float(table_max_eur),
                chip_step_eur=float(chip_step_eur),
                round_floor=bool(round_floor),
                force_min_if_frac_pos=bool(force_min_if_frac_pos),
                stop_when_br_lt_min=bool(stop_when_br_below_min),
                burn_when_zero_param=int(burn_when_zero),
                players_at_table=int(players_at_table),
                hands_are_played=bool(hands_target_is_played),
                hysteresis_enabled=bool(hyst_enabled),
                hysteresis_down_pct=float(hyst_down_pct),
            )
            # tracer le seed effectif
            res["seed"] = (int(seed) + i) if seed > 0 else None
            sim_results.append(res)

    st.session_state.multi_results = sim_results


# ============================== Affichage résultats ======================
if st.session_state.multi_results is not None:
    sims = st.session_state.multi_results

    # ---- Tableau par simulation ----
    rows = []
    for idx, r in enumerate(sims, start=1):
        rows.append(
            {
                "Simulation": idx,
                "Seed": r.get("seed"),
                "Mains jouées": r.get("hands_played", 0),
                "Profit (€)": round(r.get("total_profit_eur", 0.0), 2),
                "EV / main (unités)": round(r.get("ev_per_round_units", 0.0), 4),
                "EV / main (€)": round(r.get("ev_per_round_eur", 0.0), 4),
                "Bankroll fin (€)": round(r.get("end_bankroll_eur", 0.0), 2),
                "Win %": round(100 * r.get("rounds_win", 0) / max(1, r.get("rounds_win", 0) + r.get("rounds_push", 0) + r.get("rounds_loss", 0)), 1),
                "Push %": round(100 * r.get("rounds_push", 0) / max(1, r.get("rounds_win", 0) + r.get("rounds_push", 0) + r.get("rounds_loss", 0)), 1),
                "Loss %": round(100 * r.get("rounds_loss", 0) / max(1, r.get("rounds_win", 0) + r.get("rounds_push", 0) + r.get("rounds_loss", 0)), 1),
                "Sabots joués": r.get("shoe_sessions", 0),
                "Max Drawdown (€)": round(max_drawdown(r.get("bankroll_curve", [0, 0])), 2),
            }
        )

    df_runs = pd.DataFrame(rows)
    st.subheader("📈 Résultats – par simulation")
    st.dataframe(df_runs, hide_index=True, width="stretch")

    # ---- Statistiques agrégées ----
    profits = np.array([r.get("total_profit_eur", 0.0) for r in sims], dtype=float)
    ev_units = np.array([r.get("ev_per_round_units", 0.0) for r in sims], dtype=float)
    ev_eur = np.array([r.get("ev_per_round_eur", 0.0) for r in sims], dtype=float)
    mdds = np.array([max_drawdown(r.get("bankroll_curve", [0, 0])) for r in sims], dtype=float)

    # concat des retours unitaires pour N0
    all_unit_returns = np.concatenate(
        [np.asarray(r.get("unit_returns", []), dtype=float) for r in sims]
    ) if any(len(r.get("unit_returns", [])) > 0 for r in sims) else np.array([], dtype=float)

    def mean_std_ci(x: np.ndarray):
        if x.size == 0:
            return 0.0, 0.0, 0.0
        m = float(np.mean(x))
        s = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        # approx IC95 (normal) : m ± 1.96 * s / sqrt(n)
        half = 1.96 * s / np.sqrt(max(1, len(x)))
        return m, s, half

    m_prof, s_prof, h_prof = mean_std_ci(profits)
    m_evu, s_evu, h_evu = mean_std_ci(ev_units)
    m_eve, s_eve, h_eve = mean_std_ci(ev_eur)
    m_mdd, _, _ = mean_std_ci(mdds)

    # EV/unité globale la plus robuste = moyenne des g par main
    ev_units_global = float(np.mean(all_unit_returns)) if all_unit_returns.size > 0 else m_evu
    N0 = estimate_N0(ev_units_global, all_unit_returns)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Profit moyen (€)", f"{m_prof:,.2f}", help=f"±{h_prof:,.2f} (IC95), σ={s_prof:,.2f}")
    with c2:
        st.metric("EV / main moyen (unités)", f"{ev_units_global:+.4f}", help=f"±{h_evu:.4f} (IC95 ~ par sim), σ={s_evu:.4f}")
    with c3:
        st.metric("Max Drawdown moyen (€)", f"{m_mdd:,.2f}", help="Moyenne des max drawdowns par simulation")
    with c4:
        st.metric("N₀ (mains)", f"{N0:,}", help="Nombre de mains pour que σ ≈ edge ; plus petit = convergence plus rapide")

    st.caption(f"VaR 5% des profits (≈ PNL au pire 5%) : {var5(profits):,.2f} €")

    # ---- Agrégation par TC (pondérée) ----
    agg_returns_by_tc = defaultdict(float)
    agg_episodes_by_tc = defaultdict(int)
    agg_euros_by_tc = defaultdict(float)
    agg_sumsq_by_tc = defaultdict(float)

    for r in sims:
        for tc, v in r.get("returns_by_tc_base", {}).items():
            agg_returns_by_tc[int(tc)] += float(v)
        for tc, n in r.get("episodes_by_tc", {}).items():
            agg_episodes_by_tc[int(tc)] += int(n)
        for tc, e in r.get("euros_by_tc", {}).items():
            agg_euros_by_tc[int(tc)] += float(e)
        for tc, s2 in r.get("sumsq_by_tc", {}).items():
            agg_sumsq_by_tc[int(tc)] += float(s2)

    rows_tc = []
    total_hands = int(sum(agg_episodes_by_tc.values()))
    var_by_tc = {}
    kelly_raw_pct = {}

    for tc in range(TC_MIN, TC_MAX + 1):
        n = int(agg_episodes_by_tc.get(tc, 0))
        ret_sum = float(agg_returns_by_tc.get(tc, 0.0))
        sumsq = float(agg_sumsq_by_tc.get(tc, 0.0))
        euros_tc = float(agg_euros_by_tc.get(tc, 0.0))

        ev_tc_units = (ret_sum / n) if n > 0 else 0.0
        mean2 = (sumsq / n) if n > 0 else 0.0
        var_tc = max(0.0, mean2 - ev_tc_units * ev_tc_units) if n > 0 else 0.0
        var_by_tc[tc] = var_tc

        # Kelly approx (non cappé, non lissé)
        if var_tc <= 1e-12 or ev_tc_units <= 0.0:
            kelly_raw_pct[tc] = 0.0
        else:
            kelly_raw_pct[tc] = 100.0 * (ev_tc_units / var_tc)

        rows_tc.append(
            {
                "TC": tc,
                "Mains": n,
                "% du total": round(100 * n / max(1, total_hands), 2),
                "EV / main (unités)": round(ev_tc_units, 4),
                "Var / main (unités²)": round(var_tc, 5),
                "Kelly brut (%)": round(kelly_raw_pct[tc], 2),
                "Profit total (€)": round(euros_tc, 2),
            }
        )

    df_tc = pd.DataFrame(rows_tc)
    st.subheader("🔎 Détail agrégé par TC (toutes simulations)")
    st.dataframe(df_tc, hide_index=True, width="stretch")

    # ---- Courbe bankroll de la dernière simulation (ou sélection) ----
    st.subheader("📉 Courbe de bankroll (dernière simulation)")
    last_curve = sims[-1]["bankroll_curve"]
    st.line_chart(pd.DataFrame({"Bankroll (€)": last_curve}))

    # ---- Suggestion Kelly (cap + lissage + monotonicité) ----
    with st.expander("🤖 Suggérer un spread via Kelly (approché)"):
        kelly_cap_pct = st.number_input("Cap Kelly (%)", 0.0, 50.0, 5.0, 0.5)
        smooth = st.checkbox("Lisser (moyenne glissante ±1 TC)", value=True)
        enforce_monotone = st.checkbox("Rendre non décroissant avec le TC", value=True)
        min_samples = st.number_input("Min mains par TC pour proposer", 0, 1000000, 200, 50)

        if st.button("Proposer un spread %/TC"):
            suggest = {}
            for tc in range(TC_MIN, TC_MAX + 1):
                n = agg_episodes_by_tc.get(tc, 0)
                if n < int(min_samples):
                    suggest[tc] = 0.0
                    continue
                ev = (agg_returns_by_tc.get(tc, 0.0) / n) if n > 0 else 0.0
                var = var_by_tc.get(tc, 0.0)
                f = 0.0 if var <= 1e-12 or ev <= 0 else min(kelly_cap_pct / 100.0, ev / var)
                suggest[tc] = 100.0 * f  # en %

            if smooth:
                for tc in range(TC_MIN, TC_MAX + 1):
                    window = [suggest.get(t, 0.0) for t in range(tc - 1, tc + 2) if TC_MIN <= t <= TC_MAX]
                    suggest[tc] = float(np.mean(window)) if window else suggest.get(tc, 0.0)

            if enforce_monotone:
                last = 0.0
                for tc in range(TC_MIN, TC_MAX + 1):
                    if suggest[tc] < last:
                        suggest[tc] = last
                    last = suggest[tc]

            # Appliquer au data editor (via session_state) puis rerun
            st.session_state.fractions = suggest
            st.success("Spread proposé via Kelly (cap & lissage appliqués). Le tableau en haut est mis à jour.")
            st.rerun()

    # ---- Export CSV ----
    csv_runs = df_runs.to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Télécharger le résumé par simulation (CSV)",
        data=csv_runs,
        file_name="fractional_spread_runs.csv",
        mime="text/csv",
    )

    csv_tc = df_tc.to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Télécharger l'agrégat par TC (CSV)",
        data=csv_tc,
        file_name="fractional_spread_by_tc.csv",
        mime="text/csv",
    )


st.divider()

# ============================== Debug Q-Table (couverture) ==================
st.subheader("🔬 Sanity-check Q-Table coverage")
try:
    cov_rows = []
    # Parcourt les TC connus et estime la "densité" de la table
    for tc in range(TC_MIN, TC_MAX + 1):
        qdict = dict(getattr(st.session_state.qtab, "q_by_tc", {}).get(tc, {}))
        vdict = dict(getattr(st.session_state.qtab, "visits_by_tc", {}).get(tc, {}))
        n_states = len(qdict)
        nonzero_states = 0
        total_visits = 0
        avg_max_q = 0.0
        if n_states > 0:
            max_qs = []
            for s, q in qdict.items():
                if isinstance(q, (list, tuple, np.ndarray)):
                    q_arr = np.asarray(q, dtype=float)
                    if np.any(np.abs(q_arr) > 1e-12):
                        nonzero_states += 1
                    max_qs.append(np.max(q_arr))
                # visits s'il y en a
                v = vdict.get(s)
                if v is not None:
                    total_visits += int(np.sum(np.asarray(v)))
            if max_qs:
                avg_max_q = float(np.mean(max_qs))
        cov_rows.append({
            "TC": tc,
            "États connus": n_states,
            "États Q≠0": nonzero_states,
            "Visites totales": total_visits,
            "Max(Q) moyen": round(avg_max_q, 4),
        })
    df_cov = pd.DataFrame(cov_rows)
    st.dataframe(df_cov, hide_index=True, width="stretch")
    if df_cov["États connus"].sum() == 0:
        st.warning("Ta Q-Table semble vide (aucun état connu). Le jeu se comportera comme une politique par défaut (souvent 'Hit').")
except Exception as e:
    st.info(f"Impossible de résumer la Q-Table: {e}")

# ============================== Visualisation stratégie ==================
st.subheader("🧠 Tables de décision (issues de la Q-Table chargée)")
tc_choice = st.slider("True Count (TC)", TC_MIN, TC_MAX, 0)
show_percentages = st.checkbox("Afficher les pourcentages D/H/S/P", value=False)

try:
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
except Exception as e:
    st.warning(f"Tables de décision indisponibles : {e}")

st.caption(
    """
Règles simulées :
• Blackjack paie 3:2  
• Dealer s'arrête sur 17 (S17)  
• Pas de resplit, pas de double après split (no DAS)  
• Pas de surrender
"""
)
