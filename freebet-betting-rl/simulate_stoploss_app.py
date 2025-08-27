import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(r"C:\Project\BJMaster\BJMaster\freebet-rl")
from freebet.env import FreeBetEnv
from basic_strategy import BasicStrategy

# ---------------- CONFIG ----------------
DATA_DIR = "data"
INITIAL_BANKROLL = 100
MAX_STEPS = 200
TARGET_GAIN = 1.03  # +3%
STOP_LOSSES = [0.95, 0.90, 0.80, 0.70]
MAX_SIMULATIONS = 100000
sns.set_theme(style="whitegrid")

# ---------------- SIMULATION ----------------
def simulate_hand(game, player_cards, dealer_cards, bet_size):
    dealer_up = dealer_cards[1]
    bs = st.session_state.bs

    while True:
        total, soft = game.state_key(player_cards, dealer_up, True)[:2]
        pair_rank = 0
        action = bs.get_action(total, soft, pair_rank, dealer_up)

        if action == "H":
            player_cards.append(game.shoe.draw())
            t, _ = game.state_key(player_cards, dealer_up, False)[:2]
            if t > 21:
                return -bet_size
        elif action == "S":
            break
        elif action == "D":
            bet_size *= 2
            player_cards.append(game.shoe.draw())
            break
        else:
            break

    dealer_cards = game.dealer_play(dealer_cards)
    dealer_total, _ = game.state_key(dealer_cards, dealer_up, False)[:2]
    player_total, _ = game.state_key(player_cards, dealer_up, False)[:2]

    if player_total > 21:
        return -bet_size
    if dealer_total > 21 or player_total > dealer_total:
        return +bet_size
    elif player_total < dealer_total:
        return -bet_size
    else:
        return 0


def run_simulation(bet_size, stop_loss, initial_bankroll=INITIAL_BANKROLL, max_steps=MAX_STEPS):
    bankroll = initial_bankroll
    steps = 0
    game = FreeBetEnv(num_decks=8, penetration=0.5, dealer_hits_soft_17=True)

    # arrondi à l’entier inf
    target_value = int(initial_bankroll * TARGET_GAIN)

    while steps < max_steps:
        steps += 1
        player_cards, dealer_cards = game.initial_deal()
        outcome = simulate_hand(game, player_cards, dealer_cards, bet_size)
        bankroll += outcome

        # stop-win arrondi à l’inférieur
        if bankroll >= target_value:
            return "success", bankroll

        # stop-loss
        if bankroll <= initial_bankroll * stop_loss:
            return "fail", bankroll

    return "neutral", bankroll


# ---------------- STREAMLIT APP ----------------
st.title("📉 Simulation Stop-Loss – Flat Bet (+3% objectif)")

if "results" not in st.session_state:
    st.session_state.results = []
if "running" not in st.session_state:
    st.session_state.running = False
if "bs" not in st.session_state:
    st.session_state.bs = BasicStrategy(
        hard_csv=os.path.join(DATA_DIR, "hard_totals.csv"),
        soft_csv=os.path.join(DATA_DIR, "soft_totals.csv"),
        pairs_csv=os.path.join(DATA_DIR, "pairs.csv"),
    )
if "simulations_done" not in st.session_state:
    st.session_state.simulations_done = 0

# 🏦 Input bankroll utilisateur
INITIAL_BANKROLL = st.number_input(
    "💰 Bankroll initiale (€)", min_value=1.0, value=100.0, step=1.0, format="%.2f"
)

col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start / Resume"):
        st.session_state.running = True
with col2:
    if st.button("⏸️ Pause"):
        st.session_state.running = False

st.write(f"🎯 Objectif : +3% → {INITIAL_BANKROLL * TARGET_GAIN:.2f}€")
st.write(f"📉 Stop-loss testés : {', '.join([f'{int(sl*100)}%' for sl in STOP_LOSSES])}")
st.write(f"🎲 Mises testées : 1€ à {INITIAL_BANKROLL}€")

# --------- PROGRESS BAR ----------
progress_bar = st.progress(0)
st.metric("🧮 Simulations effectuées", st.session_state.simulations_done)

# --------- LOOP ----------
BATCH_SIZE = 2500

if st.session_state.running and st.session_state.simulations_done < MAX_SIMULATIONS:
    max_bet = max(1, 5)
    for _ in range(BATCH_SIZE):
        if st.session_state.simulations_done >= MAX_SIMULATIONS:
            st.session_state.running = False
            break
        bet = (st.session_state.simulations_done % max_bet) + 1
        sl = STOP_LOSSES[st.session_state.simulations_done % len(STOP_LOSSES)]
        outcome, final_bankroll = run_simulation(bet, sl, initial_bankroll=INITIAL_BANKROLL)
        st.session_state.results.append({
            "bet": bet,
            "stop_loss": sl,
            "outcome": outcome,
            "final_bankroll": final_bankroll
        })
        st.session_state.simulations_done += 1

    # update UI seulement après 1000 runs
    progress_bar.progress(st.session_state.simulations_done / MAX_SIMULATIONS)
    st.rerun()


# --------- VISUALISATION ----------
st.subheader("🎯 Analyse détaillée")

if len(st.session_state.results) > 0:
    df = pd.DataFrame(st.session_state.results)

    grouped = df.groupby(["bet", "stop_loss"]).outcome.value_counts(normalize=True).unstack(fill_value=0)
    grouped = grouped.reset_index().rename_axis(None, axis=1)

    grouped["success_rate"] = grouped.get("success", 0)
    grouped["bankrupt_rate"] = grouped.get("fail", 0)
    grouped["neutral_rate"] = grouped.get("neutral", 0)

    results_df = grouped[["bet", "stop_loss", "success_rate", "bankrupt_rate", "neutral_rate"]]

    st.subheader("🏆 Top stratégies")
    top = results_df.copy()
    top["score"] = top["success_rate"] - top["bankrupt_rate"]
    st.dataframe(top.sort_values("score", ascending=False).head(10))

    bet_choice = st.selectbox("Choisir une mise à analyser", sorted(results_df["bet"].unique()))
    df_bet = results_df[results_df["bet"] == bet_choice]

    fig, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(df_bet["stop_loss"], df_bet["success_rate"], label="Taux de succès (+3%)", color="blue", marker="o")
    ax1.plot(df_bet["stop_loss"], df_bet["bankrupt_rate"], label="Taux de banqueroute", color="red", marker="o")
    ax1.set_xlabel("Stop-loss")
    ax1.set_ylabel("Taux")
    ax1.legend()
    st.pyplot(fig)
    
else:
    st.info("Pas encore de résultats disponibles.")
