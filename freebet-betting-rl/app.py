import os
import torch
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from betting_env import BettingEnv
from basic_strategy import BasicStrategy
from policy import PolicyNet
from train_pg import train_one_episode

# ------------------- CONFIG -------------------
SAVE_PATH = "policy.pt"
DATA_DIR = "data"
BATCH_SIZE = 500  # nb d’épisodes par tick
sns.set_theme(style="whitegrid")

TARGET_GAIN = 1.03  # +3%

# ------------------- INIT -------------------
if "policy" not in st.session_state:
    st.session_state.policy = PolicyNet(5, 6)  # 6 actions (Stop + 1..5€)
    st.session_state.optimizer = torch.optim.Adam(st.session_state.policy.parameters(), lr=1e-3)
    st.session_state.episodes_done = 0
    st.session_state.logs = []
    st.session_state.running = False
    st.session_state.success = 0
    st.session_state.fail = 0
    st.session_state.neutral = 0

    # charger la basic strategy
    bs = BasicStrategy(
        hard_csv=os.path.join(DATA_DIR, "hard_totals.csv"),
        soft_csv=os.path.join(DATA_DIR, "soft_totals.csv"),
        pairs_csv=os.path.join(DATA_DIR, "pairs.csv"),
    )
    st.session_state.env = BettingEnv(bs, bankroll=100, max_steps=200)

# ------------------- SAVE/LOAD -------------------
def save_policy():
    torch.save({
        "policy_state": st.session_state.policy.state_dict(),
        "optimizer_state": st.session_state.optimizer.state_dict(),
        "episodes_done": st.session_state.episodes_done,
        "success": st.session_state.success,
        "fail": st.session_state.fail,
        "neutral": st.session_state.neutral,
    }, SAVE_PATH)
    st.success("✅ Policy saved.")

def load_policy():
    if os.path.exists(SAVE_PATH):
        checkpoint = torch.load(SAVE_PATH, map_location="cpu")
        st.session_state.policy.load_state_dict(checkpoint["policy_state"])
        st.session_state.optimizer.load_state_dict(checkpoint["optimizer_state"])
        st.session_state.episodes_done = checkpoint.get("episodes_done", 0)
        st.session_state.success = checkpoint.get("success", 0)
        st.session_state.fail = checkpoint.get("fail", 0)
        st.session_state.neutral = checkpoint.get("neutral", 0)
        st.success("📂 Policy loaded.")
    else:
        st.warning("No saved policy found.")

# ------------------- CONTROLS -------------------
st.title("🃏 Freebet Blackjack – Betting RL (objectif +3%)")

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("▶️ Start / Resume"):
        st.session_state.running = True
with col2:
    if st.button("⏸️ Pause"):
        st.session_state.running = False
with col3:
    if st.button("💾 Save policy"):
        save_policy()
with col4:
    if st.button("📂 Load policy"):
        load_policy()

st.metric("Episodes", st.session_state.episodes_done)
st.metric("✅ Success", st.session_state.success)
st.metric("❌ Fail", st.session_state.fail)
st.metric("➖ Neutral", st.session_state.neutral)

# ------------------- TRAINING LOOP -------------------
if st.session_state.running:
    for _ in range(BATCH_SIZE):
        total_reward, steps, outcome, logs = train_one_episode(
            st.session_state.env,
            st.session_state.policy,
            st.session_state.optimizer,
            target_gain=TARGET_GAIN,
        )
        st.session_state.episodes_done += 1
        st.session_state.logs.extend(logs)

        if outcome == "success":
            st.session_state.success += 1
        elif outcome == "fail":
            st.session_state.fail += 1
        else:
            st.session_state.neutral += 1

    # autosave
    if st.session_state.episodes_done % 10000 == 0:
        save_policy()

    st.rerun()

# ------------------- VISUALISATIONS -------------------
st.subheader("📊 Visualisations")

if len(st.session_state.logs) > 2000:
    df = pd.DataFrame(st.session_state.logs[-2000:])  # derniers points
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    sns.scatterplot(data=df, x="bankroll", y="bet", alpha=0.4, ax=axs[0,0])
    axs[0,0].set_title("Bet vs Bankroll")

    df["streak_signed"] = df["winstreak"] - df["loosestreak"]
    sns.scatterplot(data=df, x="streak_signed", y="bet", alpha=0.4, ax=axs[0,1])
    axs[0,1].set_title("Bet vs Streak (signed)")

    sns.scatterplot(data=df, x="prev_bet", y="bet", alpha=0.4, ax=axs[1,0])
    axs[1,0].set_title("Bet vs Previous Bet")

    sns.scatterplot(data=df, x="prev_gain", y="bet", alpha=0.4, ax=axs[1,1])
    axs[1,1].set_title("Bet vs Previous Gain")

    st.pyplot(fig)

    # Histogramme de résultats
    st.subheader("📈 Résultats des sessions")
    result_df = pd.DataFrame({
        "Outcome": ["Success", "Fail", "Neutral"],
        "Count": [st.session_state.success, st.session_state.fail, st.session_state.neutral],
    })
    st.bar_chart(result_df.set_index("Outcome"))
else:
    st.info("Pas assez de données pour les visualisations.")

# ------------------- CONSEILLER INTERACTIF -------------------
st.subheader("🎲 Conseiller interactif")

bankroll = st.number_input("Bankroll actuelle", min_value=0, value=100)
prev_bet = st.number_input("Mise précédente", min_value=0, value=0)
prev_gain = st.number_input("Gain précédent", min_value=-10, max_value=10, value=0)
winstreak = st.number_input("Winstreak", min_value=0, value=0)
loosestreak = st.number_input("Loosestreak", min_value=0, value=0)

if st.button("Obtenir une recommandation"):
    # On prépare l'état comme pendant l'entraînement
    state = [
        bankroll/500.0,
        prev_bet/5.0,
        prev_gain/5.0,
        min(winstreak,10)/10.0,
        min(loosestreak,10)/10.0,
    ]
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    # On passe par la policy
    probs = st.session_state.policy(state_tensor).detach().numpy()[0]
    best_action = probs.argmax()

    if best_action == 0:
        st.warning("💤 Stratégie: STOP (ne pas miser cette manche).")
    else:
        st.success(f"💰 Stratégie: miser {best_action} € (action {best_action}).")

    st.write("Distribution des choix (probas):", probs)
