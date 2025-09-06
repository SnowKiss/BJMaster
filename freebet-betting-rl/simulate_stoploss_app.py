import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RESULTS_FILE = os.path.join(os.getcwd(), "stoploss_results.csv")
MAX_BANKROLL = 10000  # doit matcher ton worker
sns.set_theme(style="whitegrid")

st.title("📊 Analyse des meilleures stratégies Stop-Loss (+3%)")

if os.path.exists(RESULTS_FILE):
    results_df = pd.read_csv(RESULTS_FILE)
    max_done = results_df["bankroll"].max()
else:
    results_df = pd.DataFrame()
    max_done = 0

st.metric("📊 Bankroll max simulée", max_done)
st.progress(max_done / MAX_BANKROLL)

# Choix de la bankroll
selected_bankroll = st.number_input(
    "Choisir une bankroll à analyser",
    min_value=1,
    value=int(max_done) if max_done > 0 else 1,
    step=1
)

# Choix du mode
mode_choice = st.selectbox("Mode de mise", ["flat", "percent"])

if not results_df.empty and selected_bankroll in results_df["bankroll"].values:
    df_b = results_df[(results_df["bankroll"] == selected_bankroll) & (results_df["mode"] == mode_choice)].copy()
    if df_b.empty:
        st.warning(f"Aucune stratégie {mode_choice} sauvegardée pour bankroll={selected_bankroll}.")
    else:
        st.subheader("🏆 Top stratégies")
        st.dataframe(df_b.sort_values("score", ascending=False))

        # Heatmap seulement si plusieurs combinaisons
        if df_b["bet"].nunique() > 1 and df_b["stop_loss"].nunique() > 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(
                df_b.pivot_table(index="bet", columns="stop_loss", values="success_rate", aggfunc="mean"),
                annot=True, fmt=".2f", cmap="YlGnBu", ax=ax
            )
            ax.set_title(f"Taux de succès ({mode_choice}) – bankroll={selected_bankroll}")
            st.pyplot(fig)
else:
    st.info("Pas encore de résultats pour cette bankroll.")
