import os
import pickle
import time
import threading
from queue import Queue, Empty
from collections import Counter, defaultdict

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
st.set_page_config(page_title="Free Bet Blackjack RL", layout="wide")

# ============================== Constants ===============================
SAVE_PATH = "qtable.pkl"
DEFAULT_COMMIT_EVERY = 10_000
DEFAULT_EPISODES_PER_TICK = 50_000
DEFAULT_AUTOSAVE_SECS = 60

# ============================== Q-Table save/load ========================
def save_qtable(qtab: QTable, path: str = SAVE_PATH) -> None:
    data = {
        "q_by_tc": {tc: dict(qtab.q_by_tc[tc]) for tc in qtab.q_by_tc},
        "visits_by_tc": {tc: dict(qtab.visits_by_tc[tc]) for tc in qtab.visits_by_tc},
        "episodes": qtab.episodes,
        "epsilon": qtab.epsilon,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_qtable(qtab: QTable, path: str = SAVE_PATH) -> None:
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
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        pass

# ============================== Init Session State =======================
if "env" not in st.session_state:
    st.session_state.env = FreeBetEnv(num_decks=8, penetration=0.5, dealer_hits_soft_17=False)

if "qtab" not in st.session_state:
    st.session_state.qtab = QTable()
    load_qtable(st.session_state.qtab)

# Flags/threads & shared objects
if "running" not in st.session_state:
    st.session_state.running = False
if "train_thread" not in st.session_state:
    st.session_state.train_thread = None
if "stop_event" not in st.session_state:
    st.session_state.stop_event = threading.Event()
if "run_event" not in st.session_state:
    st.session_state.run_event = threading.Event()
if "delta_queue" not in st.session_state:
    st.session_state.delta_queue = Queue(maxsize=100)

# Metrics
if "episodes" not in st.session_state:
    st.session_state.episodes = 0
if "stats" not in st.session_state:
    st.session_state.stats = Counter()
if "returns_sum" not in st.session_state:
    st.session_state.returns_sum = 0.0
if "returns_by_tc" not in st.session_state:
    st.session_state.returns_by_tc = defaultdict(float)
if "episodes_by_tc" not in st.session_state:
    st.session_state.episodes_by_tc = defaultdict(int)

# Speed display
if "last_eps" not in st.session_state:
    st.session_state.last_eps = 0
if "last_ts" not in st.session_state:
    st.session_state.last_ts = time.time()

# Settings
if "settings" not in st.session_state:
    st.session_state.settings = {
        "alpha": 0.05,
        "epsilon": 0.2,
        "gamma": 1.0,
        "episodes_per_tick": DEFAULT_EPISODES_PER_TICK,
        "dealer_hits_soft_17": False,
        "num_decks": 8,
        "penetration": 0.5,
        "commit_every": DEFAULT_COMMIT_EVERY,
        "autosave_secs": DEFAULT_AUTOSAVE_SECS,
    }
else:
    st.session_state.settings.setdefault("episodes_per_tick", DEFAULT_EPISODES_PER_TICK)
    st.session_state.settings.setdefault("commit_every", DEFAULT_COMMIT_EVERY)
    st.session_state.settings.setdefault("autosave_secs", DEFAULT_AUTOSAVE_SECS)

# Autosave state
if "last_autosave_ts" not in st.session_state:
    st.session_state.last_autosave_ts = time.time()
if "episodes_at_last_save" not in st.session_state:
    st.session_state.episodes_at_last_save = st.session_state.episodes

# Config partagé lu par le worker (référence stable)
if "config" not in st.session_state:
    st.session_state.config = {
        "alpha": st.session_state.settings["alpha"],
        "epsilon": st.session_state.settings["epsilon"],
        "episodes_per_tick": st.session_state.settings["episodes_per_tick"],
        "commit_every": st.session_state.settings["commit_every"],
        "env": st.session_state.env,
        "qtab": st.session_state.qtab,
    }

# ============================== Utils ==========================
def _is_win(outcomes) -> bool:
    if isinstance(outcomes, str):
        return outcomes == "win"
    if isinstance(outcomes, dict):
        return bool(outcomes.get("win"))
    return False

def _is_push(outcomes) -> bool:
    if isinstance(outcomes, str):
        return outcomes == "push"
    if isinstance(outcomes, dict):
        return bool(outcomes.get("push"))
    return False

def apply_deltas_from_queue() -> int:
    """Applique les deltas produits par le worker dans la session (thread principal)."""
    applied = 0
    q = st.session_state.delta_queue
    while True:
        try:
            d = q.get_nowait()
        except Empty:
            break
        st.session_state.episodes += d["episodes"]
        st.session_state.returns_sum += d["returns_sum"]
        st.session_state.stats["win"] += d["wins"]
        st.session_state.stats["push"] += d["push"]
        st.session_state.stats["loss"] += d["loss"]
        for k, v in d["returns_by_tc"].items():
            st.session_state.returns_by_tc[k] += v
        for k, v in d["episodes_by_tc"].items():
            st.session_state.episodes_by_tc[k] += v
        applied += 1
    return applied

def maybe_autosave():
    now = time.time()
    if now - st.session_state.last_autosave_ts >= float(st.session_state.settings["autosave_secs"]):
        if st.session_state.episodes != st.session_state.episodes_at_last_save:
            save_qtable(st.session_state.qtab)
            st.session_state.last_autosave_ts = now
            st.session_state.episodes_at_last_save = st.session_state.episodes
            st.toast("Q-Table auto-sauvegardée 💾", icon="💾")

# ============================== Training Worker ==========================
def train_worker(config: dict, delta_queue: Queue, run_event: threading.Event, stop_event: threading.Event) -> None:
    # Accumulateurs locaux
    episodes_local = 0
    returns_local = 0.0
    win_local = 0
    push_local = 0
    loss_local = 0
    returns_by_tc_local = defaultdict(float)
    episodes_by_tc_local = defaultdict(int)
    commit_counter = 0

    def flush_locals():
        nonlocal episodes_local, returns_local, win_local, push_local, loss_local, returns_by_tc_local, episodes_by_tc_local, commit_counter
        if episodes_local or returns_by_tc_local:
            payload = {
                "episodes": episodes_local,
                "returns_sum": returns_local,
                "wins": win_local,
                "push": push_local,
                "loss": loss_local,
                "returns_by_tc": dict(returns_by_tc_local),
                "episodes_by_tc": dict(episodes_by_tc_local),
            }
            try:
                delta_queue.put(payload, timeout=0.2)
            except:
                pass
            episodes_local = 0
            returns_local = 0.0
            win_local = push_local = loss_local = 0
            returns_by_tc_local.clear()
            episodes_by_tc_local.clear()
            commit_counter = 0

    while not stop_event.is_set():
        if not run_event.is_set():
            time.sleep(0.05)
            continue

        eps = float(config["epsilon"])
        alpha = float(config["alpha"])
        env: FreeBetEnv = config["env"]
        qtab: QTable = config["qtab"]
        to_do = int(config["episodes_per_tick"]) or DEFAULT_EPISODES_PER_TICK
        commit_every = int(config["commit_every"]) or DEFAULT_COMMIT_EVERY

        for _ in range(to_do):
            # ---> pause/stop immédiats
            if stop_event.is_set() or not run_event.is_set():
                break

            tc = max(-5, min(5, env.shoe.true_count()))
            trans, rewards, outcomes = env.play_round(qtab, eps, tc)
            G = 0.0
            for r in rewards:
                G += r
            if trans:
                qtab.update_episode(trans, G, alpha, tc)

            episodes_local += 1
            returns_local += G
            episodes_by_tc_local[tc] += 1
            returns_by_tc_local[tc] += G

            if _is_win(outcomes):
                win_local += 1
            elif _is_push(outcomes):
                push_local += 1
            else:
                loss_local += 1

            commit_counter += 1
            if commit_counter >= commit_every:
                flush_locals()

        # Si on a été interrompu au milieu, on flush pour ne rien perdre
        if stop_event.is_set() or not run_event.is_set():
            flush_locals()

        time.sleep(0.001)

    # flush final si stop
    flush_locals()

# ============================== Sidebar (tuning) =========================
st.sidebar.header("⚙️ Paramètres")
episodes_per_tick = st.sidebar.number_input(
    "Episodes par tick",
    min_value=1_000, max_value=2_000_000, step=10_000,
    value=int(st.session_state.settings["episodes_per_tick"]),
)
commit_every = st.sidebar.number_input(
    "Commit toutes N itérations",
    min_value=100, max_value=200_000, step=100,
    value=int(st.session_state.settings["commit_every"]),
)
epsilon_input = st.sidebar.slider(
    "Epsilon (exploration)", 0.0, 1.0,
    float(st.session_state.settings["epsilon"]), 0.01,
)
alpha_input = st.sidebar.number_input(
    "Alpha (learning rate)",
    min_value=0.001, max_value=1.0, step=0.01,
    value=float(st.session_state.settings["alpha"]),
)
autosave_secs = st.sidebar.number_input(
    "Autosave (secondes)", min_value=10, max_value=3600, step=10,
    value=int(st.session_state.settings["autosave_secs"]),
)
h17_toggle = st.sidebar.checkbox(
    "Dealer hits soft 17 (H17)", value=bool(st.session_state.settings["dealer_hits_soft_17"])
)

# Appliquer les modifs runtime + maj du config partagé
st.session_state.settings["episodes_per_tick"] = int(episodes_per_tick)
st.session_state.settings["commit_every"] = int(commit_every)
st.session_state.settings["epsilon"] = float(epsilon_input)
st.session_state.settings["alpha"] = float(alpha_input)
st.session_state.settings["autosave_secs"] = int(autosave_secs)
st.session_state.settings["dealer_hits_soft_17"] = bool(h17_toggle)

# recrée l'env si H17/S17 change, et met à jour la ref dans config
if "last_h17" not in st.session_state:
    st.session_state.last_h17 = st.session_state.settings["dealer_hits_soft_17"]
if st.session_state.last_h17 != st.session_state.settings["dealer_hits_soft_17"]:
    st.session_state.env = FreeBetEnv(
        num_decks=st.session_state.settings["num_decks"],
        penetration=st.session_state.settings["penetration"],
        dealer_hits_soft_17=st.session_state.settings["dealer_hits_soft_17"],
    )
    st.session_state.last_h17 = st.session_state.settings["dealer_hits_soft_17"]
    st.toast("Règle croupier mise à jour (H17/S17)", icon="🔁")

# pousser les valeurs dans le config partagé (le worker lira ici en live)
st.session_state.config["alpha"] = st.session_state.settings["alpha"]
st.session_state.config["epsilon"] = st.session_state.settings["epsilon"]
st.session_state.config["episodes_per_tick"] = st.session_state.settings["episodes_per_tick"]
st.session_state.config["commit_every"] = st.session_state.settings["commit_every"]
st.session_state.config["env"] = st.session_state.env
st.session_state.config["qtab"] = st.session_state.qtab

# ============================== Controls ================================
c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 1, 1, 2, 2, 1, 1])

with c1:
    if st.button("▶️ Start / Resume", type="primary", key="btn_start"):
        st.session_state.running = True
        st.session_state.run_event.set()
        if st.session_state.train_thread is None or not st.session_state.train_thread.is_alive():
            st.session_state.stop_event = threading.Event()  # nouveau stop pour un nouveau thread
            st.session_state.train_thread = threading.Thread(
                target=train_worker,
                args=(st.session_state.config, st.session_state.delta_queue, st.session_state.run_event, st.session_state.stop_event),
                daemon=True,
            )
            st.session_state.train_thread.start()

with c2:
    if st.button("⏸️ Pause", key="btn_pause"):
        st.session_state.running = False
        st.session_state.run_event.clear()   # pause immédiate (le worker break dans la boucle)

with c7:
    if st.button("⏹ Stop", key="btn_stop"):
        st.session_state.running = False
        st.session_state.run_event.clear()
        st.session_state.stop_event.set()    # fin du thread
        th = st.session_state.train_thread
        if th and th.is_alive():
            th.join(timeout=0.5)
        st.session_state.train_thread = None
        # nouveau stop_event pour le prochain start
        st.session_state.stop_event = threading.Event()

with c3:
    if st.button("⏭️ Train one tick", key="btn_tick"):
        ss = st.session_state
        env = ss.env
        qtab = ss.qtab
        eps = ss.settings["epsilon"]
        alpha = ss.settings["alpha"]
        to_do = int(ss.settings["episodes_per_tick"]) or DEFAULT_EPISODES_PER_TICK

        episodes_local = 0
        returns_local = 0.0
        win_local = 0
        push_local = 0
        loss_local = 0
        returns_by_tc_local = defaultdict(float)
        episodes_by_tc_local = defaultdict(int)

        for _ in range(to_do):
            tc = max(-5, min(5, env.shoe.true_count()))
            trans, rewards, outcomes = env.play_round(qtab, eps, tc)
            G = 0.0
            for r in rewards:
                G += r
            if trans:
                qtab.update_episode(trans, G, alpha, tc)
            episodes_local += 1
            returns_local += G
            episodes_by_tc_local[tc] += 1
            returns_by_tc_local[tc] += G
            if _is_win(outcomes):
                win_local += 1
            elif _is_push(outcomes):
                push_local += 1
            else:
                loss_local += 1

        # passe par la queue pour rester homogène
        payload = {
            "episodes": episodes_local,
            "returns_sum": returns_local,
            "wins": win_local,
            "push": push_local,
            "loss": loss_local,
            "returns_by_tc": dict(returns_by_tc_local),
            "episodes_by_tc": dict(episodes_by_tc_local),
        }
        try:
            st.session_state.delta_queue.put(payload, timeout=0.2)
        except:
            pass

with c4:
    if st.button("💾 Save Q-Table", key="btn_save"):
        save_qtable(st.session_state.qtab)
        st.session_state.episodes_at_last_save = st.session_state.episodes
        st.session_state.last_autosave_ts = time.time()
        st.success("Q-Table sauvegardée !")

with c5:
    if st.button("📂 Load Q-Table", key="btn_load"):
        load_qtable(st.session_state.qtab)
        st.success("Q-Table chargée !")

with c6:
    if st.button("♻️ Reset stats", key="btn_reset"):
        st.session_state.episodes = 0
        st.session_state.stats = Counter()
        st.session_state.returns_sum = 0.0
        st.session_state.returns_by_tc = defaultdict(float)
        st.session_state.episodes_by_tc = defaultdict(int)
        st.toast("Stats remises à zéro", icon="✅")

# ============================== Intégrer les deltas du worker ============
apply_deltas_from_queue()
maybe_autosave()

# ============================== Speed gauge ==============================
now = time.time()
dt = max(1e-6, now - st.session_state.last_ts)
eps_delta = st.session_state.episodes - st.session_state.last_eps
eps_per_sec = eps_delta / dt
st.session_state.last_ts = now
st.session_state.last_eps = st.session_state.episodes
thread_alive = bool(st.session_state.train_thread and st.session_state.train_thread.is_alive())
state_emoji = "🟢" if (thread_alive and st.session_state.running) else ("⏸️" if thread_alive else "⚫")
st.caption(f"{state_emoji} ~{eps_per_sec:,.0f} épisodes/s • thread={'alive' if thread_alive else 'dead'} • running={'True' if st.session_state.running else 'False'}")

# ============================== Strategy Tables FIRST ====================
st.subheader("🧠 Learned Best Actions (multi-tables par True Count)")
TABLE_HEIGHT = 720
col_tc, col_pct = st.columns([3, 1])
with col_tc:
    tc_choice = st.slider("True Count (TC)", -5, 5, 0)
with col_pct:
    show_percentages = st.checkbox("% D/H/S/P", value=False, key="chk_show_pcts")

htab1, htab2, htab3 = st.tabs(["Hard totals", "Soft totals", "Pairs"])
with htab1:
    df_hard = build_table_hard(st.session_state.qtab, st.session_state.env, tc_choice, show_percentages)
    st.dataframe(style_actions(df_hard), width="stretch", height=TABLE_HEIGHT, hide_index=True)
with htab2:
    df_soft = build_table_soft(st.session_state.qtab, st.session_state.env, tc_choice, show_percentages)
    st.dataframe(style_actions(df_soft), width="stretch", height=TABLE_HEIGHT, hide_index=True)
with htab3:
    df_pairs = build_table_pairs(st.session_state.qtab, st.session_state.env, tc_choice, show_percentages)
    st.dataframe(style_actions(df_pairs), width="stretch", height=TABLE_HEIGHT, hide_index=True)

st.divider()

# ============================== Metrics (then EV by TC) ==================
st.subheader("📈 Training Metrics")
colA, colB, colC, colD, colE, colF = st.columns([2, 1, 1, 1, 2, 2])
with colA:
    st.metric("Episodes", f"{st.session_state.episodes:,}")
with colB:
    total_outcomes = sum(st.session_state.stats.values()) or 1
    wins = st.session_state.stats.get("win", 0)
    st.metric("Win %", f"{100 * wins / total_outcomes:.1f} %")
with colC:
    pushes = st.session_state.stats.get("push", 0)
    st.metric("Push %", f"{100 * pushes / total_outcomes:.1f} %")
with colD:
    losses = st.session_state.stats.get("loss", 0)
    st.metric("Loss %", f"{100 * losses / total_outcomes:.1f} %")
with colE:
    avg_return = st.session_state.returns_sum / max(1, st.session_state.episodes)
    st.metric("EV per round", f"{avg_return:.4f}")
with colF:
    st.metric("Sessions (sabots joués)", st.session_state.env.shoe.sessions_played)

# ============================== EV par True Count ========================
st.subheader("📊 EV par True Count")
data = []
for tc in sorted(st.session_state.returns_by_tc.keys()):
    n_eps = st.session_state.episodes_by_tc[tc]
    if n_eps <= 0:
        continue
    ev_tc = st.session_state.returns_by_tc[tc] / n_eps
    data.append({"TC": tc, "Episodes": n_eps, "EV per round": round(ev_tc, 4)})
if data:
    df_tc = pd.DataFrame(data)
    st.dataframe(df_tc, hide_index=True, width="stretch")
    st.line_chart(df_tc.set_index("TC")["EV per round"])
else:
    st.info("Pas encore de données par TC. Lance l'entraînement pour voir les métriques.")

st.divider()

# ============================== Règles rappel ============================
rule_17 = "Dealer tire sur soft 17 (H17)." if st.session_state.settings["dealer_hits_soft_17"] else "Dealer s'arrête sur 17 (S17)."
st.caption(
    f"""
Rappels de règles implémentées :
• Blackjack paie 3:2.  
• {rule_17}
• Pas de resplit ni redouble après split.  
• Un seul split par main (simplification).  
• Règle Charlie 6 cartes : 6 cartes ≤ 21 = gagné automatiquement.  
"""
)

# ============================== Auto-refresh ==============================
if st.session_state.running:
    time.sleep(0.5)
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()
