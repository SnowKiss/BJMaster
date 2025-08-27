import pandas as pd
import numpy as np

# Actions doivent être dans le même ordre que dans env.py !
ACTIONS = ["H", "S", "D", "P"]

def style_actions(df: pd.DataFrame):
    color_map = {
        "H": "background-color: red; color: white;",
        "D": "background-color: yellow; color: black;",
        "S": "background-color: blue; color: white;",
        "P": "background-color: orange; color: black;",
    }
    def highlight(val):
        if val in color_map:
            return color_map[val]
        return ""
    best_cols = [c for c in df.columns if c.endswith("(Best)")]
    return df.style.map(highlight, subset=best_cols)


# ------------------- Agrégation Q-values -------------------
def aggregate_qvalues(qtab, states):
    """Moyenne pondérée des Q-values selon les visites, avec correction EV=-inf si action illégale."""
    q_sum = np.zeros(4, dtype=float)
    total = 0
    for s in states:
        v = qtab.visits[s].sum()
        q_sum += qtab.q[s] * max(1, v)
        total += max(1, v)

    q_avg = q_sum / max(1, total)

    # ⚠️ Correction : si pair=0 → action P illégale → EV = -inf
    if all(s[2] == 0 for s in states):  # s[2] = pair_rank
        q_avg[3] = float("-inf")

    return q_avg


# ------------------- Hard totals -------------------
def build_table_hard(qtab, env, show_ev=True):
    rows = []
    for total in range(5, 22):
        row = {"Player": str(total)}
        for dealer_card in range(2, 12):
            states = [(total, 0, 0, dealer_card, fa) for fa in (0, 1)]
            qvals = aggregate_qvalues(qtab, states)

            best_action = ACTIONS[int(np.argmax(qvals))]

            row[f"{dealer_card} (Best)"] = best_action
            if show_ev:
                row[f"{dealer_card} H_EV"] = f"{qvals[0]:.3f}"
                row[f"{dealer_card} S_EV"] = f"{qvals[1]:.3f}"
                row[f"{dealer_card} D_EV"] = f"{qvals[2]:.3f}"
                # pas de P_EV car toujours -inf
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------- Soft totals -------------------
def build_table_soft(qtab, env, show_ev=True):
    rows = []
    for val in range(2, 10):  # A2 .. A9
        total = 11 + val
        row = {"Player": f"A{val}"}
        for dealer_card in range(2, 12):
            states = [(total, 1, 0, dealer_card, fa) for fa in (0, 1)]
            qvals = aggregate_qvalues(qtab, states)

            best_action = ACTIONS[int(np.argmax(qvals))]

            row[f"{dealer_card} (Best)"] = best_action
            if show_ev:
                row[f"{dealer_card} H_EV"] = f"{qvals[0]:.3f}"
                row[f"{dealer_card} S_EV"] = f"{qvals[1]:.3f}"
                row[f"{dealer_card} D_EV"] = f"{qvals[2]:.3f}"
                # pas de P_EV car toujours -inf
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------- Pairs -------------------
def build_table_pairs(qtab, env, show_ev=True):
    rows = []
    for c in range(2, 12):  # paires 2..A
        label = f"{c},{c}" if c != 11 else "A,A"
        row = {"Player": label}
        for dealer_card in range(2, 12):
            states = [(c*2, 0, c, dealer_card, fa) for fa in (0, 1)]
            qvals = aggregate_qvalues(qtab, states)

            best_action = ACTIONS[int(np.argmax(qvals))]

            row[f"{dealer_card} (Best)"] = best_action
            if show_ev:
                row[f"{dealer_card} H_EV"] = f"{qvals[0]:.3f}"
                row[f"{dealer_card} S_EV"] = f"{qvals[1]:.3f}"
                row[f"{dealer_card} D_EV"] = f"{qvals[2]:.3f}"
                row[f"{dealer_card} P_EV"] = f"{qvals[3]:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)
