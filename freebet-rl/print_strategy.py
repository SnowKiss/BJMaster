# print_strategy.py
import argparse
import pickle
import os
import numpy as np
import pandas as pd
from collections import defaultdict

from freebet.env import FreeBetEnv
from freebet.rl.qtable import QTable
from freebet.ui.tables import (
    build_table_hard,
    build_table_soft,
    build_table_pairs,
)

def load_qtable(path: str) -> QTable:
    qtab = QTable()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return qtab
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        qtab.q_by_tc = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))
        for tc, qdict in data.get("q_by_tc", {}).items():
            qtab.q_by_tc[tc] = defaultdict(lambda: np.zeros(4), qdict)
        qtab.visits_by_tc = defaultdict(lambda: defaultdict(lambda: np.zeros(4, dtype=int)))
        for tc, vdict in data.get("visits_by_tc", {}).items():
            qtab.visits_by_tc[tc] = defaultdict(lambda: np.zeros(4, dtype=int), vdict)
        qtab.episodes = int(data.get("episodes", 0))
        qtab.epsilon = float(data.get("epsilon", qtab.epsilon))
    except Exception:
        pass
    return qtab

def main():
    ap = argparse.ArgumentParser(description="Afficher les tables de stratégie apprises.")
    ap.add_argument("--save-path", default="qtable.pkl")
    ap.add_argument("--tc", type=int, default=0, help="True Count à afficher (-5..5)")
    ap.add_argument("--h17", action="store_true", help="Dealer hits soft 17")
    ap.add_argument("--to-csv", default="", help="Chemin CSV pour exporter (vide = pas d'export)")
    args = ap.parse_args()

    qtab = load_qtable(args.save_path)
    env = FreeBetEnv(num_decks=8, penetration=0.5, dealer_hits_soft_17=bool(args.h17))

    df_hard = build_table_hard(qtab, env, args.tc, show_percent=False)
    df_soft = build_table_soft(qtab, env, args.tc, show_percent=False)
    df_pairs = build_table_pairs(qtab, env, args.tc, show_percent=False)

    print(f"\n=== HARD totals (TC={args.tc}) ===")
    print(df_hard.to_string(index=False))
    print(f"\n=== SOFT totals (TC={args.tc}) ===")
    print(df_soft.to_string(index=False))
    print(f"\n=== PAIRS (TC={args.tc}) ===")
    print(df_pairs.to_string(index=False))

    if args.to_csv:
        with open(args.to_csv, "w", encoding="utf-8") as f:
            f.write("# HARD\n")
            df_hard.to_csv(f, index=False)
            f.write("\n# SOFT\n")
            df_soft.to_csv(f, index=False)
            f.write("\n# PAIRS\n")
            df_pairs.to_csv(f, index=False)
        print(f"\n✅ Exporté vers {args.to_csv}")

if __name__ == "__main__":
    main()
