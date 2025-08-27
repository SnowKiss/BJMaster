import pandas as pd

class BasicStrategy:
    def __init__(self, hard_csv, soft_csv, pairs_csv):
        self.hard = pd.read_csv(hard_csv)
        self.soft = pd.read_csv(soft_csv)
        self.pairs = pd.read_csv(pairs_csv)

        # Normalisation des colonnes : si on a "A (Best)" on garde, sinon on le crée
        for table in [self.hard, self.soft, self.pairs]:
            cols = table.columns
            if "A (Best)" not in cols and "11 (Best)" in cols:
                table.rename(columns={"11 (Best)": "A (Best)"}, inplace=True)

    def get_action(self, total, soft, pair, dealer_up):
        """
        Retourne l’action ('H','S','D','P') d’après les CSV.
        - total : valeur totale main joueur
        - soft  : bool (main soft)
        - pair  : bool (main paire)
        - dealer_up : carte visible du croupier (2-11, où 11 = As)
        """

        # Nom de colonne
        dealer_col = "A (Best)" if dealer_up == 11 else f"{dealer_up} (Best)"

        # Choix de la bonne table
        if pair:
            table = self.pairs
        elif soft:
            table = self.soft
        else:
            table = self.hard

        # Filtrer la ligne correspondant au joueur
        if "Player" not in table.columns:
            return "S"  # fallback
        row = table[table["Player"] == total]

        if row.empty or dealer_col not in row:
            return "S"  # fallback si pas trouvé

        action = row[dealer_col].values[0]

        # Nettoyage éventuel : si la cellule contient du NaN ou vide
        if pd.isna(action):
            return "S"

        return str(action).strip().upper()
