import os
import duckdb
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, log_loss

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.path.join(DATA_DIR, "cbb.duckdb")
MODEL_PATH = os.path.join(DATA_DIR, "cbb_model.pkl")

def train_basketball_model():
    print("--- Training Basketball Game Prediction Model ---")
    conn = duckdb.connect(DB_PATH)
    df_2023 = conn.execute("SELECT * FROM ratings_2023").df()
    df_2024 = conn.execute("SELECT * FROM ratings_2024").df()
    conn.close()

    df = pd.concat([df_2023, df_2024], ignore_index=True)
    print(f"Loaded {len(df)} total team-season records from DuckDB.")

    matchups = []
    np.random.seed(42)

    for season, group in df.groupby("season"):
        teams = group.to_dict("records")
        for i in range(len(teams)):
            for j in range(i + 1, min(i + 40, len(teams))):
                team_a = teams[i]
                team_b = teams[j]
                
                # Net efficiency differential
                net_a = team_a["adj_oe"] - team_a["adj_de"]
                net_b = team_b["adj_oe"] - team_b["adj_de"]
                diff_net_eff = net_a - net_b
                
                # Four-factor differentials
                diff_efg = (team_a["efg_off"] - team_a["efg_def"]) - (team_b["efg_off"] - team_b["efg_def"])
                diff_to = -(team_a["to_rate"] - team_b["to_rate"])
                diff_reb = (team_a["or_rate"] - (100 - team_a["dr_rate"])) - (team_b["or_rate"] - (100 - team_b["dr_rate"]))
                diff_barthag = team_a["barthag"] - team_b["barthag"]

                # Expected margin using standard NCAA pace (~68 possessions)
                expected_margin = (diff_net_eff / 100.0) * 68.0
                
                # Single-game variance simulation
                noise = np.random.normal(0, 10.0)
                simulated_margin = expected_margin + noise
                team_a_won = 1 if simulated_margin > 0 else 0

                matchups.append({
                    "diff_net_eff": diff_net_eff,
                    "diff_efg": diff_efg,
                    "diff_to": diff_to,
                    "diff_reb": diff_reb,
                    "diff_barthag": diff_barthag,
                    "margin": expected_margin,
                    "team_a_won": team_a_won
                })

    data = pd.DataFrame(matchups)
    feature_cols = ["diff_net_eff", "diff_efg", "diff_to", "diff_reb", "diff_barthag"]
    X = data[feature_cols]
    y_win = data["team_a_won"]
    y_spread = data["margin"]

    clf = LogisticRegression()
    clf.fit(X, y_win)

    reg = Ridge(alpha=1.0)
    reg.fit(X, y_spread)

    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    acc = accuracy_score(y_win, preds)
    loss = log_loss(y_win, probs)

    print(f"Model Training Results:")
    print(f"  Classification Accuracy: {acc * 100:.2f}%")
    print(f"  Log-Loss: {loss:.4f}")

    artifacts = {
        "clf": clf,
        "reg": reg,
        "feature_cols": feature_cols,
        "teams_2024": df_2024.set_index("team").to_dict("index")
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"\nModel artifacts saved successfully to: {MODEL_PATH}")

if __name__ == "__main__":
    train_basketball_model()