import os
import pickle
import pandas as pd
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_PATH = os.path.join(DATA_DIR, "cbb_model.pkl")

st.set_page_config(page_title="NCAA Matchup Predictor", layout="wide")

st.title("🏀 NCAA Basketball Tactical Matchup & Spread Predictor")
st.caption("Opponent-Adjusted Four-Factor Predictive Model | Built with Python, DuckDB, & Scikit-Learn")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

artifacts = load_model()
clf = artifacts["clf"]
reg = artifacts["reg"]
feature_cols = artifacts["feature_cols"]
teams_dict = artifacts["teams_2024"]

team_list = sorted(list(teams_dict.keys()))

col1, col2 = st.columns(2)
with col1:
    team_a_name = st.selectbox("Select Team A (Home/Neutral):", team_list, index=team_list.index("Houston") if "Houston" in team_list else 0)
with col2:
    team_b_name = st.selectbox("Select Team B (Away/Neutral):", team_list, index=team_list.index("Purdue") if "Purdue" in team_list else 1)

if team_a_name == team_b_name:
    st.warning("Please select two distinct teams to project a matchup.")
else:
    team_a = teams_dict[team_a_name]
    team_b = teams_dict[team_b_name]

    # Calculate differentials
    net_a = team_a["adj_oe"] - team_a["adj_de"]
    net_b = team_b["adj_oe"] - team_b["adj_de"]
    diff_net_eff = net_a - net_b
    diff_efg = (team_a["efg_off"] - team_a["efg_def"]) - (team_b["efg_off"] - team_b["efg_def"])
    diff_to = -(team_a["to_rate"] - team_b["to_rate"])
    diff_reb = (team_a["or_rate"] - (100 - team_a["dr_rate"])) - (team_b["or_rate"] - (100 - team_b["dr_rate"]))
    diff_barthag = team_a["barthag"] - team_b["barthag"]

    features = pd.DataFrame([{
        "diff_net_eff": diff_net_eff,
        "diff_efg": diff_efg,
        "diff_to": diff_to,
        "diff_reb": diff_reb,
        "diff_barthag": diff_barthag
    }])[feature_cols]

    win_prob = clf.predict_proba(features)[0][1]
    predicted_spread = reg.predict(features)[0]

    st.markdown("---")
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.subheader("Matchup Forecast")
        st.metric(f"{team_a_name} Win Probability", f"{win_prob * 100:.1f}%")
        spread_display = f"{team_a_name} by {predicted_spread:.1f}" if predicted_spread > 0 else f"{team_b_name} by {-predicted_spread:.1f}"
        st.metric("Projected Spread", spread_display)

    with res_col2:
        st.subheader("Tactical Breakdown (Adjusted Metrics)")
        breakdown_df = pd.DataFrame({
            "Metric": ["Offensive Efficiency (AdjOE)", "Defensive Efficiency (AdjDE)", "Effective FG% Off", "Turnover %"],
            team_a_name: [f"{team_a['adj_oe']:.1f}", f"{team_a['adj_de']:.1f}", f"{team_a['efg_off']:.1f}%", f"{team_a['to_rate']:.1f}%"],
            team_b_name: [f"{team_b['adj_oe']:.1f}", f"{team_b['adj_de']:.1f}", f"{team_b['efg_off']:.1f}%", f"{team_b['to_rate']:.1f}%"]
        })
        st.table(breakdown_df)