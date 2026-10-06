import os
import pickle
import pandas as pd
import streamlit as st
import duckdb

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


    st.markdown("---")
    st.subheader("📊 Relational SQL Analytics Layer (DuckDB)")

    query_option = st.selectbox(
        "Select an Analytical SQL Query to Run:",
        [
            "Conference Power Index (Aggregations & Grouping)",
            "Elite Championship Tier (Barthag >= 0.900)",
            "Conference-Relative Efficiency (Window Functions)",
            "Top Defensive Havoc Teams (Defensive Filters)"
        ]
    )

    conn = duckdb.connect(DATA_DIR + "/cbb.duckdb")

    if query_option == "Conference Power Index (Aggregations & Grouping)":
        sql = """
        SELECT 
            conf,
            COUNT(team) AS total_programs,
            ROUND(AVG(adj_oe), 2) AS avg_offensive_rating,
            ROUND(AVG(adj_de), 2) AS avg_defensive_rating,
            ROUND(AVG(adj_oe - adj_de), 2) AS avg_net_rating,
            ROUND(AVG(barthag), 3) AS avg_barthag_power
        FROM ratings_2024
        GROUP BY conf
        HAVING COUNT(team) >= 8
        ORDER BY avg_net_rating DESC;
        """
    elif query_option == "Elite Championship Tier (Barthag >= 0.900)":
        sql = """
        SELECT 
            team, conf,
            ROUND(adj_oe, 1) AS adj_oe,
            ROUND(adj_de, 1) AS adj_de,
            ROUND(adj_oe - adj_de, 1) AS net_efficiency,
            ROUND(barthag, 4) AS barthag,
            ROUND(efg_off, 1) AS efg_off_pct,
            ROUND(to_rate, 1) AS to_rate_pct
        FROM ratings_2024
        WHERE barthag >= 0.900
        ORDER BY barthag DESC;
        """
    elif query_option == "Conference-Relative Efficiency (Window Functions)":
        sql = """
        WITH conference_baselines AS (
            SELECT 
                team, conf, adj_oe, adj_de,
                (adj_oe - adj_de) AS net_efficiency,
                barthag,
                AVG(adj_oe) OVER(PARTITION BY conf) AS conf_avg_oe,
                AVG(adj_de) OVER(PARTITION BY conf) AS conf_avg_de,
                DENSE_RANK() OVER(PARTITION BY conf ORDER BY barthag DESC) AS conf_rank,
                DENSE_RANK() OVER(ORDER BY barthag DESC) AS national_rank
            FROM ratings_2024
        )
        SELECT 
            team, conf, conf_rank, national_rank,
            ROUND(net_efficiency, 2) AS net_eff,
            ROUND(adj_oe - conf_avg_oe, 2) AS oe_above_conf_avg,
            ROUND(conf_avg_de - adj_de, 2) AS de_better_than_conf_avg,
            ROUND(barthag, 3) AS barthag
        FROM conference_baselines
        WHERE conf IN ('B12', 'B10', 'SEC', 'BE', 'ACC') AND conf_rank <= 3
        ORDER BY national_rank ASC;
        """
    else:
        sql = """
        SELECT 
            team, conf,
            ROUND(efg_def, 1) AS opp_efg_pct,
            ROUND(dr_rate, 1) AS def_rebound_pct,
            ROUND(adj_de, 1) AS adj_de,
            DENSE_RANK() OVER(ORDER BY adj_de ASC) AS def_rank
        FROM ratings_2024
        WHERE efg_def < 48.0 AND dr_rate >= 75.0
        ORDER BY adj_de ASC
        LIMIT 15;
        """

    result_df = conn.execute(sql).df()
    conn.close()

    with st.expander("Show SQL Code", expanded=False):
        st.code(sql, language="sql")

    st.dataframe(result_df, use_container_width=True)