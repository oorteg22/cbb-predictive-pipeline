# NCAA Basketball Predictive Modeling & Tactical Matchup Pipeline

An end-to-end sports analytics pipeline and interactive scouting dashboard predicting NCAA Men's Division I basketball outcomes using opponent-adjusted Four-Factor efficiency metrics.

## Architecture & Data Flow

1. **Ingestion & Data Pipeline (`src/ingest.py`):**
   - Retrieves NCAA efficiency and tempo metrics across 360+ Division I programs from Barttorvik historical archives.
   - Cleans and transforms raw records into structured relational tables stored in a local embedded **DuckDB** instance (`data/cbb.duckdb`).

2. **Feature Engineering & Statistical Modeling (`src/model.py`):**
   - Engineers possession-adjusted metric differentials: Net Rating ($\Delta \text{Net Eff}$), Effective Field Goal differential ($\Delta \text{eFG}$), Turnover Rate margin ($\Delta \text{TO}$), Rebounding differential ($\Delta \text{Reb}$), and Barthag power ratings.
   - Fits a **Logistic Regression** classifier for calibrated single-game win probabilities and a **Ridge Regression** model for projected margin of victory based on an empirical ~68-possession baseline.
   - Serializes trained model parameters and team metadata into `data/cbb_model.pkl`.

3. **Interactive Scouting Interface (`src/app.py`):**
   - Full-stack UI powered by **Streamlit** enabling scouts, coaches, and analysts to select head-to-head Division I programs.
   - Produces immediate win probability forecasts, projected point spreads, and tactical matchup breakdowns.

## Setup & Local Execution

```bash
# Clone the repository
git clone [https://github.com/oscarjortega/cbb-predictive-pipeline.git](https://github.com/oscarjortega/cbb-predictive-pipeline.git)
cd cbb-predictive-pipeline

# Configure virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run pipeline and launch interface
python src/ingest.py
python src/model.py
streamlit run src/app.py

---

### Step 5: Resume Project Entry

Add this entry directly under the **Technical Projects** section of your resume:

> **NCAA Basketball Predictive Modeling & Tactical Pipeline** | *Python, DuckDB, Scikit-Learn, Streamlit, Pandas*
> * Built an automated ingestion and storage pipeline processing season-long efficiency and tempo metrics across 360+ Division I teams into an embedded DuckDB database.
> * Engineered opponent-adjusted Four-Factor differential metrics ($\Delta \text{Net Eff}$, $\Delta \text{eFG}$, $\Delta \text{TO}$) to account for pace and possession variance.
> * Trained Logistic and Ridge regression models to forecast head-to-head win probabilities and margin spreads across simulated multi-game schedules.
> * Developed an interactive scouting dashboard in Streamlit allowing staff to simulate matchup spreads and identify tactical mismatches across offensive and defensive ratings.

---

### Step 6: Push to GitHub