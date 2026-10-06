import os
import io
import duckdb
import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.path.join(DATA_DIR, "cbb.duckdb")

def fetch_and_store_data(seasons=[2023, 2024]):
    print("--- Starting Clean NCAA Data Ingestion ---")
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = duckdb.connect(DB_PATH)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    
    for year in seasons:
        print(f"Fetching team ratings for {year}...")
        url = f"https://barttorvik.com/{year}_team_results.csv"
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code != 200:
                print(f"  HTTP error {response.status_code} fetching {url}")
                continue
            
            csv_data = io.StringIO(response.text)
            df = pd.read_csv(csv_data, header=None)
            
            # Drop repeated header lines
            df = df[df.iloc[:, 1].astype(str).str.lower() != "team"].copy()

            # Exact Barttorvik CSV metric mapping
            clean_df = pd.DataFrame()
            clean_df["team"] = df.iloc[:, 1].astype(str).str.strip()
            clean_df["conf"] = df.iloc[:, 2].astype(str).str.strip()
            clean_df["adj_oe"] = pd.to_numeric(df.iloc[:, 4], errors="coerce")
            clean_df["adj_de"] = pd.to_numeric(df.iloc[:, 6], errors="coerce")
            clean_df["barthag"] = pd.to_numeric(df.iloc[:, 5], errors="coerce")
            clean_df["efg_off"] = pd.to_numeric(df.iloc[:, 8], errors="coerce")
            clean_df["efg_def"] = pd.to_numeric(df.iloc[:, 9], errors="coerce")
            clean_df["to_rate"] = pd.to_numeric(df.iloc[:, 10], errors="coerce")
            clean_df["or_rate"] = pd.to_numeric(df.iloc[:, 11], errors="coerce")
            clean_df["dr_rate"] = pd.to_numeric(df.iloc[:, 12], errors="coerce")
            clean_df["season"] = int(year)
            
            clean_df = clean_df.dropna().reset_index(drop=True)
            
            conn.register("temp_df", clean_df)
            conn.execute(f"CREATE OR REPLACE TABLE ratings_{year} AS SELECT * FROM temp_df")
            print(f"  Loaded {len(clean_df)} clean teams for season {year} into DuckDB.")
            
        except Exception as e:
            print(f"  Error processing season {year}: {e}")
            
    conn.close()
    print(f"Ingestion complete. Database verified at: {DB_PATH}")

if __name__ == "__main__":
    fetch_and_store_data()