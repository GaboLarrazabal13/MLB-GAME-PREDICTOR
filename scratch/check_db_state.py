import sqlite3
import pandas as pd

def check_db():
    conn = sqlite3.connect('data/mlb_reentrenamiento.db')
    print("--- TABLES ---")
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    for t in tables:
        print(f"Table: {t[0]}")
    
    print("\n--- PREDICTIONS MAY 1ST ---")
    try:
        df_pred_may1 = pd.read_sql("SELECT fecha, home_team, away_team, prediccion FROM predicciones_historico WHERE fecha = '2026-05-01'", conn)
        print(df_pred_may1)
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- RECENT PREDICTIONS ---")
    try:
        df_pred = pd.read_sql("SELECT fecha, home_team, away_team, prediccion FROM predicciones_historico ORDER BY fecha DESC LIMIT 5", conn)
        print(df_pred)
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- RECENT REAL RESULTS ---")
    try:
        df_real = pd.read_sql("SELECT fecha, home_team, away_team, score_home, score_away FROM historico_real ORDER BY fecha DESC LIMIT 5", conn)
        print(df_real)
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- SYNC CONTROL ---")
    try:
        df_sync = pd.read_sql("SELECT * FROM sync_control", conn)
        print(df_sync)
    except Exception as e:
        print(f"Error: {e}")

    conn.close()

if __name__ == '__main__':
    check_db()
