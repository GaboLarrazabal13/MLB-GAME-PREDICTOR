import sqlite3
import pandas as pd
from mlb_config import DB_PATH

try:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM historico_real ORDER BY fecha DESC LIMIT 10", conn)
    print("Últimos 10 juegos en historico_real:")
    print(df[['fecha', 'home_team', 'away_team', 'score_home', 'score_away', 'ganador']])
    
    df_2026 = pd.read_sql("SELECT COUNT(*) as count FROM historico_real WHERE fecha LIKE '2026%'", conn)
    print(f"\nJuegos en 2026: {df_2026.iloc[0]['count']}")
    
    conn.close()
except Exception as e:
    print(f"Error: {e}")
