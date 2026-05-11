import sqlite3
import pandas as pd
from datetime import datetime

def test():
    conn = sqlite3.connect('data/mlb_reentrenamiento.db')
    try:
        df = pd.read_sql("SELECT * FROM historico_real", conn)
        df["home_team_norm"] = df["home_team"].astype(str).str.strip().str.upper()
        df["away_team_norm"] = df["away_team"].astype(str).str.strip().str.upper()
        
        t_code_u = "BOS"
        fecha_limite = pd.to_datetime("2026-05-11")
        anio_partido = fecha_limite.year
        
        mask_season = (
            ((df["home_team_norm"].isin([t_code_u])) | (df["away_team_norm"].isin([t_code_u])))
            & (pd.to_datetime(df["fecha"], errors="coerce") < fecha_limite)
            & (pd.to_datetime(df["fecha"], errors="coerce").dt.year == anio_partido)
        )
        partidos = df[mask_season]
        
        # Write results to a file we can view
        with open("test_output.txt", "w") as f:
            f.write(f"Total in historico_real: {len(df)}\n")
            f.write(f"Matches for BOS in 2026 before {fecha_limite}: {len(partidos)}\n")
            if len(partidos) > 0:
                f.write(f"Latest match for BOS: {partidos.iloc[0]['fecha']}\n")
    except Exception as e:
        with open("test_output.txt", "w") as f:
            f.write(f"Error: {e}\n")

test()
