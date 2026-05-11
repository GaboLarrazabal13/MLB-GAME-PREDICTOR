import sqlite3
import pandas as pd
from mlb_config import DB_PATH

try:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT count(*) as count, max(fecha) as mx, min(fecha) as mn FROM historico_real", conn)
        print("Data in historico_real:", df.to_dict('records'))
        
        df2 = pd.read_sql("SELECT * FROM historico_real LIMIT 5", conn)
        print("Columns in historico_real:", df2.columns.tolist())
except Exception as e:
    print("Error:", e)
