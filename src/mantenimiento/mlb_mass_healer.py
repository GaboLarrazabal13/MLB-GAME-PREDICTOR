import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
SCRIPTS_DIR = os.path.join(BASE_DIR, "src")
DB_PATH = os.path.join(BASE_DIR, 'data', 'mlb_reentrenamiento.db')

def get_gaps():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    query = """
    SELECT r.fecha, COUNT(DISTINCT r.home_team) as total_juegos, COUNT(DISTINCT p.home_team) as total_preds
    FROM historico_real r
    LEFT JOIN predicciones_historico p ON r.fecha = p.fecha AND r.home_team = p.home_team
    WHERE r.fecha >= '2026-03-20' AND r.fecha <= date('now')
    GROUP BY r.fecha
    HAVING total_preds < total_juegos * 0.8 OR total_preds = 0
    ORDER BY r.fecha DESC
    """
    c.execute(query)
    gaps = c.fetchall()
    conn.close()
    return gaps

def heal_date(date):
    print(f"🛠️  SANANDO: {date}")
    env = os.environ.copy()
    env['TARGET_DATE'] = date
    subprocess.run([sys.executable, os.path.join(SCRIPTS_DIR, 'mlb_daily_scraper.py')], env=env)
    subprocess.run([sys.executable, os.path.join(SCRIPTS_DIR, 'mlb_predict_engine.py')], env=env)

if __name__ == "__main__":
    gaps = get_gaps()
    for date, _, _ in gaps: heal_date(date)
    print("✅ Sanación completada")
