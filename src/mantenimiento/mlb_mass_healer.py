import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta

# Configuración de rutas robusta para src/mantenimiento/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
SCRIPTS_DIR = os.path.join(BASE_DIR, "src")

DB_PATH = os.path.join(BASE_DIR, 'data', 'mlb_reentrenamiento.db')
START_DATE = '2026-03-20'  # Inicio de temporada aproximado
END_DATE = datetime.now().strftime('%Y-%m-%d')

def get_gaps():
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: DB no encontrada en {DB_PATH}")
        return []
        
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Buscar fechas con resultados reales pero sin predicciones (o muy pocas)
    query = """
    SELECT r.fecha, COUNT(DISTINCT r.home_team) as total_juegos, COUNT(DISTINCT p.home_team) as total_preds
    FROM historico_real r
    LEFT JOIN predicciones_historico p ON r.fecha = p.fecha AND r.home_team = p.home_team
    WHERE r.fecha >= ? AND r.fecha <= ?
    GROUP BY r.fecha
    HAVING total_preds < total_juegos * 0.8 OR total_preds = 0
    ORDER BY r.fecha DESC
    """
    c.execute(query, (START_DATE, END_DATE))
    gaps = c.fetchall()
    conn.close()
    return gaps

def heal_date(date):
    print(f"\n" + "="*50)
    print(f"🛠️  SANANDO FECHA: {date}")
    print("="*50)
    
    env = os.environ.copy()
    env['TARGET_DATE'] = date
    
    # 1. Asegurar que tenemos la cartelera (scraper)
    print(f"--- Paso 1: Verificando/Scrapeando cartelera para {date} ---")
    subprocess.run([sys.executable, os.path.join(SCRIPTS_DIR, 'mlb_daily_scraper.py')], env=env)
    
    # 2. Generar predicciones de alta calidad (engine)
    print(f"--- Paso 2: Generando predicciones de calidad para {date} ---")
    subprocess.run([sys.executable, os.path.join(SCRIPTS_DIR, 'mlb_predict_engine.py')], env=env)

if __name__ == "__main__":
    gaps = get_gaps()
    
    if not gaps:
        print("✅ No se encontraron huecos de predicciones en la temporada 2026.")
        sys.exit(0)
    
    print(f"Found {len(gaps)} dates with gaps. Starting healing process...")
    
    for date, total_juegos, total_preds in gaps:
        print(f"📅 {date}: Faltan predicciones ({total_preds}/{total_juegos} encontradas)")
        heal_date(date)
        
    print("\n✅ Proceso de sanación masiva completado.")
