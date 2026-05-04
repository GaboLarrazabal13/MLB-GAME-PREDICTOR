import sqlite3
import os
import sys
import subprocess
import time
from datetime import datetime

# Ajuste de rutas para la nueva ubicación en src/mantenimiento/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DB_PATH = os.path.join(BASE_DIR, "data", "mlb_reentrenamiento.db")
SCRIPTS_DIR = os.path.join(BASE_DIR, "src")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_connection():
    return sqlite3.connect(DB_PATH)

def check_db_health():
    stats = {"missing_results": [], "fallback_predictions": [], "total_predictions": 0, "total_results": 0}
    try:
        if not os.path.exists(DB_PATH): return stats
        conn = get_connection()
        cursor = conn.cursor()
        stats["total_predictions"] = cursor.execute("SELECT COUNT(*) FROM predicciones_historico").fetchone()[0]
        stats["total_results"] = cursor.execute("SELECT COUNT(*) FROM historico_real").fetchone()[0]
        today = datetime.now().strftime("%Y-%m-%d")
        query_missing = """
            SELECT p.fecha, COUNT(*) FROM predicciones_historico p 
            LEFT JOIN historico_real hr ON p.fecha = hr.fecha 
                AND ( (p.home_team = hr.home_team AND p.away_team = hr.away_team) 
                      OR (p.home_team = hr.away_team AND p.away_team = hr.home_team) )
            WHERE hr.game_id IS NULL AND p.fecha < ?
            GROUP BY p.fecha ORDER BY p.fecha DESC
        """
        stats["missing_results"] = cursor.execute(query_missing, (today,)).fetchall()
        cursor.execute("PRAGMA table_info(predicciones_historico)")
        cols = [c[1] for c in cursor.fetchall()]
        if 'tipo' in cols:
            stats["fallback_predictions"] = cursor.execute("SELECT fecha, COUNT(*) FROM predicciones_historico WHERE tipo LIKE '%FALLBACK%' GROUP BY fecha ORDER BY fecha DESC").fetchall()
        conn.close()
    except Exception as e: print(f"❌ Error al analizar DB: {e}")
    return stats

def run_script(script_name, args=[]):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if script_name == "mlb_mass_healer.py": script_path = os.path.join(CURRENT_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"❌ Error: El script {script_name} no existe")
        return False
    cmd = [sys.executable, script_path] + args
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def push_db_to_production():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    db_relative = os.path.relpath(DB_PATH, BASE_DIR)
    print(f"\n🚀 SUBIR BASE DE DATOS A PRODUCCIÓN\n📦 Archivo: {db_relative}\n🕐 Timestamp: {now_str}")
    confirm = input("\n¿Confirmar y subir a GitHub? (s/n): ")
    if confirm.lower() != 's': return
    subprocess.run(["git", "add", db_relative], cwd=BASE_DIR)
    commit_msg = f"data: maintenance update - {now_str}"
    subprocess.run(["git", "commit", "-m", commit_msg], cwd=BASE_DIR)
    subprocess.run(["git", "push", "origin", "main"], cwd=BASE_DIR)
    print("\n✅ Base de datos subida exitosamente.")

def menu_principal():
    while True:
        clear_screen()
        print("="*60 + "\n⚾ MLB GAME PREDICTOR - MANTENIMIENTO ⚾\n" + "="*60)
        stats = check_db_health()
        print(f"\n📊 ESTADO: {stats['total_predictions']} preds / {stats['total_results']} reals")
        print("\nOPCIONES:\n1. 🔄 Recuperar resultados\n2. 🧙 Corregir FALLBACK\n3. 🧹 Optimizar DB\n4. 📋 Ver detalle\n5. 🚀 Subir a Producción\n6. 🛠️  Sanación Masiva\n7. 🚪 Salir")
        choice = input("\nSeleccione: ")
        if choice == '1':
            for f, _ in stats["missing_results"]: run_script("mlb_update_real_results.py", ["--fecha", f])
        elif choice == '2': run_script("mlb_manual_interface.py", ["--mode", "fix_fallback"])
        elif choice == '3': run_script("mlb_utils.py", ["--action", "optimize"])
        elif choice == '4': print(stats["missing_results"], stats["fallback_predictions"])
        elif choice == '5': push_db_to_production()
        elif choice == '6': run_script("mlb_mass_healer.py")
        elif choice == '7': break
        input("\nEnter para continuar...")

if __name__ == "__main__": menu_principal()
