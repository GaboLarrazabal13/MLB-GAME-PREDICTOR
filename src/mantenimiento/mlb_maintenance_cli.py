import sqlite3
import os
import sys
import subprocess
import time
from datetime import datetime

# Ajuste de rutas para la nueva ubicación en src/mantenimiento/
# El script está en src/mantenimiento/, así que BASE_DIR es dos niveles arriba
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DB_PATH = os.path.join(BASE_DIR, "data", "mlb_reentrenamiento.db")
SCRIPTS_DIR = os.path.join(BASE_DIR, "src")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_connection():
    return sqlite3.connect(DB_PATH)

def check_db_health():
    """Analiza la salud de la base de datos y retorna estadísticas de problemas."""
    stats = {
        "missing_results": [],
        "fallback_predictions": [],
        "total_predictions": 0,
        "total_results": 0
    }
    
    try:
        if not os.path.exists(DB_PATH):
            return stats
            
        conn = get_connection()
        cursor = conn.cursor()
        
        # 1. Total de registros
        stats["total_predictions"] = cursor.execute("SELECT COUNT(*) FROM predicciones_historico").fetchone()[0]
        stats["total_results"] = cursor.execute("SELECT COUNT(*) FROM historico_real").fetchone()[0]
        
        # 2. Predicciones sin resultados validados (Excluyendo hoy)
        today = datetime.now().strftime("%Y-%m-%d")
        query_missing = """
            SELECT p.fecha, COUNT(*) 
            FROM predicciones_historico p 
            LEFT JOIN historico_real hr ON p.fecha = hr.fecha 
                AND ( (p.home_team = hr.home_team AND p.away_team = hr.away_team) 
                      OR (p.home_team = hr.away_team AND p.away_team = hr.home_team) )
            WHERE hr.game_id IS NULL AND p.fecha < ?
            GROUP BY p.fecha 
            ORDER BY p.fecha DESC
        """
        stats["missing_results"] = cursor.execute(query_missing, (today,)).fetchall()
        
        # 3. Predicciones con calidad FALLBACK
        cursor.execute("PRAGMA table_info(predicciones_historico)")
        cols = [c[1] for c in cursor.fetchall()]
        if 'tipo' in cols:
            query_fallback = """
                SELECT fecha, COUNT(*) 
                FROM predicciones_historico 
                WHERE tipo LIKE '%FALLBACK%'
                GROUP BY fecha 
                ORDER BY fecha DESC
            """
            stats["fallback_predictions"] = cursor.execute(query_fallback).fetchall()
        
        conn.close()
    except Exception as e:
        print(f"❌ Error al analizar DB: {e}")
        
    return stats

def run_script(script_name, args=[]):
    """Ejecuta un script de la carpeta src con argumentos."""
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    
    # Caso especial para el healer que ahora también está en mantenimiento/
    if script_name == "mlb_mass_healer.py":
        script_path = os.path.join(CURRENT_DIR, script_name)

    if not os.path.exists(script_path):
        print(f"❌ Error: El script {script_name} no existe en {os.path.dirname(script_path)}")
        return False

    cmd = [sys.executable, script_path] + args
    print(f"\n🚀 Ejecutando: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al ejecutar {script_name}: {e}")
        return False

def push_db_to_production():
    """Hace commit y push de la base de datos a producción (GitHub)."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    db_relative = os.path.relpath(DB_PATH, BASE_DIR)

    print(f"\n{'='*60}")
    print("🚀 SUBIR BASE DE DATOS A PRODUCCIÓN")
    print(f"{'='*60}")
    print(f"\n📦 Se hará commit del archivo:")
    print(f"   {db_relative}")
    print(f"\n🕐 Timestamp: {now_str}")
    confirm = input("\n¿Confirmar y subir a GitHub? (s/n): ")
    if confirm.lower() != 's':
        print("⏭️  Operación cancelada.")
        return

    print("\n⏳ Ejecutando git add...")
    result_add = subprocess.run(
        ["git", "add", db_relative],
        cwd=BASE_DIR, capture_output=True, text=True
    )
    if result_add.returncode != 0:
        print(f"❌ Error en git add: {result_add.stderr}")
        return

    # Verificar si hay algo que commitear
    result_status = subprocess.run(
        ["git", "status", "--porcelain", db_relative],
        cwd=BASE_DIR, capture_output=True, text=True
    )
    if not result_status.stdout.strip():
        print("ℹ️  La base de datos no tiene cambios respecto al último commit. Nada que subir.")
        return

    commit_msg = f"data: maintenance update - {now_str}"
    print(f"\n⏳ Haciendo commit: '{commit_msg}'")
    result_commit = subprocess.run(
        ["git", "commit", "-m", commit_msg],
        cwd=BASE_DIR, capture_output=True, text=True
    )
    if result_commit.returncode != 0:
        print(f"❌ Error en git commit: {result_commit.stderr}")
        return
    print(result_commit.stdout.strip())

    print("\n⏳ Haciendo push a origin/main...")
    result_push = subprocess.run(
        ["git", "push", "origin", "main"],
        cwd=BASE_DIR, capture_output=True, text=True
    )
    if result_push.returncode != 0:
        print(f"❌ Error en git push: {result_push.stderr}")
        return

    print("\n✅ Base de datos subida a producción exitosamente.")

def menu_principal():
    while True:
        clear_screen()
        print("="*60)
        print("⚾ MLB GAME PREDICTOR - SISTEMA DE MANTENIMIENTO ⚾")
        print("="*60)
        
        stats = check_db_health()
        
        print(f"\n📊 RESUMEN DE ESTADO:")
        print(f"   - Predicciones totales: {stats['total_predictions']}")
        print(f"   - Resultados reales: {stats['total_results']}")
        
        total_missing = sum(count for _, count in stats["missing_results"])
        if total_missing > 0:
            print(f"   - ⚠️ Partidos sin validar: {total_missing} (en {len(stats['missing_results'])} fechas)")
        else:
            print(f"   - ✅ Todos los resultados están al día.")
            
        total_fallback = sum(count for _, count in stats["fallback_predictions"])
        if total_fallback > 0:
            print(f"   - ⚠️ Predicciones de baja calidad (FALLBACK): {total_fallback}")
        else:
            print(f"   - ✅ No hay predicciones incompletas.")
        
        print("\n" + "-"*60)
        print("OPCIONES:")
        print("1. 🔄 Recuperar resultados reales (validar aciertos)")
        print("2. 🧙 Re-predecir juegos FALLBACK (mejorar calidad)")
        print("3. 🧹 Limpiar duplicados y optimizar DB")
        print("4. 📋 Ver detalle de fechas con problemas")
        print("5. 🚀 Subir base de datos a producción (commit + push)")
        print("6. 🛠️  Sanación Masiva (Fix Gaps en toda la temporada)")
        print("7. 🚪 Salir")
        
        choice = input("\nSeleccione una opción: ")
        
        if choice == '1':
            if not stats["missing_results"]:
                print("✅ No hay resultados pendientes.")
                time.sleep(1.5)
                continue
            
            print(f"\nSe intentará recuperar resultados para {len(stats['missing_results'])} fechas.")
            confirm = input("¿Desea continuar? (s/n): ")
            if confirm.lower() == 's':
                for fecha, _ in stats["missing_results"]:
                    print(f"\n📅 Procesando fecha: {fecha}")
                    os.environ["TARGET_DATE"] = fecha
                    run_script("mlb_update_real_results.py", ["--fecha", fecha])
                input("\nPresione Enter para continuar...")

        elif choice == '2':
            print("\nMejorando calidad de predicciones FALLBACK...")
            run_script("mlb_manual_interface.py", ["--mode", "fix_fallback"])
            input("\nPresione Enter para continuar...")

        elif choice == '3':
            print("\nOptimizando base de datos...")
            run_script("mlb_utils.py", ["--action", "optimize"])
            input("\nPresione Enter para continuar...")

        elif choice == '4':
            print("\n📋 DETALLE DE FECHAS CON PROBLEMAS:")
            if stats["missing_results"]:
                print("\nResultados faltantes:")
                for f, c in stats["missing_results"]:
                    print(f" - {f}: {c} juegos")
            if stats["fallback_predictions"]:
                print("\nPredicciones FALLBACK:")
                for f, c in stats["fallback_predictions"]:
                    print(f" - {f}: {c} juegos")
            input("\nPresione Enter para continuar...")

        elif choice == '5':
            push_db_to_production()
            input("\nPresione Enter para continuar...")

        elif choice == '6':
            print("\n🔍 Buscando huecos en toda la temporada 2026...")
            run_script("mlb_mass_healer.py")
            input("\nPresione Enter para continuar...")

        elif choice == '7':
            print("👋 Saliendo...")
            break
        else:
            print("❌ Opción inválida.")
            time.sleep(1.5)

if __name__ == "__main__":
    menu_principal()
