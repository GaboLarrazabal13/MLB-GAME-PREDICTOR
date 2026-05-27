"""
MLB Batch Repredict Utility - V4.0
Regenera todas las predicciones de la temporada 2026 usando el nuevo motor V4.0.
Esto asegura consistencia y transparencia de datos en la aplicación Streamlit.
"""

import os
import sqlite3
import sys
import time

# Asegurar importación de módulos centralizados
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_config import DB_PATH
from mlb_predict_engine import predecir_juego


def repredecir_temporada_2026():
    print("=" * 80)
    print("🚀 INICIANDO RE-SCORING DE LA TEMPORADA 2026 CON EL MOTOR V4.0")
    print("=" * 80)

    if not os.path.exists(DB_PATH):
        print(f"❌ Error: No existe la base de datos en {DB_PATH}")
        return

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        partidos = conn.execute(
            """
            SELECT fecha, home_team, away_team, home_pitcher, away_pitcher, year
            FROM historico_real
            WHERE substr(fecha, 1, 4) = '2026'
            ORDER BY fecha ASC
            """
        ).fetchall()

    total_partidos = len(partidos)
    print(f"📋 Se encontraron {total_partidos} partidos para procesar.")

    if total_partidos == 0:
        print("🔭 No hay partidos registrados del 2026 en 'historico_real'.")
        return

    exitosos = 0
    errores = 0
    inicio_ts = time.time()

    for idx, p in enumerate(partidos, 1):
        fecha = p["fecha"]
        home = p["home_team"]
        away = p["away_team"]
        hp = p["home_pitcher"]
        ap = p["away_pitcher"]
        year = p["year"] or 2026

        print(f"\n🔄 [{idx}/{total_partidos}] {fecha}: {away} @ {home} ({ap} vs {hp})")

        try:
            # predecir_juego borrará automáticamente el registro anterior en predicciones_historico
            # e insertará la nueva predicción con el motor V4.0.
            res = predecir_juego(
                home_team=home,
                away_team=away,
                home_pitcher=hp,
                away_pitcher=ap,
                year=year,
                modo_auto=True,
                fecha_partido=fecha,
                hacer_scraping=True,
                guardar_db=True
            )
            if res:
                exitosos += 1
                print(f"   ✅ Predicho con V4.0: {res['prediccion']} (Prob: {max(res['prob_home'], res['prob_away']):.1f}%, Conf: {res['confianza']})")
            else:
                errores += 1
                print("   ❌ Error: El motor no devolvió predicción.")
        except Exception as e:
            errores += 1
            print(f"   ❌ Error procesando partido: {e}")

    elapsed = time.time() - inicio_ts
    print("\n" + "=" * 80)
    print(f"🏁 PROCESO COMPLETADO EN {elapsed/60:.1f} MINUTOS")
    print(f"✅ Predicciones exitosas generadas: {exitosos}")
    print(f"⚠️ Errores encontrados: {errores}")
    print("=" * 80)

if __name__ == "__main__":
    repredecir_temporada_2026()
