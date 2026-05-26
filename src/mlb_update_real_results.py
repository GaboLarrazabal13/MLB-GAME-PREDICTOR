"""
Actualizador de Resultados Reales MLB - VERSIÓN 3 (API-FIRST PURE)
Ejecuta al día siguiente a las 5 AM para capturar resultados finales y lanzadores oficiales.
100% libre de web scraping a Baseball-Reference.
"""

import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

# Importar configuración centralizada
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_config import DB_PATH, get_team_code

TEAM_CODE_ALIASES = {
    "ANA": "LAA",
    "CHN": "CHC",
    "KCA": "KCR",
    "LAN": "LAD",
    "NYN": "NYM",
    "SDN": "SDP",
    "SFN": "SFG",
    "SLN": "STL",
    "TBD": "TBR",
}
STANDARD_TO_BREF = {v: k for k, v in TEAM_CODE_ALIASES.items()}


def obtener_fechas_ayer():
    """Calcula las fechas de ayer para la ejecución automática a las 5 AM"""
    ayer = datetime.now(ZoneInfo("America/New_York")) - timedelta(days=1)
    fecha_legible = ayer.strftime("%A, %B %d, %Y")
    fecha_db = ayer.strftime("%Y-%m-%d")
    return fecha_legible, fecha_db, ayer.year


def _formatear_fecha_bref_desde_db(fecha_db):
    """Convierte fecha YYYY-MM-DD a formato legible."""
    fecha_dt = datetime.strptime(fecha_db, "%Y-%m-%d")
    return fecha_dt.strftime("%A, %B %d, %Y")


def obtener_fechas_objetivo(max_fechas=3):
    """Obtiene fechas objetivo recientes, siempre incluyendo ayer y las que no tienen resultados en la base de datos."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    hoy_db = now_et.strftime("%Y-%m-%d")
    ayer_db = (now_et - timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        with sqlite3.connect(DB_PATH) as conn:
            pendientes = conn.execute(
                """
                SELECT DISTINCT hp.fecha
                FROM historico_partidos hp
                LEFT JOIN historico_real hr ON hp.game_id = hr.game_id
                WHERE hr.game_id IS NULL
                  AND hp.fecha <= ?
                ORDER BY hp.fecha DESC
                LIMIT ?
                """,
                (hoy_db, max_fechas),
            ).fetchall()
    except Exception:
        pendientes = []

    # Prioridad: siempre ayer, luego pendientes, luego respaldo de fechas recientes.
    candidatas = [ayer_db]
    candidatas.extend([fecha_db for (fecha_db,) in pendientes])
    candidatas.extend(
        [
            (now_et - timedelta(days=2)).strftime("%Y-%m-%d"),
            (now_et - timedelta(days=3)).strftime("%Y-%m-%d"),
        ]
    )

    fechas_unicas = []
    for fecha_db in candidatas:
        if fecha_db <= hoy_db and fecha_db not in fechas_unicas:
            fechas_unicas.append(fecha_db)

    fechas = []
    for fecha_db in fechas_unicas[:max_fechas]:
        fechas.append(
            (
                _formatear_fecha_bref_desde_db(fecha_db),
                fecha_db,
                datetime.strptime(fecha_db, "%Y-%m-%d").year,
            )
        )

    return fechas


def actualizar_resultados_reales_en_fecha(fecha_legible, fecha_db, year_val):
    """
    Actualiza los resultados reales para una fecha específica usando la API oficial de la MLB.
    """
    print(f"\n{'=' * 70}")
    print(f"🕐 Actualizando resultados reales para: {fecha_legible}")
    print("   Origen de datos: API de Estadísticas de la MLB (statsapi.mlb.com)")
    print(f"{'=' * 70}")

    data_resultados = []
    partidos_procesados = 0
    completo_api = False

    url_api = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={fecha_db}&hydrate=probablePitcher,decisions"
    
    for intento in range(1, 3):
        try:
            print(f"🔍 Consultando resultados en la API de la MLB (intento {intento}/2) para {fecha_db}...")
            r = requests.get(url_api, timeout=12)
            if r.status_code == 200:
                data = r.json()
                if "dates" in data and len(data["dates"]) > 0:
                    games = data["dates"][0].get("games", [])
                    for g in games:
                        # Solo procesar juegos finalizados
                        state = g.get("status", {}).get("abstractGameState", "")
                        coded_state = g.get("status", {}).get("codedGameState", "")
                        if state != "Final" and coded_state != "F":
                            continue

                        away_name = g["teams"]["away"]["team"]["name"]
                        home_name = g["teams"]["home"]["team"]["name"]
                        away_team = get_team_code(away_name) or away_name
                        home_team = get_team_code(home_name) or home_name

                        # Normalizar códigos de equipo
                        away_team = TEAM_CODE_ALIASES.get(away_team, away_team)
                        home_team = TEAM_CODE_ALIASES.get(home_team, home_team)

                        score_away = g["teams"]["away"].get("score")
                        score_home = g["teams"]["home"].get("score")

                        if score_away is not None and score_home is not None:
                            score_away = int(score_away)
                            score_home = int(score_home)
                            ganador = 1 if score_home > score_away else 0

                            # Obtener nombres de lanzadores
                            away_pitcher = g["teams"]["away"].get("probablePitcher", {}).get("fullName")
                            home_pitcher = g["teams"]["home"].get("probablePitcher", {}).get("fullName")

                            # Fallback a decisiones si el probablePitcher no está disponible
                            decisions = g.get("decisions", {})
                            wp = decisions.get("winner", {}).get("fullName")
                            lp = decisions.get("loser", {}).get("fullName")

                            if not home_pitcher or not away_pitcher:
                                if wp and lp:
                                    if ganador == 1:
                                        home_pitcher = home_pitcher or wp
                                        away_pitcher = away_pitcher or lp
                                    else:
                                        home_pitcher = home_pitcher or lp
                                        away_pitcher = away_pitcher or wp

                            # Si todo lo demás falla, poner un marcador por defecto
                            home_pitcher = home_pitcher or "Desconocido"
                            away_pitcher = away_pitcher or "Desconocido"

                            # Generar boxscore URL relativo sintético
                            bref_home_code = STANDARD_TO_BREF.get(home_team, home_team)
                            date_clean = fecha_db.replace("-", "")
                            game_num_str = "0"
                            if g.get("doubleHeader") in ("Y", "S") and g.get("gameNumber") in (2, "2"):
                                game_num_str = "2"
                            box_score_url = f"/boxes/{bref_home_code}/{bref_home_code}{date_clean}{game_num_str}.shtml"

                            game_id = f"{fecha_db}_{home_team}_{away_team}"

                            data_resultados.append({
                                "game_id": game_id,
                                "home_team": home_team,
                                "away_team": away_team,
                                "home_pitcher": home_pitcher,
                                "away_pitcher": away_pitcher,
                                "ganador": ganador,
                                "year": year_val,
                                "fecha": fecha_db,
                                "score_home": score_home,
                                "score_away": score_away,
                                "box_score_url": box_score_url
                            })
                            partidos_procesados += 1
                            print(f"✅ {away_team} @ {home_team}: {score_away}-{score_home} (Pitchers: {away_pitcher} vs {home_pitcher})")

                    if data_resultados:
                        completo_api = True
                        break
                else:
                    print(f"  ⚠️ Intento {intento}/2 falló con datos vacíos en API")
            else:
                print(f"  ⚠️ Intento {intento}/2 falló con status {r.status_code}")
        except Exception as e:
            print(f"  ⚠️ Error consultando resultados via API: {e}")
        
        if not completo_api and intento < 2:
            time.sleep(2)

    if not data_resultados:
        print("⚠️ No se encontraron resultados finales para procesar.")
        return False

    # Guardar en base de datos
    print(f"\n💾 Guardando {len(data_resultados)} resultados reales en la base de datos...")
    with sqlite3.connect(DB_PATH) as conn:
        # Crear tabla si no existe
        conn.execute("""CREATE TABLE IF NOT EXISTS historico_real
                       (game_id TEXT PRIMARY KEY, home_team TEXT, away_team TEXT,
                        home_pitcher TEXT, away_pitcher TEXT, ganador INTEGER,
                        year INTEGER, fecha TEXT, score_home INTEGER, score_away INTEGER)""")

        for row in data_resultados:
            # Eliminar duplicados previos
            conn.execute(
                """DELETE FROM historico_real
                   WHERE fecha = ? AND home_team = ? AND away_team = ?""",
                (row["fecha"], row["home_team"], row["away_team"]),
            )
            # Insertar el nuevo resultado real
            conn.execute(
                """INSERT OR REPLACE INTO historico_real
                   (game_id, home_team, away_team, home_pitcher, away_pitcher, ganador, year, fecha, score_home, score_away)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["game_id"],
                    row["home_team"],
                    row["away_team"],
                    row["home_pitcher"],
                    row["away_pitcher"],
                    row["ganador"],
                    row["year"],
                    row["fecha"],
                    row["score_home"],
                    row["score_away"]
                )
            )
        conn.commit()

    print(f"✅ Se han guardado exitosamente {len(data_resultados)} resultados reales en 'historico_real'")
    print(f"{'=' * 70}\n")
    return True


def actualizar_resultados_reales():
    """
    Actualiza resultados reales para fechas pendientes recientes.
    """
    fechas_objetivo = obtener_fechas_objetivo(max_fechas=3)

    print("\n📅 Fechas objetivo para actualizar resultados:")
    for _, fecha_db, _ in fechas_objetivo:
        print(f"   - {fecha_db}")

    procesados = 0
    for fecha_legible, fecha_db, year_val in fechas_objetivo:
        if actualizar_resultados_reales_en_fecha(fecha_legible, fecha_db, year_val):
            procesados += 1

    if procesados == 0:
        print("\nℹ️ No se encontraron nuevos resultados finales para las fechas objetivo.")
    else:
        print(f"\n✅ Fechas procesadas con resultados guardados: {procesados}")

    return True


def verificar_juegos_pendientes():
    """Verifica si hay juegos que fueron capturados pero no tienen resultado"""
    with sqlite3.connect(DB_PATH) as conn:
        query = """
            SELECT hp.game_id, hp.fecha, hp.home_team, hp.away_team
            FROM historico_partidos hp
            LEFT JOIN historico_real hr ON hp.game_id = hr.game_id
            WHERE hr.game_id IS NULL
            ORDER BY hp.fecha DESC
            LIMIT 20
        """
        df_pendientes = pd.read_sql(query, conn)

        if not df_pendientes.empty:
            print(f"\n📋 Partidos pendientes de resultado ({len(df_pendientes)}):")
            for _, row in df_pendientes.iterrows():
                print(f"   {row['fecha']}: {row['away_team']} @ {row['home_team']}")
        else:
            print("\n✅ No hay partidos pendientes de resultado")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Actualizador de resultados reales MLB (API-First)")
    parser.add_argument("--verificar", action="store_true", help="Verificar juegos pendientes")
    parser.add_argument(
        "--fecha",
        type=str,
        default=None,
        help="Fecha específica a procesar (formato YYYY-MM-DD).",
    )
    args = parser.parse_args()

    if args.verificar:
        verificar_juegos_pendientes()
    else:
        target_date = args.fecha or os.environ.get("TARGET_DATE", "").strip()

        if target_date:
            print(f"\n🎯 Modo backfill — fecha objetivo: {target_date}")
            try:
                fecha_legible = _formatear_fecha_bref_desde_db(target_date)
                year_val = int(target_date[:4])
                resultado = actualizar_resultados_reales_en_fecha(fecha_legible, target_date, year_val)
            except Exception as exc:
                print(f"❌ Error al procesar fecha {target_date}: {exc}")
                resultado = False
        else:
            resultado = actualizar_resultados_reales()

        if resultado:
            print("\n" + "=" * 70)
            verificar_juegos_pendientes()

        sys.exit(0 if resultado else 1)
