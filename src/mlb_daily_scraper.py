"""
Scraper Diario de Partidos MLB - VERSIÓN 3 (API-FIRST PURE)
Ejecuta de forma autónoma para capturar la cartelera y estadísticas oficiales de lanzadores.
100% libre de web scraping a Baseball-Reference.
"""

import os
import re
import sqlite3
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

# Importar configuración centralizada
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_config import DB_PATH, get_team_code
from mlb_stats_api_client import obtener_stats_pitcher_api, obtener_stats_pitcher_por_nombre

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


def safe_float(val, default=0.0):
    try:
        if val is None or val == "-" or val == "":
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_int(val, default=0):
    try:
        if val is None or val == "-" or val == "":
            return default
        return int(float(val))
    except (ValueError, TypeError):
        return default


def normalizar_team_code(code):
    """Normaliza códigos a los usados en la base de datos."""
    if not code:
        return code
    code = str(code).upper().strip()
    return TEAM_CODE_ALIASES.get(code, code)


def obtener_fechas_ejecucion():
    """Obtiene fechas formateadas para la consulta de la API."""
    target_date = os.getenv("TARGET_DATE", "").strip()

    if target_date:
        try:
            ahora = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            print(f"⚠️ TARGET_DATE inválida '{target_date}'. Se usará fecha actual en ET.")
            ahora = datetime.now(ZoneInfo("America/New_York"))
    else:
        ahora = datetime.now(ZoneInfo("America/New_York"))

    # Formato para la base de datos y la API
    fecha_db = ahora.strftime("%Y-%m-%d")
    fecha_legible = ahora.strftime("%A, %B %d, %Y")

    return fecha_legible, fecha_db, ahora.year


def ejecutar_pipeline_diario():
    """
    Pipeline principal para obtener cartelera del día via la API oficial de la MLB.
    Elimina por completo el fallback HTML a Baseball-Reference.
    """
    fecha_legible, fecha_db, year_val = obtener_fechas_ejecucion()
    run_source = os.getenv("RUN_SOURCE", "local").strip().lower()
    max_intentos = int(os.getenv("SCRAPER_MAX_ATTEMPTS", "3"))
    espera_reintento = int(os.getenv("SCRAPER_RETRY_WAIT_SECONDS", "30"))

    print(f"\n{'=' * 70}")
    print(f"📅 Cargando Cartelera Oficial MLB para: {fecha_legible}")
    print("   Origen de datos: API de Estadísticas de la MLB (statsapi.mlb.com)")
    print("   100% Pura Ingesta Digital - Cero Scraping HTML")
    print(f"{'=' * 70}")

    url_api = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={fecha_db}&hydrate=probablePitcher"

    equipos_hoy = []
    completo_api = False

    for intento in range(1, max_intentos + 1):
        try:
            print(f"🔍 Consultando cartelera oficial en la API de la MLB (intento {intento}/{max_intentos}) para {fecha_db}...")
            r = requests.get(url_api, timeout=12)
            if r.status_code == 200:
                data = r.json()
                if "dates" in data and len(data["dates"]) > 0:
                    games = data["dates"][0].get("games", [])
                    for g in games:
                        away_name = g["teams"]["away"]["team"]["name"]
                        home_name = g["teams"]["home"]["team"]["name"]
                        away_std = normalizar_team_code(get_team_code(away_name))
                        home_std = normalizar_team_code(get_team_code(home_name))

                        # Generar el boxscore URL relativo sintético en formato Baseball-Reference
                        bref_home_code = STANDARD_TO_BREF.get(home_std, home_std)
                        date_clean = fecha_db.replace("-", "")
                        game_num_str = "0"
                        if g.get("doubleHeader") in ("Y", "S") and g.get("gameNumber") in (2, "2"):
                            game_num_str = "2"
                        preview_link = f"/boxes/{bref_home_code}/{bref_home_code}{date_clean}{game_num_str}.shtml"

                        # Guardar el probablePitcher de away y home
                        away_p = g["teams"]["away"].get("probablePitcher", {})
                        home_p = g["teams"]["home"].get("probablePitcher", {})

                        equipos_hoy.append({
                            "away_team": away_std,
                            "home_team": home_std,
                            "preview_link": preview_link,
                            "lanzadores_api": {
                                away_std: {"nombre": away_p.get("fullName"), "link": f"/api/v1/people/{away_p.get('id')}" if away_p.get("id") else "", "id": away_p.get("id")} if away_p.get("id") else None,
                                home_std: {"nombre": home_p.get("fullName"), "link": f"/api/v1/people/{home_p.get('id')}" if home_p.get("id") else "", "id": home_p.get("id")} if home_p.get("id") else None
                            }
                        })
                        print(f"  ✅ Encontrado: {away_std} @ {home_std} (BRef URL sintética: {preview_link})")

                    if equipos_hoy:
                        completo_api = True
                        break
                else:
                    print(f"  ⚠️ Intento {intento}/{max_intentos} falló con datos vacíos en API")
            else:
                print(f"  ⚠️ Intento {intento}/{max_intentos} falló con status {r.status_code}")
        except Exception as e:
            print(f"  ⚠️ Error consultando cartelera via API (intento {intento}/{max_intentos}): {e}")

        if intento < max_intentos:
            time.sleep(espera_reintento)

    if not completo_api or not equipos_hoy:
        print("❌ No se pudo cargar la cartelera oficial desde la API de la MLB.")
        return False

    data_partidos = []

    for i, juego_dict in enumerate(equipos_hoy, start=1):
        away_team = juego_dict["away_team"]
        home_team = juego_dict["home_team"]
        preview_link = juego_dict["preview_link"]
        lanzadores_api = juego_dict["lanzadores_api"]

        print(f"\n⚾ Procesando partido {i}/{len(equipos_hoy)}: {away_team} @ {home_team}")

        away_info = lanzadores_api.get(away_team) or {}
        home_info = lanzadores_api.get(home_team) or {}

        away_pitcher = away_info.get("nombre") or "Anunciado"
        away_pitcher_link = away_info.get("link") or ""
        away_pitcher_id = away_info.get("id") or None

        home_pitcher = home_info.get("nombre") or "Anunciado"
        home_pitcher_link = home_info.get("link") or ""
        home_pitcher_id = home_info.get("id") or None

        print(f"   Lanzadores: {away_pitcher} vs {home_pitcher}")

        # Ingesta de estadísticas de pitcheo desde la API
        s_away = None
        if away_pitcher_id:
            print(f"   🔍 Stats de {away_pitcher} via MLB API...")
            s_away = obtener_stats_pitcher_api(away_pitcher_id, year_val)

        if not s_away and away_pitcher != "Anunciado":
            print(f"   🔍 Stats de {away_pitcher} via búsqueda por nombre...")
            s_away = obtener_stats_pitcher_por_nombre(away_team, away_pitcher, year_val)

        s_home = None
        if home_pitcher_id:
            print(f"   🔍 Stats de {home_pitcher} via MLB API...")
            s_home = obtener_stats_pitcher_api(home_pitcher_id, year_val)

        if not s_home and home_pitcher != "Anunciado":
            print(f"   🔍 Stats de {home_pitcher} via búsqueda por nombre...")
            s_home = obtener_stats_pitcher_por_nombre(home_team, home_pitcher, year_val)

        # Fallbacks si no hay estadísticas disponibles
        if not s_away:
            s_away = {"ERA": 4.50, "WHIP": 1.35, "H9": 8.5, "SO9": 7.5, "W": 0, "L": 0}
        if not s_home:
            s_home = {"ERA": 4.50, "WHIP": 1.35, "H9": 8.5, "SO9": 7.5, "W": 0, "L": 0}

        game_id = f"{fecha_db}_{home_team}_{away_team}"

        data_partidos.append({
            "game_id": game_id,
            "box_score_url": preview_link,
            "fecha": fecha_db,
            "year": year_val,
            "away_team": away_team,
            "home_team": home_team,
            "away_pitcher": away_pitcher,
            "home_pitcher": home_pitcher,
            "away_pitcher_link": away_pitcher_link,
            "home_pitcher_link": home_pitcher_link,
            "away_starter_ERA": s_away["ERA"],
            "away_starter_WHIP": s_away["WHIP"],
            "away_starter_H9": s_away["H9"],
            "away_starter_SO9": s_away["SO9"],
            "away_starter_W": s_away["W"],
            "away_starter_L": s_away["L"],
            "home_starter_ERA": s_home["ERA"],
            "home_starter_WHIP": s_home["WHIP"],
            "home_starter_H9": s_home["H9"],
            "home_starter_SO9": s_home["SO9"],
            "home_starter_W": s_home["W"],
            "home_starter_L": s_home["L"],
        })
        print("   ✅ Partido procesado con éxito")

    if not data_partidos:
        print("⚠️ No hay partidos válidos procesados.")
        return True

    # Guardar en base de datos
    print(f"\n💾 Guardando {len(data_partidos)} partidos en la base de datos...")
    with sqlite3.connect(DB_PATH) as conn:
        # Verificar esquema existente y recrear si es necesario
        conn.execute("DROP TABLE IF EXISTS historico_partidos")
        conn.execute(
            """CREATE TABLE historico_partidos
                       (game_id TEXT PRIMARY KEY, box_score_url TEXT, fecha TEXT, year INTEGER,
                        away_team TEXT, home_team TEXT, away_pitcher TEXT, home_pitcher TEXT,
                        away_pitcher_link TEXT, home_pitcher_link TEXT,
                        away_starter_ERA REAL, away_starter_WHIP REAL, away_starter_H9 REAL,
                        away_starter_SO9 REAL, away_starter_W INTEGER, away_starter_L INTEGER,
                        home_starter_ERA REAL, home_starter_WHIP REAL, home_starter_H9 REAL,
                        home_starter_SO9 REAL, home_starter_W INTEGER, home_starter_L INTEGER)"""
        )

        conn.execute(
            """CREATE TABLE IF NOT EXISTS lineup_ini
                       (fecha TEXT, game_id TEXT, team TEXT, [order] TEXT, player TEXT)"""
        )

        conn.execute(
            """CREATE TABLE IF NOT EXISTS sync_control (
                       dataset TEXT,
                       source TEXT,
                       fecha TEXT,
                       updated_at TEXT,
                       PRIMARY KEY(dataset, source)
                   )"""
        )

        df_partidos = pd.DataFrame(data_partidos)
        for fecha_guardada in df_partidos["fecha"].unique():
            conn.execute("DELETE FROM historico_partidos WHERE fecha = ?", (fecha_guardada,))

        # Inserción robusta
        for _, row in df_partidos.iterrows():
            conn.execute(
                """INSERT OR REPLACE INTO historico_partidos VALUES
                           (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                tuple(row),
            )

        conn.execute(
            """INSERT INTO sync_control (dataset, source, fecha, updated_at)
                       VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                       ON CONFLICT(dataset, source)
                       DO UPDATE SET fecha = excluded.fecha,
                                     updated_at = CURRENT_TIMESTAMP""",
            ("games_today", run_source, fecha_db),
        )
        conn.commit()

    print("✅ Proceso finalizado exitosamente")
    print(f"   - Partidos insertados: {len(data_partidos)}")
    print(f"{'=' * 70}\n")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scraper diario de partidos MLB (API-First)")
    parser.add_argument("--retry", action="store_true", help="Modo reintentar")
    args = parser.parse_args()

    resultado = ejecutar_pipeline_diario()
    sys.exit(0 if resultado else 1)
