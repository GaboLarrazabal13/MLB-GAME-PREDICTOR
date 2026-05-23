"""
Actualizador de Resultados Reales MLB - REFACTORIZADO
Ejecuta al día siguiente a las 5 AM para capturar resultados finales
Versión optimizada para GitHub Actions
"""

import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import cloudscraper
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Importar configuración centralizada
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_config import DB_PATH, SCRAPING_CONFIG, get_team_code
from mlb_schedule_utils import seleccionar_seccion_schedule

# ============================================================================
# FUNCIONES DE SOPORTE
# ============================================================================


def obtener_html(url, max_retries=None):
    """Obtiene HTML con reintentos"""
    if max_retries is None:
        max_retries = SCRAPING_CONFIG["max_retries"]

    scraper = cloudscraper.create_scraper()

    for intento in range(max_retries):
        try:
            response = scraper.get(url, timeout=SCRAPING_CONFIG["timeout"])
            if response.status_code == 200:
                response.encoding = "utf-8"
                return response.text
            time.sleep(2**intento)
        except Exception as e:
            if intento == max_retries - 1:
                print(f"       Error final obteniendo {url}: {e}")
            time.sleep(2**intento)

    return None


def obtener_fechas_ayer():
    """Calcula las fechas de ayer para la ejecución automática a las 5 AM"""
    ayer = datetime.now(ZoneInfo("America/New_York")) - timedelta(days=1)

    # Formato para Baseball-Reference
    fecha_bref = ayer.strftime("%A, %B %-d, %Y" if os.name != "nt" else "%A, %B %#d, %Y")

    # Formato para base de datos
    fecha_db = ayer.strftime("%Y-%m-%d")

    return fecha_bref, fecha_db, ayer.year


def _formatear_fecha_bref_desde_db(fecha_db):
    """Convierte fecha YYYY-MM-DD a formato de encabezado de Baseball-Reference."""
    fecha_dt = datetime.strptime(fecha_db, "%Y-%m-%d")
    return fecha_dt.strftime("%A, %B %-d, %Y" if os.name != "nt" else "%A, %B %#d, %Y")


def obtener_fechas_objetivo(max_fechas=3):
    """Obtiene fechas objetivo recientes, siempre incluyendo ayer."""
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


def extraer_pitchers_desde_boxscore(box_score_url, home_team_gano):
    """Extrae lanzadores home/away desde el footer de linescore (WP/LP)."""
    if not box_score_url:
        return None, None

    html = obtener_html(f"https://www.baseball-reference.com{box_score_url}")
    if not html:
        return None, None

    soup = BeautifulSoup(html, "html.parser")
    linescore = soup.select_one("table.linescore")
    if not linescore:
        return None, None

    tfoot = linescore.find("tfoot")
    if not tfoot:
        return None, None

    footer_text = tfoot.get_text(" ", strip=True).replace("\xa0", " ")

    def extraer_nombre(tag):
        match = re.search(rf"{tag}:\s*([^•(]+?)\s*\(", footer_text)
        if match:
            return match.group(1).strip()
        match = re.search(rf"{tag}:\s*([^•]+)", footer_text)
        if match:
            return match.group(1).strip()
        return None

    wp = extraer_nombre("WP")
    lp = extraer_nombre("LP")

    if not wp or not lp:
        return None, None

    if home_team_gano:
        return wp, lp
    return lp, wp


# ============================================================================
# PROCESO PRINCIPAL
# ============================================================================


def actualizar_resultados_reales_en_fecha(fecha_bref, fecha_db, year_val):
    """
    Actualiza los resultados reales para una fecha específica
    Retorna True si procesó datos, False si no
    """
    print(f"\n{'=' * 70}")
    print(f"🕐 Actualizando resultados reales para: {fecha_bref}")
    print(f"{'=' * 70}")

    data_resultados = []
    partidos_procesados = 0
    completo_api = False

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

    # 1. INTENTAR VIA MLB STATS API (2 INTENTOS)
    url_api = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={fecha_db}"
    for intento_api in range(1, 3):
        try:
            print(f"  🔍 Consultando resultados en la API de la MLB (intento {intento_api}/2) para la fecha: {fecha_db}...")
            r = requests.get(url_api, timeout=10)
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

                            # Generar boxscore URL relativo sintético
                            bref_home_code = STANDARD_TO_BREF.get(home_team, home_team)
                            date_clean = fecha_db.replace("-", "")
                            game_num_str = "0"
                            if g.get("doubleHeader") in ("Y", "S") and g.get("gameNumber") in (2, "2"):
                                game_num_str = "2"
                            box_score_url = f"/boxes/{bref_home_code}/{bref_home_code}{date_clean}{game_num_str}.shtml"

                            game_id = f"{fecha_db}_{home_team}_{away_team}"

                            data_resultados.append(
                                {
                                    "game_id": game_id,
                                    "box_score_url": box_score_url,
                                    "fecha": fecha_db,
                                    "year": year_val,
                                    "home_team": home_team,
                                    "away_team": away_team,
                                    "score_home": score_home,
                                    "score_away": score_away,
                                    "ganador": ganador,
                                }
                            )
                            partidos_procesados += 1
                            print(f"✅ [API] {away_team} @ {home_team}: {score_away}-{score_home}")

                    if data_resultados:
                        completo_api = True
                        break
            else:
                print(f"  ⚠️ [API] Intento {intento_api}/2 falló con status {r.status_code}")
        except Exception as e:
            print(f"  ⚠️ Error consultando resultados via API (intento {intento_api}/2): {e}")
        if not completo_api and intento_api < 2:
            time.sleep(2)

    # 2. FALLBACK A BASEBALL-REFERENCE SCHEDULE HTML
    if not completo_api:
        print("  ⚠️ Falló la API de la MLB para resultados tras 2 intentos. Activando fallback HTML...")
        url_schedule = f"https://www.baseball-reference.com/leagues/majors/{year_val}-schedule.shtml"
        html = obtener_html(url_schedule)

        if not html:
            print("❌ Error de conexión con Baseball-Reference.")
            return False

        soup = BeautifulSoup(html, "html.parser")
        seccion = seleccionar_seccion_schedule(soup, fecha_db)

        if not seccion or seccion.get("date") != fecha_db:
            print(f"⚠️ No hay juegos registrados para la fecha {fecha_bref} aún.")
            return False

        print(
            "📍 Sección de resultados seleccionada: "
            f"{seccion.get('label', 'sin etiqueta')} -> {seccion.get('date', 'sin fecha')}"
        )

        for game in seccion["games"]:
            cursor = game.get("node")
            if not cursor:
                continue

            try:
                links = cursor.find_all("a")
                if len(links) >= 3:
                    box_score_url = None
                    for link in links:
                        href = link.get("href", "")
                        if href.startswith("/boxes/"):
                            box_score_url = href
                            break

                    away_team_full = links[0].text.strip()
                    home_team_full = links[1].text.strip()

                    # Convertir a códigos
                    away_team = get_team_code(away_team_full) or away_team_full
                    home_team = get_team_code(home_team_full) or home_team_full

                    # Extraer scores del texto
                    texto_juego = cursor.get_text()
                    scores = re.findall(r"\((\d+)\)", texto_juego)

                    if len(scores) >= 2:
                        score_away = int(scores[0])
                        score_home = int(scores[1])
                        ganador = 1 if score_home > score_away else 0

                        # Crear game_id consistente
                        game_id = f"{fecha_db}_{home_team}_{away_team}"

                        data_resultados.append(
                            {
                                "game_id": game_id,
                                "box_score_url": box_score_url,
                                "fecha": fecha_db,
                                "year": year_val,
                                "home_team": home_team,
                                "away_team": away_team,
                                "score_home": score_home,
                                "score_away": score_away,
                                "ganador": ganador,
                            }
                        )

                        partidos_procesados += 1
                        print(f"✅ {away_team} @ {home_team}: {score_away}-{score_home}")
                    else:
                        print(f"⚠️ Partido sin scores finales: {away_team} @ {home_team}")

            except Exception as e:
                print(f"⚠️ Error procesando partido: {e}")
                import traceback
                traceback.print_exc()

    if not data_resultados:
        print("\n⚠️ No se encontraron resultados finales para procesar.")
        return False

    # Procesar y guardar resultados
    print(f"\n{'=' * 70}")
    print(f"💾 Procesando {len(data_resultados)} resultados...")

    with sqlite3.connect(DB_PATH) as conn:
        df_res = pd.DataFrame(data_resultados)

        # Obtener lanzadores desde lineup_ini usando game_id
        query_pitchers = f"""
            SELECT game_id, team, player as pitcher
            FROM lineup_ini
            WHERE fecha='{fecha_db}' AND [order]='P'
        """
        try:
            df_p = pd.read_sql(query_pitchers, conn)
        except Exception:
            df_p = pd.DataFrame(columns=["game_id", "team", "pitcher"])

        if df_p.empty:
            print("⚠️ No se encontraron lineups previos. Intentando extraer WP/LP desde boxscore.")
            df_res["home_pitcher"] = None
            df_res["away_pitcher"] = None
        else:
            # Merge para Home Pitcher
            df_final = df_res.merge(
                df_p[df_p["team"].isin(df_res["home_team"])],
                left_on=["game_id", "home_team"],
                right_on=["game_id", "team"],
                how="left",
            )
            df_final = df_final.rename(columns={"pitcher": "home_pitcher"}).drop(columns=["team"], errors="ignore")

            # Merge para Away Pitcher
            df_final = df_final.merge(
                df_p[df_p["team"].isin(df_res["away_team"])],
                left_on=["game_id", "away_team"],
                right_on=["game_id", "team"],
                how="left",
            )
            df_final = df_final.rename(columns={"pitcher": "away_pitcher"}).drop(columns=["team"], errors="ignore")

            df_res = df_final

        # Fallback: completar lanzadores faltantes desde boxscore (WP/LP)
        faltantes = df_res[df_res["home_pitcher"].isna() | df_res["away_pitcher"].isna()].copy()
        if not faltantes.empty:
            print(f"ℹ️ Intentando completar lanzadores desde boxscore para {len(faltantes)} partidos...")
            for idx, row in faltantes.iterrows():
                home_pitcher, away_pitcher = extraer_pitchers_desde_boxscore(
                    row.get("box_score_url"), row.get("ganador") == 1
                )
                if home_pitcher and pd.isna(df_res.at[idx, "home_pitcher"]):
                    df_res.at[idx, "home_pitcher"] = home_pitcher
                if away_pitcher and pd.isna(df_res.at[idx, "away_pitcher"]):
                    df_res.at[idx, "away_pitcher"] = away_pitcher

        # Crear tabla si no existe
        conn.execute("""CREATE TABLE IF NOT EXISTS historico_real
                       (game_id TEXT PRIMARY KEY, home_team TEXT, away_team TEXT,
                        home_pitcher TEXT, away_pitcher TEXT, ganador INTEGER,
                        year INTEGER, fecha TEXT, score_home INTEGER, score_away INTEGER)""")

        # Columnas finales
        columnas_finales = [
            "game_id",
            "home_team",
            "away_team",
            "home_pitcher",
            "away_pitcher",
            "ganador",
            "year",
            "fecha",
            "score_home",
            "score_away",
        ]

        # Preparar export
        df_export = df_res[columnas_finales].copy()

        # Mostrar cuántos tienen pitchers
        con_pitchers = df_export[df_export["home_pitcher"].notna() & df_export["away_pitcher"].notna()]
        sin_pitchers = df_export[df_export["home_pitcher"].isna() | df_export["away_pitcher"].isna()]

        print("\n📊 Estadísticas:")
        print(f"   - Partidos con lanzadores: {len(con_pitchers)}")
        print(f"   - Partidos sin lanzadores: {len(sin_pitchers)}")

        if len(sin_pitchers) > 0:
            print("\n⚠️ Partidos sin datos de lanzadores:")
            for _, row in sin_pitchers.iterrows():
                print(f"      {row['away_team']} @ {row['home_team']}")

        # Guardar todos (INSERT OR REPLACE para evitar duplicados)
        for _, row in df_export.iterrows():
            # Evita duplicados en esquemas antiguos sin PK efectiva en game_id.
            conn.execute(
                """DELETE FROM historico_real
                   WHERE fecha = ? AND home_team = ? AND away_team = ?""",
                (row["fecha"], row["home_team"], row["away_team"]),
            )
            conn.execute(
                """INSERT OR REPLACE INTO historico_real VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                tuple(row),
            )

        conn.commit()

        print(f"\n✅ Se han guardado {len(df_export)} resultados reales en 'historico_real'")
        print(f"{'=' * 70}\n")

        return True


def actualizar_resultados_reales():
    """
    Actualiza resultados reales para fechas pendientes recientes.
    Retorna True para mantener estable el workflow incluso si no hay nuevos finales.
    """
    fechas_objetivo = obtener_fechas_objetivo(max_fechas=3)

    print("\n📅 Fechas objetivo para actualizar resultados:")
    for _, fecha_db, _ in fechas_objetivo:
        print(f"   - {fecha_db}")

    procesados = 0
    for fecha_bref, fecha_db, year_val in fechas_objetivo:
        if actualizar_resultados_reales_en_fecha(fecha_bref, fecha_db, year_val):
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


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Actualizador de resultados reales MLB")
    parser.add_argument("--verificar", action="store_true", help="Verificar juegos pendientes")
    parser.add_argument(
        "--fecha",
        type=str,
        default=None,
        help="Fecha específica a procesar (formato YYYY-MM-DD). Si no se indica, usa TARGET_DATE env var o auto-detección.",
    )
    args = parser.parse_args()

    if args.verificar:
        verificar_juegos_pendientes()
    else:
        # Soporte para fecha específica via argumento o variable de entorno
        target_date = args.fecha or os.environ.get("TARGET_DATE", "").strip()

        if target_date:
            print(f"\n🎯 Modo backfill — fecha objetivo: {target_date}")
            try:
                fecha_bref = _formatear_fecha_bref_desde_db(target_date)
                year_val = int(target_date[:4])
                resultado = actualizar_resultados_reales_en_fecha(fecha_bref, target_date, year_val)
            except Exception as exc:
                print(f"❌ Error al procesar fecha {target_date}: {exc}")
                resultado = False
        else:
            resultado = actualizar_resultados_reales()

        # Verificar pendientes después de actualizar
        if resultado:
            print("\n" + "=" * 70)
            verificar_juegos_pendientes()

        # Retornar código de salida para GitHub Actions
        sys.exit(0 if resultado else 1)
