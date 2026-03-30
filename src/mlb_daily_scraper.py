"""
Scraper Diario de Partidos MLB - VERSIÓN 2 (ACTUALIZADA)
Ejecuta a las 10 AM y 1 PM para capturar lineups del día
Adaptado para la nueva estructura de Baseball-Reference (2026)
"""

import os
import re
import sqlite3
import sys
import time
import unicodedata
from datetime import datetime
from io import StringIO
from zoneinfo import ZoneInfo

import cloudscraper
import pandas as pd
from bs4 import BeautifulSoup, Comment

# Importar configuración centralizada
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_config import DB_PATH, SCRAPING_CONFIG, get_team_code

# ============================================================================
# FUNCIONES DE SOPORTE Y FORMATEO
# ============================================================================


def normalizar_texto(texto):
    """Normaliza texto para comparaciones"""
    if not texto:
        return ""
    texto = str(texto).lower()
    texto = "".join(
        c
        for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )
    texto = re.sub(r"[^a-z0-9]", "", texto)
    return texto


def safe_float(val):
    """Convierte a float de forma segura"""
    try:
        if pd.isna(val) or val == "-":
            return 0.0
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def limpiar_dataframe(df):
    """Limpia dataframes de Baseball-Reference"""
    if df is None or len(df) == 0:
        return df
    name_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    df = df[
        ~df[name_col]
        .astype(str)
        .str.contains(r"Team Totals|Rank in|^\s*$", case=False, na=False, regex=True)
    ]
    return df.reset_index(drop=True)


def obtener_html(url, max_retries=None):
    """Obtiene HTML con reintentos"""
    if max_retries is None:
        max_retries = SCRAPING_CONFIG["max_retries"]

    scraper = cloudscraper.create_scraper(
        browser={
            "browser": "chrome",
            "platform": "windows",
            "desktop": True,
        }
    )
    scraper.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
            "Referer": "https://www.baseball-reference.com/",
        }
    )

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


def obtener_fechas_ejecucion():
    """Obtiene fechas formateadas para scraping"""
    # Baseball-Reference publica "today" con horario del este (MLB).
    # Usar ET evita desfasar fecha al correr en servidores UTC.
    ahora = datetime.now(ZoneInfo("America/New_York"))

    # Formato para Baseball-Reference (ej: "Monday, April 1, 2024")
    fecha_bref = ahora.strftime(
        "%A, %B %-d, %Y" if os.name != "nt" else "%A, %B %#d, %Y"
    )

    # Formato para base de datos
    fecha_db = ahora.strftime("%Y-%m-%d")

    return fecha_bref, fecha_db, ahora.year


# ============================================================================
# EXTRACCIÓN DE EQUIPOS Y LANZADORES (NUEVA ESTRUCTURA)
# ============================================================================


def extraer_equipos_del_dia(soup):
    """
    Extrae SOLO los equipos que se enfrentan hoy desde el span id='today'.
    Utiliza el h3 que contiene <span id='today'> y procesa p.game hasta el siguiente h3.
    """
    equipos_lista = []

    # Encontrar el h3 que contiene el span id="today"
    today_header = None
    for h3 in soup.find_all("h3"):
        if h3.find("span", {"id": "today"}):
            today_header = h3
            break

    if not today_header:
        print("  ⚠️ No se encontró h3 con span id='today'")
        return equipos_lista

    # Recorrer los hermanos siguientes a este h3
    for sibling in today_header.find_next_siblings():
        if sibling.name == "h3":
            # Fecha siguiente, terminamos
            break

        if sibling.name == "p" and "game" in sibling.get("class", []):
            try:
                team_links = sibling.find_all(
                    "a", href=re.compile(r"/teams/\w+/\d+\.shtml")
                )
                if len(team_links) >= 2:
                    away_href = team_links[0].get("href", "")
                    home_href = team_links[1].get("href", "")

                    away_match = re.search(r"/teams/(\w+)/", away_href)
                    home_match = re.search(r"/teams/(\w+)/", home_href)

                    if away_match and home_match:
                        away_team = away_match.group(1)
                        home_team = home_match.group(1)

                        em_tag = sibling.find("em")
                        preview_link = None
                        if em_tag:
                            preview_a = em_tag.find("a")
                            if preview_a:
                                preview_link = preview_a.get("href", "")

                        equipos_lista.append((away_team, home_team, preview_link))
                        print(f"  ✅ Encontrado: {away_team} @ {home_team}")
            except Exception as e:
                print(f"  ⚠️ Error extrayendo equipos: {e}")

    return equipos_lista


def extraer_lanzadores_del_preview(preview_url):
    """
    Extrae los nombres Y LINKS de los lanzadores iniciales de la página del preview
    Busca divs con class="section_heading assoc_sp_{TEAM_CODE}"
    Retorna diccionario: {team_code: {"nombre": pitcher_name, "link": pitcher_href}}
    """
    preview_retries = int(os.getenv("SCRAPER_PREVIEW_RETRIES", "3"))
    preview_retry_wait = int(os.getenv("SCRAPER_PREVIEW_RETRY_WAIT_SECONDS", "8"))

    html = None
    for intento in range(1, preview_retries + 1):
        html = obtener_html(f"https://www.baseball-reference.com{preview_url}")
        if html:
            break
        if intento < preview_retries:
            print(
                f"     ⚠️ Preview no disponible (intento {intento}/{preview_retries}), "
                f"reintentando en {preview_retry_wait}s..."
            )
            time.sleep(preview_retry_wait)

    if not html:
        print(f"     ⚠️ No se pudo obtener HTML del preview: {preview_url}")
        return {}

    soup = BeautifulSoup(html, "html.parser")

    # Buscar todos los divs con class que contenga "assoc_sp_"
    lanzadores = {}

    for div in soup.find_all("div", class_=re.compile(r"assoc_sp_")):
        try:
            # Extraer el código del equipo desde la clase (ej: assoc_sp_PIT -> PIT)
            class_attr = div.get("class")
            if isinstance(class_attr, list):
                class_str = " ".join(str(c) for c in class_attr)
            else:
                class_str = str(class_attr or "")
            team_match = re.search(r"assoc_sp_(\w+)", class_str)

            if team_match:
                team_code = team_match.group(1)

                # Buscar el h2 con el enlace del lanzador
                h2_tag = div.find("h2")
                if h2_tag:
                    pitcher_link = h2_tag.find("a")
                    if pitcher_link:
                        pitcher_name = pitcher_link.get_text(strip=True)
                        pitcher_href = pitcher_link.get("href", "")
                        lanzadores[team_code] = {
                            "nombre": pitcher_name,
                            "link": pitcher_href
                        }
                        print(f"     🎯 Lanzador {team_code}: {pitcher_name} | Link: {pitcher_href}")

        except Exception as e:
            print(f"     ⚠️ Error extrayendo lanzador: {e}")

    return lanzadores


def scrape_player_stats(team_code, year):
    """Extrae estadísticas de pitcheo de un equipo"""
    url = f"https://www.baseball-reference.com/teams/{team_code}/{year}.shtml"
    html = obtener_html(url)

    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")

    def buscar_tabla(table_id):
        """Busca tabla incluso si está en comentarios HTML"""
        tab = soup.find("table", {"id": table_id})
        if not tab:
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for c in comments:
                if f'id="{table_id}"' in c:
                    return BeautifulSoup(str(c), "html.parser").find("table")
        return tab

    pitching_table = buscar_tabla("players_standard_pitching")

    if pitching_table:
        try:
            return pd.read_html(StringIO(str(pitching_table)))[0]
        except Exception:
            pass

    return None


def encontrar_lanzador(pitching_df, nombre_lanzador):
    """Encuentra estadísticas de un lanzador específico"""
    if pitching_df is None or not nombre_lanzador:
        return None

    pitching_df = limpiar_dataframe(pitching_df)
    busqueda = normalizar_texto(nombre_lanzador)
    name_col = "Name" if "Name" in pitching_df.columns else pitching_df.columns[1]

    for _, fila in pitching_df.iterrows():
        nombre_tabla_limpio = normalizar_texto(str(fila[name_col]))
        if busqueda in nombre_tabla_limpio or nombre_tabla_limpio in busqueda:
            return {
                "ERA": safe_float(fila.get("ERA", 0)),
                "WHIP": safe_float(fila.get("WHIP", 0)),
                "H9": safe_float(fila.get("H9", 0)),
                "SO9": safe_float(fila.get("SO9", 0)),
                "W": safe_float(fila.get("W", 0)),
                "L": safe_float(fila.get("L", 0)),
            }

    return None


def extraer_lineups_completos(box_url):
    """Extrae lineups completos de la página del boxscore"""
    html = obtener_html(f"https://www.baseball-reference.com{box_url}")

    if not html:
        return [], [], None, None

    soup = BeautifulSoup(html, "html.parser")

    def obtener_datos(div_id):
        """Extrae bateadores y lanzador de un div de lineup"""
        div = soup.find("div", id=div_id)

        if not div:
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for c in comments:
                if f'id="{div_id}"' in c:
                    div = BeautifulSoup(str(c), "html.parser").find("div", id=div_id)
                    break

        if div:
            links = div.find_all("a")
            bateadores = [a.get_text(strip=True) for a in links[:-1]]
            lanzador = links[-1].get_text(strip=True) if links else None
            return bateadores, lanzador

        return [], None

    b_away, p_away = obtener_datos("lineups_1")
    b_home, p_home = obtener_datos("lineups_2")

    return b_away, b_home, p_away, p_home


def obtener_delay_adaptativo(default_seconds=0):
    """Lee el delay adaptativo persistido para la siguiente ejecución."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scraper_control (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            row = conn.execute(
                "SELECT value FROM scraper_control WHERE key = ?",
                ("adaptive_between_games_seconds",),
            ).fetchone()
            if not row:
                return default_seconds
            return int(row[0])
    except Exception:
        return default_seconds


def guardar_delay_adaptativo(seconds):
    """Guarda el delay adaptativo para la siguiente ejecución."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scraper_control (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                INSERT INTO scraper_control (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP
                """,
                ("adaptive_between_games_seconds", str(int(seconds))),
            )
            conn.commit()
    except Exception as e:
        print(f"  ⚠️ No se pudo persistir delay adaptativo: {e}")


# ============================================================================
# EJECUCIÓN DIARIA
# ============================================================================


def ejecutar_pipeline_diario():
    """
    Pipeline principal para scraping diario (VERSIÓN 2)
    Retorna True si encontró datos, False si no
    """
    fecha_bref, fecha_db, year_val = obtener_fechas_ejecucion()
    run_source = os.getenv("RUN_SOURCE", "local").strip().lower()
    max_intentos = int(os.getenv("SCRAPER_MAX_ATTEMPTS", "3"))
    espera_reintento = int(os.getenv("SCRAPER_RETRY_WAIT_SECONDS", "90"))
    permitir_guardado_parcial = os.getenv(
        "SCRAPER_SAVE_PARTIAL_ON_FINAL", "1"
    ).strip().lower() in {"1", "true", "yes", "on"}
    espera_entre_partidos_base = int(
        os.getenv("SCRAPER_BETWEEN_GAMES_SECONDS", "0")
    )
    delay_adaptativo_fallo = int(
        os.getenv("SCRAPER_ADAPTIVE_DELAY_ON_FAIL_SECONDS", "10")
    )
    espera_entre_partidos_adaptativa = obtener_delay_adaptativo(default_seconds=0)
    espera_entre_partidos = max(
        espera_entre_partidos_base, espera_entre_partidos_adaptativa
    )
    url_schedule = (
        f"https://www.baseball-reference.com/leagues/majors/{year_val}-schedule.shtml"
    )

    print(f"\n{'=' * 70}")
    print(f"📅 Ejecutando scraping automático para: {fecha_bref}")
    print("   Versión 2 (Estructura HTML Actualizada)")
    if espera_entre_partidos > 0:
        print(
            "   Modo throttling activo: "
            f"{espera_entre_partidos}s entre partidos "
            f"(base={espera_entre_partidos_base}, adaptativo={espera_entre_partidos_adaptativa})"
        )
    print(f"{'=' * 70}")

    for intento in range(1, max_intentos + 1):
        print(f"\n🔁 Intento de scraping {intento}/{max_intentos}")

        html = obtener_html(url_schedule)
        if not html:
            print("❌ No se pudo conectar a Baseball-Reference")
            if intento < max_intentos:
                print(f"⏳ Esperando {espera_reintento}s antes de reintentar...")
                time.sleep(espera_reintento)
                continue
            guardar_delay_adaptativo(delay_adaptativo_fallo)
            print("⚠️ Máximo de intentos alcanzado. Continuando workflow sin guardar.")
            return True

        soup = BeautifulSoup(html, "html.parser")

        # Extraer equipos usando la nueva estructura (span id="today")
        print("\n🔍 Buscando partidos de hoy en span id='today'...")
        equipos_hoy = extraer_equipos_del_dia(soup)

        if not equipos_hoy:
            print(
                "\n⚠️ No se encontraron partidos listados para hoy usando la nueva estructura."
            )
            print("   (Los lineups podrían no estar publicados aún)")
            if intento < max_intentos:
                print(f"⏳ Esperando {espera_reintento}s antes de reintentar...")
                time.sleep(espera_reintento)
                continue
            guardar_delay_adaptativo(delay_adaptativo_fallo)
            print("⚠️ Máximo de intentos alcanzado. Continuando workflow sin guardar.")
            return True

        data_partidos = []
        errores_partidos = []

        total_juegos = len(equipos_hoy)
        for i, (away_team, home_team, preview_link) in enumerate(equipos_hoy, start=1):
            try:
                print(f"\n⚾ Procesando: {away_team} @ {home_team}")

                if not preview_link:
                    print("   ⚠️ No se encontró link de preview para este partido")
                    errores_partidos.append(f"{away_team}@{home_team}:sin_preview")
                    continue

                # Extraer lanzadores del preview
                print("   🔍 Extrayendo lanzadores de preview...")
                lanzadores = extraer_lanzadores_del_preview(preview_link)

                away_pitcher = lanzadores.get(away_team, {}).get("nombre") if isinstance(lanzadores.get(away_team), dict) else lanzadores.get(away_team)
                away_pitcher_link = lanzadores.get(away_team, {}).get("link", "") if isinstance(lanzadores.get(away_team), dict) else ""
                home_pitcher = lanzadores.get(home_team, {}).get("nombre") if isinstance(lanzadores.get(home_team), dict) else lanzadores.get(home_team)
                home_pitcher_link = lanzadores.get(home_team, {}).get("link", "") if isinstance(lanzadores.get(home_team), dict) else ""

                if not away_pitcher or not home_pitcher:
                    print(
                        f"   ⚠️ No se encontraron lanzadores: {away_pitcher} vs {home_pitcher}"
                    )
                    errores_partidos.append(
                        f"{away_team}@{home_team}:lanzadores_incompletos"
                    )
                    continue

                print(f"   ✅ Lanzadores encontrados: {away_pitcher} vs {home_pitcher}")

                away_stats_code = get_team_code(away_team) or away_team
                home_stats_code = get_team_code(home_team) or home_team

                # Extraer stats de los lanzadores
                print(f"   🔍 Buscando stats de {away_pitcher}...")
                s_away = encontrar_lanzador(
                    scrape_player_stats(away_stats_code, year_val), away_pitcher
                )
                time.sleep(SCRAPING_CONFIG["min_delay"])

                print(f"   🔍 Buscando stats de {home_pitcher}...")
                s_home = encontrar_lanzador(
                    scrape_player_stats(home_stats_code, year_val), home_pitcher
                )
                time.sleep(SCRAPING_CONFIG["min_delay"])

                # Crear game_id unificado
                game_id = f"{fecha_db}_{home_team}_{away_team}"

                # Guardar datos del partido
                data_partidos.append(
                    {
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
                        "away_starter_ERA": s_away["ERA"] if s_away else 0.0,
                        "away_starter_WHIP": s_away["WHIP"] if s_away else 0.0,
                        "away_starter_H9": s_away["H9"] if s_away else 0.0,
                        "away_starter_SO9": s_away["SO9"] if s_away else 0.0,
                        "away_starter_W": s_away["W"] if s_away else 0,
                        "away_starter_L": s_away["L"] if s_away else 0,
                        "home_starter_ERA": s_home["ERA"] if s_home else 0.0,
                        "home_starter_WHIP": s_home["WHIP"] if s_home else 0.0,
                        "home_starter_H9": s_home["H9"] if s_home else 0.0,
                        "home_starter_SO9": s_home["SO9"] if s_home else 0.0,
                        "home_starter_W": s_home["W"] if s_home else 0,
                        "home_starter_L": s_home["L"] if s_home else 0,
                    }
                )

                print("   ✅ Partido procesado exitosamente")

            except Exception as e:
                print(f"  ⚠️ Error en partido {away_team} @ {home_team}: {e}")
                errores_partidos.append(f"{away_team}@{home_team}:exception")
            finally:
                if espera_entre_partidos > 0 and i < total_juegos:
                    print(
                        f"   ⏳ Esperando {espera_entre_partidos}s antes del siguiente partido..."
                    )
                    time.sleep(espera_entre_partidos)

        total_detectados = len(equipos_hoy)
        total_ok = len(data_partidos)
        completo = total_ok == total_detectados and total_detectados > 0

        print(
            f"\n📊 Resumen intento {intento}: {total_ok}/{total_detectados} partidos completos"
        )

        if not completo:
            if errores_partidos:
                print("   Detalle de errores:")
                for err in errores_partidos:
                    print(f"   - {err}")

            if intento < max_intentos:
                print(
                    f"⏳ Datos incompletos. Esperando {espera_reintento}s para reintentar..."
                )
                time.sleep(espera_reintento)
                continue

            if permitir_guardado_parcial and total_ok > 0:
                print(
                    "⚠️ Tercer intento sin datos completos. Se guardará snapshot PARCIAL "
                    f"({total_ok}/{total_detectados}) para no dejar la app sin cartelera."
                )
            else:
                print(
                    "⚠️ Tercer intento sin datos completos. No se guarda información y se continúa workflow."
                )
                guardar_delay_adaptativo(delay_adaptativo_fallo)
                return True

        # Guardar en base de datos solo si la corrida está completa.
        print(f"\n{'=' * 70}")
        if completo:
            print(f"💾 Guardando {len(data_partidos)} partidos en la base de datos...")
        else:
            print(
                "💾 Guardando snapshot PARCIAL: "
                f"{len(data_partidos)}/{total_detectados} partidos en la base de datos..."
            )

        with sqlite3.connect(DB_PATH) as conn:
            # Verificar esquema existente y migrar si es necesario
            table_info = conn.execute(
                "PRAGMA table_info(historico_partidos)"
            ).fetchall()
            if not table_info or len(table_info) < 20:
                print(
                    "  ℹ️ Creando o recreando tabla historico_partidos con esquema completo"
                )
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
            else:
                print("  ℹ️ Usando tabla historico_partidos existente")

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

            conn.execute(
                "DELETE FROM historico_partidos WHERE fecha = ?",
                (fecha_db,),
            )

            # Guardar partidos (con INSERT OR REPLACE para evitar duplicados)
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
        print(f"   - Partidos guardados: {len(data_partidos)}")
        if espera_entre_partidos_adaptativa > 0:
            guardar_delay_adaptativo(0)
            print("   - Delay adaptativo reiniciado a 0s")
        print(f"{'=' * 70}\n")
        return True

    print("⚠️ No se pudo completar scraping diario tras los reintentos.")
    return True


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scraper diario de partidos MLB (Versión 2)"
    )
    parser.add_argument(
        "--retry", action="store_true", help="Modo reintentar si no hay datos"
    )
    args = parser.parse_args()

    resultado = ejecutar_pipeline_diario()

    # Retornar código de salida para GitHub Actions
    sys.exit(0 if resultado else 1)
