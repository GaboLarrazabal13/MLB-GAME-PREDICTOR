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
from mlb_schedule_utils import seleccionar_seccion_schedule

_SCRAPER_SESSION = None

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


def get_scraper_session(force_new=False):
    """Crea/reutiliza una sesión cloudscraper para mantener cookies y bajar bloqueos."""
    global _SCRAPER_SESSION
    if _SCRAPER_SESSION is None or force_new:
        _SCRAPER_SESSION = cloudscraper.create_scraper(
            browser={
                "browser": "chrome",
                "platform": "windows",
                "desktop": True,
            }
        )
        # Headers más modernos y completos
        _SCRAPER_SESSION.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer": "https://www.google.com/",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "cross-site",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0",
            }
        )
    return _SCRAPER_SESSION

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


def normalizar_team_code(code):
    """Normaliza códigos históricos/alternos a los usados en la base."""
    if not code:
        return code
    code = str(code).upper().strip()
    return TEAM_CODE_ALIASES.get(code, code)


def extraer_fecha_desde_box_url(box_url, default_fecha):
    """Intenta obtener YYYY-MM-DD desde links /boxes/TEAM/TEAMYYYYMMDD0.shtml."""
    if not box_url:
        return default_fecha
    match = re.search(r"/boxes/[A-Z]{3}/[A-Z]{3}(\d{4})(\d{2})(\d{2})\d\.shtml", box_url)
    if not match:
        return default_fecha
    year, month, day = match.groups()
    return f"{year}-{month}-{day}"


def obtener_html(url, max_retries=None):
    """Obtiene HTML con reintentos"""
    if max_retries is None:
        max_retries = SCRAPING_CONFIG["max_retries"]

    for intento in range(max_retries):
        try:
            scraper = get_scraper_session()
            response = scraper.get(url, timeout=SCRAPING_CONFIG["timeout"])

            if response.status_code == 200:
                # Verificar si es una página real o una página de Cloudflare/Bloqueo
                if "Enable JavaScript and cookies to continue" in response.text or "Just a moment..." in response.text:
                    print(f"       ⚠️ Detectado reto Cloudflare para {url}. Reintentando...")
                    get_scraper_session(force_new=True)
                    time.sleep(10)
                    continue
                
                response.encoding = "utf-8"
                return response.text

            if response.status_code in (403, 429):
                wait_time = int((2**intento) * 10) # Aumentado el tiempo base
                print(
                    f"       Status {response.status_code} para {url}. "
                    f"Esperando {wait_time}s y renovando sesión..."
                )
                if response.status_code == 403:
                    # Log breve del contenido para diagnóstico
                    snippet = response.text[:200].replace('\n', ' ')
                    print(f"       Snippet del error: {snippet}")

                get_scraper_session(force_new=True)
                time.sleep(wait_time)
                continue

            wait_time = int(2**intento)
            print(
                f"       Status {response.status_code} para {url}. "
                f"Reintento en {wait_time}s..."
            )
            time.sleep(wait_time)
        except Exception as e:
            if intento == max_retries - 1:
                print(f"       Error final obteniendo {url}: {e}")
            time.sleep(2**intento)

    return None


def obtener_fechas_ejecucion():
    """Obtiene fechas formateadas para scraping"""
    target_date = os.getenv("TARGET_DATE", "").strip()

    if target_date:
        try:
            ahora = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            print(
                f"⚠️ TARGET_DATE inválida '{target_date}'. Se usará fecha actual en ET."
            )
            ahora = datetime.now(ZoneInfo("America/New_York"))
    else:
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


def extraer_equipos_del_dia(soup, fecha_objetivo_db):
    """
    Extrae los equipos de la jornada objetivo usando la fecha MLB real.
    """
    equipos_lista = []

    seccion = seleccionar_seccion_schedule(soup, fecha_objetivo_db)
    if not seccion:
        print("  ⚠️ No se encontró ninguna sección válida en el schedule")
        return equipos_lista

    seccion_date = seccion.get('date')
    print(
        "  📍 Sección seleccionada: "
        f"{seccion.get('label', 'sin etiqueta')} -> {seccion_date}"
    )
    
    # VALIDACIÓN CRÍTICA: La sección debe coincidir con la fecha objetivo
    if seccion_date != fecha_objetivo_db:
        if seccion.get("is_today") and seccion_date is None:
            print(f"  ℹ️ Usando sección 'Today's Games' para la fecha {fecha_objetivo_db}")
            seccion_date = fecha_objetivo_db
        else:
            print(f"  ❌ ERROR: La sección encontrada ({seccion_date}) NO coinciden con la fecha objetivo ({fecha_objetivo_db})")
            print("  Esto indica que Baseball-Reference aún no ha publicado el calendario de hoy o el scraper está viendo una versión antigua.")
            return []

    for game in seccion["games"]:
        try:
            away_team = normalizar_team_code(game["away_team"])
            home_team = normalizar_team_code(game["home_team"])
            preview_link = game.get("game_link")

            equipos_lista.append((away_team, home_team, preview_link))
            print(f"  ✅ Encontrado: {away_team} @ {home_team}")
        except Exception as e:
            print(f"  ⚠️ Error extrayendo equipos: {e}")

    return equipos_lista


def extraer_lanzadores_del_preview(preview_url, away_team=None, home_team=None):
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

    away_team = normalizar_team_code(away_team)
    home_team = normalizar_team_code(home_team)

    # Buscar todos los divs con class que contenga "assoc_sp_" (estructura preview clásica)
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
                team_code = normalizar_team_code(team_match.group(1))

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

    if lanzadores:
        return lanzadores

    # Fallback: cuando el link ya es /boxes/... usar lineups_1 y lineups_2.
    def obtener_div_desde_soup_o_comentarios(div_id):
        div = soup.find("div", id=div_id)
        if div:
            return div
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for c in comments:
            if f'id="{div_id}"' in c:
                return BeautifulSoup(str(c), "html.parser").find("div", id=div_id)
        return None

    div_away = obtener_div_desde_soup_o_comentarios("lineups_1")
    div_home = obtener_div_desde_soup_o_comentarios("lineups_2")

    def extraer_pitcher_div(div):
        if not div:
            return None, ""
        links = div.find_all("a")
        if not links:
            return None, ""
        pitcher_a = links[-1]
        return pitcher_a.get_text(strip=True), pitcher_a.get("href", "")

    away_name, away_link = extraer_pitcher_div(div_away)
    home_name, home_link = extraer_pitcher_div(div_home)

    if away_team and away_name:
        lanzadores[away_team] = {"nombre": away_name, "link": away_link}
        print(f"     🎯 Lanzador {away_team}: {away_name} | Link: {away_link}")
    if home_team and home_name:
        lanzadores[home_team] = {"nombre": home_name, "link": home_link}
        print(f"     🎯 Lanzador {home_team}: {home_name} | Link: {home_link}")

    return lanzadores


def scrape_player_stats(team_code, year):
    """Extrae estadísticas de pitcheo de un equipo"""
    team_code = normalizar_team_code(team_code)
    url = f"https://www.baseball-reference.com/teams/{team_code}/{year}.shtml"
    html = obtener_html(url)

    if not html:
        alt_code = get_team_code(team_code)
        alt_code = normalizar_team_code(alt_code)
        if alt_code and alt_code != team_code:
            alt_url = f"https://www.baseball-reference.com/teams/{alt_code}/{year}.shtml"
            html = obtener_html(alt_url)

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
        print(f"\n🔍 Buscando partidos para la fecha MLB objetivo: {fecha_db}...")
        equipos_hoy = extraer_equipos_del_dia(soup, fecha_db)

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

        def registrar_partido_pendiente(
            data_partidos_ref,
            errores_partidos_ref,
            away_team_val,
            home_team_val,
            motivo,
            fecha_partido_db,
            away_pitcher="",
            home_pitcher="",
            away_pitcher_link="",
            home_pitcher_link="",
            box_url=None,
        ):
            year_partido = int(fecha_partido_db[:4])
            game_id = f"{fecha_partido_db}_{home_team_val}_{away_team_val}"
            data_partidos_ref.append(
                {
                    "game_id": game_id,
                    "box_score_url": box_url,
                    "fecha": fecha_partido_db,
                    "year": year_partido,
                    "away_team": away_team_val,
                    "home_team": home_team_val,
                    "away_pitcher": away_pitcher or "",
                    "home_pitcher": home_pitcher or "",
                    "away_pitcher_link": away_pitcher_link or "",
                    "home_pitcher_link": home_pitcher_link or "",
                    "away_starter_ERA": 0.0,
                    "away_starter_WHIP": 0.0,
                    "away_starter_H9": 0.0,
                    "away_starter_SO9": 0.0,
                    "away_starter_W": 0,
                    "away_starter_L": 0,
                    "home_starter_ERA": 0.0,
                    "home_starter_WHIP": 0.0,
                    "home_starter_H9": 0.0,
                    "home_starter_SO9": 0.0,
                    "home_starter_W": 0,
                    "home_starter_L": 0,
                }
            )
            errores_partidos_ref.append(f"{away_team_val}@{home_team_val}:{motivo}")
            print("   ⚠️ Partido guardado como pendiente " f"({motivo}).")

        total_juegos = len(equipos_hoy)
        for i, (away_team, home_team, preview_link) in enumerate(equipos_hoy, start=1):
            try:
                print(f"\n⚾ Procesando: {away_team} @ {home_team}")

                if not preview_link:
                    print("   ⚠️ No se encontró link de preview para este partido")
                    registrar_partido_pendiente(
                        data_partidos,
                        errores_partidos,
                        away_team,
                        home_team,
                        "sin_preview",
                        fecha_db,
                        box_url=None,
                    )
                    continue

                # Extraer lanzadores del preview
                print("   🔍 Extrayendo lanzadores de preview...")
                lanzadores = extraer_lanzadores_del_preview(
                    preview_link, away_team=away_team, home_team=home_team
                )

                away_pitcher = lanzadores.get(away_team, {}).get("nombre") if isinstance(lanzadores.get(away_team), dict) else lanzadores.get(away_team)
                away_pitcher_link = lanzadores.get(away_team, {}).get("link", "") if isinstance(lanzadores.get(away_team), dict) else ""
                home_pitcher = lanzadores.get(home_team, {}).get("nombre") if isinstance(lanzadores.get(home_team), dict) else lanzadores.get(home_team)
                home_pitcher_link = lanzadores.get(home_team, {}).get("link", "") if isinstance(lanzadores.get(home_team), dict) else ""

                if not away_pitcher or not home_pitcher:
                    print(
                        f"   ⚠️ No se encontraron lanzadores: {away_pitcher} vs {home_pitcher}"
                    )
                    fecha_partido_db = extraer_fecha_desde_box_url(preview_link, fecha_db)
                    registrar_partido_pendiente(
                        data_partidos,
                        errores_partidos,
                        away_team,
                        home_team,
                        "lanzadores_incompletos",
                        fecha_partido_db,
                        away_pitcher=away_pitcher or "",
                        home_pitcher=home_pitcher or "",
                        away_pitcher_link=away_pitcher_link,
                        home_pitcher_link=home_pitcher_link,
                        box_url=preview_link,
                    )
                    continue

                print(f"   ✅ Lanzadores encontrados: {away_pitcher} vs {home_pitcher}")

                fecha_partido_db = extraer_fecha_desde_box_url(preview_link, fecha_db)
                if fecha_partido_db != fecha_db:
                    print(
                        "   ℹ️ Ajuste de fecha por boxscore URL: "
                        f"{fecha_db} -> {fecha_partido_db}"
                    )

                year_partido = int(fecha_partido_db[:4])

                away_stats_code = away_team or get_team_code(away_team)
                home_stats_code = home_team or get_team_code(home_team)

                # Extraer stats de los lanzadores
                print(f"   🔍 Buscando stats de {away_pitcher}...")
                s_away = encontrar_lanzador(
                    scrape_player_stats(away_stats_code, year_partido), away_pitcher
                )
                time.sleep(SCRAPING_CONFIG["min_delay"])

                print(f"   🔍 Buscando stats de {home_pitcher}...")
                s_home = encontrar_lanzador(
                    scrape_player_stats(home_stats_code, year_partido), home_pitcher
                )
                time.sleep(SCRAPING_CONFIG["min_delay"])

                # Crear game_id unificado
                game_id = f"{fecha_partido_db}_{home_team}_{away_team}"

                # Guardar datos del partido
                data_partidos.append(
                    {
                        "game_id": game_id,
                        "box_score_url": preview_link,
                        "fecha": fecha_partido_db,
                        "year": year_partido,
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
                registrar_partido_pendiente(
                    data_partidos,
                    errores_partidos,
                    away_team,
                    home_team,
                    "exception",
                    fecha_db,
                    box_url=preview_link,
                )
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
            if not table_info or len(table_info) < 22:
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
            fechas_guardadas = sorted(set(df_partidos["fecha"].tolist()))
            for fecha_guardada in fechas_guardadas:
                conn.execute(
                    "DELETE FROM historico_partidos WHERE fecha = ?",
                    (fecha_guardada,),
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
                ("games_today", run_source, max(fechas_guardadas)),
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
