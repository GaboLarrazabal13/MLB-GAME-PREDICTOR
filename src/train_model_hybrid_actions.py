"""
Script de Entrenamiento ML Híbrido OPTIMIZADO para Predicción MLB - VERSIÓN V3.5 REFACTORIZADA
LÓGICA HÍBRIDA INTELIGENTE:
- Partidos antiguos (2022-2024): Solo features temporales (CSV)
- Partidos recientes (2026): Features temporales + scraping
Sistema de bloques para evitar rate limiting
Cache incremental para entrenar por etapas
REFACTORIZACIÓN: Código centralizado, sin duplicación, mejor mantenibilidad
"""

import os
import pickle
import random
import re
import shutil
import sqlite3
import time
import unicodedata
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

import cloudscraper
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, Comment
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Importar módulos centralizados
from mlb_config import (
    CACHE_PATH,
    DB_PATH,
    MODEL_CONFIG,
    MODELO_BACKUP,
    MODELO_PATH,
    SCRAPING_CONFIG,
    SCRAPING_FEATURES,
    SUPER_FEATURES,
    TEMPORAL_FEATURES,
    get_team_code,
)
from mlb_feature_engineering import calcular_super_features

warnings.filterwarnings("ignore")


def alinear_features_entrenamiento(X_new, model_actual=None):
    """Asegura un esquema estable de features para reentrenamiento incremental."""
    features_esperadas = TEMPORAL_FEATURES + SCRAPING_FEATURES + SUPER_FEATURES

    columnas_faltantes = [col for col in features_esperadas if col not in X_new.columns]
    if columnas_faltantes:
        print(f"ℹ️ Añadiendo columnas faltantes con 0 para mantener esquema estable: {columnas_faltantes}")
        for columna in columnas_faltantes:
            X_new[columna] = 0

    if model_actual is None:
        return X_new

    feature_names_modelo = model_actual.get_booster().feature_names or []
    if not feature_names_modelo:
        return X_new

    columnas_faltantes_modelo = [col for col in feature_names_modelo if col not in X_new.columns]
    if columnas_faltantes_modelo:
        print(f"ℹ️ Añadiendo columnas faltantes requeridas por el modelo previo: {columnas_faltantes_modelo}")
        for columna in columnas_faltantes_modelo:
            X_new[columna] = 0

    columnas_extra_modelo = [col for col in X_new.columns if col not in feature_names_modelo]
    if columnas_extra_modelo:
        print(
            "ℹ️ Excluyendo columnas que no existen en el modelo previo para el "
            f"reentrenamiento incremental: {columnas_extra_modelo}"
        )

    return X_new.reindex(columns=feature_names_modelo, fill_value=0)


# ============================================================================
# FUNCIONES DE SCRAPING CON REINTENTOS
# ============================================================================


def obtener_html(
    url,
    max_retries=None,
    rate_limit_wait_override=None,
    fast_fail_403=False,
):
    """Obtiene HTML con reintentos y backoff exponencial"""
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
            elif response.status_code == 429:
                wait_time = (2**intento) * 5
                print(f"       Rate limit (429) detectado, esperando {wait_time}s...")
                time.sleep(wait_time)
            elif response.status_code == 403:
                if fast_fail_403:
                    print("       Error 403 (Forbidden) detectado, fast-fail activado.")
                    return None

                wait_time = (
                    rate_limit_wait_override
                    if rate_limit_wait_override is not None
                    else SCRAPING_CONFIG["rate_limit_wait"]
                )
                print(f"       Error 403 (Forbidden), esperando {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"       Error {response.status_code} al obtener URL {url}")
                if intento < max_retries - 1:
                    time.sleep(2**intento)

        except Exception as e:
            if intento == max_retries - 1:
                print(f"       Error final al obtener URL {url}: {str(e)}")
            time.sleep(2**intento)

    return None


def limpiar_dataframe(df):
    """Limpia dataframes de Baseball-Reference eliminando basura"""
    if df is None or len(df) == 0:
        return df

    if "Rk" in df.columns:
        df = df.drop("Rk", axis=1)

    name_col = df.columns[0]
    df = df.dropna(subset=[name_col])
    df = df[~df[name_col].astype(str).str.contains(r"Team Totals|Rank in|^\s*$", case=False, na=False, regex=True)]

    return df.reset_index(drop=True)


def scrape_player_stats(
    team_code,
    year,
    session_cache=None,
    max_retries_override=None,
    fast_fail_403=False,
):
    """Scrapea bateo y pitcheo de un equipo con caché de sesión"""
    team_code = str(team_code).upper().strip()

    if session_cache is not None:
        cache_key = f"{team_code}_{year}"
        if cache_key in session_cache:
            return session_cache[cache_key]

    url = f"https://www.baseball-reference.com/teams/{team_code}/{year}.shtml"
    if team_code == "OAK":
        # Fallback solicitado: si OAK falla 2 veces, intentar ATH.
        html = obtener_html(
            url,
            max_retries=max_retries_override if max_retries_override is not None else 2,
            fast_fail_403=fast_fail_403,
        )
        if not html:
            alt_code = "ATH"
            alt_url = f"https://www.baseball-reference.com/teams/{alt_code}/{year}.shtml"
            print("       OAK sin respuesta tras 2 intentos, probando ATH...")
            html = obtener_html(
                alt_url,
                max_retries=max_retries_override,
                fast_fail_403=fast_fail_403,
            )
            if html:
                team_code = alt_code
    else:
        html = obtener_html(
            url,
            max_retries=max_retries_override,
            fast_fail_403=fast_fail_403,
        )

    if not html:
        return None, None

    try:
        soup = BeautifulSoup(html, "html.parser")
        batting_table = soup.find("table", {"id": "players_standard_batting"})
        pitching_table = soup.find("table", {"id": "players_standard_pitching"})

        batting_df = None
        pitching_df = None

        if batting_table:
            batting_df = pd.read_html(str(batting_table))[0]
            batting_df = limpiar_dataframe(batting_df)

        if pitching_table:
            pitching_df = pd.read_html(str(pitching_table))[0]
            pitching_df = limpiar_dataframe(pitching_df)

        if session_cache is not None:
            session_cache[f"{team_code}_{year}"] = (batting_df, pitching_df)

        return batting_df, pitching_df

    except Exception as e:
        print(f"       Error procesando tablas para {team_code}: {e}")
        return None, None


def obtener_stats_pitcher_por_link(
    pitcher_link,
    target_year=None,
    session_cache=None,
    max_retries_override=None,
    fast_fail_403=False,
):
    """
    Obtiene stats de un pitcher directamente por su link individual.
    Usa la tabla histórica del jugador para buscar una temporada específica.
    """
    if not pitcher_link:
        return None

    if session_cache is not None:
        cache_key = f"pitcher::{pitcher_link}::{target_year}"
        if cache_key in session_cache:
            return session_cache[cache_key]

    try:
        url = str(pitcher_link).strip()
        url = url.replace("baseball-reference.coom", "baseball-reference.com")
        url = url.replace("baseball-reference.comm", "baseball-reference.com")
        if pitcher_link.startswith("/"):
            url = f"https://www.baseball-reference.com{pitcher_link}"

        html = obtener_html(
            url,
            max_retries=max_retries_override,
            fast_fail_403=fast_fail_403,
        )

        if not html:
            print(f"       ⚠️ No se pudo obtener HTML del pitcher: {pitcher_link}")
            return None

        soup = BeautifulSoup(html, "html.parser")

        def buscar_tabla(table_id):
            tabla = soup.find("table", {"id": table_id})
            if tabla:
                return tabla

            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for comment in comments:
                if f'id="{table_id}"' in comment:
                    return BeautifulSoup(str(comment), "html.parser").find("table", {"id": table_id})
            return None

        pitching_hist_table = buscar_tabla("players_standard_pitching")
        pitching_st_table = buscar_tabla("pitching_st")

        if not pitching_hist_table and not pitching_st_table:
            print(f"       ⚠️ No se encontró tabla de pitcheo utilizable para: {pitcher_link}")
            return None

        meta_title = soup.select_one("#meta h1 span")
        nombre_real = meta_title.get_text(strip=True) if meta_title else ""

        row = None

        if pitching_hist_table is not None:
            hist_df = pd.read_html(str(pitching_hist_table))[0]
            hist_df = limpiar_dataframe(hist_df)
            if not hist_df.empty and "Season" in hist_df.columns:
                season_series = pd.to_numeric(hist_df["Season"], errors="coerce")
                hist_df = hist_df[season_series.notna()].copy()
                if not hist_df.empty:
                    hist_df["Season_num"] = pd.to_numeric(hist_df["Season"], errors="coerce")
                    if target_year is not None:
                        year_match = hist_df[hist_df["Season_num"] == int(target_year)]
                        if not year_match.empty:
                            row = year_match.iloc[0]
                    if row is None and target_year is None:
                        row = hist_df.iloc[-1]

        if row is None and pitching_st_table is not None:
            current_df = pd.read_html(str(pitching_st_table))[0]
            current_df = limpiar_dataframe(current_df)
            if not current_df.empty:
                if target_year is not None and "Year" in current_df.columns:
                    year_series = pd.to_numeric(current_df["Year"], errors="coerce")
                    year_match = current_df[year_series == int(target_year)]
                    if not year_match.empty:
                        row = year_match.iloc[0]
                if row is None and target_year is None:
                    row = current_df.iloc[0]

        if row is None:
            return None

        stats = {
            "nombre_real": nombre_real,
            "ERA": safe_float(row.get("ERA", 0)),
            "WHIP": safe_float(row.get("WHIP", 0)),
            "H9": safe_float(row.get("H9", row.get("H/9", 0))),
            "SO9": safe_float(row.get("SO9", 0)),
            "W": safe_int(row.get("W", 0)),
            "L": safe_int(row.get("L", 0)),
            "IP": safe_float(row.get("IP", 0)),
            "G": safe_int(row.get("G", 0)),
            "GS": safe_int(row.get("GS", row.get("GS.1", 0))),
        }

        if session_cache is not None:
            session_cache[cache_key] = stats

        return stats

    except Exception as e:
        print(f"       ⚠️ Error obteniendo stats del pitcher {pitcher_link}: {e}")

    return None


# ============================================================================
# IDENTIFICAR TOP 3 RELEVISTAS
# ============================================================================


def extraer_top_relevistas(pitching_df):
    """Identifica al Closer y los mejores Setup men basándose en Saves y Juegos Finalizados"""
    if pitching_df is None or len(pitching_df) == 0:
        return None

    # Asegurar conversión numérica para columnas de relevo
    cols_relevo = ["SV", "GF", "ERA", "WHIP", "IP", "SO", "G", "GS"]
    for col in cols_relevo:
        if col in pitching_df.columns:
            pitching_df[col] = pd.to_numeric(pitching_df[col], errors="coerce").fillna(0)

    # Filtrar relevistas: GS < 50% de sus juegos
    bullpen = pitching_df[pitching_df["GS"] < (pitching_df["G"] * 0.5)].copy()
    if len(bullpen) == 0:
        return None

    # Ordenar por jerarquía: Saves (Closer), Juegos Finalizados (Setup), IP
    bullpen = bullpen.sort_values(by=["SV", "GF", "IP"], ascending=False)
    top_3 = bullpen.head(3)

    return {
        "bullpen_ERA_mean": top_3["ERA"].mean(),
        "bullpen_WHIP_mean": top_3["WHIP"].mean(),
    }


# ============================================================================
# FUNCIONES DE EXTRACCIÓN Y CÁLCULO
# ============================================================================


def normalizar_texto(texto):
    """Normaliza texto para comparaciones de nombres"""
    if not texto:
        return ""
    texto = str(texto).lower()
    texto = "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^a-z0-9]", "", texto)
    return texto


def safe_float(val):
    """Convierte a float de forma segura manejando errores y NaNs"""
    try:
        if pd.isna(val):
            return 0.0
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def safe_int(val):
    """Convierte a int de forma segura manejando errores y NaNs"""
    try:
        if pd.isna(val):
            return 0
        return int(float(val))
    except (ValueError, TypeError):
        return 0


def encontrar_lanzador(pitching_df, nombre_lanzador):
    """Busca un lanzador específico usando normalización agresiva"""
    if pitching_df is None or len(pitching_df) == 0:
        return None

    nombre_busqueda = normalizar_texto(nombre_lanzador)
    name_col = pitching_df.columns[0]

    mask = pitching_df[name_col].apply(
        lambda x: (nombre_busqueda in normalizar_texto(x) or normalizar_texto(x) in nombre_busqueda)
    )

    if mask.sum() == 0:
        return None

    lanzador = pitching_df[mask].iloc[0]

    return {
        "ERA": safe_float(lanzador.get("ERA", 0)),
        "WHIP": safe_float(lanzador.get("WHIP", 0)),
        "H9": safe_float(lanzador.get("H9", 0)),
        "SO9": safe_float(lanzador.get("SO9", 0)),
        "W": safe_float(lanzador.get("W", 0)),
        "L": safe_float(lanzador.get("L", 0)),
        "IP": safe_float(lanzador.get("IP", 0)),
        "G": safe_float(lanzador.get("G", 0)),
        "GS": safe_float(lanzador.get("GS", 0)),
        "nombre_real": str(lanzador[name_col]),
    }


def encontrar_mejor_bateador(batting_df):
    """Encuentra estadísticas de los mejores bateadores del equipo"""
    if batting_df is None or len(batting_df) == 0:
        return None

    if "OBP" not in batting_df.columns or "AB" not in batting_df.columns:
        return None

    df = batting_df.copy()
    df["OBP"] = pd.to_numeric(df["OBP"], errors="coerce")
    df["AB"] = pd.to_numeric(df["AB"], errors="coerce")
    df = df.dropna(subset=["OBP", "AB"])

    if len(df) == 0:
        return None

    mediana_ab = df["AB"].median()
    df_filtrado = df[df["AB"] >= mediana_ab].copy()

    if len(df_filtrado) == 0:
        df_filtrado = df

    top_3 = df_filtrado.sort_values("OBP", ascending=False).head(3)

    name_col = top_3.columns[0]

    detalles = []
    for _, row in top_3.iterrows():
        detalles.append(
            {
                "n": str(row[name_col]),
                "ba": safe_float(row.get("BA", 0)),
                "obp": safe_float(row.get("OBP", 0)),
                "slg": safe_float(row.get("SLG", 0)),
                "ops": safe_float(row.get("OPS", 0)),
                "hr": safe_float(row.get("HR", 0)),
                "rbi": safe_float(row.get("RBI", 0)),
            }
        )

    return {
        "best_bat_BA": pd.to_numeric(top_3["BA"], errors="coerce").mean(),
        "best_bat_OBP": top_3["OBP"].mean(),
        "best_bat_OPS": pd.to_numeric(top_3["OPS"], errors="coerce").mean() if "OPS" in top_3.columns else 0.750,
        "best_bat_HR": pd.to_numeric(top_3["HR"], errors="coerce").mean(),
        "best_bat_RBI": pd.to_numeric(top_3["RBI"], errors="coerce").mean(),
        "detalles_visuales": detalles,
    }


def calcular_stats_equipo(batting_df, pitching_df):
    """Calcula promedios generales del equipo (Bateo y Pitcheo)"""
    stats = {}

    if batting_df is not None and len(batting_df) > 0:
        for col in ["BA", "OBP", "SLG", "OPS", "HR", "RBI"]:
            if col in batting_df.columns:
                val = pd.to_numeric(batting_df[col], errors="coerce").mean()
                stats[f"team_{col}_mean"] = val if not pd.isna(val) else 0

    if pitching_df is not None and len(pitching_df) > 0:
        for col in ["ERA", "WHIP", "SO9", "H9", "BB9"]:
            if col in pitching_df.columns:
                val = pd.to_numeric(pitching_df[col], errors="coerce").mean()
                stats[f"team_{col}_mean"] = val if not pd.isna(val) else 0

    return stats


# ============================================================================
# FUNCIONES DE TENDENCIAS TEMPORALES
# ============================================================================


def calcular_tendencias_equipo(df, team, fecha_limite, ventana=10):
    """Calcula rendimiento reciente (ventana) y de toda la temporada de un equipo"""
    from mlb_config import get_team_code, get_team_name

    if isinstance(fecha_limite, str):
        fecha_limite = pd.to_datetime(fecha_limite)

    # Normalizar equipo
    t_code = get_team_code(team)
    t_full = get_team_name(t_code)

    # Normalizar columnas de equipo en el DF para evitar fallos por espacios o mayúsculas
    df = df.copy()
    df["home_team_norm"] = df["home_team"].astype(str).str.strip()
    df["away_team_norm"] = df["away_team"].astype(str).str.strip()

    # Filtrar todos los partidos de la misma temporada antes de la fecha límite
    anio_partido = fecha_limite.year
    mask_season = (
        ((df["home_team_norm"].isin([t_code, t_full])) | (df["away_team_norm"].isin([t_code, t_full])))
        & (pd.to_datetime(df["fecha"]) < fecha_limite)
        & (pd.to_datetime(df["fecha"]).dt.year == anio_partido)
    )

    partidos_todos = df[mask_season].sort_values("fecha", ascending=False)
    partidos_recientes = partidos_todos.head(ventana)

    if len(partidos_todos) == 0:
        return {
            "victorias_recientes": 0.5,
            "win_rate_season": 0.5,
            "carreras_anotadas_avg": 4.5,
            "carreras_recibidas_avg": 4.5,
            "racha_actual": 0,
            "diferencial_carreras": 0,
            "total_juegos_season": 0,
            "wins_season": 0,
            "losses_season": 0,
        }

    # Estadísticas Ventana (L10)
    victorias_l10 = 0
    carreras_f_l10 = 0
    carreras_c_l10 = 0

    for _, p in partidos_recientes.iterrows():
        es_home = p["home_team_norm"] in [t_code, t_full]
        ganador_val = p.get("ganador", 0)
        if ganador_val is None:
            ganador_val = 0
        ganado = (ganador_val == 1) if es_home else (ganador_val == 0)
        if ganado:
            victorias_l10 += 1
        carreras_f_l10 += float(p.get("score_home", 0) if es_home else p.get("score_away", 0) or 0)
        carreras_c_l10 += float(p.get("score_away", 0) if es_home else p.get("score_home", 0) or 0)

    # Estadísticas Temporada
    victorias_season = 0
    for _, p in partidos_todos.iterrows():
        es_home = p["home_team_norm"] in [t_code, t_full]
        ganador_val = p.get("ganador", 0)
        if ganador_val is None:
            ganador_val = 0
        ganado = (ganador_val == 1) if es_home else (ganador_val == 0)
        if ganado:
            victorias_season += 1

    win_rate_season = victorias_season / len(partidos_todos)

    # Cálculo de racha
    racha = 0
    for _, p in partidos_recientes.iterrows():
        es_home = p["home_team_norm"] in [t_code, t_full]
        ganado = (p["ganador"] == 1) if es_home else (p["ganador"] == 0)
        if racha == 0:
            racha = 1 if ganado else -1
        elif (racha > 0 and ganado) or (racha < 0 and not ganado):
            racha += 1 if ganado else -1
        else:
            break

    n_win = len(partidos_recientes)
    return {
        "victorias_recientes": victorias_l10 / n_win if n_win > 0 else 0.5,
        "win_rate_season": win_rate_season,
        "carreras_anotadas_avg": carreras_f_l10 / n_win if n_win > 0 else 4.5,
        "carreras_recibidas_avg": carreras_c_l10 / n_win if n_win > 0 else 4.5,
        "racha_actual": racha,
        "diferencial_carreras": (carreras_f_l10 - carreras_c_l10) / n_win if n_win > 0 else 0,
        "total_juegos_season": len(partidos_todos),
        "wins_season": victorias_season,
        "losses_season": len(partidos_todos) - victorias_season,
    }


# ============================================================================
# FUNCIONES DE CONTROL DE BASE DE DATOS
# ============================================================================


# Milestones fijos de la temporada regular: 2430 juegos / 5 cortes = 486 por bloque
MILESTONES_TEMPORADA = [486, 972, 1458, 1944, 2430]


def obtener_todos_los_juegos_temporada():
    """Devuelve todos los juegos de historico_real para la temporada actual, ordenados por fecha."""
    temporada_objetivo = int(os.getenv("TRAINING_SEASON_YEAR", datetime.now(ZoneInfo("America/New_York")).year))
    with sqlite3.connect(DB_PATH) as conn:
        df_real = pd.read_sql(
            "SELECT * FROM historico_real WHERE substr(fecha, 1, 4) = ? ORDER BY fecha",
            conn,
            params=[str(temporada_objetivo)],
        )

    if df_real.empty:
        return df_real

    df_real["home_team"] = df_real["home_team"].str.strip()
    df_real["away_team"] = df_real["away_team"].str.strip()
    return df_real


def verificar_milestone_reentrenamiento():
    """
    Comprueba si el total de registros en historico_real alcanzó el próximo milestone
    de la temporada (486 / 972 / 1458 / 1944 / 2430).
    Devuelve (should_train: bool, total: int, next_milestone: int | None).
    """
    temporada_objetivo = int(os.getenv("TRAINING_SEASON_YEAR", datetime.now(ZoneInfo("America/New_York")).year))
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS metadata_entrenamiento (
                   key TEXT PRIMARY KEY,
                   value TEXT
               )"""
        )
        total = conn.execute(
            "SELECT COUNT(*) FROM historico_real WHERE substr(fecha,1,4) = ?",
            (str(temporada_objetivo),),
        ).fetchone()[0]

        row = conn.execute("SELECT value FROM metadata_entrenamiento WHERE key = 'last_retrain_milestone'").fetchone()
        try:
            last_milestone = int(row[0]) if row and row[0] else 0
        except Exception:
            last_milestone = 0

    next_milestone = next((m for m in MILESTONES_TEMPORADA if m > last_milestone), None)

    if next_milestone is None:
        print(f"✅ Todos los milestones de temporada completados. Total en BD: {total}")
        return False, total, None

    should_train = total >= next_milestone
    print(
        f"📊 Juegos temporada {temporada_objetivo}: {total} "
        f"| Último milestone: {last_milestone} "
        f"| Próximo milestone: {next_milestone} "
        f"| Entrenar: {should_train}"
    )
    return should_train, total, next_milestone


def registrar_milestone_cumplido(milestone):
    """Registra que el milestone fue procesado (con o sin mejora de accuracy)."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS metadata_entrenamiento (
                   key TEXT PRIMARY KEY,
                   value TEXT
               )"""
        )
        conn.execute(
            "INSERT OR REPLACE INTO metadata_entrenamiento (key, value) VALUES (?, ?)",
            ("last_retrain_milestone", str(milestone)),
        )
        conn.commit()
    print(f"🎯 Milestone {milestone}/2430 registrado como completado.")


# ============================================================================
# EXTRACCIÓN DE FEATURES HÍBRIDA
# ============================================================================


def extraer_features_hibridas(row, df_historico=None, hacer_scraping=False, session_cache=None):
    """Extrae features combinando tendencias temporales y scraping"""
    features = {}

    # 1. TENDENCIAS TEMPORALES
    if df_historico is not None:
        fecha_dt = pd.to_datetime(row["fecha"])

        trend_h = calcular_tendencias_equipo(df_historico, row["home_team"], fecha_dt, ventana=10)
        trend_a = calcular_tendencias_equipo(df_historico, row["away_team"], fecha_dt, ventana=10)

        features["home_win_rate_10"] = trend_h.get("victorias_recientes", 0.5)
        features["home_win_rate_season"] = trend_h.get("win_rate_season", 0.5)
        features["home_racha"] = trend_h.get("racha_actual", 0)
        features["home_runs_avg"] = trend_h.get("carreras_anotadas_avg", 4.5)
        features["home_runs_diff"] = trend_h.get("diferencial_carreras", 0)
        features["home_season_record"] = f"{trend_h.get('wins_season', 0)}-{trend_h.get('losses_season', 0)}"

        features["away_win_rate_10"] = trend_a.get("victorias_recientes", 0.5)
        features["away_win_rate_season"] = trend_a.get("win_rate_season", 0.5)
        features["away_racha"] = trend_a.get("racha_actual", 0)
        features["away_runs_avg"] = trend_a.get("carreras_anotadas_avg", 4.5)
        features["away_runs_diff"] = trend_a.get("diferencial_carreras", 0)
        features["away_season_record"] = f"{trend_a.get('wins_season', 0)}-{trend_a.get('losses_season', 0)}"

    # 2. SCRAPING Y STATS DE JUGADORES
    if hacer_scraping:
        home_code = get_team_code(row["home_team"].strip())
        away_code = get_team_code(row["away_team"].strip())

        if not home_code:
            home_code = row["home_team"]
        if not away_code:
            away_code = row["away_team"]

        bat1, pit1 = scrape_player_stats(home_code, row["year"], session_cache)
        time.sleep(random.uniform(SCRAPING_CONFIG["min_delay"], SCRAPING_CONFIG["max_delay"]))
        bat2, pit2 = scrape_player_stats(away_code, row["year"], session_cache)
        time.sleep(random.uniform(SCRAPING_CONFIG["min_delay"], SCRAPING_CONFIG["max_delay"]))

        # Stats de Equipo
        stats_h = calcular_stats_equipo(bat1, pit1)
        stats_a = calcular_stats_equipo(bat2, pit2)

        if stats_h and stats_a:
            features["home_team_OPS"] = stats_h.get("team_OPS_mean", 0)
            features["away_team_OPS"] = stats_a.get("team_OPS_mean", 0)
            features["diff_team_BA"] = stats_h.get("team_BA_mean", 0) - stats_a.get("team_BA_mean", 0)
            features["diff_team_OPS"] = stats_h.get("team_OPS_mean", 0) - stats_a.get("team_OPS_mean", 0)
            features["diff_team_ERA"] = stats_a.get("team_ERA_mean", 0) - stats_h.get("team_ERA_mean", 0)

        # Abridores
        sp1 = encontrar_lanzador(pit1, row["home_pitcher"])
        sp2 = encontrar_lanzador(pit2, row["away_pitcher"])

        if sp1 and sp2:
            features["home_pitcher_name_real"] = sp1.get("nombre_real", row["home_pitcher"])
            features["away_pitcher_name_real"] = sp2.get("nombre_real", row["away_pitcher"])
            features["home_starter_SO9"] = sp1.get("SO9", 0)
            features["away_starter_SO9"] = sp2.get("SO9", 0)
            features["home_starter_WHIP"] = sp1.get("WHIP", 0)
            features["away_starter_WHIP"] = sp2.get("WHIP", 0)
            features["home_starter_ERA"] = sp1.get("ERA", 0)
            features["away_starter_ERA"] = sp2.get("ERA", 0)
            features["diff_starter_ERA"] = sp2.get("ERA", 0) - sp1.get("ERA", 0)
            features["diff_starter_WHIP"] = sp2.get("WHIP", 0) - sp1.get("WHIP", 0)
            features["diff_starter_SO9"] = sp1.get("SO9", 0) - sp2.get("SO9", 0)

        # Mejores Bateadores
        hb1 = encontrar_mejor_bateador(bat1)
        hb2 = encontrar_mejor_bateador(bat2)

        if hb1 and hb2:
            features["home_top_3_batters_details"] = hb1.get("detalles_visuales", [])
            features["away_top_3_batters_details"] = hb2.get("detalles_visuales", [])
            features["home_best_OPS"] = hb1.get("best_bat_OPS", 0)
            features["away_best_OPS"] = hb2.get("best_bat_OPS", 0)
            features["diff_best_BA"] = hb1.get("best_bat_BA", 0) - hb2.get("best_bat_BA", 0)
            features["diff_best_OPS"] = hb1.get("best_bat_OPS", 0) - hb2.get("best_bat_OPS", 0)
            features["diff_best_HR"] = hb1.get("best_bat_HR", 0) - hb2.get("best_bat_HR", 0)

        # Bullpen
        rel_h = extraer_top_relevistas(pit1)
        rel_a = extraer_top_relevistas(pit2)
        if rel_h and rel_a:
            features["home_bullpen_ERA"] = rel_h.get("bullpen_ERA_mean", 0)
            features["away_bullpen_ERA"] = rel_a.get("bullpen_ERA_mean", 0)
            features["home_bullpen_WHIP"] = rel_h.get("bullpen_WHIP_mean", 0)
            features["away_bullpen_WHIP"] = rel_a.get("bullpen_WHIP_mean", 0)
            features["diff_bullpen_ERA"] = rel_a.get("bullpen_ERA_mean", 0) - rel_h.get("bullpen_ERA_mean", 0)
            features["diff_bullpen_WHIP"] = rel_a.get("bullpen_WHIP_mean", 0) - rel_h.get("bullpen_WHIP_mean", 0)

        # Anclas
        if sp1:
            features["anchor_pitching_level"] = sp1.get("ERA", 0)
        if stats_h:
            features["anchor_offensive_level"] = stats_h.get("team_OPS_mean", 0)

    features["year"] = row["year"]

    return features


# ============================================================================
# MOTOR DE REENTRENAMIENTO INCREMENTAL
# ============================================================================


def ejecutar_reentrenamiento_incremental(bloque_size=None, pausa_entre_bloques=None):
    """Ejecuta el proceso de reentrenamiento con datos nuevos"""

    if bloque_size is None:
        bloque_size = SCRAPING_CONFIG["bloque_size"]
    if pausa_entre_bloques is None:
        pausa_entre_bloques = SCRAPING_CONFIG["pausa_entre_bloques"]

    print("\n" + "=" * 80)
    print(" INICIANDO ACTUALIZACIÓN INCREMENTAL MLB V3.5")
    print("=" * 80)

    # 1. Verificar milestone antes de proceder
    should_train, _total_real, next_milestone = verificar_milestone_reentrenamiento()
    if not should_train:
        print("ℹ️ No se alcanzó ningún milestone de reentrenamiento. Saliendo.")
        return

    print(
        f"🎯 Milestone alcanzado: {next_milestone}/2430 — "
        f"Entrenando con todos los juegos de la temporada ({_total_real} registros)."
    )

    # 2. Carga de todos los juegos de la temporada
    df_nuevos = obtener_todos_los_juegos_temporada()

    if not df_nuevos.empty:
        df_nuevos["score_home"] = pd.to_numeric(df_nuevos["score_home"], errors="coerce").fillna(0)
        df_nuevos["score_away"] = pd.to_numeric(df_nuevos["score_away"], errors="coerce").fillna(0)
        df_nuevos["ganador"] = pd.to_numeric(df_nuevos["ganador"], errors="coerce").fillna(0).astype(int)

    print(f"Conteo total detectado: {len(df_nuevos)}")

    if not df_nuevos.empty:
        print(f"Rango de fechas: {df_nuevos['fecha'].min()} hasta {df_nuevos['fecha'].max()}")

    total_juegos = len(df_nuevos)
    df_nuevos["fecha"] = pd.to_datetime(df_nuevos["fecha"], errors="coerce")
    df_nuevos["year"] = df_nuevos["fecha"].apply(lambda x: x.year if pd.notna(x) else 0)

    # 2. Extracción de Features
    X_dict_list = []
    y_list = []

    # Cargar caché si existe
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f_pkl:
                cache_previo = pickle.load(f_pkl)
                X_dict_list = cache_previo.get("X_list", [])
                y_list = cache_previo.get("y_list", [])
            print(f"📦 Caché recuperado: {len(X_dict_list)} juegos ya procesados.")
        except Exception:
            print("⚠️ Cache corrupto o no encontrado, iniciando desde cero.")

    juegos_saltados = len(X_dict_list)
    df_para_procesar = df_nuevos.iloc[juegos_saltados:]

    session_cache = {}
    juegos_procesados_completos = []

    for i, (_, row) in enumerate(df_para_procesar.iterrows(), juegos_saltados + 1):
        if i % 25 == 0 or i == 1 or i == total_juegos:
            print(f"⏳ Procesando juego {i}/{total_juegos}... (Equipos en caché: {len(session_cache)})")

        try:
            f = extraer_features_hibridas(
                row,
                df_historico=df_nuevos,
                hacer_scraping=True,
                session_cache=session_cache,
            )
            X_dict_list.append(f)
            y_list.append(row["ganador"])
            juegos_procesados_completos.append(row)

            if i % bloque_size == 0 or i == total_juegos:
                print(f"\n💾 Actualizando caché ({i}/{total_juegos})...")
                try:
                    with open(CACHE_PATH, "wb") as f_pkl:
                        pickle.dump(
                            {"X_list": X_dict_list, "y_list": y_list, "indices": []},
                            f_pkl,
                        )
                except Exception as e:
                    print(f"❌ Error al escribir caché: {e}")

                if i % bloque_size == 0 and i < total_juegos:
                    print(f"🛡️ Pausa de seguridad: {pausa_entre_bloques}s...")
                    time.sleep(pausa_entre_bloques)

        except Exception as e:
            print(f"⚠️ Error en juego {row.get('game_id')}: {e}")
            continue

    # Guardado definitivo
    with open(CACHE_PATH, "wb") as f_pkl:
        pickle.dump({"X_list": X_dict_list, "y_list": y_list, "indices": []}, f_pkl)

    print(f"\n✅ Extracción finalizada. Procesados {len(X_dict_list)} juegos.")

    # 3. Preparar datos para entrenamiento
    X_new = pd.DataFrame(X_dict_list).fillna(0)
    y_new = np.array(y_list)

    # Aplicar super features usando módulo centralizado
    print("🛠️ Calculando super features...")
    for i in range(len(X_new)):
        row_dict = X_new.iloc[i].to_dict()
        updated_dict = calcular_super_features(row_dict)
        for key, val in updated_dict.items():
            if key in [
                "super_neutralizacion_whip_ops",
                "super_resistencia_era_ops",
                "super_muro_bullpen",
            ]:
                X_new.at[i, key] = val

    # El reentrenamiento solo puede consumir columnas numéricas.
    columnas_no_numericas = X_new.select_dtypes(exclude=[np.number]).columns.tolist()
    if columnas_no_numericas:
        print(f"ℹ️ Excluyendo columnas no numéricas del entrenamiento: {columnas_no_numericas}")
        X_new = X_new.drop(columns=columnas_no_numericas)

    # 4. Cargar modelo actual y alinear esquema de features
    accuracy_actual_en_nuevos = 0
    model_actual = None

    if os.path.exists(MODELO_PATH):
        try:
            model_actual = XGBClassifier()
            model_actual.load_model(MODELO_PATH)
        except Exception as e:
            print(f"⚠️ Error cargando modelo previo: {e}")

    X_new = alinear_features_entrenamiento(X_new, model_actual=model_actual)

    # 5. Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_new.fillna(0))
    X_final = pd.DataFrame(X_scaled, columns=X_new.columns)

    # 6. División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_final,
        y_new,
        test_size=MODEL_CONFIG["test_size"],
        random_state=MODEL_CONFIG["random_state"],
    )

    # 7. Evaluar modelo actual si existe
    if model_actual is not None:
        try:
            y_pred_old = model_actual.predict(X_test)
            accuracy_actual_en_nuevos = accuracy_score(y_test, y_pred_old)
            print(f"📊 Accuracy del modelo previo: {accuracy_actual_en_nuevos:.2%}")
        except Exception as e:
            print(f"⚠️ Error evaluando modelo previo: {e}")

    # 8. Optimización de hiperparámetros
    print("🔎 Buscando la mejor combinación de hiperparámetros...")

    xgb_base = XGBClassifier(eval_metric="logloss", random_state=MODEL_CONFIG["random_state"])
    xgb_model_param = model_actual.get_booster() if model_actual else None

    grid = GridSearchCV(
        estimator=xgb_base,
        param_grid=MODEL_CONFIG["param_grid"],
        cv=MODEL_CONFIG["cv_folds"],
        scoring="accuracy",
        verbose=1,
    )

    try:
        grid.fit(X_train, y_train, xgb_model=xgb_model_param)
    except Exception as e:
        if xgb_model_param is not None and "feature_names mismatch" in str(e):
            print(
                "⚠️ Warm-start incremental falló por incompatibilidad de "
                "features con el modelo previo. Reintentando entrenamiento "
                "sin xgb_model para mantener el pipeline estable..."
            )
            grid.fit(X_train, y_train)
        else:
            raise

    model_nuevo = grid.best_estimator_
    print(f"🏆 Mejores parámetros encontrados: {grid.best_params_}")

    # 9. Validación final
    y_pred_new = model_nuevo.predict(X_test)
    accuracy_nuevo = accuracy_score(y_test, y_pred_new)
    print(f"📈 Accuracy nueva versión (Optimizado): {accuracy_nuevo:.2%}")

    # 10. Guardar si hay mejora — en cualquier caso registrar el milestone
    if accuracy_nuevo >= accuracy_actual_en_nuevos:
        print("✅ MEJORA DETECTADA. Actualizando modelo oficial.")

        # Backup del modelo anterior
        if os.path.exists(MODELO_PATH):
            shutil.copy(MODELO_PATH, MODELO_BACKUP)

        model_nuevo.get_booster().save_model(MODELO_PATH)

        registrar_milestone_cumplido(next_milestone)

        if os.path.exists(CACHE_PATH):
            shutil.copy(CACHE_PATH, CACHE_PATH + ".bak")
            print("✅ Copia de seguridad del caché creada (.bak)")
    else:
        print("⚠️ No hubo mejora con los nuevos parámetros. Manteniendo versión previa.")
        registrar_milestone_cumplido(next_milestone)


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    ejecutar_reentrenamiento_incremental()
