"""
Script de Entrenamiento ML Híbrido OPTIMIZADO para Predicción MLB - VERSIÓN V3 UNIFICADA
LÓGICA HÍBRIDA INTELIGENTE:
- Partidos antiguos (2022-2024): Solo features temporales (CSV)
- Partidos recientes (2026): Features temporales + scraping
Sistema de bloques para evitar rate limiting
Cache incremental para entrenar por etapas
NUEVAS MEJORAS: Identificación de Relevistas y Selección SelectKBest
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

import cloudscraper
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE PATHS Y CONSTANTES (RESPETADAS)
# ============================================================================
MODELO_PATH = './models/modelo_mlb_v3.5.json'
MODELO_BACKUP = './models/modelo_mlb_v3.5_backup.json'
DB_PATH = './data/mlb_reentrenamiento.db'
CACHE_PATH = './cache/features_hibridas_v3.5_cache.pkl'
# ============================================================================
# Mapeo de nombres a códigos de Baseball-Reference
# ============================================================================
TEAM_TO_CODE = {
    'Arizona D\'Backs': 'ARI', 'Atlanta Braves': 'ATL', 'Baltimore Orioles': 'BAL',
    'Boston Red Sox': 'BOS', 'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CHW',
    'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE', 'Colorado Rockies': 'COL',
    'Detroit Tigers': 'DET', 'Houston Astros': 'HOU', 'Kansas City Royals': 'KCR',
    'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD', 'Miami Marlins': 'MIA',
    'Milwaukee Brewers': 'MIL', 'Minnesota Twins': 'MIN', 'New York Mets': 'NYM',
    'New York Yankees': 'NYY', 'Oakland Athletics': 'OAK', 'Philadelphia Phillies': 'PHI',
    'Pittsburgh Pirates': 'PIT', 'San Diego Padres': 'SDP', 'San Francisco Giants': 'SFG',
    'Seattle Mariners': 'SEA', 'St. Louis Cardinals': 'STL', 'Tampa Bay Rays': 'TBR',
    'Texas Rangers': 'TEX', 'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSN',
    'Arizona Diamondbacks': 'ARI', 'Atlanta Braves': 'ATL',
    'Baltimore Orioles': 'BAL', 'Boston Red Sox': 'BOS',
    'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CHW',
    'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE',
    'Colorado Rockies': 'COL', 'Detroit Tigers': 'DET',
    'Houston Astros': 'HOU', 'Kansas City Royals': 'KCR',
    'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD',
    'Miami Marlins': 'MIA', 'Milwaukee Brewers': 'MIL',
    'Minnesota Twins': 'MIN', 'New York Mets': 'NYM',
    'New York Yankees': 'NYY', 'Oakland Athletics': 'OAK',
    'Philadelphia Phillies': 'PHI', 'Pittsburgh Pirates': 'PIT',
    'San Diego Padres': 'SDP', 'Seattle Mariners': 'SEA',
    'San Francisco Giants': 'SFG', 'St. Louis Cardinals': 'STL',
    'Tampa Bay Rays': 'TBR', 'Texas Rangers': 'TEX',
    'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSN',
    # Nombres cortos
    'Diamondbacks': 'ARI', 'Braves': 'ATL', 'Orioles': 'BAL',
    'Red Sox': 'BOS', 'Cubs': 'CHC', 'White Sox': 'CHW',
    'Reds': 'CIN', 'Guardians': 'CLE', 'Rockies': 'COL',
    'Tigers': 'DET', 'Astros': 'HOU', 'Royals': 'KCR',
    'Angels': 'LAA', 'Dodgers': 'LAD', 'Marlins': 'MIA',
    'Brewers': 'MIL', 'Twins': 'MIN', 'Mets': 'NYM',
    'Yankees': 'NYY', 'Athletics': 'OAK', 'Phillies': 'PHI',
    'Pirates': 'PIT', 'Padres': 'SDP', 'Mariners': 'SEA',
    'Giants': 'SFG', 'Cardinals': 'STL', 'Rays': 'TBR',
    'Rangers': 'TEX', 'Blue Jays': 'TOR', 'Nationals': 'WSN',
    # Códigos
    'ARI': 'ARI', 'ATL': 'ATL', 'BAL': 'BAL', 'BOS': 'BOS',
    'CHC': 'CHC', 'CHW': 'CHW', 'CIN': 'CIN', 'CLE': 'CLE',
    'COL': 'COL', 'DET': 'DET', 'HOU': 'HOU', 'KCR': 'KCR',
    'LAA': 'LAA', 'LAD': 'LAD', 'MIA': 'MIA', 'MIL': 'MIL',
    'MIN': 'MIN', 'NYM': 'NYM', 'NYY': 'NYY', 'OAK': 'OAK',
    'PHI': 'PHI', 'PIT': 'PIT', 'SDP': 'SDP', 'SEA': 'SEA',
    'SFG': 'SFG', 'STL': 'STL', 'TBR': 'TBR', 'TEX': 'TEX',
    'TOR': 'TOR', 'WSN': 'WSN'
}

# ============================================================================
# FUNCIONES DE SCRAPING CON REINTENTOS (ORIGINALES)
# ============================================================================

def obtener_html(url, max_retries=3):
    """Obtiene HTML con reintentos y backoff exponencial"""
    scraper = cloudscraper.create_scraper()

    for intento in range(max_retries):
        try:
            response = scraper.get(url, timeout=15)

            if response.status_code == 200:
                response.encoding = 'utf-8'
                return response.text
            elif response.status_code == 429:
                wait_time = (2 ** intento) * 5
                print(f"       Rate limit (429) detectado, esperando {wait_time}s...")
                time.sleep(wait_time)
            elif response.status_code == 403:
                wait_time = 10
                print(f"       Error 403 (Forbidden), esperando {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"       Error {response.status_code} al obtener URL {url}")
                if intento < max_retries - 1:
                    time.sleep(2 ** intento)

        except Exception as e:
            if intento == max_retries - 1:
                print(f"       Error final al obtener URL {url}: {str(e)}")
            time.sleep(2 ** intento)

    return None


def limpiar_dataframe(df):
    """Limpia dataframes de Baseball-Reference eliminando basura"""
    if df is None or len(df) == 0:
        return df

    if 'Rk' in df.columns:
        df = df.drop('Rk', axis=1)

    name_col = df.columns[0]
    df = df.dropna(subset=[name_col])
    df = df[~df[name_col].astype(str).str.contains(r'Team Totals|Rank in|^\s*$', case=False, na=False, regex=True)]

    return df.reset_index(drop=True)


def scrape_player_stats(team_code, year, session_cache=None):
    """Scrapea bateo y pitcheo de un equipo con caché de sesión"""
    if session_cache is not None:
        cache_key = f"{team_code}_{year}"
        if cache_key in session_cache:
            return session_cache[cache_key]

    url = f"https://www.baseball-reference.com/teams/{team_code}/{year}.shtml"
    html = obtener_html(url)

    if not html:
        return None, None

    try:
        soup = BeautifulSoup(html, 'html.parser')
        batting_table = soup.find('table', {'id': 'players_standard_batting'})
        pitching_table = soup.find('table', {'id': 'players_standard_pitching'})

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

# ============================================================================
# NUEVA FUNCIÓN: IDENTIFICAR TOP 3 RELEVISTAS (AGREGADA V3)
# ============================================================================

def extraer_top_relevistas(pitching_df):
    """Identifica al Closer y los mejores Setup men basándose en Saves y Juegos Finalizados"""
    if pitching_df is None or len(pitching_df) == 0:
        return None

    # Asegurar conversión numérica para columnas de relevo
    cols_relevo = ['SV', 'GF', 'ERA', 'WHIP', 'IP', 'SO', 'G', 'GS']
    for col in cols_relevo:
        if col in pitching_df.columns:
            pitching_df[col] = pd.to_numeric(pitching_df[col], errors='coerce').fillna(0)

    # Filtrar relevistas: GS < 50% de sus juegos
    bullpen = pitching_df[pitching_df['GS'] < (pitching_df['G'] * 0.5)].copy()
    if len(bullpen) == 0: return None

    # Ordenar por jerarquía: Saves (Closer), Juegos Finalizados (Setup), IP
    bullpen = bullpen.sort_values(by=['SV', 'GF', 'IP'], ascending=False)
    top_3 = bullpen.head(3)

    return {
        'bullpen_ERA_mean': top_3['ERA'].mean(),
        'bullpen_WHIP_mean': top_3['WHIP'].mean()
        # 'bullpen_K9': (top_3['SO'].sum() / top_3['IP'].sum() * 9) if top_3['IP'].sum() > 0 else 0
    }
# ============================================================================
# FUNCIONES DE EXTRACCIÓN Y CÁLCULO (ORIGINALES)
# ============================================================================
def normalizar_texto(texto):
    if not texto: return ""
    texto = str(texto).lower()
    texto = "".join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    texto = re.sub(r'[^a-z0-9]', '', texto)
    return texto


def safe_float(val):
    """Convierte a float de forma segura manejando errores y NaNs"""
    try:
        if pd.isna(val): return 0.0
        return float(val)
    except:
        return 0.0

def encontrar_lanzador(pitching_df, nombre_lanzador):
    """Busca un lanzador específico usando normalización agresiva (estilo API V3)"""
    if pitching_df is None or len(pitching_df) == 0:
        return None

    # --- CAMBIO: Usamos tu normalización de la API para quitar asteriscos ---
    nombre_busqueda = normalizar_texto(nombre_lanzador)
    name_col = pitching_df.columns[0] # Detecta 'Name', 'Player' o 'Name*' automáticamente

    # --- CAMBIO: Normalizamos ambos lados en la máscara ---
    mask = pitching_df[name_col].apply(
        lambda x: nombre_busqueda in normalizar_texto(x) or normalizar_texto(x) in nombre_busqueda
    )

    if mask.sum() == 0:
        return None

    lanzador = pitching_df[mask].iloc[0]

    # Mantenemos exactamente tus mismas llaves de retorno
    return {
        'ERA': safe_float(lanzador.get('ERA', 0)),
        'WHIP': safe_float(lanzador.get('WHIP', 0)),
        'H9': safe_float(lanzador.get('H9', 0)),
        'SO9': safe_float(lanzador.get('SO9', 0)),
        'W': safe_float(lanzador.get('W', 0)),
        'L': safe_float(lanzador.get('L', 0)),
        'IP': safe_float(lanzador.get('IP', 0)),
        'G': safe_float(lanzador.get('G', 0)),
        'GS': safe_float(lanzador.get('GS', 0)),
        'nombre_real': str(lanzador[name_col]) # Agregamos esto para el print sin romper lo anterior
    }

def encontrar_mejor_bateador(batting_df):
    """Encuentra estadísticas manteniendo tus llaves originales pero detectando la columna de nombre"""
    if batting_df is None or len(batting_df) == 0:
        return None

    if 'OBP' not in batting_df.columns or 'AB' not in batting_df.columns:
        return None

    df = batting_df.copy()
    df['OBP'] = pd.to_numeric(df['OBP'], errors='coerce')
    df['AB'] = pd.to_numeric(df['AB'], errors='coerce')
    df = df.dropna(subset=['OBP', 'AB'])

    if len(df) == 0:
        return None

    mediana_ab = df['AB'].median()
    df_filtrado = df[df['AB'] >= mediana_ab].copy()

    if len(df_filtrado) == 0:
        df_filtrado = df

    top_3 = df_filtrado.sort_values('OBP', ascending=False).head(3)

    name_col = top_3.columns[0]

    detalles = []
    for _, row in top_3.iterrows():
        detalles.append({
            'n': str(row[name_col]),
            'ba': safe_float(row.get('BA', 0)),
            'obp': safe_float(row.get('OBP', 0)),
            'slg': safe_float(row.get('SLG', 0)),
            'ops': safe_float(row.get('OPS', 0)),
            'hr': safe_float(row.get('HR', 0)),
            'rbi': safe_float(row.get('RBI', 0))
        })

    return {
        'best_bat_BA': pd.to_numeric(top_3['BA'], errors='coerce').mean(),
        'best_bat_OBP': top_3['OBP'].mean(),
        'best_bat_OPS': pd.to_numeric(top_3['OPS'], errors='coerce').mean() if 'OPS' in top_3.columns else 0.750,
        'best_bat_HR': pd.to_numeric(top_3['HR'], errors='coerce').mean(),
        'best_bat_RBI': pd.to_numeric(top_3['RBI'], errors='coerce').mean(),
        'detalles_visuales': detalles
    }

# def encontrar_lanzador(pitching_df, nombre_lanzador):
#     """Busca un lanzador específico en el dataframe de pitcheo y extrae sus stats"""
#     if pitching_df is None or len(pitching_df) == 0:
#         return None

#     # Normalizar nombre para búsqueda
#     nombre_busqueda = str(nombre_lanzador).lower().strip()
#     # Baseball-Reference a veces tiene caracteres especiales o * / # junto al nombre
#     name_col = pitching_df.columns[0]

#     mask = pitching_df[name_col].astype(str).str.lower().apply(
#         lambda x: nombre_busqueda in x or x in nombre_busqueda
#     )

#     if mask.sum() == 0:
#         return None

#     lanzador = pitching_df[mask].iloc[0]

#     return {
#         'ERA': safe_float(lanzador.get('ERA', 0)),
#         'WHIP': safe_float(lanzador.get('WHIP', 0)),
#         'H9': safe_float(lanzador.get('H9', 0)),
#         'SO9': safe_float(lanzador.get('SO9', 0)),
#         'W': safe_float(lanzador.get('W', 0)),
#         'L': safe_float(lanzador.get('L', 0)),
#         'IP': safe_float(lanzador.get('IP', 0)),
#         'G': safe_float(lanzador.get('G', 0)),
#         'GS': safe_float(lanzador.get('GS', 0))
#     }

# def encontrar_mejor_bateador(batting_df):
#     """Encuentra las estadísticas de los mejores bateadores del equipo"""
#     if batting_df is None or len(batting_df) == 0:
#         return None

#     if 'OBP' not in batting_df.columns or 'AB' not in batting_df.columns:
#         return None

#     # Asegurar conversión numérica para filtrar
#     df = batting_df.copy()
#     df['OBP'] = pd.to_numeric(df['OBP'], errors='coerce')
#     df['AB'] = pd.to_numeric(df['AB'], errors='coerce')
#     df = df.dropna(subset=['OBP', 'AB'])

#     if len(df) == 0:
#         return None

#     # Filtrar por mediana de turnos al bate para evitar ruidos de novatos o lesionados
#     mediana_ab = df['AB'].median()
#     df_filtrado = df[df['AB'] >= mediana_ab].copy()

#     if len(df_filtrado) == 0:
#         df_filtrado = df

#     # Ordenar por OBP y tomar los 3 mejores
#     top_3 = df_filtrado.sort_values('OBP', ascending=False).head(3)

#     return {
#         'best_bat_BA': pd.to_numeric(top_3['BA'], errors='coerce').mean(),
#         'best_bat_OBP': top_3['OBP'].mean(),
#         'best_bat_OPS': pd.to_numeric(top_3['OPS'], errors='coerce').mean() if 'OPS' in top_3.columns else 0.750,
#         'best_bat_HR': pd.to_numeric(top_3['HR'], errors='coerce').mean(),
#         'best_bat_RBI': pd.to_numeric(top_3['RBI'], errors='coerce').mean()
#     }

def calcular_stats_equipo(batting_df, pitching_df):
    """Calcula promedios generales del equipo (Bateo y Pitcheo)"""
    stats = {}

    if batting_df is not None and len(batting_df) > 0:
        for col in ['BA', 'OBP', 'SLG', 'OPS', 'HR', 'RBI']:
            if col in batting_df.columns:
                val = pd.to_numeric(batting_df[col], errors='coerce').mean()
                stats[f'team_{col}_mean'] = val if not pd.isna(val) else 0

    if pitching_df is not None and len(pitching_df) > 0:
        for col in ['ERA', 'WHIP', 'SO9', 'H9', 'BB9']:
            if col in pitching_df.columns:
                val = pd.to_numeric(pitching_df[col], errors='coerce').mean()
                stats[f'team_{col}_mean'] = val if not pd.isna(val) else 0

    return stats

# ============================================================================
# FUNCIONES DE TENDENCIAS TEMPORALES (ORIGINALES)
# ============================================================================

def calcular_tendencias_equipo(df, team, fecha_limite, ventana=10):
    """Calcula rendimiento reciente de un equipo antes de la fecha del partido"""
    if isinstance(fecha_limite, str):
        fecha_limite = pd.to_datetime(fecha_limite)

    mask = (
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (pd.to_datetime(df['fecha']) < fecha_limite)
    )
    partidos_previos = df[mask].sort_values('fecha', ascending=False).head(ventana)

    if len(partidos_previos) == 0:
        return {
            'victorias_recientes': 0.5, 'carreras_anotadas_avg': 4.5,
            'carreras_recibidas_avg': 4.5, 'racha_actual': 0, 'diferencial_carreras': 0
        }

    victorias = 0
    carreras_f = 0
    carreras_c = 0

    for _, p in partidos_previos.iterrows():
        es_home = (p['home_team'] == team)
        # Usamos .get() o aseguramos que no sea None
        ganador_val = p.get('ganador', 0)
        if ganador_val is None: ganador_val = 0

        ganado = (ganador_val == 1) if es_home else (ganador_val == 0)
        if ganado: victorias += 1

        carreras_f += float(p.get('score_home', 0) if es_home else p.get('score_away', 0) or 0)
        carreras_c += float(p.get('score_away', 0) if es_home else p.get('score_home', 0) or 0)

    # Cálculo de racha (W/L)
    racha = 0
    for _, p in partidos_previos.iterrows():
        es_home = (p['home_team'] == team)
        ganado = (p['ganador'] == 1) if es_home else (p['ganador'] == 0)
        if racha == 0:
            racha = 1 if ganado else -1
        elif (racha > 0 and ganado) or (racha < 0 and not ganado):
            racha += 1 if ganado else -1
        else:
            break

    n = len(partidos_previos)
    return {
        'victorias_recientes': victorias / n,
        'carreras_anotadas_avg': carreras_f / n,
        'carreras_recibidas_avg': carreras_c / n,
        'racha_actual': racha,
        'diferencial_carreras': (carreras_f - carreras_c) / n
    }

def calcular_tendencias_lanzador(df, equipo, lanzador, fecha_limite, ventana=5):
    """Calcula rendimiento reciente del lanzador específico"""
    if isinstance(fecha_limite, str):
        fecha_limite = pd.to_datetime(fecha_limite)

    mask = (
        (((df['home_team'] == equipo) & (df['home_pitcher'] == lanzador)) |
         ((df['away_team'] == equipo) & (df['away_pitcher'] == lanzador))) &
        (pd.to_datetime(df['fecha']) < fecha_limite)
    )
    partidos = df[mask].sort_values('fecha', ascending=False).head(ventana)

    if len(partidos) == 0:
        return {'victorias_lanzador': 0.5, 'carreras_permitidas_avg': 4.5}

    victorias = 0
    c_permitidas = 0
    for _, p in partidos.iterrows():
        es_home = (p['home_team'] == equipo)
        ganado = (p['ganador'] == 1) if es_home else (p['ganador'] == 0)
        if ganado: victorias += 1
        c_permitidas += p['score_away'] if es_home else p['score_home']

    return {
        'victorias_lanzador': victorias / len(partidos),
        'carreras_permitidas_avg': c_permitidas / len(partidos)
    }

def calcular_historial_enfrentamientos(df, t1, t2, fecha_limite, ventana=10):
    """Calcula historial directo entre dos equipos (H2H)"""
    if isinstance(fecha_limite, str):
        fecha_limite = pd.to_datetime(fecha_limite)

    mask = (
        (((df['home_team'] == t1) & (df['away_team'] == t2)) |
         ((df['home_team'] == t2) & (df['away_team'] == t1))) &
        (pd.to_datetime(df['fecha']) < fecha_limite)
    )
    enfrentamientos = df[mask].sort_values('fecha', ascending=False).head(ventana)

    if len(enfrentamientos) == 0:
        return {'h2h_win_rate_t1': 0.5, 'h2h_runs_avg_t1': 4.5, 'h2h_runs_avg_t2': 4.5}

    wins_t1 = 0
    runs_t1 = 0
    runs_t2 = 0
    for _, p in enfrentamientos.iterrows():
        t1_es_home = (p['home_team'] == t1)
        t1_gano = (p['ganador'] == 1) if t1_es_home else (p['ganador'] == 0)
        if t1_gano: wins_t1 += 1

        runs_t1 += p['score_home'] if t1_es_home else p['score_away']
        runs_t2 += p['score_away'] if t1_es_home else p['score_home']

    n = len(enfrentamientos)
    return {
        'h2h_win_rate_t1': wins_t1 / n,
        'h2h_runs_avg_t1': runs_t1 / n,
        'h2h_runs_avg_t2': runs_t2 / n
    }

# ============================================================================
# NUEVAS FUNCIONES DE CONTROL DE BASE DE DATOS
# ============================================================================

def registrar_juegos_entrenados(df_procesado):
    """Guarda los IDs de los juegos procesados de forma masiva y segura"""
    if df_procesado.empty:
        return

    # Extraemos solo los IDs únicos para evitar redundancia
    game_ids = df_procesado[['game_id']].drop_duplicates()

    with sqlite3.connect(DB_PATH) as conn:
        # Creamos una tabla temporal para cargar los datos rápido
        game_ids.to_sql('temp_entrenados', conn, if_exists='replace', index=False)

        # Insertamos en la tabla real solo los que no existan (o reemplazamos)
        conn.execute("""
            INSERT OR REPLACE INTO control_entrenamiento (game_id)
            SELECT game_id FROM temp_entrenados
        """)

        # Limpiamos la tabla temporal
        conn.execute("DROP TABLE temp_entrenados")
        conn.commit()

    print(f"✅ Se registraron {len(game_ids)} juegos en la base de datos de control.")


def obtener_juegos_no_entrenados():
    with sqlite3.connect(DB_PATH) as conn:
        # 1. Aseguramos que exista la tabla de control
        conn.execute("CREATE TABLE IF NOT EXISTS control_entrenamiento (game_id TEXT PRIMARY KEY)")

        # 2. Leemos los resultados reales
        df_real = pd.read_sql("SELECT * FROM historico_real", conn)

        if df_real.empty:
            return df_real

        # 3. Creamos el game_id dinámico: FECHA_HOME_AWAY (Ej: 2026-04-01_NYY_LAD)
        # Limpiamos espacios por si acaso
        df_real['home_team'] = df_real['home_team'].str.strip()
        df_real['away_team'] = df_real['away_team'].str.strip()

        df_real['game_id'] = (
            df_real['fecha'].astype(str).str.cat(
                df_real['home_team'].astype(str), sep="_"
            ).str.cat(
                df_real['away_team'].astype(str), sep="_"
            )
        )

        # 4. Consultamos qué IDs ya han sido entrenados
        ids_entrenados = pd.read_sql("SELECT game_id FROM control_entrenamiento", conn)['game_id'].tolist()

        # 5. Filtramos los que NO han sido entrenados
        df_nuevos = df_real[~df_real['game_id'].isin(ids_entrenados)].copy()

        return df_nuevos

# ============================================================================
# EXTRACCIÓN DE FEATURES HÍBRIDA (UNIFICADA)
# ============================================================================
def extraer_features_hibridas(row, df_historico=None, hacer_scraping=False, session_cache=None):
    features = {}

    # =========================================================================
    # 1. TENDENCIAS TEMPORALES (Tu función robusta integrada)
    # =========================================================================
    if df_historico is not None:
        fecha_dt = pd.to_datetime(row['fecha'])

        # Calculamos tendencias para Home y Away usando tu lógica original
        trend_h = calcular_tendencias_equipo(df_historico, row['home_team'], fecha_dt, ventana=10)
        trend_a = calcular_tendencias_equipo(df_historico, row['away_team'], fecha_dt, ventana=10)

        # Insertamos tus métricas en el diccionario de features
        features['home_win_rate_10'] = trend_h.get('victorias_recientes', 0.5)
        features['home_racha'] = trend_h.get('racha_actual', 0)
        features['home_runs_avg'] = trend_h.get('carreras_anotadas_avg', 4.5)
        features['home_runs_diff'] = trend_h.get('diferencial_carreras', 0)

        features['away_win_rate_10'] = trend_a.get('victorias_recientes', 0.5)
        features['away_racha'] = trend_a.get('racha_actual', 0)
        features['away_runs_avg'] = trend_a.get('carreras_anotadas_avg', 4.5)
        features['away_runs_diff'] = trend_a.get('diferencial_carreras', 0)

    # =========================================================================
    # 2. SCRAPING Y STATS DE JUGADORES (Tu lógica original con METADATOS extra)
    # =========================================================================
    if hacer_scraping:
        # Convertir nombres a códigos antes de scrapear
        home_code = TEAM_TO_CODE.get(row['home_team'].strip(), row['home_team'])
        away_code = TEAM_TO_CODE.get(row['away_team'].strip(), row['away_team'])

        bat1, pit1 = scrape_player_stats(home_code, row['year'], session_cache)
        time.sleep(random.uniform(2, 4))  # Pausa para evitar rate limiting
        bat2, pit2 = scrape_player_stats(away_code, row['year'], session_cache)
        time.sleep(random.uniform(2, 4))  # Pausa para evitar rate limiting

        # Stats de Equipo (Individuales y Diferenciales)
        stats_h = calcular_stats_equipo(bat1, pit1)
        stats_a = calcular_stats_equipo(bat2, pit2)

        if stats_h and stats_a:
            features['home_team_OPS'] = stats_h.get('team_OPS_mean', 0)
            features['away_team_OPS'] = stats_a.get('team_OPS_mean', 0)
            features['diff_team_BA'] = stats_h.get('team_BA_mean', 0) - stats_a.get('team_BA_mean', 0)
            features['diff_team_OPS'] = stats_h.get('team_OPS_mean', 0) - stats_a.get('team_OPS_mean', 0)
            features['diff_team_ERA'] = stats_a.get('team_ERA_mean', 0) - stats_h.get('team_ERA_mean', 0)

        # Abridores (Individuales y Diferenciales)
        sp1 = encontrar_lanzador(pit1, row['home_pitcher'])
        sp2 = encontrar_lanzador(pit2, row['away_pitcher'])

        if sp1 and sp2:
            # --- NUEVO: Metadatos visuales (Nombres Reales) ---
            features['home_pitcher_name_real'] = sp1.get('nombre_real', row['home_pitcher'])
            features['away_pitcher_name_real'] = sp2.get('nombre_real', row['away_pitcher'])
            features['home_starter_SO9'] = sp1.get('SO9', 0)
            features['away_starter_SO9'] = sp2.get('SO9', 0)
            # --------------------------------------------------
            features['home_starter_WHIP'] = sp1.get('WHIP', 0)
            features['away_starter_WHIP'] = sp2.get('WHIP', 0)
            features['home_starter_ERA'] = sp1.get('ERA', 0)
            features['away_starter_ERA'] = sp2.get('ERA', 0)
            features['diff_starter_ERA'] = sp2.get('ERA', 0) - sp1.get('ERA', 0)
            features['diff_starter_WHIP'] = sp2.get('WHIP', 0) - sp1.get('WHIP', 0)
            features['diff_starter_SO9'] = sp1.get('SO9', 0) - sp2.get('SO9', 0)

        # Mejores Bateadores (Individuales y Diferenciales)
        hb1 = encontrar_mejor_bateador(bat1)
        hb2 = encontrar_mejor_bateador(bat2)

        if hb1 and hb2:
            # --- SECCIÓN CORREGIDA: Metadatos visuales sin error de columna 'Name' ---
            # Usamos '.get' para evitar errores si la llave no existe por algún fallo de red
            features['home_top_3_batters_details'] = hb1.get('detalles_visuales', [])
            features['away_top_3_batters_details'] = hb2.get('detalles_visuales', [])

            # MANTENEMOS TUS VARIABLES ORIGINALES PARA EL MODELO (NO TOCAR)
            features['home_best_OPS'] = hb1.get('best_bat_OPS', 0)
            features['away_best_OPS'] = hb2.get('best_bat_OPS', 0)
            features['diff_best_BA'] = hb1.get('best_bat_BA', 0) - hb2.get('best_bat_BA', 0)
            features['diff_best_OPS'] = hb1.get('best_bat_OPS', 0) - hb2.get('best_bat_OPS', 0)
            features['diff_best_HR'] = hb1.get('best_bat_HR', 0) - hb2.get('best_bat_HR', 0)

        # Bullpen (Individuales y Diferenciales)
        rel_h = extraer_top_relevistas(pit1)
        rel_a = extraer_top_relevistas(pit2)
        if rel_h and rel_a:
            features['home_bullpen_ERA'] = rel_h.get('bullpen_ERA_mean', 0)
            features['away_bullpen_ERA'] = rel_a.get('bullpen_ERA_mean', 0)
            features['home_bullpen_WHIP'] = rel_h.get('bullpen_WHIP_mean', 0)
            features['away_bullpen_WHIP'] = rel_a.get('bullpen_WHIP_mean', 0)
            features['diff_bullpen_ERA'] = rel_a.get('bullpen_ERA_mean', 0) - rel_h.get('bullpen_ERA_mean', 0)
            features['diff_bullpen_WHIP'] = rel_a.get('bullpen_WHIP_mean', 0) - rel_h.get('bullpen_WHIP_mean', 0)

        # Anclas
        if sp1: features['anchor_pitching_level'] = sp1.get('ERA', 0)
        if stats_h: features['anchor_offensive_level'] = stats_h.get('team_OPS_mean', 0)

    # Añadimos el año para que el modelo tenga contexto de la temporada
    features['year'] = row['year']

    return features
# def extraer_features_hibridas(row, df_historico=None, hacer_scraping=False, session_cache=None):
#     features = {}

#     # =========================================================================
#     # 1. TENDENCIAS TEMPORALES (Tu función robusta integrada)
#     # =========================================================================
#     if df_historico is not None:
#         fecha_dt = pd.to_datetime(row['fecha'])

#         # Calculamos tendencias para Home y Away usando tu lógica original
#         trend_h = calcular_tendencias_equipo(df_historico, row['home_team'], fecha_dt, ventana=10)
#         trend_a = calcular_tendencias_equipo(df_historico, row['away_team'], fecha_dt, ventana=10)

#         # Insertamos tus métricas en el diccionario de features
#         features['home_win_rate_10'] = trend_h.get('victorias_recientes', 0.5)
#         features['home_racha'] = trend_h.get('racha_actual', 0)
#         features['home_runs_avg'] = trend_h.get('carreras_anotadas_avg', 4.5)
#         features['home_runs_diff'] = trend_h.get('diferencial_carreras', 0)

#         features['away_win_rate_10'] = trend_a.get('victorias_recientes', 0.5)
#         features['away_racha'] = trend_a.get('racha_actual', 0)
#         features['away_runs_avg'] = trend_a.get('carreras_anotadas_avg', 4.5)
#         features['away_runs_diff'] = trend_a.get('diferencial_carreras', 0)

#     # =========================================================================
#     # 2. SCRAPING Y STATS DE JUGADORES (Tu lógica original sin cambios)
#     # =========================================================================
#     if hacer_scraping:
#         # Convertir nombres a códigos antes de scrapear
#         home_code = TEAM_TO_CODE.get(row['home_team'].strip(), row['home_team'])
#         away_code = TEAM_TO_CODE.get(row['away_team'].strip(), row['away_team'])

#         bat1, pit1 = scrape_player_stats(home_code, row['year'], session_cache)
#         time.sleep(random.uniform(2, 4))  # Pausa para evitar rate limiting
#         bat2, pit2 = scrape_player_stats(away_code, row['year'], session_cache)
#         time.sleep(random.uniform(2, 4))  # Pausa para evitar rate limiting
#         # Stats de Equipo (Individuales y Diferenciales)
#         stats_h = calcular_stats_equipo(bat1, pit1)
#         stats_a = calcular_stats_equipo(bat2, pit2)

#         if stats_h and stats_a:
#             features['home_team_OPS'] = stats_h.get('team_OPS_mean', 0)
#             features['away_team_OPS'] = stats_a.get('team_OPS_mean', 0)
#             features['diff_team_BA'] = stats_h.get('team_BA_mean', 0) - stats_a.get('team_BA_mean', 0)
#             features['diff_team_OPS'] = stats_h.get('team_OPS_mean', 0) - stats_a.get('team_OPS_mean', 0)
#             features['diff_team_ERA'] = stats_a.get('team_ERA_mean', 0) - stats_h.get('team_ERA_mean', 0)

#         # Abridores (Individuales y Diferenciales)
#         sp1 = encontrar_lanzador(pit1, row['home_pitcher'])
#         sp2 = encontrar_lanzador(pit2, row['away_pitcher'])

#         if sp1 and sp2:
#             features['home_starter_WHIP'] = sp1.get('WHIP', 0)
#             features['away_starter_WHIP'] = sp2.get('WHIP', 0)
#             features['home_starter_ERA'] = sp1.get('ERA', 0)
#             features['away_starter_ERA'] = sp2.get('ERA', 0)
#             features['diff_starter_ERA'] = sp2.get('ERA', 0) - sp1.get('ERA', 0)
#             features['diff_starter_WHIP'] = sp2.get('WHIP', 0) - sp1.get('WHIP', 0)
#             # Descomenta estos si los usas en encontrar_lanzador:
#             # features['diff_starter_W'] = sp1.get('W', 0) - sp2.get('W', 0)
#             # features['diff_starter_L'] = sp2.get('L', 0) - sp1.get('L', 0)
#             features['diff_starter_SO9'] = sp1.get('SO9', 0) - sp2.get('SO9', 0)

#         # Mejores Bateadores (Individuales y Diferenciales)
#         hb1 = encontrar_mejor_bateador(bat1)
#         hb2 = encontrar_mejor_bateador(bat2)

#         if hb1 and hb2:
#             features['home_best_OPS'] = hb1.get('best_bat_OPS', 0)
#             features['away_best_OPS'] = hb2.get('best_bat_OPS', 0)
#             features['diff_best_BA'] = hb1.get('best_bat_BA', 0) - hb2.get('best_bat_BA', 0)
#             # features['diff_best_R'] = hb1.get('best_bat_R', 0) - hb2.get('best_bat_R', 0)
#             features['diff_best_OPS'] = hb1.get('best_bat_OPS', 0) - hb2.get('best_bat_OPS', 0)
#             features['diff_best_HR'] = hb1.get('best_bat_HR', 0) - hb2.get('best_bat_HR', 0)

#         # Bullpen (Individuales y Diferenciales)
#         rel_h = extraer_top_relevistas(pit1)
#         rel_a = extraer_top_relevistas(pit2)
#         if rel_h and rel_a:
#             features['home_bullpen_WHIP'] = rel_h.get('bullpen_WHIP_mean', 0)
#             features['away_bullpen_WHIP'] = rel_a.get('bullpen_WHIP_mean', 0)
#             features['diff_bullpen_ERA'] = rel_a.get('bullpen_ERA_mean', 0) - rel_h.get('bullpen_ERA_mean', 0)
#             features['diff_bullpen_WHIP'] = rel_a.get('bullpen_WHIP_mean', 0) - rel_h.get('bullpen_WHIP_mean', 0)
#             # features['diff_bullpen_K9'] = rel_h.get('bullpen_K9', 0) - rel_a.get('bullpen_K9', 0)

#         # Anclas
#         if sp1: features['anchor_pitching_level'] = sp1.get('ERA', 0)
#         if stats_h: features['anchor_offensive_level'] = stats_h.get('team_OPS_mean', 0)

#     # Añadimos el año para que el modelo tenga contexto de la temporada
#     features['year'] = row['year']

#     return features


# ============================================================================
# EL NUEVO MOTOR (SUSTITUYE AL ANTERIOR)
# ============================================================================

def ejecutar_reentrenamiento_incremental(bloque_size=150, pausa_entre_bloques=45):
    print("\n" + "="*80)
    print(" INICIANDO ACTUALIZACIÓN INCREMENTAL MLB V3.5")
    print("="*80)

    # 1. Carga de datos desde SQL
    df_nuevos = obtener_juegos_no_entrenados()

    # --- 1.5 LIMPIEZA CRÍTICA DE NULOS ---
    if not df_nuevos.empty:
        df_nuevos['score_home'] = pd.to_numeric(df_nuevos['score_home'], errors='coerce').fillna(0)
        df_nuevos['score_away'] = pd.to_numeric(df_nuevos['score_away'], errors='coerce').fillna(0)
        df_nuevos['ganador'] = pd.to_numeric(df_nuevos['ganador'], errors='coerce').fillna(0).astype(int)

    print(f"Conteo total detectado: {len(df_nuevos)}")

    if not df_nuevos.empty:
        print(f"Rango de fechas: {df_nuevos['fecha'].min()} hasta {df_nuevos['fecha'].max()}")

    # if len(df_nuevos) < 150:
    #     print(f"⌛ Solo hay {len(df_nuevos)} juegos nuevos. Esperando a llegar a 150.")
    #     return

    total_juegos = len(df_nuevos)
    df_nuevos['fecha'] = pd.to_datetime(df_nuevos['fecha'], errors='coerce')
    df_nuevos['year'] = df_nuevos['fecha'].apply(lambda x: x.year if pd.notna(x) else 0)

    # 2. Extracción de Features con CARGA DE CACHÉ
    X_dict_list = []
    y_list = []

    # Intentar cargar progreso previo (Aquí es donde se mantiene la base acumulada)
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, 'rb') as f_pkl:
                cache_previo = pickle.load(f_pkl)
                X_dict_list = cache_previo.get('X_list', [])
                y_list = cache_previo.get('y_list', [])
            print(f"📦 Caché recuperado: {len(X_dict_list)} juegos ya procesados.")
        except:
            print("⚠️ Cache corrupto o no encontrado, iniciando desde cero.")

    # Filtrar para procesar solo lo que falta (Lógica acumulativa)
    juegos_saltados = len(X_dict_list)
    df_para_procesar = df_nuevos.iloc[juegos_saltados:]

    session_cache = {}
    juegos_procesados_completos = []

    # El loop inicia desde el siguiente índice después del caché
    for i, (_, row) in enumerate(df_para_procesar.iterrows(), juegos_saltados + 1):

        if i % 25 == 0 or i == 1 or i == total_juegos:
            print(f"⏳ Procesando juego {i}/{total_juegos}... (Equipos en caché: {len(session_cache)})")

        try:
            f = extraer_features_hibridas(row, df_historico=df_nuevos, hacer_scraping=True, session_cache=session_cache)
            X_dict_list.append(f)
            y_list.append(row['ganador'])
            juegos_procesados_completos.append(row)

            # Guardado preventivo cada bloque_size
            if i % bloque_size == 0 or i == total_juegos:
                print(f"\n💾 Actualizando caché ({i}/{total_juegos})...")
                try:
                    with open(CACHE_PATH, 'wb') as f_pkl:
                        pickle.dump({'X_list': X_dict_list, 'y_list': y_list, 'indices': []}, f_pkl)
                except Exception as e:
                    print(f"❌ Error al escribir caché: {e}")

                if i % bloque_size == 0 and i < total_juegos:
                    print(f"🛡️ Pausa de seguridad: {pausa_entre_bloques}s...")
                    time.sleep(pausa_entre_bloques)

        except Exception as e:
            print(f"⚠️ Error en juego {row.get('game_id')}: {e}")
            continue

    # --- DOBLE SEGURIDAD: Guardado definitivo al salir del bucle ---
    with open(CACHE_PATH, 'wb') as f_pkl:
        pickle.dump({'X_list': X_dict_list, 'y_list': y_list, 'indices': []}, f_pkl)

    print(f"\n✅ Extracción finalizada. Procesados {len(X_dict_list)} juegos.")

    X_new = pd.DataFrame(X_dict_list).fillna(0)
    y_new = np.array(y_list)

    # --- INTEGRACIÓN DE NEUTRALIZACIÓN ---
    if 'home_starter_WHIP' in X_new.columns and 'away_team_OPS' in X_new.columns:
        print("🛠️ Calculando variables de neutralización...")
        X_new['super_neutralizacion_whip_ops'] = (X_new['home_starter_WHIP'] * X_new['away_team_OPS']) - \
                                                 (X_new['away_starter_WHIP'] * X_new['home_team_OPS'])
        X_new['super_resistencia_era_ops'] = (X_new['home_starter_ERA'] / (X_new['away_team_OPS'] + 0.01)) - \
                                             (X_new['away_starter_ERA'] / (X_new['home_team_OPS'] + 0.01))
        X_new['super_muro_bullpen'] = (X_new['home_bullpen_WHIP'] * X_new['away_best_OPS']) - \
                                      (X_new['away_bullpen_WHIP'] * X_new['home_best_OPS'])

    # --- ESCALADO DE DATOS ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_new.fillna(0))
    X_final = pd.DataFrame(X_scaled, columns=X_new.columns)

    # --- DIVISIÓN DE DATOS ---
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_new, test_size=0.20, random_state=42)

    accuracy_actual_en_nuevos = 0
    model_actual = None

    if os.path.exists(MODELO_PATH):
        try:
            model_actual = XGBClassifier()
            model_actual.load_model(MODELO_PATH)
            y_pred_old = model_actual.predict(X_test)
            accuracy_actual_en_nuevos = accuracy_score(y_test, y_pred_old)
            print(f"📊 Accuracy del modelo previo: {accuracy_actual_en_nuevos:.2%}")
        except Exception as e:
            print(f"⚠️ Error evaluando modelo previo: {e}")

    # --- OPTIMIZACIÓN DE HIPERPARÁMETROS ---
    print("🔎 Buscando la mejor combinación de hiperparámetros...")
    param_grid = {
        'n_estimators': [200, 300, 400],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.03, 0.05],
        'gamma': [0.1, 0.2]
    }

    xgb_base = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb_model_param = model_actual.get_booster() if model_actual else None

    grid = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        verbose=1
    )

    # Entrenamiento incremental pasando el booster anterior
    grid.fit(X_train, y_train, xgb_model=xgb_model_param)

    model_nuevo = grid.best_estimator_
    print(f"🏆 Mejores parámetros encontrados: {grid.best_params_}")

    # --- VALIDACIÓN FINAL Y GUARDADO ---
    y_pred_new = model_nuevo.predict(X_test)
    accuracy_nuevo = accuracy_score(y_test, y_pred_new)
    print(f"📈 Accuracy nueva versión (Optimizado): {accuracy_nuevo:.2%}")

    if accuracy_nuevo >= accuracy_actual_en_nuevos:
        print("✅ MEJORA DETECTADA. Actualizando modelo oficial.")
        # Se usa get_booster() para evitar errores de serialización
        model_nuevo.get_booster().save_model(MODELO_PATH)

        # Registrar juegos en SQL
        registrar_juegos_entrenados(df_nuevos)

        # Mantenemos el caché y creamos un backup
        if os.path.exists(CACHE_PATH):
            shutil.copy(CACHE_PATH, CACHE_PATH + ".bak")
            print("✅ Copia de seguridad del caché creada (.bak)")
    else:
        print("⚠️ No hubo mejora con los nuevos parámetros. Manteniendo versión previa.")
# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    ejecutar_reentrenamiento_incremental()
# ============================================================================
# EJECUCIÓN (ORIGINAL)
# ============================================================================
