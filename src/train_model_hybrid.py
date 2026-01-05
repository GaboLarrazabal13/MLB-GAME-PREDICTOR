"""
Script de Entrenamiento ML Híbrido OPTIMIZADO para Predicción MLB
LÓGICA HÍBRIDA INTELIGENTE:
- Partidos antiguos (2022-2024): Solo features temporales (CSV)
- Partidos recientes (2025): Features temporales + scraping
Sistema de bloques para evitar rate limiting
Cache incremental para entrenar por etapas
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                            roc_auc_score, f1_score)
import xgboost as xgb
from xgboost import XGBClassifier
import pickle
import cloudscraper
from bs4 import BeautifulSoup
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# FUNCIONES DE SCRAPING CON REINTENTOS
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
            elif response.status_code == 429:  # Rate limit
                wait_time = (2 ** intento) * 5  # 5s, 10s, 20s
                print(f"      ⏳ Rate limit detectado, esperando {wait_time}s...")
                time.sleep(wait_time)
            elif response.status_code == 403:  # Forbidden
                wait_time = 10
                print(f"      ⏳ Error 403, esperando {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"      ⚠️ Error {response.status_code} en {url}")
                if intento < max_retries - 1:
                    time.sleep(2 ** intento)
        except Exception as e:
            if intento < max_retries - 1:
                print(f"      ⚠️ Excepción: {e}, reintentando...")
                time.sleep(2 ** intento)
            else:
                print(f"      ❌ Error final de conexión: {e}")
    
    return None


def limpiar_dataframe(df):
    """Limpia el DataFrame eliminando filas no deseadas"""
    if df is None or len(df) == 0:
        return df
    
    if 'Rk' in df.columns:
        df = df.drop('Rk', axis=1)
    
    name_col = df.columns[0]
    df = df.dropna(subset=[name_col])
    df = df[~df[name_col].astype(str).str.contains(
        r'Team Totals|Rank in|^\s*$', 
        case=False, na=False, regex=True
    )]
    return df.reset_index(drop=True)


def scrape_player_stats(team_code, year=2025, session_cache=None):
    """
    Extrae estadísticas con cache de sesión
    """
    # Verificar cache de sesión
    if session_cache is not None:
        cache_key = f"{team_code}_{year}"
        if cache_key in session_cache:
            return session_cache[cache_key]
    
    url = f"https://www.baseball-reference.com/teams/{team_code}/{year}.shtml"
    
    html = obtener_html(url, max_retries=3)
    if not html:
        return None, None
    
    soup = BeautifulSoup(html, 'html.parser')
    
    batting_table = soup.find('table', {'id': 'players_standard_batting'})
    pitching_table = soup.find('table', {'id': 'players_standard_pitching'})
    
    batting_df = pitching_df = None
    
    if batting_table:
        try:
            batting_df = pd.read_html(str(batting_table))[0]
            batting_df = limpiar_dataframe(batting_df)
        except:
            pass
    
    if pitching_table:
        try:
            pitching_df = pd.read_html(str(pitching_table))[0]
            pitching_df = limpiar_dataframe(pitching_df)
        except:
            pass
    
    # Guardar en cache de sesión
    if session_cache is not None:
        cache_key = f"{team_code}_{year}"
        session_cache[cache_key] = (batting_df, pitching_df)
    
    return batting_df, pitching_df


def safe_float(val):
    """Convierte valores a float de forma segura"""
    try:
        return float(val)
    except:
        return 0.0


def encontrar_lanzador(pitching_df, nombre_lanzador):
    """Busca un lanzador y extrae sus stats: ERA, WHIP, H9"""
    if pitching_df is None or len(pitching_df) == 0:
        return None
    
    nombre_busqueda = nombre_lanzador.lower().strip()
    name_col = pitching_df.columns[0]
    
    mask = pitching_df[name_col].astype(str).str.lower().str.contains(nombre_busqueda, na=False)
    
    if mask.sum() == 0:
        return None
    
    lanzador = pitching_df[mask].iloc[0]
    
    return {
        'ERA': safe_float(lanzador.get('ERA', 0)),
        'WHIP': safe_float(lanzador.get('WHIP', 0)),
        'H9': safe_float(lanzador.get('H9', 0)),
        'W': safe_float(lanzador.get('W', 0)),
        'L': safe_float(lanzador.get('L', 0)),
        'IP': safe_float(lanzador.get('IP', 0))
    }


def encontrar_mejor_bateador(batting_df):
    """Encuentra los 3 mejores bateadores según OBP"""
    if batting_df is None or len(batting_df) == 0:
        return None
    
    if 'OBP' not in batting_df.columns or 'AB' not in batting_df.columns:
        return None
    
    batting_df['OBP'] = pd.to_numeric(batting_df['OBP'], errors='coerce')
    batting_df['AB'] = pd.to_numeric(batting_df['AB'], errors='coerce')
    batting_df = batting_df.dropna(subset=['OBP', 'AB'])
    
    if len(batting_df) == 0:
        return None
    
    mediana_ab = batting_df['AB'].median()
    batting_filtrado = batting_df[batting_df['AB'] > mediana_ab].copy()
    
    if len(batting_filtrado) == 0:
        batting_filtrado = batting_df
    
    batting_filtrado = batting_filtrado.sort_values('OBP', ascending=False)
    top_3 = batting_filtrado.head(3)
    
    stats_promedio = {'BA': 0, 'OBP': 0, 'RBI': 0, 'R': 0}
    
    count = 0
    for idx, bateador in top_3.iterrows():
        stats_promedio['BA'] += safe_float(bateador.get('BA', 0))
        stats_promedio['OBP'] += safe_float(bateador.get('OBP', 0))
        stats_promedio['RBI'] += safe_float(bateador.get('RBI', 0))
        stats_promedio['R'] += safe_float(bateador.get('R', 0))
        count += 1
    
    if count > 0:
        for key in stats_promedio:
            stats_promedio[key] /= count
    
    return stats_promedio


def calcular_stats_equipo(batting_df, pitching_df):
    """Calcula estadísticas agregadas del equipo"""
    stats = {}
    
    if batting_df is not None and len(batting_df) > 0:
        for col in ['BA', 'OBP', 'RBI', 'R']:
            if col in batting_df.columns:
                batting_df[col] = pd.to_numeric(batting_df[col], errors='coerce')
                stats[f'team_{col}_mean'] = batting_df[col].mean()
    
    if pitching_df is not None and len(pitching_df) > 0:
        for col in ['ERA', 'WHIP', 'H9']:
            if col in pitching_df.columns:
                pitching_df[col] = pd.to_numeric(pitching_df[col], errors='coerce')
                stats[f'team_{col}_mean'] = pitching_df[col].mean()
    
    return stats


# ============================================================================
# FUNCIONES DE FEATURES TEMPORALES
# ============================================================================

def calcular_tendencias_equipo(df, team, fecha_limite, ventana=10):
    """Calcula tendencias de un equipo en sus últimos N partidos"""
    if isinstance(fecha_limite, str):
        fecha_limite = pd.to_datetime(fecha_limite)
    
    partidos_equipo = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (pd.to_datetime(df['fecha']) < fecha_limite)
    ].copy()
    
    partidos_equipo = partidos_equipo.sort_values('fecha', ascending=False).head(ventana)
    
    if len(partidos_equipo) == 0:
        return {
            'victorias_recientes': 0.5,
            'carreras_anotadas_avg': 4.5,
            'carreras_recibidas_avg': 4.5,
            'racha_actual': 0,
            'diferencial_carreras': 0
        }
    
    victorias = 0
    carreras_anotadas = []
    carreras_recibidas = []
    
    for idx, partido in partidos_equipo.iterrows():
        es_local = (partido['home_team'] == team)
        
        if es_local:
            gano = (partido['ganador'] == 1)
            carreras_anotadas.append(partido['score_home'])
            carreras_recibidas.append(partido['score_away'])
        else:
            gano = (partido['ganador'] == 0)
            carreras_anotadas.append(partido['score_away'])
            carreras_recibidas.append(partido['score_home'])
        
        if gano:
            victorias += 1
    
    # Calcular racha
    racha_actual = 0
    for idx, partido in partidos_equipo.iterrows():
        es_local = (partido['home_team'] == team)
        gano = (partido['ganador'] == 1) if es_local else (partido['ganador'] == 0)
        
        if racha_actual == 0:
            racha_actual = 1 if gano else -1
        elif (racha_actual > 0 and gano) or (racha_actual < 0 and not gano):
            racha_actual += 1 if gano else -1
        else:
            break
    
    n_partidos = len(partidos_equipo)
    tasa_victorias = victorias / n_partidos if n_partidos > 0 else 0.5
    avg_anotadas = np.mean(carreras_anotadas) if carreras_anotadas else 4.5
    avg_recibidas = np.mean(carreras_recibidas) if carreras_recibidas else 4.5
    diferencial = avg_anotadas - avg_recibidas
    
    return {
        'victorias_recientes': tasa_victorias,
        'carreras_anotadas_avg': avg_anotadas,
        'carreras_recibidas_avg': avg_recibidas,
        'racha_actual': racha_actual,
        'diferencial_carreras': diferencial
    }


def calcular_tendencias_lanzador(df, equipo, lanzador, fecha_limite, ventana=5):
    """Calcula tendencias de un lanzador"""
    if isinstance(fecha_limite, str):
        fecha_limite = pd.to_datetime(fecha_limite)
    
    partidos_lanzador = df[
        (
            ((df['home_team'] == equipo) & (df['home_pitcher'] == lanzador)) |
            ((df['away_team'] == equipo) & (df['away_pitcher'] == lanzador))
        ) &
        (pd.to_datetime(df['fecha']) < fecha_limite)
    ].copy()
    
    partidos_lanzador = partidos_lanzador.sort_values('fecha', ascending=False).head(ventana)
    
    if len(partidos_lanzador) == 0:
        return {
            'victorias_lanzador': 0.5,
            'carreras_permitidas_avg': 4.5
        }
    
    victorias = 0
    carreras_permitidas = []
    
    for idx, partido in partidos_lanzador.iterrows():
        es_local = (partido['home_team'] == equipo)
        
        if es_local:
            gano = (partido['ganador'] == 1)
            carreras_permitidas.append(partido['score_away'])
        else:
            gano = (partido['ganador'] == 0)
            carreras_permitidas.append(partido['score_home'])
        
        if gano:
            victorias += 1
    
    n_aperturas = len(partidos_lanzador)
    tasa_victorias = victorias / n_aperturas if n_aperturas > 0 else 0.5
    avg_permitidas = np.mean(carreras_permitidas) if carreras_permitidas else 4.5
    
    return {
        'victorias_lanzador': tasa_victorias,
        'carreras_permitidas_avg': avg_permitidas
    }


def calcular_historial_enfrentamientos(df, team1, team2, fecha_limite, ventana=10):
    """Calcula historial H2H"""
    if isinstance(fecha_limite, str):
        fecha_limite = pd.to_datetime(fecha_limite)
    
    enfrentamientos = df[
        (
            ((df['home_team'] == team1) & (df['away_team'] == team2)) |
            ((df['home_team'] == team2) & (df['away_team'] == team1))
        ) &
        (pd.to_datetime(df['fecha']) < fecha_limite)
    ].copy()
    
    enfrentamientos = enfrentamientos.sort_values('fecha', ascending=False).head(ventana)
    
    if len(enfrentamientos) == 0:
        return {
            'victorias_team1': 0.5,
            'carreras_avg_team1': 4.5,
            'carreras_avg_team2': 4.5
        }
    
    victorias_team1 = 0
    carreras_team1 = []
    carreras_team2 = []
    
    for idx, partido in enfrentamientos.iterrows():
        team1_es_local = (partido['home_team'] == team1)
        
        if team1_es_local:
            gano_team1 = (partido['ganador'] == 1)
            carreras_team1.append(partido['score_home'])
            carreras_team2.append(partido['score_away'])
        else:
            gano_team1 = (partido['ganador'] == 0)
            carreras_team1.append(partido['score_away'])
            carreras_team2.append(partido['score_home'])
        
        if gano_team1:
            victorias_team1 += 1
    
    n_enfrentamientos = len(enfrentamientos)
    tasa_victorias = victorias_team1 / n_enfrentamientos if n_enfrentamientos > 0 else 0.5
    
    return {
        'victorias_team1': tasa_victorias,
        'carreras_avg_team1': np.mean(carreras_team1) if carreras_team1 else 4.5,
        'carreras_avg_team2': np.mean(carreras_team2) if carreras_team2 else 4.5
    }


# ============================================================================
# EXTRACCIÓN HÍBRIDA DE FEATURES
# ============================================================================

def extraer_features_hibridas(row, df, hacer_scraping=False, session_cache=None):
    """
    Extrae features con lógica híbrida:
    - SIEMPRE: features temporales (rápido)
    - CONDICIONAL: features de scraping (solo si hacer_scraping=True)
    """
    
    features = {}
    
    # ===== PARTE 1: FEATURES TEMPORALES (SIEMPRE) =====
    
    # Tendencias de equipos
    tend_home = calcular_tendencias_equipo(df, row['home_team'], row['fecha'], ventana=10)
    tend_away = calcular_tendencias_equipo(df, row['away_team'], row['fecha'], ventana=10)
    
    features['home_victorias_L10'] = tend_home['victorias_recientes']
    features['home_runs_anotadas_L10'] = tend_home['carreras_anotadas_avg']
    features['home_runs_recibidas_L10'] = tend_home['carreras_recibidas_avg']
    features['home_racha'] = tend_home['racha_actual']
    features['home_run_diff_L10'] = tend_home['diferencial_carreras']
    
    features['away_victorias_L10'] = tend_away['victorias_recientes']
    features['away_runs_anotadas_L10'] = tend_away['carreras_anotadas_avg']
    features['away_runs_recibidas_L10'] = tend_away['carreras_recibidas_avg']
    features['away_racha'] = tend_away['racha_actual']
    features['away_run_diff_L10'] = tend_away['diferencial_carreras']
    
    # Tendencias de lanzadores
    tend_pitcher_home = calcular_tendencias_lanzador(
        df, row['home_team'], row['home_pitcher'], row['fecha'], ventana=5
    )
    tend_pitcher_away = calcular_tendencias_lanzador(
        df, row['away_team'], row['away_pitcher'], row['fecha'], ventana=5
    )
    
    features['home_pitcher_victorias_L5'] = tend_pitcher_home['victorias_lanzador']
    features['home_pitcher_runs_permitidas_L5'] = tend_pitcher_home['carreras_permitidas_avg']
    features['away_pitcher_victorias_L5'] = tend_pitcher_away['victorias_lanzador']
    features['away_pitcher_runs_permitidas_L5'] = tend_pitcher_away['carreras_permitidas_avg']
    
    # H2H
    h2h = calcular_historial_enfrentamientos(
        df, row['home_team'], row['away_team'], row['fecha'], ventana=10
    )
    
    features['h2h_home_win_rate'] = h2h['victorias_team1']
    features['h2h_home_runs_avg'] = h2h['carreras_avg_team1']
    features['h2h_away_runs_avg'] = h2h['carreras_avg_team2']
    
    # ===== PARTE 2: FEATURES DE SCRAPING (CONDICIONAL) =====
    
    if hacer_scraping:
        # Obtener stats de equipos
        batting1, pitching1 = scrape_player_stats(row['home_team'], row['year'], session_cache)
        batting2, pitching2 = scrape_player_stats(row['away_team'], row['year'], session_cache)
        
        if batting1 is not None and batting2 is not None:
            # Stats del equipo
            stats_team1 = calcular_stats_equipo(batting1, pitching1)
            stats_team2 = calcular_stats_equipo(batting2, pitching2)
            
            for key, val in stats_team1.items():
                features[f'home_{key}'] = val
            
            for key, val in stats_team2.items():
                features[f'away_{key}'] = val
            
            # Lanzadores
            pitcher1_stats = encontrar_lanzador(pitching1, row['home_pitcher'])
            pitcher2_stats = encontrar_lanzador(pitching2, row['away_pitcher'])
            
            if pitcher1_stats:
                features['home_pitcher_ERA'] = pitcher1_stats['ERA']
                features['home_pitcher_WHIP'] = pitcher1_stats['WHIP']
                features['home_pitcher_H9'] = pitcher1_stats['H9']
                features['home_pitcher_W'] = pitcher1_stats['W']
                features['home_pitcher_L'] = pitcher1_stats['L']
            else:
                features.update({
                    'home_pitcher_ERA': 0, 'home_pitcher_WHIP': 0,
                    'home_pitcher_H9': 0, 'home_pitcher_W': 0, 'home_pitcher_L': 0
                })
            
            if pitcher2_stats:
                features['away_pitcher_ERA'] = pitcher2_stats['ERA']
                features['away_pitcher_WHIP'] = pitcher2_stats['WHIP']
                features['away_pitcher_H9'] = pitcher2_stats['H9']
                features['away_pitcher_W'] = pitcher2_stats['W']
                features['away_pitcher_L'] = pitcher2_stats['L']
            else:
                features.update({
                    'away_pitcher_ERA': 0, 'away_pitcher_WHIP': 0,
                    'away_pitcher_H9': 0, 'away_pitcher_W': 0, 'away_pitcher_L': 0
                })
            
            # Mejores bateadores
            best_batter1 = encontrar_mejor_bateador(batting1)
            best_batter2 = encontrar_mejor_bateador(batting2)
            
            if best_batter1:
                features['home_best_BA'] = best_batter1['BA']
                features['home_best_OBP'] = best_batter1['OBP']
                features['home_best_RBI'] = best_batter1['RBI']
                features['home_best_R'] = best_batter1['R']
            else:
                features.update({
                    'home_best_BA': 0, 'home_best_OBP': 0,
                    'home_best_RBI': 0, 'home_best_R': 0
                })
            
            if best_batter2:
                features['away_best_BA'] = best_batter2['BA']
                features['away_best_OBP'] = best_batter2['OBP']
                features['away_best_RBI'] = best_batter2['RBI']
                features['away_best_R'] = best_batter2['R']
            else:
                features.update({
                    'away_best_BA': 0, 'away_best_OBP': 0,
                    'away_best_RBI': 0, 'away_best_R': 0
                })
            
            # Features derivadas de scraping
            features['pitcher_ERA_diff'] = features['away_pitcher_ERA'] - features['home_pitcher_ERA']
            features['pitcher_WHIP_diff'] = features['away_pitcher_WHIP'] - features['home_pitcher_WHIP']
            features['pitcher_H9_diff'] = features['away_pitcher_H9'] - features['home_pitcher_H9']
            features['team_BA_diff'] = features['home_team_BA_mean'] - features['away_team_BA_mean']
            features['team_OBP_diff'] = features['home_team_OBP_mean'] - features['away_team_OBP_mean']
    
    # Features derivadas temporales
    features['victorias_diff'] = features['home_victorias_L10'] - features['away_victorias_L10']
    features['run_diff_diff'] = features['home_run_diff_L10'] - features['away_run_diff_L10']
    features['pitcher_win_rate_diff'] = features['home_pitcher_victorias_L5'] - features['away_pitcher_victorias_L5']
    features['pitcher_runs_allowed_diff'] = features['away_pitcher_runs_permitidas_L5'] - features['home_pitcher_runs_permitidas_L5']
    features['racha_diff'] = features['home_racha'] - features['away_racha']
    
    return features


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def entrenar_modelo_hibrido_optimizado(
    csv_path='./data/processed/datos_ml_ready.csv',
    test_size=0.2,
    usar_cache=True,
    cache_path='./cache/features_hibridas_optimizadas_cache.pkl',
    optimizar_hiperparametros=False,
    validacion_temporal=True,
    year_scraping=2025,  # Solo hacer scraping a partidos de este año
    bloque_size=200,     # Procesar en bloques de 200 partidos
    pausa_entre_bloques=60  # Pausa en segundos entre bloques
):
    """
    Entrena modelo híbrido con scraping inteligente por bloques
    """
    
    print("\n" + "="*80)
    print(" ENTRENAMIENTO MODELO ML HÍBRIDO OPTIMIZADO - PREDICTOR MLB")
    print(" LÓGICA HÍBRIDA: Scraping solo para partidos de 2025")
    print("="*80)
    
    # Cargar datos
    print(f"\n📂 Cargando datos desde: {csv_path}")
    df = pd.read_csv(csv_path)
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values('fecha').reset_index(drop=True)
    
    print(f"   Total de partidos: {len(df)}")
    print(f"   Rango de fechas: {df['fecha'].min()} a {df['fecha'].max()}")
    
    # Identificar partidos para scraping
    df['year_partido'] = df['fecha'].dt.year
    partidos_scraping = df[df['year_partido'] == year_scraping].index.tolist()
    partidos_sin_scraping = df[df['year_partido'] != year_scraping].index.tolist()
    
    print(f"\n🎯 Estrategia de scraping:")
    print(f"   Partidos {year_scraping} (CON scraping): {len(partidos_scraping)}")
    print(f"   Partidos anteriores (SIN scraping): {len(partidos_sin_scraping)}")
    
    # Distribución
    print(f"\n📊 Distribución de ganadores:")
    victorias_local = (df['ganador'] == 1).sum()
    victorias_visitante = (df['ganador'] == 0).sum()
    print(f"   Victorias locales (1): {victorias_local} ({victorias_local/len(df)*100:.1f}%)")
    print(f"   Victorias visitantes (0): {victorias_visitante} ({victorias_visitante/len(df)*100:.1f}%)")
    
    # Intentar cargar cache
    usar_cache_exitoso = False
    if usar_cache:
        try:
            print(f"\n💾 Intentando cargar features desde cache...")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                X = cache_data['X']
                y = cache_data['y']
                indices_procesados = cache_data.get('indices_procesados', [])
            print(f"   ✅ Cache cargado exitosamente")
            print(f"   Partidos en cache: {len(X)}")
            print(f"   Índices procesados: {len(indices_procesados)}")
            usar_cache_exitoso = True
        except FileNotFoundError:
            print(f"   ⚠️  No se encontró cache, extrayendo features desde cero...")
            indices_procesados = []
        except Exception as e:
            print(f"   ⚠️  Error al cargar cache: {e}")
            indices_procesados = []
    else:
        indices_procesados = []
    
    if not usar_cache_exitoso or len(indices_procesados) < len(df):
        # Determinar qué partidos quedan por procesar
        todos_indices = set(range(len(df)))
        indices_pendientes = list(todos_indices - set(indices_procesados))
        indices_pendientes.sort()
        
        print(f"\n🔄 Extrayendo features...")
        print(f"   Total partidos: {len(df)}")
        print(f"   Ya procesados: {len(indices_procesados)}")
        print(f"   Pendientes: {len(indices_pendientes)}")
        
        # Inicializar listas
        if usar_cache_exitoso:
            features_list = X.to_dict('records')
            labels = y.tolist()
        else:
            features_list = []
            labels = []
        
        # Cache de sesión para scraping
        session_cache = {}
        
        partidos_sin_historia = 0
        partidos_fallidos = 0
        
        # Separar partidos pendientes en dos grupos
        indices_pendientes_sin_scraping = [idx for idx in indices_pendientes if idx not in partidos_scraping]
        indices_pendientes_con_scraping = [idx for idx in indices_pendientes if idx in partidos_scraping]
        
        print(f"\n   📝 Plan de procesamiento:")
        print(f"      Partidos sin scraping (rápido): {len(indices_pendientes_sin_scraping)}")
        print(f"      Partidos con scraping (bloques): {len(indices_pendientes_con_scraping)}")
        
        # ===== FASE 1: PROCESAR PARTIDOS SIN SCRAPING (RÁPIDO) =====
        if len(indices_pendientes_sin_scraping) > 0:
            print(f"\n   🚀 FASE 1: Procesando partidos sin scraping...")
            
            for i, idx in enumerate(indices_pendientes_sin_scraping):
                if (i + 1) % 100 == 0:
                    print(f"      Progreso: {i+1}/{len(indices_pendientes_sin_scraping)} ({(i+1)/len(indices_pendientes_sin_scraping)*100:.1f}%)")
                
                row = df.iloc[idx]
                
                # Verificar historia mínima
                fecha_actual = row['fecha']
                partidos_previos = df[pd.to_datetime(df['fecha']) < fecha_actual]
                
                if len(partidos_previos) < 20:
                    partidos_sin_historia += 1
                    continue
                
                try:
                    features = extraer_features_hibridas(
                        row, 
                        df=df, 
                        hacer_scraping=False,  # SIN scraping
                        session_cache=session_cache
                    )
                    
                    features_list.append(features)
                    labels.append(row['ganador'])
                    indices_procesados.append(idx)
                    
                except Exception as e:
                    partidos_fallidos += 1
                    continue
            
            print(f"      ✅ Completado: {len(indices_pendientes_sin_scraping)} partidos sin scraping")
            
            # Guardar progreso después de fase 1
            print(f"\n   💾 Guardando progreso (Fase 1 completada)...")
            X_temp = pd.DataFrame(features_list)
            y_temp = np.array(labels)
            
            import os
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'X': X_temp,
                    'y': y_temp,
                    'indices_procesados': indices_procesados
                }, f)
            print(f"      ✅ Progreso guardado: {len(features_list)} partidos")
        
        # ===== FASE 2: PROCESAR PARTIDOS CON SCRAPING (POR BLOQUES) =====
        if len(indices_pendientes_con_scraping) > 0:
            print(f"\n   🔄 FASE 2: Procesando partidos con scraping (por bloques)...")
            print(f"      Tamaño de bloque: {bloque_size} partidos")
            print(f"      Pausa entre bloques: {pausa_entre_bloques}s")
            
            bloques_procesados = 0
            partidos_scraping_procesados = 0
            
            for i, idx in enumerate(indices_pendientes_con_scraping):
                if (i + 1) % 10 == 0:
                    print(f"      Progreso: {i+1}/{len(indices_pendientes_con_scraping)} ({(i+1)/len(indices_pendientes_con_scraping)*100:.1f}%)")
                
                row = df.iloc[idx]
                
                # Verificar historia mínima
                fecha_actual = row['fecha']
                partidos_previos = df[pd.to_datetime(df['fecha']) < fecha_actual]
                
                if len(partidos_previos) < 20:
                    partidos_sin_historia += 1
                    continue
                
                try:
                    features = extraer_features_hibridas(
                        row, 
                        df=df, 
                        hacer_scraping=True,  # CON scraping
                        session_cache=session_cache
                    )
                    
                    features_list.append(features)
                    labels.append(row['ganador'])
                    indices_procesados.append(idx)
                    partidos_scraping_procesados += 1
                    
                except Exception as e:
                    partidos_fallidos += 1
                    if partidos_fallidos % 5 == 0:
                        print(f"         ⚠️  {partidos_fallidos} partidos fallidos hasta ahora")
                    continue
                
                # Pausa entre requests individuales
                time.sleep(1.5)
                
                # Pausa entre bloques (solo para partidos con scraping)
                if partidos_scraping_procesados > 0 and partidos_scraping_procesados % bloque_size == 0:
                    bloques_procesados += 1
                    print(f"\n      ⏸️  BLOQUE {bloques_procesados} COMPLETADO")
                    print(f"         Partidos con scraping procesados: {partidos_scraping_procesados}")
                    print(f"      ⏳ Esperando {pausa_entre_bloques}s antes del siguiente bloque...")
                    print(f"      💾 Guardando progreso incremental...")
                    
                    # Guardar progreso incremental
                    X_temp = pd.DataFrame(features_list)
                    y_temp = np.array(labels)
                    
                    import os
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, 'wb') as f:
                        pickle.dump({
                            'X': X_temp,
                            'y': y_temp,
                            'indices_procesados': indices_procesados
                        }, f)
                    
                    print(f"      ✅ Progreso guardado: {len(features_list)} partidos totales")
                    time.sleep(pausa_entre_bloques)
                    print(f"      ▶️  Continuando con siguiente bloque...\n")
            
            print(f"      ✅ Completado: {partidos_scraping_procesados} partidos con scraping")
        
        print(f"\n   ✅ Features extraídas: {len(features_list)} partidos")
        print(f"   ⚠️  Partidos sin historia: {partidos_sin_historia}")
        print(f"   ⚠️  Partidos fallidos: {partidos_fallidos}")
        
        # Crear DataFrames finales
        X = pd.DataFrame(features_list)
        y = np.array(labels)
        
        # Guardar cache final
        print(f"\n💾 Guardando cache final...")
        import os
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'X': X,
                'y': y,
                'indices_procesados': indices_procesados
            }, f)
        print(f"   ✅ Cache guardado: {cache_path}")
    
    # Manejar valores faltantes
    X = X.fillna(0)
    
    # Información de features
    print(f"\n📊 Shape final de datos:")
    print(f"   Features (X): {X.shape}")
    print(f"   Labels (y): {y.shape}")
    print(f"\n" + "="*80)
    print(f" TOTAL DE FEATURES GENERADAS: {len(X.columns)}")
    print("="*80)
    print(f"\n   Columnas de features ({len(X.columns)}):")
    for i, col in enumerate(X.columns, 1):
        print(f"   {i:2d}. {col}")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\n📊 División de datos:")
    print(f"   Train: {len(X_train)} partidos ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Test:  {len(X_test)} partidos ({len(X_test)/len(X)*100:.1f}%)")
    
    # Escalar features
    print(f"\n⚙️  Escalando features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelos
    print(f"\n" + "="*80)
    print(" ENTRENANDO MODELOS")
    print("="*80)
    
    modelos = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
    }
    
    resultados = {}
    
    for nombre, modelo in modelos.items():
        print(f"\n🤖 Entrenando {nombre}...")
        
        # Entrenar
        modelo.fit(X_train_scaled, y_train)
        
        # Predicciones
        y_pred = modelo.predict(X_test_scaled)
        y_proba = modelo.predict_proba(X_test_scaled)[:, 1]
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        
        # Cross-validation
        if validacion_temporal:
            cv = TimeSeriesSplit(n_splits=5)
            print(f"   Usando TimeSeriesSplit para validación temporal...")
        else:
            cv = 5
        
        cv_scores = cross_val_score(modelo, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        
        resultados[nombre] = {
            'modelo': modelo,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        print(f"   Accuracy: {accuracy*100:.2f}%")
        print(f"   ROC-AUC: {roc_auc:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Optimización XGBoost (opcional)
    if optimizar_hiperparametros:
        print(f"\n" + "="*80)
        print(" OPTIMIZACIÓN DE HIPERPARÁMETROS (XGBoost)")
        print("="*80)
        
        print(f"\n🔍 Buscando mejores hiperparámetros...")
        
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'n_estimators': [150, 200, 250],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        xgb_base = XGBClassifier(
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        grid_search = GridSearchCV(
            xgb_base,
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        print(f"\n   ✅ Mejores parámetros:")
        for param, value in grid_search.best_params_.items():
            print(f"      {param}: {value}")
        
        best_xgb = grid_search.best_estimator_
        y_pred_opt = best_xgb.predict(X_test_scaled)
        y_proba_opt = best_xgb.predict_proba(X_test_scaled)[:, 1]
        
        accuracy_opt = accuracy_score(y_test, y_pred_opt)
        roc_auc_opt = roc_auc_score(y_test, y_proba_opt)
        f1_opt = f1_score(y_test, y_pred_opt)
        
        print(f"\n   Accuracy: {accuracy_opt*100:.2f}%")
        print(f"   ROC-AUC: {roc_auc_opt:.4f}")
        
        resultados['XGBoost Optimizado'] = {
            'modelo': best_xgb,
            'accuracy': accuracy_opt,
            'roc_auc': roc_auc_opt,
            'f1_score': f1_opt,
            'cv_mean': grid_search.best_score_,
            'cv_std': 0,
            'y_pred': y_pred_opt,
            'y_proba': y_proba_opt
        }
    
    # Seleccionar mejor
    mejor_modelo_nombre = max(resultados.items(), key=lambda x: x[1]['accuracy'])[0]
    mejor_modelo_data = resultados[mejor_modelo_nombre]
    mejor_modelo = mejor_modelo_data['modelo']
    
    print(f"\n" + "="*80)
    print(f" MEJOR MODELO: {mejor_modelo_nombre}")
    print("="*80)
    
    print(f"\n📊 MÉTRICAS DEL MODELO:")
    print(f"   Accuracy: {mejor_modelo_data['accuracy']*100:.2f}%")
    print(f"   ROC-AUC: {mejor_modelo_data['roc_auc']:.4f}")
    print(f"   F1-Score: {mejor_modelo_data['f1_score']:.4f}")
    print(f"   CV Score: {mejor_modelo_data['cv_mean']:.4f} (+/- {mejor_modelo_data['cv_std']:.4f})")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, mejor_modelo_data['y_pred'], 
                                target_names=['Away Win', 'Home Win']))
    
    print(f"\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, mejor_modelo_data['y_pred'])
    print(f"                 Predicted")
    print(f"                 Away  Home")
    print(f"   Actual Away   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"   Actual Home   {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Feature importance
    if hasattr(mejor_modelo, 'feature_importances_'):
        print(f"\n🔝 Top 20 Features más importantes:")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': mejor_modelo.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance.head(20).iterrows():
            print(f"   {row['feature']:45s} {row['importance']:.4f}")
        
        import os
        os.makedirs('./models', exist_ok=True)
        feature_importance.to_csv('./models/feature_importance_hybrid.csv', index=False)
        print(f"\n   💾 Feature importance guardada")
    
    # Análisis de confianza
    print(f"\n📊 Análisis de confianza de predicciones:")
    probabilidades = mejor_modelo_data['y_proba']
    confianza_alta = ((probabilidades > 0.7) | (probabilidades < 0.3)).sum()
    confianza_media = ((probabilidades >= 0.55) & (probabilidades <= 0.7) | 
                       (probabilidades >= 0.3) & (probabilidades <= 0.45)).sum()
    confianza_baja = ((probabilidades > 0.45) & (probabilidades < 0.55)).sum()
    
    print(f"   Alta confianza (>70% o <30%): {confianza_alta} ({confianza_alta/len(probabilidades)*100:.1f}%)")
    print(f"   Media confianza (55-70% o 30-45%): {confianza_media} ({confianza_media/len(probabilidades)*100:.1f}%)")
    print(f"   Baja confianza (45-55%): {confianza_baja} ({confianza_baja/len(probabilidades)*100:.1f}%)")
    
    # Guardar modelo
    print(f"\n💾 Guardando modelo...")
    
    import os
    os.makedirs('./models', exist_ok=True)
    
    with open('./models/mlb_model_hybrid_optimized.pkl', 'wb') as f:
        pickle.dump(mejor_modelo, f)
    
    with open('./models/mlb_scaler_hybrid_optimized.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('./models/mlb_feature_names_hybrid_optimized.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    model_info = {
        'nombre': mejor_modelo_nombre,
        'accuracy': mejor_modelo_data['accuracy'],
        'roc_auc': mejor_modelo_data['roc_auc'],
        'f1_score': mejor_modelo_data['f1_score'],
        'cv_mean': mejor_modelo_data['cv_mean'],
        'cv_std': mejor_modelo_data['cv_std'],
        'n_features': len(X.columns),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'feature_names': list(X.columns),
        'validacion_temporal': validacion_temporal,
        'year_scraping': year_scraping
    }
    
    with open('./models/mlb_model_info_hybrid_optimized.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"   ✅ Archivos guardados en ./models/")
    
    print(f"\n" + "="*80)
    print(" ✅ ENTRENAMIENTO COMPLETADO")
    print("="*80)
    print(f"\n Resumen:")
    print(f"   Modelo: {mejor_modelo_nombre}")
    print(f"   Accuracy: {mejor_modelo_data['accuracy']*100:.2f}%")
    print(f"   ROC-AUC: {mejor_modelo_data['roc_auc']:.4f}")
    print(f"   Features: {len(X.columns)}")
    print(f"   Partidos entrenados: {len(X_train)}")
    print(f"   Validación: {'TimeSeriesSplit' if validacion_temporal else 'KFold'}")
    print(f"\n💡 Modelo listo para predicciones!")
    
    return mejor_modelo, scaler, list(X.columns), model_info


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    
    modelo, scaler, feature_names, info = entrenar_modelo_hibrido_optimizado(
        csv_path='./data/processed/datos_ml_ready.csv',
        test_size=0.2,
        usar_cache=True,
        cache_path='./cache/features_hibridas_optimizadas_cache.pkl',
        optimizar_hiperparametros=True,  # True para GridSearchCV (tarda más)
        validacion_temporal=True,          # True = TimeSeriesSplit, False = KFold
        year_scraping=2025,                # Solo scraping a partidos de 2025
        bloque_size=200,                   # Procesar 200 partidos por bloque
        pausa_entre_bloques=60             # Pausar 60s entre bloques
    )
    
    print("\n ¡Entrenamiento completado!")