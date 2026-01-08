"""
Script de Entrenamiento ML Híbrido OPTIMIZADO para Predicción MLB - VERSIÓN V3 UNIFICADA
LÓGICA HÍBRIDA INTELIGENTE:
- Partidos antiguos (2022-2024): Solo features temporales (CSV)
- Partidos recientes (2025): Features temporales + scraping
Sistema de bloques para evitar rate limiting
Cache incremental para entrenar por etapas
NUEVAS MEJORAS: Identificación de Relevistas y Selección SelectKBest
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif  # NUEVO V3
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                            roc_auc_score, f1_score)
import xgboost as xgb
from xgboost import XGBClassifier
import pickle
import cloudscraper
from bs4 import BeautifulSoup
import time
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


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

def safe_float(val):
    """Convierte a float de forma segura manejando errores y NaNs"""
    try:
        if pd.isna(val): return 0.0
        return float(val)
    except:
        return 0.0

def encontrar_lanzador(pitching_df, nombre_lanzador):
    """Busca un lanzador específico en el dataframe de pitcheo y extrae sus stats"""
    if pitching_df is None or len(pitching_df) == 0:
        return None
    
    # Normalizar nombre para búsqueda
    nombre_busqueda = str(nombre_lanzador).lower().strip()
    # Baseball-Reference a veces tiene caracteres especiales o * / # junto al nombre
    name_col = pitching_df.columns[0]
    
    mask = pitching_df[name_col].astype(str).str.lower().apply(
        lambda x: nombre_busqueda in x or x in nombre_busqueda
    )
    
    if mask.sum() == 0:
        return None
    
    lanzador = pitching_df[mask].iloc[0]
    
    return {
        'ERA': safe_float(lanzador.get('ERA', 0)),
        'WHIP': safe_float(lanzador.get('WHIP', 0)),
        'H9': safe_float(lanzador.get('H9', 0)),
        'SO9': safe_float(lanzador.get('SO9', 0)),
        # 'W': safe_float(lanzador.get('W', 0)),
        # 'L': safe_float(lanzador.get('L', 0)),
        'IP': safe_float(lanzador.get('IP', 0)),
        'G': safe_float(lanzador.get('G', 0)),
        'GS': safe_float(lanzador.get('GS', 0))
    }

def encontrar_mejor_bateador(batting_df):
    """Encuentra las estadísticas de los mejores bateadores del equipo"""
    if batting_df is None or len(batting_df) == 0:
        return None
    
    if 'OBP' not in batting_df.columns or 'AB' not in batting_df.columns:
        return None
    
    # Asegurar conversión numérica para filtrar
    df = batting_df.copy()
    df['OBP'] = pd.to_numeric(df['OBP'], errors='coerce')
    df['AB'] = pd.to_numeric(df['AB'], errors='coerce')
    df = df.dropna(subset=['OBP', 'AB'])
    
    if len(df) == 0:
        return None
    
    # Filtrar por mediana de turnos al bate para evitar ruidos de novatos o lesionados
    mediana_ab = df['AB'].median()
    df_filtrado = df[df['AB'] >= mediana_ab].copy()
    
    if len(df_filtrado) == 0:
        df_filtrado = df
        
    # Ordenar por OBP y tomar los 3 mejores
    top_3 = df_filtrado.sort_values('OBP', ascending=False).head(3)
    
    return {
        'best_bat_BA': pd.to_numeric(top_3['BA'], errors='coerce').mean(),
        'best_bat_OBP': top_3['OBP'].mean(),
        'best_bat_OPS': pd.to_numeric(top_3['OPS'], errors='coerce').mean() if 'OPS' in top_3.columns else 0.750,
        'best_bat_HR': pd.to_numeric(top_3['HR'], errors='coerce').mean(),
        'best_bat_RBI': pd.to_numeric(top_3['RBI'], errors='coerce').mean()
    }

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
        ganado = (p['ganador'] == 1) if es_home else (p['ganador'] == 0)
        if ganado: victorias += 1
        
        carreras_f += p['score_home'] if es_home else p['score_away']
        carreras_c += p['score_away'] if es_home else p['score_home']
        
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
# EXTRACCIÓN DE FEATURES HÍBRIDA (UNIFICADA)
# ============================================================================
def extraer_features_hibridas(row, hacer_scraping=False, session_cache=None):
    features = {}
    
    if hacer_scraping:
        bat1, pit1 = scrape_player_stats(row['home_team'], row['year'], session_cache)
        bat2, pit2 = scrape_player_stats(row['away_team'], row['year'], session_cache)
        
        # 1. Stats de Equipo (Individuales y Diferenciales)
        stats_h = calcular_stats_equipo(bat1, pit1)
        stats_a = calcular_stats_equipo(bat2, pit2)
        
        if stats_h and stats_a:
            # Valores Individuales (NUEVO para Súper Features)
            features['home_team_OPS'] = stats_h.get('team_OPS_mean', 0)
            features['away_team_OPS'] = stats_a.get('team_OPS_mean', 0)
            
            # Diferenciales originales
            features['diff_team_BA'] = stats_h.get('team_BA_mean', 0) - stats_a.get('team_BA_mean', 0)
            features['diff_team_OPS'] = stats_h.get('team_OPS_mean', 0) - stats_a.get('team_OPS_mean', 0)
            features['diff_team_ERA'] = stats_a.get('team_ERA_mean', 0) - stats_h.get('team_ERA_mean', 0)

        # 2. Abridores (Individuales y Diferenciales)
        sp1 = encontrar_lanzador(pit1, row['home_pitcher'])
        sp2 = encontrar_lanzador(pit2, row['away_pitcher'])
        
        if sp1 and sp2:
            # Valores Individuales (NUEVO para Súper Features)
            features['home_starter_WHIP'] = sp1.get('WHIP', 0)
            features['away_starter_WHIP'] = sp2.get('WHIP', 0)
            features['home_starter_ERA'] = sp1.get('ERA', 0)
            features['away_starter_ERA'] = sp2.get('ERA', 0)

            # Diferenciales originales
            features['diff_starter_ERA'] = sp2.get('ERA', 0) - sp1.get('ERA', 0)
            features['diff_starter_WHIP'] = sp2.get('WHIP', 0) - sp1.get('WHIP', 0)
            features['diff_starter_W'] = sp1.get('W', 0) - sp2.get('W', 0)
            features['diff_starter_L'] = sp2.get('L', 0) - sp1.get('L', 0)
            features['diff_starter_SO9'] = sp1.get('SO9', 0) - sp2.get('SO9', 0)

        # 3. Mejores Bateadores (Individuales y Diferenciales)
        hb1 = encontrar_mejor_bateador(bat1)
        hb2 = encontrar_mejor_bateador(bat2)
        
        if hb1 and hb2:
            # Valores Individuales (NUEVO para Súper Features)
            features['home_best_OPS'] = hb1.get('best_bat_OPS', 0)
            features['away_best_OPS'] = hb2.get('best_bat_OPS', 0)

            # Diferenciales originales
            features['diff_best_BA'] = hb1.get('best_bat_BA', 0) - hb2.get('best_bat_BA', 0)
            features['diff_best_R'] = hb1.get('best_bat_R', 0) - hb2.get('best_bat_R', 0)
            features['diff_best_OPS'] = hb1.get('best_bat_OPS', 0) - hb2.get('best_bat_OPS', 0)
            features['diff_best_HR'] = hb1.get('best_bat_HR', 0) - hb2.get('best_bat_HR', 0)

        # 4. Bullpen (Individuales y Diferenciales)
        rel_h = extraer_top_relevistas(pit1)
        rel_a = extraer_top_relevistas(pit2)
        if rel_h and rel_a:
            # Valores Individuales (NUEVO para Súper Features)
            features['home_bullpen_WHIP'] = rel_h.get('bullpen_WHIP_mean', 0)
            features['away_bullpen_WHIP'] = rel_a.get('bullpen_WHIP_mean', 0)

            # Diferenciales originales
            features['diff_bullpen_ERA'] = rel_a.get('bullpen_ERA_mean', 0) - rel_h.get('bullpen_ERA_mean', 0)
            features['diff_bullpen_WHIP'] = rel_a.get('bullpen_WHIP_mean', 0) - rel_h.get('bullpen_WHIP_mean', 0)
            features['diff_bullpen_K9'] = rel_h.get('bullpen_K9', 0) - rel_a.get('bullpen_K9', 0)

        # 5. Anclas
        if sp1: features['anchor_pitching_level'] = sp1.get('ERA', 0)
        if stats_h: features['anchor_offensive_level'] = stats_h.get('team_OPS_mean', 0)

    return features
# ============================================================================
# ENTRENAMIENTO DEL MODELO
# ============================================================================
def entrenar_modelo_hibrido_optimizado(
    csv_path='./data/processed/datos_ml_ready.csv',
    test_size=0.2,
    usar_cache=True,
    cache_path='./cache/features_hibridas_v3_cache.pkl',
    modelo_path='./models/modelo_mlb_v3.json',
    optimizar_hiperparametros=True,
    validacion_temporal=True,
    year_scraping=2023,
    bloque_size=150,
    pausa_entre_bloques=45
):
    print("\n" + "="*80)
    print(" INICIANDO ENTRENAMIENTO HÍBRIDO MLB V3")
    print("="*80)
    
    if not os.path.exists(csv_path):
        print(f" Error: Archivo {csv_path} no encontrado.")
        return None, None, None, None

    # Carga de datos originales
    df = pd.read_csv(csv_path)
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values('fecha').reset_index(drop=True)
    df['year'] = df['fecha'].dt.year

    # Manejo de Cache Incremental
    X_dict_list = []
    y_list = []
    indices_procesados = []
    
    if usar_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                data_cache = pickle.load(f)
                X_dict_list = data_cache['X_list']
                y_list = data_cache['y_list']
                indices_procesados = data_cache['indices']
            print(f" Cache cargado: {len(indices_procesados)} partidos ya procesados.")
        except:
            print(" Error cargando cache, iniciando desde cero.")
    
    # Identificar partidos pendientes
    todos_indices = set(range(len(df)))
    pendientes = sorted(list(todos_indices - set(indices_procesados)))
    
    if pendientes:
        print(f" Procesando {len(pendientes)} partidos restantes en bloques de {bloque_size}...")
        session_cache = {}
        
        for i, idx in enumerate(pendientes):
            row = df.iloc[idx]
            hacer_scraping = (row['year'] >= year_scraping)
            
            try:
                f = extraer_features_hibridas(row, df, hacer_scraping, session_cache)
                X_dict_list.append(f)
                y_list.append(row['ganador'])
                indices_procesados.append(idx)
            except Exception as e:
                print(f"       Error en idx {idx}: {e}")
                continue
            
            # Control de flujo y guardado por bloques
            if (i + 1) % bloque_size == 0:
                print(f"       Guardando bloque... ({len(indices_procesados)} total)")
                with open(cache_path, 'wb') as f:
                    pickle.dump({'X_list': X_dict_list, 'y_list': y_list, 'indices': indices_procesados}, f)
                if hacer_scraping:
                    print(f"       Pausa de seguridad ({pausa_entre_bloques}s)...")
                    time.sleep(pausa_entre_bloques)
        
        # Guardado final tras procesar todos
        with open(cache_path, 'wb') as f:
            pickle.dump({'X_list': X_dict_list, 'y_list': y_list, 'indices': indices_procesados}, f)

    # Convertir a DataFrame y limpiar
    X = pd.DataFrame(X_dict_list).fillna(0)
    y = np.array(y_list)
    
    # --- INTEGRACIÓN DE NEUTRALIZACIÓN ---
    if 'home_starter_WHIP' in X.columns:
        X['super_neutralizacion_whip_ops'] = (X['home_starter_WHIP'] * X['away_team_OPS']) - \
                                             (X['away_starter_WHIP'] * X['home_team_OPS'])
        X['super_resistencia_era_ops'] = (X['home_starter_ERA'] / (X['away_team_OPS'] + 0.01)) - \
                                         (X['away_starter_ERA'] / (X['home_team_OPS'] + 0.01))
        X['super_muro_bullpen'] = (X['home_bullpen_WHIP'] * X['away_best_OPS']) - \
                                   (X['away_bullpen_WHIP'] * X['home_best_OPS'])

    print(f"\n Matriz de features finalizada: {X.shape[0]} filas, {X.shape[1]} columnas.")

    # 1. Escalado de datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. SELECCIÓN DE LAS MEJORES FEATURES
    print(" Seleccionando las 26 mejores variables mediante SelectKBest...")
    selector = SelectKBest(score_func=f_classif, k=min(26, X.shape[1]))
    X_selected = selector.fit_transform(X_scaled, y)
    cols_seleccionadas = X.columns[selector.get_support()].tolist()

    # 3. División de datos
    if validacion_temporal:
        split_idx = int(len(X_selected) * (1 - test_size))
        X_train, X_test = X_selected[:split_idx], X_selected[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=test_size, random_state=42, stratify=y
        )

    # 4. Entrenamiento Exclusivo con XGBoost
    print("\n Entrenando el motor XGBoost con hiperparámetros de alta precisión...")
    
    xgb_base = XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=2,
        subsample=0.8,
        colsample_bytree=0.8
    )

    if optimizar_hiperparametros:
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.03, 0.05],
            'gamma': [0.1, 0.2]
        }
        cv_strategy = TimeSeriesSplit(n_splits=5) if validacion_temporal else 5
        grid = GridSearchCV(xgb_base, param_grid, cv=cv_strategy, scoring='accuracy', verbose=1)
        grid.fit(X_train, y_train)
        mejor_modelo = grid.best_estimator_
    else:
        mejor_modelo = xgb_base.fit(X_train, y_train)

    acc = accuracy_score(y_test, mejor_modelo.predict(X_test))
    
    # --- REPORTE ORGANIZADO DE RESULTADOS ---
    print("\n" + "="*70)
    print(" RESUMEN DE MODELO MLB V3")
    print("="*70)
    print(f" Partidos procesados:      {len(X)}")
    print(f" Variables seleccionadas:   {len(cols_seleccionadas)}")
    print("-" * 70)
    print(" LISTA DE VARIABLES CLAVE (TOP 26):")
    for i in range(0, len(cols_seleccionadas), 2):
        cols = cols_seleccionadas[i:i+2]
        line = "  ".join(f"- {c:<30}" for c in cols)
        print(line)
    print("-" * 70)
    print(" CONFIGURACION GANADORA:")
    params = mejor_modelo.get_params()
    print(f" > max_depth:      {params['max_depth']}")
    print(f" > learning_rate:  {params['learning_rate']}")
    print(f" > n_estimators:   {params['n_estimators']}")
    print(f" > gamma:          {params['gamma']}")
    print("-" * 70)
    print(f" RENDIMIENTO FINAL (ACCURACY): {acc:.2%}")
    print("="*70 + "\n")

    # Guardado del modelo
    save_path = modelo_path.replace('.pkl', '.json')
    print(f" Guardando modelo en {save_path}")
    mejor_modelo.get_booster().save_model(save_path)

    return mejor_modelo, X_test, y_test, selector
# ============================================================================
# EJECUCIÓN (ORIGINAL)
# ============================================================================

if __name__ == "__main__":
    entrenar_modelo_hibrido_optimizado(
        optimizar_hiperparametros=True, # Cambiar a False para rapidez
        validacion_temporal=True
    )