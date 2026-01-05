"""
API REST para Predicción de Partidos MLB - MODELO HÍBRIDO
FastAPI + modelo ML híbrido entrenado (features temporales + scraping)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import cloudscraper
from bs4 import BeautifulSoup
from typing import Optional
import uvicorn

# ============================================================================
# MODELOS DE DATOS (Pydantic)
# ============================================================================

class PartidoRequest(BaseModel):
    home_team: str
    away_team: str
    home_pitcher: str
    away_pitcher: str
    year: Optional[int] = 2026

class PartidoResponse(BaseModel):
    ganador: str
    prob_home: float
    prob_away: float
    confianza: float
    year_usado: int
    mensaje: Optional[str] = None

class ModeloInfo(BaseModel):
    nombre: str
    accuracy: float
    roc_auc: float
    n_features: int
    n_train: int
    n_test: int

# ============================================================================
# INICIALIZACIÓN DE LA API
# ============================================================================

app = FastAPI(
    title="MLB Game Predictor API - Hybrid Model",
    description="API para predecir resultados de partidos de MLB usando modelo híbrido (features temporales + scraping)",
    version="2.0.0"
)

# CORS para permitir requests desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales para el modelo
modelo = None
scaler = None
feature_names = None
model_info = None

# ============================================================================
# FUNCIONES DE SCRAPING (copias necesarias)
# ============================================================================

def obtener_html(url):
    """Accede a la URL usando cloudscraper."""
    scraper = cloudscraper.create_scraper()
    try:
        response = scraper.get(url, timeout=15)
        response.encoding = 'utf-8'
        if response.status_code == 200:
            return response.text
        return None
    except:
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
        'Team Totals|Rank in|^\s*$', 
        case=False, na=False, regex=True
    )]
    return df.reset_index(drop=True)

def scrape_player_stats(team_code, year=2026):
    """Extrae estadísticas de jugadores de un equipo"""
    url = f"https://www.baseball-reference.com/teams/{team_code}/{year}.shtml"
    
    html = obtener_html(url)
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
    
    return batting_df, pitching_df

def safe_float(val):
    """Convierte valores a float de forma segura"""
    try:
        return float(val)
    except:
        return 0.0

def encontrar_lanzador(pitching_df, nombre_lanzador):
    """Busca un lanzador y extrae sus stats"""
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
        'L': safe_float(lanzador.get('L', 0))
    }

def encontrar_mejor_bateador(batting_df):
    """Encuentra los 3 mejores bateadores con filtro de mediana AB"""
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
    
    stats_promedio = {'BA': 0.0, 'OBP': 0.0, 'RBI': 0.0, 'R': 0.0}
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

def simular_features_temporales():
    """
    Para predicción en vivo, usamos valores neutros/promedio
    ya que no tenemos historial del partido que se va a jugar
    """
    return {
        'home_victorias_L10': 0.5,
        'home_runs_anotadas_L10': 4.5,
        'home_runs_recibidas_L10': 4.5,
        'home_racha': 0,
        'home_run_diff_L10': 0,
        'away_victorias_L10': 0.5,
        'away_runs_anotadas_L10': 4.5,
        'away_runs_recibidas_L10': 4.5,
        'away_racha': 0,
        'away_run_diff_L10': 0,
        'home_pitcher_victorias_L5': 0.5,
        'home_pitcher_runs_permitidas_L5': 4.5,
        'away_pitcher_victorias_L5': 0.5,
        'away_pitcher_runs_permitidas_L5': 4.5,
        'h2h_home_win_rate': 0.5,
        'h2h_home_runs_avg': 4.5,
        'h2h_away_runs_avg': 4.5,
        'victorias_diff': 0,
        'run_diff_diff': 0,
        'pitcher_win_rate_diff': 0,
        'pitcher_runs_allowed_diff': 0,
        'racha_diff': 0
    }

def extraer_features(home_team, away_team, home_pitcher, away_pitcher, year=2026):
    """Extrae todas las features necesarias para la predicción (híbridas)"""
    
    # Intentar múltiples años
    años_a_probar = [year, year-1, 2025]
    batting1 = batting2 = pitching1 = pitching2 = None
    year_usado = None
    
    for año_intento in años_a_probar:
        batting1, pitching1 = scrape_player_stats(home_team, año_intento)
        batting2, pitching2 = scrape_player_stats(away_team, año_intento)
        
        if batting1 is not None and batting2 is not None and pitching1 is not None and pitching2 is not None:
            year_usado = año_intento
            break
    
    if batting1 is None or batting2 is None:
        raise ValueError(f"No se pudieron obtener datos para {home_team} vs {away_team}")
    
    # Stats del equipo
    stats_team1 = calcular_stats_equipo(batting1, pitching1)
    stats_team2 = calcular_stats_equipo(batting2, pitching2)
    
    # Lanzadores
    pitcher1_stats = encontrar_lanzador(pitching1, home_pitcher)
    pitcher2_stats = encontrar_lanzador(pitching2, away_pitcher)
    
    # Bateadores
    best_batters1 = encontrar_mejor_bateador(batting1)
    best_batters2 = encontrar_mejor_bateador(batting2)
    
    # Crear vector de features
    features = {}
    
    # Features temporales (simuladas para predicción en vivo)
    features_temporales = simular_features_temporales()
    features.update(features_temporales)
    
    # Features de scraping
    for key, val in stats_team1.items():
        features[f'home_{key}'] = val
    
    for key, val in stats_team2.items():
        features[f'away_{key}'] = val
    
    # Lanzadores con fallback
    if pitcher1_stats:
        features.update({f'home_pitcher_{k}': v for k, v in pitcher1_stats.items()})
    else:
        features.update({
            'home_pitcher_ERA': stats_team1.get('team_ERA_mean', 4.0),
            'home_pitcher_WHIP': stats_team1.get('team_WHIP_mean', 1.3),
            'home_pitcher_H9': stats_team1.get('team_H9_mean', 9.0),
            'home_pitcher_W': 0,
            'home_pitcher_L': 0
        })
    
    if pitcher2_stats:
        features.update({f'away_pitcher_{k}': v for k, v in pitcher2_stats.items()})
    else:
        features.update({
            'away_pitcher_ERA': stats_team2.get('team_ERA_mean', 4.0),
            'away_pitcher_WHIP': stats_team2.get('team_WHIP_mean', 1.3),
            'away_pitcher_H9': stats_team2.get('team_H9_mean', 9.0),
            'away_pitcher_W': 0,
            'away_pitcher_L': 0
        })
    
    # Bateadores con fallback
    if best_batters1:
        features.update({f'home_best_{k}': v for k, v in best_batters1.items()})
    else:
        features.update({
            'home_best_BA': stats_team1.get('team_BA_mean', 0.250),
            'home_best_OBP': stats_team1.get('team_OBP_mean', 0.320),
            'home_best_RBI': 0,
            'home_best_R': 0
        })
    
    if best_batters2:
        features.update({f'away_best_{k}': v for k, v in best_batters2.items()})
    else:
        features.update({
            'away_best_BA': stats_team2.get('team_BA_mean', 0.250),
            'away_best_OBP': stats_team2.get('team_OBP_mean', 0.320),
            'away_best_RBI': 0,
            'away_best_R': 0
        })
    
    # Features derivadas
    features['pitcher_ERA_diff'] = features['away_pitcher_ERA'] - features['home_pitcher_ERA']
    features['pitcher_WHIP_diff'] = features['away_pitcher_WHIP'] - features['home_pitcher_WHIP']
    features['pitcher_H9_diff'] = features['away_pitcher_H9'] - features['home_pitcher_H9']
    features['team_BA_diff'] = features['home_team_BA_mean'] - features['away_team_BA_mean']
    features['team_OBP_diff'] = features['home_team_OBP_mean'] - features['away_team_OBP_mean']
    
    return features, year_usado

# ============================================================================
# EVENTOS DE LA API
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Cargar el modelo híbrido al iniciar la API"""
    global modelo, scaler, feature_names, model_info
    
    try:
        with open('./models/mlb_model_hybrid_optimized.pkl', 'rb') as f:
            modelo = pickle.load(f)
        with open('./models/mlb_scaler_hybrid_optimized.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('./models/mlb_feature_names_hybrid_optimized.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        with open('./models/mlb_model_info_hybrid_optimized.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        print("✅ Modelo híbrido cargado exitosamente")
        print(f"   Modelo: {model_info['nombre']}")
        print(f"   Accuracy: {model_info['accuracy']*100:.2f}%")
        print(f"   Features: {model_info['n_features']}")
        
    except FileNotFoundError:
        print("❌ Error: Archivos del modelo híbrido no encontrados")
        print("   Ejecuta 'python train_model_hybrid_optimized.py' primero")

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Endpoint raíz - información de la API"""
    return {
        "message": "MLB Game Predictor API - Hybrid Model",
        "version": "2.0.0",
        "model": "Hybrid (Temporal Features + Scraping)",
        "endpoints": {
            "/predict": "POST - Predecir resultado de un partido",
            "/predict/detailed": "POST - Predecir con stats detalladas",
            "/info": "GET - Información del modelo",
            "/health": "GET - Estado de la API"
        }
    }

@app.get("/health")
async def health_check():
    """Verifica el estado de la API"""
    model_loaded = modelo is not None
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "model_type": "hybrid_optimized"
    }

@app.get("/info", response_model=ModeloInfo)
async def get_model_info():
    """Obtiene información del modelo cargado"""
    if model_info is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    return model_info

@app.post("/predict", response_model=PartidoResponse)
async def predict_game(partido: PartidoRequest):
    """
    Predice el resultado de un partido usando modelo híbrido
    
    - **home_team**: Código del equipo local (ej: BOS, NYY, LAD)
    - **away_team**: Código del equipo visitante
    - **home_pitcher**: Nombre o apellido del lanzador local
    - **away_pitcher**: Nombre o apellido del lanzador visitante
    - **year**: Temporada (default: 2026)
    """
    
    if modelo is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        # Extraer features
        features, year_usado = extraer_features(
            partido.home_team,
            partido.away_team,
            partido.home_pitcher,
            partido.away_pitcher,
            partido.year
        )
        
        # Preparar para predicción
        features_df = pd.DataFrame([features])
        
        for col in feature_names:
            if col not in features_df.columns:
                features_df[col] = 0
        
        features_df = features_df[feature_names]
        features_df = features_df.fillna(0)
        
        # Predecir
        features_scaled = scaler.transform(features_df)
        prediccion = modelo.predict(features_scaled)[0]
        probabilidades = modelo.predict_proba(features_scaled)[0]
        
        ganador = partido.home_team if prediccion == 1 else partido.away_team
        
        mensaje = None
        if year_usado != partido.year:
            mensaje = f"Usando datos de temporada {year_usado} (no disponibles para {partido.year})"
        
        return PartidoResponse(
            ganador=ganador,
            prob_home=float(probabilidades[1]),
            prob_away=float(probabilidades[0]),
            confianza=float(max(probabilidades)),
            year_usado=year_usado,
            mensaje=mensaje
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@app.post("/predict/detailed")
async def predict_game_detailed(partido: PartidoRequest):
    """
    Predice el resultado de un partido CON STATS DETALLADAS
    Incluye stats de lanzadores y top 3 bateadores
    """
    
    if modelo is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        # Extraer features Y stats detalladas
        features, year_usado, stats_detalladas = extraer_features_con_stats(
            partido.home_team,
            partido.away_team,
            partido.home_pitcher,
            partido.away_pitcher,
            partido.year
        )
        
        # Preparar para predicción
        features_df = pd.DataFrame([features])
        
        for col in feature_names:
            if col not in features_df.columns:
                features_df[col] = 0
        
        features_df = features_df[feature_names]
        features_df = features_df.fillna(0)
        
        # Predecir
        features_scaled = scaler.transform(features_df)
        prediccion = modelo.predict(features_scaled)[0]
        probabilidades = modelo.predict_proba(features_scaled)[0]
        
        ganador = partido.home_team if prediccion == 1 else partido.away_team
        
        mensaje = None
        if year_usado != partido.year:
            mensaje = f"Usando datos de temporada {year_usado}"
        
        return {
            "ganador": ganador,
            "prob_home": float(probabilidades[1]),
            "prob_away": float(probabilidades[0]),
            "confianza": float(max(probabilidades)),
            "year_usado": year_usado,
            "mensaje": mensaje,
            "stats_detalladas": stats_detalladas
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


def extraer_features_con_stats(home_team, away_team, home_pitcher, away_pitcher, year=2026):
    """
    Extrae features Y guarda stats detalladas de lanzadores y bateadores
    """
    
    # Intentar múltiples años
    años_a_probar = [year, year-1, 2025]
    batting1 = batting2 = pitching1 = pitching2 = None
    year_usado = None
    
    for año_intento in años_a_probar:
        batting1, pitching1 = scrape_player_stats(home_team, año_intento)
        batting2, pitching2 = scrape_player_stats(away_team, año_intento)
        
        if batting1 is not None and batting2 is not None and pitching1 is not None and pitching2 is not None:
            year_usado = año_intento
            break
    
    if batting1 is None or batting2 is None:
        raise ValueError(f"No se pudieron obtener datos para {home_team} vs {away_team}")
    
    # Stats del equipo
    stats_team1 = calcular_stats_equipo(batting1, pitching1)
    stats_team2 = calcular_stats_equipo(batting2, pitching2)
    
    # Lanzadores CON DETALLES
    pitcher1_stats = encontrar_lanzador(pitching1, home_pitcher)
    pitcher2_stats = encontrar_lanzador(pitching2, away_pitcher)
    
    # Top 3 bateadores CON DETALLES
    best_batters1_detailed = encontrar_top_3_bateadores_detailed(batting1)
    best_batters2_detailed = encontrar_top_3_bateadores_detailed(batting2)
    
    # Promedios de top 3 para el modelo
    best_batters1 = promediar_top_3(best_batters1_detailed) if best_batters1_detailed else None
    best_batters2 = promediar_top_3(best_batters2_detailed) if best_batters2_detailed else None
    
    # Crear vector de features
    features = {}
    
    # Features temporales (simuladas)
    features_temporales = simular_features_temporales()
    features.update(features_temporales)
    
    # Features de scraping
    for key, val in stats_team1.items():
        features[f'home_{key}'] = val
    
    for key, val in stats_team2.items():
        features[f'away_{key}'] = val
    
    # Lanzadores
    if pitcher1_stats:
        features.update({f'home_pitcher_{k}': v for k, v in pitcher1_stats.items()})
    else:
        features.update({
            'home_pitcher_ERA': stats_team1.get('team_ERA_mean', 4.0),
            'home_pitcher_WHIP': stats_team1.get('team_WHIP_mean', 1.3),
            'home_pitcher_H9': stats_team1.get('team_H9_mean', 9.0),
            'home_pitcher_W': 0,
            'home_pitcher_L': 0
        })
    
    if pitcher2_stats:
        features.update({f'away_pitcher_{k}': v for k, v in pitcher2_stats.items()})
    else:
        features.update({
            'away_pitcher_ERA': stats_team2.get('team_ERA_mean', 4.0),
            'away_pitcher_WHIP': stats_team2.get('team_WHIP_mean', 1.3),
            'away_pitcher_H9': stats_team2.get('team_H9_mean', 9.0),
            'away_pitcher_W': 0,
            'away_pitcher_L': 0
        })
    
    # Bateadores
    if best_batters1:
        features.update({f'home_best_{k}': v for k, v in best_batters1.items()})
    else:
        features.update({
            'home_best_BA': stats_team1.get('team_BA_mean', 0.250),
            'home_best_OBP': stats_team1.get('team_OBP_mean', 0.320),
            'home_best_RBI': 0,
            'home_best_R': 0
        })
    
    if best_batters2:
        features.update({f'away_best_{k}': v for k, v in best_batters2.items()})
    else:
        features.update({
            'away_best_BA': stats_team2.get('team_BA_mean', 0.250),
            'away_best_OBP': stats_team2.get('team_OBP_mean', 0.320),
            'away_best_RBI': 0,
            'away_best_R': 0
        })
    
    # Features derivadas
    features['pitcher_ERA_diff'] = features['away_pitcher_ERA'] - features['home_pitcher_ERA']
    features['pitcher_WHIP_diff'] = features['away_pitcher_WHIP'] - features['home_pitcher_WHIP']
    features['pitcher_H9_diff'] = features['away_pitcher_H9'] - features['home_pitcher_H9']
    features['team_BA_diff'] = features['home_team_BA_mean'] - features['away_team_BA_mean']
    features['team_OBP_diff'] = features['home_team_OBP_mean'] - features['away_team_OBP_mean']
    
    # Stats detalladas para mostrar
    stats_detalladas = {
        'home_pitcher': pitcher1_stats,
        'away_pitcher': pitcher2_stats,
        'home_batters': best_batters1_detailed,
        'away_batters': best_batters2_detailed
    }
    
    return features, year_usado, stats_detalladas


def encontrar_top_3_bateadores_detailed(batting_df):
    """
    Encuentra los 3 mejores bateadores con detalles completos
    """
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
    
    name_col = top_3.columns[0]
    
    bateadores = []
    for idx, bateador in top_3.iterrows():
        bateadores.append({
            'nombre': str(bateador[name_col]),
            'BA': safe_float(bateador.get('BA', 0)),
            'OBP': safe_float(bateador.get('OBP', 0)),
            'SLG': safe_float(bateador.get('SLG', 0)),
            'OPS': safe_float(bateador.get('OPS', 0)),
            'RBI': int(safe_float(bateador.get('RBI', 0))),
            'R': int(safe_float(bateador.get('R', 0))),
            'HR': int(safe_float(bateador.get('HR', 0))),
            'AB': int(safe_float(bateador.get('AB', 0)))
        })
    
    return bateadores


def promediar_top_3(bateadores_list):
    """Calcula el promedio de los top 3 bateadores"""
    if not bateadores_list:
        return None
    
    return {
        'BA': np.mean([b['BA'] for b in bateadores_list]),
        'OBP': np.mean([b['OBP'] for b in bateadores_list]),
        'RBI': np.mean([b['RBI'] for b in bateadores_list]),
        'R': np.mean([b['R'] for b in bateadores_list])
    }


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api_hybrid:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )