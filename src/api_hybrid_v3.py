"""
API REST para Predicción de Partidos MLB - MODELO HÍBRIDO V3
FastAPI + XGBoost + Super Features + Bullpen Tracking
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import unicodedata
import re
import cloudscraper
from bs4 import BeautifulSoup
from typing import Optional
import uvicorn
import xgboost as xgb

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
    n_features: int
    version: str
    features_seleccionadas: int

# ============================================================================
# INICIALIZACIÓN DE LA API
# ============================================================================

app = FastAPI(
    title="MLB Game Predictor API - V3",
    description="API con modelo híbrido V3: Super Features + Bullpen + SelectKBest",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
modelo = None
selector = None
feature_names_v3 = None
model_info = None

# Features V3 (las 26 seleccionadas)
FEATURES_V3 = [
    'home_team_OPS', 'away_team_OPS', 'diff_team_BA', 'diff_team_OPS', 
    'diff_team_ERA', 'home_starter_WHIP', 'away_starter_WHIP', 
    'home_starter_ERA', 'away_starter_ERA', 'diff_starter_ERA', 
    'diff_starter_WHIP', 'diff_starter_SO9', 'home_best_OPS', 
    'away_best_OPS', 'diff_best_BA', 'diff_best_OPS', 'diff_best_HR',
    'home_bullpen_WHIP', 'away_bullpen_WHIP', 'diff_bullpen_ERA', 
    'diff_bullpen_WHIP', 'anchor_pitching_level', 'anchor_offensive_level',
    'super_neutralizacion_whip_ops', 'super_resistencia_era_ops', 
    'super_muro_bullpen'
]

# ============================================================================
# FUNCIONES DE SCRAPING
# ============================================================================
def normalizar_texto(texto):
    if not texto:
        return ""
    texto = str(texto).lower()
    # Eliminar acentos
    texto = "".join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    # ELIMINAR TODO LO QUE NO SEA LETRA (Quita asteriscos como el de Rodón*, puntos, etc.)
    texto = re.sub(r'[^a-z0-9]', '', texto)
    return texto

def obtener_html(url, max_retries=3):
    scraper = cloudscraper.create_scraper()
    for intento in range(max_retries):
        try:
            response = scraper.get(url, timeout=15)
            if response.status_code == 200:
                response.encoding = 'utf-8'
                return response.text
            elif response.status_code in [429, 403]:
                import time
                time.sleep((2 ** intento) * 5)
        except Exception as e:
            if intento == max_retries - 1:
                return None
    return None

def limpiar_dataframe(df):
    if df is None or len(df) == 0:
        return df
    if 'Rk' in df.columns:
        df = df.drop('Rk', axis=1)
    name_col = df.columns[0]
    df = df.dropna(subset=[name_col])
    df = df[~df[name_col].astype(str).str.contains(
        r'Team Totals|Rank in|^\s*$', case=False, na=False, regex=True
    )]
    return df.reset_index(drop=True)

def scrape_player_stats(team_code, year, session_cache=None):
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
        
        batting_df = pitching_df = None
        
        if batting_table:
            batting_df = pd.read_html(str(batting_table))[0]
            batting_df = limpiar_dataframe(batting_df)
        
        if pitching_table:
            pitching_df = pd.read_html(str(pitching_table))[0]
            pitching_df = limpiar_dataframe(pitching_df)
        
        if session_cache is not None:
            session_cache[cache_key] = (batting_df, pitching_df)
        
        return batting_df, pitching_df
    except Exception as e:
        return None, None

def safe_float(val):
    try:
        if pd.isna(val): 
            return 0.0
        return float(val)
    except:
        return 0.0

# ============================================================================
# FUNCIONES ESPECÍFICAS V3
# ============================================================================

def encontrar_lanzador(pitching_df, nombre_lanzador):
    if pitching_df is None or len(pitching_df) == 0:
        return None
    
    nombre_busqueda = normalizar_texto(nombre_lanzador)
    name_col = pitching_df.columns[0]
    
    # Buscamos usando tu normalización en ambos lados
    mask = pitching_df[name_col].apply(lambda x: nombre_busqueda in normalizar_texto(x) or normalizar_texto(x) in nombre_busqueda)
    
    if mask.sum() == 0:
        return None
    
    lanzador = pitching_df[mask].iloc[0]
    
    return {
        'nombre': str(lanzador[name_col]),
        'ERA': safe_float(lanzador.get('ERA', 0)),
        'WHIP': safe_float(lanzador.get('WHIP', 0)),
        'H9': safe_float(lanzador.get('H9', 0)),
        'SO9': safe_float(lanzador.get('SO9', 0)),
        'W': safe_float(lanzador.get('W', 0)),
        'L': safe_float(lanzador.get('L', 0)),
        'IP': safe_float(lanzador.get('IP', 0))
    }

def extraer_top_relevistas(pitching_df):
    """Identifica Closer y Setup men (TOP 3 relevistas)"""
    if pitching_df is None or len(pitching_df) == 0:
        return None
    
    cols_relevo = ['SV', 'GF', 'ERA', 'WHIP', 'IP', 'G', 'GS']
    for col in cols_relevo:
        if col in pitching_df.columns:
            pitching_df[col] = pd.to_numeric(pitching_df[col], errors='coerce').fillna(0)
    
    # Filtrar relevistas: GS < 50% de G
    bullpen = pitching_df[pitching_df['GS'] < (pitching_df['G'] * 0.5)].copy()
    if len(bullpen) == 0:
        return None
    
    # Ordenar por jerarquía: Saves > Games Finished > IP
    bullpen = bullpen.sort_values(by=['SV', 'GF', 'IP'], ascending=False)
    top_3 = bullpen.head(3)
    
    return {
        'bullpen_ERA_mean': top_3['ERA'].mean(),
        'bullpen_WHIP_mean': top_3['WHIP'].mean()
    }

def encontrar_mejor_bateador(batting_df):
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
    
    return {
        'best_bat_BA': pd.to_numeric(top_3['BA'], errors='coerce').mean(),
        'best_bat_OBP': top_3['OBP'].mean(),
        'best_bat_OPS': pd.to_numeric(top_3['OPS'], errors='coerce').mean() if 'OPS' in top_3.columns else 0.750,
        'best_bat_HR': pd.to_numeric(top_3['HR'], errors='coerce').mean()
    }

def calcular_stats_equipo(batting_df, pitching_df):
    stats = {}
    
    if batting_df is not None and len(batting_df) > 0:
        for col in ['BA', 'OBP', 'SLG', 'OPS', 'HR']:
            if col in batting_df.columns:
                val = pd.to_numeric(batting_df[col], errors='coerce').mean()
                stats[f'team_{col}_mean'] = val if not pd.isna(val) else 0
    
    if pitching_df is not None and len(pitching_df) > 0:
        for col in ['ERA', 'WHIP', 'SO9', 'H9']:
            if col in pitching_df.columns:
                val = pd.to_numeric(pitching_df[col], errors='coerce').mean()
                stats[f'team_{col}_mean'] = val if not pd.isna(val) else 0
    
    return stats

def encontrar_top_3_bateadores_detailed(batting_df):
    """Para stats detalladas en /predict/detailed"""
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
    
    bateadores = []
    for idx, bateador in top_3.iterrows():
        bateadores.append({
            'nombre': str(bateador[name_col]),
            'BA': safe_float(bateador.get('BA', 0)),
            'OBP': safe_float(bateador.get('OBP', 0)),
            'SLG': safe_float(bateador.get('SLG', 0)),
            'OPS': safe_float(bateador.get('OPS', 0)),
            'HR': int(safe_float(bateador.get('HR', 0))),
            'RBI': int(safe_float(bateador.get('RBI', 0))),
            'R': int(safe_float(bateador.get('R', 0))),
            'AB': int(safe_float(bateador.get('AB', 0)))
        })
    
    return bateadores

# ============================================================================
# EXTRACCIÓN DE FEATURES V3
# ============================================================================

def extraer_features_v3(home_team, away_team, home_pitcher, away_pitcher, year=2026, session_cache=None):
    """
    Extrae las 26 features seleccionadas por SelectKBest
    Incluye super features y bullpen tracking
    """
    
    # Intentar múltiples años
    años_a_probar = [year, year-1, 2025, 2024, 2023]
    batting1 = batting2 = pitching1 = pitching2 = None
    year_usado = None
    
    for año_intento in años_a_probar:
        batting1, pitching1 = scrape_player_stats(home_team, año_intento, session_cache)
        batting2, pitching2 = scrape_player_stats(away_team, año_intento, session_cache)
        
        if all([batting1 is not None, batting2 is not None, 
                pitching1 is not None, pitching2 is not None]):
            year_usado = año_intento
            break
    
    if batting1 is None or batting2 is None:
        raise ValueError(f"No se pudieron obtener datos para {home_team} vs {away_team}")
    
    # 1. Stats de equipos
    stats_h = calcular_stats_equipo(batting1, pitching1)
    stats_a = calcular_stats_equipo(batting2, pitching2)
    
    # 2. Abridores
    sp1 = encontrar_lanzador(pitching1, home_pitcher)
    sp2 = encontrar_lanzador(pitching2, away_pitcher)
    
    # 3. Top 3 bateadores
    hb1 = encontrar_mejor_bateador(batting1)
    hb2 = encontrar_mejor_bateador(batting2)
    
    # 4. Bullpen
    rel_h = extraer_top_relevistas(pitching1)
    rel_a = extraer_top_relevistas(pitching2)
    
    # Construir vector de features
    features = {}
    
    # Features de equipos
    features['home_team_OPS'] = stats_h.get('team_OPS_mean', 0.750)
    features['away_team_OPS'] = stats_a.get('team_OPS_mean', 0.750)
    features['diff_team_BA'] = stats_h.get('team_BA_mean', 0.250) - stats_a.get('team_BA_mean', 0.250)
    features['diff_team_OPS'] = stats_h.get('team_OPS_mean', 0.750) - stats_a.get('team_OPS_mean', 0.750)
    features['diff_team_ERA'] = stats_a.get('team_ERA_mean', 4.0) - stats_h.get('team_ERA_mean', 4.0)
    
    # Features de abridores
    if sp1 and sp2:
        features['home_starter_WHIP'] = sp1['WHIP']
        features['away_starter_WHIP'] = sp2['WHIP']
        features['home_starter_ERA'] = sp1['ERA']
        features['away_starter_ERA'] = sp2['ERA']
        features['diff_starter_ERA'] = sp2['ERA'] - sp1['ERA']
        features['diff_starter_WHIP'] = sp2['WHIP'] - sp1['WHIP']
        features['diff_starter_SO9'] = sp1['SO9'] - sp2['SO9']
    else:
        # Fallback con promedios
        features['home_starter_WHIP'] = stats_h.get('team_WHIP_mean', 1.3)
        features['away_starter_WHIP'] = stats_a.get('team_WHIP_mean', 1.3)
        features['home_starter_ERA'] = stats_h.get('team_ERA_mean', 4.0)
        features['away_starter_ERA'] = stats_a.get('team_ERA_mean', 4.0)
        features['diff_starter_ERA'] = 0
        features['diff_starter_WHIP'] = 0
        features['diff_starter_SO9'] = 0
    
    # Features de bateadores
    if hb1 and hb2:
        features['home_best_OPS'] = hb1['best_bat_OPS']
        features['away_best_OPS'] = hb2['best_bat_OPS']
        features['diff_best_BA'] = hb1['best_bat_BA'] - hb2['best_bat_BA']
        features['diff_best_OPS'] = hb1['best_bat_OPS'] - hb2['best_bat_OPS']
        features['diff_best_HR'] = hb1['best_bat_HR'] - hb2['best_bat_HR']
    else:
        features['home_best_OPS'] = 0.750
        features['away_best_OPS'] = 0.750
        features['diff_best_BA'] = 0
        features['diff_best_OPS'] = 0
        features['diff_best_HR'] = 0
    
    # Features de bullpen
    if rel_h and rel_a:
        features['home_bullpen_WHIP'] = rel_h['bullpen_WHIP_mean']
        features['away_bullpen_WHIP'] = rel_a['bullpen_WHIP_mean']
        features['diff_bullpen_ERA'] = rel_a['bullpen_ERA_mean'] - rel_h['bullpen_ERA_mean']
        features['diff_bullpen_WHIP'] = rel_a['bullpen_WHIP_mean'] - rel_h['bullpen_WHIP_mean']
    else:
        features['home_bullpen_WHIP'] = 1.3
        features['away_bullpen_WHIP'] = 1.3
        features['diff_bullpen_ERA'] = 0
        features['diff_bullpen_WHIP'] = 0
    
    # Anclas
    features['anchor_pitching_level'] = features['home_starter_ERA']
    features['anchor_offensive_level'] = features['home_team_OPS']
    
    # SUPER FEATURES (las que suben el accuracy)
    features['super_neutralizacion_whip_ops'] = (
        features['home_starter_WHIP'] * features['away_team_OPS']
    ) - (
        features['away_starter_WHIP'] * features['home_team_OPS']
    )
    
    features['super_resistencia_era_ops'] = (
        features['home_starter_ERA'] / (features['away_team_OPS'] + 0.01)
    ) - (
        features['away_starter_ERA'] / (features['home_team_OPS'] + 0.01)
    )
    
    features['super_muro_bullpen'] = (
        features['home_bullpen_WHIP'] * features['away_best_OPS']
    ) - (
        features['away_bullpen_WHIP'] * features['home_best_OPS']
    )
    
    return features, year_usado

# ============================================================================
# EVENTOS DE LA API
# ============================================================================

@app.on_event("startup")
async def startup_event():
    global modelo, model_info
    try:
        # Carga nativa para modelos entrenados con train_model_hybrid_v3.py
        modelo = xgb.Booster()
        modelo.load_model('./models/modelo_mlb_v3.json')
        
        model_info = {
            'nombre': 'XGBoost V3 - Super Features',
            'accuracy': 0.607,
            'n_features': 26,
            'version': '3.0.0',
            'features_seleccionadas': 26
        }
        print("✅ Modelo V3 cargado exitosamente y listo para predicciones")
    except Exception as e:
        print(f"❌ Error crítico al cargar el modelo: {e}")

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "MLB Game Predictor API - V3",
        "version": "3.0.0",
        "features": "Super Features + Bullpen + SelectKBest",
        "accuracy": "60.70%",
        "endpoints": {
            "/predict": "POST - Predicción básica",
            "/predict/detailed": "POST - Predicción con stats detalladas",
            "/info": "GET - Información del modelo",
            "/health": "GET - Estado de la API"
        }
    }

@app.get("/health")
async def health_check():
    model_loaded = modelo is not None
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "model_version": "3.0.0",
        "accuracy": "60.70%"
    }

@app.get("/info", response_model=ModeloInfo)
async def get_model_info():
    if model_info is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return model_info

@app.post("/predict", response_model=PartidoResponse)
async def predict_game(partido: PartidoRequest):
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    try:
        session_cache = {}
        features, year_usado = extraer_features_v3(
            partido.home_team, partido.away_team,
            partido.home_pitcher, partido.away_pitcher,
            partido.year, session_cache
        )
        
        # Orden exacto de entrenamiento
        df = pd.DataFrame([features]).reindex(columns=FEATURES_V3, fill_value=0)
        
        # Predicción nativa
        dmatrix = xgb.DMatrix(df)
        prob_home = float(modelo.predict(dmatrix)[0])
        
        return PartidoResponse(
            ganador=partido.home_team if prob_home > 0.5 else partido.away_team,
            prob_home=round(prob_home, 4),
            prob_away=round(1 - prob_home, 4),
            confianza=round(max(prob_home, 1 - prob_home), 4),
            year_usado=year_usado,
            mensaje=f"Datos de temporada {year_usado}" if year_usado != partido.year else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/detailed")
async def predict_game_detailed(partido: PartidoRequest):
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    try:
        session_cache = {}
        features, year_usado = extraer_features_v3(
            partido.home_team, partido.away_team,
            partido.home_pitcher, partido.away_pitcher,
            partido.year, session_cache
        )
        
        bat1, pit1 = scrape_player_stats(partido.home_team, year_usado, session_cache)
        bat2, pit2 = scrape_player_stats(partido.away_team, year_usado, session_cache)
        
        df = pd.DataFrame([features]).reindex(columns=FEATURES_V3, fill_value=0)
        prob_home = float(modelo.predict(xgb.DMatrix(df))[0])
        
        # --- BLOQUE CORREGIDO ---
        return {
            "ganador": partido.home_team if prob_home > 0.5 else partido.away_team,
            "prob_home": round(prob_home, 4),
            "prob_away": round(1 - prob_home, 4),
            "confianza": round(max(prob_home, 1 - prob_home), 4),
            "year_usado": year_usado,
            "features_usadas": features,  # <--- AÑADIMOS ESTA LÍNEA PARA LA WEB APP
            "stats_detalladas": {
                'home_pitcher': encontrar_lanzador(pit1, partido.home_pitcher),
                'away_pitcher': encontrar_lanzador(pit2, partido.away_pitcher),
                'home_batters': encontrar_top_3_bateadores_detailed(bat1),
                'away_batters': encontrar_top_3_bateadores_detailed(bat2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api_hybrid_v3:app",
        host="0.0.0.0",
        port=8000, 
        reload=True
    )