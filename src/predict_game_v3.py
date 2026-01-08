"""
Script de Predicción CLI - Modelo V3
Usa modelo XGBoost con Super Features + Bullpen Tracking
"""

import pickle
import sys
import pandas as pd
import numpy as np
import cloudscraper
from bs4 import BeautifulSoup
import xgboost as xgb
import unicodedata
import re
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# MAPEO DE EQUIPOS MLB
# ============================================================================

TEAM_MAPPING = {
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

CODE_TO_FULL_NAME = {
    'ARI': 'Arizona Diamondbacks', 'ATL': 'Atlanta Braves',
    'BAL': 'Baltimore Orioles', 'BOS': 'Boston Red Sox',
    'CHC': 'Chicago Cubs', 'CHW': 'Chicago White Sox',
    'CIN': 'Cincinnati Reds', 'CLE': 'Cleveland Guardians',
    'COL': 'Colorado Rockies', 'DET': 'Detroit Tigers',
    'HOU': 'Houston Astros', 'KCR': 'Kansas City Royals',
    'LAA': 'Los Angeles Angels', 'LAD': 'Los Angeles Dodgers',
    'MIA': 'Miami Marlins', 'MIL': 'Milwaukee Brewers',
    'MIN': 'Minnesota Twins', 'NYM': 'New York Mets',
    'NYY': 'New York Yankees', 'OAK': 'Oakland Athletics',
    'PHI': 'Philadelphia Phillies', 'PIT': 'Pittsburgh Pirates',
    'SDP': 'San Diego Padres', 'SEA': 'Seattle Mariners',
    'SFG': 'San Francisco Giants', 'STL': 'St. Louis Cardinals',
    'TBR': 'Tampa Bay Rays', 'TEX': 'Texas Rangers',
    'TOR': 'Toronto Blue Jays', 'WSN': 'Washington Nationals'
}

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
# FUNCIONES DE VALIDACIÓN
# ============================================================================
def normalizar_texto(texto):
    if not texto:
        return ""
    # Convertir a string y minúsculas
    texto = str(texto).lower()
    # Eliminar acentos
    texto = "".join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    # ELIMINAR TODO LO QUE NO SEA LETRA (Quita asteriscos, puntos, etc.)
    texto = re.sub(r'[^a-z0-9]', '', texto)
    return texto

def normalizar_equipo(team_input):
    if not team_input:
        return None, None
    team_clean = team_input.strip()
    for key, code in TEAM_MAPPING.items():
        if key.lower() == team_clean.lower():
            return code, CODE_TO_FULL_NAME.get(code, code)
    team_lower = team_clean.lower()
    matches = []
    for key, code in TEAM_MAPPING.items():
        if team_lower in key.lower() or key.lower() in team_lower:
            matches.append((code, CODE_TO_FULL_NAME.get(code, code), key))
    if len(matches) == 1:
        return matches[0][0], matches[0][1]
    if len(matches) > 1:
        print(f"\n  '{team_input}' es ambiguo. ¿Te refieres a:")
        for i, (code, full_name, key) in enumerate(matches, 1):
            print(f"   {i}. {full_name} ({code})")
        return None, None
    return None, None

def mostrar_equipos_disponibles():
    print("\n EQUIPOS MLB DISPONIBLES:")
    print("="*70)
    for code in sorted(CODE_TO_FULL_NAME.keys()):
        print(f"   {code:4s} = {CODE_TO_FULL_NAME[code]}")
    print("="*70)

def validar_y_mostrar_equipo(team_input, role="Equipo"):
    code, full_name = normalizar_equipo(team_input)
    if code:
        print(f"    {role}: {full_name} ({code})")
        return code
    else:
        print(f"    No se reconoce '{team_input}'")
        return None

# ============================================================================
# FUNCIONES DE SCRAPING (VERSIÓN V3)
# ============================================================================

def obtener_html(url):
    scraper = cloudscraper.create_scraper()
    try:
        response = scraper.get(url, timeout=15)
        response.encoding = 'utf-8'
        return response.text if response.status_code == 200 else None
    except:
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

def scrape_player_stats(team_code, year=2025):
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
    try:
        if pd.isna(val):
            return 0.0
        return float(val)
    except:
        return 0.0

def encontrar_lanzador(pitching_df, nombre_lanzador):
    if pitching_df is None or len(pitching_df) == 0:
        return None
    
    # Lo que el usuario escribió (ej: "rondon" -> "rondon")
    busqueda = normalizar_texto(nombre_lanzador)
    
    # Intentamos encontrar la columna de nombres (suele ser 'Name' o la primera)
    name_col = 'Name' if 'Name' in pitching_df.columns else pitching_df.columns[0]
    
    # Buscamos en el DataFrame
    for _, fila in pitching_df.iterrows():
        nombre_tabla_original = str(fila[name_col])
        nombre_tabla_limpio = normalizar_texto(nombre_tabla_original)
        
        # Comparación: ¿Está el texto buscado dentro del nombre de la tabla?
        if busqueda in nombre_tabla_limpio or nombre_tabla_limpio in busqueda:
            return {
                'nombre': nombre_tabla_original,
                'ERA': safe_float(fila.get('ERA', 0)),
                'WHIP': safe_float(fila.get('WHIP', 0)),
                'H9': safe_float(fila.get('H9', 0)),
                'SO9': safe_float(fila.get('SO9', 0)),
                'W': safe_float(fila.get('W', 0)),
                'L': safe_float(fila.get('L', 0))
            }
    
    return None

def extraer_top_relevistas(pitching_df):
    if pitching_df is None or len(pitching_df) == 0:
        return None
    cols_relevo = ['SV', 'GF', 'ERA', 'WHIP', 'IP', 'G', 'GS']
    for col in cols_relevo:
        if col in pitching_df.columns:
            pitching_df[col] = pd.to_numeric(pitching_df[col], errors='coerce').fillna(0)
    bullpen = pitching_df[pitching_df['GS'] < (pitching_df['G'] * 0.5)].copy()
    if len(bullpen) == 0:
        return None
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

# ============================================================================
# FUNCIÓN DE PREDICCIÓN
# ============================================================================

def predecir_partido(home_team, away_team, home_pitcher, away_pitcher, year=2026):
    print("\n" + "="*70)
    print("  PREDICTOR MLB - MODELO V3 (SUPER FEATURES)")
    print("="*70)
    
    home_full = CODE_TO_FULL_NAME.get(home_team, home_team)
    away_full = CODE_TO_FULL_NAME.get(away_team, away_team)
    
    print(f"\n  {home_full} ({home_team}) vs {away_full} ({away_team})")
    print(f" Temporada: {year}")
    
    # Cargar modelo V3
    try:
        print(f"\n Cargando modelo V3...")
        modelo = xgb.Booster()
        modelo.load_model('./models/modelo_mlb_v3.json')
        print(f"   ✅ Modelo XGBoost V3")
        print(f"   ✅ Accuracy: 60.70%")
        print(f"   ✅ Features: 26 (SelectKBest)")
    except FileNotFoundError:
        print(f"\n❌ Error: Modelo V3 no encontrado")
        print(f"   Ejecuta: python train_model_hybrid_v3.py")
        return None
        
    # --- SCRAPING DIRECTO POR AÑO ---
    print("\n" + "="*70)
    print(f" EXTRAYENDO DATOS DE LA TEMPORADA {year}")
    print("="*70)
    
    # Intentamos primero el año solicitado de forma directa
    batting1, pitching1 = scrape_player_stats(home_team, year)
    batting2, pitching2 = scrape_player_stats(away_team, year)
    
    year_usado = year

    # Solo si el año solicitado falla totalmente, avisamos y buscamos el anterior
    if batting1 is None or batting2 is None:
        print(f"Advertencia: No hay datos para {year}. Buscando temporada previa...")
        year_usado = year - 1
        batting1, pitching1 = scrape_player_stats(home_team, year_usado)
        batting2, pitching2 = scrape_player_stats(away_team, year_usado)
    
    if batting1 is None:
        print(f"Error: Imposible obtener datos para {home_team}")
        return None
    
    # Stats de equipos
    stats_h = calcular_stats_equipo(batting1, pitching1)
    stats_a = calcular_stats_equipo(batting2, pitching2)
    
    print(f"\n Stats Equipo {home_team}:")
    print(f"   OPS: {stats_h.get('team_OPS_mean', 0):.3f}")
    print(f"   ERA: {stats_h.get('team_ERA_mean', 0):.2f}")
    
    print(f"\n Stats Equipo {away_team}:")
    print(f"   OPS: {stats_a.get('team_OPS_mean', 0):.3f}")
    print(f"   ERA: {stats_a.get('team_ERA_mean', 0):.2f}")
    
    # Lanzadores
    print(f"\n Lanzador {home_team}: '{home_pitcher}'")
    sp1 = encontrar_lanzador(pitching1, home_pitcher)
    if sp1:
        print(f"  ✓ {sp1['nombre']}")
        print(f"    ERA: {sp1['ERA']:.2f} | WHIP: {sp1['WHIP']:.3f} | SO9: {sp1['SO9']:.2f}")
    else:
        print(f"     No encontrado, usando promedio del equipo")
    
    print(f"\n Lanzador {away_team}: '{away_pitcher}'")
    sp2 = encontrar_lanzador(pitching2, away_pitcher)
    if sp2:
        print(f"  ✓ {sp2['nombre']}")
        print(f"    ERA: {sp2['ERA']:.2f} | WHIP: {sp2['WHIP']:.3f} | SO9: {sp2['SO9']:.2f}")
    else:
        print(f"     No encontrado, usando promedio del equipo")
    
    # Bateadores
    hb1 = encontrar_mejor_bateador(batting1)
    hb2 = encontrar_mejor_bateador(batting2)
    
    print(f"\n Top 3 Bateadores {home_team}: OPS Promedio = {hb1['best_bat_OPS']:.3f}" if hb1 else "")
    print(f" Top 3 Bateadores {away_team}: OPS Promedio = {hb2['best_bat_OPS']:.3f}" if hb2 else "")
    
    # Bullpen
    rel_h = extraer_top_relevistas(pitching1)
    rel_a = extraer_top_relevistas(pitching2)
    
    print(f"\n Bullpen {home_team}: WHIP = {rel_h['bullpen_WHIP_mean']:.3f}" if rel_h else "")
    print(f" Bullpen {away_team}: WHIP = {rel_a['bullpen_WHIP_mean']:.3f}" if rel_a else "")
    
    # Construir features V3
    features = {}
    
    features['home_team_OPS'] = stats_h.get('team_OPS_mean', 0.750)
    features['away_team_OPS'] = stats_a.get('team_OPS_mean', 0.750)
    features['diff_team_BA'] = stats_h.get('team_BA_mean', 0.250) - stats_a.get('team_BA_mean', 0.250)
    features['diff_team_OPS'] = features['home_team_OPS'] - features['away_team_OPS']
    features['diff_team_ERA'] = stats_a.get('team_ERA_mean', 4.0) - stats_h.get('team_ERA_mean', 4.0)
    
    if sp1 and sp2:
        features['home_starter_WHIP'] = sp1['WHIP']
        features['away_starter_WHIP'] = sp2['WHIP']
        features['home_starter_ERA'] = sp1['ERA']
        features['away_starter_ERA'] = sp2['ERA']
        features['diff_starter_ERA'] = sp2['ERA'] - sp1['ERA']
        features['diff_starter_WHIP'] = sp2['WHIP'] - sp1['WHIP']
        features['diff_starter_SO9'] = sp1['SO9'] - sp2['SO9']
    else:
        features['home_starter_WHIP'] = stats_h.get('team_WHIP_mean', 1.3)
        features['away_starter_WHIP'] = stats_a.get('team_WHIP_mean', 1.3)
        features['home_starter_ERA'] = stats_h.get('team_ERA_mean', 4.0)
        features['away_starter_ERA'] = stats_a.get('team_ERA_mean', 4.0)
        features['diff_starter_ERA'] = 0
        features['diff_starter_WHIP'] = 0
        features['diff_starter_SO9'] = 0
    
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
    
    features['anchor_pitching_level'] = features['home_starter_ERA']
    features['anchor_offensive_level'] = features['home_team_OPS']
    
    # SUPER FEATURES
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
    
    # Preparar DataFrame con el orden exacto de FEATURES_V3
    features_df = pd.DataFrame([features]).reindex(columns=FEATURES_V3, fill_value=0)
    
    # Predecir usando el formato nativo DMatrix
    dmatrix = xgb.DMatrix(features_df)
    resultado = modelo.predict(dmatrix)
    
    # Extraer probabilidad (XGBoost nativo devuelve un array con la prob. de Home)
    prob_home = float(resultado[0])
    prob_away = 1.0 - prob_home
    
    ganador = home_team if prob_home > 0.5 else away_team
    confianza = max(prob_home, prob_away)
    
    # Mostrar resultado
    print("\n" + "="*70)
    print(" RESULTADO DE LA PREDICCION")
    print("="*70)
    
    ganador_nombre = CODE_TO_FULL_NAME.get(ganador, ganador)
    print(f"\nGANADOR PREDICHO: {ganador_nombre}")
    
    print(f"\nProbabilidades:")
    print(f"   {home_team} (Local):     {prob_home*100:5.1f}%")
    print(f"   {away_team} (Visitante): {prob_away*100:5.1f}%")
    
    print(f"\nNivel de Confianza: {confianza*100:.1f}%")
    
    if confianza > 0.70:
        nivel = "MUY ALTA"
    elif confianza > 0.60:
        nivel = "ALTA"
    elif confianza > 0.55:
        nivel = "MODERADA"
    else:
        nivel = "BAJA (Partido muy parejo)"
    
    print(f"Estatus: {nivel}")
    
    # Factores clave de desempeño con nombres especificos
    print("\nFactores clave de desempeno:")
    
    # Nombres o etiquetas para mayor claridad
    name_h = sp1['nombre'] if sp1 else f"Abridor {home_team}"
    name_a = sp2['nombre'] if sp2 else f"Abridor {away_team}"

    # 1. Neutralizacion (WHIP vs OPS rival)
    neut = features.get('super_neutralizacion_whip_ops', 0)
    if neut < 0:
        print(f"   Neutralizacion: {neut:.4f} (Ventaja {home_team}: {name_h} controla mejor el bateo de {away_team})")
    else:
        print(f"   Neutralizacion: {neut:.4f} (Ventaja {away_team}: {name_a} controla mejor el bateo de {home_team})")

    # 2. Resistencia (ERA vs OPS rival)
    res = features.get('super_resistencia_era_ops', 0)
    if res < 0:
        print(f"   Resistencia:    {res:.4f} (Ventaja {home_team}: {name_h} es mas solido ante el poder de {away_team})")
    else:
        print(f"   Resistencia:    {res:.4f} (Ventaja {away_team}: {name_a} es mas solido ante el poder de {home_team})")

    # 3. Muro Bullpen (Bullpen vs Top Bateadores rivales)
    muro = features.get('super_muro_bullpen', 0)
    if muro < 0:
        print(f"   Muro Bullpen:   {muro:.4f} (Ventaja {home_team}: El bullpen de {home_team} domina a los mejores de {away_team})")
    else:
        print(f"   Muro Bullpen:   {muro:.4f} (Ventaja {away_team}: El bullpen de {away_team} domina a los mejores de {home_team})")
    
    if sp1 is None or sp2 is None:
        print(f"\n    Nota: Lanzador(es) no encontrado(s), usando stats del equipo")
    
    print(f"\n     Datos de temporada {year_usado}")
    print(f"\n" + "="*70)
    
    return {
        'ganador': ganador,
        'prob_home': prob_home,
        'prob_away': prob_away,
        'confianza': confianza,
        'year_usado': year_usado
    }

# ============================================================================
# EJECUCIÓN
# ============================================================================
if __name__ == "__main__":
    
    # 1. Caso de ayuda/lista de equipos
    if len(sys.argv) == 2 and sys.argv[1] in ['--equipos', '-e']:
        mostrar_equipos_disponibles()
        sys.exit()

    # 2. Caso de argumentos directos (se asume 2026 por defecto si no se pasa)
    if len(sys.argv) == 5:
        home_code = validar_y_mostrar_equipo(sys.argv[1], "Local")
        away_code = validar_y_mostrar_equipo(sys.argv[2], "Visitante")
        if home_code and away_code:
            predecir_partido(home_code, away_code, sys.argv[3], sys.argv[4], year=2026)

    # 3. MODO INTERACTIVO (EL QUE TÚ BUSCAS)
    else:
        print("\n" + "="*70)
        print(" MLB PREDICTOR V3 - CONFIGURACION DE ENCUENTRO")
        print("="*70)
        
        try:
            # Validación para Equipo Local
            home_code = None
            while not home_code:
                h_input = input("Equipo Local: ")
                home_code = validar_y_mostrar_equipo(h_input, "Local")
                if not home_code:
                    print("Sugerencia: Escribe el nombre de la ciudad, el equipo o el codigo (ej: NYY).")
                    mostrar_equipos_disponibles()

            # Validación para Equipo Visitante
            away_code = None
            while not away_code:
                a_input = input("Equipo Visitante: ")
                away_code = validar_y_mostrar_equipo(a_input, "Visitante")
                if not away_code:
                    print("Sugerencia: Revisa la lista de codigos arriba.")
                
            # Seleccion de Lanzadores
            p_home = input(f"Lanzador abridor de {home_code}: ")
            p_away = input(f"Lanzador abridor de {away_code}: ")
                 
            print("-" * 30)
            year_input = input("Introduce el año del Encuetro: ")
            
            # Validacion simple del año
            try:
                year_val = int(year_input) if year_input.strip() else 2026
            except ValueError:
                print("Año no valido, usando 2026 por defecto.")
                year_val = 2026
            print("-" * 30)
            
            # Ejecutar prediccion con el año elegido
            predecir_partido(home_code, away_code, p_home, p_away, year=year_val)
            
        except KeyboardInterrupt:
            print("\n\nProceso cancelado.")
            sys.exit()