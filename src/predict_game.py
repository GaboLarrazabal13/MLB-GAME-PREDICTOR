"""
Script de Predicción de Partidos MLB
Usa el modelo entrenado para predecir nuevos partidos
"""

# ============================================================================
# MAPEO DE EQUIPOS MLB
# ============================================================================

TEAM_MAPPING = {
    # Nombres completos
    'Arizona Diamondbacks': 'ARI',
    'Atlanta Braves': 'ATL',
    'Baltimore Orioles': 'BAL',
    'Boston Red Sox': 'BOS',
    'Chicago Cubs': 'CHC',
    'Chicago White Sox': 'CHW',
    'Cincinnati Reds': 'CIN',
    'Cleveland Guardians': 'CLE',
    'Colorado Rockies': 'COL',
    'Detroit Tigers': 'DET',
    'Houston Astros': 'HOU',
    'Kansas City Royals': 'KCR',
    'Los Angeles Angels': 'LAA',
    'Los Angeles Dodgers': 'LAD',
    'Miami Marlins': 'MIA',
    'Milwaukee Brewers': 'MIL',
    'Minnesota Twins': 'MIN',
    'New York Mets': 'NYM',
    'New York Yankees': 'NYY',
    'Oakland Athletics': 'OAK',
    'Philadelphia Phillies': 'PHI',
    'Pittsburgh Pirates': 'PIT',
    'San Diego Padres': 'SDP',
    'Seattle Mariners': 'SEA',
    'San Francisco Giants': 'SFG',
    'St. Louis Cardinals': 'STL',
    'Tampa Bay Rays': 'TBR',
    'Texas Rangers': 'TEX',
    'Toronto Blue Jays': 'TOR',
    'Washington Nationals': 'WSN',
    
    # Nombres cortos (sin ciudad)
    'Diamondbacks': 'ARI',
    'Braves': 'ATL',
    'Orioles': 'BAL',
    'Red Sox': 'BOS',
    'Cubs': 'CHC',
    'White Sox': 'CHW',
    'Reds': 'CIN',
    'Guardians': 'CLE',
    'Rockies': 'COL',
    'Tigers': 'DET',
    'Astros': 'HOU',
    'Royals': 'KCR',
    'Angels': 'LAA',
    'Dodgers': 'LAD',
    'Marlins': 'MIA',
    'Brewers': 'MIL',
    'Twins': 'MIN',
    'Mets': 'NYM',
    'Yankees': 'NYY',
    'Athletics': 'OAK',
    'Phillies': 'PHI',
    'Pirates': 'PIT',
    'Padres': 'SDP',
    'Mariners': 'SEA',
    'Giants': 'SFG',
    'Cardinals': 'STL',
    'Rays': 'TBR',
    'Rangers': 'TEX',
    'Blue Jays': 'TOR',
    'Nationals': 'WSN',
    
    # Códigos (ya en mayúsculas)
    'ARI': 'ARI',
    'ATL': 'ATL',
    'BAL': 'BAL',
    'BOS': 'BOS',
    'CHC': 'CHC',
    'CHW': 'CHW',
    'CIN': 'CIN',
    'CLE': 'CLE',
    'COL': 'COL',
    'DET': 'DET',
    'HOU': 'HOU',
    'KCR': 'KCR',
    'LAA': 'LAA',
    'LAD': 'LAD',
    'MIA': 'MIA',
    'MIL': 'MIL',
    'MIN': 'MIN',
    'NYM': 'NYM',
    'NYY': 'NYY',
    'OAK': 'OAK',
    'PHI': 'PHI',
    'PIT': 'PIT',
    'SDP': 'SDP',
    'SEA': 'SEA',
    'SFG': 'SFG',
    'STL': 'STL',
    'TBR': 'TBR',
    'TEX': 'TEX',
    'TOR': 'TOR',
    'WSN': 'WSN'
}

# Mapeo inverso (código a nombre completo)
CODE_TO_FULL_NAME = {
    'ARI': 'Arizona Diamondbacks',
    'ATL': 'Atlanta Braves',
    'BAL': 'Baltimore Orioles',
    'BOS': 'Boston Red Sox',
    'CHC': 'Chicago Cubs',
    'CHW': 'Chicago White Sox',
    'CIN': 'Cincinnati Reds',
    'CLE': 'Cleveland Guardians',
    'COL': 'Colorado Rockies',
    'DET': 'Detroit Tigers',
    'HOU': 'Houston Astros',
    'KCR': 'Kansas City Royals',
    'LAA': 'Los Angeles Angels',
    'LAD': 'Los Angeles Dodgers',
    'MIA': 'Miami Marlins',
    'MIL': 'Milwaukee Brewers',
    'MIN': 'Minnesota Twins',
    'NYM': 'New York Mets',
    'NYY': 'New York Yankees',
    'OAK': 'Oakland Athletics',
    'PHI': 'Philadelphia Phillies',
    'PIT': 'Pittsburgh Pirates',
    'SDP': 'San Diego Padres',
    'SEA': 'Seattle Mariners',
    'SFG': 'San Francisco Giants',
    'STL': 'St. Louis Cardinals',
    'TBR': 'Tampa Bay Rays',
    'TEX': 'Texas Rangers',
    'TOR': 'Toronto Blue Jays',
    'WSN': 'Washington Nationals'
}


def normalizar_equipo(team_input):
    """
    Normaliza el nombre del equipo a su código de 3 letras
    Acepta: código, nombre completo, o nombre corto
    
    Args:
        team_input: String con el nombre o código del equipo
    
    Returns:
        tuple: (código_normalizado, nombre_completo) o (None, None) si no se encuentra
    """
    if not team_input:
        return None, None
    
    # Limpiar y normalizar input
    team_clean = team_input.strip()
    
    # Buscar en el mapeo (case-insensitive)
    for key, code in TEAM_MAPPING.items():
        if key.lower() == team_clean.lower():
            full_name = CODE_TO_FULL_NAME.get(code, code)
            return code, full_name
    
    # Si no se encuentra exacto, buscar coincidencia parcial
    team_lower = team_clean.lower()
    matches = []
    
    for key, code in TEAM_MAPPING.items():
        if team_lower in key.lower() or key.lower() in team_lower:
            full_name = CODE_TO_FULL_NAME.get(code, code)
            matches.append((code, full_name, key))
    
    # Si hay exactamente una coincidencia, usarla
    if len(matches) == 1:
        return matches[0][0], matches[0][1]
    
    # Si hay múltiples coincidencias, mostrar opciones
    if len(matches) > 1:
        print(f"\n⚠️  '{team_input}' es ambiguo. ¿Te refieres a alguno de estos?")
        for i, (code, full_name, key) in enumerate(matches, 1):
            print(f"   {i}. {full_name} ({code})")
        return None, None
    
    # No se encontró
    return None, None


def mostrar_equipos_disponibles():
    """Muestra todos los equipos disponibles con sus códigos"""
    print("\n📋 EQUIPOS MLB DISPONIBLES:")
    print("="*70)
    
    # Agrupar por división (simplificado por orden alfabético)
    for code in sorted(CODE_TO_FULL_NAME.keys()):
        full_name = CODE_TO_FULL_NAME[code]
        print(f"   {code:4s} = {full_name}")
    
    print("="*70)
    print("\n💡 Puedes usar el código (ej: NYY) o el nombre completo (ej: New York Yankees)")
    print("   También funciona con nombres cortos (ej: Yankees)")


def sugerir_equipo(team_input):
    """
    Sugiere equipos similares si no se encuentra una coincidencia exacta
    """
    if not team_input:
        return []
    
    team_lower = team_input.lower()
    suggestions = []
    
    # Buscar coincidencias parciales
    for key, code in TEAM_MAPPING.items():
        if team_lower in key.lower():
            full_name = CODE_TO_FULL_NAME.get(code, code)
            suggestions.append((code, full_name))
    
    # Eliminar duplicados manteniendo el orden
    seen = set()
    unique_suggestions = []
    for code, full_name in suggestions:
        if code not in seen:
            seen.add(code)
            unique_suggestions.append((code, full_name))
    
    return unique_suggestions


def validar_y_mostrar_equipo(team_input, role="Equipo"):
    """
    Valida el input del equipo y muestra información
    
    Args:
        team_input: Input del usuario
        role: "Local" o "Visitante" para el mensaje
    
    Returns:
        str: Código del equipo o None si no es válido
    """
    code, full_name = normalizar_equipo(team_input)
    
    if code:
        print(f"   ✅ {role}: {full_name} ({code})")
        return code
    else:
        print(f"   ❌ No se reconoce '{team_input}'")
        
        # Buscar sugerencias
        suggestions = sugerir_equipo(team_input)
        
        if suggestions:
            print(f"\n   💡 ¿Quisiste decir alguno de estos?")
            for code, full_name in suggestions[:5]:  # Mostrar máximo 5
                print(f"      • {full_name} ({code})")
        else:
            print(f"\n   💡 Usa el comando con --equipos para ver todos los equipos disponibles")
        
        return None
import pickle
import sys
import pandas as pd
import numpy as np
import cloudscraper
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')


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
        r'Team Totals|Rank in|^\s*$', 
        case=False, na=False, regex=True
    )]
    return df.reset_index(drop=True)


def scrape_player_stats(team_code, year=2025):
    """Extrae estadísticas de jugadores de un equipo"""
    url = f"https://www.baseball-reference.com/teams/{team_code}/{year}.shtml"
    
    html = obtener_html(url)
    if not html:
        print(f"❌ No se pudo obtener el HTML de {team_code} para {year}")
        return None, None
    
    soup = BeautifulSoup(html, 'html.parser')
    
    batting_table = soup.find('table', {'id': 'players_standard_batting'})
    if not batting_table:
        print(f"⚠️ Tabla de bateo no encontrada para {team_code}. Revisa si la tabla está comentada en el HTML.")
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
    """Busca un lanzador y extrae sus stats: ERA, WHIP, H9"""
    if pitching_df is None or len(pitching_df) == 0:
        return None
    
    nombre_busqueda = nombre_lanzador.lower().strip()
    name_col = pitching_df.columns[0]
    
    mask = pitching_df[name_col].astype(str).str.lower().str.contains(nombre_busqueda, na=False)
    
    if mask.sum() == 0:
        print(f"  ⚠️  Lanzador '{nombre_lanzador}' no encontrado")
        return None
    
    lanzador = pitching_df[mask].iloc[0]
    nombre_completo = lanzador[name_col]
    
    stats = {
        'nombre': nombre_completo,
        'ERA': safe_float(lanzador.get('ERA', 0)),
        'WHIP': safe_float(lanzador.get('WHIP', 0)),
        'H9': safe_float(lanzador.get('H9', 0)),
        'W': safe_float(lanzador.get('W', 0)),
        'L': safe_float(lanzador.get('L', 0)),
        'IP': safe_float(lanzador.get('IP', 0))
    }
    
    print(f"  ✓ {nombre_completo}")
    print(f"    ERA: {stats['ERA']:.2f} | WHIP: {stats['WHIP']:.2f} | H9: {stats['H9']:.1f} | W-L: {stats['W']}-{stats['L']}")
    
    return stats


def encontrar_mejor_bateador(batting_df):
    """
    Encuentra los 3 mejores bateadores según OBP
    Filtro: deben tener más turnos al bate (AB) que la mediana del equipo
    Stats: BA, OBP, RBI, R
    """
    if batting_df is None or len(batting_df) == 0:
        return None
    
    name_col = batting_df.columns[0]
    
    # Verificar que existan las columnas necesarias
    if 'OBP' not in batting_df.columns or 'AB' not in batting_df.columns:
        return None
    
    # Convertir a numérico
    batting_df['OBP'] = pd.to_numeric(batting_df['OBP'], errors='coerce')
    batting_df['AB'] = pd.to_numeric(batting_df['AB'], errors='coerce')
    batting_df = batting_df.dropna(subset=['OBP', 'AB'])
    
    if len(batting_df) == 0:
        return None
    
    # Calcular mediana de turnos al bate del equipo
    mediana_ab = batting_df['AB'].median()
    
    # Filtrar: solo jugadores con AB > mediana
    batting_filtrado = batting_df[batting_df['AB'] > mediana_ab].copy()
    
    if len(batting_filtrado) == 0:
        print(f"  ⚠️  Ningún bateador supera la mediana de AB ({mediana_ab:.0f})")
        batting_filtrado = batting_df  # Usar todos si no hay suficientes
    
    # Ordenar por OBP y tomar top 3
    batting_filtrado = batting_filtrado.sort_values('OBP', ascending=False)
    top_3 = batting_filtrado.head(3)
    
    print(f"   Mediana de AB del equipo: {mediana_ab:.0f}")
    print(f"   Top 3 Bateadores (con AB > {mediana_ab:.0f}):")
    
    # Calcular promedios de los top 3
    stats_promedio = {
        'BA': 0,
        'OBP': 0,
        'RBI': 0,
        'R': 0
    }
    
    count = 0
    for idx, bateador in top_3.iterrows():
        nombre = bateador[name_col]
        ba = safe_float(bateador.get('BA', 0))
        obp = safe_float(bateador.get('OBP', 0))
        rbi = safe_float(bateador.get('RBI', 0))
        r = safe_float(bateador.get('R', 0))
        ab = safe_float(bateador.get('AB', 0))
        
        print(f"     {count+1}. {nombre}")
        print(f"        BA: {ba:.3f} | OBP: {obp:.3f} | RBI: {rbi} | R: {r} | AB: {ab:.0f}")
        
        stats_promedio['BA'] += ba
        stats_promedio['OBP'] += obp
        stats_promedio['RBI'] += rbi
        stats_promedio['R'] += r
        count += 1
    
    # Promediar
    if count > 0:
        for key in stats_promedio:
            stats_promedio[key] /= count
    
    print(f"  📈 Promedio Top 3: BA={stats_promedio['BA']:.3f}, OBP={stats_promedio['OBP']:.3f}")
    
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
# FUNCIÓN DE PREDICCIÓN
# ============================================================================

def predecir_partido(home_team, away_team, home_pitcher, away_pitcher, year=2026):
    """
    Predice el resultado de un partido
    
    Args:
        home_team: Código del equipo local (ej: 'BOS')
        away_team: Código del equipo visitante (ej: 'NYY')
        home_pitcher: Nombre del lanzador del equipo local (busca por coincidencia)
        away_pitcher: Nombre del lanzador del equipo visitante (busca por coincidencia)
        year: Año de la temporada (default 2026 para predicciones futuras)
    """
    
    print("\n" + "="*70)
    print(" PREDICTOR DE PARTIDOS MLB")
    print("="*70)
    
    # Obtener nombres completos
    home_full = CODE_TO_FULL_NAME.get(home_team, home_team)
    away_full = CODE_TO_FULL_NAME.get(away_team, away_team)
    
    print(f"\n🏟️  {home_full} ({home_team}) vs {away_full} ({away_team})")
    print(f"📅 Temporada: {year}")
    
    # Cargar modelo
    try:
        print(f"\n📦 Cargando modelo entrenado...")
        with open('./models/mlb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('./models/mlb_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('./models/mlb_feature_names.pkl', 'rb') as f:
            expected_features = pickle.load(f)
        with open('./models/mlb_model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        print(f"   ✅ Modelo: {model_info['nombre']}")
        print(f"   ✅ Accuracy: {model_info['accuracy']*100:.2f}%")
        print(f"   ✅ Features: {model_info['n_features']}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: No se encontró el modelo entrenado.")
        print(f"   Primero ejecuta: python train_model.py")
        return None
    
    # Extraer datos actualizados - INTENTAR MÚLTIPLES AÑOS
    print(f"\n" + "="*70)
    print(f" EXTRAYENDO DATOS ACTUALIZADOS")
    print("="*70)
    
    # Intentar año especificado primero, luego el anterior
    años_a_probar = [year, year-1, 2025]
    batting1 = batting2 = pitching1 = pitching2 = None
    year_usado = None
    
    for año_intento in años_a_probar:
        print(f"\n📡 Intentando obtener datos de temporada {año_intento}...")
        
        print(f"   Equipo Local: {home_team}")
        batting1, pitching1 = scrape_player_stats(home_team, año_intento)
        
        print(f"   Equipo Visitante: {away_team}")
        batting2, pitching2 = scrape_player_stats(away_team, año_intento)
        
        if batting1 is not None and batting2 is not None and pitching1 is not None and pitching2 is not None:
            year_usado = año_intento
            print(f"\n   ✅ Datos obtenidos exitosamente de temporada {año_intento}")
            break
        else:
            print(f"   ⚠️  Datos incompletos para temporada {año_intento}")
    
    if batting1 is None or batting2 is None or pitching1 is None or pitching2 is None:
        print("\n❌ Error: No se pudieron obtener datos de ninguna temporada")
        print("   Verifica los códigos de los equipos y que tengan datos disponibles.")
        return None
    
    # Stats del equipo
    stats_team1 = calcular_stats_equipo(batting1, pitching1)
    stats_team2 = calcular_stats_equipo(batting2, pitching2)
    
    # Mostrar stats del equipo
    print(f"\n Stats del Equipo {home_team} (temporada {year_usado}):")
    print(f"   BA: {stats_team1.get('team_BA_mean', 0):.3f} | OBP: {stats_team1.get('team_OBP_mean', 0):.3f}")
    print(f"   ERA: {stats_team1.get('team_ERA_mean', 0):.2f} | WHIP: {stats_team1.get('team_WHIP_mean', 0):.2f}")
    
    print(f"\n Stats del Equipo {away_team} (temporada {year_usado}):")
    print(f"   BA: {stats_team2.get('team_BA_mean', 0):.3f} | OBP: {stats_team2.get('team_OBP_mean', 0):.3f}")
    print(f"   ERA: {stats_team2.get('team_ERA_mean', 0):.2f} | WHIP: {stats_team2.get('team_WHIP_mean', 0):.2f}")
    
    # Lanzadores iniciales - BUSCAR EN LA TABLA ACTUAL
    print(f"\n Buscando Lanzador Inicial {home_team}: '{home_pitcher}'")
    pitcher1_stats = encontrar_lanzador(pitching1, home_pitcher)
    
    if pitcher1_stats is None:
        print(f"   ⚠️  No se encontró '{home_pitcher}' en el roster de {home_team}")
        print(f"   📋 Lanzadores disponibles en {home_team}:")
        if pitching1 is not None and len(pitching1) > 0:
            name_col = pitching1.columns[0]
            for idx, row in pitching1.head(10).iterrows():
                print(f"      - {row[name_col]}")
        print(f"   ℹ️  Usando stats promedio del equipo como fallback")
    
    print(f"\n Buscando Lanzador Inicial {away_team}: '{away_pitcher}'")
    pitcher2_stats = encontrar_lanzador(pitching2, away_pitcher)
    
    if pitcher2_stats is None:
        print(f"   ⚠️  No se encontró '{away_pitcher}' en el roster de {away_team}")
        print(f"   📋 Lanzadores disponibles en {away_team}:")
        if pitching2 is not None and len(pitching2) > 0:
            name_col = pitching2.columns[0]
            for idx, row in pitching2.head(10).iterrows():
                print(f"      - {row[name_col]}")
        print(f"   ℹ️  Usando stats promedio del equipo como fallback")
    
    # Top 3 bateadores (con filtro de mediana de AB)
    print(f"\n Top 3 Bateadores {home_team}:")
    best_batters1 = encontrar_mejor_bateador(batting1)
    
    print(f"\n Top 3 Bateadores {away_team}:")
    best_batters2 = encontrar_mejor_bateador(batting2)
    
    # Crear vector de features
    features = {}
    
    for key, val in stats_team1.items():
        features[f'home_{key}'] = val
    
    for key, val in stats_team2.items():
        features[f'away_{key}'] = val
    
    # Si no se encontró el lanzador, usar promedio del equipo
    if pitcher1_stats:
        features['home_pitcher_ERA'] = pitcher1_stats['ERA']
        features['home_pitcher_WHIP'] = pitcher1_stats['WHIP']
        features['home_pitcher_H9'] = pitcher1_stats['H9']
        features['home_pitcher_W'] = pitcher1_stats['W']
        features['home_pitcher_L'] = pitcher1_stats['L']
    else:
        # Usar stats promedio del equipo como fallback
        features['home_pitcher_ERA'] = stats_team1.get('team_ERA_mean', 4.0)
        features['home_pitcher_WHIP'] = stats_team1.get('team_WHIP_mean', 1.3)
        features['home_pitcher_H9'] = stats_team1.get('team_H9_mean', 9.0)
        features['home_pitcher_W'] = 0
        features['home_pitcher_L'] = 0
    
    if pitcher2_stats:
        features['away_pitcher_ERA'] = pitcher2_stats['ERA']
        features['away_pitcher_WHIP'] = pitcher2_stats['WHIP']
        features['away_pitcher_H9'] = pitcher2_stats['H9']
        features['away_pitcher_W'] = pitcher2_stats['W']
        features['away_pitcher_L'] = pitcher2_stats['L']
    else:
        # Usar stats promedio del equipo como fallback
        features['away_pitcher_ERA'] = stats_team2.get('team_ERA_mean', 4.0)
        features['away_pitcher_WHIP'] = stats_team2.get('team_WHIP_mean', 1.3)
        features['away_pitcher_H9'] = stats_team2.get('team_H9_mean', 9.0)
        features['away_pitcher_W'] = 0
        features['away_pitcher_L'] = 0
    
    if best_batters1:
        features['home_best_BA'] = best_batters1['BA']
        features['home_best_OBP'] = best_batters1['OBP']
        features['home_best_RBI'] = best_batters1['RBI']
        features['home_best_R'] = best_batters1['R']
    else:
        features.update({
            'home_best_BA': stats_team1.get('team_BA_mean', 0.250),
            'home_best_OBP': stats_team1.get('team_OBP_mean', 0.320),
            'home_best_RBI': 0,
            'home_best_R': 0
        })
    
    if best_batters2:
        features['away_best_BA'] = best_batters2['BA']
        features['away_best_OBP'] = best_batters2['OBP']
        features['away_best_RBI'] = best_batters2['RBI']
        features['away_best_R'] = best_batters2['R']
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
    
    # Preparar para predicción
    features_df = pd.DataFrame([features])
    
    for col in expected_features:
        if col not in features_df.columns:
            features_df[col] = 0
    
    features_df = features_df[expected_features]
    features_df = features_df.fillna(0)
    
    # Predecir
    features_scaled = scaler.transform(features_df)
    prediccion = model.predict(features_scaled)[0]
    probabilidades = model.predict_proba(features_scaled)[0]
    
    # Mostrar resultado
    print(f"\n" + "="*70)
    print(" 🎯 PREDICCIÓN")
    print("="*70)
    
    ganador = home_team if prediccion == 1 else away_team
    prob_home = probabilidades[1]
    prob_away = probabilidades[0]
    confianza = max(prob_home, prob_away)
    
    print(f"\n🏆 GANADOR PREDICHO: {ganador}")
    print(f"\n📊 Probabilidades:")
    print(f"   {home_team} (Local):     {prob_home*100:5.1f}%  {'█' * int(prob_home*50)}")
    print(f"   {away_team} (Visitante): {prob_away*100:5.1f}%  {'█' * int(prob_away*50)}")
    
    print(f"\n Nivel de Confianza: {confianza*100:.1f}%")
    
    if confianza > 0.70:
        nivel = "MUY ALTA ✅✅✅"
    elif confianza > 0.60:
        nivel = "ALTA ✅✅"
    elif confianza > 0.55:
        nivel = "MODERADA ⚠️"
    else:
        nivel = "BAJA ❌ (Partido muy parejo)"
    
    print(f"   {nivel}")
    
    # Factores clave
    print(f"\n🔑 Factores Clave:")
    
    if features['pitcher_ERA_diff'] < -0.5:
        print(f"   ✅ Ventaja importante en ERA del lanzador local (diff: {features['pitcher_ERA_diff']:.2f})")
    elif features['pitcher_ERA_diff'] > 0.5:
        print(f"   ✅ Ventaja importante en ERA del lanzador visitante (diff: {features['pitcher_ERA_diff']:.2f})")
    
    if features['team_BA_diff'] > 0.020:
        print(f"   ✅ Mejor bateo del equipo local (+{features['team_BA_diff']:.3f})")
    elif features['team_BA_diff'] < -0.020:
        print(f"   ✅ Mejor bateo del equipo visitante ({features['team_BA_diff']:.3f})")
    
    if pitcher1_stats is None or pitcher2_stats is None:
        print(f"\n   ⚠️  Nota: Se usaron stats promedio del equipo para lanzador(es) no encontrado(s)")
    
    print(f"\n   ℹ️  Datos basados en temporada {year_usado}")
    
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
    
    # Modo CLI
    if len(sys.argv) == 5:
        home = sys.argv[1]
        away = sys.argv[2]
        pitcher_home = sys.argv[3]
        pitcher_away = sys.argv[4]
        
        print("\n" + "="*70)
        print(" VALIDANDO EQUIPOS")
        print("="*70)
        
        # Validar equipos
        home_code = validar_y_mostrar_equipo(home, "Local")
        away_code = validar_y_mostrar_equipo(away, "Visitante")
        
        if home_code and away_code:
            predecir_partido(home_code, away_code, pitcher_home, pitcher_away)
        else:
            print("\n❌ Por favor, verifica los nombres de los equipos e intenta de nuevo")
            print("   Usa: python predict_game.py --equipos  para ver la lista completa")
    
    # Mostrar lista de equipos
    elif len(sys.argv) == 2 and sys.argv[1] in ['--equipos', '--teams', '-e', '--help-teams']:
        mostrar_equipos_disponibles()
    
    # Ayuda
    elif len(sys.argv) == 2 and sys.argv[1] in ['--help', '-h']:
        print("\n" + "="*70)
        print(" PREDICTOR DE PARTIDOS MLB - Ayuda")
        print("="*70)
        print("\n📖 USO:")
        print("\n1. Modo línea de comandos:")
        print("   python predict_game.py <equipo_local> <equipo_visitante> <lanzador_local> <lanzador_visitante>")
        print("\n   Ejemplos:")
        print("   python predict_game.py NYY BOS Cole Bello")
        print("   python predict_game.py \"New York Yankees\" \"Boston Red Sox\" Cole Bello")
        print("   python predict_game.py Yankees \"Red Sox\" Cole Bello")
        print("\n2. Modo interactivo:")
        print("   python predict_game.py")
        print("\n3. Ver lista de equipos:")
        print("   python predict_game.py --equipos")
        print("\n" + "="*70)
        print("\n💡 FORMATO DE EQUIPOS:")
        print("   • Código de 3 letras: NYY, BOS, LAD, etc.")
        print("   • Nombre completo: New York Yankees, Boston Red Sox, etc.")
        print("   • Nombre corto: Yankees, Red Sox, Dodgers, etc.")
        print("\n   Ejemplos válidos:")
        print("   • NYY = New York Yankees = Yankees")
        print("   • BOS = Boston Red Sox = Red Sox")
        print("   • LAD = Los Angeles Dodgers = Dodgers")
        print("   • CHC = Chicago Cubs = Cubs")
        print("\n" + "="*70)
    
    # Modo interactivo
    else:
        if len(sys.argv) > 1:
            print(f"\n❌ Argumentos inválidos")
            print("   Usa: python predict_game.py --help  para ver la ayuda")
            sys.exit(1)
        
        print("\n" + "="*70)
        print(" PREDICTOR DE PARTIDOS MLB - Modo Interactivo")
        print("="*70)
        
        print("\n💡 Puedes usar códigos (NYY) o nombres completos (New York Yankees)")
        print("   Escribe '--equipos' para ver todos los equipos disponibles\n")
        
        # Input con validación de equipo local
        while True:
            home_input = input("  Equipo Local: ").strip()
            
            if home_input.lower() == '--equipos':
                mostrar_equipos_disponibles()
                continue
            
            home_code = validar_y_mostrar_equipo(home_input, "Local")
            if home_code:
                break
            print("   ❌ Inténtalo de nuevo\n")
        
        # Input con validación de equipo visitante
        while True:
            away_input = input("\n  Equipo Visitante: ").strip()
            
            if away_input.lower() == '--equipos':
                mostrar_equipos_disponibles()
                continue
            
            away_code = validar_y_mostrar_equipo(away_input, "Visitante")
            if away_code:
                break
            print("   ❌ Inténtalo de nuevo\n")
        
        # Lanzadores (sin validación especial por ahora)
        pitcher_home = input("\n  Lanzador Local (apellido o nombre): ").strip()
        pitcher_away = input("  Lanzador Visitante (apellido o nombre): ").strip()
        
        year_input = input("\n  Temporada (default 2026): ").strip()
        year = int(year_input) if year_input else 2026
        
        predecir_partido(home_code, away_code, pitcher_home, pitcher_away, year)