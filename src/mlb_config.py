"""
Configuración Centralizada para el Sistema MLB Predictor V3.5
Evita duplicación de código y facilita mantenimiento
"""

import os

# ============================================================================
# RUTAS DE ARCHIVOS
# ============================================================================
# Detectar si estamos en subcarpeta src/ o en la raíz del proyecto
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Si estamos en src/, subir un nivel para llegar a la raíz del proyecto
if os.path.basename(CURRENT_DIR) == 'src':
    BASE_DIR = os.path.dirname(CURRENT_DIR)
else:
    BASE_DIR = CURRENT_DIR

MODELO_PATH = os.path.join(BASE_DIR, 'models', 'modelo_mlb_v3.5.json')
MODELO_BACKUP = os.path.join(BASE_DIR, 'models', 'modelo_mlb_v3.5_backup.json')
DB_PATH = os.path.join(BASE_DIR, 'data', 'mlb_reentrenamiento.db')
CACHE_PATH = os.path.join(BASE_DIR, 'cache', 'features_hibridas_v3.5_cache.pkl')

# ============================================================================
# MAPEO UNIFICADO DE EQUIPOS
# ============================================================================
TEAM_CODE_TO_NAME = {
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

# Crear mapeo inverso automáticamente
TEAM_NAME_TO_CODE = {}

# Nombres completos
for code, full_name in TEAM_CODE_TO_NAME.items():
    TEAM_NAME_TO_CODE[full_name] = code
    TEAM_NAME_TO_CODE[full_name.lower()] = code
    
    # Nombre corto (última palabra)
    short_name = full_name.split()[-1]
    if short_name not in TEAM_NAME_TO_CODE:
        TEAM_NAME_TO_CODE[short_name] = code
        TEAM_NAME_TO_CODE[short_name.lower()] = code
    
    # Códigos (ya normalizados)
    TEAM_NAME_TO_CODE[code] = code
    TEAM_NAME_TO_CODE[code.lower()] = code

# Casos especiales
TEAM_NAME_TO_CODE.update({
    "d'backs": 'ARI',
    "diamondbacks": 'ARI',
    "arizona d'backs": 'ARI',
    "guardians": 'CLE',
    "white sox": 'CHW',
    "red sox": 'BOS',
    "blue jays": 'TOR',
})

# ============================================================================
# CONFIGURACIÓN DE SCRAPING
# ============================================================================
SCRAPING_CONFIG = {
    'max_retries': 3,
    'timeout': 15,
    'min_delay': 2,
    'max_delay': 4,
    'rate_limit_wait': 10,
    'bloque_size': 150,
    'pausa_entre_bloques': 45
}

# ============================================================================
# CONFIGURACIÓN DE MODELO
# ============================================================================
MODEL_CONFIG = {
    'test_size': 0.20,
    'random_state': 42,
    'cv_folds': 3,
    'param_grid': {
        'n_estimators': [200, 300, 400],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.03, 0.05],
        'gamma': [0.1, 0.2]
    }
}

# ============================================================================
# FEATURES ESPERADAS (Para validación)
# ============================================================================
TEMPORAL_FEATURES = [
    'home_win_rate_10', 'home_racha', 'home_runs_avg', 'home_runs_diff',
    'away_win_rate_10', 'away_racha', 'away_runs_avg', 'away_runs_diff',
    'year'
]

SCRAPING_FEATURES = [
    'home_team_OPS', 'away_team_OPS', 'diff_team_BA', 'diff_team_OPS', 'diff_team_ERA',
    'home_starter_WHIP', 'away_starter_WHIP', 'home_starter_ERA', 'away_starter_ERA',
    'home_starter_SO9', 'away_starter_SO9',
    'diff_starter_ERA', 'diff_starter_WHIP', 'diff_starter_SO9',
    'home_best_OPS', 'away_best_OPS', 'diff_best_BA', 'diff_best_OPS', 'diff_best_HR',
    'home_bullpen_ERA', 'away_bullpen_ERA', 'home_bullpen_WHIP', 'away_bullpen_WHIP',
    'diff_bullpen_ERA', 'diff_bullpen_WHIP',
    'anchor_pitching_level', 'anchor_offensive_level'
]

SUPER_FEATURES = [
    'super_neutralizacion_whip_ops',
    'super_resistencia_era_ops',
    'super_muro_bullpen'
]

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================
def get_team_code(team_input):
    """
    Convierte cualquier variación de nombre de equipo a código MLB
    
    Args:
        team_input: Nombre, código o apodo del equipo
        
    Returns:
        Código de 3 letras o None si no se encuentra
    """
    if not team_input:
        return None
    
    team_clean = str(team_input).strip().lower()
    return TEAM_NAME_TO_CODE.get(team_clean)


def get_team_name(team_code):
    """
    Convierte código MLB a nombre completo
    
    Args:
        team_code: Código de 3 letras (ej: 'NYY')
        
    Returns:
        Nombre completo o el código si no se encuentra
    """
    if not team_code:
        return None
    
    return TEAM_CODE_TO_NAME.get(team_code.upper(), team_code)


def ensure_directories():
    """Crea directorios necesarios si no existen"""
    dirs = ['models', 'data', 'cache']
    for d in dirs:
        dir_path = os.path.join(BASE_DIR, d)
        os.makedirs(dir_path, exist_ok=True)
        
    # Verificar que los directorios se crearon correctamente
    for d in dirs:
        dir_path = os.path.join(BASE_DIR, d)
        if not os.path.exists(dir_path):
            print(f"⚠️ No se pudo crear el directorio: {dir_path}")


# ============================================================================
# VALIDACIÓN DE CONFIGURACIÓN
# ============================================================================
def validate_config():
    """Valida que todos los paths y configuraciones sean correctos"""
    errors = []
    
    # Verificar directorios
    ensure_directories()
    
    # Verificar que hay exactamente 30 equipos
    if len(TEAM_CODE_TO_NAME) != 30:
        errors.append(f"Se esperan 30 equipos MLB, se encontraron {len(TEAM_CODE_TO_NAME)}")
    
    # Verificar coherencia de mapeos
    if len(TEAM_CODE_TO_NAME) != len(set(TEAM_CODE_TO_NAME.values())):
        errors.append("Hay nombres de equipos duplicados")
    
    return errors


if __name__ == "__main__":
    # Auto-validación al ejecutar
    print(f"📁 Directorio base detectado: {BASE_DIR}")
    print(f"📁 Directorio actual: {CURRENT_DIR}")
    print()
    
    errors = validate_config()
    if errors:
        print("❌ Errores de configuración:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("✅ Configuración validada correctamente")
        print(f"📁 Modelo: {MODELO_PATH}")
        print(f"   Existe: {'✅ SÍ' if os.path.exists(MODELO_PATH) else '❌ NO'}")
        print(f"📁 DB: {DB_PATH}")
        print(f"   Existe: {'✅ SÍ' if os.path.exists(DB_PATH) else '❌ NO'}")
        print(f"📁 Cache: {CACHE_PATH}")
        print(f"   Existe: {'✅ SÍ' if os.path.exists(CACHE_PATH) else '⚠️ NO (se creará al entrenar)'}")
        print(f"📊 Equipos configurados: {len(TEAM_CODE_TO_NAME)}")