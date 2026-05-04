import sys
import warnings

from mlb_predict_engine import predecir_juego

warnings.filterwarnings('ignore')

# ============================================================================
# MAPEO Y VALIDACIÓN (Heredado e Intacto)
# ============================================================================
TEAM_MAPPING = {
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

def normalizar_equipo(team_input):
    if not team_input: return None
    team_clean = team_input.strip().lower()
    # 1. BUSQUEDA EXACTA
    for name, code in TEAM_MAPPING.items():
        if team_clean == code.lower() or team_clean == name.lower():
            return code

    # 2. SI NO HAY EXACTA, BUSQUEDA PARCIAL (Solo si es necesario)
    matches = []
    for name, code in TEAM_MAPPING.items():
        if team_clean in name.lower():
            if code not in [m[0] for m in matches]:
                matches.append((code, name))

    if len(matches) == 1:
        return matches[0][0]
    elif len(matches) > 1:
        print(f"\n🤔 '{team_input}' es ambiguo. ¿Te refieres a:")
        for i, (code, name) in enumerate(matches, 1):
            print(f"   {i}. {name} ({code})")
        return None

    return None

def ejecutar_cli_manual():
    print("\n" + "="*70)
    print("⚾ MLB PREDICTOR V3.5 - INTERFAZ DE PRUEBA MANUAL ⚾")
    print("="*70)

    try:
        # 1. Selección de Equipos
        home_code = None
        while not home_code:
            h_input = input("🏠 Equipo Local (Nombre/Código): ")
            home_code = normalizar_equipo(h_input)
            if not home_code: print("❌ No reconozco el equipo.")

        away_code = None
        while not away_code:
            a_input = input("✈️ Equipo Visitante (Nombre/Código): ")
            away_code = normalizar_equipo(a_input)
            if away_code == home_code:
                print("❌ El equipo visitante no puede ser el mismo que el local.")
                away_code = None
            if not away_code: print("❌ No reconozco el equipo.")

        # 3. Lanzadores y Año
        print("-" * 40)
        p_home = input(f"👤 Lanzador abridor de {home_code}: ")
        p_away = input(f"👤 Lanzador abridor de {away_code}: ")

        year_input = input("📅 Año para el scraping de stats (ej: 2024, 2021): ")
        try:
            year_val = int(year_input) if year_input.strip() else 2026
        except ValueError:
            year_val = 2026

        print("\n" + "-" * 40)
        print(f"🚀 Iniciando Predicción: {away_code} vs {home_code} ({year_val})")
        print("🔎 Buscando estadísticas en Baseball-Reference...")

        # 4. Llamada al MOTOR UNIFICADO (Contiene la lógica de scraping y predicción)
        # El motor devuelve un diccionario con el resultado o None si falla
        resultado = predecir_juego(
            home_team=home_code,
            away_team=away_code,
            home_pitcher=p_home,
            away_pitcher=p_away,
            year=year_val,
            modo_auto=False
        )

        if resultado:
            # --- SECCIÓN DE REPORTE VISUAL ROBUSTO ---
            # Nota: predecir_juego ya hizo los prints básicos,
            # pero aquí podemos confirmar que se guardó.
            print("\n✅ Predicción completada con éxito.")
            print("💾 Guardado en DB 'predicciones_historico' con tipo 'MANUAL'")
        else:
            # El error 'Name' ya no ocurrirá porque predecir_juego usa
            # las funciones robustas que definimos antes.
            print("\n❌ Error: No se pudo completar la predicción.")

    except KeyboardInterrupt:
        print("\n\nSaliendo del predictor...")
        sys.exit()

if __name__ == "__main__":
    ejecutar_cli_manual()
