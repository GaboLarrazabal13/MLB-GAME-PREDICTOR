"""
Transformador de CSV de partidos MLB
Identifica a qué equipo pertenece cada lanzador y estructura el CSV para el modelo
"""

import pandas as pd
import cloudscraper
from bs4 import BeautifulSoup
import time
import pickle
from collections import defaultdict

# ============================================================================
# FUNCIONES DE SCRAPING
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
    except Exception as e:
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
    df = df.reset_index(drop=True)
    return df


def obtener_roster_equipo(team_code, year=2025):
    """
    Obtiene la lista de lanzadores de un equipo
    Returns: lista de nombres de lanzadores
    """
    url = f"https://www.baseball-reference.com/teams/{team_code}/{year}.shtml"
    
    html = obtener_html(url)
    if not html:
        return []
    
    soup = BeautifulSoup(html, 'html.parser')
    pitching_table = soup.find('table', {'id': 'players_standard_pitching'})
    
    if not pitching_table:
        return []
    
    try:
        pitching_df = pd.read_html(str(pitching_table))[0]
        pitching_df = limpiar_dataframe(pitching_df)
        
        if len(pitching_df) > 0:
            name_col = pitching_df.columns[0]
            # Obtener lista de nombres, normalizar
            nombres = pitching_df[name_col].astype(str).str.strip().tolist()
            return nombres
    except:
        pass
    
    return []


def crear_diccionario_lanzadores(equipos, year=2025, cache_file='./cachepitcher_cache.pkl'):
    """
    Crea un diccionario que mapea cada lanzador a su equipo
    
    Returns:
        dict: {nombre_lanzador: team_code}
    """
    print("\n" + "="*70)
    print(" CREANDO BASE DE DATOS DE LANZADORES")
    print("="*70)
    
    # Intentar cargar cache
    try:
        with open(cache_file, 'rb') as f:
            pitcher_dict = pickle.load(f)
            print(f"✅ Cache cargado: {len(pitcher_dict)} lanzadores")
            return pitcher_dict
    except FileNotFoundError:
        print("📝 No se encontró cache, creando nuevo...")
    
    pitcher_dict = {}
    
    for i, team in enumerate(equipos, 1):
        print(f"\n[{i}/{len(equipos)}] Scrapeando roster de {team}...")
        
        roster = obtener_roster_equipo(team, year)
        
        if roster:
            for pitcher in roster:
                # Normalizar nombre para búsqueda flexible
                pitcher_normalized = pitcher.lower().strip()
                pitcher_dict[pitcher_normalized] = team
                print(f"  + {pitcher} → {team}")
        
        time.sleep(2)  # Ser amigable con el servidor
    
    # Guardar cache
    with open(cache_file, 'wb') as f:
        pickle.dump(pitcher_dict, f)
    
    print(f"\n✅ Base de datos creada: {len(pitcher_dict)} lanzadores")
    print(f"💾 Cache guardado en: {cache_file}")
    
    return pitcher_dict


def buscar_equipo_lanzador(nombre_lanzador, pitcher_dict):
    """
    Busca a qué equipo pertenece un lanzador
    
    Args:
        nombre_lanzador: Nombre del lanzador a buscar
        pitcher_dict: Diccionario de lanzadores
    
    Returns:
        team_code o None si no se encuentra
    """
    if not nombre_lanzador or nombre_lanzador == 'N/A':
        return None
    
    nombre_busqueda = nombre_lanzador.lower().strip()
    
    # Búsqueda exacta
    if nombre_busqueda in pitcher_dict:
        return pitcher_dict[nombre_busqueda]
    
    # Búsqueda parcial (apellido)
    apellido = nombre_busqueda.split()[-1] if ' ' in nombre_busqueda else nombre_busqueda
    
    for pitcher_name, team in pitcher_dict.items():
        if apellido in pitcher_name:
            return team
    
    return None


def transformar_csv(input_csv, output_csv='datos_ml_ready.csv', year=2025):
    """
    Transforma el CSV original al formato requerido para el modelo ML
    
    CSV Original:
    - date, away_team, a_team_name, R_A, home_team, h_team_name, R_H
    - Ganador_Pitcher, Perdedor_Pitcher, Salvado_Pitcher, ganador
    
    CSV Output:
    - home_team, away_team, home_pitcher, away_pitcher
    - ganador, year, fecha, score_home, score_away
    """
    print("\n" + "="*70)
    print(" TRANSFORMACIÓN DE CSV")
    print("="*70)
    
    # Leer CSV original
    try:
        df = pd.read_csv(input_csv)
        print(f"\n📂 CSV cargado: {len(df)} partidos")
        print(f"Columnas: {list(df.columns)}")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{input_csv}'")
        print("   Asegúrate de que el archivo esté en el mismo directorio.")
        return None
    
    # Obtener lista única de equipos
    equipos_home = df['home_team'].unique()
    equipos_away = df['away_team'].unique()
    todos_equipos = list(set(list(equipos_home) + list(equipos_away)))
    
    print(f"\n📋 Equipos únicos: {len(todos_equipos)}")
    print(f"   {sorted(todos_equipos)}")
    
    # Crear diccionario de lanzadores
    pitcher_dict = crear_diccionario_lanzadores(todos_equipos, year)
    
    # Procesar cada partido
    print("\n" + "="*70)
    print(" PROCESANDO PARTIDOS")
    print("="*70)
    
    datos_transformados = []
    partidos_sin_datos = []
    lanzadores_no_encontrados = defaultdict(int)
    
    for idx, row in df.iterrows():
        if (idx + 1) % 50 == 0:
            print(f"  Procesando partido {idx+1}/{len(df)}...")
        
        ganador_pitcher = row['Ganador_Pitcher']
        perdedor_pitcher = row['Perdedor_Pitcher']
        
        # Identificar equipos de los lanzadores
        equipo_ganador = buscar_equipo_lanzador(ganador_pitcher, pitcher_dict)
        equipo_perdedor = buscar_equipo_lanzador(perdedor_pitcher, pitcher_dict)
        
        home_team = row['home_team']
        away_team = row['away_team']
        
        # Determinar quién es home_pitcher y away_pitcher
        home_pitcher = None
        away_pitcher = None
        
        # Lógica: El lanzador pertenece al equipo que ganó o perdió
        if equipo_ganador == home_team:
            home_pitcher = ganador_pitcher
            away_pitcher = perdedor_pitcher
        elif equipo_ganador == away_team:
            away_pitcher = ganador_pitcher
            home_pitcher = perdedor_pitcher
        elif equipo_perdedor == home_team:
            home_pitcher = perdedor_pitcher
            away_pitcher = ganador_pitcher
        elif equipo_perdedor == away_team:
            away_pitcher = perdedor_pitcher
            home_pitcher = ganador_pitcher
        
        # Si no pudimos identificar, intentar con lógica de ganador
        if not home_pitcher or not away_pitcher:
            # Si ganó el equipo local (ganador=1), el ganador_pitcher es del home
            if row['ganador'] == 1:
                home_pitcher = ganador_pitcher
                away_pitcher = perdedor_pitcher
            else:
                away_pitcher = ganador_pitcher
                home_pitcher = perdedor_pitcher
        
        # Verificar que tengamos datos completos
        if not home_pitcher or not away_pitcher or home_pitcher == 'N/A' or away_pitcher == 'N/A':
            partidos_sin_datos.append({
                'idx': idx,
                'fecha': row['date'],
                'home': home_team,
                'away': away_team,
                'ganador_p': ganador_pitcher,
                'perdedor_p': perdedor_pitcher
            })
            
            # Registrar lanzadores no encontrados
            if ganador_pitcher != 'N/A':
                lanzadores_no_encontrados[ganador_pitcher] += 1
            if perdedor_pitcher != 'N/A':
                lanzadores_no_encontrados[perdedor_pitcher] += 1
            
            continue
        
        # Crear registro transformado
        registro = {
            'home_team': home_team,
            'away_team': away_team,
            'home_pitcher': home_pitcher,
            'away_pitcher': away_pitcher,
            'ganador': row['ganador'],
            'year': year,
            'fecha': row['date'],
            'score_home': row['R_H'],
            'score_away': row['R_A']
        }
        
        datos_transformados.append(registro)
    
    # Crear DataFrame transformado
    df_transformado = pd.DataFrame(datos_transformados)
    
    print(f"\n✅ Partidos procesados exitosamente: {len(df_transformado)}")
    print(f"⚠️  Partidos sin datos completos: {len(partidos_sin_datos)}")
    
    if len(partidos_sin_datos) > 0:
        print(f"\n📋 REPORTE DE PARTIDOS SIN DATOS:")
        print(f"   Total: {len(partidos_sin_datos)}")
        print(f"\n   Primeros 5 ejemplos:")
        for partido in partidos_sin_datos[:5]:
            print(f"   - {partido['fecha']}: {partido['home']} vs {partido['away']}")
            print(f"     Ganador: {partido['ganador_p']}, Perdedor: {partido['perdedor_p']}")
    
    if lanzadores_no_encontrados:
        print(f"\n📋 LANZADORES NO ENCONTRADOS (Top 10):")
        sorted_pitchers = sorted(lanzadores_no_encontrados.items(), key=lambda x: x[1], reverse=True)
        for pitcher, count in sorted_pitchers[:10]:
            print(f"   - {pitcher}: {count} veces")
    
    # Mostrar muestra
    print(f"\n📊 MUESTRA DEL CSV TRANSFORMADO:")
    print("="*70)
    print(df_transformado.head(10).to_string(index=False))
    
    # Guardar CSV transformado
    df_transformado.to_csv(output_csv, index=False)
    print(f"\n💾 CSV guardado: {output_csv}")
    
    # Estadísticas
    print(f"\n📈 ESTADÍSTICAS:")
    print(f"   Total partidos: {len(df_transformado)}")
    print(f"   Victorias locales: {(df_transformado['ganador'] == 1).sum()}")
    print(f"   Victorias visitantes: {(df_transformado['ganador'] == 0).sum()}")
    print(f"   Equipos únicos: {len(set(df_transformado['home_team'].unique()) | set(df_transformado['away_team'].unique()))}")
    
    return df_transformado


def verificar_transformacion(csv_transformado):
    """
    Verifica que el CSV transformado tenga la estructura correcta
    """
    print("\n" + "="*70)
    print(" VERIFICACIÓN DEL CSV TRANSFORMADO")
    print("="*70)
    
    df = pd.read_csv(csv_transformado)
    
    columnas_requeridas = ['home_team', 'away_team', 'home_pitcher', 'away_pitcher', 'ganador']
    
    print(f"\n✓ Verificando columnas requeridas...")
    for col in columnas_requeridas:
        if col in df.columns:
            print(f"  ✅ {col}: OK")
        else:
            print(f"  ❌ {col}: FALTA")
    
    print(f"\n✓ Verificando datos faltantes...")
    for col in columnas_requeridas:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"  ⚠️  {col}: {missing} valores faltantes")
        else:
            print(f"  ✅ {col}: Sin datos faltantes")
    
    print(f"\n✓ Distribución de ganadores:")
    print(df['ganador'].value_counts())
    
    print(f"\n✅ CSV listo para entrenamiento del modelo!")


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    
    # PASO 1: Transformar CSV
    print("🔄 TRANSFORMADOR DE CSV PARA MODELO ML")
    
    df_transformado = transformar_csv(
        input_csv='./data/raw/resultados_béisbol_season_2022_2023_2024_2025.csv',  # Tu CSV original
        output_csv='./data/processed/datos_ml_ready.csv',  # CSV para el modelo
        year=2025
    )
    
    # PASO 2: Verificar resultado
    if df_transformado is not None and len(df_transformado) > 0:
        verificar_transformacion('./data/processed/datos_ml_ready.csv')
        
        print("\n" + "="*70)
        print("✅ PROCESO COMPLETADO")
        print("="*70)
