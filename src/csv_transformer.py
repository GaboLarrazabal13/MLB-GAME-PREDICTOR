"""
Transformador de CSV de partidos MLB
Identifica a qué equipo pertenece cada lanzador y estructura el CSV para el modelo
"""

import pickle
import time

import cloudscraper
import pandas as pd
from bs4 import BeautifulSoup

# ============================================================================
# FUNCIONES DE SCRAPING
# ============================================================================


def obtener_html(url):
    """Accede a la URL usando cloudscraper."""
    scraper = cloudscraper.create_scraper()
    try:
        response = scraper.get(url, timeout=15)
        response.encoding = "utf-8"
        if response.status_code == 200:
            return response.text
        return None
    except Exception:
        return None


def limpiar_dataframe(df):
    """Limpia el DataFrame eliminando filas no deseadas"""
    if df is None or len(df) == 0:
        return df

    if "Rk" in df.columns:
        df = df.drop("Rk", axis=1)

    name_col = df.columns[0]
    df = df.dropna(subset=[name_col])
    df = df[~df[name_col].astype(str).str.contains(r"Team Totals|Rank in|^\s*$", case=False, na=False, regex=True)]
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

    soup = BeautifulSoup(html, "html.parser")
    pitching_table = soup.find("table", {"id": "players_standard_pitching"})

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
    except Exception:
        # Capturar excepciones e ignorar si no se puede extraer la tabla
        return []

    return []


def crear_diccionario_lanzadores(equipos, year=2025, cache_file="./cachepitcher_cache.pkl"):
    """
    Crea un diccionario que mapea cada lanzador a su equipo

    Returns:
        dict: {nombre_lanzador: team_code}
    """
    print("\n" + "=" * 70)
    print(" CREANDO BASE DE DATOS DE LANZADORES")
    print("=" * 70)

    # Intentar cargar cache
    try:
        with open(cache_file, "rb") as f:
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
    with open(cache_file, "wb") as f:
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
    if not nombre_lanzador or nombre_lanzador == "N/A":
        return None

    nombre_busqueda = nombre_lanzador.lower().strip()

    # Búsqueda exacta
    if nombre_busqueda in pitcher_dict:
        return pitcher_dict[nombre_busqueda]

    # Búsqueda parcial (apellido)
    apellido = nombre_busqueda.split()[-1] if " " in nombre_busqueda else nombre_busqueda

    for pitcher_name, team in pitcher_dict.items():
        if apellido in pitcher_name:
            return team

    return None


def transformar_csv(input_csv, output_csv="datos_ml_ready.csv"):
    """
    Versión Corregida: Extrae el año dinámicamente de la fecha del partido.
    """
    print("\n" + "=" * 70)
    print(" TRANSFORMACIÓN DE CSV DINÁMICA (POR AÑO)")
    print("=" * 70)

    try:
        df = pd.read_csv(input_csv)
        # Convertimos la columna date a datetime para extraer el año real
        df["date"] = pd.to_datetime(df["date"])
        print(f"\n📂 CSV cargado: {len(df)} partidos")
    except Exception as e:
        print(f"❌ Error al cargar CSV: {e}")
        return None

    # Mapeo de lanzadores por año para evitar confusiones de roster
    # {año: {lanzador: equipo}}
    cache_lanzadores_por_anio = {}

    datos_transformados = []

    for _idx, row in df.iterrows():
        # 1. EXTRAER EL AÑO REAL DEL PARTIDO
        anio_partido = row["date"].year

        # 2. OBTENER/CREAR DICCIONARIO PARA ESE AÑO ESPECÍFICO
        if anio_partido not in cache_lanzadores_por_anio:
            print(f"\n📅 Detectado año {anio_partido}. Cargando rosters de esa temporada...")
            equipos = list(set(df["home_team"].unique()) | set(df["away_team"].unique()))
            # Creamos un cache específico para ese año
            cache_lanzadores_por_anio[anio_partido] = crear_diccionario_lanzadores(
                equipos,
                year=anio_partido,
                cache_file=f"./cache/pitcher_cache_{anio_partido}.pkl",
            )

        pitcher_dict = cache_lanzadores_por_anio[anio_partido]

        # 3. LÓGICA DE IDENTIFICACIÓN (Se mantiene tu lógica de búsqueda)
        ganador_pitcher = row["Ganador_Pitcher"]
        perdedor_pitcher = row["Perdedor_Pitcher"]

        equipo_ganador = buscar_equipo_lanzador(ganador_pitcher, pitcher_dict)

        home_team = row["home_team"]
        away_team = row["away_team"]

        # ... (aquí va tu lógica de asignación de home_pitcher/away_pitcher) ...
        # [Mantenemos tu bloque de IF/ELIF para asignar los pitchers]
        if equipo_ganador == home_team:
            home_pitcher, away_pitcher = ganador_pitcher, perdedor_pitcher
        elif equipo_ganador == away_team:
            away_pitcher, home_pitcher = ganador_pitcher, perdedor_pitcher
        else:
            # Si el diccionario falla, usamos la columna 'ganador' como respaldo
            if row["ganador"] == 1:
                home_pitcher, away_pitcher = ganador_pitcher, perdedor_pitcher
            else:
                away_pitcher, home_pitcher = ganador_pitcher, perdedor_pitcher

        # 4. CREAR REGISTRO CON EL AÑO CORRECTO
        registro = {
            "home_team": home_team,
            "away_team": away_team,
            "home_pitcher": home_pitcher,
            "away_pitcher": away_pitcher,
            "ganador": row["ganador"],
            "year": anio_partido,  # <-- AHORA ES DINÁMICO
            "fecha": row["date"].strftime("%Y-%m-%d"),
            "score_home": row["R_H"],
            "score_away": row["R_A"],
        }
        datos_transformados.append(registro)

    df_transformado = pd.DataFrame(datos_transformados)
    df_transformado.to_csv(output_csv, index=False)
    return df_transformado


def verificar_transformacion(csv_transformado):
    """
    Verifica que el CSV transformado tenga la estructura correcta
    """
    print("\n" + "=" * 70)
    print(" VERIFICACIÓN DEL CSV TRANSFORMADO")
    print("=" * 70)

    df = pd.read_csv(csv_transformado)

    columnas_requeridas = [
        "home_team",
        "away_team",
        "home_pitcher",
        "away_pitcher",
        "ganador",
    ]

    print("\n✓ Verificando columnas requeridas...")
    for col in columnas_requeridas:
        if col in df.columns:
            print(f"  ✅ {col}: OK")
        else:
            print(f"  ❌ {col}: FALTA")

    print("\n✓ Verificando datos faltantes...")
    for col in columnas_requeridas:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"  ⚠️  {col}: {missing} valores faltantes")
        else:
            print(f"  ✅ {col}: Sin datos faltantes")

    print("\n✓ Distribución de ganadores:")
    print(df["ganador"].value_counts())

    print("\n✅ CSV listo para entrenamiento del modelo!")


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # PASO 1: Transformar CSV
    print("🔄 TRANSFORMADOR DE CSV PARA MODELO ML")

    df_transformado = transformar_csv(
        input_csv="./data/raw/resultados_béisbol_season_2022_2023_2024_2025.csv",  # Tu CSV original
        output_csv="./data/processed/datos_ml_ready.csv",  # CSV para el modelo
    )

    # PASO 2: Verificar resultado
    if df_transformado is not None and len(df_transformado) > 0:
        verificar_transformacion("./data/processed/datos_ml_ready.csv")

        print("\n" + "=" * 70)
        print("✅ PROCESO COMPLETADO")
        print("=" * 70)
