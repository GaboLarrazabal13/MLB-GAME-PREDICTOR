"""
Transformador de CSV de partidos MLB - VERSIÓN 3 (API-FIRST PURE)
Identifica a qué equipo pertenece cada lanzador y estructura el CSV para el modelo.
100% libre de web scraping a Baseball-Reference.
"""

import pickle
import time
import os
import sys
import pandas as pd
import requests

# Importar cliente de la API oficial de la MLB
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_stats_api_client import obtener_stats_jugadores_equipo_api, TEAM_CODE_TO_ID


def obtener_roster_equipo(team_code, year=2025):
    """
    Obtiene la lista de lanzadores de un equipo a través de la API oficial de la MLB.
    Returns: lista de nombres de lanzadores
    """
    try:
        splits = obtener_stats_jugadores_equipo_api(team_code, year, group="pitching")
        if not splits:
            return []
        
        nombres = []
        for split in splits:
            player_info = split.get("player", {})
            full_name = player_info.get("fullName")
            if full_name:
                nombres.append(full_name.strip())
        return nombres
    except Exception as e:
        print(f"  ⚠️ Error al obtener roster API para {team_code}: {e}")
        return []


def crear_diccionario_lanzadores(equipos, year=2025, cache_file="./cache/pitcher_cache.pkl"):
    """
    Crea un diccionario que mapea cada lanzador a su equipo.
    """
    print("\n" + "=" * 70)
    print(f" CREANDO BASE DE DATOS DE LANZADORES ({year}) VIA MLB API")
    print("=" * 70)

    # Asegurar que el directorio de cache existe
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

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
        print(f"\n[{i}/{len(equipos)}] Obteniendo roster de {team}...")
        roster = obtener_roster_equipo(team, year)

        if roster:
            for pitcher in roster:
                pitcher_normalized = pitcher.lower().strip()
                pitcher_dict[pitcher_normalized] = team
                print(f"  + {pitcher} → {team}")

        time.sleep(0.5)  # Ser amigable con la API

    # Guardar cache
    with open(cache_file, "wb") as f:
        pickle.dump(pitcher_dict, f)

    print(f"\n✅ Base de datos creada: {len(pitcher_dict)} lanzadores")
    print(f"💾 Cache guardado en: {cache_file}")

    return pitcher_dict


def buscar_equipo_lanzador(nombre_lanzador, pitcher_dict):
    """
    Busca a qué equipo pertenece un lanzador.
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
    Transforma el CSV original extrayendo el año dinámicamente de la fecha del partido.
    """
    print("\n" + "=" * 70)
    print(" TRANSFORMACIÓN DE CSV DINÁMICA (POR AÑO)")
    print("=" * 70)

    try:
        df = pd.read_csv(input_csv)
        df["date"] = pd.to_datetime(df["date"])
        print(f"\n📂 CSV cargado: {len(df)} partidos")
    except Exception as e:
        print(f"❌ Error al cargar CSV: {e}")
        return None

    cache_lanzadores_por_anio = {}
    datos_transformados = []

    for _idx, row in df.iterrows():
        anio_partido = row["date"].year

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

        ganador_pitcher = row["Ganador_Pitcher"]
        perdedor_pitcher = row["Perdedor_Pitcher"]

        equipo_ganador = buscar_equipo_lanzador(ganador_pitcher, pitcher_dict)

        home_team = row["home_team"]
        away_team = row["away_team"]

        if equipo_ganador == home_team:
            home_pitcher, away_pitcher = ganador_pitcher, perdedor_pitcher
        elif equipo_ganador == away_team:
            away_pitcher, home_pitcher = ganador_pitcher, perdedor_pitcher
        else:
            if row["ganador"] == 1:
                home_pitcher, away_pitcher = ganador_pitcher, perdedor_pitcher
            else:
                away_pitcher, home_pitcher = ganador_pitcher, perdedor_pitcher

        registro = {
            "home_team": home_team,
            "away_team": away_team,
            "home_pitcher": home_pitcher,
            "away_pitcher": away_pitcher,
            "ganador": row["ganador"],
            "year": anio_partido,
            "fecha": row["date"].strftime("%Y-%m-%d"),
            "score_home": row["R_H"],
            "score_away": row["R_A"],
        }
        datos_transformados.append(registro)

    df_transformado = pd.DataFrame(datos_transformados)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_transformado.to_csv(output_csv, index=False)
    return df_transformado


def verificar_transformacion(csv_transformado):
    """
    Verifica que el CSV transformado tenga la estructura correcta.
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


if __name__ == "__main__":
    print("🔄 TRANSFORMADOR DE CSV PARA MODELO ML (API OFICIAL)")
    
    df_transformado = transformar_csv(
        input_csv="./data/raw/resultados_béisbol_season_2022_2023_2024_2025.csv",
        output_csv="./data/processed/datos_ml_ready.csv",
    )

    if df_transformado is not None and len(df_transformado) > 0:
        verificar_transformacion("./data/processed/datos_ml_ready.csv")
        print("\n" + "=" * 70)
        print("✅ PROCESO COMPLETADO")
        print("=" * 70)
