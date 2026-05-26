"""
Descargador de Temporadas MLB - VERSIÓN 3 (API-FIRST PURE)
Extrae resultados y estadísticas de partidos históricos de la API oficial de la MLB.
100% libre de web scraping a Baseball-Reference.
"""

import os
import re
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

TEAM_CODES = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP",
    "Seattle Mariners": "SEA",
    "San Francisco Giants": "SFG",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN",
    # Abreviaciones
    "Diamondbacks": "ARI",
    "Braves": "ATL",
    "Orioles": "BAL",
    "Red Sox": "BOS",
    "Cubs": "CHC",
    "White Sox": "CHW",
    "Reds": "CIN",
    "Guardians": "CLE",
    "Rockies": "COL",
    "Tigers": "DET",
    "Astros": "HOU",
    "Royals": "KCR",
    "Angels": "LAA",
    "Dodgers": "LAD",
    "Marlins": "MIA",
    "Brewers": "MIL",
    "Twins": "MIN",
    "Mets": "NYM",
    "Yankees": "NYY",
    "Athletics": "OAK",
    "Phillies": "PHI",
    "Pirates": "PIT",
    "Padres": "SDP",
    "Mariners": "SEA",
    "Giants": "SFG",
    "Cardinals": "STL",
    "Rays": "TBR",
    "Rangers": "TEX",
    "Blue Jays": "TOR",
    "Nationals": "WSN",
}


def obtener_codigo_equipo(nombre_completo):
    """Devuelve el código de 3 letras basado en el nombre del equipo."""
    return TEAM_CODES.get(nombre_completo, "UNKNOWN")


def extraer_rango_fechas(fecha_inicio_str, fecha_fin_str, mostrar_progreso=True):
    """
    Obtiene los resultados de partidos para un rango de fechas directamente de la API oficial de la MLB.
    Extremadamente rápido (una sola petición por temporada en lugar de peticiones diarias).
    """
    if mostrar_progreso:
        print(f"📊 Consultando API oficial de la MLB para el rango {fecha_inicio_str} a {fecha_fin_str}...")

    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={fecha_inicio_str}&endDate={fecha_fin_str}&hydrate=probablePitcher,decisions"
    
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            print(f"❌ Error al consultar la API: Status {r.status_code}")
            return pd.DataFrame()

        data = r.json()
        resultados = []

        dates = data.get("dates", [])
        for d in dates:
            fecha = d.get("date")
            games = d.get("games", [])
            for g in games:
                # Solo procesar juegos finalizados
                state = g.get("status", {}).get("abstractGameState", "")
                coded_state = g.get("status", {}).get("codedGameState", "")
                if state != "Final" and coded_state != "F":
                    continue

                equipo_v_name = g["teams"]["away"]["team"]["name"]
                equipo_l_name = g["teams"]["home"]["team"]["name"]
                
                score_v = g["teams"]["away"].get("score", "N/A")
                score_l = g["teams"]["home"].get("score", "N/A")

                # Obtener lanzadores desde decisiones
                decisions = g.get("decisions", {})
                w_pitcher = decisions.get("winner", {}).get("fullName", "N/A")
                l_pitcher = decisions.get("loser", {}).get("fullName", "N/A")
                s_pitcher = decisions.get("save", {}).get("fullName", "NotApplyed")

                # Si no hay decisiones (ej: suspendido, empatado, etc) pero sí probable pitchers
                if w_pitcher == "N/A":
                    w_pitcher = g["teams"]["home"].get("probablePitcher", {}).get("fullName", "N/A")
                if l_pitcher == "N/A":
                    l_pitcher = g["teams"]["away"].get("probablePitcher", {}).get("fullName", "N/A")

                # Ganador
                ganador_col = None
                try:
                    if score_l != "N/A" and score_v != "N/A":
                        r_v_int = int(score_v)
                        r_l_int = int(score_l)
                        ganador_col = 1 if r_l_int > r_v_int else 0
                except (ValueError, TypeError):
                    pass

                resultados.append({
                    "date": fecha,
                    "away_team": obtener_codigo_equipo(equipo_v_name),
                    "a_team_name": equipo_v_name,
                    "R_A": str(score_v),
                    "home_team": obtener_codigo_equipo(equipo_l_name),
                    "h_team_name": equipo_l_name,
                    "R_H": str(score_l),
                    "Ganador_Pitcher": w_pitcher,
                    "Perdedor_Pitcher": l_pitcher,
                    "Salvado_Pitcher": s_pitcher,
                    "ganador": ganador_col,
                })

        df_temp = pd.DataFrame(resultados)
        columnas_ordenadas = [
            "date",
            "away_team",
            "a_team_name",
            "R_A",
            "home_team",
            "h_team_name",
            "R_H",
            "Ganador_Pitcher",
            "Perdedor_Pitcher",
            "Salvado_Pitcher",
            "ganador",
        ]

        if not df_temp.empty:
            return df_temp[columnas_ordenadas]
        return df_temp

    except Exception as e:
        print(f"⚠️ Error general extrayendo rango de fechas: {e}")
        return pd.DataFrame()


def solicitar_temporadas():
    """
    Solicita al usuario la cantidad de temporadas y sus rangos de fechas.
    """
    print("\n" + "=" * 70)
    print(" DESCARGA DE TEMPORADAS MLB (API OFICIAL)")
    print("=" * 70)

    while True:
        try:
            num_temporadas = int(input("\n📋 ¿Cuántas temporadas deseas descargar? (1-10): "))
            if 1 <= num_temporadas <= 10:
                break
            print("   ❌ Por favor, ingresa un número entre 1 y 10")
        except ValueError:
            print("   ❌ Por favor, ingresa un número válido")

    temporadas = []

    for i in range(num_temporadas):
        print(f"\n{'─' * 70}")
        print(f"📅 TEMPORADA {i + 1} de {num_temporadas}")
        print(f"{'─' * 70}")

        while True:
            try:
                año = int(input(f"\nAño para la temporada {i + 1}: "))
                if 2000 <= año <= 2030:
                    break
                print("   ❌ Por favor, ingresa un año entre 2000 y 2030")
            except ValueError:
                print("   ❌ Por favor, ingresa un año válido")

        while True:
            fecha_inicio = input(f"Fecha de inicio de temporada {año} (YYYY-MM-DD): ").strip()
            try:
                datetime.strptime(fecha_inicio, "%Y-%m-%d")
                break
            except ValueError:
                print("   ❌ Formato incorrecto. Usa YYYY-MM-DD (ejemplo: 2023-03-27)")

        while True:
            fecha_fin = input(f"Fecha de fin de temporada {año} (YYYY-MM-DD): ").strip()
            try:
                fecha_fin_dt = datetime.strptime(fecha_fin, "%Y-%m-%d")
                fecha_inicio_dt = datetime.strptime(fecha_inicio, "%Y-%m-%d")

                if fecha_fin_dt >= fecha_inicio_dt:
                    break
                print("   ❌ La fecha de fin debe ser posterior o igual a la fecha de inicio")
            except ValueError:
                print("   ❌ Formato incorrecto. Usa YYYY-MM-DD (ejemplo: 2023-10-01)")

        temporadas.append({"año": año, "fecha_inicio": fecha_inicio, "fecha_fin": fecha_fin})
        print(f"   ✅ Temporada {año} configurada: {fecha_inicio} al {fecha_fin}")

    return temporadas


def procesar_temporadas(temporadas):
    """
    Procesa múltiples temporadas combinando los resultados.
    """
    print("\n" + "=" * 70)
    print(" PROCESAMIENTO DE TEMPORADAS")
    print("=" * 70)

    todos_los_datos = []
    años_procesados = []

    for i, temp in enumerate(temporadas, 1):
        print(f"\n🔄 Procesando Temporada {i}/{len(temporadas)}: {temp['año']}")
        print(f"   Rango: {temp['fecha_inicio']} → {temp['fecha_fin']}")

        df_temporada = extraer_rango_fechas(temp["fecha_inicio"], temp["fecha_fin"], mostrar_progreso=True)

        if not df_temporada.empty:
            df_temporada["Salvado_Pitcher"] = df_temporada["Salvado_Pitcher"].replace("N/A", "NotApplyed")
            todos_los_datos.append(df_temporada)
            años_procesados.append(str(temp["año"]))
            print(f"   ✅ {len(df_temporada)} juegos extraídos para la temporada {temp['año']}")
        else:
            print(f"   ⚠️  No se obtuvieron datos para la temporada {temp['año']}")

    if todos_los_datos:
        df_combinado = pd.concat(todos_los_datos, ignore_index=True)
        return df_combinado, años_procesados
    else:
        return pd.DataFrame(), []


def guardar_resultados(df, años):
    """
    Guarda el DataFrame en un archivo CSV.
    """
    if df.empty:
        print("\n❌ No hay datos para guardar")
        return

    # Crear directorio si no existe
    os.makedirs("./data/raw", exist_ok=True)
    
    años_str = "_".join(años)
    nombre_archivo = f"./data/raw/resultados_béisbol_season_{años_str}.csv"

    df.to_csv(nombre_archivo, index=False)

    print("\n" + "=" * 70)
    print(" ✅ EXTRACCIÓN COMPLETADA")
    print("=" * 70)
    print(f"\n📁 Archivo guardado: {nombre_archivo}")
    print(f"📊 Total de juegos: {len(df)}")
    print(f"📅 Temporadas incluidas: {', '.join(años)}")

    print("\n📋 Primeras 5 filas:")
    print(df.head().to_string())

    print("\n📊 Resumen por temporada:")
    if "date" in df.columns:
        df["year"] = pd.to_datetime(df["date"]).dt.year
        resumen = df.groupby("year").size()
        for año, cantidad in resumen.items():
            print(f"   {año}: {cantidad} juegos")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" DESCARGADOR DE TEMPORADAS MLB (API OFICIAL)")
    print(" Extrae resultados de partidos de la API oficial de la MLB")
    print("=" * 70)

    temporadas = solicitar_temporadas()

    print("\n" + "=" * 70)
    print(" RESUMEN DE TEMPORADAS A DESCARGAR")
    print("=" * 70)

    for i, temp in enumerate(temporadas, 1):
        print(f"{i}. Temporada {temp['año']}: {temp['fecha_inicio']} a {temp['fecha_fin']}")

    confirmar = input("\n¿Deseas continuar con la descarga? (s/n): ").strip().lower()
    if confirmar != "s":
        print("\n Descarga cancelada por el usuario")
        sys.exit(0)

    df_final, años = procesar_temporadas(temporadas)

    if not df_final.empty:
        guardar_resultados(df_final, años)
    else:
        print("\n No se pudieron extraer datos de ninguna temporada")
