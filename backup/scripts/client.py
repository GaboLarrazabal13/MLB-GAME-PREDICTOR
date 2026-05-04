"""
Cliente Interactivo para API MLB Predictor
Ejecutar: python client.py
"""

from datetime import datetime

import requests

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

API_URL = "http://localhost:8000"

# Mapeo de equipos para ayuda
EQUIPOS_MLB = {
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

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def limpiar_pantalla():
    """Limpia la pantalla (multiplataforma)"""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


def mostrar_banner():
    """Muestra el banner de la aplicación"""
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║             MLB GAME PREDICTOR - CLIENTE INTERACTIVO          ║
    ║                                                                   ║
    ║           Predicciones de partidos MLB con Machine Learning      ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)


def verificar_api():
    """Verifica que la API esté disponible"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('model_loaded'):
                return True, "✅ API conectada y modelo cargado"
            else:
                return False, "⚠️  API conectada pero modelo no cargado"
        else:
            return False, f"❌ API respondió con código {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, f"❌ No se pudo conectar a {API_URL}"
    except requests.exceptions.Timeout:
        return False, "❌ Timeout al conectar con la API"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


def mostrar_equipos():
    """Muestra la lista de equipos disponibles"""
    print("\n EQUIPOS MLB DISPONIBLES:")
    print("="*70)

    # Dividir en 2 columnas
    equipos_lista = list(EQUIPOS_MLB.items())
    mitad = len(equipos_lista) // 2

    for i in range(mitad):
        equipo1 = equipos_lista[i]
        equipo2 = equipos_lista[i + mitad] if i + mitad < len(equipos_lista) else None

        linea1 = f"   {equipo1[0]:4s} = {equipo1[1]:30s}"
        linea2 = f"   {equipo2[0]:4s} = {equipo2[1]}" if equipo2 else ""

        print(f"{linea1}  {linea2}")

    print("="*70)


def validar_equipo(equipo_input):
    """Valida que el equipo existe"""
    equipo_upper = equipo_input.strip().upper()

    # Buscar por código
    if equipo_upper in EQUIPOS_MLB:
        return equipo_upper, EQUIPOS_MLB[equipo_upper]

    # Buscar por nombre
    for codigo, nombre in EQUIPOS_MLB.items():
        if equipo_input.lower() in nombre.lower():
            return codigo, nombre

    return None, None


def solicitar_equipo(tipo="Local"):
    """Solicita un equipo con validación"""
    while True:
        equipo = input(f"  Equipo {tipo} (código o nombre, '?' para lista): ").strip()

        if equipo == '?':
            mostrar_equipos()
            continue

        codigo, nombre = validar_equipo(equipo)

        if codigo:
            print(f"     ✅ {nombre} ({codigo})")
            return codigo
        else:
            print(f"     ❌ '{equipo}' no reconocido. Intenta de nuevo o escribe '?' para ver la lista")


def hacer_prediccion(home_team, away_team, home_pitcher, away_pitcher, year):
    """Realiza la predicción llamando a la API"""

    print("\n Realizando predicción...")
    print("   Esto puede tomar 10-30 segundos...")

    data = {
        "home_team": home_team,
        "away_team": away_team,
        "home_pitcher": home_pitcher,
        "away_pitcher": away_pitcher,
        "year": year
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=data,
            timeout=120  # 2 minutos de timeout
        )

        if response.status_code == 200:
            return True, response.json()
        elif response.status_code == 400:
            error = response.json()
            return False, f"Error en los datos: {error.get('detail', 'Error desconocido')}"
        elif response.status_code == 503:
            return False, "El modelo no está cargado en la API"
        else:
            return False, f"Error {response.status_code}: {response.text}"

    except requests.exceptions.Timeout:
        return False, "Timeout: La predicción tardó demasiado. Verifica que la API esté funcionando."
    except Exception as e:
        return False, f"Error al conectar con la API: {str(e)}"


def mostrar_resultado(resultado):
    """Muestra el resultado de la predicción de forma visual"""
    print("\n" + "="*70)
    print("  RESULTADO DE LA PREDICCIÓN")
    print("="*70)

    home = resultado.get('equipo_local', 'Local')
    away = resultado.get('equipo_visitante', 'Visitante')

    # Si la API no devuelve los nombres, buscarlos
    if 'equipo_local' not in resultado:
        # Asumimos que los códigos están en prob_home/prob_away context
        home = "Local"
        away = "Visitante"

    ganador = resultado.get('ganador')
    prob_home = resultado.get('prob_home', 0)
    prob_away = resultado.get('prob_away', 0)
    confianza = resultado.get('confianza', 0)

    # Información del partido
    print("\n  PARTIDO:")
    print(f"   Local: {home}")
    print(f"   Visitante: {away}")

    # Ganador predicho
    print(f"\n🏆 GANADOR PREDICHO: {ganador}")

    # Probabilidades con barras visuales
    print("\n PROBABILIDADES:")

    barra_home = "█" * int(prob_home * 50)
    barra_away = "█" * int(prob_away * 50)

    print(f"   Local:     {prob_home*100:5.1f}%  {barra_home}")
    print(f"   Visitante: {prob_away*100:5.1f}%  {barra_away}")

    # Nivel de confianza
    print(f"\n CONFIANZA: {confianza*100:.1f}%")

    if confianza > 0.70:
        nivel = "MUY ALTA ✅✅✅"
        emoji = "🔥"
    elif confianza > 0.60:
        nivel = "ALTA ✅✅"
        emoji = "👍"
    elif confianza > 0.55:
        nivel = "MODERADA ⚠️"
        emoji = "🤔"
    else:
        nivel = "BAJA ❌ (Partido muy parejo)"
        emoji = "🤷"

    print(f"   {emoji} {nivel}")

    # Mensaje adicional si existe
    if resultado.get('mensaje'):
        print(f"\nℹ️  {resultado.get('mensaje')}")

    print("\n" + "="*70)


# ============================================================================
# MENÚ PRINCIPAL
# ============================================================================

def menu_prediccion():
    """Menú para hacer una predicción"""
    print("\n" + "="*70)
    print("  NUEVA PREDICCIÓN")
    print("="*70)

    print("\n Puedes usar códigos (NYY) o nombres completos (Yankees)")
    print("   Escribe '?' en cualquier momento para ver todos los equipos\n")

    # Solicitar datos
    home_team = solicitar_equipo("Local")
    away_team = solicitar_equipo("Visitante")

    print("\n LANZADORES:")
    home_pitcher = input("  Lanzador Local (apellido o nombre): ").strip()
    away_pitcher = input("  Lanzador Visitante (apellido o nombre): ").strip()

    year_input = input("\n Temporada (Enter para 2026): ").strip()
    year = int(year_input) if year_input else 2026

    # Confirmar
    print("\n RESUMEN:")
    print(f"   {EQUIPOS_MLB.get(home_team, home_team)} vs {EQUIPOS_MLB.get(away_team, away_team)}")
    print(f"   Lanzadores: {home_pitcher} vs {away_pitcher}")
    print(f"   Temporada: {year}")

    confirmar = input("\n¿Realizar predicción? (s/n): ").strip().lower()

    if confirmar != 's':
        print(" Predicción cancelada")
        return

    # Hacer predicción
    exito, resultado = hacer_prediccion(home_team, away_team, home_pitcher, away_pitcher, year)

    if exito:
        mostrar_resultado(resultado)

        # Guardar resultado
        guardar = input("\n💾 ¿Guardar resultado? (s/n): ").strip().lower()
        if guardar == 's':
            guardar_prediccion(home_team, away_team, home_pitcher, away_pitcher, year, resultado)
    else:
        print(f"\n❌ ERROR: {resultado}")


def guardar_prediccion(home_team, away_team, home_pitcher, away_pitcher, year, resultado):
    """Guarda la predicción en un archivo"""
    import csv
    from pathlib import Path

    archivo = Path('predicciones_historial.csv')

    # Crear archivo con headers si no existe
    if not archivo.exists():
        with open(archivo, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'fecha', 'home_team', 'away_team', 'home_pitcher', 'away_pitcher',
                'year', 'ganador_predicho', 'prob_home', 'prob_away', 'confianza'
            ])

    # Agregar predicción
    with open(archivo, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            home_team,
            away_team,
            home_pitcher,
            away_pitcher,
            year,
            resultado.get('ganador'),
            resultado.get('prob_home'),
            resultado.get('prob_away'),
            resultado.get('confianza')
        ])

    print(f"✅ Predicción guardada en {archivo}")


def menu_historial():
    """Muestra el historial de predicciones"""
    import csv
    from pathlib import Path

    archivo = Path('predicciones_historial.csv')

    if not archivo.exists():
        print("\n No hay predicciones guardadas aún")
        return

    print("\n" + "="*70)
    print(" HISTORIAL DE PREDICCIONES")
    print("="*70)

    with open(archivo) as f:
        reader = csv.DictReader(f)
        predicciones = list(reader)

    if not predicciones:
        print("\n No hay predicciones guardadas")
        return

    print(f"\nTotal: {len(predicciones)} predicciones\n")

    for i, pred in enumerate(reversed(predicciones[-10:]), 1):
        print(f"{i}. {pred['fecha']}")
        print(f"   {pred['home_team']} vs {pred['away_team']}")
        print(f"   Ganador: {pred['ganador_predicho']} ({float(pred['confianza'])*100:.0f}%)")
        print()


def menu_configuracion():
    """Menú de configuración"""
    global API_URL

    print("\n" + "="*70)
    print(" ⚙️  CONFIGURACIÓN")
    print("="*70)

    print(f"\n🔗 URL actual de API: {API_URL}")

    nueva_url = input("\nNueva URL (Enter para mantener): ").strip()

    if nueva_url:
        API_URL = nueva_url
        print(f"✅ URL actualizada: {API_URL}")

        # Verificar conexión
        print("\n🔍 Verificando conexión...")
        exito, mensaje = verificar_api()
        print(mensaje)


def menu_principal():
    """Menú principal de la aplicación"""
    while True:
        limpiar_pantalla()
        mostrar_banner()

        # Verificar API
        exito, mensaje = verificar_api()
        print(f" Estado: {mensaje}")
        print(f"🔗 URL: {API_URL}\n")

        print("="*70)
        print(" MENÚ PRINCIPAL")
        print("="*70)
        print("\n1. 🎯 Hacer nueva predicción")
        print("2. 📜 Ver historial de predicciones")
        print("3. 📋 Ver lista de equipos MLB")
        print("4. ⚙️  Configuración")
        print("5. 🚪 Salir")

        opcion = input("\nSelecciona una opción (1-5): ").strip()

        if opcion == "1":
            if not exito:
                print(f"\n❌ {mensaje}")
                print("   Verifica que la API esté corriendo:")
                print("   uvicorn api:app --reload")
                input("\nPresiona Enter para continuar...")
            else:
                menu_prediccion()
                input("\nPresiona Enter para continuar...")

        elif opcion == "2":
            menu_historial()
            input("\nPresiona Enter para continuar...")

        elif opcion == "3":
            mostrar_equipos()
            input("\nPresiona Enter para continuar...")

        elif opcion == "4":
            menu_configuracion()
            input("\nPresiona Enter para continuar...")

        elif opcion == "5":
            print("\n ¡Hasta luego!")
            break

        else:
            print("\n❌ Opción inválida")
            input("\nPresiona Enter para continuar...")


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    try:
        menu_principal()
    except KeyboardInterrupt:
        print("\n\n ¡Hasta luego!")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()


