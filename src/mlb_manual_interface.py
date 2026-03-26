"""
Interfaz Manual CLI para MLB Predictor V3.5 - REFACTORIZADO
Usa módulos centralizados para evitar duplicación
"""

import sys
import warnings

# Importar módulos centralizados
from mlb_config import TEAM_CODE_TO_NAME, get_team_code, get_team_name
from mlb_predict_engine import predecir_juego

warnings.filterwarnings('ignore')


# ============================================================================
# FUNCIONES DE VALIDACIÓN
# ============================================================================

def normalizar_equipo(team_input):
    """
    Normaliza el input de equipo a código MLB

    Args:
        team_input: Nombre, código o apodo del equipo

    Returns:
        Código de 3 letras o None si es ambiguo/no encontrado
    """
    if not team_input:
        return None

    # Usar la función centralizada
    code = get_team_code(team_input)

    if code:
        return code

    # Si no se encuentra, buscar coincidencias parciales
    team_clean = team_input.strip().lower()
    matches = []

    for code, full_name in TEAM_CODE_TO_NAME.items():
        if team_clean in full_name.lower() or team_clean in code.lower():
            if code not in [m[0] for m in matches]:
                matches.append((code, full_name))

    if len(matches) == 1:
        return matches[0][0]
    elif len(matches) > 1:
        print(f"\n🤔 '{team_input}' es ambiguo. ¿Te refieres a:")
        for i, (code, name) in enumerate(matches, 1):
            print(f"   {i}. {name} ({code})")
        return None

    return None


def mostrar_equipos_disponibles():
    """Muestra una lista de todos los equipos MLB disponibles"""
    print("\n📋 EQUIPOS MLB DISPONIBLES:")
    print("="*60)

    # Agrupar por división (simplificado)
    al_east = ['BAL', 'BOS', 'NYY', 'TBR', 'TOR']
    al_central = ['CHW', 'CLE', 'DET', 'KCR', 'MIN']
    al_west = ['HOU', 'LAA', 'OAK', 'SEA', 'TEX']
    nl_east = ['ATL', 'MIA', 'NYM', 'PHI', 'WSN']
    nl_central = ['CHC', 'CIN', 'MIL', 'PIT', 'STL']
    nl_west = ['ARI', 'COL', 'LAD', 'SDP', 'SFG']

    divisiones = [
        ("AL East", al_east),
        ("AL Central", al_central),
        ("AL West", al_west),
        ("NL East", nl_east),
        ("NL Central", nl_central),
        ("NL West", nl_west)
    ]

    for div_name, teams in divisiones:
        print(f"\n{div_name}:")
        for team_code in teams:
            team_name = get_team_name(team_code)
            print(f"  {team_code:<4} - {team_name}")

    print("="*60)


def validar_year(year_input):
    """
    Valida y normaliza el año ingresado

    Args:
        year_input: String con el año

    Returns:
        Int con el año validado o 2026 por defecto
    """
    if not year_input or not year_input.strip():
        return 2026

    try:
        year = int(year_input)
        if 2015 <= year <= 2026:
            return year
        else:
            print(f"⚠️ Año {year} fuera de rango (2015-2026). Usando 2026.")
            return 2026
    except ValueError:
        print(f"⚠️ '{year_input}' no es un año válido. Usando 2026.")
        return 2026


# ============================================================================
# INTERFAZ CLI PRINCIPAL
# ============================================================================

def ejecutar_cli_manual():
    """Ejecuta la interfaz de línea de comandos para predicción manual"""
    print("\n" + "="*70)
    print("⚾ MLB PREDICTOR V3.5 - INTERFAZ DE PRUEBA MANUAL ⚾")
    print("="*70)
    print("\nEscribe 'help' para ver la lista de equipos disponibles")
    print("Escribe 'quit' o 'exit' para salir\n")

    try:
        while True:
            # 1. Selección de Equipo Local
            home_code = None
            while not home_code:
                h_input = input("🏠 Equipo Local (Nombre/Código): ").strip()

                if h_input.lower() in ['quit', 'exit', 'salir']:
                    print("\n👋 ¡Hasta luego!")
                    sys.exit(0)

                if h_input.lower() == 'help':
                    mostrar_equipos_disponibles()
                    continue

                home_code = normalizar_equipo(h_input)
                if not home_code:
                    print("❌ No reconozco el equipo. Intenta de nuevo o escribe 'help'.")

            # Confirmar equipo seleccionado
            home_name = get_team_name(home_code)
            print(f"✅ Equipo local: {home_name} ({home_code})")

            # 2. Selección de Equipo Visitante
            away_code = None
            while not away_code:
                a_input = input("✈️  Equipo Visitante (Nombre/Código): ").strip()

                if a_input.lower() in ['quit', 'exit', 'salir']:
                    print("\n👋 ¡Hasta luego!")
                    sys.exit(0)

                if a_input.lower() == 'help':
                    mostrar_equipos_disponibles()
                    continue

                away_code = normalizar_equipo(a_input)

                if away_code == home_code:
                    print("❌ El equipo visitante no puede ser el mismo que el local.")
                    away_code = None
                    continue

                if not away_code:
                    print("❌ No reconozco el equipo. Intenta de nuevo o escribe 'help'.")

            # Confirmar equipo seleccionado
            away_name = get_team_name(away_code)
            print(f"✅ Equipo visitante: {away_name} ({away_code})")

            # 3. Lanzadores
            print("-" * 40)
            p_home = input(f"👤 Lanzador abridor de {home_code}: ").strip()
            if not p_home:
                print("⚠️ Debes ingresar un nombre de lanzador")
                continue

            p_away = input(f"👤 Lanzador abridor de {away_code}: ").strip()
            if not p_away:
                print("⚠️ Debes ingresar un nombre de lanzador")
                continue

            # 4. Año para scraping
            year_input = input("📅 Año para el scraping de stats (Enter=2026): ").strip()
            year_val = validar_year(year_input)

            # 5. Confirmación
            print("\n" + "-" * 40)
            print("📋 RESUMEN DE LA PREDICCIÓN:")
            print(f"  Local:     {home_name} ({home_code})")
            print(f"  Visitante: {away_name} ({away_code})")
            print(f"  Lanzadores: {p_home} vs {p_away}")
            print(f"  Año de stats: {year_val}")
            print("-" * 40)

            confirmar = input("\n¿Continuar con la predicción? (S/n): ").strip().lower()
            if confirmar and confirmar not in ['s', 'si', 'sí', 'yes', 'y']:
                print("❌ Predicción cancelada\n")
                continue

            print("\n" + "-" * 40)
            print(f"🚀 Iniciando Predicción: {away_code} @ {home_code} ({year_val})")
            print("🔎 Buscando estadísticas en Baseball-Reference...")
            print("⏳ Este proceso puede tardar 30-60 segundos...\n")

            # 6. Llamada al motor de predicción
            resultado = predecir_juego(
                home_team=home_code,
                away_team=away_code,
                home_pitcher=p_home,
                away_pitcher=p_away,
                year=year_val,
                modo_auto=False
            )

            if resultado:
                print("\n✅ Predicción completada con éxito.")
                print("💾 Guardado en DB 'predicciones_historico' con tipo 'MANUAL'")
            else:
                print("\n❌ Error: No se pudo completar la predicción.")
                print("Posibles causas:")
                print("  - Nombres de lanzadores incorrectos")
                print("  - Problemas de conexión con Baseball-Reference")
                print("  - Datos insuficientes para el año seleccionado")

            # 7. Preguntar si desea hacer otra predicción
            print("\n" + "="*70)
            otra = input("¿Realizar otra predicción? (S/n): ").strip().lower()
            if otra and otra not in ['s', 'si', 'sí', 'yes', 'y']:
                print("\n👋 ¡Hasta luego!")
                break

            print("\n" + "="*70 + "\n")

    except KeyboardInterrupt:
        print("\n\n👋 Saliendo del predictor...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def modo_rapido(home, away, hp, ap, year=2026):
    """
    Modo rápido para predicción sin interacción

    Args:
        home: Código equipo local
        away: Código equipo visitante
        hp: Lanzador local
        ap: Lanzador visitante
        year: Año para stats
    """
    home_code = normalizar_equipo(home)
    away_code = normalizar_equipo(away)

    if not home_code or not away_code:
        print("❌ Error: Equipos no válidos")
        return None

    print(f"\n🚀 Modo Rápido: {away_code} @ {home_code}")

    resultado = predecir_juego(
        home_team=home_code,
        away_team=away_code,
        home_pitcher=hp,
        away_pitcher=ap,
        year=year,
        modo_auto=False
    )

    return resultado


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    # Detectar si se pasan argumentos por línea de comandos
    if len(sys.argv) > 1:
        # Modo rápido con argumentos
        # Uso: python mlb_manual_interface.py NYY BOS "Gerrit Cole" "Tanner Houck" 2024
        if len(sys.argv) >= 5:
            home = sys.argv[1]
            away = sys.argv[2]
            hp = sys.argv[3]
            ap = sys.argv[4]
            year = int(sys.argv[5]) if len(sys.argv) > 5 else 2026

            modo_rapido(home, away, hp, ap, year)
        else:
            print("❌ Uso: python mlb_manual_interface.py <home> <away> <home_pitcher> <away_pitcher> [year]")
            print("Ejemplo: python mlb_manual_interface.py NYY BOS \"Gerrit Cole\" \"Tanner Houck\" 2024")
    else:
        # Modo interactivo
        ejecutar_cli_manual()
