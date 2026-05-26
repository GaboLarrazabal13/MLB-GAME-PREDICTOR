"""
Utilidades adicionales para el Sistema MLB Predictor V3.5
Funciones helper para análisis, reportes y mantenimiento
"""

import os
import sqlite3
from datetime import datetime, timedelta

import pandas as pd

from mlb_config import CACHE_PATH, DB_PATH, MODELO_PATH, get_team_name

# ============================================================================
# ANÁLISIS DE RENDIMIENTO DEL MODELO
# ============================================================================


def analizar_accuracy_historico(dias=30):
    """
    Analiza el accuracy del modelo en predicciones recientes

    Args:
        dias: Número de días hacia atrás para analizar

    Returns:
        DataFrame con estadísticas de rendimiento
    """
    if not os.path.exists(DB_PATH):
        print("❌ Base de datos no encontrada")
        return None

    fecha_limite = datetime.now() - timedelta(days=dias)

    with sqlite3.connect(DB_PATH) as conn:
        # Obtener predicciones
        query_pred = f"""
            SELECT * FROM predicciones_historico
            WHERE fecha >= '{fecha_limite.strftime("%Y-%m-%d")}'
            ORDER BY fecha DESC
        """
        df_pred = pd.read_sql(query_pred, conn)

        # Obtener resultados reales
        query_real = f"""
            SELECT * FROM historico_real
            WHERE fecha >= '{fecha_limite.strftime("%Y-%m-%d")}'
        """
        df_real = pd.read_sql(query_real, conn)

    if df_pred.empty or df_real.empty:
        print(f"⚠️ No hay datos suficientes para los últimos {dias} días")
        return None

    # Merge para comparar
    df_pred["match_key"] = df_pred["fecha"] + "_" + df_pred["home_team"] + "_" + df_pred["away_team"]
    df_real["match_key"] = df_real["fecha"].astype(str) + "_" + df_real["home_team"] + "_" + df_real["away_team"]

    merged = df_pred.merge(df_real[["match_key", "ganador"]], on="match_key", how="inner")

    if merged.empty:
        print("⚠️ No se pudieron emparejar predicciones con resultados")
        return None

    # Calcular aciertos
    merged["acierto"] = merged.apply(
        lambda row: (
            (row["prediccion"] == row["home_team"] and row["ganador"] == 1)
            or (row["prediccion"] == row["away_team"] and row["ganador"] == 0)
        ),
        axis=1,
    )

    # Estadísticas generales
    total = len(merged)
    aciertos = merged["acierto"].sum()
    accuracy = (aciertos / total * 100) if total > 0 else 0

    # Estadísticas por confianza
    stats_confianza = merged.groupby("confianza").agg({"acierto": ["count", "sum", "mean"]}).round(3)

    print("\n" + "=" * 60)
    print(f"📊 ANÁLISIS DE RENDIMIENTO - Últimos {dias} días")
    print("=" * 60)
    print(f"Total de predicciones: {total}")
    print(f"Aciertos: {aciertos}")
    print(f"Accuracy General: {accuracy:.2f}%")
    print("\n📈 Accuracy por Nivel de Confianza:")
    print(stats_confianza)
    print("=" * 60 + "\n")

    return merged


def generar_reporte_equipos(equipo_code, ultimos_n=20):
    """
    Genera un reporte del rendimiento del modelo para un equipo específico

    Args:
        equipo_code: Código del equipo (ej: 'NYY')
        ultimos_n: Número de partidos recientes a analizar
    """
    if not os.path.exists(DB_PATH):
        print("❌ Base de datos no encontrada")
        return

    with sqlite3.connect(DB_PATH) as conn:
        query = f"""
            SELECT * FROM predicciones_historico
            WHERE home_team = '{equipo_code}' OR away_team = '{equipo_code}'
            ORDER BY fecha DESC
            LIMIT {ultimos_n}
        """
        df = pd.read_sql(query, conn)

    if df.empty:
        print(f"⚠️ No hay predicciones para {equipo_code}")
        return

    team_name = get_team_name(equipo_code)

    print("\n" + "=" * 60)
    print(f"📊 REPORTE DE PREDICCIONES - {team_name} ({equipo_code})")
    print("=" * 60)

    # Estadísticas como local y visitante
    como_local = df[df["home_team"] == equipo_code]
    como_visitante = df[df["away_team"] == equipo_code]

    predicho_ganar_local = (como_local["prediccion"] == equipo_code).sum()
    predicho_ganar_visitante = (como_visitante["prediccion"] == equipo_code).sum()

    print(f"\nComo Local: {len(como_local)} juegos")
    print(f"  Predicho ganar: {predicho_ganar_local} ({predicho_ganar_local / len(como_local) * 100:.1f}%)")

    print(f"\nComo Visitante: {len(como_visitante)} juegos")
    print(f"  Predicho ganar: {predicho_ganar_visitante} ({predicho_ganar_visitante / len(como_visitante) * 100:.1f}%)")

    # Probabilidad promedio
    prob_promedio_local = como_local["prob_home"].mean() if len(como_local) > 0 else 0
    prob_promedio_visitante = (100 - como_visitante["prob_home"]).mean() if len(como_visitante) > 0 else 0

    print(f"\nProbabilidad promedio como local: {prob_promedio_local:.1f}%")
    print(f"Probabilidad promedio como visitante: {prob_promedio_visitante:.1f}%")

    print("\n📋 Últimas 5 predicciones:")
    print("-" * 60)
    for _, row in df.head(5).iterrows():
        es_local = row["home_team"] == equipo_code
        rival = row["away_team"] if es_local else row["home_team"]
        prob = row["prob_home"] if es_local else row["prob_away"]
        print(f"{row['fecha']}: vs {rival} - Pred: {row['prediccion']} ({prob:.1f}%) [{row['confianza']}]")

    print("=" * 60 + "\n")


# ============================================================================
# MANTENIMIENTO DE BASE DE DATOS
# ============================================================================


def limpiar_cache():
    """Elimina el caché de features para forzar re-scraping"""
    if os.path.exists(CACHE_PATH):
        backup_path = CACHE_PATH + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.rename(CACHE_PATH, backup_path)
        print(f"✅ Caché respaldado en: {backup_path}")
        print("✅ Caché limpiado. El próximo entrenamiento volverá a scrapear datos.")
    else:
        print("⚠️ No hay caché para limpiar")


def compactar_base_datos():
    """Compacta y optimiza la base de datos SQLite"""
    if not os.path.exists(DB_PATH):
        print("❌ Base de datos no encontrada")
        return

    with sqlite3.connect(DB_PATH) as conn:
        # Obtener tamaño antes
        cursor = conn.cursor()
        cursor.execute("PRAGMA page_count")
        page_count_before = cursor.fetchone()[0]
        cursor.execute("PRAGMA page_size")
        page_size = cursor.fetchone()[0]
        size_before = page_count_before * page_size / 1024 / 1024  # MB

        # Compactar
        conn.execute("VACUUM")

        # Obtener tamaño después
        cursor.execute("PRAGMA page_count")
        page_count_after = cursor.fetchone()[0]
        size_after = page_count_after * page_size / 1024 / 1024  # MB

        print("✅ Base de datos compactada")
        print(f"   Antes: {size_before:.2f} MB")
        print(f"   Después: {size_after:.2f} MB")
        print(f"   Ahorro: {size_before - size_after:.2f} MB")


def verificar_integridad_db():
    """Verifica la integridad de la base de datos"""
    if not os.path.exists(DB_PATH):
        print("❌ Base de datos no encontrada")
        return

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        # Verificar integridad
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()[0]

        if result == "ok":
            print("✅ Integridad de base de datos: OK")
        else:
            print(f"❌ Problemas de integridad: {result}")

        # Listar tablas
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tablas = cursor.fetchall()

        print("\n📋 Tablas en la base de datos:")
        for tabla in tablas:
            cursor.execute(f"SELECT COUNT(*) FROM {tabla[0]}")
            count = cursor.fetchone()[0]
            print(f"   {tabla[0]}: {count} registros")


# ============================================================================
# EXPORTACIÓN DE DATOS
# ============================================================================


def exportar_predicciones_csv(output_path="predicciones_export.csv", dias=30):
    """
    Exporta predicciones recientes a CSV

    Args:
        output_path: Ruta del archivo CSV de salida
        dias: Número de días hacia atrás para exportar
    """
    if not os.path.exists(DB_PATH):
        print("❌ Base de datos no encontrada")
        return

    fecha_limite = datetime.now() - timedelta(days=dias)

    with sqlite3.connect(DB_PATH) as conn:
        query = f"""
            SELECT * FROM predicciones_historico
            WHERE fecha >= '{fecha_limite.strftime("%Y-%m-%d")}'
            ORDER BY fecha DESC
        """
        df = pd.read_sql(query, conn)

    if df.empty:
        print(f"⚠️ No hay predicciones en los últimos {dias} días")
        return

    df.to_csv(output_path, index=False)
    print(f"✅ {len(df)} predicciones exportadas a: {output_path}")


def exportar_resultados_csv(output_path="resultados_export.csv", dias=30):
    """
    Exporta resultados reales a CSV

    Args:
        output_path: Ruta del archivo CSV de salida
        dias: Número de días hacia atrás para exportar
    """
    if not os.path.exists(DB_PATH):
        print("❌ Base de datos no encontrada")
        return

    fecha_limite = datetime.now() - timedelta(days=dias)

    with sqlite3.connect(DB_PATH) as conn:
        query = f"""
            SELECT * FROM historico_real
            WHERE fecha >= '{fecha_limite.strftime("%Y-%m-%d")}'
            ORDER BY fecha DESC
        """
        df = pd.read_sql(query, conn)

    if df.empty:
        print(f"⚠️ No hay resultados en los últimos {dias} días")
        return

    df.to_csv(output_path, index=False)
    print(f"✅ {len(df)} resultados exportados a: {output_path}")


# ============================================================================
# MONITOREO DEL MODELO
# ============================================================================


def verificar_estado_modelo():
    """Verifica el estado del modelo y archivos necesarios"""
    print("\n" + "=" * 60)
    print("🔍 VERIFICACIÓN DEL SISTEMA MLB PREDICTOR")
    print("=" * 60)

    # Modelo
    if os.path.exists(MODELO_PATH):
        size = os.path.getsize(MODELO_PATH) / 1024  # KB
        mtime = datetime.fromtimestamp(os.path.getmtime(MODELO_PATH))
        print(f"✅ Modelo: {MODELO_PATH}")
        print(f"   Tamaño: {size:.2f} KB")
        print(f"   Última modificación: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"❌ Modelo NO encontrado: {MODELO_PATH}")

    # Base de datos
    if os.path.exists(DB_PATH):
        size = os.path.getsize(DB_PATH) / 1024 / 1024  # MB
        print(f"\n✅ Base de datos: {DB_PATH}")
        print(f"   Tamaño: {size:.2f} MB")
    else:
        print(f"\n❌ Base de datos NO encontrada: {DB_PATH}")

    # Caché
    if os.path.exists(CACHE_PATH):
        size = os.path.getsize(CACHE_PATH) / 1024 / 1024  # MB
        mtime = datetime.fromtimestamp(os.path.getmtime(CACHE_PATH))
        print(f"\n✅ Caché: {CACHE_PATH}")
        print(f"   Tamaño: {size:.2f} MB")
        print(f"   Última actualización: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"\n⚠️ Caché NO encontrado: {CACHE_PATH}")

    print("=" * 60 + "\n")


# ============================================================================
# FUNCIONES AUXILIARES DE MACHINE LEARNING E INGENIERÍA DE FEATURES
# ============================================================================

def normalizar_texto(texto):
    """Normaliza texto para comparaciones de nombres"""
    import re
    import unicodedata
    if not texto:
        return ""
    texto = str(texto).lower()
    texto = "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^a-z0-9]", "", texto)
    return texto


def safe_float(val):
    """Convierte a float de forma segura manejando errores y NaNs"""
    try:
        if pd.isna(val):
            return 0.0
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def safe_int(val):
    """Convierte a int de forma segura manejando errores y NaNs"""
    try:
        if pd.isna(val):
            return 0
        return int(float(val))
    except (ValueError, TypeError):
        return 0


def alinear_features_entrenamiento(X_new, model_actual=None):
    """Asegura un esquema estable de features para reentrenamiento incremental."""
    from mlb_config import SCRAPING_FEATURES, SUPER_FEATURES, TEMPORAL_FEATURES
    features_esperadas = TEMPORAL_FEATURES + SCRAPING_FEATURES + SUPER_FEATURES

    columnas_faltantes = [col for col in features_esperadas if col not in X_new.columns]
    if columnas_faltantes:
        for columna in columnas_faltantes:
            X_new[columna] = 0

    if model_actual is None:
        return X_new

    feature_names_modelo = model_actual.get_booster().feature_names or []
    if not feature_names_modelo:
        return X_new

    columnas_faltantes_modelo = [col for col in feature_names_modelo if col not in X_new.columns]
    if columnas_faltantes_modelo:
        for columna in columnas_faltantes_modelo:
            X_new[columna] = 0

    return X_new.reindex(columns=feature_names_modelo, fill_value=0)


def calcular_tendencias_equipo(df, team, fecha_limite, ventana=10):
    """Calcula rendimiento reciente (ventana) y de toda la temporada de un equipo"""
    from mlb_config import get_team_code, get_team_name

    if isinstance(fecha_limite, str):
        fecha_limite = pd.to_datetime(fecha_limite)

    # Normalizar equipo
    t_code = get_team_code(team)
    t_full = get_team_name(t_code)

    # Normalizar columnas de equipo en el DF para evitar fallos por espacios o mayúsculas
    df = df.copy()
    df["home_team_norm"] = df["home_team"].astype(str).str.strip().str.upper()
    df["away_team_norm"] = df["away_team"].astype(str).str.strip().str.upper()

    t_code_u = t_code.upper()
    t_full_u = t_full.upper()

    # Filtrar todos los partidos de la misma temporada antes de la fecha límite
    anio_partido = fecha_limite.year
    mask_season = (
        ((df["home_team_norm"].isin([t_code_u, t_full_u])) | (df["away_team_norm"].isin([t_code_u, t_full_u])))
        & (pd.to_datetime(df["fecha"], errors="coerce") < fecha_limite)
        & (pd.to_datetime(df["fecha"], errors="coerce").dt.year == anio_partido)
    )

    partidos_todos = df[mask_season].sort_values("fecha", ascending=False)
    partidos_recientes = partidos_todos.head(ventana)

    if len(partidos_todos) == 0:
        return {
            "victorias_recientes": 0.5,
            "win_rate_season": 0.5,
            "carreras_anotadas_avg": 4.5,
            "carreras_recibidas_avg": 4.5,
            "racha_actual": 0,
            "diferencial_carreras": 0,
            "total_juegos_season": 0,
            "wins_season": 0,
            "losses_season": 0,
            "season_record": "0-0",
        }

    # Estadísticas Ventana (L10)
    victorias_l10 = 0
    carreras_f_l10 = 0
    carreras_c_l10 = 0

    for _, p in partidos_recientes.iterrows():
        es_home = p["home_team_norm"] in [t_code_u, t_full_u]
        ganador_val = p.get("ganador", 0)
        if ganador_val is None:
            ganador_val = 0
        ganado = (ganador_val == 1) if es_home else (ganador_val == 0)
        if ganado:
            victorias_l10 += 1
        carreras_f_l10 += float(p.get("score_home", 0) if es_home else p.get("score_away", 0) or 0)
        carreras_c_l10 += float(p.get("score_away", 0) if es_home else p.get("score_home", 0) or 0)

    # Estadísticas Temporada
    victorias_season = 0
    for _, p in partidos_todos.iterrows():
        es_home = p["home_team_norm"] in [t_code_u, t_full_u]
        ganador_val = p.get("ganador", 0)
        if ganador_val is None:
            ganador_val = 0
        ganado = (ganador_val == 1) if es_home else (ganador_val == 0)
        if ganado:
            victorias_season += 1

    win_rate_season = victorias_season / len(partidos_todos)

    # Cálculo de racha
    racha = 0
    for _, p in partidos_recientes.iterrows():
        es_home = p["home_team_norm"] in [t_code_u, t_full_u]
        ganado = (p["ganador"] == 1) if es_home else (p["ganador"] == 0)
        if racha == 0:
            racha = 1 if ganado else -1
        elif (racha > 0 and ganado) or (racha < 0 and not ganado):
            racha += 1 if ganado else -1
        else:
            break

    n_win = len(partidos_recientes)
    return {
        "victorias_recientes": victorias_l10 / n_win if n_win > 0 else 0.5,
        "win_rate_season": win_rate_season,
        "carreras_anotadas_avg": carreras_f_l10 / n_win if n_win > 0 else 4.5,
        "carreras_recibidas_avg": carreras_c_l10 / n_win if n_win > 0 else 4.5,
        "racha_actual": racha,
        "diferencial_carreras": (carreras_f_l10 - carreras_c_l10) / n_win if n_win > 0 else 0,
        "total_juegos_season": len(partidos_todos),
        "wins_season": victorias_season,
        "losses_season": len(partidos_todos) - victorias_season,
        "season_record": f"{victorias_season}-{len(partidos_todos) - victorias_season}",
    }


def verificar_milestone_reentrenamiento():
    """
    Comprueba si el total de registros en historico_real alcanzó el próximo milestone
    de la temporada (486 / 972 / 1458 / 1944 / 2430).
    Devuelve (should_train: bool, total: int, next_milestone: int | None).
    """
    import sqlite3
    from datetime import datetime
    from zoneinfo import ZoneInfo

    MILESTONES_TEMPORADA = [486, 972, 1458, 1944, 2430]
    temporada_objetivo = int(os.getenv("TRAINING_SEASON_YEAR", datetime.now(ZoneInfo("America/New_York")).year))

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS metadata_entrenamiento (
                   key TEXT PRIMARY KEY,
                   value TEXT
               )"""
        )
        total = conn.execute(
            "SELECT COUNT(*) FROM historico_real WHERE substr(fecha,1,4) = ?",
            (str(temporada_objetivo),),
        ).fetchone()[0]

        row = conn.execute("SELECT value FROM metadata_entrenamiento WHERE key = 'last_retrain_milestone'").fetchone()
        try:
            last_milestone = int(row[0]) if row and row[0] else 0
        except Exception:
            last_milestone = 0

    next_milestone = next((m for m in MILESTONES_TEMPORADA if m > last_milestone), None)

    if next_milestone is None:
        return False, total, None

    should_train = total >= next_milestone
    return should_train, total, next_milestone


def registrar_milestone_cumplido(milestone):
    """Registra que el milestone fue procesado (con o sin mejora de accuracy)."""
    import sqlite3
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS metadata_entrenamiento (
                   key TEXT PRIMARY KEY,
                   value TEXT
               )"""
        )
        conn.execute(
            "INSERT OR REPLACE INTO metadata_entrenamiento (key, value) VALUES (?, ?)",
            ("last_retrain_milestone", str(milestone)),
        )
        conn.commit()
    print(f"🎯 Milestone {milestone}/2430 registrado como completado.")


def extraer_features_hibridas(row, df_historico=None, hacer_scraping=False, session_cache=None):
    """Extrae features combinando tendencias temporales y scraping"""
    from mlb_stats_api_client import obtener_fecha_ayer, obtener_stats_completas_api

    features = {}

    # 1. TENDENCIAS TEMPORALES
    if df_historico is not None:
        fecha_dt = pd.to_datetime(row["fecha"])

        trend_h = calcular_tendencias_equipo(df_historico, row["home_team"], fecha_dt, ventana=10)
        trend_a = calcular_tendencias_equipo(df_historico, row["away_team"], fecha_dt, ventana=10)

        features["home_win_rate_10"] = trend_h.get("victorias_recientes", 0.5)
        features["home_win_rate_season"] = trend_h.get("win_rate_season", 0.5)
        features["home_racha"] = trend_h.get("racha_actual", 0)
        features["home_runs_avg"] = trend_h.get("carreras_anotadas_avg", 4.5)
        features["home_runs_diff"] = trend_h.get("diferencial_carreras", 0)
        features["home_season_record"] = f"{trend_h.get('wins_season', 0)}-{trend_h.get('losses_season', 0)}"

        features["away_win_rate_10"] = trend_a.get("victorias_recientes", 0.5)
        features["away_win_rate_season"] = trend_a.get("win_rate_season", 0.5)
        features["away_racha"] = trend_a.get("racha_actual", 0)
        features["away_runs_avg"] = trend_a.get("carreras_anotadas_avg", 4.5)
        features["away_runs_diff"] = trend_a.get("diferencial_carreras", 0)
        features["away_season_record"] = f"{trend_a.get('wins_season', 0)}-{trend_a.get('losses_season', 0)}"

    # 2. PETICIONES A LA API DE MLB EN LUGAR DE SCRAPING DE BASEBALL-REFERENCE
    if hacer_scraping:
        if isinstance(row["fecha"], str):
            fecha_str = row["fecha"][:10]
        else:
            fecha_str = row["fecha"].strftime("%Y-%m-%d")

        ayer_str = obtener_fecha_ayer(fecha_str)
        year_val = safe_int(row["year"])
        if year_val == 0:
            try:
                year_val = int(fecha_str[:4])
            except Exception:
                year_val = datetime.now().year

        api_features = obtener_stats_completas_api(
            home_team=row["home_team"],
            away_team=row["away_team"],
            home_pitcher=row["home_pitcher"],
            away_pitcher=row["away_pitcher"],
            year=year_val,
            up_to_date=ayer_str
        )
        features.update(api_features)

    features["year"] = row["year"]

    return features


# ============================================================================
# EJECUCIÓN DE UTILIDADES
# ============================================================================


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python mlb_utils.py <comando> [argumentos]")
        print("\nComandos disponibles:")
        print("  accuracy [dias]           - Analizar accuracy del modelo")
        print("  equipo <codigo> [n]       - Reporte de un equipo")
        print("  limpiar_cache             - Limpiar caché de features")
        print("  compactar                 - Compactar base de datos")
        print("  verificar                 - Verificar integridad DB")
        print("  exportar_pred [dias]      - Exportar predicciones a CSV")
        print("  exportar_real [dias]      - Exportar resultados a CSV")
        print("  estado                    - Verificar estado del sistema")
        sys.exit(0)

    comando = sys.argv[1].lower()

    if comando == "accuracy":
        dias = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        analizar_accuracy_historico(dias)

    elif comando == "equipo":
        if len(sys.argv) < 3:
            print("❌ Debes especificar el código del equipo")
        else:
            codigo = sys.argv[2].upper()
            n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
            generar_reporte_equipos(codigo, n)

    elif comando == "limpiar_cache":
        limpiar_cache()

    elif comando == "compactar":
        compactar_base_datos()

    elif comando == "verificar":
        verificar_integridad_db()

    elif comando == "exportar_pred":
        dias = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        exportar_predicciones_csv(dias=dias)

    elif comando == "exportar_real":
        dias = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        exportar_resultados_csv(dias=dias)

    elif comando == "estado":
        verificar_estado_modelo()

    else:
        print(f"❌ Comando desconocido: {comando}")
