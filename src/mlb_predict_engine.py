"""
Motor de Predicción MLB V3.5 - REFACTORIZADO
Usa módulos centralizados para evitar duplicación de código
"""

import json
import os
import sqlite3
import time
import warnings

import pandas as pd
import xgboost as xgb

# Importar módulos centralizados
from mlb_config import DB_PATH, MODELO_PATH, get_team_name
from mlb_feature_engineering import (
    calcular_estadisticas_agregadas,
    calcular_super_features,
    detectar_outliers,
)

# Importar extracción de features del módulo de entrenamiento
from train_model_hybrid_actions import extraer_features_hibridas, normalizar_texto

warnings.filterwarnings("ignore")


# ============================================================================
# FUNCIONES DE SOPORTE
# ============================================================================


def obtener_nivel_confianza(prob_pct):
    """Determina el nivel de confianza basado en la probabilidad"""
    confianza = max(prob_pct, 100 - prob_pct) / 100
    if confianza > 0.70:
        return "MUY ALTA"
    if confianza > 0.60:
        return "ALTA"
    if confianza > 0.55:
        return "MODERADA"
    return "BAJA (Partido muy parejo)"


# ============================================================================
# MOTOR DE PREDICCIÓN
# ============================================================================


def predecir_juego(
    home_team,
    away_team,
    home_pitcher,
    away_pitcher,
    year=2026,
    modo_auto=False,
    fecha_partido=None,
    hacer_scraping=True,
    guardar_db=True,
    debug=False,
):
    """
    Predice el resultado de un juego de MLB

    Args:
        home_team: Código del equipo local
        away_team: Código del equipo visitante
        home_pitcher: Nombre del lanzador abridor local
        away_pitcher: Nombre del lanzador abridor visitante
        year: Año para scraping de estadísticas
        modo_auto: Si es True, suprime algunos prints
        hacer_scraping: Si es False, solo usa features temporales (sin HTTP a Baseball-Ref)

    Returns:
        Dict con resultado de la predicción o None si hay error
    """
    debug_info = {
        "enabled": bool(debug),
        "stages": [],
        "error": None,
    }

    def _debug_stage(name, start_ts, ok=True, extra=None):
        if not debug:
            return
        stage = {
            "name": name,
            "ok": ok,
            "elapsed_ms": round((time.perf_counter() - start_ts) * 1000, 2),
        }
        if extra is not None:
            stage["extra"] = extra
        debug_info["stages"].append(stage)

    total_start = time.perf_counter()

    # 1. Validar que existe el modelo
    stage_start = time.perf_counter()
    if not os.path.exists(MODELO_PATH):
        print(f"❌ Error: No existe el modelo en {MODELO_PATH}")
        _debug_stage("model_exists", stage_start, ok=False)
        debug_info["error"] = f"No existe el modelo en {MODELO_PATH}"
        _debug_stage("total", total_start, ok=False)
        if debug:
            return {"error": debug_info["error"], "_debug": debug_info}
        return None
    _debug_stage("model_exists", stage_start)

    # 2. Cargar modelo
    try:
        stage_start = time.perf_counter()
        model = xgb.Booster()
        model.load_model(MODELO_PATH)
        expected_features = model.feature_names
        _debug_stage("model_load", stage_start)
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        _debug_stage("model_load", stage_start, ok=False, extra={"error": str(e)})
        debug_info["error"] = f"Error cargando modelo: {e}"
        _debug_stage("total", total_start, ok=False)
        if debug:
            return {"error": debug_info["error"], "_debug": debug_info}
        return None

    # 3. Normalizar nombres de lanzadores
    stage_start = time.perf_counter()
    p_home_clean = normalizar_texto(home_pitcher)
    p_away_clean = normalizar_texto(away_pitcher)
    _debug_stage("normalize_inputs", stage_start)

    # 4. Preparar datos del partido
    row_data = {
        "home_team": home_team,
        "away_team": away_team,
        "home_pitcher": home_pitcher,
        "away_pitcher": away_pitcher,
        "home_pitcher_clean": p_home_clean,
        "away_pitcher_clean": p_away_clean,
        "year": year,
        "fecha": fecha_partido or pd.Timestamp.now().strftime("%Y-%m-%d"),
    }

    try:
        # 5. Cargar histórico de partidos
        stage_start = time.perf_counter()
        with sqlite3.connect(DB_PATH) as conn:
            try:
                df_historico = pd.read_sql("SELECT * FROM historico_real", conn)
            except Exception:
                df_historico = pd.DataFrame()
        _debug_stage(
            "load_historical",
            stage_start,
            extra={"rows": int(len(df_historico))},
        )

        # 6. Extracción de features híbrida (temporal + scraping)
        stage_start = time.perf_counter()
        features_dict = extraer_features_hibridas(
            row_data,
            df_historico=df_historico,
            hacer_scraping=hacer_scraping,
            session_cache={},
        )
        _debug_stage(
            "extract_features",
            stage_start,
            ok=bool(features_dict),
            extra={"count": int(len(features_dict or {})), "hacer_scraping": bool(hacer_scraping)},
        )

        if not features_dict:
            print("❌ No se pudieron extraer las features necesarias")
            debug_info["error"] = "No se pudieron extraer las features necesarias"
            _debug_stage("total", total_start, ok=False)
            if debug:
                return {"error": debug_info["error"], "_debug": debug_info}
            return None

        # 7. Aplicar super features usando módulo centralizado
        stage_start = time.perf_counter()
        features_dict = calcular_super_features(features_dict)
        _debug_stage("super_features", stage_start)

        # 8. Validar datos extraídos
        stage_start = time.perf_counter()
        warnings_data = detectar_outliers(features_dict)
        _debug_stage(
            "detect_outliers",
            stage_start,
            extra={"warnings": int(len(warnings_data or []))},
        )
        if warnings_data and not modo_auto:
            print("\n⚠️ Advertencias de datos:")
            for w in warnings_data:
                print(f"  {w}")

        # 9. Preparar DataFrame para predicción
        stage_start = time.perf_counter()
        X_df = pd.DataFrame([features_dict])
        X_df = X_df.reindex(columns=expected_features, fill_value=0)
        _debug_stage(
            "prepare_matrix",
            stage_start,
            extra={"feature_columns": int(len(X_df.columns))},
        )

        # 10. Realizar predicción
        stage_start = time.perf_counter()
        dmatrix = xgb.DMatrix(X_df)
        prob_home = model.predict(dmatrix)[0]
        _debug_stage("predict", stage_start)

        prob_home_pct = round(float(prob_home) * 100, 2)
        prob_away_pct = round(100 - prob_home_pct, 2)
        conf_label = obtener_nivel_confianza(prob_home_pct)

        # 11. Determinar ganador
        ganador_code = home_team if prob_home > 0.5 else away_team
        ganador_full = get_team_name(ganador_code)

        # 12. Calcular estadísticas agregadas para análisis adicional
        stage_start = time.perf_counter()
        stats_agregadas = calcular_estadisticas_agregadas(features_dict)
        _debug_stage("aggregate_stats", stage_start)

        # 13. OUTPUT VISUAL ENRIQUECIDO
        if not modo_auto:
            print("\n" + "=" * 75)
            print("   ⚾ MLB PREDICTOR V3.5 - ANÁLISIS ESTADÍSTICO")
            print("=" * 75)

            # Nombres reales de lanzadores
            h_p_name = features_dict.get("home_pitcher_name_real", home_pitcher)
            a_p_name = features_dict.get("away_pitcher_name_real", away_pitcher)

            print(f" Encuentro: {home_team} vs {away_team}")
            print(f" Temporada: {year} | Scraping: Baseball-Reference")

            print("\n📊 COMPARATIVA DE EQUIPOS:")
            print(
                f" 🏠  {home_team}: OPS: {features_dict.get('home_team_OPS', 0):.3f} | Bullpen WHIP: {features_dict.get('home_bullpen_WHIP', 0):.3f}"
            )
            print(
                f" ✈️  {away_team}: OPS: {features_dict.get('away_team_OPS', 0):.3f} | Bullpen WHIP: {features_dict.get('away_bullpen_WHIP', 0):.3f}"
            )

            print("\n👤 LANZADORES ABRIDORES:")
            print(
                f" 🏠 {h_p_name}: ERA: {features_dict.get('home_starter_ERA', 0):.2f} | WHIP: {features_dict.get('home_starter_WHIP', 0):.3f} | SO9: {features_dict.get('home_starter_SO9', 0):.2f}"
            )
            print(
                f" ✈️  {a_p_name}: ERA: {features_dict.get('away_starter_ERA', 0):.2f} | WHIP: {features_dict.get('away_starter_WHIP', 0):.3f} | SO9: {features_dict.get('away_starter_SO9', 0):.2f}"
            )

            # Análisis de Bullpen
            print("\n🧱 ANÁLISIS DE BULLPEN:")
            print(
                f" 🏠 {home_team}: ERA: {features_dict.get('home_bullpen_ERA', 0):.3f} | WHIP: {features_dict.get('home_bullpen_WHIP', 0):.3f}"
            )
            print(
                f" ✈️  {away_team}: ERA: {features_dict.get('away_bullpen_ERA', 0):.3f} | WHIP: {features_dict.get('away_bullpen_WHIP', 0):.3f}"
            )

            d_era = features_dict.get("diff_bullpen_ERA", 0)
            print(f" 📊 Diferencial ERA: {d_era:+.2f}")

            # Tabla de Bateadores
            print("\n🔥 TOP 3 BATEADORES ANALIZADOS:")
            for label, team_key in [
                ("🏠 " + home_team, "home_top_3_batters_details"),
                ("✈️  " + away_team, "away_top_3_batters_details"),
            ]:
                print(f"\n {label}:")
                print(
                    f" {'Nombre':<22} | {'BA':<5} | {'OBP':<5} | {'SLG':<5} | {'OPS':<5} | {'HR':<3} | {'RBI'}"
                )
                print("-" * 75)
                for b in features_dict.get(team_key, []):
                    nombre = b.get("n", "Desconocido")
                    ba = b.get("ba", 0)
                    obp = b.get("obp", 0)
                    slg = b.get("slg", 0)
                    ops = b.get("ops", b.get("o", 0))
                    hr = b.get("hr", 0)
                    rbi = b.get("rbi", 0)
                    print(
                        f" {nombre:<22} | {ba:.3f} | {obp:.3f} | {slg:.3f} | {ops:.3f} | {int(hr):<3} | {int(rbi)}"
                    )

            # Tendencias recientes
            print("\n📈 TENDENCIAS RECIENTES (Últimos 10 juegos):")
            print(
                f" 🏠 {home_team}: Win Rate: {features_dict.get('home_win_rate_10', 0.5):.1%} | Racha: {features_dict.get('home_racha', 0):+d}"
            )
            print(
                f" ✈️  {away_team}: Win Rate: {features_dict.get('away_win_rate_10', 0.5):.1%} | Racha: {features_dict.get('away_racha', 0):+d}"
            )

            print("\n" + "=" * 75)
            print(f" 🏆 GANADOR PREDICHO: {ganador_full}")
            print("=" * 75)
            print(
                f" Probabilidades: {home_team} {prob_home_pct}% | {away_team} {prob_away_pct}%"
            )
            print(f" Confianza: {conf_label}")

            # Diagnóstico de super features
            print("\n🚀 DIAGNÓSTICO DE SUPER FEATURES:")
            s_neut = features_dict.get("super_neutralizacion_whip_ops", 0)
            s_res = features_dict.get("super_resistencia_era_ops", 0)
            s_muro = features_dict.get("super_muro_bullpen", 0)

            n_v = home_team if s_neut < 0 else away_team
            print(f" 🛡️ Neutralización: {s_neut:.4f} (Ventaja {n_v})")
            r_v = home_team if s_res < 0 else away_team
            print(f" 📉 Resistencia:    {s_res:.4f} (Ventaja {r_v})")
            m_v = home_team if s_muro < 0 else away_team
            print(f" 🧱 Muro Bullpen:   {s_muro:.4f} (Ventaja {m_v})")

            # Análisis agregado
            print("\n💡 ANÁLISIS COMPUESTO:")
            print(
                f" Ventaja Pitcheo: {stats_agregadas.get('pitching_advantage', 0):+.3f}"
            )
            print(
                f" Ventaja Bateo:   {stats_agregadas.get('batting_advantage', 0):+.3f}"
            )
            print(
                f" Ventaja Momentum: {stats_agregadas.get('momentum_advantage', 0):+.3f}"
            )
            print(
                f" Score Compuesto: {stats_agregadas.get('composite_advantage', 0):+.3f}"
            )

            print("=" * 75 + "\n")

        # 14. Guardar predicción en base de datos
        def _formatear_top_bateadores(feature_key):
            bateadores = []
            for b in features_dict.get(feature_key, []) or []:
                if not isinstance(b, dict):
                    continue
                bateadores.append(
                    {
                        "nombre": b.get("n", "N/A"),
                        "BA": float(b.get("ba", 0) or 0),
                        "OBP": float(b.get("obp", 0) or 0),
                        "SLG": float(b.get("slg", 0) or 0),
                        "OPS": float(b.get("ops", 0) or 0),
                        "HR": int(float(b.get("hr", 0) or 0)),
                        "RBI": int(float(b.get("rbi", 0) or 0)),
                    }
                )
            return bateadores

        stage_start = time.perf_counter()
        detalles = {
            "year_usado": int(year),
            "features_usadas": features_dict,
            "stats_agregadas": stats_agregadas,
            "stats_detalladas": {
                "home_pitcher": {
                    "nombre": features_dict.get("home_pitcher_name_real", home_pitcher),
                    "ERA": float(features_dict.get("home_starter_ERA", 0) or 0),
                    "WHIP": float(features_dict.get("home_starter_WHIP", 0) or 0),
                    "H9": float(features_dict.get("home_starter_H9", 0) or 0),
                    "SO9": float(features_dict.get("home_starter_SO9", 0) or 0),
                    "W": int(float(features_dict.get("home_starter_W", 0) or 0)),
                    "L": int(float(features_dict.get("home_starter_L", 0) or 0)),
                },
                "away_pitcher": {
                    "nombre": features_dict.get("away_pitcher_name_real", away_pitcher),
                    "ERA": float(features_dict.get("away_starter_ERA", 0) or 0),
                    "WHIP": float(features_dict.get("away_starter_WHIP", 0) or 0),
                    "H9": float(features_dict.get("away_starter_H9", 0) or 0),
                    "SO9": float(features_dict.get("away_starter_SO9", 0) or 0),
                    "W": int(float(features_dict.get("away_starter_W", 0) or 0)),
                    "L": int(float(features_dict.get("away_starter_L", 0) or 0)),
                },
                "home_batters": _formatear_top_bateadores("home_top_3_batters_details"),
                "away_batters": _formatear_top_bateadores("away_top_3_batters_details"),
            },
        }
        _debug_stage("assemble_details", stage_start)

        detalles_json = json.dumps(detalles)

        db_data = {
            "fecha": row_data["fecha"],
            "home_team": home_team,
            "away_team": away_team,
            "home_pitcher": home_pitcher,
            "away_pitcher": away_pitcher,
            "prob_home": prob_home_pct,
            "prob_away": prob_away_pct,
            "prediccion": ganador_code,
            "confianza": conf_label,
            "tipo": "AUTOMATICO" if modo_auto else "MANUAL",
            "detalles": detalles_json,
        }

        resultado_data = {**db_data, "detalles": detalles}

        if guardar_db:
            stage_start = time.perf_counter()
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""CREATE TABLE IF NOT EXISTS predicciones_historico
                               (fecha TEXT, home_team TEXT, away_team TEXT, home_pitcher TEXT,
                                away_pitcher TEXT, prob_home REAL, prob_away REAL,
                                prediccion TEXT, confianza TEXT, tipo TEXT, detalles TEXT)""")
                
                # Check if detalles column exists, if not, add it
                cursor = conn.execute("PRAGMA table_info(predicciones_historico)")
                columns = [col[1] for col in cursor.fetchall()]
                if 'detalles' not in columns:
                    conn.execute("ALTER TABLE predicciones_historico ADD COLUMN detalles TEXT")

                # Reemplazo por juego para evitar duplicados y mantener la corrida no destructiva.
                conn.execute(
                    """
                    DELETE FROM predicciones_historico
                    WHERE fecha = ? AND home_team = ? AND away_team = ?
                    """,
                    (row_data["fecha"], home_team, away_team),
                )
                pd.DataFrame([db_data]).to_sql(
                    "predicciones_historico", conn, if_exists="append", index=False
                )
            _debug_stage("db_write", stage_start)

        if debug:
            _debug_stage("total", total_start)
            resultado_data["_debug"] = debug_info

        return resultado_data

    except Exception as e:
        print(f"❌ Error crítico en motor: {e}")
        import traceback

        traceback.print_exc()
        debug_info["error"] = str(e)
        _debug_stage("total", total_start, ok=False)
        if debug:
            return {"error": str(e), "_debug": debug_info}
        return None


def ejecutar_flujo_diario():
    """Ejecuta predicciones automáticas para los juegos del día"""
    print("🚀 Iniciando Motor de Predicción Diario...")

    if not os.path.exists(DB_PATH):
        print("❌ Error: Base de datos no encontrada.")
        return

    run_source = os.getenv("RUN_SOURCE", "local").strip().lower()
    target_date = os.getenv("TARGET_DATE", "").strip()

    with sqlite3.connect(DB_PATH) as conn:
        try:
            if target_date:
                fecha_objetivo = target_date
            else:
                fecha_objetivo = conn.execute(
                    "SELECT MAX(fecha) FROM historico_partidos"
                ).fetchone()[0]

            if not fecha_objetivo:
                print("🔭 No hay juegos registrados en historico_partidos.")
                return

            df_hoy = pd.read_sql(
                "SELECT * FROM historico_partidos WHERE fecha = ?",
                conn,
                params=[fecha_objetivo],
            )

            # Backfill manual: si se fuerza TARGET_DATE y no existe en historico_partidos,
            # intentamos reconstruir desde historico_real para no perder esa jornada.
            if df_hoy.empty and target_date:
                df_hoy = pd.read_sql(
                    """
                    SELECT
                        fecha,
                        home_team,
                        away_team,
                        home_pitcher,
                        away_pitcher,
                        COALESCE(year, 2026) AS year
                    FROM historico_real
                    WHERE fecha = ?
                    """,
                    conn,
                    params=[fecha_objetivo],
                )

            conn.execute(
                """CREATE TABLE IF NOT EXISTS sync_control (
                           dataset TEXT,
                           source TEXT,
                           fecha TEXT,
                           updated_at TEXT,
                           PRIMARY KEY(dataset, source)
                       )"""
            )
            conn.commit()
        except Exception:
            print("🔭 Tabla 'historico_partidos' no encontrada.")
            return

    if df_hoy.empty:
        if target_date:
            print(
                f"🔭 No hay juegos registrados para TARGET_DATE={fecha_objetivo} en historico_partidos ni historico_real."
            )
        else:
            print("🔭 No hay juegos registrados para hoy.")
        return

    print(f"📅 Se encontraron {len(df_hoy)} juegos para hoy\n")

    resultados = []
    for idx, row in df_hoy.iterrows():
        print(
            f"Procesando juego {idx + 1}/{len(df_hoy)}: {row['away_team']} @ {row['home_team']}"
        )

        try:
            resultado = predecir_juego(
                row["home_team"],
                row["away_team"],
                row["home_pitcher"],
                row["away_pitcher"],
                year=row.get("year", 2026),
                modo_auto=True,
                fecha_partido=row.get("fecha", fecha_objetivo),
                hacer_scraping=True
            )
        except Exception as e:
            print(f"⚠️ Error en scraping detallado para {row['away_team']} @ {row['home_team']}: {e}")
            resultado = None
            
        if not resultado:
            print("⚠️ Reintentando en modo de datos temporales (hacer_scraping=False)...")
            try:
                resultado = predecir_juego(
                    row["home_team"],
                    row["away_team"],
                    row["home_pitcher"],
                    row["away_pitcher"],
                    year=row.get("year", 2026),
                    modo_auto=True,
                    fecha_partido=row.get("fecha", fecha_objetivo),
                    hacer_scraping=False
                )
            except Exception as e:
                print(f"❌ Error en modo temporal para {row['away_team']} @ {row['home_team']}: {e}")
                resultado = None

        if resultado:
            resultados.append(resultado)
            print(
                f"✅ Predicción: {resultado['prediccion']} (Confianza: {resultado['confianza']})\n"
            )
        else:
            print("❌ Error total en predicción: No se pudo predecir de ninguna forma\n")

    if resultados:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """INSERT INTO sync_control (dataset, source, fecha, updated_at)
                           VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                           ON CONFLICT(dataset, source)
                           DO UPDATE SET fecha = excluded.fecha,
                                         updated_at = CURRENT_TIMESTAMP""",
                ("predictions_today", run_source, fecha_objetivo),
            )
            conn.commit()
    else:
        print(
            "⚠️ No se actualizará sync_control para predictions_today: no hubo predicciones exitosas."
        )

    print(
        f"\n✅ Proceso completado: {len(resultados)}/{len(df_hoy)} predicciones exitosas"
    )


if __name__ == "__main__":
    ejecutar_flujo_diario()
