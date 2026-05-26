"""
Script de Entrenamiento ML Híbrido OPTIMIZADO para Predicción MLB - VERSIÓN V3.5 REFACTORIZADA
LÓGICA HÍBRIDA INTELIGENTE:
- Partidos antiguos (2022-2024): Solo features temporales (CSV)
- Partidos recientes (2026): Features temporales + scraping
Sistema de bloques para evitar rate limiting
Cache incremental para entrenar por etapas
REFACTORIZACIÓN: Código centralizado, sin duplicación, mejor mantenibilidad
"""

import os
import pickle
import random
import re
import shutil
import sqlite3
import time
import unicodedata
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Importar módulos centralizados
from mlb_config import (
    CACHE_PATH,
    DB_PATH,
    MODEL_CONFIG,
    MODELO_BACKUP,
    MODELO_PATH,
    SCRAPING_CONFIG,
    SCRAPING_FEATURES,
    SUPER_FEATURES,
    TEMPORAL_FEATURES,
    get_team_code,
)
from mlb_feature_engineering import calcular_super_features
from mlb_stats_api_client import obtener_stats_completas_api, obtener_fecha_ayer

warnings.filterwarnings("ignore")


def alinear_features_entrenamiento(X_new, model_actual=None):
    """Asegura un esquema estable de features para reentrenamiento incremental."""
    features_esperadas = TEMPORAL_FEATURES + SCRAPING_FEATURES + SUPER_FEATURES

    columnas_faltantes = [col for col in features_esperadas if col not in X_new.columns]
    if columnas_faltantes:
        print(f"ℹ️ Añadiendo columnas faltantes con 0 para mantener esquema estable: {columnas_faltantes}")
        for columna in columnas_faltantes:
            X_new[columna] = 0

    if model_actual is None:
        return X_new

    feature_names_modelo = model_actual.get_booster().feature_names or []
    if not feature_names_modelo:
        return X_new

    columnas_faltantes_modelo = [col for col in feature_names_modelo if col not in X_new.columns]
    if columnas_faltantes_modelo:
        print(f"ℹ️ Añadiendo columnas faltantes requeridas por el modelo previo: {columnas_faltantes_modelo}")
        for columna in columnas_faltantes_modelo:
            X_new[columna] = 0

    columnas_extra_modelo = [col for col in X_new.columns if col not in feature_names_modelo]
    if columnas_extra_modelo:
        print(
            "ℹ️ Excluyendo columnas que no existen en el modelo previo para el "
            f"reentrenamiento incremental: {columnas_extra_modelo}"
        )

    return X_new.reindex(columns=feature_names_modelo, fill_value=0)


def normalizar_texto(texto):
    """Normaliza texto para comparaciones de nombres"""
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


# ============================================================================
# FUNCIONES DE TENDENCIAS TEMPORALES
# ============================================================================


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


# ============================================================================
# FUNCIONES DE CONTROL DE BASE DE DATOS
# ============================================================================


# Milestones fijos de la temporada regular: 2430 juegos / 5 cortes = 486 por bloque
MILESTONES_TEMPORADA = [486, 972, 1458, 1944, 2430]


def obtener_todos_los_juegos_temporada():
    """Devuelve todos los juegos de historico_real para la temporada actual, ordenados por fecha."""
    temporada_objetivo = int(os.getenv("TRAINING_SEASON_YEAR", datetime.now(ZoneInfo("America/New_York")).year))
    with sqlite3.connect(DB_PATH) as conn:
        df_real = pd.read_sql(
            "SELECT * FROM historico_real WHERE substr(fecha, 1, 4) = ? ORDER BY fecha",
            conn,
            params=[str(temporada_objetivo)],
        )

    if df_real.empty:
        return df_real

    df_real["home_team"] = df_real["home_team"].str.strip()
    df_real["away_team"] = df_real["away_team"].str.strip()
    return df_real


def verificar_milestone_reentrenamiento():
    """
    Comprueba si el total de registros en historico_real alcanzó el próximo milestone
    de la temporada (486 / 972 / 1458 / 1944 / 2430).
    Devuelve (should_train: bool, total: int, next_milestone: int | None).
    """
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
        print(f"✅ Todos los milestones de temporada completados. Total en BD: {total}")
        return False, total, None

    should_train = total >= next_milestone
    print(
        f"📊 Juegos temporada {temporada_objetivo}: {total} "
        f"| Último milestone: {last_milestone} "
        f"| Próximo milestone: {next_milestone} "
        f"| Entrenar: {should_train}"
    )
    return should_train, total, next_milestone


def registrar_milestone_cumplido(milestone):
    """Registra que el milestone fue procesado (con o sin mejora de accuracy)."""
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


# ============================================================================
# EXTRACCIÓN DE FEATURES HÍBRIDA
# ============================================================================


def extraer_features_hibridas(row, df_historico=None, hacer_scraping=False, session_cache=None):
    """Extrae features combinando tendencias temporales y scraping"""
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
        # Convertir fecha a string YYYY-MM-DD
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
# MOTOR DE REENTRENAMIENTO INCREMENTAL
# ============================================================================


def ejecutar_reentrenamiento_incremental(bloque_size=None, pausa_entre_bloques=None):
    """Ejecuta el proceso de reentrenamiento con datos nuevos"""

    if bloque_size is None:
        bloque_size = SCRAPING_CONFIG["bloque_size"]
    if pausa_entre_bloques is None:
        pausa_entre_bloques = SCRAPING_CONFIG["pausa_entre_bloques"]

    print("\n" + "=" * 80)
    print(" INICIANDO ACTUALIZACIÓN INCREMENTAL MLB V3.5")
    print("=" * 80)

    # 1. Verificar milestone antes de proceder
    should_train, _total_real, next_milestone = verificar_milestone_reentrenamiento()
    if not should_train:
        print("ℹ️ No se alcanzó ningún milestone de reentrenamiento. Saliendo.")
        return

    print(
        f"🎯 Milestone alcanzado: {next_milestone}/2430 — "
        f"Entrenando con todos los juegos de la temporada ({_total_real} registros)."
    )

    # 2. Carga de todos los juegos de la temporada
    df_nuevos = obtener_todos_los_juegos_temporada()

    if not df_nuevos.empty:
        df_nuevos["score_home"] = pd.to_numeric(df_nuevos["score_home"], errors="coerce").fillna(0)
        df_nuevos["score_away"] = pd.to_numeric(df_nuevos["score_away"], errors="coerce").fillna(0)
        df_nuevos["ganador"] = pd.to_numeric(df_nuevos["ganador"], errors="coerce").fillna(0).astype(int)

    print(f"Conteo total detectado: {len(df_nuevos)}")

    if not df_nuevos.empty:
        print(f"Rango de fechas: {df_nuevos['fecha'].min()} hasta {df_nuevos['fecha'].max()}")

    total_juegos = len(df_nuevos)
    df_nuevos["fecha"] = pd.to_datetime(df_nuevos["fecha"], errors="coerce")
    df_nuevos["year"] = df_nuevos["fecha"].apply(lambda x: x.year if pd.notna(x) else 0)

    # 2. Extracción de Features
    X_dict_list = []
    y_list = []

    # Cargar caché si existe
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f_pkl:
                cache_previo = pickle.load(f_pkl)
                X_dict_list = cache_previo.get("X_list", [])
                y_list = cache_previo.get("y_list", [])
            print(f"📦 Caché recuperado: {len(X_dict_list)} juegos ya procesados.")
        except Exception:
            print("⚠️ Cache corrupto o no encontrado, iniciando desde cero.")

    juegos_saltados = len(X_dict_list)
    df_para_procesar = df_nuevos.iloc[juegos_saltados:]

    session_cache = {}
    juegos_procesados_completos = []

    for i, (_, row) in enumerate(df_para_procesar.iterrows(), juegos_saltados + 1):
        if i % 25 == 0 or i == 1 or i == total_juegos:
            print(f"⏳ Procesando juego {i}/{total_juegos}... (Equipos en caché: {len(session_cache)})")

        try:
            f = extraer_features_hibridas(
                row,
                df_historico=df_nuevos,
                hacer_scraping=True,
                session_cache=session_cache,
            )
            X_dict_list.append(f)
            y_list.append(row["ganador"])
            juegos_procesados_completos.append(row)

            if i % bloque_size == 0 or i == total_juegos:
                print(f"\n💾 Actualizando caché ({i}/{total_juegos})...")
                try:
                    with open(CACHE_PATH, "wb") as f_pkl:
                        pickle.dump(
                            {"X_list": X_dict_list, "y_list": y_list, "indices": []},
                            f_pkl,
                        )
                except Exception as e:
                    print(f"❌ Error al escribir caché: {e}")

                if i % bloque_size == 0 and i < total_juegos:
                    print(f"🛡️ Pausa de seguridad: {pausa_entre_bloques}s...")
                    time.sleep(pausa_entre_bloques)

        except Exception as e:
            print(f"⚠️ Error en juego {row.get('game_id')}: {e}")
            continue

    # Guardado definitivo
    with open(CACHE_PATH, "wb") as f_pkl:
        pickle.dump({"X_list": X_dict_list, "y_list": y_list, "indices": []}, f_pkl)

    print(f"\n✅ Extracción finalizada. Procesados {len(X_dict_list)} juegos.")

    # 3. Preparar datos para entrenamiento
    X_new = pd.DataFrame(X_dict_list).fillna(0)
    y_new = np.array(y_list)

    # Aplicar super features usando módulo centralizado
    print("🛠️ Calculando super features...")
    for i in range(len(X_new)):
        row_dict = X_new.iloc[i].to_dict()
        updated_dict = calcular_super_features(row_dict)
        for key, val in updated_dict.items():
            if key in [
                "super_neutralizacion_whip_ops",
                "super_resistencia_era_ops",
                "super_muro_bullpen",
            ]:
                X_new.at[i, key] = val

    # El reentrenamiento solo puede consumir columnas numéricas.
    columnas_no_numericas = X_new.select_dtypes(exclude=[np.number]).columns.tolist()
    if columnas_no_numericas:
        print(f"ℹ️ Excluyendo columnas no numéricas del entrenamiento: {columnas_no_numericas}")
        X_new = X_new.drop(columns=columnas_no_numericas)

    # 4. Cargar modelo actual y alinear esquema de features
    accuracy_actual_en_nuevos = 0
    model_actual = None

    if os.path.exists(MODELO_PATH):
        try:
            model_actual = XGBClassifier()
            model_actual.load_model(MODELO_PATH)
        except Exception as e:
            print(f"⚠️ Error cargando modelo previo: {e}")

    X_new = alinear_features_entrenamiento(X_new, model_actual=model_actual)

    # 5. Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_new.fillna(0))
    X_final = pd.DataFrame(X_scaled, columns=X_new.columns)

    # 6. División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_final,
        y_new,
        test_size=MODEL_CONFIG["test_size"],
        random_state=MODEL_CONFIG["random_state"],
    )

    # 7. Evaluar modelo actual si existe
    if model_actual is not None:
        try:
            y_pred_old = model_actual.predict(X_test)
            accuracy_actual_en_nuevos = accuracy_score(y_test, y_pred_old)
            print(f"📊 Accuracy del modelo previo: {accuracy_actual_en_nuevos:.2%}")
        except Exception as e:
            print(f"⚠️ Error evaluando modelo previo: {e}")

    # 8. Optimización de hiperparámetros con Optuna
    print("🔎 Buscando la mejor combinación de hiperparámetros con Optuna...")

    xgb_model_param = model_actual.get_booster() if model_actual else None

    # Desactivar logs detallados de optuna para no inundar la consola
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 450),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
            "eval_metric": "logloss",
            "random_state": MODEL_CONFIG["random_state"],
            "n_jobs": -1
        }
        
        cv = StratifiedKFold(n_splits=MODEL_CONFIG["cv_folds"], shuffle=True, random_state=MODEL_CONFIG["random_state"])
        scores = []
        
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model = XGBClassifier(**params)
            
            # Intentar entrenar con el booster anterior si está disponible y es compatible
            try:
                if xgb_model_param is not None:
                    model.fit(X_tr, y_tr, xgb_model=xgb_model_param)
                else:
                    model.fit(X_tr, y_tr)
            except Exception:
                # Si falla el warm start, hacer fallback transparente
                model.fit(X_tr, y_tr)
                
            y_pred = model.predict(X_val)
            scores.append(accuracy_score(y_val, y_pred))
            
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=35)
    
    best_params = study.best_params
    print(f"🏆 Mejores parámetros encontrados por Optuna: {best_params}")
    print(f"📈 Mejor Accuracy en CV: {study.best_value:.2%}")
    
    # Entrenar el modelo final con los mejores parámetros
    model_nuevo = XGBClassifier(
        **best_params,
        eval_metric="logloss",
        random_state=MODEL_CONFIG["random_state"],
        n_jobs=-1
    )
    
    try:
        if xgb_model_param is not None:
            model_nuevo.fit(X_train, y_train, xgb_model=xgb_model_param)
        else:
            model_nuevo.fit(X_train, y_train)
    except Exception as e:
        if xgb_model_param is not None and "feature_names mismatch" in str(e):
            print(
                "⚠️ Warm-start incremental falló por incompatibilidad de "
                "features con el modelo previo. Reintentando entrenamiento "
                "sin xgb_model para mantener el pipeline estable..."
            )
            model_nuevo.fit(X_train, y_train)
        else:
            raise

    print(f"🏆 Mejores parámetros encontrados: {best_params}")

    # 9. Validación final
    y_pred_new = model_nuevo.predict(X_test)
    accuracy_nuevo = accuracy_score(y_test, y_pred_new)
    print(f"📈 Accuracy nueva versión (Optimizado): {accuracy_nuevo:.2%}")

    # 10. Guardar si hay mejora — en cualquier caso registrar el milestone
    if accuracy_nuevo >= accuracy_actual_en_nuevos:
        print("✅ MEJORA DETECTADA. Actualizando modelo oficial.")

        # Backup del modelo anterior
        if os.path.exists(MODELO_PATH):
            shutil.copy(MODELO_PATH, MODELO_BACKUP)

        model_nuevo.get_booster().save_model(MODELO_PATH)

        registrar_milestone_cumplido(next_milestone)

        if os.path.exists(CACHE_PATH):
            shutil.copy(CACHE_PATH, CACHE_PATH + ".bak")
            print("✅ Copia de seguridad del caché creada (.bak)")
    else:
        print("⚠️ No hubo mejora con los nuevos parámetros. Manteniendo versión previa.")
        registrar_milestone_cumplido(next_milestone)


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    ejecutar_reentrenamiento_incremental()
