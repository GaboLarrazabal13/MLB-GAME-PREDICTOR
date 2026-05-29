"""
MLB Production ML Trainer - V4.0 (PRODUCTION SYSTEM)
Implementa la Estrategia B: Reentrenamiento completo sobre datos históricos (2022-2025)
más los nuevos partidos de la temporada 2026, aplicando ponderación por recencia (Sample Weights).
Integra MLflow SQLite persistente para registrar los experimentos del pipeline de producción.
"""

import os
import pickle
import shutil
import sqlite3
import sys
import time
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Asegurar importación de módulos centralizados
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_config import (
    CACHE_PATH,
    DB_PATH,
    MODEL_CONFIG,
    MODELO_BACKUP,
    MODELO_PATH,
    SCRAPING_FEATURES,
    SUPER_FEATURES,
    TEAM_CODE_TO_NAME,
    TEMPORAL_FEATURES,
    get_team_code,
)
from mlb_feature_engineering import calcular_super_features
from mlb_utils import alinear_features_entrenamiento, extraer_features_hibridas

# MLflow SQLite Tracking URI in the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")

def cargar_dataset_produccion():
    """
    Carga todos los partidos históricos (2022-2025) y los de la temporada 2026,
    fusionando sus estadísticas y features en una sola matriz sin leakage de datos.
    """
    print("\n🛠️ Cargando y estructurando dataset de producción (2022-2026)...")

    with sqlite3.connect(DB_PATH) as conn:
        # 1. Cargar todos los partidos ordenados cronológicamente
        df_juegos = pd.read_sql(
            "SELECT * FROM historico_real ORDER BY fecha ASC",
            conn
        )

        # 2. Cargar todas las features precalculadas (2022-2025)
        df_features_db = pd.read_sql(
            "SELECT * FROM features_juegos",
            conn
        )

    if df_juegos.empty:
        raise ValueError("❌ La tabla historico_real está vacía. No hay datos para entrenar.")

    print(f"📊 Partidos totales cargados: {len(df_juegos)}")

    df_juegos["fecha"] = pd.to_datetime(df_juegos["fecha"])
    df_juegos["year"] = df_juegos["year"].astype(int)
    df_juegos["ganador"] = df_juegos["ganador"].astype(int)

    # 3. Mapear las features precalculadas indexadas por game_id para un acceso rápido O(1)
    features_db_dict = {row["game_id"]: row for _, row in df_features_db.iterrows()}

    # 4. Intentar cargar el caché local para los partidos de la temporada 2026
    cache_2026 = {}
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f_pkl:
                cache_data = pickle.load(f_pkl)
                X_list = cache_data.get("X_list", [])
                for x in X_list:
                    if x.get("game_id"):
                        cache_2026[x["game_id"]] = x
            print(f"📦 Caché recuperado con {len(cache_2026)} juegos ya procesados.")
        except Exception:
            print("⚠️ Caché no encontrado o corrupto para juegos nuevos.")

    # 5. Fusionar features base (SCRAPING_FEATURES) para cada juego
    print("   🧱 Fusionando features base de partidos...")
    base_feats_list = []

    session_cache = {}

    for idx, row in df_juegos.iterrows():
        g_id = row["game_id"]
        yr = row["year"]

        # Si el partido es de 2022-2025, cargamos de features_juegos en la DB
        if yr < 2026 and g_id in features_db_dict:
            feat_data = features_db_dict[g_id]
            base_feats_list.append({col: feat_data.get(col, 0.0) for col in SCRAPING_FEATURES})
        # Si es de 2026, cargamos del caché o extraemos usando la API
        else:
            if g_id in cache_2026:
                feat_data = cache_2026[g_id]
                base_feats_list.append({col: feat_data.get(col, 0.0) for col in SCRAPING_FEATURES})
            else:
                # Fallback: extraer features vía API en tiempo real
                try:
                    f = extraer_features_hibridas(
                        row,
                        df_historico=df_juegos,
                        hacer_scraping=True,
                        session_cache=session_cache,
                    )
                    base_feats_list.append({col: f.get(col, 0.0) for col in SCRAPING_FEATURES})
                    # Guardar en caché para futuras ejecuciones
                    f["game_id"] = g_id
                    cache_2026[g_id] = f
                except Exception as e:
                    print(f"⚠️ Error extrayendo features para el juego {g_id}: {e}")
                    # Completar con ceros en caso de fallo crítico
                    base_feats_list.append({col: 0.0 for col in SCRAPING_FEATURES})

    # Guardar caché actualizado de 2026
    if cache_2026:
        try:
            with open(CACHE_PATH, "wb") as f_pkl:
                pickle.dump({"X_list": list(cache_2026.values()), "y_list": [], "indices": []}, f_pkl)
        except Exception as e:
            print(f"❌ Error guardando caché actualizado: {e}")

    df_base_feats = pd.DataFrame(base_feats_list)
    df_dataset = pd.concat([df_juegos.reset_index(drop=True), df_base_feats.reset_index(drop=True)], axis=1)

    # 6. Calcular Features Temporales y ELO (O(N) cronológico sin leakage)
    print("   ⏱️ Calculando tendencias temporales y ELO mediante tracking cronológico acumulativo...")
    temp_feats_list = []
    team_games = {}

    # Inicializar ELOs
    elo_dict = {code: 1500.0 for code in TEAM_CODE_TO_NAME.keys()}
    K = 20.0
    HOME_ADVANTAGE = 24.0

    for idx, row in df_dataset.iterrows():
        h_team = row["home_team"]
        a_team = row["away_team"]
        yr = row["year"]
        sc_h = float(row["score_home"] or 0)
        sc_a = float(row["score_away"] or 0)
        gan = int(row["ganador"] or 0)

        h_code = get_team_code(h_team)
        a_code = get_team_code(a_team)

        hist_h = team_games.get((h_team, yr), [])
        hist_a = team_games.get((a_team, yr), [])

        # Home Team metrics
        if not hist_h:
            h_wr10, h_wrs, h_strk, h_ravg, h_rdiff = 0.5, 0.5, 0, 4.5, 0.0
        else:
            rec_h = hist_h[-10:]
            h_wr10 = sum(1 for g in rec_h if g["ganado"]) / len(rec_h)
            h_wrs = sum(1 for g in hist_h if g["ganado"]) / len(hist_h)
            h_ravg = sum(g["runs_sc"] for g in rec_h) / len(rec_h)
            h_rdiff = (sum(g["runs_sc"] for g in rec_h) - sum(g["runs_al"] for g in rec_h)) / len(rec_h)

            streak = 0
            ultimo_ganado = hist_h[-1]["ganado"]
            for g in reversed(hist_h):
                if g["ganado"] == ultimo_ganado:
                    streak += 1
                else:
                    break
            h_strk = streak if ultimo_ganado else -streak

        # Away Team metrics
        if not hist_a:
            a_wr10, a_wrs, a_strk, a_ravg, a_rdiff = 0.5, 0.5, 0, 4.5, 0.0
        else:
            rec_a = hist_a[-10:]
            a_wr10 = sum(1 for g in rec_a if g["ganado"]) / len(rec_a)
            a_wrs = sum(1 for g in hist_a if g["ganado"]) / len(hist_a)
            a_ravg = sum(g["runs_sc"] for g in rec_a) / len(rec_a)
            a_rdiff = (sum(g["runs_sc"] for g in rec_a) - sum(g["runs_al"] for g in rec_a)) / len(rec_a)

            streak = 0
            ultimo_ganado = hist_a[-1]["ganado"]
            for g in reversed(hist_a):
                if g["ganado"] == ultimo_ganado:
                    streak += 1
                else:
                    break
            a_strk = streak if ultimo_ganado else -streak

        # Obtener ELO antes de que inicie el juego (sin leakage)
        if h_code and a_code:
            h_elo_before = elo_dict[h_code]
            a_elo_before = elo_dict[a_code]
            diff_elo_before = h_elo_before - a_elo_before
        else:
            h_elo_before = 1500.0
            a_elo_before = 1500.0
            diff_elo_before = 0.0

        temp_feats_list.append({
            "home_win_rate_10": h_wr10,
            "home_win_rate_season": h_wrs,
            "home_racha": h_strk,
            "home_runs_avg": h_ravg,
            "home_runs_diff": h_rdiff,
            "away_win_rate_10": a_wr10,
            "away_win_rate_season": a_wrs,
            "away_racha": a_strk,
            "away_runs_avg": a_ravg,
            "away_runs_diff": a_rdiff,
            "home_elo": h_elo_before,
            "away_elo": a_elo_before,
            "diff_elo": diff_elo_before,
        })

        # Actualizar historial
        h_win = (gan == 1)
        a_win = (gan == 0)

        if (h_team, yr) not in team_games:
            team_games[(h_team, yr)] = []
        team_games[(h_team, yr)].append({"ganado": h_win, "runs_sc": sc_h, "runs_al": sc_a})

        if (a_team, yr) not in team_games:
            team_games[(a_team, yr)] = []
        team_games[(a_team, yr)].append({"ganado": a_win, "runs_sc": sc_a, "runs_al": sc_h})

        # Actualizar ELO para futuros partidos
        if h_code and a_code:
            e_home = 1.0 / (10.0 ** (-(h_elo_before + HOME_ADVANTAGE - a_elo_before) / 400.0) + 1.0)
            s_home = 1.0 if gan == 1 else 0.0
            elo_dict[h_code] = h_elo_before + K * (s_home - e_home)
            elo_dict[a_code] = a_elo_before + K * ((1.0 - s_home) - (1.0 - e_home))

    df_temp = pd.DataFrame(temp_feats_list)
    df_dataset = pd.concat([df_dataset.reset_index(drop=True), df_temp.reset_index(drop=True)], axis=1)

    # 7. Calcular Super Features
    print("   🧱 Calculando Super Features...")
    super_feats_list = []
    for idx, row in df_dataset.iterrows():
        row_dict = row.to_dict()
        sf = calcular_super_features(row_dict)
        super_feats_list.append({
            "super_neutralizacion_whip_ops": sf.get("super_neutralizacion_whip_ops", 0.0),
            "super_resistencia_era_ops": sf.get("super_resistencia_era_ops", 0.0),
            "super_muro_bullpen": sf.get("super_muro_bullpen", 0.0)
        })

    df_sf = pd.DataFrame(super_feats_list)
    df_dataset = pd.concat([df_dataset.reset_index(drop=True), df_sf.reset_index(drop=True)], axis=1)

    # Alinear columnas finales de features
    features_esperadas = TEMPORAL_FEATURES + SCRAPING_FEATURES + SUPER_FEATURES
    X = df_dataset[features_esperadas].fillna(0)
    y = df_dataset["ganador"].values

    # Inyectar metadatos para ponderación de pesos
    X["year"] = df_dataset["year"].values

    return X, y

def evaluar_modelo(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "LogLoss": log_loss(y_true, y_prob),
        "ROC_AUC": roc_auc_score(y_true, y_prob)
    }

def ejecutar_reentrenamiento():
    """Ejecuta el reentrenamiento completo con optimización de hiperparámetros y pesos."""
    mlflow.set_experiment("MLB-Predictor-Production-Experiment")

    X, y = cargar_dataset_produccion()

    # 1. Calcular pesos de entrenamiento (Sample Weights)
    # Asignar peso 1.5 a la temporada 2026 para capturar tendencias recientes, y 1.0 a la historia
    sample_weights = np.where(X["year"] == 2026, 1.5, 1.0)

    # Quitar columna year para que no se use como feature en XGBoost
    X_train_data = X.drop(columns=["year"])

    # 2. Dividir y Escalar datos
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X_train_data, y, sample_weights, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    accuracy_champion = 0
    model_champion = None

    # 3. Cargar Champion Actual
    if os.path.exists(MODELO_PATH):
        try:
            model_champion = XGBClassifier()
            model_champion.load_model(MODELO_PATH)
            y_pred_champ = model_champion.predict(X_test_scaled)
            accuracy_champion = accuracy_score(y_test, y_pred_champ)
            print(f"\n🏆 Accuracy de Champion en Test: {accuracy_champion:.2%}")
        except Exception as e:
            print(f"⚠️ Error cargando Champion previo: {e}")

    # 4. Optimización de Hiperparámetros con Optuna
    print("\n🔎 Iniciando Optimización Bayesiana de Optuna (35 trials) con pesos por recencia...")

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
            "random_state": 42,
            "n_jobs": -1
        }

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in cv.split(X_train_scaled, y_train):
            X_tr, X_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            w_tr = w_train[train_idx]

            model = XGBClassifier(**params)
            # Aplicar sample_weight en el ajuste
            model.fit(X_tr, y_tr, sample_weight=w_tr)

            y_pred = model.predict(X_val)
            scores.append(accuracy_score(y_val, y_pred))

        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=35)

    best_params = study.best_params
    print(f"   🏆 Optuna completado. Mejores parámetros: {best_params}")

    # 5. Entrenar Challenger final con toda la data y pesos
    print("\n🚀 Entrenando Challenger con mejores hiperparámetros y pesos...")
    model_challenger = XGBClassifier(
        **best_params,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    with mlflow.start_run(run_name="V4.0_Production_Retraining"):
        model_challenger.fit(X_train_scaled, y_train, sample_weight=w_train)

        # Evaluar
        y_pred = model_challenger.predict(X_test_scaled)
        y_prob = model_challenger.predict_proba(X_test_scaled)[:, 1]
        metrics = evaluar_modelo(y_test, y_pred, y_prob)

        # Registrar en MLflow
        mlflow.log_param("tuning_method", "Optuna")
        mlflow.log_param("reconstruction_strategy", "Option_B_Full_Retrain")
        mlflow.log_param("weight_season_2026", 1.5)
        for key, val in best_params.items():
            mlflow.log_param(key, val)
        for key, val in metrics.items():
            mlflow.log_metric(key, val)

        mlflow.xgboost.log_model(model_challenger, artifact_path="model")

        mlflow.set_tag("version", "4.0_prod")
        mlflow.set_tag("phase", "Production upgrade Challenger")

        accuracy_challenger = metrics["Accuracy"]
        print(f"📈 Accuracy de Challenger en Test: {accuracy_challenger:.2%}")

        # 6. Promoción Champion vs Challenger
        if accuracy_challenger >= accuracy_champion:
            print("🟢 Challenger SUPERÓ o IGUALÓ al Champion anterior. Promocionando a Producción...")
            if os.path.exists(MODELO_PATH):
                shutil.copy(MODELO_PATH, MODELO_BACKUP)

            # Guardar el modelo ganador de producción
            model_challenger.get_booster().save_model(MODELO_PATH)
            print(f"✅ ¡Modelo ganador '{MODELO_PATH}' guardado exitosamente!")

            # Registrar milestone
            try:
                from mlb_utils import registrar_milestone_cumplido, verificar_milestone_reentrenamiento
                _, _, next_m = verificar_milestone_reentrenamiento()
                if next_m:
                    registrar_milestone_cumplido(next_m)
            except Exception as e:
                print(f"⚠️ No se pudo registrar el milestone: {e}")
        else:
            print("🔴 Champion anterior retuvo un mejor rendimiento en test. Manteniendo Champion actual en producción.")

if __name__ == "__main__":
    ejecutar_reentrenamiento()
