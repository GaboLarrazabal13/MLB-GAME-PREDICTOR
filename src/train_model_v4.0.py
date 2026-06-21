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

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")

# Asegurar importación de módulos centralizados
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_config import (
    CACHE_PATH,
    DB_PATH,
    MODEL_CONFIG,
    MODELO_BACKUP,
    MODELO_LGBM_BACKUP,
    MODELO_LGBM_PATH,
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
        try:
            df_features_db = pd.read_sql("SELECT * FROM features_juegos", conn)
        except Exception:
            print("⚠️ Tabla features_juegos no encontrada; partidos históricos usarán ceros para scraping features.")
            df_features_db = pd.DataFrame()

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

        if yr < 2026:
            # Partidos históricos: usar features precalculadas o ceros si no están disponibles
            if g_id in features_db_dict:
                feat_data = features_db_dict[g_id]
                base_feats_list.append({col: feat_data.get(col, 0.0) for col in SCRAPING_FEATURES})
            else:
                base_feats_list.append({col: 0.0 for col in SCRAPING_FEATURES})
        else:
            # Partidos de 2026: intentar caché y luego API
            if g_id in cache_2026:
                feat_data = cache_2026[g_id]
                base_feats_list.append({col: feat_data.get(col, 0.0) for col in SCRAPING_FEATURES})
            else:
                try:
                    f = extraer_features_hibridas(
                        row,
                        df_historico=df_juegos,
                        hacer_scraping=True,
                        session_cache=session_cache,
                    )
                    base_feats_list.append({col: f.get(col, 0.0) for col in SCRAPING_FEATURES})
                    f["game_id"] = g_id
                    cache_2026[g_id] = f
                except Exception as e:
                    print(f"⚠️ Error extrayendo features para el juego {g_id}: {e}")
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
    # 2026 (temporada actual): peso 2.5 — patrones recientes son los más relevantes
    # 2025 (temporada anterior): peso 1.3 — contexto cercano útil
    # 2023-2024: peso 1.0 — base histórica de patrones
    sample_weights = np.where(X["year"] == 2026, 2.5, np.where(X["year"] == 2025, 1.3, 1.0))

    # Quitar columna year para que no se use como feature
    X_train_data = X.drop(columns=["year"])

    # 2. Dividir datos (sin escalar: CatBoost no lo requiere y el motor de predicción tampoco lo aplica)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X_train_data, y, sample_weights, test_size=0.20, random_state=42, stratify=y
    )

    accuracy_champion_ensemble = 0
    model_champion_cb = None
    model_champion_lgbm = None

    # 3. Cargar Champions actuales (CatBoost + LightGBM)
    if os.path.exists(MODELO_PATH):
        try:
            model_champion_cb = CatBoostClassifier()
            model_champion_cb.load_model(MODELO_PATH)
        except Exception as e:
            print(f"⚠️ Error cargando Champion CatBoost: {e}")

    if os.path.exists(MODELO_LGBM_PATH):
        try:
            model_champion_lgbm = joblib.load(MODELO_LGBM_PATH)
        except Exception as e:
            print(f"⚠️ Error cargando Champion LightGBM: {e}")

    if model_champion_cb:
        prob_cb = model_champion_cb.predict_proba(X_test)[:, 1]
        if model_champion_lgbm:
            prob_lgbm = model_champion_lgbm.predict_proba(X_test)[:, 1]
            prob_ensemble = (prob_cb + prob_lgbm) / 2
        else:
            prob_ensemble = prob_cb
        y_pred_champ = (prob_ensemble >= 0.5).astype(int)
        accuracy_champion_ensemble = accuracy_score(y_test, y_pred_champ)
        print(f"\n🏆 Accuracy de Champion Ensemble en Test: {accuracy_champion_ensemble:.2%}")

    # 4a. Optimización de Hiperparámetros con Optuna (CatBoost)
    print("\n🔎 [CatBoost] Optimización Bayesiana Optuna (30 trials)...")

    def objective_cb(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 150, 450),
            "depth": trial.suggest_int("depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_seed": 42,
            "thread_count": -1,
            "verbose": False,
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            w_tr = w_train[train_idx]
            m = CatBoostClassifier(**params)
            m.fit(X_tr, y_tr, sample_weight=w_tr)
            scores.append(accuracy_score(y_val, m.predict(X_val)))
        return np.mean(scores)

    study_cb = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_cb.optimize(objective_cb, n_trials=30)
    best_params_cb = study_cb.best_params
    print(f"   ✅ CatBoost mejores params: {best_params_cb}")

    # 4b. Optimización de Hiperparámetros con Optuna (LightGBM)
    print("\n🔎 [LightGBM] Optimización Bayesiana Optuna (30 trials)...")

    def objective_lgbm(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 80),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            w_tr = w_train[train_idx]
            m = LGBMClassifier(**params)
            m.fit(X_tr, y_tr, sample_weight=w_tr)
            scores.append(accuracy_score(y_val, m.predict(X_val)))
        return np.mean(scores)

    study_lgbm = optuna.create_study(direction="maximize")
    study_lgbm.optimize(objective_lgbm, n_trials=30)
    best_params_lgbm = study_lgbm.best_params
    print(f"   ✅ LightGBM mejores params: {best_params_lgbm}")

    # 5. Entrenar Challengers finales
    print("\n🚀 Entrenando Challengers CatBoost + LightGBM con pesos por recencia...")
    model_challenger_cb = CatBoostClassifier(**best_params_cb, random_seed=42, thread_count=-1, verbose=False)
    model_challenger_cb.fit(X_train, y_train, sample_weight=w_train)

    model_challenger_lgbm = LGBMClassifier(**best_params_lgbm, random_state=42, n_jobs=-1, verbosity=-1)
    model_challenger_lgbm.fit(X_train, y_train, sample_weight=w_train)

    # 6. Evaluar Ensemble Challenger
    prob_cb_ch = model_challenger_cb.predict_proba(X_test)[:, 1]
    prob_lgbm_ch = model_challenger_lgbm.predict_proba(X_test)[:, 1]
    prob_ensemble_ch = (prob_cb_ch + prob_lgbm_ch) / 2
    y_pred_ensemble = (prob_ensemble_ch >= 0.5).astype(int)
    metrics_ensemble = evaluar_modelo(y_test, y_pred_ensemble, prob_ensemble_ch)
    metrics_cb = evaluar_modelo(y_test, (prob_cb_ch >= 0.5).astype(int), prob_cb_ch)
    metrics_lgbm = evaluar_modelo(y_test, (prob_lgbm_ch >= 0.5).astype(int), prob_lgbm_ch)

    accuracy_challenger_ensemble = metrics_ensemble["Accuracy"]
    print(f"   📊 CatBoost solo:   {metrics_cb['Accuracy']:.2%}")
    print(f"   📊 LightGBM solo:   {metrics_lgbm['Accuracy']:.2%}")
    print(f"   📊 Ensemble (avg):  {accuracy_challenger_ensemble:.2%}")

    with mlflow.start_run(run_name="V4.0_Ensemble_Retraining"):
        mlflow.log_param("tuning_method", "Optuna")
        mlflow.log_param("ensemble", "CatBoost+LightGBM_avg")
        mlflow.log_param("weight_2026", 2.5)
        mlflow.log_param("weight_2025", 1.3)
        mlflow.log_dict(best_params_cb, "catboost_params.json")
        mlflow.log_dict(best_params_lgbm, "lgbm_params.json")
        for key, val in metrics_ensemble.items():
            mlflow.log_metric(f"ensemble_{key}", val)
        for key, val in metrics_cb.items():
            mlflow.log_metric(f"catboost_{key}", val)
        for key, val in metrics_lgbm.items():
            mlflow.log_metric(f"lgbm_{key}", val)
        mlflow.set_tag("version", "4.1_ensemble")
        mlflow.set_tag("phase", "Production Ensemble Challenger")

        # 7. Promoción: ensemble challenger vs ensemble champion
        if accuracy_challenger_ensemble >= accuracy_champion_ensemble:
            print("\n🟢 Ensemble Challenger SUPERÓ al Champion. Promocionando a Producción...")
            if os.path.exists(MODELO_PATH):
                shutil.copy(MODELO_PATH, MODELO_BACKUP)
            if os.path.exists(MODELO_LGBM_PATH):
                shutil.copy(MODELO_LGBM_PATH, MODELO_LGBM_BACKUP)

            model_challenger_cb.save_model(MODELO_PATH)
            joblib.dump(model_challenger_lgbm, MODELO_LGBM_PATH)
            print(f"   ✅ CatBoost guardado: {MODELO_PATH}")
            print(f"   ✅ LightGBM guardado: {MODELO_LGBM_PATH}")

            try:
                from mlb_utils import registrar_milestone_cumplido, verificar_milestone_reentrenamiento
                _, _, next_m = verificar_milestone_reentrenamiento()
                if next_m:
                    registrar_milestone_cumplido(next_m)
            except Exception as e:
                print(f"⚠️ No se pudo registrar el milestone: {e}")
        else:
            print("\n🔴 Champion Ensemble retuvo mejor rendimiento. Manteniendo modelos actuales en producción.")

if __name__ == "__main__":
    ejecutar_reentrenamiento()
