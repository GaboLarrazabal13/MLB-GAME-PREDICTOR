"""
Script de Comparación y Validación Científica: Optuna vs. GridSearch (Modelo V3.5)
Carga el caché de features, evalúa el modelo Champion actual, realiza la optimización
de Optuna y genera un reporte comparativo completo de métricas.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import accuracy_score, log_loss, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import optuna

warnings.filterwarnings("ignore")

# Asegurar importación de módulos centralizados
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_config import (
    CACHE_PATH,
    MODELO_PATH,
    MODEL_CONFIG,
    TEMPORAL_FEATURES,
    SCRAPING_FEATURES,
    SUPER_FEATURES,
)
from mlb_feature_engineering import calcular_super_features

def alinear_features_entrenamiento(X_new, model_actual=None):
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

def main():
    print("=" * 80)
    print(" 🧪 MLB PREDICTOR V3.5 - COMPARACIÓN CIENTÍFICA OPTUNA VS CHAMPION")
    print("=" * 80)
    
    # 1. Cargar caché de features preprocesadas
    if not os.path.exists(CACHE_PATH):
        print(f"❌ No se encontró el caché en {CACHE_PATH}. Debe ejecutarse una ingesta previa.")
        sys.exit(1)
        
    print(f"📦 Cargando caché de features desde: {CACHE_PATH}...")
    with open(CACHE_PATH, "rb") as f_pkl:
        cache_data = pickle.load(f_pkl)
        X_dict_list = cache_data.get("X_list", [])
        y_list = cache_data.get("y_list", [])
        
    print(f"   Juegos cargados en caché: {len(X_dict_list)}")
    
    # 2. Convertir a DataFrame y calcular super features
    X_raw = pd.DataFrame(X_dict_list).fillna(0)
    y_raw = np.array(y_list)
    
    print("🛠️ Calculando super features en los datos de caché...")
    for i in range(len(X_raw)):
        row_dict = X_raw.iloc[i].to_dict()
        updated_dict = calcular_super_features(row_dict)
        for key, val in updated_dict.items():
            if key in SUPER_FEATURES:
                X_raw.at[i, key] = val
                
    # Excluir no numéricas
    columnas_no_numericas = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()
    if columnas_no_numericas:
        X_raw = X_raw.drop(columns=columnas_no_numericas)
        
    # 3. Cargar modelo Champion (Actual)
    model_champion = None
    if os.path.exists(MODELO_PATH):
        try:
            model_champion = XGBClassifier()
            model_champion.load_model(MODELO_PATH)
            print("🏆 Modelo Champion actual cargado exitosamente.")
        except Exception as e:
            print(f"⚠️ Error cargando modelo previo: {e}")
            
    # Alinear features
    X_aligned = alinear_features_entrenamiento(X_raw, model_actual=model_champion)
    
    # 4. Escalado y división de datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_aligned.fillna(0))
    X_final = pd.DataFrame(X_scaled, columns=X_aligned.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_final,
        y_raw,
        test_size=MODEL_CONFIG["test_size"],
        random_state=MODEL_CONFIG["random_state"],
    )
    
    print(f"📊 Dataset dividido: {len(X_train)} entrenamiento, {len(X_test)} test.")
    
    # 5. Evaluar Champion en el conjunto de test
    champion_metrics = {}
    if model_champion is not None:
        try:
            y_pred_champ = model_champion.predict(X_test)
            y_prob_champ = model_champion.predict_proba(X_test)
            
            champion_metrics = {
                "Accuracy": accuracy_score(y_test, y_pred_champ),
                "Log Loss": log_loss(y_test, y_prob_champ),
                "F1-Score": f1_score(y_test, y_pred_champ, average="binary"),
                "Precision": precision_score(y_test, y_pred_champ, average="binary"),
                "Recall": recall_score(y_test, y_pred_champ, average="binary")
            }
            print("\n📊 Métricas del Modelo Champion (Actual) en conjunto de test:")
            for k, v in champion_metrics.items():
                print(f"   - {k}: {v:.4f}")
        except Exception as e:
            print(f"❌ Error evaluando Champion: {e}")
            
    # 6. Optimización de hiperparámetros con Optuna
    print("\n🔎 Buscando la mejor combinación de hiperparámetros con Optuna (35 trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    xgb_model_param = model_champion.get_booster() if model_champion else None
    
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
            
            try:
                if xgb_model_param is not None:
                    model.fit(X_tr, y_tr, xgb_model=xgb_model_param)
                else:
                    model.fit(X_tr, y_tr)
            except Exception:
                model.fit(X_tr, y_tr)
                
            y_pred = model.predict(X_val)
            scores.append(accuracy_score(y_val, y_pred))
            
        return np.mean(scores)
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=35)
    
    best_params = study.best_params
    print(f"\n🏆 Mejores parámetros encontrados por Optuna:")
    for k, v in best_params.items():
        print(f"   - {k}: {v}")
    print(f"📈 Mejor Accuracy en CV: {study.best_value:.2%}")
    
    # 7. Entrenar Challenger (Nuevo) con mejores parámetros de Optuna
    print("\n🚀 Entrenando el modelo Challenger final...")
    model_challenger = XGBClassifier(
        **best_params,
        eval_metric="logloss",
        random_state=MODEL_CONFIG["random_state"],
        n_jobs=-1
    )
    
    try:
        if xgb_model_param is not None:
            model_challenger.fit(X_train, y_train, xgb_model=xgb_model_param)
        else:
            model_challenger.fit(X_train, y_train)
    except Exception:
        model_challenger.fit(X_train, y_train)
        
    # 8. Evaluar Challenger en el conjunto de test
    y_pred_chal = model_challenger.predict(X_test)
    y_prob_chal = model_challenger.predict_proba(X_test)
    
    challenger_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_chal),
        "Log Loss": log_loss(y_test, y_prob_chal),
        "F1-Score": f1_score(y_test, y_pred_chal, average="binary"),
        "Precision": precision_score(y_test, y_pred_chal, average="binary"),
        "Recall": recall_score(y_test, y_pred_chal, average="binary")
    }
    
    # 9. Mostrar Reporte de Comparación
    print("\n" + "=" * 80)
    print(" 📊 REPORTE COMPARATIVO FINAL: CHAMPION VS. CHALLENGER (OPTUNA)")
    print("=" * 80)
    
    df_compare = pd.DataFrame({
        "Métrica": list(challenger_metrics.keys()),
        "Champion (Actual)": [f"{champion_metrics.get(m, 0.0):.4f}" for m in challenger_metrics.keys()],
        "Challenger (Optuna)": [f"{challenger_metrics[m]:.4f}" for m in challenger_metrics.keys()],
    })
    
    # Calcular diferencia
    differences = []
    for m in challenger_metrics.keys():
        diff = challenger_metrics[m] - champion_metrics.get(m, 0.0)
        sign = "+" if diff >= 0 else ""
        # Para Log Loss, menor es mejor, así que revertimos el color interpretado
        if m == "Log Loss":
            pref = "🟢 " if diff < 0 else "🔴 "
        else:
            pref = "🟢 " if diff > 0 else ("🔴 " if diff < 0 else "⚪ ")
        differences.append(f"{pref}{sign}{diff:.4f}")
        
    df_compare["Diferencia (Challenger - Champion)"] = differences
    
    print(df_compare.to_string(index=False))
    print("=" * 80)
    
    # Comparar detallado
    print("\n📝 Clasificación Detallada (Challenger):")
    print(classification_report(y_test, y_pred_chal, target_names=["Derrota Local (0)", "Victoria Local (1)"]))
    
    # Evaluar si hay mejora
    if challenger_metrics["Accuracy"] >= champion_metrics.get("Accuracy", 0.0):
        print("🟢 RESULTADO: ¡El modelo ajustado con Optuna ha superado o igualado al Champion!")
        print("   Se recomienda guardar y persistir los hiperparámetros de Optuna.")
    else:
        print("🔴 RESULTADO: El modelo Champion actual retiene un mejor desempeño en test.")
        
if __name__ == "__main__":
    main()
