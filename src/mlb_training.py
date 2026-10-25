"""
Reentrenamiento honesto del modelo MLB (champion / challenger).

Reglas:
- El challenger se entrena solo con partidos anteriores a una "ventana sombra" (los últimos
  30 días con resultado) y se evalúa en esa ventana.
- El champion se evalúa con las predicciones que realmente publicó en esa ventana
  (`predicciones_historico`), que son fuera de muestra por construcción.
- Se promueve solo si el challenger mejora el log loss sin perder más de 1 punto de accuracy.
- Al promover, se reentrena con todos los datos y se escribe `models/model_card.json`.
- Cada evaluación queda registrada en `models/training_log.jsonl`.
"""

import json
import os
import pickle
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_config import (
    CACHE_PATH,
    DB_PATH,
    FEATURES_HISTORICAS_PATH,
    MODEL_FEATURES,
    MODELO_BACKUP,
    MODELO_PATH,
    SCRAPING_FEATURES,
    TRAINING_LOG_PATH,
)
from mlb_feature_engineering import (
    calcular_super_features,
    cargar_partidos_historicos,
    features_cronologicas_dataset,
    preparar_matriz_modelo,
    unir_partidos,
)
from mlb_model_card import escribir_model_card, leer_model_card, version_modelo_actual

DIAS_SOMBRA = 30
MIN_PARTIDOS_SOMBRA = 150
MIN_COBERTURA_PREDICCIONES = 0.80
MIN_COBERTURA_HISTORICA = 0.50
MAX_PERDIDA_ACCURACY = 0.01
MIN_DIAS_ENTRE_ENTRENAMIENTOS = 7
MIN_PARTIDOS_NUEVOS = 50
PESOS_RECENCIA = {0: 2.5, 1: 1.3}  # distancia en temporadas respecto a la actual -> peso


# ============================================================================
# REGISTRO
# ============================================================================
def registrar_training_log(entrada, path=TRAINING_LOG_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entrada, ensure_ascii=False, default=str) + "\n")


# ============================================================================
# CADENCIA
# ============================================================================
def debe_entrenar(conn, card, hoy=None, min_dias=MIN_DIAS_ENTRE_ENTRENAMIENTOS, min_partidos=MIN_PARTIDOS_NUEVOS):
    """Devuelve (True/False, razón). Entrena si pasaron `min_dias` y hay `min_partidos` nuevos con resultado."""
    hoy = hoy or datetime.now(timezone.utc).date()
    if not card:
        return True, "no existe model_card.json"

    trained_at = datetime.fromisoformat(card["trained_at"]).date()
    dias = (hoy - trained_at).days
    if dias < min_dias:
        return False, f"último entrenamiento hace {dias} días (mínimo {min_dias})"

    nuevos = conn.execute(
        "SELECT COUNT(*) FROM historico_real WHERE fecha > ? AND ganador IS NOT NULL",
        (card["trained_through"],),
    ).fetchone()[0]
    if nuevos < min_partidos:
        return False, f"solo {nuevos} partidos nuevos desde {card['trained_through']} (mínimo {min_partidos})"
    return True, f"{dias} días y {nuevos} partidos nuevos desde el último entrenamiento"


# ============================================================================
# DATASET
# ============================================================================
def cargar_features_historicas(conn=None, path=FEATURES_HISTORICAS_PATH):
    """Features de la API para temporadas pasadas: CSV versionado y, si no existe, la tabla de la DB."""
    if os.path.exists(path):
        return pd.read_csv(path)
    if conn is not None:
        try:
            return pd.read_sql("SELECT * FROM features_juegos", conn)
        except Exception:
            pass
    return pd.DataFrame(columns=["game_id", *SCRAPING_FEATURES])


def _cargar_cache(cache_path):
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "rb") as f:
            return {x["game_id"]: x for x in pickle.load(f).get("X_list", []) if x.get("game_id")}
    except Exception:
        return {}


def _guardar_cache(cache, cache_path):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"X_list": list(cache.values()), "y_list": [], "indices": []}, f)


def construir_dataset(conn, cache_path=CACHE_PATH, scraping_para_faltantes=True):
    """
    Devuelve un DataFrame ordenado por fecha con metadatos (game_id, fecha, year, home_team,
    away_team, ganador) y todas las MODEL_FEATURES. Aborta si las temporadas pasadas no tienen
    estadísticas reales (es la situación que arruinó los reentrenamientos de 2026).
    """
    real = pd.read_sql("SELECT * FROM historico_real WHERE ganador IS NOT NULL ORDER BY fecha ASC", conn)
    df = unir_partidos(real, cargar_partidos_historicos())
    if df.empty:
        raise ValueError("No hay partidos con resultado para entrenar")
    temporada_actual = int(df["year"].max())

    hist = cargar_features_historicas(conn).drop_duplicates("game_id").set_index("game_id")
    cache = _cargar_cache(cache_path)
    cache_crecio = False
    base = []
    for row in df.itertuples(index=False):
        if row.year < temporada_actual:
            feats = hist.loc[row.game_id].to_dict() if row.game_id in hist.index else {}
        elif row.game_id in cache:
            feats = cache[row.game_id]
        elif scraping_para_faltantes:
            from mlb_utils import extraer_features_hibridas

            try:
                feats = extraer_features_hibridas(row._asdict(), df_historico=None, hacer_scraping=True, session_cache={})
                feats["game_id"] = row.game_id
                cache[row.game_id] = feats
                cache_crecio = True
            except Exception as e:  # noqa: BLE001 - un partido sin features no debe tumbar el entrenamiento
                print(f"⚠️ Sin features para {row.game_id}: {e}")
                feats = {}
        else:
            feats = {}
        base.append({c: feats.get(c, 0.0) for c in SCRAPING_FEATURES})
    if cache_crecio:
        _guardar_cache(cache, cache_path)

    df = pd.concat([df, pd.DataFrame(base), features_cronologicas_dataset(df)], axis=1)
    df = pd.concat([df, pd.DataFrame([calcular_super_features(r) for r in df.to_dict("records")])[
        ["super_neutralizacion_whip_ops", "super_resistencia_era_ops", "super_muro_bullpen"]
    ]], axis=1)

    pasadas = df[df["year"] < temporada_actual]
    if len(pasadas):
        cobertura = float((pasadas["home_team_OPS"] != 0).mean())
        if cobertura < MIN_COBERTURA_HISTORICA:
            raise ValueError(
                f"Solo el {cobertura:.0%} de los partidos de temporadas pasadas tiene estadísticas reales. "
                f"Revisa {FEATURES_HISTORICAS_PATH} antes de entrenar."
            )
    return df


def pesos_recencia(years, temporada_actual=None):
    years = np.asarray(years, dtype=int)
    temporada_actual = temporada_actual or int(years.max())
    return np.array([PESOS_RECENCIA.get(temporada_actual - y, 1.0) for y in years], dtype=float)


# ============================================================================
# VENTANA SOMBRA Y MÉTRICAS
# ============================================================================
def definir_ventana_sombra(fechas, dias=DIAS_SOMBRA, min_partidos=MIN_PARTIDOS_SOMBRA):
    """(inicio, fin) como strings YYYY-MM-DD. Amplía hacia atrás hasta reunir `min_partidos`."""
    f = pd.to_datetime(pd.Series(fechas)).sort_values()
    fin = f.iloc[-1]
    inicio = fin - timedelta(days=dias - 1)
    while (f >= inicio).sum() < min_partidos and inicio > f.iloc[0]:
        inicio -= timedelta(days=7)
    inicio = max(inicio, f.iloc[0])
    return inicio.strftime("%Y-%m-%d"), fin.strftime("%Y-%m-%d")


def metricas(y, p):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }
    out["roc_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return out


def predicciones_publicadas(conn, inicio, fin, version=None):
    """Predicciones que el champion publicó en la ventana (última por partido)."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(predicciones_historico)")}
    sel = "fecha, home_team, away_team, prob_home" + (", model_version" if "model_version" in cols else "")
    df = pd.read_sql(
        f"SELECT {sel} FROM predicciones_historico WHERE fecha BETWEEN ? AND ?", conn, params=[inicio, fin]
    )
    if df.empty:
        return df
    if version and "model_version" in df.columns:
        df = df[df["model_version"].isna() | (df["model_version"] == version)]
    df = df.drop_duplicates(["fecha", "home_team", "away_team"], keep="last")
    df["p_home_champion"] = df["prob_home"].astype(float) / 100.0
    return df[["fecha", "home_team", "away_team", "p_home_champion"]]


def evaluar_champion(conn, sombra, card, modelo=None, min_cobertura=MIN_COBERTURA_PREDICCIONES):
    """
    Devuelve (metricas, fuente, sombra_evaluable).
    fuente: 'predicciones' (lo publicado), 'modelo' (predict_proba, solo si el champion no vio la ventana) o None.
    """
    inicio, fin = sombra["fecha"].min(), sombra["fecha"].max()
    pub = predicciones_publicadas(conn, inicio, fin, version=(card or {}).get("version"))
    comunes = sombra.merge(pub, on=["fecha", "home_team", "away_team"], how="inner") if not pub.empty else sombra.iloc[0:0]
    if len(comunes) >= min_cobertura * len(sombra) and len(comunes) >= MIN_PARTIDOS_SOMBRA // 2:
        return metricas(comunes["ganador"], comunes["p_home_champion"]), "predicciones", comunes

    trained_through = (card or {}).get("trained_through")
    if modelo is not None and trained_through and trained_through < inicio:
        X = preparar_matriz_modelo(sombra, modelo.feature_names_, nan_missing=bool((card or {}).get("nan_missing", False)))
        return metricas(sombra["ganador"], modelo.predict_proba(X)[:, 1]), "modelo", sombra
    return None, None, sombra


def decidir_promocion(m_champion, m_challenger, max_perdida_accuracy=MAX_PERDIDA_ACCURACY):
    if m_champion is None:
        return False, "no hay métricas honestas del champion en la ventana; se mantiene el modelo actual"
    if m_challenger["logloss"] >= m_champion["logloss"]:
        return False, f"logloss challenger {m_challenger['logloss']:.4f} ≥ champion {m_champion['logloss']:.4f}"
    if m_challenger["accuracy"] < m_champion["accuracy"] - max_perdida_accuracy:
        return False, (
            f"accuracy challenger {m_challenger['accuracy']:.3f} cae más de {max_perdida_accuracy:.0%} "
            f"frente a champion {m_champion['accuracy']:.3f}"
        )
    return True, (
        f"logloss {m_champion['logloss']:.4f} → {m_challenger['logloss']:.4f}, "
        f"accuracy {m_champion['accuracy']:.3f} → {m_challenger['accuracy']:.3f}"
    )


# ============================================================================
# ENTRENAMIENTO
# ============================================================================
def folds_temporales(n, n_folds=3, fraccion_val=0.10):
    """Validación expansiva: cada fold valida un bloque posterior a todo su entrenamiento."""
    bloque = max(int(n * fraccion_val), 1)
    folds = []
    for k in range(n_folds, 0, -1):
        fin_val = n - (k - 1) * bloque
        ini_val = fin_val - bloque
        if ini_val <= 0:
            continue
        folds.append((np.arange(0, ini_val), np.arange(ini_val, fin_val)))
    return folds


def _catboost(params):
    return CatBoostClassifier(
        **params, random_seed=42, thread_count=-1, verbose=False, allow_writing_files=False, bootstrap_type="Bernoulli"
    )


def optimizar_catboost(X, y, w, n_trials=20, folds=None):
    folds = folds or folds_temporales(len(X))

    def objetivo(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 200, 800),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        }
        perdidas = []
        for tr, va in folds:
            m = _catboost(params)
            m.fit(X.iloc[tr], y[tr], sample_weight=w[tr])
            perdidas.append(log_loss(y[va], np.clip(m.predict_proba(X.iloc[va])[:, 1], 1e-6, 1 - 1e-6), labels=[0, 1]))
        return float(np.mean(perdidas))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    estudio.optimize(objetivo, n_trials=n_trials)
    return estudio.best_params, float(estudio.best_value)


def entrenar(X, y, w, params):
    m = _catboost(params)
    m.fit(X, y, sample_weight=w)
    return m


def ejecutar_reentrenamiento(n_trials=20, dias_sombra=DIAS_SOMBRA, solo_evaluar=False, db_path=DB_PATH):
    """Orquesta la evaluación champion/challenger y, si procede, la promoción. Devuelve el registro."""
    ahora = datetime.now(timezone.utc)
    card = leer_model_card()
    print(f"\n🏷️ Champion actual: {version_modelo_actual(card)}")

    with sqlite3.connect(db_path) as conn:
        df = construir_dataset(conn)
        inicio, fin = definir_ventana_sombra(df["fecha"], dias=dias_sombra)
        sombra = df[(df["fecha"] >= inicio) & (df["fecha"] <= fin)].reset_index(drop=True)
        entren = df[df["fecha"] < inicio].reset_index(drop=True)
        print(f"📊 Dataset: {len(df)} partidos | entrenamiento: {len(entren)} | ventana sombra {inicio} → {fin}: {len(sombra)}")

        champion = None
        if os.path.exists(MODELO_PATH):
            champion = CatBoostClassifier()
            champion.load_model(MODELO_PATH)
        m_champ, fuente, comunes = evaluar_champion(conn, sombra, card, champion)

    temporada_actual = int(df["year"].max())
    X_tr = preparar_matriz_modelo(entren)
    y_tr = entren["ganador"].to_numpy()
    w_tr = pesos_recencia(entren["year"], temporada_actual)

    print(f"🔎 Optuna ({n_trials} trials, validación temporal)...")
    params, cv_logloss = optimizar_catboost(X_tr, y_tr, w_tr, n_trials=n_trials)
    challenger = entrenar(X_tr, y_tr, w_tr, params)

    eval_set = comunes if fuente == "predicciones" else sombra
    m_chall = metricas(eval_set["ganador"], challenger.predict_proba(preparar_matriz_modelo(eval_set))[:, 1])
    if m_champ:
        print(f"🏆 Champion ({fuente}): acc {m_champ['accuracy']:.3%} | logloss {m_champ['logloss']:.4f} | n={m_champ['n']}")
    else:
        print("⚠️ Sin métricas honestas del champion en la ventana (no publicó predicciones y ya vio esos partidos).")
    print(f"🥊 Challenger:          acc {m_chall['accuracy']:.3%} | logloss {m_chall['logloss']:.4f} | n={m_chall['n']}")

    promover, razon = decidir_promocion(m_champ, m_chall)
    registro = {
        "timestamp": ahora.isoformat(timespec="seconds"),
        "modo": "solo_evaluar" if solo_evaluar else "entrenamiento",
        "n_trials": n_trials,
        "ventana": {"inicio": inicio, "fin": fin, "n": int(len(sombra)), "evaluados": int(len(eval_set))},
        "fuente_champion": fuente,
        "champion": {"version": version_modelo_actual(card), "metricas": m_champ},
        "challenger": {"metricas": m_chall, "cv_logloss": cv_logloss, "params": params},
        "promovido": bool(promover and not solo_evaluar),
        "razon": razon,
    }

    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mlflow.db')}")
        mlflow.set_experiment("MLB-Predictor-Champion-Challenger")
        with mlflow.start_run(run_name=f"sombra_{inicio}_{fin}"):
            mlflow.log_params(params)
            mlflow.log_param("fuente_champion", fuente)
            for k, v in m_chall.items():
                mlflow.log_metric(f"challenger_{k}", v)
            for k, v in (m_champ or {}).items():
                mlflow.log_metric(f"champion_{k}", v)
            mlflow.set_tag("promovido", str(registro["promovido"]))
    except Exception as e:  # noqa: BLE001 - MLflow es opcional
        print(f"ℹ️ MLflow no disponible: {e}")

    if solo_evaluar:
        print(f"🧪 Solo evaluación. Decisión que se habría tomado: {'PROMOVER' if promover else 'RETENER'} — {razon}")
    elif promover:
        print(f"🟢 Promoviendo challenger: {razon}")
        X_all = preparar_matriz_modelo(df)
        final = entrenar(X_all, df["ganador"].to_numpy(), pesos_recencia(df["year"], temporada_actual), params)
        if os.path.exists(MODELO_PATH):
            shutil.copy(MODELO_PATH, MODELO_BACKUP)
        final.save_model(MODELO_PATH)
        nueva_version = f"catboost-{ahora.strftime('%Y%m%d')}"
        escribir_model_card(
            {
                "version": nueva_version,
                "version_anterior": version_modelo_actual(card),
                "trained_at": ahora.isoformat(timespec="seconds"),
                "trained_through": str(df["fecha"].max()),
                "n_train": int(len(df)),
                "features": list(X_all.columns),
                "nan_missing": True,
                "params": params,
                "pesos_recencia": {str(k): v for k, v in PESOS_RECENCIA.items()},
                "sombra": registro["ventana"] | {"fuente_champion": fuente, "champion": m_champ, "challenger": m_chall},
            }
        )
        registro["version_nueva"] = nueva_version
        print(f"   ✅ Modelo guardado: {MODELO_PATH} ({nueva_version})")
    else:
        print(f"🔴 Se mantiene el champion: {razon}")

    registrar_training_log(registro)
    return registro
