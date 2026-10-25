"""
Módulo de Ingeniería de Features para MLB Predictor
Centraliza el cálculo de features para garantizar consistencia
entre entrenamiento y predicción
"""

import os

import numpy as np
import pandas as pd

from mlb_config import (
    DESCANSO_MAX_DIAS,
    ELO_CARRYOVER,
    ELO_HOME_ADVANTAGE,
    ELO_K,
    MODEL_FEATURES,
    PARTIDOS_HISTORICOS_PATH,
    SCRAPING_FEATURES,
    SUPER_FEATURES,
    get_team_code,
)

COLUMNAS_PARTIDO = ["game_id", "fecha", "year", "home_team", "away_team", "home_pitcher", "away_pitcher", "score_home", "score_away", "ganador"]


def cargar_partidos_historicos(path=PARTIDOS_HISTORICOS_PATH):
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNAS_PARTIDO)
    return pd.read_csv(path, dtype={"fecha": str, "game_id": str})[COLUMNAS_PARTIDO]


def _clave_partido(df):
    """fecha + códigos de equipo: identifica un partido aunque el game_id tenga otro formato."""
    code = lambda t: get_team_code(t) or str(t).strip().upper()  # noqa: E731
    return df["fecha"].str[:10] + "_" + df["home_team"].map(code) + "_" + df["away_team"].map(code)


def unir_partidos(historico_real, partidos_historicos):
    """Temporadas pasadas del CSV + partidos de historico_real que no estén ya en él, sin duplicados."""
    real = historico_real.dropna(subset=["ganador"]).copy()
    hist = partidos_historicos.copy()
    if len(hist):
        real = real[~_clave_partido(real).isin(set(_clave_partido(hist)))]
    df = pd.concat([hist, real[COLUMNAS_PARTIDO]], ignore_index=True)
    df = df.drop_duplicates("game_id", keep="last").sort_values(["fecha", "game_id"], kind="stable").reset_index(drop=True)
    df["year"] = df["year"].astype(int)
    df["ganador"] = df["ganador"].astype(int)
    return df


FEATURES_CRONOLOGICAS = [
    "home_win_rate_10", "home_racha", "home_runs_avg", "home_runs_diff",
    "away_win_rate_10", "away_racha", "away_runs_avg", "away_runs_diff",
    "home_elo", "away_elo", "diff_elo", "home_descanso", "away_descanso",
]


class TrackerCronologico:
    """
    Recorre los partidos en orden y, para cada uno, da las features que solo dependen de lo
    ocurrido ANTES: tendencias de la temporada, ELO y días de descanso.

    Es la única implementación: el entrenamiento recorre todo el histórico con `registrar`
    y la inferencia recorre los partidos anteriores al que se predice. Así no hay desfase
    entre lo que ve el modelo al entrenar y al predecir.

    - ELO: K=20, +24 de localía; al cambiar de temporada regresa un tercio hacia 1500, de
      modo que la fuerza del año anterior sirve de prior en abril.
    - Descanso: días desde el último partido del equipo en la temporada, topado en 7.
    """

    def __init__(self):
        self.elo = {}
        self.temporada = {}
        self.ultima_fecha = {}
        self.historial = {}

    @staticmethod
    def _code(team):
        return get_team_code(team) or str(team).strip().upper()

    def _elo_previo(self, code, year):
        elo = self.elo.get(code, 1500.0)
        if self.temporada.get(code) not in (None, year):
            elo = 1500.0 + (elo - 1500.0) * ELO_CARRYOVER
        return elo

    def _descanso(self, code, year, fecha):
        if self.temporada.get(code) != year or code not in self.ultima_fecha:
            return DESCANSO_MAX_DIAS
        return int(min((fecha - self.ultima_fecha[code]).days, DESCANSO_MAX_DIAS))

    @staticmethod
    def _tendencias(hist):
        if not hist:
            return 0.5, 0, 4.5, 0.0
        rec = hist[-10:]
        wr10 = sum(g["ganado"] for g in rec) / len(rec)
        ravg = sum(g["runs_sc"] for g in rec) / len(rec)
        rdiff = (sum(g["runs_sc"] for g in rec) - sum(g["runs_al"] for g in rec)) / len(rec)
        ultimo = hist[-1]["ganado"]
        racha = 0
        for g in reversed(hist):
            if g["ganado"] != ultimo:
                break
            racha += 1
        return wr10, (racha if ultimo else -racha), ravg, rdiff

    def features_previas(self, home_team, away_team, fecha, year):
        year = int(year)
        fecha = pd.Timestamp(fecha)
        h, a = self._code(home_team), self._code(away_team)
        th, ta = self._tendencias(self.historial.get((h, year), [])), self._tendencias(self.historial.get((a, year), []))
        eh, ea = self._elo_previo(h, year), self._elo_previo(a, year)
        return {
            "home_win_rate_10": th[0], "home_racha": th[1], "home_runs_avg": th[2], "home_runs_diff": th[3],
            "away_win_rate_10": ta[0], "away_racha": ta[1], "away_runs_avg": ta[2], "away_runs_diff": ta[3],
            "home_elo": eh, "away_elo": ea, "diff_elo": eh - ea,
            "home_descanso": self._descanso(h, year, fecha), "away_descanso": self._descanso(a, year, fecha),
        }

    def registrar(self, home_team, away_team, fecha, year, ganador, score_home=0, score_away=0):
        year = int(year)
        fecha = pd.Timestamp(fecha)
        h, a = self._code(home_team), self._code(away_team)
        eh, ea = self._elo_previo(h, year), self._elo_previo(a, year)
        esperado_home = 1.0 / (10.0 ** (-(eh + ELO_HOME_ADVANTAGE - ea) / 400.0) + 1.0)
        s_home = 1.0 if int(ganador) == 1 else 0.0
        self.elo[h] = eh + ELO_K * (s_home - esperado_home)
        self.elo[a] = ea + ELO_K * ((1.0 - s_home) - (1.0 - esperado_home))
        self.temporada[h] = self.temporada[a] = year
        self.ultima_fecha[h] = self.ultima_fecha[a] = fecha
        sh, sa = float(score_home or 0), float(score_away or 0)
        self.historial.setdefault((h, year), []).append({"ganado": s_home == 1.0, "runs_sc": sh, "runs_al": sa})
        self.historial.setdefault((a, year), []).append({"ganado": s_home == 0.0, "runs_sc": sa, "runs_al": sh})

    def registrar_dataframe(self, df, hasta_fecha=None):
        """Registra los partidos con resultado de `df` (opcionalmente solo los anteriores a `hasta_fecha`)."""
        if df is None or len(df) == 0:
            return self
        d = df.dropna(subset=["ganador"]).copy()
        d["fecha"] = pd.to_datetime(d["fecha"], errors="coerce")
        d = d.dropna(subset=["fecha"])
        if hasta_fecha is not None:
            d = d[d["fecha"] < pd.Timestamp(hasta_fecha)]
        d = d.sort_values("fecha", kind="stable")
        for r in d.itertuples(index=False):
            year = getattr(r, "year", None) or r.fecha.year
            self.registrar(r.home_team, r.away_team, r.fecha, year, r.ganador,
                           getattr(r, "score_home", 0), getattr(r, "score_away", 0))
        return self


_HISTORIA_CACHE = {}


def historia_para_tracker(df_historico_real):
    """
    Historia que alimenta al TrackerCronologico en inferencia: temporadas pasadas del CSV +
    historico_real, unidas igual que en el entrenamiento (así el ELO coincide).
    El CSV se lee una vez por proceso.
    """
    if "hist" not in _HISTORIA_CACHE:
        _HISTORIA_CACHE["hist"] = cargar_partidos_historicos()
    real = df_historico_real if df_historico_real is not None else pd.DataFrame(columns=COLUMNAS_PARTIDO)
    for col in COLUMNAS_PARTIDO:
        if col not in real.columns:
            real = real.assign(**{col: None})
    real = real.assign(year=pd.to_numeric(real["year"], errors="coerce").fillna(pd.to_datetime(real["fecha"], errors="coerce").dt.year))
    real = real.dropna(subset=["year", "ganador"])
    return unir_partidos(real, _HISTORIA_CACHE["hist"])


def features_cronologicas_dataset(df):
    """Features cronológicas de cada fila de `df` (ordenado por fecha), sin mirar el futuro."""
    tracker = TrackerCronologico()
    filas = []
    for r in df.itertuples(index=False):
        filas.append(tracker.features_previas(r.home_team, r.away_team, r.fecha, r.year))
        tracker.registrar(r.home_team, r.away_team, r.fecha, r.year, r.ganador, r.score_home, r.score_away)
    return pd.DataFrame(filas, columns=FEATURES_CRONOLOGICAS)


def preparar_matriz_modelo(df, features=None, nan_missing=True):
    """
    Deja un DataFrame listo para el modelo: columnas en el orden esperado, valores numéricos
    y, si `nan_missing`, estadísticas ausentes marcadas como NaN.

    Un ERA, OPS o WHIP exactamente 0 nunca es un dato real: es un valor que el scraper no
    consiguió. Marcarlo como NaN deja que CatBoost lo trate como faltante en lugar de
    aprender que "0" es un lanzador perfecto. Las super features derivadas de un dato
    ausente también se marcan como NaN. Debe usarse igual en entrenamiento e inferencia;
    `nan_missing=False` reproduce el comportamiento de los modelos entrenados con ceros.
    """
    features = list(features) if features is not None else list(MODEL_FEATURES)
    X = df.reindex(columns=features)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    if not nan_missing:
        return X

    stats = [c for c in SCRAPING_FEATURES if c in X.columns]
    X[stats] = X[stats].replace(0.0, np.nan)

    inputs_super = {
        "super_neutralizacion_whip_ops": ["home_starter_WHIP", "away_starter_WHIP", "home_team_OPS", "away_team_OPS"],
        "super_resistencia_era_ops": ["home_starter_ERA", "away_starter_ERA", "home_team_OPS", "away_team_OPS"],
        "super_muro_bullpen": ["home_bullpen_WHIP", "away_bullpen_WHIP", "home_best_OPS", "away_best_OPS"],
    }
    for sf in SUPER_FEATURES:
        deps = [c for c in inputs_super.get(sf, []) if c in X.columns]
        if sf in X.columns and deps:
            X.loc[X[deps].isna().any(axis=1), sf] = np.nan
    return X


def calcular_super_features(features_dict):
    """
    Calcula las "super features" derivadas a partir de features base.
    Debe usarse tanto en entrenamiento como en predicción.

    Args:
        features_dict: Diccionario con features base

    Returns:
        Diccionario actualizado con super features
    """
    # Valores por defecto seguros
    defaults = {
        "home_starter_WHIP": 1.3,
        "away_starter_WHIP": 1.3,
        "home_team_OPS": 0.75,
        "away_team_OPS": 0.75,
        "home_starter_ERA": 4.0,
        "away_starter_ERA": 4.0,
        "home_bullpen_WHIP": 1.3,
        "away_bullpen_WHIP": 1.3,
        "home_best_OPS": 0.85,
        "away_best_OPS": 0.85,
    }

    # Extraer valores con fallback a defaults
    def get_safe(key):
        return features_dict.get(key, defaults.get(key, 0))

    # Super Feature 1: Neutralización WHIP vs OPS
    # Mide cómo el WHIP del pitcher neutraliza el OPS del equipo contrario
    home_whip = get_safe("home_starter_WHIP")
    away_whip = get_safe("away_starter_WHIP")
    home_ops = get_safe("home_team_OPS")
    away_ops = get_safe("away_team_OPS")

    features_dict["super_neutralizacion_whip_ops"] = (home_whip * away_ops) - (away_whip * home_ops)

    # Super Feature 2: Resistencia ERA vs OPS
    # Mide la capacidad del pitcher de resistir la ofensiva rival
    home_era = get_safe("home_starter_ERA")
    away_era = get_safe("away_starter_ERA")

    features_dict["super_resistencia_era_ops"] = (home_era / (away_ops + 0.01)) - (away_era / (home_ops + 0.01))

    # Super Feature 3: Muro del Bullpen
    # Mide efectividad del bullpen contra mejores bateadores rivales
    home_bull_whip = get_safe("home_bullpen_WHIP")
    away_bull_whip = get_safe("away_bullpen_WHIP")
    home_best_ops = get_safe("home_best_OPS")
    away_best_ops = get_safe("away_best_OPS")

    features_dict["super_muro_bullpen"] = (home_bull_whip * away_best_ops) - (away_bull_whip * home_best_ops)

    return features_dict


def validar_features(features_dict, expected_features=None):
    """
    Valida que un diccionario de features tenga los campos esperados

    Args:
        features_dict: Diccionario con features
        expected_features: Lista de features esperadas (opcional)

    Returns:
        Tuple (is_valid, missing_features, extra_features)
    """
    if expected_features is None:
        return True, [], []

    current_features = set(features_dict.keys())
    expected_set = set(expected_features)

    missing = expected_set - current_features
    extra = current_features - expected_set

    is_valid = len(missing) == 0

    return is_valid, list(missing), list(extra)


def normalizar_features_dataframe(df, expected_features):
    """
    Asegura que un DataFrame tenga exactamente las columnas esperadas
    en el orden correcto, rellenando con 0 las faltantes

    Args:
        df: DataFrame con features
        expected_features: Lista ordenada de features esperadas

    Returns:
        DataFrame normalizado
    """
    return df.reindex(columns=expected_features, fill_value=0)


def calcular_estadisticas_agregadas(features_dict):
    """
    Calcula estadísticas agregadas útiles para análisis

    Args:
        features_dict: Diccionario con features

    Returns:
        Diccionario con estadísticas adicionales
    """
    stats = {}

    # Ventaja general de pitcheo
    home_pitch_quality = (features_dict.get("home_starter_ERA", 4.0) + features_dict.get("home_bullpen_ERA", 4.0)) / 2

    away_pitch_quality = (features_dict.get("away_starter_ERA", 4.0) + features_dict.get("away_bullpen_ERA", 4.0)) / 2

    stats["pitching_advantage"] = away_pitch_quality - home_pitch_quality

    # Ventaja general de bateo
    stats["batting_advantage"] = features_dict.get("home_team_OPS", 0.75) - features_dict.get("away_team_OPS", 0.75)

    # Ventaja de momentum
    stats["momentum_advantage"] = features_dict.get("home_win_rate_10", 0.5) - features_dict.get(
        "away_win_rate_10", 0.5
    )

    # Score compuesto
    stats["composite_advantage"] = (
        stats["pitching_advantage"] * 0.4 + stats["batting_advantage"] * 0.4 + stats["momentum_advantage"] * 0.2
    )

    return stats


def detectar_outliers(features_dict, thresholds=None):
    """
    Detecta valores atípicos que podrían indicar errores de scraping

    Args:
        features_dict: Diccionario con features
        thresholds: Dict con umbrales personalizados

    Returns:
        Lista de warnings sobre valores sospechosos
    """
    if thresholds is None:
        thresholds = {
            "ERA": (0.5, 10.0),
            "WHIP": (0.5, 3.0),
            "OPS": (0.4, 1.2),
            "win_rate": (0.0, 1.0),
            "racha": (-15, 15),
        }

    warnings = []

    # Validar ERAs
    for prefix in ["home_starter", "away_starter", "home_bullpen", "away_bullpen"]:
        era_key = f"{prefix}_ERA"
        if era_key in features_dict:
            era = features_dict[era_key]
            if era < thresholds["ERA"][0] or era > thresholds["ERA"][1]:
                warnings.append(f"⚠️ {era_key} fuera de rango: {era:.2f}")

    # Validar WHIPs
    for prefix in ["home_starter", "away_starter", "home_bullpen", "away_bullpen"]:
        whip_key = f"{prefix}_WHIP"
        if whip_key in features_dict:
            whip = features_dict[whip_key]
            if whip < thresholds["WHIP"][0] or whip > thresholds["WHIP"][1]:
                warnings.append(f"⚠️ {whip_key} fuera de rango: {whip:.3f}")

    # Validar OPS
    for key in ["home_team_OPS", "away_team_OPS", "home_best_OPS", "away_best_OPS"]:
        if key in features_dict:
            ops = features_dict[key]
            if ops < thresholds["OPS"][0] or ops > thresholds["OPS"][1]:
                warnings.append(f"⚠️ {key} fuera de rango: {ops:.3f}")

    return warnings


def generar_feature_importance_report(model, feature_names, top_n=20):
    """
    Genera un reporte de importancia de features

    Args:
        model: Modelo XGBoost entrenado
        feature_names: Lista de nombres de features
        top_n: Número de features top a mostrar

    Returns:
        DataFrame con ranking de importancia
    """
    try:
        importance = model.get_booster().get_score(importance_type="weight")

        # Convertir a DataFrame
        importance_df = (
            pd.DataFrame([{"feature": k, "importance": v} for k, v in importance.items()])
            .sort_values("importance", ascending=False)
            .head(top_n)
        )

        return importance_df
    except Exception as e:
        print(f"Error generando reporte de importancia: {e}")
        return pd.DataFrame()


def crear_features_interaccion(features_dict):
    """
    Crea features de interacción adicionales que podrían mejorar el modelo

    Args:
        features_dict: Diccionario con features base

    Returns:
        Diccionario actualizado con features de interacción
    """
    # Interacción pitcheo-bateo
    if "home_starter_ERA" in features_dict and "away_best_OPS" in features_dict:
        features_dict["home_pitch_vs_away_bat"] = features_dict["home_starter_ERA"] * features_dict["away_best_OPS"]

    if "away_starter_ERA" in features_dict and "home_best_OPS" in features_dict:
        features_dict["away_pitch_vs_home_bat"] = features_dict["away_starter_ERA"] * features_dict["home_best_OPS"]

    # Interacción racha-calidad
    if "home_racha" in features_dict and "home_team_OPS" in features_dict:
        features_dict["home_momentum_quality"] = features_dict["home_racha"] * features_dict["home_team_OPS"]

    if "away_racha" in features_dict and "away_team_OPS" in features_dict:
        features_dict["away_momentum_quality"] = features_dict["away_racha"] * features_dict["away_team_OPS"]

    return features_dict


if __name__ == "__main__":
    # Test de funciones
    test_features = {
        "home_starter_ERA": 3.2,
        "away_starter_ERA": 4.1,
        "home_starter_WHIP": 1.1,
        "away_starter_WHIP": 1.35,
        "home_team_OPS": 0.78,
        "away_team_OPS": 0.72,
        "home_bullpen_ERA": 3.8,
        "away_bullpen_ERA": 4.2,
        "home_bullpen_WHIP": 1.25,
        "away_bullpen_WHIP": 1.40,
        "home_best_OPS": 0.92,
        "away_best_OPS": 0.87,
        "home_win_rate_10": 0.65,
        "away_win_rate_10": 0.55,
        "home_racha": 3,
        "away_racha": -2,
    }

    print("🧪 Testing Feature Engineering...")

    # Calcular super features
    result = calcular_super_features(test_features.copy())
    print("\n✅ Super Features:")
    for key in [
        "super_neutralizacion_whip_ops",
        "super_resistencia_era_ops",
        "super_muro_bullpen",
    ]:
        print(f"  {key}: {result.get(key, 0):.4f}")

    # Detectar outliers
    warnings = detectar_outliers(test_features)
    if warnings:
        print("\n⚠️ Warnings detectados:")
        for w in warnings:
            print(f"  {w}")
    else:
        print("\n✅ No se detectaron outliers")

    # Estadísticas agregadas
    stats = calcular_estadisticas_agregadas(test_features)
    print("\n📊 Estadísticas Agregadas:")
    for key, val in stats.items():
        print(f"  {key}: {val:.4f}")
