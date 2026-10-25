"""
Variables sabermétricas cronológicas y el modelo logístico que las usa.

Variables (solo con información anterior al día del partido):
  pitag    % de victorias esperado por carreras de la temporada (exponente 1.83), regresado hacia
           la temporada anterior (a su vez regresada hacia 0.5) según los partidos jugados.
  rs / ra  carreras anotadas / recibidas por partido en la temporada, regresadas a la media de liga.
  fip_sp   FIP del abridor ((13·HR + 3·BB − 2·K)/IP + c) en los últimos 365 días, regresado con 40 IP.
  fip_bp   FIP de los relevistas del equipo en los últimos 30 días, regresado con 30 IP.

Las constantes se fijaron a priori (literatura sabermétrica), no ajustadas a los datos de prueba.

Modelo: regresión logística (imputación por mediana + estandarización, C=1) con 7 variables.
En la validación cronológica mes a mes 2024-2026 (7,209 partidos) acertó 56.6% frente a 55.3%
del CatBoost de 38 variables, mejor en cada uno de los tres años.
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from mlb_config import BASE_DIR, get_team_code

MODELO_SABER_PATH = os.path.join(BASE_DIR, "models", "modelo_saber.pkl")
MODELO_SABER_CARD = os.path.join(BASE_DIR, "models", "modelo_saber.json")

EXP_PITAG = 1.83
K_PITAG = 30.0
REGRESION_ANIO_ANTERIOR = 0.6
K_CARRERAS = 10.0
K_FIP_SP = 40.0
K_FIP_BP = 30.0
C_FIP = 3.15
FIP_LIGA = 4.15
RPG_LIGA = 4.4

FEATURES_SABER = [
    "home_pitag", "away_pitag", "diff_pitag", "home_rs", "away_rs", "home_ra", "away_ra",
    "home_fip_sp", "away_fip_sp", "diff_fip_sp", "home_fip_bp", "away_fip_bp", "diff_fip_bp",
]
FEATURES_MODELO_SABER = ["diff_pitag", "diff_fip_sp", "diff_fip_bp", "home_rs", "away_rs", "home_ra", "away_ra"]


def _code(team):
    return get_team_code(team) or str(team).strip().upper()


class Sabermetria:
    """
    Estado cronológico de carreras por equipo y temporada + registro de lanzadores.
    `registrar` añade partidos jugados; `features` consulta un partido usando solo lo registrado
    y los lanzamientos de fechas anteriores.
    """

    def __init__(self, logs):
        from mlb_feature_engineering import normalizar_nombre

        self._norm = normalizar_nombre
        self.rs, self.ra, self.g = defaultdict(float), defaultdict(float), defaultdict(int)
        self.pitag_final = {}
        lg = logs.copy() if logs is not None else pd.DataFrame(columns=["fecha", "pitcher", "team", "es_abridor", "outs", "hr", "bb", "so"])
        lg["fecha"] = pd.to_datetime(lg["fecha"])
        lg["nombre"] = lg["pitcher"].map(normalizar_nombre)
        lg["ip"] = lg["outs"].astype(float) / 3.0
        cols = ["ip", "hr", "bb", "so"]
        self.sp = {n: (g["fecha"].to_numpy(), g[cols].to_numpy(float)) for n, g in lg[lg["es_abridor"] == 1].sort_values("fecha").groupby("nombre")}
        dia = lg[lg["es_abridor"] == 0].groupby(["team", "fecha"])[cols].sum().reset_index()
        self.bp = {t: (g["fecha"].to_numpy(), np.cumsum(g[cols].to_numpy(float), axis=0)) for t, g in dia.sort_values("fecha").groupby("team")}

    # ----------------------------------------------------------------- carreras
    def _pitag(self, k):
        rs, ra = self.rs[k], self.ra[k]
        return rs ** EXP_PITAG / (rs ** EXP_PITAG + ra ** EXP_PITAG) if rs + ra > 0 else 0.5

    def registrar(self, home_team, away_team, year, score_home, score_away):
        year = int(year)
        sh, sa = float(score_home or 0), float(score_away or 0)
        for t, a, b in ((_code(home_team), sh, sa), (_code(away_team), sa, sh)):
            k = (t, year)
            self.rs[k] += a
            self.ra[k] += b
            self.g[k] += 1
            self.pitag_final[k] = self._pitag(k)

    def registrar_dataframe(self, df, hasta_fecha=None):
        if df is None or len(df) == 0:
            return self
        d = df.dropna(subset=["ganador"])
        if hasta_fecha is not None:
            d = d[d["fecha"].astype(str).str[:10] < str(hasta_fecha)[:10]]
        for r in d.sort_values("fecha", kind="stable").itertuples(index=False):
            self.registrar(r.home_team, r.away_team, r.year, r.score_home, r.score_away)
        return self

    def _equipo(self, team, year):
        t, year = _code(team), int(year)
        k = (t, year)
        prior = 0.5 + (self.pitag_final.get((t, year - 1), 0.5) - 0.5) * REGRESION_ANIO_ANTERIOR
        if not self.g[k]:
            return prior, RPG_LIGA, RPG_LIGA
        peso = self.g[k] / (self.g[k] + K_PITAG)
        return (peso * self._pitag(k) + (1 - peso) * prior,
                (self.rs[k] + RPG_LIGA * K_CARRERAS) / (self.g[k] + K_CARRERAS),
                (self.ra[k] + RPG_LIGA * K_CARRERAS) / (self.g[k] + K_CARRERAS))

    # ----------------------------------------------------------------- FIP
    @staticmethod
    def _fip(ip, hr, bb, so, k):
        crudo = (13 * hr + 3 * bb - 2 * so) / ip + C_FIP if ip > 0 else FIP_LIGA
        return (crudo * ip + FIP_LIGA * k) / (ip + k)

    def _fip_sp(self, nombre, fecha):
        n = self._norm(nombre)
        if n not in self.sp:
            return FIP_LIGA
        f, v = self.sp[n]
        a = np.searchsorted(f, fecha - np.timedelta64(365, "D"), side="left")
        b = np.searchsorted(f, fecha, side="left")
        return self._fip(*(v[a:b].sum(axis=0) if b > a else np.zeros(4)), K_FIP_SP)

    def _fip_bp(self, team, fecha):
        t = _code(team)
        if t not in self.bp:
            return FIP_LIGA
        f, cum = self.bp[t]
        a = np.searchsorted(f, fecha - np.timedelta64(30, "D"), side="left")
        b = np.searchsorted(f, fecha, side="left")
        if b == 0 or b <= a:
            return FIP_LIGA
        return self._fip(*(cum[b - 1] - (cum[a - 1] if a > 0 else 0)), K_FIP_BP)

    # ----------------------------------------------------------------- consulta
    def features(self, home_team, away_team, home_pitcher, away_pitcher, fecha, year):
        f = np.datetime64(pd.Timestamp(fecha).normalize())
        hp, hrs, hra = self._equipo(home_team, year)
        ap, ars, ara = self._equipo(away_team, year)
        hfs, afs = self._fip_sp(home_pitcher, f), self._fip_sp(away_pitcher, f)
        hfb, afb = self._fip_bp(home_team, f), self._fip_bp(away_team, f)
        return {
            "home_pitag": hp, "away_pitag": ap, "diff_pitag": hp - ap,
            "home_rs": hrs, "away_rs": ars, "home_ra": hra, "away_ra": ara,
            "home_fip_sp": hfs, "away_fip_sp": afs, "diff_fip_sp": afs - hfs,  # positivo = ventaja local
            "home_fip_bp": hfb, "away_fip_bp": afb, "diff_fip_bp": afb - hfb,
        }


_BASE = {}


def sabermetria_hasta(df_historia, fecha):
    """
    Sabermetria con la historia anterior a `fecha`. El registro de lanzadores se carga una vez por
    proceso (lo caro); las carreras se recalculan en cada consulta sobre una copia.
    """
    import copy

    from mlb_config import PITCHING_LOGS_PATH

    mtime = os.path.getmtime(PITCHING_LOGS_PATH) if os.path.exists(PITCHING_LOGS_PATH) else None
    if _BASE.get("mtime") != mtime or "s" not in _BASE:
        logs = pd.read_csv(PITCHING_LOGS_PATH, dtype={"fecha": str}) if mtime else None
        _BASE.update(s=Sabermetria(logs), mtime=mtime)
    s = copy.copy(_BASE["s"])
    s.rs, s.ra, s.g, s.pitag_final = defaultdict(float), defaultdict(float), defaultdict(int), {}
    return s.registrar_dataframe(df_historia, hasta_fecha=fecha)


def features_sabermetricas_dataset(df, logs):
    """Features de cada fila de `df` (con resultado), día a día: los partidos del mismo día no se ven entre sí."""
    s = Sabermetria(logs)
    d = df.reset_index(drop=True)
    filas = [None] * len(d)
    for _, idx in d.groupby(d["fecha"].astype(str).str[:10], sort=True).groups.items():
        for i in idx:
            r = d.loc[i]
            filas[i] = s.features(r["home_team"], r["away_team"], r["home_pitcher"], r["away_pitcher"], r["fecha"], r["year"])
        for i in idx:
            r = d.loc[i]
            s.registrar(r["home_team"], r["away_team"], r["year"], r["score_home"], r["score_away"])
    return pd.DataFrame(filas, columns=FEATURES_SABER)


# ============================================================================
# MODELO LOGÍSTICO
# ============================================================================
def _pesos(years):
    years = np.asarray(years, dtype=int)
    actual = years.max()
    return np.where(years == actual, 2.5, np.where(years == actual - 1, 1.3, 1.0))


def entrenar_modelo_saber(dataset, path=MODELO_SABER_PATH, card_path=MODELO_SABER_CARD):
    """`dataset` necesita FEATURES_MODELO_SABER, year, fecha y ganador. Guarda el modelo y su ficha."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = dataset[FEATURES_MODELO_SABER].astype(float)
    y = dataset["ganador"].astype(int).to_numpy()
    modelo = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    modelo.fit(X, y, logisticregression__sample_weight=_pesos(dataset["year"]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(modelo, path)
    coef = modelo.named_steps["logisticregression"].coef_[0]
    card = {
        "version": f"saber-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_through": str(dataset["fecha"].max())[:10],
        "n_train": int(len(dataset)),
        "features": FEATURES_MODELO_SABER,
        "coeficientes_estandarizados": dict(zip(FEATURES_MODELO_SABER, np.round(coef, 4).tolist())),
        "intercepto": round(float(modelo.named_steps["logisticregression"].intercept_[0]), 4),
    }
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return modelo, card


_CACHE = {}


def cargar_modelo_saber(path=MODELO_SABER_PATH):
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    if _CACHE.get("mtime") != mtime:
        _CACHE.update(modelo=joblib.load(path), mtime=mtime)
    return _CACHE["modelo"]


def predecir_saber(features, modelo=None):
    """Probabilidad del local según el modelo sabermétrico (None si no hay modelo entrenado)."""
    modelo = modelo or cargar_modelo_saber()
    if modelo is None:
        return None
    X = pd.DataFrame([{k: features.get(k, np.nan) for k in FEATURES_MODELO_SABER}]).astype(float)
    return float(modelo.predict_proba(X)[0, 1])
