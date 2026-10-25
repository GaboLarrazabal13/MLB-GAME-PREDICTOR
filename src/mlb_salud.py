"""
Salud del modelo en producción: acierto reciente de las predicciones publicadas frente a dos
referencias simples (elegir al equipo con mejor récord previo y elegir siempre al local).

Solo cuenta predicciones honestas: partidos posteriores a la fecha de corte del modelo que las
generó. Las predicciones de partidos ya jugados no se reescriben (ver guardar_prediccion), así que
desde ahora todo lo publicado es fuera de muestra.

Estados:
  ok       el modelo supera a la referencia de mejor récord en la ventana
  vigilar  queda por debajo en la ventana
  alerta   queda por debajo en las últimas 3 semanas completas (≥30 partidos cada una)
"""

from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

from mlb_model_card import leer_model_card

CORTE_LEGADO = "2026-05-27"  # el champion de mayo se entrenó con partidos hasta esta fecha
MIN_PARTIDOS_SEMANA = 30


def _predicciones_con_resultado(conn):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(predicciones_historico)")}
    extra = ", p.model_version" if "model_version" in cols else ", NULL AS model_version"
    df = pd.read_sql(
        f"""
        SELECT p.fecha, p.home_team, p.away_team, p.prob_home{extra}, r.ganador, r.year
        FROM predicciones_historico p
        INNER JOIN historico_real r
            ON r.fecha = p.fecha AND r.home_team = p.home_team AND r.away_team = p.away_team
        WHERE r.ganador IS NOT NULL
        """,
        conn,
    )
    return df.drop_duplicates(["fecha", "home_team", "away_team"], keep="last")


def _referencia_mejor_record(conn, temporadas):
    """Para cada partido de las temporadas pedidas: 1 si el local tenía mejor récord previo (empate → local)."""
    if not temporadas:
        return {}
    marcas = ",".join("?" * len(temporadas))
    real = pd.read_sql(
        f"SELECT fecha, home_team, away_team, ganador, year FROM historico_real WHERE ganador IS NOT NULL AND year IN ({marcas}) ORDER BY fecha",
        conn,
        params=list(temporadas),
    )
    w, g, out = defaultdict(int), defaultdict(int), {}
    for r in real.itertuples(index=False):
        kh, ka = (r.year, r.home_team), (r.year, r.away_team)
        ph = w[kh] / g[kh] if g[kh] else 0.5
        pa = w[ka] / g[ka] if g[ka] else 0.5
        out[(r.fecha, r.home_team, r.away_team)] = 1 if ph >= pa else 0
        g[kh] += 1
        g[ka] += 1
        w[kh if r.ganador == 1 else ka] += 1
    return out


def _metricas(d):
    if d.empty:
        return {"n": 0, "accuracy": None, "logloss": None, "mejor_record": None, "local": None}
    p = np.clip(d["p"].to_numpy(), 1e-6, 1 - 1e-6)
    y = d["ganador"].to_numpy()
    return {
        "n": int(len(d)),
        "accuracy": float(((p >= 0.5) == y).mean()),
        "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "mejor_record": float((d["ref"] == y).mean()),
        "local": float(y.mean()),
    }


def calcular_salud(conn, hoy=None, ventana_dias=30, card=None):
    hoy = hoy or date.today()
    card = card if card is not None else (leer_model_card() or {})
    df = _predicciones_con_resultado(conn)
    corte = card.get("trained_through", CORTE_LEGADO) if card.get("version", "").endswith("legacy") else CORTE_LEGADO
    # honestas: modelo legado solo después de su corte; versiones nuevas siempre (se publican antes del partido)
    legado = df["model_version"].isna() | df["model_version"].astype(str).str.endswith("legacy")
    df = df[~legado | (df["fecha"] > corte)].copy()
    if df.empty:
        return {"estado": "sin_datos", "modelo": card.get("version"), "ventana": _metricas(df), "semanas": [], "motivo": "no hay predicciones honestas con resultado"}

    ref = _referencia_mejor_record(conn, sorted(df["year"].dropna().astype(int).unique().tolist()))
    df["ref"] = [ref.get((r.fecha, r.home_team, r.away_team), 1) for r in df.itertuples(index=False)]
    df["p"] = df["prob_home"].astype(float) / 100.0
    df["fecha_dt"] = pd.to_datetime(df["fecha"])

    fin = pd.Timestamp(min(hoy, df["fecha_dt"].max().date()))
    ventana = df[df["fecha_dt"] > fin - timedelta(days=ventana_dias)]
    df["semana"] = df["fecha_dt"].dt.to_period("W-SUN")
    semanas = []
    for per, g in df.groupby("semana"):
        semanas.append({"semana": str(per.start_time.date()), **_metricas(g)})

    m = _metricas(ventana)
    completas = [s for s in semanas if s["n"] >= MIN_PARTIDOS_SEMANA][-3:]
    if len(completas) == 3 and all(s["accuracy"] < s["mejor_record"] for s in completas):
        estado, motivo = "alerta", "por debajo de la referencia de mejor récord en las últimas 3 semanas"
    elif m["n"] and m["accuracy"] < m["mejor_record"]:
        estado, motivo = "vigilar", f"por debajo de la referencia de mejor récord en los últimos {ventana_dias} días"
    else:
        estado, motivo = "ok", f"por encima de la referencia de mejor récord en los últimos {ventana_dias} días"

    return {
        "estado": estado,
        "motivo": motivo,
        "modelo": card.get("version"),
        "desde": str(df["fecha"].min()),
        "hasta": str(df["fecha"].max()),
        "ventana_dias": ventana_dias,
        "ventana": m,
        "temporada": _metricas(df[df["year"] == df["year"].max()]),
        "semanas": semanas,
    }
