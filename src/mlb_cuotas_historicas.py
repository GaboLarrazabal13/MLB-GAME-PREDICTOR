"""
Probabilidad implícita del mercado para temporadas pasadas (2022-2026), desde la API interna de
ESPN (no documentada). Herramienta de uso local y puntual: no se ejecuta en GitHub Actions.

Por partido se guarda solo lo derivado:
  prob_home_cierre    media entre casas de la probabilidad del local sin margen con la cuota final
  prob_home_apertura  igual con la cuota de apertura, cuando ESPN la da (2024 en adelante)
  n_casas             cuántas casas entraron en la media

Cobertura conocida: 2022-2023 ~15 casas por partido; 2024-2026 una casa (ESPN BET / DraftKings)
con apertura y cierre.

    python mlb_cuotas_historicas.py --desde 2022-04-01 --hasta 2026-11-05
"""

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_config import BASE_DIR, get_team_code
from mlb_cuotas import prob_sin_margen

CUOTAS_HISTORICAS_PATH = os.path.join(BASE_DIR, "data", "cuotas_historicas.csv")
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
ODDS = "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/events/{id}/competitions/{id}/odds"
COLUMNAS = ["espn_id", "fecha", "home_team", "away_team", "tipo", "n_casas", "prob_home_cierre", "prob_home_apertura", "n_casas_apertura"]
ESPN_A_CODIGO = {"TB": "TBR", "CWS": "CHW", "KC": "KCR", "WSH": "WSN", "SD": "SDP", "SF": "SFG", "AZ": "ARI", "OAK": "ATH", "LV": "ATH"}
EXCLUIR = ("live", "consensus", "teamrankings")
PAUSA = 0.25  # segundos entre peticiones por hilo

_local = threading.local()


def _sesion():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers["User-Agent"] = "Mozilla/5.0 (mlb-game-predictor; uso personal)"
    return _local.s


def _get(url, params=None, intentos=4):
    for i in range(intentos):
        try:
            r = _sesion().get(url, params=params, timeout=30)
            if r.status_code == 200:
                time.sleep(PAUSA)
                return r.json()
            if r.status_code == 404:
                return {}
        except requests.RequestException:
            pass
        time.sleep(3 * (i + 1))
    raise RuntimeError(f"ESPN no respondió: {url}")


def _codigo(team):
    return ESPN_A_CODIGO.get(team.get("abbreviation"), team.get("abbreviation")) if team.get("abbreviation") else get_team_code(team.get("displayName"))


def _american(x):
    if isinstance(x, dict):
        x = x.get("american") or x.get("value")
    if x in (None, "", 0, "0", "EVEN"):
        return "+100" if x == "EVEN" else None
    return str(x)


def resumir_cuotas(items):
    """De los items de ESPN a (n, prob_cierre, n_apertura, prob_apertura) sin guardar cuotas en bruto."""
    cierre, apertura = [], []
    for it in items:
        nombre = str(it.get("provider", {}).get("name", "")).lower()
        if any(x in nombre for x in EXCLUIR):
            continue
        h, a = it.get("homeTeamOdds") or {}, it.get("awayTeamOdds") or {}
        mh, ma = _american(h.get("moneyLine")), _american(a.get("moneyLine"))
        if mh and ma:
            try:
                cierre.append(prob_sin_margen(mh, ma))
            except ValueError:
                pass
        oh, oa = _american((h.get("open") or {}).get("moneyLine")), _american((a.get("open") or {}).get("moneyLine"))
        if oh and oa:
            try:
                apertura.append(prob_sin_margen(oh, oa))
            except ValueError:
                pass
    media = lambda v: round(sum(v) / len(v), 4) if v else None  # noqa: E731
    return len(cierre), media(cierre), len(apertura), media(apertura)


def eventos_del_dia(dia):
    data = _get(SCOREBOARD, {"dates": dia.strftime("%Y%m%d")})
    out = []
    for e in data.get("events", []):
        c = (e.get("competitions") or [{}])[0]
        if not c.get("status", {}).get("type", {}).get("completed"):
            continue
        lados = {x["homeAway"]: x["team"] for x in c.get("competitors", [])}
        if "home" not in lados:
            continue
        out.append({"espn_id": e["id"], "fecha": dia.isoformat(), "home_team": _codigo(lados["home"]), "away_team": _codigo(lados["away"]),
                    "tipo": e.get("season", {}).get("type")})
    return out


def cuotas_de_evento(ev):
    data = _get(ODDS.format(id=ev["espn_id"]))
    n, pc, na, pa = resumir_cuotas(data.get("items", []))
    return {**ev, "n_casas": n, "prob_home_cierre": pc, "prob_home_apertura": pa, "n_casas_apertura": na}


def descargar(desde, hasta, path=CUOTAS_HISTORICAS_PATH, workers=3, temporada_de=(3, 11)):
    previas = pd.read_csv(path, dtype={"espn_id": str}) if os.path.exists(path) else pd.DataFrame(columns=COLUMNAS)
    hechos = set(previas["espn_id"].astype(str))
    dias, d = [], date.fromisoformat(desde)
    while d <= date.fromisoformat(hasta):
        if temporada_de[0] <= d.month <= temporada_de[1]:
            dias.append(d)
        d += timedelta(days=1)
    fechas_hechas = set(previas["fecha"]) if len(previas) else set()
    dias = [d for d in dias if d.isoformat() not in fechas_hechas]
    print(f"📅 {len(dias)} días por revisar ({len(hechos)} partidos ya descargados)")

    filas, inicio = [], time.time()

    def guardar():
        nonlocal previas, filas
        if filas:
            nuevas = pd.DataFrame(filas, columns=COLUMNAS)
            previas = (pd.concat([previas, nuevas], ignore_index=True) if len(previas) else nuevas).drop_duplicates("espn_id", keep="last")
            previas.sort_values(["fecha", "espn_id"]).to_csv(path, index=False)
            filas = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, dia in enumerate(dias, 1):
            try:
                eventos = [e for e in eventos_del_dia(dia) if e["espn_id"] not in hechos]
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ {dia}: {e}")
                continue
            for fut in as_completed([ex.submit(cuotas_de_evento, e) for e in eventos]):
                try:
                    filas.append(fut.result())
                except Exception as e:  # noqa: BLE001
                    print(f"⚠️ evento: {e}")
            if i % 15 == 0:
                guardar()
                print(f"   {i}/{len(dias)} días · {len(previas)} partidos · {time.time() - inicio:.0f}s")
    guardar()
    con = previas["prob_home_cierre"].notna().mean() if len(previas) else 0
    print(f"✅ {len(previas)} partidos ({con:.0%} con cuota de cierre) en {time.time() - inicio:.0f}s")


def main():
    p = argparse.ArgumentParser(description="Descarga histórica de probabilidades de mercado (ESPN)")
    p.add_argument("--desde", required=True)
    p.add_argument("--hasta", required=True)
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    descargar(a.desde, a.hasta, workers=a.workers)


if __name__ == "__main__":
    main()
