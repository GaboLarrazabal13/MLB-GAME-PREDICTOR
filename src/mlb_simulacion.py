"""
Simulación de la temporada y los playoffs (Monte Carlo).

Parte de la clasificación real, simula los partidos que faltan con la fuerza ELO de cada equipo
y juega la llave de playoffs completa. Devuelve, por equipo, la probabilidad de clasificar, ganar
la división, ganar la liga y ser campeón.

    python mlb_simulacion.py            # simula la temporada en curso y guarda data/simulacion_playoffs.json

Formato de playoffs (2022+): por liga, 3 campeones de división (sembrados 1-3 por récord) y 3
comodines (4-6). Comodín al mejor de 3 (todo en casa del mejor sembrado: 3 vs 6, 4 vs 5); División
al mejor de 5 (1 vs ganador 4-5, 2 vs ganador 3-6); Campeonato y Serie Mundial al mejor de 7.
"""

import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_config import BASE_DIR, DB_PATH, DIVISION_DE, DIVISIONES, ELO_HOME_ADVANTAGE, get_team_code
from mlb_feature_engineering import TrackerCronologico, historia_para_tracker

SIMULACION_PATH = os.path.join(BASE_DIR, "data", "simulacion_playoffs.json")
FORMATOS = {3: [1, 1, 1], 5: [1, 1, 0, 0, 1], 7: [1, 1, 0, 0, 0, 1, 1]}  # 1 = en casa del mejor sembrado


def prob_local(elo_home, elo_away):
    return 1.0 / (1.0 + 10.0 ** (-(elo_home + ELO_HOME_ADVANTAGE - elo_away) / 400.0))


def sembrar(victorias, desempate, rng=None):
    """{liga: [6 equipos en orden de siembra]} a partir de victorias por equipo."""
    rng = rng or np.random.default_rng(0)
    out = {}
    for liga in ("AL", "NL"):
        equipos = [t for d, ts in DIVISIONES.items() if d.startswith(liga) for t in ts]
        orden = sorted(equipos, key=lambda t: (victorias[t], desempate.get(t, 0), rng.random()), reverse=True)
        campeones = []
        for d in (d for d in DIVISIONES if d.startswith(liga)):
            campeones.append(next(t for t in orden if DIVISION_DE[t] == d))
        campeones.sort(key=orden.index)
        comodines = [t for t in orden if t not in campeones][:3]
        out[liga] = campeones + comodines
    return out


def _serie(alto, bajo, n, elo, rng):
    necesarias, a, b = n // 2 + 1, 0, 0
    for en_casa_alto in FORMATOS[n]:
        p = prob_local(elo[alto], elo[bajo]) if en_casa_alto else 1 - prob_local(elo[bajo], elo[alto])
        if rng.random() < p:
            a += 1
        else:
            b += 1
        if a == necesarias or b == necesarias:
            break
    return alto if a == necesarias else bajo


def jugar_llave(siembra, elo, victorias, rng):
    campeones_liga = {}
    for liga, s in siembra.items():
        g36 = _serie(s[2], s[5], 3, elo, rng)
        g45 = _serie(s[3], s[4], 3, elo, rng)
        d1 = _serie(s[0], g45, 5, elo, rng)
        d2 = _serie(s[1], g36, 5, elo, rng)
        alto, bajo = sorted([d1, d2], key=s.index)
        campeones_liga[liga] = _serie(alto, bajo, 7, elo, rng)
    al, nl = campeones_liga["AL"], campeones_liga["NL"]
    alto, bajo = (al, nl) if victorias[al] >= victorias[nl] else (nl, al)
    return _serie(alto, bajo, 7, elo, rng), campeones_liga


def simular(victorias_actuales, pendientes, elo, n=10000, seed=2026):
    """
    victorias_actuales: {equipo: victorias}; pendientes: [(home, away)] por jugar en temporada regular;
    elo: {equipo: rating}. Devuelve {equipo: {playoffs, division, liga, campeon, victorias_esperadas}}.
    """
    rng = np.random.default_rng(seed)
    equipos = list(DIVISION_DE)
    cuenta = {t: defaultdict(float) for t in equipos}
    p = np.array([prob_local(elo[h], elo[a]) for h, a in pendientes])
    for _ in range(n):
        v = dict(victorias_actuales)
        if len(pendientes):
            gana_local = rng.random(len(pendientes)) < p
            for (h, a), gl in zip(pendientes, gana_local):
                v[h if gl else a] += 1
        siembra = sembrar(v, elo, rng)
        campeon, campeones_liga = jugar_llave(siembra, elo, v, rng)
        for t in equipos:
            cuenta[t]["victorias_esperadas"] += v[t]
        for s in siembra.values():
            for i, t in enumerate(s):
                cuenta[t]["playoffs"] += 1
                if i < 3:
                    cuenta[t]["division"] += 1
        for t in campeones_liga.values():
            cuenta[t]["liga"] += 1
        cuenta[campeon]["campeon"] += 1
    return {t: {k: c[k] / n for k in ("playoffs", "division", "liga", "campeon", "victorias_esperadas")} for t, c in cuenta.items()}


def calendario_pendiente(temporada):
    """Partidos de temporada regular aún no jugados según la API oficial: [(home, away)]."""
    from mlb_pitching_logs import _get, codigo_equipo

    data = _get("https://statsapi.mlb.com/api/v1/schedule", {"sportId": 1, "season": temporada, "gameType": "R", "hydrate": "team"})
    out = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameState") in ("Preview", "Live"):
                h, a = g["teams"]["home"]["team"], g["teams"]["away"]["team"]
                out.append((codigo_equipo(h.get("abbreviation"), h.get("name")), codigo_equipo(a.get("abbreviation"), a.get("name"))))
    return out


def victorias_oficiales(temporada):
    """Victorias de temporada regular por equipo según la clasificación oficial de la API."""
    from mlb_pitching_logs import _get, codigo_equipo

    data = _get("https://statsapi.mlb.com/api/v1/standings", {"leagueId": "103,104", "season": temporada, "standingsTypes": "regularSeason", "hydrate": "team"})
    out = {}
    for rec in data.get("records", []):
        for t in rec.get("teamRecords", []):
            out[codigo_equipo(t["team"].get("abbreviation"), t["team"].get("name"))] = int(t.get("wins", 0))
    return out


def ejecutar(temporada=None, n=10000, db_path=DB_PATH, guardar=True, pendientes=None, victorias=None):
    temporada = temporada or date.today().year
    with sqlite3.connect(db_path) as conn:
        real = pd.read_sql("SELECT * FROM historico_real WHERE ganador IS NOT NULL", conn)
    historia = historia_para_tracker(real)
    tracker = TrackerCronologico().registrar_dataframe(historia)
    elo = {t: tracker.elo.get(t, 1500.0) for t in DIVISION_DE}

    if victorias is None:
        try:
            victorias = victorias_oficiales(temporada)
        except Exception:  # noqa: BLE001 - sin API se cuentan los partidos guardados (puede incluir postemporada)
            victorias = {}
    if set(victorias) != set(DIVISION_DE):
        victorias = dict.fromkeys(DIVISION_DE, 0)
        for r in historia[historia["year"] == temporada].itertuples(index=False):
            ganador = get_team_code(r.home_team if r.ganador == 1 else r.away_team)
            if ganador in victorias:
                victorias[ganador] += 1
    if pendientes is None:
        pendientes = calendario_pendiente(temporada)
    pendientes = [(h, a) for h, a in pendientes if h in elo and a in elo]

    res = simular(victorias, pendientes, elo, n=n)
    salida = {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "temporada": temporada,
        "partidos_pendientes": len(pendientes),
        "simulaciones": n,
        "equipos": sorted(
            [{"equipo": t, "division_nombre": DIVISION_DE[t], "victorias": victorias[t], "elo": round(elo[t], 1),
              **{k: round(v, 4) for k, v in r.items()}} for t, r in res.items()],
            key=lambda e: -e["campeon"],
        ),
    }
    if guardar:
        with open(SIMULACION_PATH, "w", encoding="utf-8") as f:
            json.dump(salida, f, ensure_ascii=False, indent=1)
    return salida


if __name__ == "__main__":
    s = ejecutar()
    print(f"Temporada {s['temporada']}: {s['partidos_pendientes']} partidos pendientes, {s['simulaciones']} simulaciones")
    for e in s["equipos"][:10]:
        print(f"  {e['equipo']}: campeón {e['campeon']:.1%} | liga {e['liga']:.1%} | playoffs {e['playoffs']:.1%} | V esperadas {e['victorias_esperadas']:.1f}")
