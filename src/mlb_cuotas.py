"""
Probabilidad implícita del mercado de apuestas para cada partido (moneyline), capturada desde la
API oficial de la MLB (`schedule?hydrate=preGameOdds`, casas BetMGM y FanDuel).

Solo se guardan datos derivados: la probabilidad del local sin el margen de la casa (media,
mínimo y máximo entre casas). Las cuotas en bruto no se almacenan porque el repositorio es público
y las condiciones de uso de las fuentes no permiten redistribuirlas.

Cada ejecución añade una captura por partido a `data/cuotas_mercado.csv`; la primera es la
apertura que vemos y la última antes del inicio, el cierre.

    python mlb_cuotas.py              # partidos de hoy y mañana
"""

import argparse
import os
import sys
from datetime import date, datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_config import BASE_DIR

CUOTAS_PATH = os.path.join(BASE_DIR, "data", "cuotas_mercado.csv")
COLUMNAS = ["game_pk", "fecha", "inicio_utc", "home_team", "away_team", "captura_utc", "n_casas",
            "prob_home_mercado", "prob_home_min", "prob_home_max"]


def prob_implicita(moneyline):
    """Cuota americana -> probabilidad implícita (con margen). '-150' -> 0.6, '+130' -> 0.4348."""
    ml = float(str(moneyline).replace("+", ""))
    return -ml / (-ml + 100.0) if ml < 0 else 100.0 / (ml + 100.0)


def prob_sin_margen(ml_home, ml_away):
    """Probabilidad del local normalizada para que local + visitante sumen 1."""
    ph, pa = prob_implicita(ml_home), prob_implicita(ml_away)
    return ph / (ph + pa)


def capturas_de_partido(g, captura_utc):
    from mlb_pitching_logs import codigo_equipo

    probs = []
    for casa in g.get("preGameOdds") or []:
        lados = {o.get("homeTeam"): o.get("moneyline") for o in casa.get("teamOdds", [])}
        if lados.get(True) and lados.get(False):
            try:
                probs.append(prob_sin_margen(lados[True], lados[False]))
            except ValueError:
                continue
    if not probs:
        return None
    h, a = g["teams"]["home"]["team"], g["teams"]["away"]["team"]
    return {
        "game_pk": g["gamePk"], "fecha": g.get("officialDate"), "inicio_utc": g.get("gameDate"),
        "home_team": codigo_equipo(h.get("abbreviation"), h.get("name")), "away_team": codigo_equipo(a.get("abbreviation"), a.get("name")),
        "captura_utc": captura_utc, "n_casas": len(probs), "prob_home_mercado": round(sum(probs) / len(probs), 4),
        "prob_home_min": round(min(probs), 4), "prob_home_max": round(max(probs), 4),
    }


def capturar(desde, hasta, path=CUOTAS_PATH):
    from mlb_pitching_logs import _get

    ahora = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    data = _get("https://statsapi.mlb.com/api/v1/schedule",
                {"sportId": 1, "startDate": desde, "endDate": hasta, "hydrate": "preGameOdds,team", "gameType": "R,F,D,L,W"})
    filas = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            # solo antes de que empiece: una cuota en vivo no es una predicción previa
            if g.get("status", {}).get("abstractGameState") != "Preview":
                continue
            fila = capturas_de_partido(g, ahora)
            if fila:
                filas.append(fila)
    if filas:
        nuevas = pd.DataFrame(filas, columns=COLUMNAS)
        if os.path.exists(path):
            nuevas = pd.concat([pd.read_csv(path), nuevas], ignore_index=True)
        nuevas.to_csv(path, index=False)
    print(f"💹 {len(filas)} partidos con cuotas capturados ({ahora})")
    return filas


def cierre_por_partido(path=CUOTAS_PATH):
    """Última captura antes del inicio de cada partido (la línea de cierre que tenemos)."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNAS)
    df = pd.read_csv(path)
    df = df[df["captura_utc"] < df["inicio_utc"]]
    return df.sort_values("captura_utc").groupby("game_pk", as_index=False).last()


def main():
    parser = argparse.ArgumentParser(description="Captura la probabilidad implícita del mercado")
    hoy = date.today()
    parser.add_argument("--desde", default=hoy.isoformat())
    parser.add_argument("--hasta", default=(hoy + timedelta(days=1)).isoformat())
    args = parser.parse_args()
    capturar(args.desde, args.hasta)


if __name__ == "__main__":
    main()
