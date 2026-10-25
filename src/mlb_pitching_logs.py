"""
Registro de lanzadores por partido (una fila por lanzador y partido) desde la API oficial de la MLB.

Es la base de las features de forma reciente del abridor y de cansancio del bullpen. Se guarda en
`data/pitching_logs.csv.gz` y se amplía de forma incremental (los partidos ya descargados no se
vuelven a pedir).

    python mlb_pitching_logs.py --desde 2022-03-01 --hasta 2026-11-30   # backfill
    python mlb_pitching_logs.py                                        # últimos 4 días (uso diario)
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_config import PITCHING_LOGS_PATH, get_team_code

API = "https://statsapi.mlb.com/api/v1"
TIPOS_PARTIDO = "R,F,D,L,W"  # temporada regular + postemporada
COLUMNAS = [
    "game_pk", "fecha", "game_type", "game_number", "home_team", "away_team", "team", "lado",
    "pitcher_id", "pitcher", "orden", "es_abridor", "outs", "er", "h", "bb", "so", "hr", "pitches", "bf",
]
# Abreviaturas de la API que difieren de los códigos del proyecto
API_A_CODIGO = {"TB": "TBR", "CWS": "CHW", "KC": "KCR", "WSH": "WSN", "SD": "SDP", "SF": "SFG", "AZ": "ARI", "OAK": "ATH"}


def codigo_equipo(abbr, nombre=None):
    return API_A_CODIGO.get(abbr, abbr) if abbr else get_team_code(nombre)


def _get(url, params=None, intentos=4):
    for i in range(intentos):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"La API no respondió: {url}")


def partidos_finalizados(desde, hasta):
    """[{game_pk, fecha, game_type, game_number}] de partidos terminados entre dos fechas (YYYY-MM-DD)."""
    out = []
    ini = date.fromisoformat(desde)
    fin = date.fromisoformat(hasta)
    while ini <= fin:
        tramo_fin = min(ini + timedelta(days=30), fin)
        data = _get(f"{API}/schedule", {"sportId": 1, "startDate": ini.isoformat(), "endDate": tramo_fin.isoformat(), "gameType": TIPOS_PARTIDO})
        for d in data.get("dates", []):
            for g in d.get("games", []):
                if g.get("status", {}).get("abstractGameState") == "Final" and g.get("status", {}).get("detailedState") != "Postponed":
                    out.append({"game_pk": g["gamePk"], "fecha": g.get("officialDate", d["date"]), "game_type": g.get("gameType"), "game_number": g.get("gameNumber", 1)})
        ini = tramo_fin + timedelta(days=1)
    return out


def lineas_de_pitcheo(partido):
    """Filas de COLUMNAS para todos los lanzadores de un partido."""
    box = _get(f"{API}/game/{partido['game_pk']}/boxscore")
    equipos = {lado: codigo_equipo(box["teams"][lado]["team"].get("abbreviation"), box["teams"][lado]["team"].get("name")) for lado in ("home", "away")}
    filas = []
    for lado in ("home", "away"):
        t = box["teams"][lado]
        for orden, pid in enumerate(t.get("pitchers", [])):
            jugador = t["players"].get(f"ID{pid}", {})
            st = jugador.get("stats", {}).get("pitching", {})
            filas.append({
                **partido, "home_team": equipos["home"], "away_team": equipos["away"], "team": equipos[lado], "lado": lado,
                "pitcher_id": pid, "pitcher": jugador.get("person", {}).get("fullName"), "orden": orden,
                "es_abridor": int(orden == 0), "outs": int(st.get("outs", 0) or 0), "er": int(st.get("earnedRuns", 0) or 0),
                "h": int(st.get("hits", 0) or 0), "bb": int(st.get("baseOnBalls", 0) or 0), "so": int(st.get("strikeOuts", 0) or 0),
                "hr": int(st.get("homeRuns", 0) or 0), "pitches": int(st.get("numberOfPitches", 0) or 0), "bf": int(st.get("battersFaced", 0) or 0),
            })
    return filas


def cargar_logs(path=PITCHING_LOGS_PATH):
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNAS)
    return pd.read_csv(path, dtype={"fecha": str})


def actualizar_logs(desde, hasta, path=PITCHING_LOGS_PATH, workers=6, guardar_cada=500):
    """Descarga los partidos terminados del rango que aún no estén en el registro. Devuelve el total de filas."""
    logs = cargar_logs(path)
    hechos = set(logs["game_pk"].astype(int)) if len(logs) else set()
    pendientes = [p for p in partidos_finalizados(desde, hasta) if p["game_pk"] not in hechos]
    print(f"📋 {len(pendientes)} partidos nuevos entre {desde} y {hasta} ({len(hechos)} ya en el registro)")
    if not pendientes:
        return len(logs)

    nuevas, errores, inicio = [], 0, time.time()

    def volcar():
        nonlocal logs, nuevas
        if nuevas:
            logs = pd.concat([logs, pd.DataFrame(nuevas, columns=COLUMNAS)], ignore_index=True)
            logs = logs.drop_duplicates(["game_pk", "pitcher_id"], keep="last").sort_values(["fecha", "game_pk", "lado", "orden"])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            logs.to_csv(path, index=False, compression="gzip")
            nuevas = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futuros = {ex.submit(lineas_de_pitcheo, p): p for p in pendientes}
        for i, fut in enumerate(as_completed(futuros), 1):
            try:
                nuevas.extend(fut.result())
            except Exception as e:  # noqa: BLE001 - un boxscore fallido no detiene el resto
                errores += 1
                print(f"⚠️ {futuros[fut]['game_pk']}: {e}")
            if i % guardar_cada == 0:
                volcar()
                print(f"   {i}/{len(pendientes)} partidos ({time.time() - inicio:.0f}s, {errores} errores)")
    volcar()
    print(f"✅ Registro: {len(logs)} filas ({errores} partidos con error) en {time.time() - inicio:.0f}s")
    return len(logs)


def main():
    parser = argparse.ArgumentParser(description="Descarga el registro de lanzadores por partido")
    hoy = date.today()
    parser.add_argument("--desde", default=(hoy - timedelta(days=4)).isoformat())
    parser.add_argument("--hasta", default=(hoy - timedelta(days=1)).isoformat())
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    actualizar_logs(args.desde, args.hasta, workers=args.workers)


if __name__ == "__main__":
    main()
