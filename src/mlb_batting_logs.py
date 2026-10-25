"""
Registro de bateadores por partido (alineación titular y línea de bateo) desde los boxscores de la
API oficial. Base para medir la calidad de la alineación de cada día (descansos, lesiones).

Se guarda en `data/batting_logs.csv.gz` y se amplía de forma incremental.

    python mlb_batting_logs.py --desde 2022-04-01 --hasta 2026-10-25   # backfill
    python mlb_batting_logs.py                                        # últimos 4 días
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_config import BASE_DIR
from mlb_pitching_logs import API, _get, codigo_equipo, partidos_finalizados

BATTING_LOGS_PATH = os.path.join(BASE_DIR, "data", "batting_logs.csv.gz")
COLUMNAS = ["game_pk", "fecha", "game_type", "team", "lado", "batter_id", "batter", "orden_bate",
            "pa", "ab", "h", "doubles", "triples", "hr", "bb", "hbp", "sf", "so"]


def lineas_de_bateo(partido):
    """Titulares (battingOrder múltiplo de 100) de ambos equipos con su línea del partido."""
    box = _get(f"{API}/game/{partido['game_pk']}/boxscore")
    filas = []
    for lado in ("home", "away"):
        t = box["teams"][lado]
        team = codigo_equipo(t["team"].get("abbreviation"), t["team"].get("name"))
        for jugador in t.get("players", {}).values():
            orden = str(jugador.get("battingOrder") or "")
            if not orden.isdigit() or int(orden) % 100 != 0:
                continue  # suplentes / cambios entran como 101, 102...
            st = jugador.get("stats", {}).get("batting", {})
            filas.append({
                "game_pk": partido["game_pk"], "fecha": partido["fecha"], "game_type": partido["game_type"], "team": team, "lado": lado,
                "batter_id": jugador.get("person", {}).get("id"), "batter": jugador.get("person", {}).get("fullName"),
                "orden_bate": int(orden) // 100, "pa": int(st.get("plateAppearances", 0) or 0), "ab": int(st.get("atBats", 0) or 0),
                "h": int(st.get("hits", 0) or 0), "doubles": int(st.get("doubles", 0) or 0), "triples": int(st.get("triples", 0) or 0),
                "hr": int(st.get("homeRuns", 0) or 0), "bb": int(st.get("baseOnBalls", 0) or 0), "hbp": int(st.get("hitByPitch", 0) or 0),
                "sf": int(st.get("sacFlies", 0) or 0), "so": int(st.get("strikeOuts", 0) or 0),
            })
    return filas


def actualizar_logs(desde, hasta, path=BATTING_LOGS_PATH, workers=6, guardar_cada=500):
    logs = pd.read_csv(path, dtype={"fecha": str}) if os.path.exists(path) else pd.DataFrame(columns=COLUMNAS)
    hechos = set(logs["game_pk"].astype(int)) if len(logs) else set()
    pendientes = [p for p in partidos_finalizados(desde, hasta) if p["game_pk"] not in hechos]
    print(f"📋 {len(pendientes)} partidos nuevos entre {desde} y {hasta} ({len(hechos)} ya en el registro)")
    if not pendientes:
        return len(logs)
    nuevas, errores, inicio = [], 0, time.time()

    def volcar():
        nonlocal logs, nuevas
        if nuevas:
            nuevo = pd.DataFrame(nuevas, columns=COLUMNAS)
            logs = (pd.concat([logs, nuevo], ignore_index=True) if len(logs) else nuevo)
            logs = logs.drop_duplicates(["game_pk", "batter_id"], keep="last").sort_values(["fecha", "game_pk", "lado", "orden_bate"])
            logs.to_csv(path, index=False, compression="gzip")
            nuevas = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futuros = {ex.submit(lineas_de_bateo, p): p for p in pendientes}
        for i, fut in enumerate(as_completed(futuros), 1):
            try:
                nuevas.extend(fut.result())
            except Exception as e:  # noqa: BLE001
                errores += 1
                print(f"⚠️ {futuros[fut]['game_pk']}: {e}")
            if i % guardar_cada == 0:
                volcar()
                print(f"   {i}/{len(pendientes)} partidos ({time.time() - inicio:.0f}s, {errores} errores)", flush=True)
    volcar()
    print(f"✅ Registro: {len(logs)} filas ({errores} partidos con error) en {time.time() - inicio:.0f}s")
    return len(logs)


def main():
    parser = argparse.ArgumentParser(description="Descarga el registro de bateadores titulares por partido")
    hoy = date.today()
    parser.add_argument("--desde", default=(hoy - timedelta(days=4)).isoformat())
    parser.add_argument("--hasta", default=(hoy - timedelta(days=1)).isoformat())
    parser.add_argument("--workers", type=int, default=6)
    a = parser.parse_args()
    actualizar_logs(a.desde, a.hasta, workers=a.workers)


if __name__ == "__main__":
    main()
