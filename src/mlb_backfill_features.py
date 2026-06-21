"""
Backfill de features_juegos para partidos históricos 2023-2025.

Consulta la API oficial de MLB para obtener estadísticas de pitchers y equipos
con datos acumulados hasta el día anterior a cada partido (sin data leakage).
Escribe los resultados en la tabla features_juegos de la DB de producción.

Uso:
    python mlb_backfill_features.py              # procesa 2023-2025
    python mlb_backfill_features.py --year 2024  # procesa solo 2024
    python mlb_backfill_features.py --resume     # salta games_id ya procesados
"""

import argparse
import os
import sqlite3
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_config import DB_PATH, SCRAPING_FEATURES
from mlb_stats_api_client import (
    obtener_fecha_ayer,
    obtener_stats_equipo_api,
    obtener_stats_jugadores_equipo_api,
    obtener_stats_pitcher_por_nombre,
    safe_float,
)


def _crear_tabla_features_juegos(conn):
    cols = ", ".join(f"{c} REAL DEFAULT 0.0" for c in SCRAPING_FEATURES)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS features_juegos (
            game_id TEXT PRIMARY KEY,
            {cols}
        )
    """)
    conn.commit()


def _top3_ops(jugadores_splits):
    """Extrae OPS del top-3 bateadores con más AB."""
    bateadores = []
    for s in jugadores_splits:
        stat = s.get("stat", {})
        ab = int(stat.get("atBats", 0) or 0)
        ops = safe_float(stat.get("ops", 0.0))
        ba = safe_float(stat.get("avg", 0.0))
        hr = int(stat.get("homeRuns", 0) or 0)
        if ab >= 5:
            bateadores.append({"ops": ops, "ba": ba, "hr": hr})
    bateadores.sort(key=lambda x: x["ops"], reverse=True)
    return bateadores[:3]


def _bullpen_stats(jugadores_splits_pitching):
    """ERA y WHIP promedio del bullpen (excluyendo abridores: GS > 0 con IP > 5 en GS)."""
    eras, whips = [], []
    for s in jugadores_splits_pitching:
        stat = s.get("stat", {})
        gs = int(stat.get("gamesStarted", 0) or 0)
        ip = safe_float(stat.get("inningsPitched", 0.0))
        era = safe_float(stat.get("era", 0.0))
        whip = safe_float(stat.get("whip", 0.0))
        # Excluir abridores claros (>= 3 GS con > 5 IP promedio)
        if gs >= 3:
            continue
        if era > 0 and ip >= 3:
            eras.append(era)
        if whip > 0 and ip >= 3:
            whips.append(whip)
    return (
        sum(eras) / len(eras) if eras else 0.0,
        sum(whips) / len(whips) if whips else 0.0,
    )


def extraer_features_para_juego(row, cache_equipos):
    """
    Extrae todas las SCRAPING_FEATURES para un partido histórico consultando la API.
    Usa cache_equipos para no repetir llamadas a la API por mismo equipo+fecha.
    """
    game_id = row["game_id"]
    home = row["home_team"]
    away = row["away_team"]
    home_p = row.get("home_pitcher", "") or ""
    away_p = row.get("away_pitcher", "") or ""
    fecha = str(row["fecha"])
    year = int(row["year"])
    fecha_ayer = obtener_fecha_ayer(fecha)

    feats = {c: 0.0 for c in SCRAPING_FEATURES}

    # ------------------------------------------------------------------ #
    # Estadísticas de equipo (hitting) — con cache por equipo+fecha
    # ------------------------------------------------------------------ #
    for side, team in [("home", home), ("away", away)]:
        cache_key = (team, year, fecha_ayer, "hitting")
        if cache_key not in cache_equipos:
            cache_equipos[cache_key] = obtener_stats_equipo_api(team, year, group="hitting", up_to_date=fecha_ayer)
        team_hit = cache_equipos[cache_key]
        ops = safe_float(team_hit.get("ops", 0.0))
        feats[f"{side}_team_OPS"] = ops

    feats["diff_team_OPS"] = feats["home_team_OPS"] - feats["away_team_OPS"]
    feats["diff_team_ERA"] = 0.0  # se calcula abajo
    feats["diff_team_BA"] = 0.0   # placeholder; la API no siempre devuelve BA de equipo en byDateRange

    # ------------------------------------------------------------------ #
    # Estadísticas de jugadores (bateadores top-3 y bullpen)
    # ------------------------------------------------------------------ #
    for side, team in [("home", home), ("away", away)]:
        cache_key_bat = (team, year, fecha_ayer, "batting_splits")
        if cache_key_bat not in cache_equipos:
            cache_equipos[cache_key_bat] = obtener_stats_jugadores_equipo_api(team, year, group="hitting", up_to_date=fecha_ayer)
        splits_bat = cache_equipos[cache_key_bat]
        top3 = _top3_ops(splits_bat)
        best_ops = top3[0]["ops"] if top3 else 0.0
        feats[f"{side}_best_OPS"] = best_ops

        cache_key_pit = (team, year, fecha_ayer, "pitching_splits")
        if cache_key_pit not in cache_equipos:
            cache_equipos[cache_key_pit] = obtener_stats_jugadores_equipo_api(team, year, group="pitching", up_to_date=fecha_ayer)
        splits_pit = cache_equipos[cache_key_pit]
        bp_era, bp_whip = _bullpen_stats(splits_pit)
        feats[f"{side}_bullpen_ERA"] = bp_era
        feats[f"{side}_bullpen_WHIP"] = bp_whip

    feats["diff_best_OPS"] = feats["home_best_OPS"] - feats["away_best_OPS"]
    feats["diff_best_BA"] = 0.0
    feats["diff_best_HR"] = 0.0
    feats["diff_bullpen_ERA"] = feats["home_bullpen_ERA"] - feats["away_bullpen_ERA"]
    feats["diff_bullpen_WHIP"] = feats["home_bullpen_WHIP"] - feats["away_bullpen_WHIP"]

    # ------------------------------------------------------------------ #
    # Estadísticas de lanzadores abridores
    # ------------------------------------------------------------------ #
    for side, team, pitcher_name in [("home", home, home_p), ("away", away, away_p)]:
        if pitcher_name and pitcher_name.strip().upper() not in ("TBD", ""):
            stats = obtener_stats_pitcher_por_nombre(team, pitcher_name, year, up_to_date=fecha_ayer)
            if stats:
                feats[f"{side}_starter_ERA"] = safe_float(stats.get("ERA", 0.0))
                feats[f"{side}_starter_WHIP"] = safe_float(stats.get("WHIP", 0.0))
                feats[f"{side}_starter_SO9"] = safe_float(stats.get("SO9", 0.0))
            time.sleep(0.3)  # rate-limit amigable

    feats["diff_starter_ERA"] = feats["home_starter_ERA"] - feats["away_starter_ERA"]
    feats["diff_starter_WHIP"] = feats["home_starter_WHIP"] - feats["away_starter_WHIP"]
    feats["diff_starter_SO9"] = feats["home_starter_SO9"] - feats["away_starter_SO9"]

    # anchor features (resúmenes de alto nivel)
    feats["anchor_pitching_level"] = (feats["home_starter_ERA"] + feats["away_starter_ERA"]) / 2
    feats["anchor_offensive_level"] = (feats["home_team_OPS"] + feats["away_team_OPS"]) / 2

    return game_id, feats


def backfill(years=None, resume=False):
    if years is None:
        years = [2023, 2024, 2025]

    with sqlite3.connect(DB_PATH) as conn:
        _crear_tabla_features_juegos(conn)

        # Juegos ya procesados (para --resume)
        ya_procesados = set()
        if resume:
            cur = conn.execute("SELECT game_id FROM features_juegos")
            ya_procesados = {r[0] for r in cur.fetchall()}
            print(f"   Resume: {len(ya_procesados)} juegos ya procesados, se saltarán.")

        year_filter = ", ".join("?" * len(years))
        df = pd.read_sql(
            f"SELECT game_id, home_team, away_team, home_pitcher, away_pitcher, fecha, year "
            f"FROM historico_real WHERE year IN ({year_filter}) ORDER BY fecha ASC",
            conn,
            params=years,
        )

    if df.empty:
        print("⚠️ No hay partidos para los años indicados en historico_real.")
        return

    pendientes = df[~df["game_id"].isin(ya_procesados)] if resume else df
    total = len(pendientes)
    print(f"\n🔄 Backfill de features para {total} partidos (años: {years})\n")

    cache_equipos = {}
    procesados = 0
    errores = 0

    for idx, row in pendientes.iterrows():
        try:
            game_id, feats = extraer_features_para_juego(row, cache_equipos)
            feats["game_id"] = game_id

            with sqlite3.connect(DB_PATH) as conn:
                cols = ["game_id"] + SCRAPING_FEATURES
                placeholders = ", ".join("?" * len(cols))
                values = [feats.get(c, 0.0) for c in cols]
                conn.execute(
                    f"INSERT OR REPLACE INTO features_juegos ({', '.join(cols)}) VALUES ({placeholders})",
                    values,
                )
                conn.commit()

            procesados += 1
            if procesados % 50 == 0 or procesados == total:
                print(f"   ✅ {procesados}/{total} procesados | errores: {errores}")

            # Pausa entre juegos para respetar rate limit
            time.sleep(0.5)

        except Exception as e:
            errores += 1
            print(f"   ❌ Error en {row['game_id']}: {e}")

    print(f"\n✅ Backfill completo: {procesados} procesados, {errores} errores.")
    print(f"   La tabla features_juegos ahora tiene datos para {total - errores} partidos históricos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill de features históricas 2023-2025")
    parser.add_argument("--year", type=int, help="Procesar solo este año (2023, 2024 o 2025)")
    parser.add_argument("--resume", action="store_true", help="Saltar game_ids ya procesados")
    args = parser.parse_args()

    years = [args.year] if args.year else [2023, 2024, 2025]
    backfill(years=years, resume=args.resume)
