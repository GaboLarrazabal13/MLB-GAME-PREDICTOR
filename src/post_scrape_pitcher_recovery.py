"""
Validacion y recuperacion post-scraping para juegos con datos incompletos.

Objetivo:
- Detectar anomalias en historico_partidos/predicciones_historico para la fecha mas reciente.
- Re-scrapear SOLO los partidos afectados (lanzadores o datos faltantes).
- Re-generar prediccion solo para esos partidos.
- Reintentar hasta N veces con espera entre intentos.
"""

import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from mlb_config import DB_PATH
from mlb_daily_scraper import (
    ejecutar_pipeline_diario,
    encontrar_lanzador,
    extraer_lanzadores_del_preview,
    scrape_player_stats,
)
from mlb_predict_engine import ejecutar_flujo_diario, predecir_juego

MISSING_TOKENS = {
    "",
    "tbd",
    "none",
    "null",
    "nan",
    "n/a",
    "na",
    "por anunciar",
    "pendiente",
}


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_missing_text(value: Any) -> bool:
    text = _to_text(value)
    return (not text) or (text.lower() in MISSING_TOKENS)


def write_outputs(**kwargs: Any) -> None:
    output_path = os.getenv("GITHUB_OUTPUT", "").strip()
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as fh:
        for key, value in kwargs.items():
            fh.write(f"{key}={value}\n")


def obtener_fecha_objetivo(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT MAX(fecha) FROM historico_partidos").fetchone()
    return row[0] if row and row[0] else None


def obtener_max_fecha(conn: sqlite3.Connection, table: str) -> str | None:
    row = conn.execute(f"SELECT MAX(fecha) FROM {table}").fetchone()
    return row[0] if row and row[0] else None


def evaluar_estado_fechas(conn: sqlite3.Connection) -> dict[str, Any]:
    max_games = obtener_max_fecha(conn, "historico_partidos")
    max_preds = obtener_max_fecha(conn, "predicciones_historico")

    now_et = datetime.now(ZoneInfo("America/New_York"))
    today_et = now_et.strftime("%Y-%m-%d")
    yesterday_et = (now_et - timedelta(days=1)).strftime("%Y-%m-%d")
    valid_dates = {today_et, yesterday_et}

    stale_games = bool(max_games) and max_games not in valid_dates
    desync_pred_gt_games = bool(max_preds) and (
        (not max_games) or (max_preds > max_games)
    )

    return {
        "max_games": max_games,
        "max_preds": max_preds,
        "today_et": today_et,
        "yesterday_et": yesterday_et,
        "stale_games": stale_games,
        "desync_pred_gt_games": desync_pred_gt_games,
    }


def ejecutar_refuerzo_scrape() -> None:
    print("[post-scrape] Ejecutando ronda de refuerzo de scraping...")

    prev = {
        "RUN_SOURCE": os.getenv("RUN_SOURCE"),
        "SCRAPER_MAX_ATTEMPTS": os.getenv("SCRAPER_MAX_ATTEMPTS"),
        "SCRAPER_RETRY_WAIT_SECONDS": os.getenv("SCRAPER_RETRY_WAIT_SECONDS"),
        "SCRAPER_SAVE_PARTIAL_ON_FINAL": os.getenv("SCRAPER_SAVE_PARTIAL_ON_FINAL"),
    }

    try:
        os.environ["RUN_SOURCE"] = "post_scrape_validate"
        os.environ["SCRAPER_MAX_ATTEMPTS"] = "1"
        os.environ["SCRAPER_RETRY_WAIT_SECONDS"] = "10"
        os.environ["SCRAPER_SAVE_PARTIAL_ON_FINAL"] = "1"

        ejecutar_pipeline_diario()
        ejecutar_flujo_diario()
    finally:
        for key, value in prev.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def detectar_anomalias(conn: sqlite3.Connection, fecha: str) -> list[dict[str, Any]]:
    query = """
        SELECT
            hp.game_id,
            hp.fecha,
            hp.away_team,
            hp.home_team,
            hp.box_score_url,
            hp.away_pitcher,
            hp.home_pitcher,
            hp.away_pitcher_link,
            hp.home_pitcher_link,
            hp.away_starter_ERA,
            hp.away_starter_WHIP,
            hp.away_starter_SO9,
            hp.home_starter_ERA,
            hp.home_starter_WHIP,
            hp.home_starter_SO9,
            p.prediccion,
            p.prob_home,
            p.prob_away
        FROM historico_partidos hp
        LEFT JOIN predicciones_historico p
          ON hp.fecha = p.fecha
         AND hp.home_team = p.home_team
         AND hp.away_team = p.away_team
        WHERE hp.fecha = ?
    """
    rows = conn.execute(query, (fecha,)).fetchall()

    columns = [
        "game_id",
        "fecha",
        "away_team",
        "home_team",
        "box_score_url",
        "away_pitcher",
        "home_pitcher",
        "away_pitcher_link",
        "home_pitcher_link",
        "away_starter_ERA",
        "away_starter_WHIP",
        "away_starter_SO9",
        "home_starter_ERA",
        "home_starter_WHIP",
        "home_starter_SO9",
        "prediccion",
        "prob_home",
        "prob_away",
    ]

    anomalias: list[dict[str, Any]] = []
    for row in rows:
        data = dict(zip(columns, row, strict=False))

        missing_pitchers = is_missing_text(data.get("away_pitcher")) or is_missing_text(
            data.get("home_pitcher")
        )
        missing_stats = any(
            data.get(k) is None
            for k in (
                "away_starter_ERA",
                "away_starter_WHIP",
                "away_starter_SO9",
                "home_starter_ERA",
                "home_starter_WHIP",
                "home_starter_SO9",
            )
        )
        missing_prediction = (
            is_missing_text(data.get("prediccion"))
            or data.get("prob_home") is None
            or data.get("prob_away") is None
        )

        if missing_pitchers or missing_stats or missing_prediction:
            reasons = []
            if missing_pitchers:
                reasons.append("missing_pitchers")
            if missing_stats:
                reasons.append("missing_stats")
            if missing_prediction:
                reasons.append("missing_prediction")
            data["reasons"] = ",".join(reasons)
            anomalias.append(data)

    return anomalias


def refrescar_partido(conn: sqlite3.Connection, partido: dict[str, Any]) -> bool:
    away_team = _to_text(partido.get("away_team"))
    home_team = _to_text(partido.get("home_team"))
    box_url = _to_text(partido.get("box_score_url"))
    fecha = _to_text(partido.get("fecha"))

    if not box_url:
        print(
            f"   - SKIP {away_team}@{home_team}: sin box_score_url para refrescar ({partido.get('reasons')})"
        )
        return False

    print(
        f"   - Reprocesando {away_team}@{home_team} ({fecha}) | razones: {partido.get('reasons')}"
    )

    lanzadores = extraer_lanzadores_del_preview(
        box_url,
        away_team=away_team,
        home_team=home_team,
    )

    away_info = lanzadores.get(away_team) if isinstance(lanzadores.get(away_team), dict) else {}
    home_info = lanzadores.get(home_team) if isinstance(lanzadores.get(home_team), dict) else {}

    away_pitcher = away_info.get("nombre") or _to_text(partido.get("away_pitcher"))
    home_pitcher = home_info.get("nombre") or _to_text(partido.get("home_pitcher"))
    away_pitcher_link = away_info.get("link") or _to_text(partido.get("away_pitcher_link"))
    home_pitcher_link = home_info.get("link") or _to_text(partido.get("home_pitcher_link"))

    if is_missing_text(away_pitcher) or is_missing_text(home_pitcher):
        print(f"     x No se pudieron obtener lanzadores: {away_pitcher} vs {home_pitcher}")
        return False

    year = int(fecha[:4]) if fecha else 2026

    away_df = scrape_player_stats(away_team, year)
    home_df = scrape_player_stats(home_team, year)

    away_stats = encontrar_lanzador(away_df, away_pitcher)
    home_stats = encontrar_lanzador(home_df, home_pitcher)

    conn.execute(
        """
        UPDATE historico_partidos
        SET
            away_pitcher = ?,
            home_pitcher = ?,
            away_pitcher_link = ?,
            home_pitcher_link = ?,
            away_starter_ERA = ?,
            away_starter_WHIP = ?,
            away_starter_H9 = ?,
            away_starter_SO9 = ?,
            away_starter_W = ?,
            away_starter_L = ?,
            home_starter_ERA = ?,
            home_starter_WHIP = ?,
            home_starter_H9 = ?,
            home_starter_SO9 = ?,
            home_starter_W = ?,
            home_starter_L = ?
        WHERE game_id = ?
        """,
        (
            away_pitcher,
            home_pitcher,
            away_pitcher_link,
            home_pitcher_link,
            away_stats.get("ERA", 0.0) if away_stats else 0.0,
            away_stats.get("WHIP", 0.0) if away_stats else 0.0,
            away_stats.get("H9", 0.0) if away_stats else 0.0,
            away_stats.get("SO9", 0.0) if away_stats else 0.0,
            away_stats.get("W", 0) if away_stats else 0,
            away_stats.get("L", 0) if away_stats else 0,
            home_stats.get("ERA", 0.0) if home_stats else 0.0,
            home_stats.get("WHIP", 0.0) if home_stats else 0.0,
            home_stats.get("H9", 0.0) if home_stats else 0.0,
            home_stats.get("SO9", 0.0) if home_stats else 0.0,
            home_stats.get("W", 0) if home_stats else 0,
            home_stats.get("L", 0) if home_stats else 0,
            _to_text(partido.get("game_id")),
        ),
    )

    pred = predecir_juego(
        home_team=home_team,
        away_team=away_team,
        home_pitcher=home_pitcher,
        away_pitcher=away_pitcher,
        year=year,
        modo_auto=True,
        fecha_partido=fecha,
        hacer_scraping=True,
    )

    if not pred:
        print("     x Fallo al regenerar prediccion")
        return False

    print("     + Partido refrescado y prediccion regenerada")
    return True


def main() -> int:
    max_intentos = int(os.getenv("POST_SCRAPE_MAX_ATTEMPTS", "3"))
    wait_seconds = int(os.getenv("POST_SCRAPE_WAIT_SECONDS", "300"))
    initial_max_games = ""
    initial_max_preds = ""
    final_max_games = ""
    final_max_preds = ""
    initial_anomalies = -1
    final_anomalies = -1
    total_fixed = 0

    print("[post-scrape] Inicio de validacion y recuperacion")
    print(f"[post-scrape] Config -> max_intentos={max_intentos}, wait_seconds={wait_seconds}")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predicciones_historico
            (
                fecha TEXT,
                home_team TEXT,
                away_team TEXT,
                home_pitcher TEXT,
                away_pitcher TEXT,
                prob_home REAL,
                prob_away REAL,
                prediccion TEXT,
                confianza TEXT,
                tipo TEXT
            )
            """
        )
        conn.commit()

    for intento in range(1, max_intentos + 1):
        requiere_refuerzo_fecha = False
        with sqlite3.connect(DB_PATH) as conn:
            estado = evaluar_estado_fechas(conn)
            if intento == 1:
                initial_max_games = estado["max_games"] or ""
                initial_max_preds = estado["max_preds"] or ""
            print(
                "[post-scrape] Estado fechas -> "
                f"max_games={estado['max_games']}, max_preds={estado['max_preds']}, "
                f"today_et={estado['today_et']}, yesterday_et={estado['yesterday_et']}"
            )

            if estado["stale_games"] or estado["desync_pred_gt_games"]:
                print(
                    "[post-scrape] Detectado desfase de fecha/consistencia. "
                    "Se intentará ronda extra de scraping."
                )
                requiere_refuerzo_fecha = True

            if requiere_refuerzo_fecha:
                continue

            fecha_objetivo = obtener_fecha_objetivo(conn)
            if not fecha_objetivo:
                print("[post-scrape] No hay historico_partidos para validar. Finaliza en exito.")
                write_outputs(status="ok", attempts=intento, anomalies=0, fecha="")
                return 0

            anomalias = detectar_anomalias(conn, fecha_objetivo)
            if intento == 1:
                initial_anomalies = len(anomalias)
            print(
                f"[post-scrape] Intento {intento}/{max_intentos} -> fecha={fecha_objetivo}, anomalias={len(anomalias)}"
            )

            if not anomalias:
                final_estado = evaluar_estado_fechas(conn)
                final_max_games = final_estado["max_games"] or ""
                final_max_preds = final_estado["max_preds"] or ""
                final_anomalies = 0
                write_outputs(status="ok", attempts=intento, anomalies=0, fecha=fecha_objetivo)
                print("[post-scrape] Sin anomalias. Validacion aprobada.")
                write_outputs(
                    max_games_before=initial_max_games,
                    max_preds_before=initial_max_preds,
                    max_games_after=final_max_games,
                    max_preds_after=final_max_preds,
                    anomalies_before=initial_anomalies,
                    anomalies_after=final_anomalies,
                    fixed_total=total_fixed,
                )
                return 0

            fixed = 0
            for partido in anomalias:
                try:
                    if refrescar_partido(conn, partido):
                        fixed += 1
                except Exception as exc:
                    print(
                        f"   - ERROR inesperado en {partido.get('away_team')}@{partido.get('home_team')}: {exc}"
                    )
            total_fixed += fixed
            conn.commit()

            pendientes = detectar_anomalias(conn, fecha_objetivo)
            print(
                f"[post-scrape] Fin intento {intento}: reparados={fixed}, pendientes={len(pendientes)}"
            )

            if not pendientes:
                final_estado = evaluar_estado_fechas(conn)
                final_max_games = final_estado["max_games"] or ""
                final_max_preds = final_estado["max_preds"] or ""
                final_anomalies = 0
                write_outputs(
                    status="ok",
                    attempts=intento,
                    anomalies=0,
                    fixed=fixed,
                    fecha=fecha_objetivo,
                )
                print("[post-scrape] Recuperacion completada con exito.")
                write_outputs(
                    max_games_before=initial_max_games,
                    max_preds_before=initial_max_preds,
                    max_games_after=final_max_games,
                    max_preds_after=final_max_preds,
                    anomalies_before=initial_anomalies,
                    anomalies_after=final_anomalies,
                    fixed_total=total_fixed,
                )
                return 0

        if requiere_refuerzo_fecha:
            ejecutar_refuerzo_scrape()
            if intento < max_intentos:
                print(
                    f"[post-scrape] Esperando {wait_seconds}s tras refuerzo de fecha..."
                )
                time.sleep(wait_seconds)
            continue

        if intento < max_intentos:
            ejecutar_refuerzo_scrape()
            print(f"[post-scrape] Esperando {wait_seconds}s antes del siguiente intento...")
            time.sleep(wait_seconds)

    with sqlite3.connect(DB_PATH) as conn:
        fecha_objetivo = obtener_fecha_objetivo(conn) or ""
        pendientes = detectar_anomalias(conn, fecha_objetivo) if fecha_objetivo else []
        final_estado = evaluar_estado_fechas(conn)
        final_max_games = final_estado["max_games"] or ""
        final_max_preds = final_estado["max_preds"] or ""
        final_anomalies = len(pendientes)

    print(
        f"[post-scrape] FALLO: persisten {len(pendientes)} anomalias tras {max_intentos} intentos."
    )
    write_outputs(
        status="failed",
        attempts=max_intentos,
        anomalies=len(pendientes),
        fecha=fecha_objetivo,
        max_games_before=initial_max_games,
        max_preds_before=initial_max_preds,
        max_games_after=final_max_games,
        max_preds_after=final_max_preds,
        anomalies_before=initial_anomalies,
        anomalies_after=final_anomalies,
        fixed_total=total_fixed,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
