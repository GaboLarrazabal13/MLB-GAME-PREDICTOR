"""
Validación y recuperación post-scraping de calidad de datos y predicciones.
Versión 4.0 - Adaptado para Ingesta API-First MLB.
"""

import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from mlb_config import DB_PATH
from mlb_daily_scraper import ejecutar_pipeline_diario
from mlb_predict_engine import ejecutar_flujo_diario

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
    "anunciado"
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
    try:
        row = conn.execute(f"SELECT MAX(fecha) FROM {table}").fetchone()
        return row[0] if row and row[0] else None
    except sqlite3.OperationalError:
        return None

def evaluar_estado_fechas(conn: sqlite3.Connection) -> dict[str, Any]:
    max_games = obtener_max_fecha(conn, "historico_partidos")
    max_preds = obtener_max_fecha(conn, "predicciones_historico")

    now_et = datetime.now(ZoneInfo("America/New_York"))
    today_et = now_et.strftime("%Y-%m-%d")
    yesterday_et = (now_et - timedelta(days=1)).strftime("%Y-%m-%d")

    # Verificar si hoy hay partidos programados en la API oficial de la MLB
    has_games_today = True
    try:
        import requests
        url_api = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today_et}"
        r = requests.get(url_api, timeout=10)
        if r.status_code == 200:
            data = r.json()
            games_count = len(data["dates"][0].get("games", [])) if data.get("dates") else 0
            has_games_today = games_count > 0
            if not has_games_today:
                print(f"[post-scrape] Hoy ({today_et}) es un día sin partidos programados (off-day).")
    except Exception as e:
        print(f"[post-scrape] Error al verificar el calendario de la MLB: {e}")

    # Si max_games es menor que hoy y ya son más de las 10 AM ET, consideramos que los datos están "stale"
    # únicamente si hoy tiene partidos programados en la cartelera.
    is_past_morning = now_et.hour >= 10
    today_missing = bool(max_games) and max_games < today_et and is_past_morning and has_games_today

    valid_dates = {today_et, yesterday_et}
    stale_games = (bool(max_games) and max_games not in valid_dates) or today_missing

    if stale_games:
        print(
            f"[post-scrape] Motivo stale: max_games({max_games}) no en {valid_dates} o today_missing({today_missing})"
        )

    desync_pred_gt_games = bool(max_preds) and ((not max_games) or (max_preds > max_games))

    return {
        "max_games": max_games,
        "max_preds": max_preds,
        "today_et": today_et,
        "yesterday_et": yesterday_et,
        "stale_games": stale_games,
        "today_missing": today_missing,
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
            hp.away_pitcher,
            hp.home_pitcher,
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
    try:
        rows = conn.execute(query, (fecha,)).fetchall()
    except sqlite3.OperationalError:
        return []

    columns = [
        "game_id",
        "fecha",
        "away_team",
        "home_team",
        "away_pitcher",
        "home_pitcher",
        "prediccion",
        "prob_home",
        "prob_away",
    ]

    anomalias: list[dict[str, Any]] = []
    for row in rows:
        data = dict(zip(columns, row, strict=False))

        missing_pitchers = is_missing_text(data.get("away_pitcher")) or is_missing_text(data.get("home_pitcher"))
        missing_prediction = (
            is_missing_text(data.get("prediccion")) or data.get("prob_home") is None or data.get("prob_away") is None
        )

        if missing_pitchers or missing_prediction:
            reasons = []
            if missing_pitchers:
                reasons.append("missing_pitchers")
            if missing_prediction:
                reasons.append("missing_prediction")
            data["reasons"] = ",".join(reasons)
            anomalias.append(data)

    return anomalias

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

    print("[post-scrape] Inicio de validación y recuperación API-First")
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
                print("[post-scrape] Detectado desfase de fecha/consistencia. Se intentará ronda extra de scraping.")
                requiere_refuerzo_fecha = True

        if requiere_refuerzo_fecha:
            print(f"[post-scrape] Intento {intento}/{max_intentos}: Reforzando scraping de fecha...")
            ejecutar_refuerzo_scrape()
            if intento < max_intentos:
                print(f"[post-scrape] Esperando {wait_seconds}s tras refuerzo de fecha...")
                time.sleep(wait_seconds)
            continue

        with sqlite3.connect(DB_PATH) as conn:
            fecha_objetivo = obtener_fecha_objetivo(conn)
            if not fecha_objetivo:
                print("[post-scrape] No hay historico_partidos para validar. Finaliza en éxito.")
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
                print("[post-scrape] Sin anomalías. Validación aprobada.")
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

            print(f"[post-scrape] Se detectaron {len(anomalias)} anomalías en {fecha_objetivo}. Reforzando scraping...")
            ejecutar_refuerzo_scrape()
            total_fixed += len(anomalias)  # Contar como reintentado

            pendientes = detectar_anomalias(conn, fecha_objetivo)
            print(f"[post-scrape] Fin intento {intento}: anomalías pendientes={len(pendientes)}")

            if not pendientes:
                final_estado = evaluar_estado_fechas(conn)
                final_max_games = final_estado["max_games"] or ""
                final_max_preds = final_estado["max_preds"] or ""
                final_anomalies = 0
                write_outputs(
                    status="ok",
                    attempts=intento,
                    anomalies=0,
                    fecha=fecha_objetivo,
                )
                print("[post-scrape] Recuperación completada con éxito.")
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

        if intento < max_intentos:
            print(f"[post-scrape] Esperando {wait_seconds}s antes del siguiente intento...")
            time.sleep(wait_seconds)

    with sqlite3.connect(DB_PATH) as conn:
        fecha_objetivo = obtener_fecha_objetivo(conn) or ""
        pendientes = detectar_anomalias(conn, fecha_objetivo) if fecha_objetivo else []
        final_estado = evaluar_estado_fechas(conn)
        final_max_games = final_estado["max_games"] or ""
        final_max_preds = final_estado["max_preds"] or ""
        final_anomalies = len(pendientes)

    print(f"[post-scrape] FALLO: persisten {len(pendientes)} anomalías tras {max_intentos} intentos.")
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
    import sys
    sys.exit(main())
