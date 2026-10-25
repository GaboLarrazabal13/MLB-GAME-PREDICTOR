"""
Las predicciones publicadas de un partido ya jugado nunca se sobrescriben.
"""

import sqlite3

import pytest

from mlb_predict_engine import guardar_prediccion, partido_ya_jugado


@pytest.fixture
def conn():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE historico_real (game_id TEXT, fecha TEXT, home_team TEXT, away_team TEXT,
           score_home INTEGER, score_away INTEGER, ganador INTEGER)"""
    )
    conn.execute("INSERT INTO historico_real VALUES ('g1', '2026-06-01', 'NYY', 'BOS', 5, 3, 1)")
    return conn


def _pred(fecha, home, away, prob, version="v1"):
    return {
        "fecha": fecha, "home_team": home, "away_team": away, "home_pitcher": "A", "away_pitcher": "B",
        "prob_home": prob, "prob_away": 100 - prob, "prediccion": home if prob > 50 else away,
        "confianza": "ALTA", "tipo": "AUTOMATICO", "detalles": "{}", "model_version": version,
        "generated_at": "2026-06-01T10:00:00Z",
    }


def test_partido_ya_jugado(conn):
    assert partido_ya_jugado(conn, "2026-06-01", "NYY", "BOS")
    assert not partido_ya_jugado(conn, "2026-06-02", "NYY", "BOS")


def test_partido_futuro_se_reemplaza_en_historico(conn):
    assert guardar_prediccion(conn, _pred("2026-06-02", "LAD", "SFG", 60.0)) == "predicciones_historico"
    assert guardar_prediccion(conn, _pred("2026-06-02", "LAD", "SFG", 65.0)) == "predicciones_historico"
    filas = conn.execute("SELECT prob_home, model_version, generated_at FROM predicciones_historico").fetchall()
    assert filas == [(65.0, "v1", "2026-06-01T10:00:00Z")]


def test_partido_jugado_va_a_rescore_y_no_toca_lo_publicado(conn):
    guardar_prediccion(conn, _pred("2026-06-01", "NYY", "BOS", 58.0, version="v1"))  # ya tiene resultado
    assert conn.execute("SELECT COUNT(*) FROM predicciones_historico").fetchone()[0] == 0
    assert conn.execute("SELECT prob_home FROM predicciones_rescore").fetchall() == [(58.0,)]
    # un re-score con otra versión convive; con la misma versión se reemplaza
    guardar_prediccion(conn, _pred("2026-06-01", "NYY", "BOS", 61.0, version="v2"))
    guardar_prediccion(conn, _pred("2026-06-01", "NYY", "BOS", 62.0, version="v2"))
    assert sorted(conn.execute("SELECT model_version, prob_home FROM predicciones_rescore").fetchall()) == [("v1", 58.0), ("v2", 62.0)]


def test_tabla_antigua_gana_columnas_nuevas(conn):
    # Esquema que tenía producción antes de versionar las predicciones
    conn.execute(
        """CREATE TABLE predicciones_historico (fecha TEXT, home_team TEXT, away_team TEXT, home_pitcher TEXT,
           away_pitcher TEXT, prob_home REAL, prob_away REAL, prediccion TEXT, confianza TEXT, tipo TEXT)"""
    )
    guardar_prediccion(conn, _pred("2026-06-03", "CHC", "STL", 55.0))
    cols = {c[1] for c in conn.execute("PRAGMA table_info(predicciones_historico)")}
    assert {"detalles", "model_version", "generated_at", "prob_modelo", "prob_mercado"} <= cols
    assert conn.execute("SELECT model_version FROM predicciones_historico").fetchall() == [("v1",)]
