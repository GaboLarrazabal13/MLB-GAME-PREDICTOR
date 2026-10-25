"""
Salud del modelo: solo predicciones honestas, referencia de mejor récord y estados.
"""

import sqlite3
from datetime import date, timedelta

import pytest

from mlb_salud import calcular_salud

CARD = {"version": "catboost-20260527-legacy", "trained_through": "2026-05-27"}


def _db(partidos):
    """partidos: [(fecha, home, away, ganador, prob_home, model_version)]"""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE historico_real (fecha TEXT, home_team TEXT, away_team TEXT, ganador INTEGER, year INTEGER)")
    conn.execute("CREATE TABLE predicciones_historico (fecha TEXT, home_team TEXT, away_team TEXT, prob_home REAL, model_version TEXT)")
    for f, h, a, g, p, v in partidos:
        conn.execute("INSERT INTO historico_real VALUES (?, ?, ?, ?, ?)", (f, h, a, g, int(f[:4])))
        if p is not None:
            conn.execute("INSERT INTO predicciones_historico VALUES (?, ?, ?, ?, ?)", (f, h, a, p, v))
    return conn


def _temporada(inicio, dias, acierta, version=None):
    """NYY siempre gana en casa; el modelo acierta o falla siempre."""
    out = []
    for i in range(dias):
        f = (date.fromisoformat(inicio) + timedelta(days=i)).isoformat()
        for k in range(6):
            out.append((f, f"H{k}", f"A{k}", 1, 70.0 if acierta else 30.0, version))
    return out


def test_excluye_predicciones_anteriores_al_corte_del_modelo_legado():
    conn = _db(_temporada("2026-05-01", 20, acierta=True) + _temporada("2026-06-01", 10, acierta=False))
    s = calcular_salud(conn, hoy=date(2026, 6, 10), card=CARD)
    assert s["desde"] >= "2026-05-28"
    assert s["ventana"]["accuracy"] == 0.0  # solo cuentan los fallos de junio


def test_versiones_nuevas_cuentan_siempre():
    conn = _db(_temporada("2026-05-01", 5, acierta=True, version="catboost-20260601"))
    s = calcular_salud(conn, hoy=date(2026, 5, 6), card={"version": "catboost-20260601"})
    assert s["ventana"]["n"] == 30 and s["ventana"]["accuracy"] == 1.0


def test_estado_ok_y_alerta():
    ok = calcular_salud(_db(_temporada("2026-06-01", 28, acierta=True)), hoy=date(2026, 6, 28), card=CARD)
    assert ok["estado"] == "ok"
    mal = calcular_salud(_db(_temporada("2026-06-01", 28, acierta=False)), hoy=date(2026, 6, 28), card=CARD)
    assert mal["estado"] == "alerta"
    assert len(mal["semanas"]) >= 3


def test_referencia_mejor_record_se_calcula_con_partidos_previos():
    # BOS gana los dos primeros; en el tercero (BOS visitante) la referencia elige a BOS
    partidos = [
        ("2026-06-01", "BOS", "NYY", 1, None, None),
        ("2026-06-02", "BOS", "NYY", 1, None, None),
        ("2026-06-03", "NYY", "BOS", 0, 60.0, None),
    ]
    s = calcular_salud(_db(partidos), hoy=date(2026, 6, 3), card=CARD)
    assert s["ventana"]["n"] == 1
    assert s["ventana"]["mejor_record"] == 1.0  # acertó eligiendo a BOS
    assert s["ventana"]["accuracy"] == 0.0  # el modelo eligió a NYY (60%)


def test_sin_datos():
    s = calcular_salud(_db([]), hoy=date(2026, 6, 3), card=CARD)
    assert s["estado"] == "sin_datos"


@pytest.mark.parametrize("dias", [7, 30])
def test_ventana_configurable(dias):
    s = calcular_salud(_db(_temporada("2026-06-01", 40, acierta=True)), hoy=date(2026, 7, 10), ventana_dias=dias, card=CARD)
    assert s["ventana"]["n"] == dias * 6
