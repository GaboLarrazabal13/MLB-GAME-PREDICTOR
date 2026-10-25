"""
Resumen de cuotas históricas de ESPN: solo probabilidades derivadas, sin casas en vivo ni agregados.
"""

import pytest

from mlb_cuotas import prob_sin_margen
from mlb_cuotas_historicas import resumir_cuotas


def _item(nombre, ml_h, ml_a, open_h=None, open_a=None):
    h = {"moneyLine": ml_h}
    a = {"moneyLine": ml_a}
    if open_h:
        h["open"] = {"moneyLine": {"american": open_h}}
        a["open"] = {"moneyLine": {"american": open_a}}
    return {"provider": {"name": nombre}, "homeTeamOdds": h, "awayTeamOdds": a}


def test_media_entre_casas_excluyendo_live_y_consenso():
    items = [
        _item("Bet365", -165, 145),
        _item("MGM", -155, 135),
        _item("ESPN BET - Live Odds", -400, 300),
        _item("consensus", -900, 600),
        _item("teamrankings", -900, 600),
    ]
    n, cierre, na, apertura = resumir_cuotas(items)
    assert n == 2 and na == 0 and apertura is None
    esperado = (prob_sin_margen("-165", "+145") + prob_sin_margen("-155", "+135")) / 2
    assert cierre == pytest.approx(esperado, abs=1e-4)


def test_apertura_cuando_existe():
    n, cierre, na, apertura = resumir_cuotas([_item("DraftKings", -198, 162, "-207", "+169")])
    assert n == 1 and na == 1
    assert apertura == pytest.approx(prob_sin_margen("-207", "+169"), abs=1e-4)


def test_cuotas_vacias_o_cero():
    assert resumir_cuotas([_item("X", 0, 0)]) == (0, None, 0, None)
    assert resumir_cuotas([]) == (0, None, 0, None)
