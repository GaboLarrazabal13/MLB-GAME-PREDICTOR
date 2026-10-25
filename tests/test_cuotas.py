"""
Conversión de cuotas americanas a probabilidad y captura derivada (sin cuotas en bruto).
"""

import pytest

from mlb_cuotas import COLUMNAS, capturas_de_partido, cierre_por_partido, prob_implicita, prob_sin_margen


def test_prob_implicita():
    assert prob_implicita("-150") == pytest.approx(0.6)
    assert prob_implicita("+130") == pytest.approx(100 / 230)
    assert prob_implicita("+100") == pytest.approx(0.5)


def test_prob_sin_margen_suma_uno():
    ph = prob_sin_margen("-118", "-102")
    pa = prob_sin_margen("-102", "-118")
    assert ph + pa == pytest.approx(1.0) and ph > 0.5


GAME = {
    "gamePk": 849804, "officialDate": "2026-10-26", "gameDate": "2026-10-27T00:08:00Z",
    "teams": {"home": {"team": {"abbreviation": "TB", "name": "Tampa Bay Rays"}}, "away": {"team": {"abbreviation": "MIL", "name": "Milwaukee Brewers"}}},
    "preGameOdds": [
        {"provider": {"name": "BetMGM"}, "teamOdds": [{"homeTeam": True, "moneyline": "-118"}, {"homeTeam": False, "moneyline": "-102"}]},
        {"provider": {"name": "FanDuel"}, "teamOdds": [{"homeTeam": True, "moneyline": "-120"}, {"homeTeam": False, "moneyline": "+102"}]},
    ],
}


def test_captura_guarda_solo_datos_derivados():
    fila = capturas_de_partido(GAME, "2026-10-25T20:00:00Z")
    assert set(fila) == set(COLUMNAS)
    assert fila["home_team"] == "TBR" and fila["away_team"] == "MIL" and fila["n_casas"] == 2
    assert fila["prob_home_min"] <= fila["prob_home_mercado"] <= fila["prob_home_max"]
    assert "moneyline" not in str(fila)


def test_partido_sin_cuotas():
    assert capturas_de_partido({**GAME, "preGameOdds": []}, "x") is None


def test_cierre_es_la_ultima_captura_antes_del_inicio(tmp_path):
    import pandas as pd

    base = capturas_de_partido(GAME, "2026-10-25T12:00:00Z")
    filas = [base, {**base, "captura_utc": "2026-10-26T23:00:00Z", "prob_home_mercado": 0.55},
             {**base, "captura_utc": "2026-10-27T01:00:00Z", "prob_home_mercado": 0.9}]  # en vivo: no cuenta
    p = tmp_path / "c.csv"
    pd.DataFrame(filas).to_csv(p, index=False)
    c = cierre_por_partido(p)
    assert len(c) == 1 and c.iloc[0]["prob_home_mercado"] == 0.55
