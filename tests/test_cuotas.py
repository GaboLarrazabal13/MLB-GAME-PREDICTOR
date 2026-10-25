"""
Conversión de cuotas americanas a probabilidad y captura derivada (sin cuotas en bruto).
"""

import pytest

from mlb_cuotas import (
    COLUMNAS,
    capturas_de_partido,
    cierre_por_partido,
    combinar_probabilidades,
    prob_implicita,
    prob_mercado_para,
    prob_sin_margen,
)


def test_combinar_probabilidades():
    assert combinar_probabilidades(0.6, 0.6) == pytest.approx(0.6)
    assert combinar_probabilidades(0.5, 0.5) == pytest.approx(0.5)
    p = combinar_probabilidades(0.70, 0.55)
    assert 0.55 < p < 0.70
    assert combinar_probabilidades(0.7, 0.3, peso_mercado=0.0) == pytest.approx(0.7)
    assert combinar_probabilidades(0.7, 0.3, peso_mercado=1.0) == pytest.approx(0.3)


def test_prob_mercado_para_usa_la_ultima_captura_previa(tmp_path):
    import pandas as pd

    base = {"game_pk": 1, "fecha": "2027-04-01", "inicio_utc": "2027-04-01T23:05:00Z", "home_team": "NYY", "away_team": "BOS",
            "n_casas": 2, "prob_home_min": 0.5, "prob_home_max": 0.6}
    filas = [{**base, "captura_utc": "2027-04-01T12:00:00Z", "prob_home_mercado": 0.55},
             {**base, "captura_utc": "2027-04-01T16:00:00Z", "prob_home_mercado": 0.58},
             {**base, "captura_utc": "2027-04-02T00:00:00Z", "prob_home_mercado": 0.95}]  # en vivo
    p = tmp_path / "c.csv"
    pd.DataFrame(filas).to_csv(p, index=False)
    assert prob_mercado_para("2027-04-01", "NYY", "BOS", path=p) == 0.58
    assert prob_mercado_para("2027-04-01", "NYY", "BOS", path=p, ahora_utc="2027-04-01T13:00:00Z") == 0.55
    assert prob_mercado_para("2027-04-01", "BOS", "NYY", path=p) is None
    assert prob_mercado_para("2027-04-01", "NYY", "BOS", path=tmp_path / "no.csv") is None


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
