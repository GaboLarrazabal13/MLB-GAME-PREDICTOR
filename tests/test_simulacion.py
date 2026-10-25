"""
Simulación de temporada y playoffs.
"""

import numpy as np
import pytest

from mlb_config import DIVISION_DE, DIVISIONES
from mlb_simulacion import jugar_llave, prob_local, sembrar, simular


def test_divisiones_completas():
    assert len(DIVISION_DE) == 30 and all(len(t) == 5 for t in DIVISIONES.values())


def test_prob_local():
    assert prob_local(1500, 1500) > 0.5  # ventaja de localía
    assert prob_local(1600, 1400) > prob_local(1500, 1500) > prob_local(1400, 1600)


def test_siembra_campeones_de_division_primero():
    v = dict.fromkeys(DIVISION_DE, 70)
    v.update({"NYY": 100, "BOS": 99, "TBR": 98, "CLE": 80, "HOU": 81})  # AL Este muy fuerte
    s = sembrar(v, {})
    assert s["AL"][:3] == ["NYY", "HOU", "CLE"]  # campeones de división por récord
    assert s["AL"][3:5] == ["BOS", "TBR"]  # comodines aunque tengan más victorias que HOU
    assert len(set(s["AL"]) | set(s["NL"])) == 12


def test_llave_produce_un_campeon_de_los_sembrados():
    v = dict.fromkeys(DIVISION_DE, 80)
    elo = dict.fromkeys(DIVISION_DE, 1500.0)
    s = sembrar(v, elo, np.random.default_rng(1))
    campeon, ligas = jugar_llave(s, elo, v, np.random.default_rng(1))
    assert campeon in ligas.values()
    assert ligas["AL"] in s["AL"] and ligas["NL"] in s["NL"]


def test_temporada_terminada_probabilidades_coherentes():
    v = dict.fromkeys(DIVISION_DE, 75)
    v["LAD"] = 105
    elo = dict.fromkeys(DIVISION_DE, 1500.0)
    elo["LAD"] = 1650.0
    r = simular(v, [], elo, n=2000)
    assert r["LAD"]["playoffs"] == 1.0 and r["LAD"]["division"] == 1.0
    assert sum(t["campeon"] for t in r.values()) == pytest.approx(1.0)
    assert sum(t["playoffs"] for t in r.values()) == pytest.approx(12.0)
    assert r["LAD"]["campeon"] == max(t["campeon"] for t in r.values())


def test_partidos_pendientes_suman_victorias():
    v = dict.fromkeys(DIVISION_DE, 0)
    elo = dict.fromkeys(DIVISION_DE, 1500.0)
    r = simular(v, [("NYY", "BOS")] * 10, elo, n=500)
    assert r["NYY"]["victorias_esperadas"] + r["BOS"]["victorias_esperadas"] == pytest.approx(10)
    assert r["NYY"]["victorias_esperadas"] > 5  # localía
