"""
HistorialPitcheo: forma reciente del abridor y cansancio del bullpen sin mirar al futuro.
"""

import numpy as np
import pandas as pd
import pytest

from mlb_feature_engineering import FEATURES_PITCHEO_RECIENTE, HistorialPitcheo, normalizar_nombre


def _fila(game_pk, fecha, team, pitcher, orden, outs, er=0, h=0, bb=0, so=0, bf=0, pitches=0):
    return {"game_pk": game_pk, "fecha": fecha, "team": team, "pitcher": pitcher, "orden": orden,
            "es_abridor": int(orden == 0), "outs": outs, "er": er, "h": h, "bb": bb, "so": so, "bf": bf, "pitches": pitches}


@pytest.fixture
def historial():
    logs = pd.DataFrame([
        _fila(1, "2026-04-01", "NYY", "Gerrit Cole", 0, outs=18, er=2, h=5, bb=1, so=8, bf=24, pitches=95),
        _fila(1, "2026-04-01", "NYY", "Clay Holmes", 1, outs=3, pitches=15),
        _fila(2, "2026-04-06", "NYY", "Gerrit Cole", 0, outs=21, er=1, h=4, bb=0, so=10, bf=25, pitches=100),
        _fila(3, "2026-04-08", "NYY", "Clay Holmes", 1, outs=6, pitches=30),
        _fila(4, "2026-04-09", "NYY", "Luke Weaver", 1, outs=3, pitches=20),
        _fila(5, "2026-04-12", "NYY", "Gerrit Cole", 0, outs=9, er=6, h=8, bb=3, so=2, bf=20, pitches=80),
    ])
    return HistorialPitcheo(logs)


def test_normalizar_nombre():
    assert normalizar_nombre("José Ramírez Jr.") == "jose ramirez"
    assert normalizar_nombre("Ke'Bryan Hayes") == "kebryan hayes"
    assert normalizar_nombre(None) == ""


def test_forma_del_abridor_solo_con_salidas_anteriores(historial):
    f = historial.features("NYY", "BOS", "Gerrit Cole", "Desconocido", "2026-04-12")
    # solo cuentan las salidas del 1 y del 6 de abril (la del 12 es el propio partido)
    assert f["home_sp_era_3"] == pytest.approx(3 * 27 / 39)
    assert f["home_sp_whip_3"] == pytest.approx((9 + 1) * 3 / 39)
    assert f["home_sp_kbb_3"] == pytest.approx((18 - 1) / 49)
    assert f["home_sp_outs_3"] == pytest.approx(19.5)
    assert f["home_sp_descanso"] == 6
    # abridor sin historial: faltante, no cero
    assert np.isnan(f["away_sp_era_3"]) and np.isnan(f["away_sp_descanso"])


def test_cansancio_del_bullpen_ultimos_3_dias(historial):
    # ventana para el 10-abr: 7, 8 y 9 de abril -> 30 + 20 lanzamientos, 6 + 3 outs
    f = historial.features("NYY", "BOS", "Gerrit Cole", "X", "2026-04-10")
    assert f["home_bullpen_pitches_3d"] == 50 and f["home_bullpen_outs_3d"] == 9
    assert f["away_bullpen_pitches_3d"] == 0
    # el bullpen del propio día no cuenta
    assert historial.features("NYY", "BOS", "X", "Y", "2026-04-08")["home_bullpen_pitches_3d"] == 0


def test_features_dataset(historial):
    df = pd.DataFrame([{"home_team": "NYY", "away_team": "BOS", "home_pitcher": "Gerrit Cole", "away_pitcher": "X", "fecha": "2026-04-12"}])
    out = historial.features_dataset(df)
    assert list(out.columns) == FEATURES_PITCHEO_RECIENTE and len(out) == 1
