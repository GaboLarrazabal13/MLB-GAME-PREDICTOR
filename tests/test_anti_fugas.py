"""
Anti-fugas: ninguna feature de un partido puede depender de su propio resultado ni de partidos
posteriores. Se comprueba con los datos reales versionados: las features de un partido calculadas
con toda la historia deben ser idénticas a las calculadas con la historia cortada antes de su fecha,
y no deben cambiar si se invierte su resultado.
"""

import numpy as np
import pandas as pd
import pytest

from mlb_config import PARTIDOS_HISTORICOS_PATH
from mlb_feature_engineering import (
    FEATURES_CRONOLOGICAS,
    HistorialPitcheo,
    TrackerCronologico,
    features_cronologicas_dataset,
)

pytestmark = pytest.mark.skipif(not __import__("os").path.exists(PARTIDOS_HISTORICOS_PATH), reason="sin datos históricos")


@pytest.fixture(scope="module")
def partidos():
    df = pd.read_csv(PARTIDOS_HISTORICOS_PATH, dtype={"fecha": str})
    return df[df["year"] == 2024].sort_values(["fecha", "game_id"], kind="stable").reset_index(drop=True)


@pytest.fixture(scope="module")
def features_completas(partidos):
    return features_cronologicas_dataset(partidos)


def _muestra(partidos, n=25, seed=3):
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(np.arange(200, len(partidos)), size=n, replace=False))


def test_futuro_no_cambia_las_features(partidos, features_completas):
    for i in _muestra(partidos):
        fila = partidos.iloc[i]
        tracker = TrackerCronologico().registrar_dataframe(partidos, hasta_fecha=fila.fecha)
        corto = tracker.features_previas(fila.home_team, fila.away_team, fila.fecha, fila.year)
        completo = features_completas.iloc[i].to_dict()
        assert corto == pytest.approx(completo), f"fuga en {fila.game_id}"


def test_el_propio_resultado_no_cambia_las_features(partidos, features_completas):
    for i in _muestra(partidos, seed=4):
        alterado = partidos.copy()
        alterado.loc[i, "ganador"] = 1 - alterado.loc[i, "ganador"]
        alterado.loc[i, ["score_home", "score_away"]] = alterado.loc[i, ["score_away", "score_home"]].to_numpy()
        f = features_cronologicas_dataset(alterado).iloc[i]
        assert f[FEATURES_CRONOLOGICAS].to_dict() == pytest.approx(features_completas.iloc[i][FEATURES_CRONOLOGICAS].to_dict())


def test_pitcheo_reciente_ignora_el_dia_del_partido_y_lo_posterior():
    base = pd.DataFrame([
        {"game_pk": 1, "fecha": "2024-05-01", "team": "NYY", "pitcher": "Gerrit Cole", "orden": 0, "es_abridor": 1,
         "outs": 18, "er": 2, "h": 5, "bb": 1, "so": 7, "bf": 24, "pitches": 95},
        {"game_pk": 1, "fecha": "2024-05-01", "team": "NYY", "pitcher": "Clay Holmes", "orden": 1, "es_abridor": 0,
         "outs": 3, "er": 0, "h": 0, "bb": 0, "so": 1, "bf": 3, "pitches": 12},
    ])
    futuro = pd.DataFrame([
        {**base.iloc[0].to_dict(), "game_pk": 2, "fecha": "2024-05-06", "er": 9},  # el propio partido
        {**base.iloc[1].to_dict(), "game_pk": 2, "fecha": "2024-05-06", "pitches": 60},
        {**base.iloc[0].to_dict(), "game_pk": 3, "fecha": "2024-05-11", "er": 0},  # posterior
    ])
    f_base = HistorialPitcheo(base).features("NYY", "BOS", "Gerrit Cole", "X", "2024-05-06")
    f_todo = HistorialPitcheo(pd.concat([base, futuro])).features("NYY", "BOS", "Gerrit Cole", "X", "2024-05-06")
    assert f_base == pytest.approx(f_todo, nan_ok=True)
