"""
Variables sabermétricas (pitagórica, FIP), modelo logístico y combinación del conjunto.
"""

import numpy as np
import pandas as pd
import pytest

from mlb_config import PARTIDOS_HISTORICOS_PATH
from mlb_cuotas import combinar_lista
from mlb_sabermetria import (
    FEATURES_MODELO_SABER,
    FEATURES_SABER,
    FIP_LIGA,
    RPG_LIGA,
    Sabermetria,
    entrenar_modelo_saber,
    features_sabermetricas_dataset,
    predecir_saber,
)


def _logs(filas):
    return pd.DataFrame(filas, columns=["fecha", "team", "pitcher", "es_abridor", "outs", "hr", "bb", "so"])


def _partidos(filas):
    return pd.DataFrame(filas, columns=["fecha", "year", "home_team", "away_team", "home_pitcher", "away_pitcher", "score_home", "score_away", "ganador"])


class TestFeatures:
    def test_inicio_de_temporada_neutral(self):
        f = Sabermetria(_logs([])).features("NYY", "BOS", "A", "B", "2026-04-01", 2026)
        assert f["home_pitag"] == f["away_pitag"] == 0.5 and f["diff_pitag"] == 0
        assert f["home_rs"] == RPG_LIGA and f["home_fip_sp"] == FIP_LIGA and f["home_fip_bp"] == FIP_LIGA

    def test_temporada_anterior_como_prior_regresado(self):
        s = Sabermetria(_logs([]))
        for _ in range(10):
            s.registrar("NYY", "BOS", 2025, 6, 2)
        f = s.features("NYY", "BOS", "A", "B", "2026-03-27", 2026)
        assert 0.5 < f["home_pitag"] < s.pitag_final[("NYY", 2025)]  # regresado hacia 0.5
        assert f["away_pitag"] < 0.5

    def test_pitagorica_pesa_mas_con_mas_partidos(self):
        s = Sabermetria(_logs([]))
        s.registrar("NYY", "BOS", 2026, 8, 2)
        uno = s.features("NYY", "BOS", "A", "B", "2026-04-02", 2026)["home_pitag"]
        for _ in range(40):
            s.registrar("NYY", "BOS", 2026, 8, 2)
        muchos = s.features("NYY", "BOS", "A", "B", "2026-06-01", 2026)["home_pitag"]
        assert 0.5 < uno < muchos < 1.0

    def test_fip_del_abridor_regresado_y_sin_el_propio_dia(self):
        logs = _logs([
            ("2026-04-01", "NYY", "Gerrit Cole", 1, 18, 0, 0, 10),  # 6 IP, 10 K
            ("2026-04-06", "NYY", "Gerrit Cole", 1, 3, 3, 3, 0),    # el propio día: no cuenta
        ])
        f = Sabermetria(logs).features("NYY", "BOS", "Gerrit Cole", "Nadie", "2026-04-06", 2026)
        crudo = (-2 * 10) / 6 + 3.15
        assert f["home_fip_sp"] == pytest.approx((crudo * 6 + FIP_LIGA * 40) / 46)
        assert f["home_fip_sp"] < FIP_LIGA == f["away_fip_sp"]
        assert f["diff_fip_sp"] > 0  # ventaja local

    def test_fip_bullpen_ventana_30_dias(self):
        logs = _logs([
            ("2026-03-01", "NYY", "Viejo", 0, 30, 5, 5, 0),     # fuera de la ventana
            ("2026-04-10", "NYY", "Relevista", 0, 9, 0, 0, 6),
        ])
        f = Sabermetria(logs).features("NYY", "BOS", "A", "B", "2026-04-20", 2026)
        crudo = (-2 * 6) / 3 + 3.15
        assert f["home_fip_bp"] == pytest.approx((crudo * 3 + FIP_LIGA * 30) / 33)

    def test_dataset_partidos_del_mismo_dia_no_se_ven(self):
        df = _partidos([
            ("2026-04-01", 2026, "NYY", "BOS", "A", "B", 9, 0, 1),
            ("2026-04-01", 2026, "NYY", "TOR", "A", "C", 9, 0, 1),
            ("2026-04-02", 2026, "NYY", "BOS", "A", "B", 1, 0, 1),
        ])
        f = features_sabermetricas_dataset(df, _logs([]))
        assert list(f.columns) == FEATURES_SABER
        assert f.loc[0, "home_pitag"] == f.loc[1, "home_pitag"] == 0.5
        assert f.loc[2, "home_pitag"] > 0.5


@pytest.mark.skipif(not __import__("os").path.exists(PARTIDOS_HISTORICOS_PATH), reason="sin datos históricos")
def test_anti_fuga_con_datos_reales():
    df = pd.read_csv(PARTIDOS_HISTORICOS_PATH, dtype={"fecha": str})
    df = df[df["year"] == 2024].sort_values("fecha", kind="stable").reset_index(drop=True)
    completo = features_sabermetricas_dataset(df, _logs([]))
    rng = np.random.default_rng(1)
    for i in rng.choice(np.arange(300, len(df)), size=15, replace=False):
        r = df.iloc[i]
        corto = Sabermetria(_logs([])).registrar_dataframe(df, hasta_fecha=r["fecha"]).features(
            r["home_team"], r["away_team"], r["home_pitcher"], r["away_pitcher"], r["fecha"], r["year"])
        assert corto == pytest.approx(completo.iloc[i].to_dict())


def test_modelo_aprende_la_direccion(tmp_path):
    rng = np.random.default_rng(0)
    n = 3000
    d = pd.DataFrame({c: rng.normal(0, 0.1, n) for c in FEATURES_MODELO_SABER})
    d["ganador"] = (rng.random(n) < 1 / (1 + np.exp(-(0.1 + 6 * d["diff_pitag"])))).astype(int)
    d["year"], d["fecha"] = 2026, "2026-09-01"
    modelo, card = entrenar_modelo_saber(d, path=tmp_path / "m.pkl", card_path=tmp_path / "m.json")
    assert card["coeficientes_estandarizados"]["diff_pitag"] > 0.3
    fuerte = predecir_saber({**dict.fromkeys(FEATURES_MODELO_SABER, 0.0), "diff_pitag": 0.2}, modelo)
    debil = predecir_saber({**dict.fromkeys(FEATURES_MODELO_SABER, 0.0), "diff_pitag": -0.2}, modelo)
    assert fuerte > 0.5 > debil


def test_combinar_lista():
    assert combinar_lista([0.6, None, 0.6]) == pytest.approx(0.6)
    assert combinar_lista([None, None]) is None
    assert combinar_lista([0.7]) == pytest.approx(0.7)
    p = combinar_lista([0.7, 0.5, 0.6])
    assert 0.5 < p < 0.7
