"""
TrackerCronologico: ELO con arrastre entre temporadas, descanso y tendencias sin mirar al futuro.
"""

import pandas as pd

from mlb_config import DESCANSO_MAX_DIAS, ELO_CARRYOVER, MODEL_FEATURES
from mlb_feature_engineering import FEATURES_CRONOLOGICAS, TrackerCronologico, features_cronologicas_dataset


def _df(filas):
    return pd.DataFrame(filas, columns=["fecha", "home_team", "away_team", "year", "ganador", "score_home", "score_away"])


class TestTracker:
    def test_primer_partido_es_neutral(self):
        f = TrackerCronologico().features_previas("NYY", "BOS", "2026-04-01", 2026)
        assert f["home_elo"] == f["away_elo"] == 1500.0 and f["diff_elo"] == 0
        assert f["home_descanso"] == f["away_descanso"] == DESCANSO_MAX_DIAS
        assert f["home_win_rate_10"] == 0.5 and f["home_racha"] == 0

    def test_elo_sube_al_ganador_y_baja_al_perdedor(self):
        t = TrackerCronologico()
        t.registrar("NYY", "BOS", "2026-04-01", 2026, ganador=1, score_home=5, score_away=2)
        f = t.features_previas("BOS", "NYY", "2026-04-02", 2026)
        assert f["away_elo"] > 1500.0 > f["home_elo"]
        assert f["home_descanso"] == 1 and f["away_descanso"] == 1
        assert f["away_win_rate_10"] == 1.0 and f["away_racha"] == 1 and f["home_racha"] == -1

    def test_nombres_y_codigos_se_unifican(self):
        t = TrackerCronologico()
        t.registrar("New York Yankees", "Boston Red Sox", "2025-04-01", 2025, ganador=1)
        assert t.features_previas("NYY", "BOS", "2025-04-02", 2025)["home_elo"] > 1500.0

    def test_cambio_de_temporada_regresa_hacia_1500_y_resetea_descanso(self):
        t = TrackerCronologico()
        for d in range(1, 21):
            t.registrar("NYY", "BOS", f"2025-04-{d:02d}", 2025, ganador=1)
        elo_fin = t.features_previas("NYY", "BOS", "2025-04-21", 2025)["home_elo"]
        f = t.features_previas("NYY", "BOS", "2026-03-26", 2026)
        assert abs(f["home_elo"] - (1500 + (elo_fin - 1500) * ELO_CARRYOVER)) < 1e-9
        assert 1500 < f["home_elo"] < elo_fin
        assert f["home_descanso"] == DESCANSO_MAX_DIAS
        assert f["home_win_rate_10"] == 0.5  # las tendencias sí empiezan de cero cada temporada

    def test_descanso_se_topa(self):
        t = TrackerCronologico()
        t.registrar("NYY", "BOS", "2026-04-01", 2026, ganador=1)
        assert t.features_previas("NYY", "TOR", "2026-04-30", 2026)["home_descanso"] == DESCANSO_MAX_DIAS

    def test_registrar_dataframe_respeta_hasta_fecha(self):
        df = _df([
            ("2026-04-01", "NYY", "BOS", 2026, 1, 4, 1),
            ("2026-04-02", "NYY", "BOS", 2026, 1, 4, 1),
            ("2026-04-03", "NYY", "BOS", 2026, 1, 4, 1),
        ])
        t = TrackerCronologico().registrar_dataframe(df, hasta_fecha="2026-04-02")
        assert len(t.historial[("NYY", 2026)]) == 1

    def test_dataset_no_mira_al_futuro_y_coincide_con_inferencia(self):
        df = _df([
            ("2026-04-01", "NYY", "BOS", 2026, 1, 4, 1),
            ("2026-04-02", "BOS", "NYY", 2026, 0, 2, 6),
            ("2026-04-04", "NYY", "TOR", 2026, 0, 1, 3),
        ])
        feats = features_cronologicas_dataset(df)
        assert list(feats.columns) == FEATURES_CRONOLOGICAS and len(feats) == 3
        assert feats.loc[0, "home_elo"] == 1500.0
        # la inferencia del tercer partido, hecha con el histórico anterior, da exactamente lo mismo
        inf = TrackerCronologico().registrar_dataframe(df, hasta_fecha="2026-04-04").features_previas("NYY", "TOR", "2026-04-04", 2026)
        assert inf == feats.loc[2].to_dict()
        assert inf["home_descanso"] == 2 and inf["away_descanso"] == DESCANSO_MAX_DIAS

    def test_model_features_incluye_elo_y_descanso(self):
        assert {"home_elo", "away_elo", "diff_elo", "home_descanso", "away_descanso"} <= set(MODEL_FEATURES)
