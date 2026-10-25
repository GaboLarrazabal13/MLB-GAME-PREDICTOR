"""
Tests del reentrenamiento champion/challenger (mlb_training) y de la limpieza de features.
"""

import sqlite3
from datetime import date

import numpy as np
import pandas as pd
import pytest

from mlb_config import MODEL_FEATURES, SCRAPING_FEATURES
from mlb_feature_engineering import preparar_matriz_modelo
from mlb_training import (
    debe_entrenar,
    decidir_promocion,
    definir_ventana_sombra,
    evaluar_champion,
    folds_temporales,
    metricas,
    pesos_recencia,
)


class TestPrepararMatriz:
    def test_orden_y_columnas(self):
        df = pd.DataFrame([{"home_team_OPS": 0.75, "basura": 1}])
        X = preparar_matriz_modelo(df)
        assert list(X.columns) == MODEL_FEATURES
        assert "basura" not in X.columns
        assert {"home_elo", "home_descanso"} <= set(X.columns)

    def test_ceros_de_estadisticas_son_faltantes(self):
        fila = dict.fromkeys(MODEL_FEATURES, 1.0)
        fila["home_starter_ERA"] = 0.0
        fila["home_racha"] = 0  # una racha 0 sí es un valor real
        X = preparar_matriz_modelo(pd.DataFrame([fila]))
        assert np.isnan(X.loc[0, "home_starter_ERA"])
        assert X.loc[0, "home_racha"] == 0
        # la super feature que depende del ERA ausente también queda como faltante
        assert np.isnan(X.loc[0, "super_resistencia_era_ops"])
        assert X.loc[0, "super_muro_bullpen"] == 1.0

    def test_modo_legado_conserva_ceros(self):
        fila = dict.fromkeys(MODEL_FEATURES, 0.0)
        X = preparar_matriz_modelo(pd.DataFrame([fila]), nan_missing=False)
        assert not X.isna().any().any()
        assert (X.loc[0, SCRAPING_FEATURES] == 0).all()


class TestVentanaSombra:
    def test_ultimos_30_dias(self):
        fechas = pd.date_range("2026-04-01", "2026-09-30", freq="D").strftime("%Y-%m-%d").repeat(10)
        inicio, fin = definir_ventana_sombra(fechas, dias=30, min_partidos=150)
        assert fin == "2026-09-30"
        assert inicio == "2026-09-01"

    def test_se_amplia_si_faltan_partidos(self):
        fechas = pd.date_range("2026-04-01", "2026-09-30", freq="D").strftime("%Y-%m-%d")  # 1 partido/día
        inicio, fin = definir_ventana_sombra(fechas, dias=30, min_partidos=150)
        assert (pd.Timestamp(fin) - pd.Timestamp(inicio)).days + 1 >= 150

    def test_no_pasa_del_primer_partido(self):
        fechas = ["2026-04-01", "2026-04-02", "2026-04-03"]
        inicio, fin = definir_ventana_sombra(fechas, dias=30, min_partidos=150)
        assert inicio == "2026-04-01"


class TestDecision:
    champ = {"n": 200, "accuracy": 0.580, "logloss": 0.680, "brier": 0.24, "roc_auc": 0.6}

    def test_promueve_si_mejora_logloss_sin_perder_accuracy(self):
        ok, _ = decidir_promocion(self.champ, {**self.champ, "logloss": 0.670, "accuracy": 0.575})
        assert ok

    def test_no_promueve_si_empeora_logloss(self):
        ok, razon = decidir_promocion(self.champ, {**self.champ, "logloss": 0.690, "accuracy": 0.60})
        assert not ok and "logloss" in razon

    def test_no_promueve_si_pierde_mas_de_un_punto_de_accuracy(self):
        ok, razon = decidir_promocion(self.champ, {**self.champ, "logloss": 0.660, "accuracy": 0.565})
        assert not ok and "accuracy" in razon

    def test_sin_champion_honesto_retiene(self):
        ok, razon = decidir_promocion(None, self.champ)
        assert not ok and "champion" in razon


class TestCadencia:
    @pytest.fixture
    def conn(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE historico_real (game_id TEXT, fecha TEXT, ganador INTEGER)")
        conn.executemany(
            "INSERT INTO historico_real VALUES (?, ?, ?)",
            [(f"g{i}", f"2026-07-{(i % 28) + 1:02d}", i % 2) for i in range(120)],
        )
        return conn

    def test_sin_model_card_entrena(self, conn):
        assert debe_entrenar(conn, None)[0]

    def test_respeta_minimo_de_dias(self, conn):
        card = {"trained_at": "2026-07-25T00:00:00+00:00", "trained_through": "2026-06-30"}
        ok, razon = debe_entrenar(conn, card, hoy=date(2026, 7, 28))
        assert not ok and "días" in razon

    def test_respeta_minimo_de_partidos_nuevos(self, conn):
        card = {"trained_at": "2026-07-01T00:00:00+00:00", "trained_through": "2026-07-27"}
        ok, razon = debe_entrenar(conn, card, hoy=date(2026, 8, 15))
        assert not ok and "partidos nuevos" in razon

    def test_entrena_cuando_toca(self, conn):
        card = {"trained_at": "2026-07-01T00:00:00+00:00", "trained_through": "2026-06-30"}
        ok, _ = debe_entrenar(conn, card, hoy=date(2026, 8, 15))
        assert ok


class TestChampionHonesto:
    def _sombra(self, n=200):
        return pd.DataFrame(
            {
                "fecha": [f"2026-09-{(i % 30) + 1:02d}" for i in range(n)],
                "home_team": [f"H{i}" for i in range(n)],
                "away_team": [f"A{i}" for i in range(n)],
                "ganador": [i % 2 for i in range(n)],
            }
        )

    def test_usa_predicciones_publicadas(self):
        sombra = self._sombra()
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE predicciones_historico (fecha TEXT, home_team TEXT, away_team TEXT, prob_home REAL, model_version TEXT)")
        conn.executemany(
            "INSERT INTO predicciones_historico VALUES (?, ?, ?, ?, ?)",
            [(r.fecha, r.home_team, r.away_team, 70.0 if r.ganador else 30.0, "v1") for r in sombra.itertuples()],
        )
        m, fuente, comunes = evaluar_champion(conn, sombra, {"version": "v1"}, modelo=None)
        assert fuente == "predicciones"
        assert m["accuracy"] == 1.0 and len(comunes) == 200

    def test_sin_predicciones_ni_modelo_no_hay_metricas(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE predicciones_historico (fecha TEXT, home_team TEXT, away_team TEXT, prob_home REAL)")
        m, fuente, _ = evaluar_champion(conn, self._sombra(), {"trained_through": "2026-09-30"}, modelo=None)
        assert m is None and fuente is None


class TestUtilidades:
    def test_folds_temporales_no_miran_al_futuro(self):
        for tr, va in folds_temporales(1000, n_folds=3, fraccion_val=0.1):
            assert tr.max() < va.min()
        assert len(folds_temporales(1000)) == 3

    def test_pesos_recencia(self):
        assert pesos_recencia([2026, 2025, 2024, 2023]).tolist() == [2.5, 1.3, 1.0, 1.0]

    def test_metricas(self):
        m = metricas([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.4])
        assert m["accuracy"] == 1.0 and 0 < m["logloss"] < 0.5 and m["n"] == 4
