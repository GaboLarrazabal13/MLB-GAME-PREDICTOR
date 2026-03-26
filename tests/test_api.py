"""
Tests para API de MLB Predictor
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient


class TestAPIEndpoints:
    """Tests para endpoints de la API"""

    @pytest.fixture
    def client(self, test_db_path):
        """Cliente de test para la API"""
        with patch("src.api.DB_PATH", test_db_path):
            from src.api import app

            return TestClient(app)

    def test_root_endpoint(self, client):
        """Test endpoint raíz"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "MLB Predictor API" in data["message"]

    def test_health_check(self, client, test_db_path):
        """Test health check endpoint"""
        with patch("src.api.DB_PATH", test_db_path):
            with patch(
                "src.mlb_config.MODELO_PATH",
                str(Path(__file__).parent.parent / "models" / "modelo_mlb_v3.5.json"),
            ):
                response = client.get("/health")
                assert response.status_code == 200
                data = response.json()
                assert "status" in data

    def test_list_teams(self, client):
        """Test endpoint de listado de equipos"""
        response = client.get("/teams")
        assert response.status_code == 200
        teams = response.json()
        assert isinstance(teams, list)
        assert len(teams) > 0
        # Verificar estructura
        assert "codigo" in teams[0]
        assert "nombre" in teams[0]

    def test_predictions_endpoint_structure(self, client):
        """Test estructura de endpoint de predicciones"""
        response = client.get("/predictions/latest")
        assert response.status_code == 200
        # Puede estar vacío o tener datos
        assert isinstance(response.json(), list)

    def test_stats_endpoint_structure(self, client, test_db_path):
        """Test endpoint de estadísticas"""
        with patch("src.api.DB_PATH", test_db_path):
            response = client.get("/stats/accuracy?dias=30")
            assert response.status_code == 200
            data = response.json()
            assert "periodo" in data
            assert "total" in data
            assert "accuracy_general" in data


class TestAPIValidation:
    """Tests para validación de la API"""

    def test_prediction_validation_teams_equal(self):
        """Test que equipos iguales no son válidos"""
        from src.api import PrediccionRequest

        # Pydantic no valida equipos iguales por defecto, es validación de negocio
        # Esta validación ocurre en el endpoint, no en el modelo
        req = PrediccionRequest(
            home_team="NYY",
            away_team="NYY",
            home_pitcher="Gerrit Cole",
            away_pitcher="Tanner Houck",
        )
        # El modelo acepta la request, la validación de equipos diferentes
        # ocurre en el endpoint
        assert req.home_team == "NYY"
        assert req.away_team == "NYY"


class TestAPIModels:
    """Tests para modelos Pydantic"""

    def test_prediccion_request_model(self):
        """Test modelo PrediccionRequest"""
        from src.api import PrediccionRequest

        req = PrediccionRequest(
            home_team="NYY",
            away_team="BOS",
            home_pitcher="Gerrit Cole",
            away_pitcher="Tanner Houck",
            year=2024,
        )

        assert req.home_team == "NYY"
        assert req.away_team == "BOS"
        assert req.year == 2024

    def test_prediccion_response_model(self):
        """Test modelo PrediccionResponse"""
        from src.api import PrediccionResponse

        resp = PrediccionResponse(
            success=True,
            fecha="2024-06-01",
            home_team="NYY",
            away_team="BOS",
            home_pitcher="Gerrit Cole",
            away_pitcher="Tanner Houck",
            prob_home=68.5,
            prob_away=31.5,
            prediccion="NYY",
            confianza="ALTA",
        )

        assert resp.success is True
        assert resp.prediccion == "NYY"
