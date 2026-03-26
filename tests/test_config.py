"""
Tests para mlb_config.py
"""

import pytest
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlb_config import (
    get_team_code,
    get_team_name,
    TEAM_CODE_TO_NAME,
    TEAM_NAME_TO_CODE,
    validate_config,
    SCRAPING_CONFIG,
    MODEL_CONFIG,
)


class TestTeamMapping:
    """Tests para mapeo de equipos"""

    def test_get_team_code_with_code(self):
        """Test get_team_code con código válido"""
        assert get_team_code("NYY") == "NYY"
        assert get_team_code("BOS") == "BOS"
        assert get_team_code("LAD") == "LAD"

    def test_get_team_code_with_name(self):
        """Test get_team_code con nombre completo"""
        assert get_team_code("New York Yankees") == "NYY"
        assert get_team_code("Boston Red Sox") == "BOS"

    def test_get_team_code_with_short_name(self):
        """Test get_team_code con nombre corto"""
        assert get_team_code("Yankees") == "NYY"
        assert get_team_code("Red Sox") == "BOS"

    def test_get_team_code_case_insensitive(self):
        """Test get_team_code es case-insensitive"""
        assert get_team_code("nyy") == "NYY"
        assert get_team_code("YANKEES") == "NYY"

    def test_get_team_code_invalid(self):
        """Test get_team_code con equipo inválido"""
        assert get_team_code("INVALID_TEAM") is None
        assert get_team_code("") is None
        assert get_team_code(None) is None

    def test_get_team_name(self):
        """Test get_team_name"""
        assert get_team_name("NYY") == "New York Yankees"
        assert get_team_name("BOS") == "Boston Red Sox"

    def test_get_team_name_invalid(self):
        """Test get_team_name con código inválido"""
        assert get_team_name("XXX") == "XXX"
        assert get_team_name(None) is None

    def test_all_teams_have_names(self):
        """Test que todos los equipos tienen nombre"""
        assert len(TEAM_CODE_TO_NAME) == 30
        for code, name in TEAM_CODE_TO_NAME.items():
            assert name is not None
            assert len(name) > 0


class TestConfiguration:
    """Tests para configuración"""

    def test_scraping_config_exists(self):
        """Test que scraping config existe"""
        assert SCRAPING_CONFIG is not None
        assert "max_retries" in SCRAPING_CONFIG
        assert "timeout" in SCRAPING_CONFIG

    def test_model_config_exists(self):
        """Test que model config existe"""
        assert MODEL_CONFIG is not None
        assert "test_size" in MODEL_CONFIG
        assert "random_state" in MODEL_CONFIG

    def test_validate_config(self):
        """Test validación de configuración"""
        errors = validate_config()
        # No debería haber errores críticos
        assert isinstance(errors, list)


class TestFeaturesLists:
    """Tests para listas de features"""

    def test_temporal_features_exist(self):
        """Test que features temporales existen"""
        from mlb_config import TEMPORAL_FEATURES

        assert len(TEMPORAL_FEATURES) > 0
        assert "home_win_rate_10" in TEMPORAL_FEATURES

    def test_scraping_features_exist(self):
        """Test que features de scraping existen"""
        from mlb_config import SCRAPING_FEATURES

        assert len(SCRAPING_FEATURES) > 0

    def test_super_features_exist(self):
        """Test que super features existen"""
        from mlb_config import SUPER_FEATURES

        assert len(SUPER_FEATURES) > 0
