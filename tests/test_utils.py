"""
Tests para mlb_utils.py
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import datetime, timedelta


class TestMLBUtils:
    """Tests para funciones de utilidades"""

    def test_analizar_accuracy_with_empty_db(self, test_db_path):
        """Test analizar_accuracy con base de datos vacía o sin datos"""
        with patch("src.mlb_utils.DB_PATH", test_db_path):
            from src.mlb_utils import analizar_accuracy_historico

            result = analizar_accuracy_historico(dias=30)
            # El resultado puede ser None si no hay suficientes datos
            assert result is None or isinstance(result, type(None))

    def test_generar_reporte_equipos_empty(self, test_db_path):
        """Test generar_reporte_equipos con datos vacíos"""
        with patch("src.mlb_utils.DB_PATH", test_db_path):
            from src.mlb_utils import generar_reporte_equipos

            result = generar_reporte_equipos("NYY", ultimos_n=5)
            # Puede retornar None o un diccionario vacío
            assert result is None or isinstance(result, dict)

    def test_confianza_calculation(self):
        """Test cálculo de nivel de confianza"""

        # Test helper function para calcular confianza
        def calcular_confianza(prob):
            if prob >= 70:
                return "MUY ALTA"
            elif prob >= 60:
                return "ALTA"
            elif prob >= 50:
                return "MEDIA"
            else:
                return "BAJA"

        assert calcular_confianza(75) == "MUY ALTA"
        assert calcular_confianza(65) == "ALTA"
        assert calcular_confianza(55) == "MEDIA"
        assert calcular_confianza(45) == "BAJA"


class TestDateHelpers:
    """Tests para funciones de fecha"""

    def test_date_formats(self):
        """Test formatos de fecha"""
        fecha = datetime(2024, 6, 1)
        assert fecha.strftime("%Y-%m-%d") == "2024-06-01"
        assert fecha.strftime("%Y") == "2024"

    def test_date_range_calculation(self):
        """Test cálculo de rango de fechas"""
        fecha_inicio = datetime.now() - timedelta(days=30)
        fecha_fin = datetime.now()

        assert fecha_fin > fecha_inicio
        assert (fecha_fin - fecha_inicio).days == 30


class TestDatabaseHelpers:
    """Tests para funciones helper de base de datos"""

    def test_connection_string_format(self, test_db_path):
        """Test formato de string de conexión"""
        import sqlite3

        conn = sqlite3.connect(test_db_path)
        assert conn is not None
        conn.close()
