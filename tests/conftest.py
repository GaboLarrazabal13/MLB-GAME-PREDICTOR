"""
Fixtures compartidos para tests de MLB Predictor
"""

import os
import sys
import sqlite3
import pytest
from pathlib import Path

# Agregar src al path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope="session")
def test_db_path(tmp_path_factory):
    """Crea una base de datos SQLite temporal para tests"""
    tmp_dir = tmp_path_factory.mktemp("test_data")
    db_path = tmp_dir / "test_mlb.db"

    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    # Crear tablas necesarias
    c.execute("""
        CREATE TABLE IF NOT EXISTS historico_real (
            game_id TEXT PRIMARY KEY,
            fecha TEXT,
            home_team TEXT,
            away_team TEXT,
            home_pitcher TEXT,
            away_pitcher TEXT,
            score_home INTEGER,
            score_away INTEGER,
            ganador INTEGER
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS predicciones_historico (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT,
            home_team TEXT,
            away_team TEXT,
            home_pitcher TEXT,
            away_pitcher TEXT,
            prob_home REAL,
            prob_away REAL,
            prediccion TEXT,
            confianza TEXT,
            tipo TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS historico_partidos (
            game_id TEXT PRIMARY KEY,
            fecha TEXT,
            home_team TEXT,
            away_team TEXT,
            home_pitcher TEXT,
            away_pitcher TEXT
        )
    """)

    # Insertar datos de prueba
    c.execute("""
        INSERT INTO historico_real VALUES 
        ('test_001', '2024-06-01', 'NYY', 'BOS', 'Gerrit Cole', 'Tanner Houck', 5, 3, 1),
        ('test_002', '2024-06-02', 'LAD', 'SFG', 'Clayton Kershaw', 'Logan Webb', 4, 2, 1),
        ('test_003', '2024-06-03', 'CHC', 'STL', 'Justin Steele', 'Sonny Gray', 2, 5, 0)
    """)

    c.execute("""
        INSERT INTO predicciones_historico VALUES 
        (1, '2024-06-01', 'NYY', 'BOS', 'Gerrit Cole', 'Tanner Houck', 68.5, 31.5, 'NYY', 'ALTA', 'MANUAL'),
        (2, '2024-06-02', 'LAD', 'SFG', 'Clayton Kershaw', 'Logan Webb', 72.0, 28.0, 'LAD', 'ALTA', 'MANUAL')
    """)

    conn.commit()
    conn.close()

    yield str(db_path)

    # Cleanup - cerrar cualquier conexión abierta
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
    except PermissionError:
        # En Windows, el archivo puede estar bloqueado
        pass


@pytest.fixture
def mock_team_codes():
    """Códigos de equipos de prueba"""
    return {
        "NYY": "New York Yankees",
        "BOS": "Boston Red Sox",
        "LAD": "Los Angeles Dodgers",
        "SFG": "San Francisco Giants",
        "CHC": "Chicago Cubs",
        "STL": "St. Louis Cardinals",
    }


@pytest.fixture
def sample_prediction_data():
    """Datos de predicción de ejemplo"""
    return {
        "fecha": "2024-06-01",
        "home_team": "NYY",
        "away_team": "BOS",
        "home_pitcher": "Gerrit Cole",
        "away_pitcher": "Tanner Houck",
        "prob_home": 68.5,
        "prob_away": 31.5,
        "prediccion": "NYY",
        "confianza": "ALTA",
    }


@pytest.fixture
def sample_features_dict():
    """Features de ejemplo para testing"""
    return {
        "home_win_rate_10": 0.7,
        "home_racha": 5,
        "home_runs_avg": 5.2,
        "home_runs_diff": 1.5,
        "away_win_rate_10": 0.4,
        "away_racha": -3,
        "away_runs_avg": 3.8,
        "away_runs_diff": -1.2,
        "home_team_OPS": 0.782,
        "away_team_OPS": 0.758,
        "home_starter_ERA": 3.12,
        "away_starter_ERA": 3.89,
        "home_starter_WHIP": 1.089,
        "away_starter_WHIP": 1.234,
    }
