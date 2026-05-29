"""
MLB Game Predictor V3.5 - Aplicación Streamlit
Ejecutar: streamlit run app.py
"""

import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Importar configuración del proyecto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from mlb_config import DB_PATH, TEAM_CODE_TO_NAME, get_team_code
except Exception as e:
    TEAM_CODE_TO_NAME = {}
    DB_PATH = "./data/mlb_reentrenamiento.db"
    def get_team_code(name):
        return name
    st.warning(f"Config fallback due to error: {e}")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

st.set_page_config(
    page_title="MLB Predictor Pro V4.0",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme_selector" not in st.session_state:
    st.session_state.theme_selector = "🌙 Oscuro"

theme = "Oscuro" if "🌙 Oscuro" in st.session_state.theme_selector else "Claro"

# API Configuration
API_URL = os.getenv("API_URL")
has_secrets_api = False
if not API_URL:
    try:
        API_URL = st.secrets.get("API_URL", "http://localhost:8000")
        has_secrets_api = "API_URL" in st.secrets
    except Exception:
        API_URL = "http://localhost:8000"


# Logos MLB oficiales
EQUIPOS_MLB = {
    "ARI": {
        "nombre": "Arizona Diamondbacks",
        "logo": "https://www.mlbstatic.com/team-logos/109.svg",
        "color": "#A71930",
    },
    "ATL": {
        "nombre": "Atlanta Braves",
        "logo": "https://www.mlbstatic.com/team-logos/144.svg",
        "color": "#CE1141",
    },
    "BAL": {
        "nombre": "Baltimore Orioles",
        "logo": "https://www.mlbstatic.com/team-logos/110.svg",
        "color": "#DF4601",
    },
    "BOS": {
        "nombre": "Boston Red Sox",
        "logo": "https://www.mlbstatic.com/team-logos/111.svg",
        "color": "#BD3039",
    },
    "CHC": {
        "nombre": "Chicago Cubs",
        "logo": "https://www.mlbstatic.com/team-logos/112.svg",
        "color": "#0E3386",
    },
    "CHW": {
        "nombre": "Chicago White Sox",
        "logo": "https://www.mlbstatic.com/team-logos/145.svg",
        "color": "#27251F",
    },
    "CIN": {
        "nombre": "Cincinnati Reds",
        "logo": "https://www.mlbstatic.com/team-logos/113.svg",
        "color": "#C6011F",
    },
    "CLE": {
        "nombre": "Cleveland Guardians",
        "logo": "https://www.mlbstatic.com/team-logos/114.svg",
        "color": "#0C2340",
    },
    "COL": {
        "nombre": "Colorado Rockies",
        "logo": "https://www.mlbstatic.com/team-logos/115.svg",
        "color": "#33006F",
    },
    "DET": {
        "nombre": "Detroit Tigers",
        "logo": "https://www.mlbstatic.com/team-logos/116.svg",
        "color": "#0C2340",
    },
    "HOU": {
        "nombre": "Houston Astros",
        "logo": "https://www.mlbstatic.com/team-logos/117.svg",
        "color": "#002D62",
    },
    "KCR": {
        "nombre": "Kansas City Royals",
        "logo": "https://www.mlbstatic.com/team-logos/118.svg",
        "color": "#004687",
    },
    "LAA": {
        "nombre": "Los Angeles Angels",
        "logo": "https://www.mlbstatic.com/team-logos/108.svg",
        "color": "#BA0021",
    },
    "LAD": {
        "nombre": "Los Angeles Dodgers",
        "logo": "https://www.mlbstatic.com/team-logos/119.svg",
        "color": "#005A9C",
    },
    "MIA": {
        "nombre": "Miami Marlins",
        "logo": "https://www.mlbstatic.com/team-logos/146.svg",
        "color": "#00A3E0",
    },
    "MIL": {
        "nombre": "Milwaukee Brewers",
        "logo": "https://www.mlbstatic.com/team-logos/158.svg",
        "color": "#12284B",
    },
    "MIN": {
        "nombre": "Minnesota Twins",
        "logo": "https://www.mlbstatic.com/team-logos/142.svg",
        "color": "#002B5C",
    },
    "NYM": {
        "nombre": "New York Mets",
        "logo": "https://www.mlbstatic.com/team-logos/121.svg",
        "color": "#002D72",
    },
    "NYY": {
        "nombre": "New York Yankees",
        "logo": "https://www.mlbstatic.com/team-logos/147.svg",
        "color": "#003087",
    },
    "ATH": {
        "nombre": "Oakland Athletics",
        "logo": "https://www.mlbstatic.com/team-logos/133.svg",
        "color": "#003831",
    },
    "PHI": {
        "nombre": "Philadelphia Phillies",
        "logo": "https://www.mlbstatic.com/team-logos/143.svg",
        "color": "#E81828",
    },
    "PIT": {
        "nombre": "Pittsburgh Pirates",
        "logo": "https://www.mlbstatic.com/team-logos/134.svg",
        "color": "#27251F",
    },
    "SDP": {
        "nombre": "San Diego Padres",
        "logo": "https://www.mlbstatic.com/team-logos/135.svg",
        "color": "#2F241D",
    },
    "SEA": {
        "nombre": "Seattle Mariners",
        "logo": "https://www.mlbstatic.com/team-logos/136.svg",
        "color": "#0C2C56",
    },
    "SFG": {
        "nombre": "San Francisco Giants",
        "logo": "https://www.mlbstatic.com/team-logos/137.svg",
        "color": "#FD5A1E",
    },
    "STL": {
        "nombre": "St. Louis Cardinals",
        "logo": "https://www.mlbstatic.com/team-logos/138.svg",
        "color": "#C41E3A",
    },
    "TBR": {
        "nombre": "Tampa Bay Rays",
        "logo": "https://www.mlbstatic.com/team-logos/139.svg",
        "color": "#092C5C",
    },
    "TEX": {
        "nombre": "Texas Rangers",
        "logo": "https://www.mlbstatic.com/team-logos/140.svg",
        "color": "#003278",
    },
    "TOR": {
        "nombre": "Toronto Blue Jays",
        "logo": "https://www.mlbstatic.com/team-logos/141.svg",
        "color": "#134A8E",
    },
    "WSN": {
        "nombre": "Washington Nationals",
        "logo": "https://www.mlbstatic.com/team-logos/120.svg",
        "color": "#AB0003",
    },
}

# ============================================================================
# ESTILOS CSS PROFESIONALES
# ============================================================================
if theme == "Oscuro":
    st.markdown(
        """
<style>
    :root {
        --primary-blue: #00c8ff;
        --secondary-blue: #0088cc;
        --success-green: #00e676;
        --warning-yellow: #f59e0b;
        --danger-red: #ef4444;
        --dark-bg: #070e1c;
        --light-bg: #050c1a;
    }

    .stApp {
        background-color: #070e1c !important;
        color: #d8eef8 !important;
    }

    header, [data-testid="stHeader"], .stApp header {
        background-color: transparent !important;
        background: transparent !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: #8b949e !important;
    }
    .stTabs [aria-selected="true"] {
        color: #fff !important;
        border-bottom-color: #00c8ff !important;
    }

    /* Style Streamlit expander boxes (details summary / details>div) */
    details summary {
        background-color: #050c1a !important;
        border: 1px solid #1e3f6a !important;
        border-radius: 8px !important;
        color: #00c8ff !important;
        font-weight: 700 !important;
        padding: 0.7rem 1rem !important;
        cursor: pointer;
    }
    details[open] summary {
        border-radius: 8px 8px 0 0 !important;
        border-bottom: 1px solid #1e3f6a !important;
    }
    details>div {
        background-color: #070e1c !important;
        border: 1px solid #1e3f6a !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        padding: 1rem !important;
        color: #d8eef8 !important;
    }

    /* Style Streamlit standard buttons */
    .stButton>button {
        background-color: #050c1a !important;
        color: #00c8ff !important;
        border: 1px solid #1e3f6a !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        background-color: #0a1628 !important;
        border-color: #00c8ff !important;
        color: #ffffff !important;
        box-shadow: 0 0 10px rgba(0, 200, 255, 0.2) !important;
    }

    /* Style inputs, selectors, and dropdown fields globally in Oscuro mode */
    div[data-baseweb="input"] {
        background-color: #050c1a !important;
        border: 1px solid #1e3f6a !important;
        border-radius: 8px !important;
        color: #d8eef8 !important;
    }
    div[data-baseweb="input"] input {
        color: #d8eef8 !important;
        background-color: transparent !important;
    }

    /* Comprehensive override for selectboxes and dropdowns in Dark Mode to prevent white background */
    div[data-baseweb="select"],
    div[data-baseweb="select"] > div,
    div[role="button"],
    .stSelectbox > div {
        background-color: #050c1a !important;
        border: 1px solid #1e3f6a !important;
        border-radius: 8px !important;
        color: #d8eef8 !important;
    }

    /* Inner text and inputs in selectbox */
    div[data-baseweb="select"] div,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input {
        background-color: transparent !important;
        color: #d8eef8 !important;
    }

    /* Opened dropdown options popup/popover menu */
    div[data-baseweb="popover"],
    ul[role="listbox"],
    li[role="option"] {
        background-color: #050c1a !important;
        color: #d8eef8 !important;
        border: 1px solid #1e3f6a !important;
    }

    /* Hover and active selections in the dropdown options list */
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
        background-color: #0a1628 !important;
        color: #00c8ff !important;
    }
    /* Style dataframes */
    .stDataFrame {
        border: 1px solid #1e3f6a !important;
        border-radius: 8px !important;
        background-color: #050c1a !important;
    }

    /* Streamlit Metric Value/Labels */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #d8eef8 !important;
    }
    [data-testid="stMetricDelta"] {
        color: #00e676 !important;
    }

    /* Labels and legends of standard widgets to fix 'no se ven algunas letras' */
    .stWidgetLabel p, .stWidgetLabel, label, label p, [data-testid="stWidgetLabel"] {
        color: #d8eef8 !important;
    }
    /* Small help texts or descriptions below widgets */
    .stMarkdown p, .stCaption p, .stCaption, [data-testid="stCaptionContainer"] {
        color: #90cce8 !important;
    }
    /* Radio options */
    div[role="radiogroup"] label p {
        color: #d8eef8 !important;
    }

    /* Premium Table styling for Oscuro mode */
    .mlb-results-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1.5rem 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        background: #070e1c !important;
        border: 1px solid #1e3f6a !important;
        box-shadow: 0 4px 20px rgba(0, 200, 255, 0.05) !important;
        border-radius: 12px;
        overflow: hidden;
    }
    .mlb-results-table th {
        background: #050c1a !important;
        color: #90cce8 !important;
        font-weight: 700;
        text-align: left;
        padding: 0.85rem 1.25rem;
        border-bottom: 2px solid #1e3f6a !important;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.5px;
    }
    .mlb-results-table td {
        padding: 0.85rem 1.25rem;
        border-bottom: 1px solid #1e3f6a !important;
        color: #d8eef8 !important;
    }
    .mlb-results-table tr:last-child td {
        border-bottom: none;
    }
    .mlb-results-table tr:hover {
        background: #0a1628 !important;
    }
    .table-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.75rem;
        text-align: center;
    }
    .table-badge-success {
        background: rgba(0, 230, 118, 0.15) !important;
        color: #00e676 !important;
        border: 1px solid rgba(0, 230, 118, 0.3) !important;
    }
    .table-badge-error {
        background: rgba(239, 68, 68, 0.15) !important;
        color: #ef4444 !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
    }

    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #00c8ff 0%, #0088cc 50%, #90cce8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 1rem 0;
        text-shadow: 2px 2px 8px rgba(0,200,255,0.2);
    }

    .subtitle {
        text-align: center;
        color: #90cce8;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }

    .badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-weight: 700;
        font-size: 0.875rem;
        margin: 0.25rem;
    }

    .badge-pro {
        background: linear-gradient(135deg, #0055aa 0%, #00c8ff 100%);
        color: #05090f;
        box-shadow: 0 4px 15px rgba(0, 200, 255, 0.4);
    }

    .badge-live {
        background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    .winner-box {
        background: linear-gradient(135deg, #070e1c 0%, #0a1628 100%);
        padding: 2.5rem;
        border-radius: 1.25rem;
        text-align: center;
        color: #d8eef8;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0, 200, 255, 0.15);
        border: 2px solid #1e3f6a;
    }

    .winner-title {
        font-size: 2rem;
        font-weight: 900;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #00c8ff !important;
    }

    .winner-team {
        font-size: 3.5rem;
        font-weight: 900;
        margin: 1rem 0;
        color: #ffffff;
    }

    .stats-card {
        background: #070e1c;
        padding: 1.75rem;
        border-radius: 1rem;
        box-shadow: 0 4px 20px rgba(0, 200, 255, 0.05);
        border-left: 5px solid #00c8ff;
        border-top: 1px solid #1e3f6a;
        border-right: 1px solid #1e3f6a;
        border-bottom: 1px solid #1e3f6a;
        margin: 1rem 0;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .stats-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 200, 255, 0.12);
        border-color: #00c8ff;
    }

    .stats-card h4 {
        color: #90cce8 !important;
    }

    .stats-card p, .stats-card span {
        color: #d8eef8;
    }

    .stats-card-mini {
        background: #070e1c;
        padding: 0.75rem 1rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 15px rgba(0, 200, 255, 0.04);
        border-left: 4px solid #00c8ff;
        border-top: 1px solid #1e3f6a;
        border-right: 1px solid #1e3f6a;
        border-bottom: 1px solid #1e3f6a;
        margin: 0.5rem 0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stats-card-mini:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 200, 255, 0.08);
        border-color: #00c8ff;
    }
    .stats-card-mini h4 {
        margin: 0 0 0.15rem 0;
        font-size: 0.8rem;
        font-weight: 700;
        color: #90cce8 !important;
    }

    .stats-card-val {
        font-size: 1.15rem;
        font-weight: 900;
        margin-bottom: 0.15rem;
    }
    .value-blue {
        color: #00c8ff !important;
    }
    .value-purple {
        color: #b388ff !important;
    }
    .stats-card-desc {
        color: #8b949e !important;
        margin: 0;
        font-size: 0.725rem;
    }

    .pitcher-card {
        background: linear-gradient(135deg, #070e1c 0%, #0a1628 100%);
        padding: 2rem;
        border-radius: 1rem;
        border-left: 8px solid #00c8ff;
        border-top: 1px solid #1e3f6a;
        border-right: 1px solid #1e3f6a;
        border-bottom: 1px solid #1e3f6a;
        margin: 1.5rem 0;
        box-shadow: 0 6px 25px rgba(0,0,0,0.2);
    }

    .pitcher-name {
        font-size: 1.75rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 1rem;
    }

    .super-feature {
        background: #070e1c;
        padding: 1.5rem;
        border-radius: 1rem;
        color: #d8eef8;
        margin: 1rem 0;
        min-height: 220px;
        box-shadow: 0 4px 15px rgba(0, 200, 255, 0.04);
        border-left: 5px solid #00c8ff;
        border-top: 1px solid #1e3f6a;
        border-right: 1px solid #1e3f6a;
        border-bottom: 1px solid #1e3f6a;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .super-feature:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 200, 255, 0.08);
        border-color: #00c8ff;
    }

    .super-feature h4 {
        font-size: 1.15rem;
        font-weight: 800;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #00c8ff !important;
    }

    .super-feature-advantage {
        font-size: 0.825rem;
        font-weight: 700;
        background: rgba(0, 200, 255, 0.1);
        color: #00c8ff;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        display: inline-block;
        margin: 0.5rem 0;
    }

    .super-feature p, .super-feature strong {
        color: #cbd5e1 !important;
    }

    .alert-card-info {
        background: #070e1c;
        border-left: 5px solid #00c8ff;
        border-top: 1px solid #1e3f6a;
        border-right: 1px solid #1e3f6a;
        border-bottom: 1px solid #1e3f6a;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin: 0.75rem 0;
        font-family: 'Inter', sans-serif;
    }
    .alert-card-info-title {
        color: #00c8ff;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .alert-card-info-message {
        color: #90cce8;
        font-size: 0.875rem;
    }

    .alert-card-warning {
        background: #070e1c;
        border-left: 5px solid #f59e0b;
        border-top: 1px solid #1e3f6a;
        border-right: 1px solid #1e3f6a;
        border-bottom: 1px solid #1e3f6a;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin: 0.75rem 0;
        font-family: 'Inter', sans-serif;
    }
    .alert-card-warning-title {
        color: #f59e0b;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .alert-card-warning-message {
        color: #fed7aa;
        font-size: 0.875rem;
    }

    .alert-card-success {
        background: #070e1c;
        border-left: 5px solid #10b981;
        border-top: 1px solid #1e3f6a;
        border-right: 1px solid #1e3f6a;
        border-bottom: 1px solid #1e3f6a;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin: 0.75rem 0;
        font-family: 'Inter', sans-serif;
    }
    .alert-card-success-title {
        color: #10b981;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .alert-card-success-message {
        color: #bbf7d0;
        font-size: 0.875rem;
    }

    .game-card {
        background: #070e1c;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 200, 255, 0.05);
        border: 1px solid #1e3f6a;
        border-top: 4px solid #00c8ff;
    }
    .game-card:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(0, 200, 255, 0.12);
        border-color: #00c8ff;
    }

    .mlb-scoreboard-card {
        background: #070e1c;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        padding: 0;
        margin-bottom: 20px;
        border: 1px solid #1e3f6a;
        overflow: hidden;
        transition: transform 0.2s;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .mlb-scoreboard-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 18px rgba(0,200,255,0.1);
        border-color: #00c8ff;
    }
    .mlb-card-header {
        padding: 8px 15px;
        background: #050c1a;
        border-bottom: 1px solid #1e3f6a;
        font-size: 0.7rem;
        font-weight: 700;
        color: #90cce8;
        display: flex;
        justify-content: space-between;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .mlb-card-body {
        padding: 15px;
    }
    .mlb-team-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
    }
    .mlb-team-row:last-child {
        margin-bottom: 0;
    }
    .mlb-team-info {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .mlb-team-name {
        font-size: 1rem;
        font-weight: 600;
        color: #ffffff;
    }
    .mlb-team-record {
        font-size: 0.75rem;
        color: #8b949e;
        font-weight: 400;
    }
    .mlb-score {
        font-size: 1.4rem;
        font-weight: 800;
        color: #ffffff;
    }
    .mlb-card-footer {
        padding: 10px 15px;
        background: #070e1c;
        border-top: 1px solid #1e3f6a;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .mlb-prediction-badge {
        font-size: 0.75rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 4px 10px;
        border-radius: 4px;
        background: #050c1a;
        border: 1px solid #1e3f6a;
        color: #00c8ff;
    }

    [data-testid="stSidebar"] {
        background-color: #050c1a !important;
        border-right: 1px solid #1e3f6a !important;
    }

    [data-testid="stSidebar"] h2 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.5px !important;
    }

    [data-testid="stSidebar"] .sidebar-section-title {
        font-size: 0.92rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #90cce8;
        margin: 0.5rem 0 0.75rem 0;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 0.6rem;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label {
        background: linear-gradient(180deg, #070e1c 0%, #0a1628 100%) !important;
        border: 1px solid #1e3f6a !important;
        border-radius: 14px;
        padding: 0.55rem 0.75rem;
        margin: 0.35rem 0;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.3);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        transform: translateY(-1px);
        border-color: #00c8ff !important;
        box-shadow: 0 10px 24px rgba(0, 200, 255, 0.1);
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, #0a1628 0%, #0d1e38 100%) !important;
        border-color: #00c8ff !important;
        box-shadow: 0 12px 26px rgba(0, 200, 255, 0.15);
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-size: 1rem;
        font-weight: 700;
        color: #90cce8 !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
        color: #00c8ff !important;
    }

    .confidence-muy-alta { color: #00e676; font-weight: 900; font-size: 2.5rem; }
    .confidence-alta { color: #00c8ff; font-weight: 900; font-size: 2.5rem; }
    .confidence-moderada { color: #f59e0b; font-weight: 900; font-size: 2.5rem; }
    .confidence-baja { color: #ef4444; font-weight: 900; font-size: 2.5rem; }

    .mlb-matchup-container {
        background: #070e1c;
        border: 1px solid #1e3f6a;
        border-radius: 0.75rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin-bottom: 2rem;
        overflow: hidden;
        font-family: 'Inter', sans-serif;
    }

    .mlb-matchup-top-header {
        background: #050c1a;
        border-bottom: 1px solid #1e3f6a;
        padding: 0.6rem 1.25rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .mlb-matchup-pred-badge {
        background: rgba(0, 200, 255, 0.08);
        border: 1px solid rgba(0, 200, 255, 0.15);
        color: #00c8ff;
        padding: 0.25rem 0.75rem;
        border-radius: 2rem;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }

    .mlb-matchup-conf-pill {
        background: rgba(0, 230, 118, 0.08);
        border: 1px solid rgba(0, 230, 118, 0.15);
        color: #00e676;
        padding: 0.25rem 0.75rem;
        border-radius: 2rem;
        font-size: 0.725rem;
        font-weight: 700;
    }

    .mlb-matchup-grid {
        display: grid;
        grid-template-columns: 2.2fr 1fr 2.2fr;
        align-items: center;
        padding: 1.75rem 1.25rem;
        background: #070e1c;
    }

    .mlb-matchup-team {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        position: relative;
        overflow: hidden;
        width: 100%;
        padding: 0.5rem 0;
    }

    .mlb-matchup-logo-container,
    .mlb-matchup-status-label,
    .mlb-matchup-team-name,
    .mlb-matchup-pitcher-name,
    .mlb-matchup-prob-value,
    .mlb-matchup-prob-label {
        position: relative;
        z-index: 2;
    }

    .mlb-matchup-logo-container {
        width: 80px;
        height: 80px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.75rem;
        filter: drop-shadow(0 4px 6px rgba(0,0,0,0.3));
    }

    @keyframes logoFloat {
        0% { transform: translateY(-50%) scale(1); opacity: .07; }
        50% { transform: translateY(-54%) scale(1.04); opacity: .12; }
        100% { transform: translateY(-50%) scale(1); opacity: .07; }
    }
    @keyframes logoFloatL {
        0% { transform: translateY(-50%) scale(1); opacity: .07; }
        50% { transform: translateY(-54%) scale(1.04); opacity: .12; }
        100% { transform: translateY(-50%) scale(1); opacity: .07; }
    }

    .bg-logo {
        position: absolute;
        width: 180px;
        height: 180px;
        top: 50%;
        right: -25px;
        object-fit: contain;
        pointer-events: none;
        filter: saturate(0) brightness(1.5);
        animation: logoFloat 8s ease-in-out infinite;
        z-index: 1;
    }
    .mlb-matchup-grid .mlb-matchup-team:nth-child(3) .bg-logo {
        right: auto;
        left: -25px;
        animation: logoFloatL 8s ease-in-out infinite;
    }

    .mlb-matchup-status-label {
        font-size: 0.675rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
        padding: 0.15rem 0.6rem;
        border-radius: 0.25rem;
    }

    .status-favorite {
        color: #00c8ff;
        background: rgba(0, 200, 255, 0.1);
    }

    .status-underdog {
        color: #8b949e;
        background: rgba(139, 148, 158, 0.08);
    }

    .mlb-matchup-team-name {
        color: #ffffff;
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0.25rem 0;
    }

    .mlb-matchup-pitcher-name {
        color: #90cce8;
        font-size: 0.8rem;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }

    .mlb-matchup-prob-value {
        font-size: 2.5rem;
        font-weight: 900;
        line-height: 1;
        margin-bottom: 0.15rem;
    }

    .prob-fav {
        color: #00c8ff;
        opacity: 1.0;
    }

    .prob-und {
        color: #8b949e;
        opacity: 0.45;
    }

    .mlb-p-bar {
        height: 6px;
        background: #050c1a;
        border-radius: 3px;
        margin-top: 8px;
        overflow: hidden;
        width: 80%;
        position: relative;
        z-index: 2;
    }

    .mlb-p-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #0055aa, #00c8ff);
        border-radius: 3px;
    }

    .mlb-matchup-prob-label {
        color: #8b949e;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .mlb-matchup-center {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }

    .mlb-matchup-vs {
        font-size: 1.15rem;
        font-weight: 900;
        color: #1e3f6a;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }

    .mlb-matchup-edge-box {
        background: rgba(0, 230, 118, 0.05);
        border: 1px solid rgba(0, 230, 118, 0.2);
        border-radius: 0.5rem;
        padding: 0.4rem 0.75rem;
        display: inline-flex;
        flex-direction: column;
        align-items: center;
    }

    .mlb-matchup-edge-value {
        color: #00e676;
        font-size: 1.15rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .mlb-matchup-edge-label {
        color: #00c8ff;
        font-size: 0.6rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-top: 0.15rem;
    }

    .mlb-matchup-stadium {
        color: #90cce8;
        font-size: 0.7rem;
        margin-top: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .mlb-momentum-bar {
        background: #050c1a;
        border-top: 1px solid #1e3f6a;
        display: grid;
        grid-template-columns: 1fr 1fr;
        padding: 0.75rem 1.5rem;
        font-size: 0.8rem;
    }

    .mlb-momentum-column {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .column-away {
        justify-content: flex-start;
        border-right: 1px solid #1e3f6a;
        padding-right: 1rem;
    }

    .column-home {
        justify-content: flex-end;
        padding-left: 1rem;
    }

    .mlb-momentum-team-badge {
        font-weight: 800;
        font-size: 0.75rem;
        color: #00c8ff;
        background: rgba(0, 200, 255, 0.1);
        padding: 0.15rem 0.5rem;
        border-radius: 0.25rem;
        border: 1px solid rgba(0, 200, 255, 0.2);
    }

    .mlb-momentum-stat {
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    .mlb-momentum-stat-label {
        color: #8b949e;
        font-size: 0.575rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.15rem;
    }

    .mlb-momentum-stat-value {
        color: #ffffff;
        font-weight: 700;
    }

    .racha-pill {
        padding: 0.1rem 0.5rem;
        border-radius: 2rem;
        font-size: 0.7rem;
        font-weight: 800;
        display: inline-flex;
        align-items: center;
        gap: 0.15rem;
    }

    .racha-win {
        background: rgba(0, 230, 118, 0.1);
        color: #00e676;
        border: 1px solid rgba(0, 230, 118, 0.2);
    }

    .racha-loss {
        background: rgba(239, 68, 68, 0.1);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }

    .rendimiento-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #070e1c !important;
        border: 1px solid #1e3f6a !important;
        border-left: 6px solid #00c8ff !important;
        border-radius: 14px;
        padding: 1.25rem 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 25px rgba(0, 200, 255, 0.08) !important;
        flex-wrap: wrap;
        gap: 1.5rem;
    }

    .rendimiento-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #ffffff !important;
        letter-spacing: 0.5px;
    }

    .rendimiento-subtitle {
        font-size: 0.75rem;
        color: #90cce8 !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 2px;
    }

    .rendimiento-stats {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .rendimiento-stat-box {
        background: #050c1a !important;
        border: 1px solid #1e3f6a !important;
        border-radius: 10px;
        padding: 0.75rem 1.25rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-width: 110px;
        box-shadow: none !important;
    }

    .rendimiento-stat-val {
        font-size: 1.5rem;
        font-weight: 900;
        color: #00c8ff !important;
        line-height: 1.1;
    }

    .rendimiento-stat-lbl {
        font-size: 0.6rem;
        font-weight: 700;
        color: #80bcd8 !important;
        letter-spacing: 1px;
        margin-top: 4px;
        text-transform: uppercase;
        text-align: center;
    }
</style>
""",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
<style>
    header, [data-testid="stHeader"], .stApp header {
        background-color: transparent !important;
        background: transparent !important;
    }

    :root {
        --primary-blue: #1e40af;
        --secondary-blue: #3b82f6;
        --success-green: #10b981;
        --warning-yellow: #f59e0b;
        --danger-red: #ef4444;
        --dark-bg: #0f172a;
        --light-bg: #f8fafc;
    }

    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }

    .badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-weight: 700;
        font-size: 0.875rem;
        margin: 0.25rem;
    }

    .badge-pro {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(118, 75, 162, 0.4);
    }

    .badge-live {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    .winner-box {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2.5rem;
        border-radius: 1.25rem;
        text-align: center;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(30, 58, 138, 0.3);
        border: 2px solid rgba(255,255,255,0.1);
    }

    .winner-title {
        font-size: 2rem;
        font-weight: 900;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .winner-team {
        font-size: 3.5rem;
        font-weight: 900;
        margin: 1rem 0;
    }

    .stats-card {
        background: white;
        padding: 1.75rem;
        border-radius: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 5px solid var(--primary-blue);
        margin: 1rem 0;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .stats-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }

    .stats-card-mini {
        background: white;
        padding: 0.75rem 1rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        border-left: 4px solid var(--primary-blue);
        margin: 0.5rem 0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stats-card-mini:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    .stats-card-mini h4 {
        margin: 0 0 0.15rem 0;
        font-size: 0.8rem;
        font-weight: 700;
        color: #64748b;
    }

    .stats-card-val {
        font-size: 1.15rem;
        font-weight: 900;
        margin-bottom: 0.15rem;
    }
    .value-blue {
        color: #1e40af;
    }
    .value-purple {
        color: #7c3aed;
    }
    .stats-card-desc {
        color: #64748b;
        margin: 0;
        font-size: 0.725rem;
    }

    .pitcher-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem;
        border-radius: 1rem;
        border-left: 8px solid var(--secondary-blue);
        margin: 1.5rem 0;
        box-shadow: 0 6px 25px rgba(0,0,0,0.1);
    }

    .pitcher-name {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 1rem;
    }

    .super-feature {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 1rem;
        color: #0f172a;
        margin: 1rem 0;
        min-height: 220px;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05);
        border-left: 5px solid var(--primary-blue);
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .super-feature:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.08);
    }

    .super-feature h4 {
        font-size: 1.15rem;
        font-weight: 800;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #0f172a;
    }

    .super-feature-advantage {
        font-size: 0.825rem;
        font-weight: 700;
        background: rgba(37, 99, 235, 0.08);
        color: #1d4ed8;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        display: inline-block;
        margin: 0.5rem 0;
    }

    .alert-card-info {
        background: #ffffff;
        border-left: 5px solid #3b82f6;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05);
        margin: 0.75rem 0;
        font-family: 'Inter', sans-serif;
    }
    .alert-card-info-title {
        color: #1e3a8a;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .alert-card-info-message {
        color: #1e40af;
        font-size: 0.875rem;
    }

    .alert-card-warning {
        background: #ffffff;
        border-left: 5px solid #f59e0b;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05);
        margin: 0.75rem 0;
        font-family: 'Inter', sans-serif;
    }
    .alert-card-warning-title {
        color: #854d0e;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .alert-card-warning-message {
        color: #a16207;
        font-size: 0.875rem;
    }

    .alert-card-success {
        background: #ffffff;
        border-left: 5px solid #10b981;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05);
        margin: 0.75rem 0;
        font-family: 'Inter', sans-serif;
    }
    .alert-card-success-title {
        color: #065f46;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .alert-card-success-message {
        color: #047857;
        font-size: 0.875rem;
    }

    .game-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-top: 4px solid var(--primary-blue);
    }
    .game-card:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }

    .mlb-scoreboard-card {
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        padding: 0;
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
        overflow: hidden;
        transition: transform 0.2s;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .mlb-scoreboard-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.15);
    }
    .mlb-card-header {
        padding: 8px 15px;
        background: #f8f9fa;
        border-bottom: 1px solid #eeeeee;
        font-size: 0.7rem;
        font-weight: 700;
        color: #616161;
        display: flex;
        justify-content: space-between;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .mlb-card-body {
        padding: 15px;
    }
    .mlb-team-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
    }
    .mlb-team-row:last-child {
        margin-bottom: 0;
    }
    .mlb-team-info {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .mlb-team-name {
        font-size: 1rem;
        font-weight: 600;
        color: #212121;
    }
    .mlb-team-record {
        font-size: 0.75rem;
        color: #757575;
        font-weight: 400;
    }
    .mlb-score {
        font-size: 1.4rem;
        font-weight: 800;
        color: #212121;
    }
    .mlb-card-footer {
        padding: 10px 15px;
        background: #ffffff;
        border-top: 1px solid #f0f0f0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .mlb-prediction-badge {
        font-size: 0.75rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 4px 10px;
        border-radius: 4px;
        background: #f1f5f9;
    }

    [data-testid="stSidebar"] h2 {
        color: #0f172a !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.5px !important;
    }

    [data-testid="stSidebar"] .sidebar-section-title {
        font-size: 0.92rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748b;
        margin: 0.5rem 0 0.75rem 0;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 0.6rem;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 14px;
        padding: 0.55rem 0.75rem;
        margin: 0.35rem 0;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        transform: translateY(-1px);
        border-color: rgba(59, 130, 246, 0.45);
        box-shadow: 0 10px 24px rgba(30, 64, 175, 0.12);
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-color: rgba(37, 99, 235, 0.7);
        box-shadow: 0 12px 26px rgba(37, 99, 235, 0.18);
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-size: 1rem;
        font-weight: 700;
        color: #0f172a;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
        color: #1d4ed8;
    }

    .confidence-muy-alta { color: #10b981; font-weight: 900; font-size: 2.5rem; }
    .confidence-alta { color: #3b82f6; font-weight: 900; font-size: 2.5rem; }
    .confidence-moderada { color: #f59e0b; font-weight: 900; font-size: 2.5rem; }
    .confidence-baja { color: #ef4444; font-weight: 900; font-size: 2.5rem; }

    .mlb-matchup-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
        overflow: hidden;
        font-family: 'Inter', sans-serif;
    }

    .mlb-matchup-top-header {
        background: #f8fafc;
        border-bottom: 1px solid #e2e8f0;
        padding: 0.6rem 1.25rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .mlb-matchup-pred-badge {
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.15);
        color: #1e40af;
        padding: 0.25rem 0.75rem;
        border-radius: 2rem;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }

    .mlb-matchup-conf-pill {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.15);
        color: #10b981;
        padding: 0.25rem 0.75rem;
        border-radius: 2rem;
        font-size: 0.725rem;
        font-weight: 700;
    }

    .mlb-matchup-grid {
        display: grid;
        grid-template-columns: 2.2fr 1fr 2.2fr;
        align-items: center;
        padding: 1.75rem 1.25rem;
        background: #ffffff;
    }

    .mlb-matchup-team {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        position: relative;
        overflow: hidden;
        width: 100%;
        padding: 0.5rem 0;
    }

    .mlb-matchup-logo-container,
    .mlb-matchup-status-label,
    .mlb-matchup-team-name,
    .mlb-matchup-pitcher-name,
    .mlb-matchup-prob-value,
    .mlb-matchup-prob-label {
        position: relative;
        z-index: 2;
    }

    .mlb-matchup-logo-container {
        width: 80px;
        height: 80px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.75rem;
        filter: drop-shadow(0 4px 6px rgba(0,0,0,0.08));
    }

    @keyframes logoFloat {
        0% { transform: translateY(-50%) scale(1); opacity: .07; }
        50% { transform: translateY(-54%) scale(1.04); opacity: .12; }
        100% { transform: translateY(-50%) scale(1); opacity: .07; }
    }
    @keyframes logoFloatL {
        0% { transform: translateY(-50%) scale(1); opacity: .07; }
        50% { transform: translateY(-54%) scale(1.04); opacity: .12; }
        100% { transform: translateY(-50%) scale(1); opacity: .07; }
    }

    .bg-logo {
        position: absolute;
        width: 180px;
        height: 180px;
        top: 50%;
        right: -25px;
        object-fit: contain;
        pointer-events: none;
        filter: saturate(0) brightness(0.65);
        animation: logoFloat 8s ease-in-out infinite;
        z-index: 1;
    }
    .mlb-matchup-grid .mlb-matchup-team:nth-child(3) .bg-logo {
        right: auto;
        left: -25px;
        animation: logoFloatL 8s ease-in-out infinite;
    }

    .mlb-matchup-status-label {
        font-size: 0.675rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
        padding: 0.15rem 0.6rem;
        border-radius: 0.25rem;
    }

    .status-favorite {
        color: #10b981;
        background: rgba(16, 185, 129, 0.1);
    }

    .status-underdog {
        color: #64748b;
        background: rgba(100, 116, 139, 0.08);
    }

    .mlb-matchup-team-name {
        color: #0f172a;
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0.25rem 0;
    }

    .mlb-matchup-pitcher-name {
        color: #64748b;
        font-size: 0.8rem;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }

    .mlb-matchup-prob-value {
        font-size: 2.5rem;
        font-weight: 900;
        line-height: 1;
        margin-bottom: 0.15rem;
    }

    .prob-fav {
        color: #10b981;
        opacity: 1.0;
    }

    .prob-und {
        color: #64748b;
        opacity: 0.45;
    }

    .mlb-p-bar {
        height: 6px;
        background: #f1f5f9;
        border-radius: 3px;
        margin-top: 8px;
        overflow: hidden;
        width: 80%;
        position: relative;
        z-index: 2;
    }

    .mlb-p-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #10b981);
        border-radius: 3px;
    }

    .mlb-matchup-prob-label {
        color: #94a3b8;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .mlb-matchup-center {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }

    .mlb-matchup-vs {
        font-size: 1.15rem;
        font-weight: 900;
        color: #cbd5e1;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }

    .mlb-matchup-edge-box {
        background: rgba(16, 185, 129, 0.05);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 0.5rem;
        padding: 0.4rem 0.75rem;
        display: inline-flex;
        flex-direction: column;
        align-items: center;
    }

    .mlb-matchup-edge-value {
        color: #10b981;
        font-size: 1.15rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .mlb-matchup-edge-label {
        color: #059669;
        font-size: 0.6rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-top: 0.15rem;
    }

    .mlb-matchup-stadium {
        color: #64748b;
        font-size: 0.7rem;
        margin-top: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .mlb-momentum-bar {
        background: #f8fafc;
        border-top: 1px solid #e2e8f0;
        display: grid;
        grid-template-columns: 1fr 1fr;
        padding: 0.75rem 1.5rem;
        font-size: 0.8rem;
    }

    .mlb-momentum-column {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .column-away {
        justify-content: flex-start;
        border-right: 1px solid #e2e8f0;
        padding-right: 1rem;
    }

    .column-home {
        justify-content: flex-end;
        padding-left: 1rem;
    }

    .mlb-momentum-team-badge {
        font-weight: 800;
        font-size: 0.75rem;
        color: #1e293b;
        background: rgba(15, 23, 42, 0.05);
        padding: 0.15rem 0.5rem;
        border-radius: 0.25rem;
    }

    .mlb-momentum-stat {
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    .mlb-momentum-stat-label {
        color: #64748b;
        font-size: 0.575rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.15rem;
    }

    .mlb-momentum-stat-value {
        color: #0f172a;
        font-weight: 700;
    }

    .racha-pill {
        padding: 0.1rem 0.5rem;
        border-radius: 2rem;
        font-size: 0.7rem;
        font-weight: 800;
        display: inline-flex;
        align-items: center;
        gap: 0.15rem;
    }

    .racha-win {
        background: rgba(16, 185, 129, 0.1);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }

    .racha-loss {
        background: rgba(239, 68, 68, 0.1);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }

    .rendimiento-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 6px solid var(--secondary-blue);
        border-radius: 14px;
        padding: 1.25rem 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        flex-wrap: wrap;
        gap: 1.5rem;
    }

    .rendimiento-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: 0.5px;
    }

    .rendimiento-subtitle {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 2px;
    }

    .rendimiento-stats {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .rendimiento-stat-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem 1.25rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-width: 110px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    }

    .rendimiento-stat-val {
        font-size: 1.5rem;
        font-weight: 900;
        color: #0f172a;
        line-height: 1.1;
    }

    .rendimiento-stat-lbl {
        font-size: 0.6rem;
        font-weight: 700;
        color: #64748b;
        letter-spacing: 1px;
        margin-top: 4px;
        text-transform: uppercase;
        text-align: center;
    }

    /* Premium Table styling for Claro mode */
    .mlb-results-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1.5rem 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        background: #ffffff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }
    .mlb-results-table th {
        background: #f8fafc;
        color: #0f172a;
        font-weight: 700;
        text-align: left;
        padding: 0.85rem 1.25rem;
        border-bottom: 2px solid #e2e8f0;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.5px;
    }
    .mlb-results-table td {
        padding: 0.85rem 1.25rem;
        border-bottom: 1px solid #f1f5f9;
        color: #334155;
    }
    .mlb-results-table tr:last-child td {
        border-bottom: none;
    }
    .mlb-results-table tr:hover {
        background: #f8fafc;
    }
    .table-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.75rem;
        text-align: center;
    }
    .table-badge-success {
        background: #dcfce7;
        color: #166534;
    }
    .table-badge-error {
        background: #fee2e2;
        color: #991b1b;
    }
</style>
""",
        unsafe_allow_html=True,
    )

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================


def render_custom_alert(tipo, titulo, mensaje=""):
    """
    Renderiza una alerta premium estilo tarjeta con borde izquierdo de color,
    alineado con la estética de las tarjetas de estado de la imagen 3.
    """
    if tipo == "info":
        card_class = "alert-card-info"
        title_class = "alert-card-info-title"
        msg_class = "alert-card-info-message"
        icon = "ℹ️"
    elif tipo == "warning":
        card_class = "alert-card-warning"
        title_class = "alert-card-warning-title"
        msg_class = "alert-card-warning-message"
        icon = "⚠️"
    elif tipo == "success":
        card_class = "alert-card-success"
        title_class = "alert-card-success-title"
        msg_class = "alert-card-success-message"
        icon = "✅"
    else:
        card_class = "alert-card-info"
        title_class = "alert-card-info-title"
        msg_class = "alert-card-info-message"
        icon = "🔔"

    html = f"""<div class="{card_class}">
<div class="{title_class}">
<span>{icon}</span> <b>{titulo}</b>
</div>
{f'<div class="{msg_class}">{mensaje}</div>' if mensaje else ''}
</div>"""
    return html


# ============================================================================
# SISTEMA DE CLASIFICACIÓN ELO Y POWER RANKINGS
# ============================================================================
DIVISION_MAP = {
    "BAL": "ALE", "BOS": "ALE", "NYY": "ALE", "TBR": "ALE", "TOR": "ALE",
    "CHW": "ALC", "CLE": "ALC", "DET": "ALC", "KCR": "ALC", "MIN": "ALC",
    "HOU": "ALO", "LAA": "ALO", "ATH": "ALO", "SEA": "ALO", "TEX": "ALO",
    "ATL": "NLE", "MIA": "NLE", "NYM": "NLE", "PHI": "NLE", "WSN": "NLE",
    "CHC": "NLC", "CIN": "NLC", "MIL": "NLC", "PIT": "NLC", "STL": "NLC",
    "ARI": "NLO", "COL": "NLO", "LAD": "NLO", "SDP": "NLO", "SFG": "NLO"
}

@st.cache_data(ttl=3600)
def get_elo_data_in_memory():
    elo_dict = {code: 1500.0 for code in EQUIPOS_MLB.keys()}
    wins_dict = {code: 0 for code in EQUIPOS_MLB.keys()}
    losses_dict = {code: 0 for code in EQUIPOS_MLB.keys()}

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT home_team, away_team, ganador
                FROM historico_real
                ORDER BY fecha ASC, game_id ASC
            """)
            games = cursor.fetchall()

            K = 20.0
            HOME_ADVANTAGE = 24.0

            for home, away, ganador in games:
                home_code = get_team_code(home)
                away_code = get_team_code(away)

                if not home_code or not away_code:
                    continue

                if home_code not in elo_dict:
                    elo_dict[home_code] = 1500.0
                    wins_dict[home_code] = 0
                    losses_dict[home_code] = 0
                if away_code not in elo_dict:
                    elo_dict[away_code] = 1500.0
                    wins_dict[away_code] = 0
                    losses_dict[away_code] = 0

                r_home = elo_dict[home_code]
                r_away = elo_dict[away_code]

                e_home = 1.0 / (10.0 ** (-(r_home + HOME_ADVANTAGE - r_away) / 400.0) + 1.0)
                s_home = 1.0 if ganador == 1 else 0.0

                elo_dict[home_code] = r_home + K * (s_home - e_home)
                elo_dict[away_code] = r_away + K * ((1.0 - s_home) - (1.0 - e_home))

                if ganador == 1:
                    wins_dict[home_code] += 1
                    losses_dict[away_code] += 1
                else:
                    wins_dict[away_code] += 1
                    losses_dict[home_code] += 1
    except Exception as e:
        print(f"Error calculating ELO in memory: {e}")

    rankings = []
    for code in elo_dict.keys():
        rankings.append((code, elo_dict[code], wins_dict.get(code, 0), losses_dict.get(code, 0)))
    rankings.sort(key=lambda x: x[1], reverse=True)

    return elo_dict, rankings


def get_elo_power_rankings():
    try:
        elo_dict, rankings = get_elo_data_in_memory()
        return rankings
    except Exception as ex:
        st.error(f"Error cargando Power Rankings ELO: {ex}")
        return []

def render_power_rankings_table_html(elo_list):
    """Genera HTML para tabla premium de Power Rankings ELO"""
    theme_act = "Oscuro" if "theme_selector" in st.session_state and "🌙 Oscuro" in st.session_state.theme_selector else "Claro"

    div_names = {
        "ALE": "AL Este", "ALC": "AL Central", "ALO": "AL Oeste",
        "NLE": "NL Este", "NLC": "NL Central", "NLO": "NL Oeste"
    }

    if theme_act == "Oscuro":
        styles = """<style>
.mlb-elo-table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 1rem 0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    background: #070e1c !important;
    border: 1px solid #1e3f6a !important;
    box-shadow: 0 4px 20px rgba(0, 200, 255, 0.05) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
.mlb-elo-table th {
    background: #050c1a !important;
    color: #00c8ff !important;
    font-weight: 700 !important;
    text-align: left !important;
    padding: 0.85rem 1.1rem !important;
    border-bottom: 2px solid #1e3f6a !important;
    text-transform: uppercase !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.5px !important;
}
.mlb-elo-table td {
    padding: 0.85rem 1.1rem !important;
    border-bottom: 1px solid #1e3f6a !important;
    color: #d8eef8 !important;
    vertical-align: middle !important;
}
.mlb-elo-table tr:last-child td {
    border-bottom: none !important;
}
.mlb-elo-table tr:hover {
    background: #0a1628 !important;
}
.elo-rank {
    font-weight: bold !important;
    color: #90cce8 !important;
    font-size: 1rem !important;
    text-align: center !important;
    width: 50px !important;
}
.elo-team-col {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}
.elo-logo {
    width: 28px !important;
    height: 28px !important;
    object-fit: contain !important;
}
.elo-team-name {
    font-weight: 700 !important;
    color: #ffffff !important;
}
.elo-badge-tier {
    display: inline-block !important;
    padding: 0.2rem 0.5rem !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    font-size: 0.7rem !important;
    text-align: center !important;
}
.elo-tier-elite {
    background: rgba(0, 230, 118, 0.15) !important;
    color: #00e676 !important;
    border: 1px solid rgba(0, 230, 118, 0.3) !important;
}
.elo-tier-strong {
    background: rgba(0, 200, 255, 0.15) !important;
    color: #00c8ff !important;
    border: 1px solid rgba(0, 200, 255, 0.3) !important;
}
.elo-tier-competitive {
    background: rgba(251, 191, 36, 0.15) !important;
    color: #fbbf24 !important;
    border: 1px solid rgba(251, 191, 36, 0.3) !important;
}
.elo-tier-developing {
    background: rgba(244, 63, 94, 0.15) !important;
    color: #f43f5e !important;
    border: 1px solid rgba(244, 63, 94, 0.3) !important;
}
.elo-tier-rebuilding {
    background: rgba(148, 163, 184, 0.15) !important;
    color: #94a3b8 !important;
    border: 1px solid rgba(148, 163, 184, 0.3) !important;
}
.elo-bar-bg {
    background: #050c1a !important;
    border-radius: 4px !important;
    height: 8px !important;
    width: 100px !important;
    overflow: hidden !important;
    display: inline-block !important;
    vertical-align: middle !important;
}
.elo-bar-fill {
    height: 100% !important;
    border-radius: 4px !important;
}
</style>"""
    else:
        styles = """<style>
.mlb-elo-table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 1rem 0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
.mlb-elo-table th {
    background: #f8fafc !important;
    color: #1e3a8a !important;
    font-weight: 700 !important;
    text-align: left !important;
    padding: 0.85rem 1.1rem !important;
    border-bottom: 2px solid #e2e8f0 !important;
    text-transform: uppercase !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.5px !important;
}
.mlb-elo-table td {
    padding: 0.85rem 1.1rem !important;
    border-bottom: 1px solid #e2e8f0 !important;
    color: #334155 !important;
    vertical-align: middle !important;
}
.mlb-elo-table tr:last-child td {
    border-bottom: none !important;
}
.mlb-elo-table tr:hover {
    background: #f1f5f9 !important;
}
.elo-rank {
    font-weight: bold !important;
    color: #475569 !important;
    font-size: 1rem !important;
    text-align: center !important;
    width: 50px !important;
}
.elo-team-col {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}
.elo-logo {
    width: 28px !important;
    height: 28px !important;
    object-fit: contain !important;
}
.elo-team-name {
    font-weight: 700 !important;
    color: #0f172a !important;
}
.elo-badge-tier {
    display: inline-block !important;
    padding: 0.2rem 0.5rem !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    font-size: 0.7rem !important;
    text-align: center !important;
}
.elo-tier-elite {
    background: rgba(16, 185, 129, 0.15) !important;
    color: #059669 !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
}
.elo-tier-strong {
    background: rgba(59, 130, 246, 0.15) !important;
    color: #2563eb !important;
    border: 1px solid rgba(59, 130, 246, 0.3) !important;
}
.elo-tier-competitive {
    background: rgba(245, 158, 11, 0.15) !important;
    color: #d97706 !important;
    border: 1px solid rgba(245, 158, 11, 0.3) !important;
}
.elo-tier-developing {
    background: rgba(244, 63, 94, 0.15) !important;
    color: #e11d48 !important;
    border: 1px solid rgba(244, 63, 94, 0.3) !important;
}
.elo-tier-rebuilding {
    background: rgba(100, 116, 139, 0.15) !important;
    color: #475569 !important;
    border: 1px solid rgba(100, 116, 139, 0.3) !important;
}
.elo-bar-bg {
    background: #e2e8f0 !important;
    border-radius: 4px !important;
    height: 8px !important;
    width: 100px !important;
    overflow: hidden !important;
    display: inline-block !important;
    vertical-align: middle !important;
}
.elo-bar-fill {
    height: 100% !important;
    border-radius: 4px !important;
}
</style>"""

    table_html = styles + """<table class="mlb-elo-table">
<thead>
<tr>
<th style="text-align: center; width: 60px;">#</th>
<th>Equipo</th>
<th>División</th>
<th>Récord</th>
<th>Nivel</th>
<th style="text-align: right;">ELO</th>
<th style="text-align: center; width: 150px;">Fuerza</th>
</tr>
</thead>
<tbody>"""

    elo_vals = [row[1] for row in elo_list]
    max_elo = max(elo_vals) if elo_vals else 1650
    min_elo = min(elo_vals) if elo_vals else 1350
    elo_range = max(1.0, max_elo - min_elo)

    for idx, (code, elo, wins, losses) in enumerate(elo_list):
        rank = idx + 1
        team_info = EQUIPOS_MLB.get(code, {"nombre": code, "logo": "", "color": "#1e40af"})
        logo_url = team_info.get("logo", "")
        team_name = team_info.get("nombre", code)

        div_code = DIVISION_MAP.get(code, "MLB")
        div_desc = div_names.get(div_code, div_code)

        if elo >= 1580:
            tier_class = "elo-tier-elite"
            tier_lbl = "🔥 Elite"
        elif elo >= 1520:
            tier_class = "elo-tier-strong"
            tier_lbl = "💪 Fuerte"
        elif elo >= 1480:
            tier_class = "elo-tier-competitive"
            tier_lbl = "📊 Competitivo"
        elif elo >= 1420:
            tier_class = "elo-tier-developing"
            tier_lbl = "⚠️ En Desarrollo"
        else:
            tier_class = "elo-tier-rebuilding"
            tier_lbl = "📉 En Reconstrucción"

        pct = max(5.0, min(100.0, (elo - min_elo) / elo_range * 100.0))

        if theme_act == "Oscuro":
            if elo >= 1580:
                bar_color = "linear-gradient(90deg, #00e676, #bbf7d0)"
            elif elo >= 1520:
                bar_color = "linear-gradient(90deg, #0055aa, #00c8ff)"
            elif elo >= 1480:
                bar_color = "linear-gradient(90deg, #d97706, #fbbf24)"
            else:
                bar_color = "linear-gradient(90deg, #475569, #94a3b8)"
        else:
            if elo >= 1580:
                bar_color = "#10b981"
            elif elo >= 1520:
                bar_color = "#3b82f6"
            elif elo >= 1480:
                bar_color = "#f59e0b"
            else:
                bar_color = "#94a3b8"

        table_html += f"""<tr>
<td class="elo-rank">{rank}</td>
<td>
<div class="elo-team-col">
<img src="{logo_url}" class="elo-logo" onerror="this.style.display='none'">
<span class="elo-team-name">{team_name}</span>
</div>
</td>
<td style="font-weight: 600;">{div_desc}</td>
<td style="font-family: monospace; font-size: 0.95rem; font-weight: 700;">{wins} - {losses}</td>
<td><span class="elo-badge-tier {tier_class}">{tier_lbl}</span></td>
<td style="text-align: right; font-weight: 900; font-size: 1rem; font-family: monospace;">{elo:.1f}</td>
<td style="text-align: center; vertical-align: middle;">
<div class="elo-bar-bg">
<div class="elo-bar-fill" style="width: {pct}%; background: {bar_color};"></div>
</div>
</td>
</tr>"""

    table_html += "</tbody></table>"
    cleaned_lines = [line.strip() for line in table_html.splitlines() if line.strip()]
    return "".join(cleaned_lines)


def render_resultados_table_html(df):
    """Genera HTML para tabla premium de resultados históricos"""
    import re

    # Obtener el tema actual desde session state
    theme_act = "Oscuro" if "theme_selector" in st.session_state and "🌙 Oscuro" in st.session_state.theme_selector else "Claro"

    # Definir variables de estilo según el tema
    if theme_act == "Oscuro":
        bar_color = "#00c8ff"
        styles = """<style>
.mlb-results-table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 1.5rem 0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    background: #070e1c !important;
    border: 1px solid #1e3f6a !important;
    box-shadow: 0 4px 20px rgba(0, 200, 255, 0.05) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
.mlb-results-table th {
    background: #050c1a !important;
    color: #00c8ff !important;
    font-weight: 700 !important;
    text-align: left !important;
    padding: 0.85rem 1.1rem !important;
    border-bottom: 2px solid #1e3f6a !important;
    text-transform: uppercase !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.5px !important;
}
.mlb-results-table td {
    padding: 0.85rem 1.1rem !important;
    border-bottom: 1px solid rgba(30, 63, 106, 0.4) !important;
    color: #d8eef8 !important;
    vertical-align: middle !important;
}
.mlb-results-table tr:last-child td {
    border-bottom: none !important;
}
.mlb-results-table tr:hover {
    background: #0a1628 !important;
}
.mlb-row-success {
    background: rgba(0, 230, 118, 0.03) !important;
}
.mlb-row-success td:first-child {
    border-left: 4px solid #00e676 !important;
}
.mlb-row-error {
    background: rgba(239, 68, 68, 0.03) !important;
}
.mlb-row-error td:first-child {
    border-left: 4px solid #ef4444 !important;
}
.table-badge {
    display: inline-block !important;
    padding: 0.25rem 0.6rem !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    font-size: 0.72rem !important;
    text-align: center !important;
    white-space: nowrap !important;
}
.table-badge-success {
    background: rgba(0, 230, 118, 0.15) !important;
    color: #00e676 !important;
    border: 1px solid rgba(0, 230, 118, 0.3) !important;
}
.table-badge-error {
    background: rgba(239, 68, 68, 0.15) !important;
    color: #ef4444 !important;
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
}
</style>"""
    else:
        bar_color = "#3b82f6"
        styles = """<style>
.mlb-results-table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 1.5rem 0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.02) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
.mlb-results-table th {
    background: #f8fafc !important;
    color: #0f172a !important;
    font-weight: 700 !important;
    text-align: left !important;
    padding: 0.85rem 1.1rem !important;
    border-bottom: 2px solid #e2e8f0 !important;
    text-transform: uppercase !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.5px !important;
}
.mlb-results-table td {
    padding: 0.85rem 1.1rem !important;
    border-bottom: 1px solid #f1f5f9 !important;
    color: #1e293b !important;
    vertical-align: middle !important;
}
.mlb-results-table tr:last-child td {
    border-bottom: none !important;
}
.mlb-results-table tr:hover {
    background: #f8fafc !important;
}
.mlb-row-success {
    background: rgba(16, 185, 129, 0.03) !important;
}
.mlb-row-success td:first-child {
    border-left: 4px solid #10b981 !important;
}
.mlb-row-error {
    background: rgba(239, 68, 68, 0.03) !important;
}
.mlb-row-error td:first-child {
    border-left: 4px solid #ef4444 !important;
}
.table-badge {
    display: inline-block !important;
    padding: 0.25rem 0.6rem !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    font-size: 0.72rem !important;
    text-align: center !important;
    white-space: nowrap !important;
}
.table-badge-success {
    background: #dcfce7 !important;
    color: #166534 !important;
    border: 1px solid #bbf7d0 !important;
}
.table-badge-error {
    background: #fee2e2 !important;
    color: #991b1b !important;
    border: 1px solid #fecaca !important;
}
</style>"""

    html = styles + '<div style="overflow-x: auto;"><table class="mlb-results-table">'
    html += '<thead><tr>'
    for col in df.columns:
        html += f'<th>{col}</th>'
    html += '</tr></thead><tbody>'

    for _, row in df.iterrows():
        # Determinar clase de fila basada en el Estado
        estado_val = str(row["Estado"]).upper()
        if "ACERTADO" in estado_val or "✅" in estado_val or "TRUE" in estado_val or "1" in estado_val:
            row_class = "mlb-row-success"
        else:
            row_class = "mlb-row-error"

        html += f'<tr class="{row_class}">'
        for col in df.columns:
            val = row[col]

            if col == "Estado":
                if "mlb-row-success" in row_class:
                    html += '<td><span class="table-badge table-badge-success">✅ Acertado</span></td>'
                else:
                    html += '<td><span class="table-badge table-badge-error">❌ Error</span></td>'

            elif col == "Encuentro":
                # Limpiar emojis
                enc_clean = str(val).replace("🏟️", "").strip()
                # Parsear Away @ Home
                teams = enc_clean.split("@")
                if len(teams) == 2:
                    away = teams[0].strip()
                    home = teams[1].strip()
                    away_logo = get_team_logo_html(away, 24)
                    home_logo = get_team_logo_html(home, 24)
                    html += f'<td><div style="display: flex; align-items: center; gap: 8px;">{away_logo} <b>{away}</b> <span style="color: #64748b; font-size: 0.8rem;">@</span> {home_logo} <b>{home}</b></div></td>'
                else:
                    html += f'<td>{val}</td>'

            elif col == "Predicción del Modelo":
                val_clean = str(val).replace("⚾", "").strip()
                parts = val_clean.split("|")
                if len(parts) == 2:
                    pred_team = parts[0].strip()
                    prob_str = parts[1].strip()
                    prob_num_match = re.search(r"[\d\.]+", prob_str)
                    prob_num = float(prob_num_match.group(0)) if prob_num_match else 50.0

                    # Calcular opacidad/intensidad basada en probabilidad
                    opacity = 0.4 + (prob_num - 50.0) / 50.0 * 0.6
                    opacity = max(0.4, min(1.0, opacity))

                    pred_logo = get_team_logo_html(pred_team, 24)

                    html += f'''<td>
<div style="display: flex; align-items: center; gap: 8px;">
{pred_logo}
<div style="display: flex; flex-direction: column; width: 100%; min-width: 100px;">
<div style="display: flex; justify-content: space-between; align-items: center; opacity: {opacity:.2f};">
<span><b>{pred_team}</b></span>
<span style="font-weight: 700; font-size: 0.82rem;">{prob_str}</span>
</div>
<div style="width: 100%; height: 4px; background: rgba(100,116,139,0.15); border-radius: 2px; margin-top: 3px; overflow: hidden;">
<div style="width: {prob_num}%; height: 100%; background: {bar_color}; opacity: {opacity:.2f}; border-radius: 2px;"></div>
</div>
</div>
</div>
</td>'''
                else:
                    html += f'<td>{val}</td>'

            elif col == "Ganador Real":
                winner = str(val).strip()
                winner_logo = get_team_logo_html(winner, 24)
                html += f'<td><div style="display: flex; align-items: center; gap: 8px;">{winner_logo} <b>{winner}</b></div></td>'

            else:
                html += f'<td>{val}</td>'

        html += '</tr>'
    html += '</tbody></table></div>'
    cleaned_lines = [line.strip() for line in html.splitlines() if line.strip()]
    return "".join(cleaned_lines)


@st.cache_data(ttl=60)
def verificar_api_salud():
    """Verifica el estado de la API"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=15)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except Exception as e:
        return False, {"error": str(e)}


def get_team_logo_html(team_code, size=40):
    """Genera HTML para logo de equipo"""
    if team_code in EQUIPOS_MLB:
        logo_url = EQUIPOS_MLB[team_code]["logo"]
        return f'<img src="{logo_url}" width="{size}" style="vertical-align: middle; margin-right: 10px; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));">'
    return ""


def get_team_display_name(team_code):
    """Obtiene nombre completo del equipo"""
    if team_code in EQUIPOS_MLB:
        return EQUIPOS_MLB[team_code]["nombre"]
    return team_code


def ejecutar_scraper_manual():
    """Ejecuta scraper y predicciones manuales para refrescar la jornada visible."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scraper_path = os.path.join(script_dir, "mlb_daily_scraper.py")
    predict_path = os.path.join(script_dir, "mlb_predict_engine.py")
    env = os.environ.copy()
    env["RUN_SOURCE"] = "manual_app"
    env.setdefault("SCRAPER_PREVIEW_RETRIES", "3")
    env.setdefault("SCRAPER_PREVIEW_RETRY_WAIT_SECONDS", "8")
    env.setdefault("SCRAPER_SAVE_PARTIAL_ON_FINAL", "1")

    try:
        with st.spinner("🔍 Actualizando partidos y predicciones del día..."):
            scrape_result = subprocess.run(
                [sys.executable, scraper_path],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=script_dir,
                env=env,
            )

            if scrape_result.returncode != 0:
                error_output = (scrape_result.stderr or scrape_result.stdout or "").strip()
                if error_output:
                    return False, f"❌ El scraper falló: {error_output[-300:]}"
                return False, "⚠️ El scraper no devolvió datos para hoy"

            predict_result = subprocess.run(
                [sys.executable, predict_path],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=script_dir,
                env=env,
            )
            if predict_result.returncode == 0:
                return True, "✅ Partidos y predicciones actualizados exitosamente"

            error_output = (predict_result.stderr or predict_result.stdout or "").strip()
            if error_output:
                return False, f"❌ Las predicciones fallaron: {error_output[-300:]}"
            return (
                False,
                "⚠️ Se actualizaron los partidos, pero falló la generación de predicciones",
            )
    except subprocess.TimeoutExpired:
        return False, "❌ Timeout: El proceso tardó más de 5 minutos"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


def render_system_status_panel(api_ok, api_data):
    """Muestra el estado operativo dentro de la sección informativa."""
    entorno = "Producción" if (os.getenv("API_URL") or has_secrets_api) else "Local"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
        <div class="stats-card">
            <h4>API Backend</h4>
            <p style="font-size: 1.05rem; color: {"#10b981" if api_ok else "#ef4444"}; font-weight: 800;">{"Conectada" if api_ok else "No disponible"}</p>
            <p style="color: #64748b; margin: 0;">Salud general del servicio</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        modelo_ok = bool(api_data and api_data.get("modelo_disponible"))
        st.markdown(
            f"""
        <div class="stats-card">
            <h4>Modelo</h4>
            <p style="font-size: 1.05rem; color: {"#10b981" if modelo_ok else "#ef4444"}; font-weight: 800;">{"Cargado" if modelo_ok else "No disponible"}</p>
            <p style="color: #64748b; margin: 0;">Artefacto predictivo principal</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        db_ok = bool(api_data and api_data.get("base_datos_disponible"))
        st.markdown(
            f"""
        <div class="stats-card">
            <h4>Entorno</h4>
            <p style="font-size: 1.05rem; color: {"#10b981" if db_ok else "#f59e0b"}; font-weight: 800;">{entorno}</p>
            <p style="color: #64748b; margin: 0;">Base de datos {"disponible" if db_ok else "requiere revisión"}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )


def obtener_estadisticas_motor_total():
    """Devuelve métricas históricas del motor usando la base local."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            total_predicciones = conn.execute("SELECT COUNT(*) FROM predicciones_historico").fetchone()[0]

            df_eval = pd.read_sql(
                """
                SELECT
                    CASE
                        WHEN (r.ganador = 1 AND p.prediccion = r.home_team) OR
                             (r.ganador = 0 AND p.prediccion = r.away_team)
                        THEN 1
                        ELSE 0
                    END as acierto
                FROM historico_real r
                INNER JOIN predicciones_historico p
                    ON r.fecha = p.fecha
                    AND r.home_team = p.home_team
                    AND r.away_team = p.away_team
                """,
                conn,
            )

        total_validados = int(len(df_eval))
        aciertos = int(df_eval["acierto"].sum()) if total_validados else 0
        tasa_exito = round((aciertos / total_validados) * 100, 2) if total_validados else 0.0

        return {
            "total_predicciones": int(total_predicciones),
            "total_validados": total_validados,
            "aciertos": aciertos,
            "tasa_exito": tasa_exito,
            "error": None,
        }
    except Exception as e:
        return {
            "total_predicciones": 0,
            "total_validados": 0,
            "aciertos": 0,
            "tasa_exito": 0.0,
            "error": str(e),
        }


def render_motor_lifetime_panel():
    """Renderiza métricas de por vida del motor de predicción."""
    stats = obtener_estadisticas_motor_total()

    total = stats["total_predicciones"]
    validados = stats["total_validados"]
    aciertos = stats["aciertos"]
    tasa = stats["tasa_exito"]
    pendientes = max(0, total - validados)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
        <div class="stats-card">
            <h4>Predicciones Totales</h4>
            <p style="font-size: 1.35rem; color: #1e40af; font-weight: 900;">{total:,}</p>
            <p style="color: #64748b; margin: 0;">Generadas por el motor desde el inicio</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="stats-card">
            <h4>Con Resultado Real</h4>
            <p style="font-size: 1.35rem; color: #7c3aed; font-weight: 900;">{validados:,}</p>
            <p style="color: #64748b; margin: 0;">Partidos jugados y con resultado conocido</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="stats-card">
            <h4>Tasa de Acierto</h4>
            <p style="font-size: 1.35rem; color: #10b981; font-weight: 900;">{tasa:.1f}%</p>
            <p style="color: #64748b; margin: 0;">{aciertos:,} aciertos de {validados:,} validados</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
        <div class="stats-card">
            <h4>Pendientes de Validar</h4>
            <p style="font-size: 1.35rem; color: #f59e0b; font-weight: 900;">{pendientes:,}</p>
            <p style="color: #64748b; margin: 0;">Partidos sin resultado registrado aún</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Barra de progreso visual
    if validados > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("**Cobertura de validación** (partidos con resultado real vs. totales)")
            cobertura = validados / total if total > 0 else 0
            st.progress(cobertura, text=f"{validados:,} de {total:,} ({cobertura * 100:.1f}%)")
        with col_g2:
            st.markdown("**Tasa de acierto** (predicciones correctas sobre validadas)")
            tasa_frac = aciertos / validados if validados > 0 else 0
            color_text = "verde" if tasa_frac >= 0.55 else ("amarillo" if tasa_frac >= 0.50 else "rojo")
            _ = color_text  # solo para referencia semántica
            st.progress(tasa_frac, text=f"{aciertos:,} correctas de {validados:,} ({tasa:.1f}%)")

    if stats["error"]:
        st.warning(f"No se pudieron cargar estadísticas históricas del motor: {stats['error']}")


api_ok, api_data = verificar_api_salud()


def normalizar_probabilidad(prob):
    """
    FIX CRÍTICO: Normaliza probabilidades a rango 0-1
    La API puede devolver en porcentaje (82.5) o decimal (0.825)
    """
    if prob is None:
        return 0.5
    if prob > 1:
        return prob / 100.0
    return prob


def crear_grafico_probabilidades(prob_home, prob_away, home_team, away_team):
    """Crea gráfico de barras de probabilidades - CORREGIDO"""
    # Normalizar probabilidades
    prob_home = normalizar_probabilidad(prob_home)
    prob_away = normalizar_probabilidad(prob_away)

    theme_act = "Oscuro" if "theme_selector" in st.session_state and "🌙 Oscuro" in st.session_state.theme_selector else "Claro"

    if theme_act == "Oscuro":
        colors = [
            "#00c8ff" if prob_home > prob_away else "#1e3f6a",
            "#00c8ff" if prob_away > prob_home else "#1e3f6a",
        ]
        text_color = "#d8eef8"
        grid_color = "rgba(30, 63, 106, 0.5)"
        line_color = "#070e1c"
    else:
        colors = [
            "#3b82f6" if prob_home > prob_away else "#94a3b8",
            "#ef4444" if prob_away > prob_home else "#94a3b8",
        ]
        text_color = "#334155"
        grid_color = "rgba(0,0,0,0.1)"
        line_color = "white"

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[prob_home * 100, prob_away * 100],
            y=[f"{home_team}", f"{away_team}"],
            orientation="h",
            marker={"color": colors, "line": {"color": line_color, "width": 3}},
            text=[f"{prob_home * 100:.1f}%", f"{prob_away * 100:.1f}%"],
            textposition="auto",
            textfont={"size": 18, "color": "white", "family": "Arial Black"},
            hovertemplate="<b>%{y}</b><br>Probabilidad: %{x:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": "Probabilidades de Victoria",
            "font": {"size": 24, "weight": "bold", "color": text_color},
        },
        xaxis_title="Probabilidad (%)",
        height=350,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"size": 14, "family": "Arial", "color": text_color},
        xaxis={"range": [0, 100], "gridcolor": grid_color, "tickfont": {"color": text_color}, "title_font": {"color": text_color}},
        yaxis={"gridcolor": "rgba(0,0,0,0)", "tickfont": {"color": text_color}},
    )

    return fig


def crear_gauge_confianza(confianza):
    """Crea gauge de nivel de confianza - CORREGIDO"""
    # Normalizar confianza
    confianza = normalizar_probabilidad(confianza)

    theme_act = "Oscuro" if "theme_selector" in st.session_state and "🌙 Oscuro" in st.session_state.theme_selector else "Claro"

    if theme_act == "Oscuro":
        axis_color = "#00c8ff"
        bar_color = "#00c8ff"
        bg_gauge = "#050c1a"
        border_color = "#1e3f6a"
        text_color = "#d8eef8"
        steps = [
            {"range": [0, 55], "color": "rgba(239, 68, 68, 0.15)"},
            {"range": [55, 60], "color": "rgba(245, 158, 11, 0.15)"},
            {"range": [60, 70], "color": "rgba(0, 200, 255, 0.15)"},
            {"range": [70, 100], "color": "rgba(0, 230, 118, 0.25)"},
        ]
    else:
        axis_color = "#1e40af"
        bar_color = "#3b82f6"
        bg_gauge = "white"
        border_color = "#cbd5e1"
        text_color = "#334155"
        steps = [
            {"range": [0, 55], "color": "#fee2e2"},
            {"range": [55, 60], "color": "#fed7aa"},
            {"range": [60, 70], "color": "#d9f99d"},
            {"range": [70, 100], "color": "#bbf7d0"},
        ]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confianza * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": "Nivel de Confianza",
                "font": {"size": 24, "weight": "bold", "color": text_color},
            },
            number={"suffix": "%", "font": {"size": 48, "weight": "bold", "color": text_color}},
            gauge={
                "axis": {"range": [None, 100], "tickwidth": 2, "tickcolor": axis_color, "tickfont": {"color": text_color}},
                "bar": {"color": bar_color, "thickness": 0.8},
                "bgcolor": bg_gauge,
                "borderwidth": 3,
                "bordercolor": border_color,
                "steps": steps,
                "threshold": {
                    "line": {"color": "#ef4444", "width": 6},
                    "thickness": 0.85,
                    "value": 70,
                },
            },
        )
    )

    fig.update_layout(
        height=350,
        margin={"l": 30, "r": 30, "t": 60, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": text_color, "family": "Arial"},
    )

    return fig


def obtener_prediccion_detallada_partido(home_team, away_team, home_pitcher, away_pitcher, year, fecha=None):
    """Obtiene la predicción detallada para un partido usando la API."""
    try:
        response = requests.post(
            f"{API_URL}/predict/detailed",
            json={
                "home_team": home_team,
                "away_team": away_team,
                "home_pitcher": home_pitcher,
                "away_pitcher": away_pitcher,
                "year": year,
                "fecha": fecha,
            },
            timeout=120,
        )

        if response.status_code == 200:
            return True, response.json()

        try:
            error = response.json()
            return False, error.get("detail", "Error desconocido")
        except Exception:
            return False, f"Error HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "Timeout: la API tardó más de 120 segundos"
    except Exception as e:
        return False, str(e)


def render_tendencias_html(
    home_team, away_team, win_rate_h, racha_h, win_rate_s_h, record_s_h, win_rate_a, racha_a, win_rate_s_a, record_s_a
):
    """Genera el HTML para la sección de tendencias y momentum"""

    def _racha_badge(racha):
        if racha > 0:
            return f'<span style="background:#1a6b3a;color:#fff;padding:2px 10px;border-radius:12px;font-weight:700">🔥 {racha}G</span>'
        elif racha < 0:
            return f'<span style="background:#8b1a1a;color:#fff;padding:2px 10px;border-radius:12px;font-weight:700">❄️ {abs(racha)}P</span>'
        return '<span style="background:#555;color:#fff;padding:2px 10px;border-radius:12px">—</span>'

    # Cálculo de récord W-L basado en win_rate (L10)
    w_h = int(round(win_rate_h / 10))
    l_h = 10 - w_h
    record_l10_h = f"{w_h}-{l_h}"

    w_a = int(round(win_rate_a / 10))
    l_a = 10 - w_a
    record_l10_a = f"{w_a}-{l_a}"

    # Cargar ELO dinámico en memoria
    elo_dict = {}
    try:
        elo_dict, _ = get_elo_data_in_memory()
    except Exception:
        pass

    elo_h = elo_dict.get(home_team, 1500.0)
    elo_a = elo_dict.get(away_team, 1500.0)

    elo_vals = list(elo_dict.values()) if elo_dict else [1500.0]
    max_elo = max(elo_vals) if elo_vals else 1650
    min_elo = min(elo_vals) if elo_vals else 1350
    elo_range = max(1.0, max_elo - min_elo)

    pct_h = max(5.0, min(100.0, (elo_h - min_elo) / elo_range * 100.0))
    pct_a = max(5.0, min(100.0, (elo_a - min_elo) / elo_range * 100.0))

    theme_act = "Oscuro" if "theme_selector" in st.session_state and "🌙 Oscuro" in st.session_state.theme_selector else "Claro"

    if theme_act == "Oscuro":
        if elo_h >= 1580:
            bar_color_h = "linear-gradient(90deg, #00e676, #bbf7d0)"
        elif elo_h >= 1520:
            bar_color_h = "linear-gradient(90deg, #0055aa, #00c8ff)"
        elif elo_h >= 1480:
            bar_color_h = "linear-gradient(90deg, #d97706, #fbbf24)"
        else:
            bar_color_h = "linear-gradient(90deg, #475569, #94a3b8)"
            
        if elo_a >= 1580:
            bar_color_a = "linear-gradient(90deg, #00e676, #bbf7d0)"
        elif elo_a >= 1520:
            bar_color_a = "linear-gradient(90deg, #0055aa, #00c8ff)"
        elif elo_a >= 1480:
            bar_color_a = "linear-gradient(90deg, #d97706, #fbbf24)"
        else:
            bar_color_a = "linear-gradient(90deg, #475569, #94a3b8)"
    else:
        if elo_h >= 1580:
            bar_color_h = "#10b981"
        elif elo_h >= 1520:
            bar_color_h = "#3b82f6"
        elif elo_h >= 1480:
            bar_color_h = "#f59e0b"
        else:
            bar_color_h = "#64748b"
            
        if elo_a >= 1580:
            bar_color_a = "#10b981"
        elif elo_a >= 1520:
            bar_color_a = "#3b82f6"
        elif elo_a >= 1480:
            bar_color_a = "#f59e0b"
        else:
            bar_color_a = "#64748b"

    html = f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1rem">
        <div style="background:rgba(255,255,255,0.05);border-radius:12px;padding:1.2rem;border:1px solid rgba(255,255,255,0.1)">
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem">
                {get_team_logo_html(home_team, 32)}
                <strong style="font-size:1.1rem">{home_team}</strong>
            </div>
            <div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:0.5rem;text-align:center;margin-bottom:1rem">
                <div>
                    <div style="font-size:0.65rem;opacity:0.7;margin-bottom:4px">W% SEASON</div>
                    <div style="font-size:1.1rem;font-weight:700;color:{"#10b981" if win_rate_s_h >= 50 else "#ef4444"}">{win_rate_s_h:.0f}%</div>
                </div>
                <div>
                    <div style="font-size:0.65rem;opacity:0.7;margin-bottom:4px">RÉCORD</div>
                    <div style="font-size:1.1rem;font-weight:700">{record_s_h}</div>
                </div>
                <div>
                    <div style="font-size:0.65rem;opacity:0.7;margin-bottom:4px">RÉCORD L10</div>
                    <div style="font-size:1.1rem;font-weight:700">{record_l10_h}</div>
                </div>
                <div>
                    <div style="font-size:0.65rem;opacity:0.7;margin-bottom:4px">RACHA</div>
                    <div style="margin-top:2px">{_racha_badge(racha_h)}</div>
                </div>
            </div>
            <hr style="border:0;border-top:1px solid rgba(255,255,255,0.1);margin:0.8rem 0">
            <div style="display:flex;align-items:center;justify-content:space-between">
                <div>
                    <div style="font-size:0.65rem;opacity:0.7;margin-bottom:4px">ELO</div>
                    <div style="font-size:1.1rem;font-weight:700;color:#00c8ff">{elo_h:.1f}</div>
                </div>
                <div style="text-align:right">
                    <div style="font-size:0.65rem;opacity:0.7;margin-bottom:4px;text-align:right;">FUERZA</div>
                    <div style="background:{"#050c1a" if theme_act == "Oscuro" else "#e2e8f0"} !important; border-radius:4px !important; height:8px !important; width:180px !important; overflow:hidden !important; display:inline-block !important; vertical-align:middle !important; border:1px solid {"rgba(255,255,255,0.15)" if theme_act == "Oscuro" else "rgba(0,0,0,0.1)"} !important;">
                        <div style="display:block !important; min-height:8px !important; height:8px !important; border-radius:4px !important; width:{pct_h:.1f}% !important; background:{bar_color_h} !important; float:left !important;"></div>
                    </div>
                </div>
            </div>
        </div>
        <div style="background:rgba(255,255,255,0.05);border-radius:12px;padding:1.2rem;border:1px solid rgba(255,255,255,0.1)">
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem">
                {get_team_logo_html(away_team, 32)}
                <strong style="font-size:1.1rem">{away_team}</strong>
            </div>
            <div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:0.5rem;text-align:center;margin-bottom:1rem">
                <div>
                    <div style="font-size:0.65rem;opacity:0.7;margin-bottom:4px">W% SEASON</div>
                    <div style="font-size:1.1rem;font-weight:700;color:{"#10b981" if win_rate_s_a >= 50 else "#ef4444"}">{win_rate_s_a:.0f}%</div>
                </div>
                <div>
                    <div style="font-size:0.65rem;opacity:0.7;margin-bottom:4px">RÉCORD</div>
                    <div style="font-size:1.1rem;font-weight:700">{record_s_a}</div>
                </div>
                <div>
                    <div style="font-size:0.65rem;opacity:0.7;margin-bottom:4px">RÉCORD L10</div>
                    <div style="font-size:1.1rem;font-weight:700">{record_l10_a}</div>
                </div>
                <div>
                    <div style="font-size:0.65rem;opacity:0.7;margin-bottom:4px">RACHA</div>
                    <div style="margin-top:2px">{_racha_badge(racha_a)}</div>
                </div>
            </div>
            <hr style="border:0;border-top:1px solid rgba(255,255,255,0.1);margin:0.8rem 0">
            <div style="display:flex;align-items:center;justify-content:space-between">
                <div>
                    <div style="font-size:0.65rem;opacity:0.7;margin-bottom:4px">ELO</div>
                    <div style="font-size:1.1rem;font-weight:700;color:#00c8ff">{elo_a:.1f}</div>
                </div>
                <div style="text-align:right">
                    <div style="font-size:0.65rem;opacity:0.7;margin-bottom:4px;text-align:right;">FUERZA</div>
                    <div style="background:{"#050c1a" if theme_act == "Oscuro" else "#e2e8f0"} !important; border-radius:4px !important; height:8px !important; width:180px !important; overflow:hidden !important; display:inline-block !important; vertical-align:middle !important; border:1px solid {"rgba(255,255,255,0.15)" if theme_act == "Oscuro" else "rgba(0,0,0,0.1)"} !important;">
                        <div style="display:block !important; min-height:8px !important; height:8px !important; border-radius:4px !important; width:{pct_a:.1f}% !important; background:{bar_color_a} !important; float:left !important;"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    return "\n".join(line.strip() for line in html.splitlines())


def render_lanzadores_html(home_team, away_team, home_pitcher, away_pitcher, home_pitcher_stats, away_pitcher_stats):
    """Genera HTML premium para la sección de Lanzadores Iniciales en forma de cajas"""
    
    # 1. Home Pitcher Card
    if home_pitcher_stats and isinstance(home_pitcher_stats, dict):
        nombre_h = home_pitcher_stats.get("nombre", home_pitcher)
        era_h = home_pitcher_stats.get("ERA", 0.0)
        whip_h = home_pitcher_stats.get("WHIP", 0.0)
        so9_h = home_pitcher_stats.get("SO9", 0.0)
        
        # Color del ERA
        color_era_h = "#10b981" if era_h < 3.5 else ("#ef4444" if era_h > 4.5 else "#fbbf24")
        # Color del WHIP
        color_whip_h = "#10b981" if whip_h < 1.20 else ("#ef4444" if whip_h > 1.40 else "#fbbf24")
        
        inner_html_h = f"""
            <div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:0.5rem;text-align:center">
                <div>
                    <div style="font-size:0.65rem;opacity:0.7;margin-bottom:4px">ERA</div>
                    <div style="font-size:1.3rem;font-weight:700;color:{color_era_h}">{era_h:.2f}</div>
                </div>
                <div>
                    <div style="font-size:0.65rem;opacity:0.7;margin-bottom:4px">WHIP</div>
                    <div style="font-size:1.3rem;font-weight:700;color:{color_whip_h}">{whip_h:.3f}</div>
                </div>
                <div>
                    <div style="font-size:0.65rem;opacity:0.7;margin-bottom:4px">SO9</div>
                    <div style="font-size:1.3rem;font-weight:700;color:#10b981">{so9_h:.2f}</div>
                </div>
            </div>
        """
    else:
        inner_html_h = f"""
            <div style="text-align:center;padding:1rem;color:#94a3b8;font-size:0.8rem;font-style:italic">
                ⚠️ Sin estadísticas para {home_pitcher}
            </div>
        """
        nombre_h = home_pitcher

    # 2. Away Pitcher Card
    if away_pitcher_stats and isinstance(away_pitcher_stats, dict):
        nombre_a = away_pitcher_stats.get("nombre", away_pitcher)
        era_a = away_pitcher_stats.get("ERA", 0.0)
        whip_a = away_pitcher_stats.get("WHIP", 0.0)
        so9_a = away_pitcher_stats.get("SO9", 0.0)
        
        color_era_a = "#10b981" if era_a < 3.5 else ("#ef4444" if era_a > 4.5 else "#fbbf24")
        color_whip_a = "#10b981" if whip_a < 1.20 else ("#ef4444" if whip_a > 1.40 else "#fbbf24")
        
        inner_html_a = f"""
            <div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:0.5rem;text-align:center">
                <div>
                    <div style="font-size:0.65rem;opacity:0.7;margin-bottom:4px">ERA</div>
                    <div style="font-size:1.3rem;font-weight:700;color:{color_era_a}">{era_a:.2f}</div>
                </div>
                <div>
                    <div style="font-size:0.65rem;opacity:0.7;margin-bottom:4px">WHIP</div>
                    <div style="font-size:1.3rem;font-weight:700;color:{color_whip_a}">{whip_a:.3f}</div>
                </div>
                <div>
                    <div style="font-size:0.65rem;opacity:0.7;margin-bottom:4px">SO9</div>
                    <div style="font-size:1.3rem;font-weight:700;color:#10b981">{so9_a:.2f}</div>
                </div>
            </div>
        """
    else:
        inner_html_a = f"""
            <div style="text-align:center;padding:1rem;color:#94a3b8;font-size:0.8rem;font-style:italic">
                ⚠️ Sin estadísticas para {away_pitcher}
            </div>
        """
        nombre_a = away_pitcher

    html = f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1rem">
        <div style="background:rgba(255,255,255,0.05);border-radius:12px;padding:1.2rem;border:1px solid rgba(255,255,255,0.1)">
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem">
                {get_team_logo_html(home_team, 32)}
                <strong style="font-size:1.1rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{home_team} — {nombre_h}</strong>
            </div>
            {inner_html_h}
        </div>
        <div style="background:rgba(255,255,255,0.05);border-radius:12px;padding:1.2rem;border:1px solid rgba(255,255,255,0.1)">
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem">
                {get_team_logo_html(away_team, 32)}
                <strong style="font-size:1.1rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{away_team} — {nombre_a}</strong>
            </div>
            {inner_html_a}
        </div>
    </div>
    """
    return "\n".join(line.strip() for line in html.splitlines())


def renderizar_analisis_detallado_partido(resultado_detallado, home_team, away_team, home_pitcher, away_pitcher):
    """Renderiza en UI el mismo análisis detallado usado en Predicción Manual."""
    if resultado_detallado.get("modo_degradado"):
        st.warning(
            resultado_detallado.get(
                "mensaje",
                "Analisis rapido activado: la fuente externa no respondio a tiempo.",
            )
        )

    ganador = resultado_detallado.get("ganador", home_team)
    prob_home = normalizar_probabilidad(resultado_detallado.get("prob_home", 0.5))
    prob_away = normalizar_probabilidad(resultado_detallado.get("prob_away", 0.5))

    conf_val = max(prob_home, prob_away)
    if conf_val > 0.70:
        conf_label = "MUY ALTA"
        conf_class = "confidence-muy-alta"
    elif conf_val > 0.60:
        conf_label = "ALTA"
        conf_class = "confidence-alta"
    elif conf_val > 0.55:
        conf_label = "MODERADA"
        conf_class = "confidence-moderada"
    else:
        conf_label = "BAJA"
        conf_class = "confidence-baja"

    # Métricas principales
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"**{home_team}**")
        st.metric(
            "Probabilidad",
            f"{prob_home * 100:.1f}%",
            delta=f"{(prob_home - 0.5) * 100:+.1f}% vs 50%",
        )
    with col_b:
        st.markdown(f"**{away_team}**")
        st.metric(
            "Probabilidad",
            f"{prob_away * 100:.1f}%",
            delta=f"{(prob_away - 0.5) * 100:+.1f}% vs 50%",
        )
    with col_c:
        st.metric("Confianza", f"{conf_val * 100:.1f}%")
        st.markdown(
            f'<p class="{conf_class}" style="text-align: center;">{conf_label}</p>',
            unsafe_allow_html=True,
        )

    # Mostrar información sobre fallback de años si aplica
    year_solicitado = resultado_detallado.get("year_solicitado", resultado_detallado.get("year_usado_home", 2026))
    year_home = resultado_detallado.get("year_usado_home", year_solicitado)
    year_away = resultado_detallado.get("year_usado_away", year_solicitado)
    razon_home = resultado_detallado.get("razon_fallback_home")
    razon_away = resultado_detallado.get("razon_fallback_away")

    hubo_fallback_home = bool(razon_home) or (year_home != year_solicitado)
    hubo_fallback_away = bool(razon_away) or (year_away != year_solicitado)

    if hubo_fallback_home or hubo_fallback_away:
        st.divider()
        st.markdown("### 📊 Información sobre los datos utilizados")

        # Mostrar info del lanzador local si tiene fallback
        if hubo_fallback_home:
            ip_home = resultado_detallado.get("ip_home", 0)
            detalle_home = razon_home if razon_home else f"No había muestra suficiente/estable en {year_solicitado}"
            st.info(
                f"📌 **{home_team} - {home_pitcher}**: Se usaron estadísticas de la temporada "
                f"**{year_home}** para mejorar la estabilidad del análisis. "
                f"Motivo: {detalle_home}."
                f" IP consideradas: **{ip_home}**."
            )

        # Mostrar info del lanzador visitante si tiene fallback
        if hubo_fallback_away:
            ip_away = resultado_detallado.get("ip_away", 0)
            detalle_away = razon_away if razon_away else f"No había muestra suficiente/estable en {year_solicitado}"
            st.info(
                f"📌 **{away_team} - {away_pitcher}**: Se usaron estadísticas de la temporada "
                f"**{year_away}** para mejorar la estabilidad del análisis. "
                f"Motivo: {detalle_away}."
                f" IP consideradas: **{ip_away}**."
            )

        st.markdown(
            "💡 *Este ajuste es normal al inicio de temporada: cuando la muestra de innings "
            "todavía es baja, el sistema usa una temporada previa para mantener predicciones más robustas. "
            "A medida que se acumulen más innings en la temporada actual, el fallback desaparecerá automáticamente.*"
        )
        st.divider()

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.plotly_chart(
            crear_grafico_probabilidades(prob_home, prob_away, home_team, away_team),
            use_container_width=True,
            key=f"prob_chart_{home_team}_{away_team}_{ganador}",
        )
    with col_g2:
        st.plotly_chart(
            crear_gauge_confianza(conf_val),
            use_container_width=True,
            key=f"gauge_chart_{home_team}_{away_team}_{ganador}",
        )

    # Super Features
    st.markdown("### Super Features: Análisis Dinámico de Matchups")
    features = resultado_detallado.get("features_usadas", {})

    if not features or not any("super_" in k for k in features):
        st.info("ℹ️ Las Super Features no están disponibles en esta respuesta de la API")
    else:
        col1, col2, col3 = st.columns(3)

        neut = features.get("super_neutralizacion_whip_ops", 0)
        ventaja_n = home_team if neut < 0 else away_team
        pitcher_n = home_pitcher if neut < 0 else away_pitcher
        rival_n = away_team if neut < 0 else home_team

        with col1:
            st.markdown(
                f"""
            <div class="super-feature">
                <h4>Neutralización WHIP vs OPS</h4>
                <div class="super-feature-advantage">Ventaja: {ventaja_n}</div>
                <p style="font-size: 0.925rem; line-height: 1.5; color: #334155; margin-top: 0.75rem;">
                    <strong>{pitcher_n}</strong> tiene un WHIP que neutraliza efectivamente
                    el OPS del lineup de <strong>{rival_n}</strong>, limitando su producción ofensiva.
                </p>
                <div style="margin-top: 0.75rem; font-size: 0.775rem; color: #64748b; font-weight: 600;">
                    Impacto: <strong>{abs(neut):.4f}</strong>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        res = features.get("super_resistencia_era_ops", 0)
        ventaja_r = home_team if res < 0 else away_team
        pitcher_r = home_pitcher if res < 0 else away_pitcher

        with col2:
            st.markdown(
                f"""
            <div class="super-feature">
                <h4>Resistencia ERA vs Poder</h4>
                <div class="super-feature-advantage">Ventaja: {ventaja_r}</div>
                <p style="font-size: 0.925rem; line-height: 1.5; color: #334155; margin-top: 0.75rem;">
                    <strong>{pitcher_r}</strong> demuestra mayor resistencia ante bateadores
                    de poder, manteniendo su ERA bajo presión ofensiva.
                </p>
                <div style="margin-top: 0.75rem; font-size: 0.775rem; color: #64748b; font-weight: 600;">
                    Impacto: <strong>{abs(res):.4f}</strong>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        muro = features.get("super_muro_bullpen", 0)
        ventaja_m = home_team if muro < 0 else away_team

        with col3:
            st.markdown(
                f"""
            <div class="super-feature">
                <h4>Muro del Bullpen</h4>
                <div class="super-feature-advantage">Ventaja: {ventaja_m}</div>
                <p style="font-size: 0.925rem; line-height: 1.5; color: #334155; margin-top: 0.75rem;">
                    El bullpen de <strong>{ventaja_m}</strong> es más efectivo
                    contra los mejores bateadores rivales en innings críticos.
                </p>
                <div style="margin-top: 0.75rem; font-size: 0.775rem; color: #64748b; font-weight: 600;">
                    Impacto: <strong>{abs(muro):.4f}</strong>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # --- SECCIÓN DE TENDENCIAS Y MOMENTUM ---
    stats_det_t = resultado_detallado.get("stats_detalladas", {})
    tendencias_obj = stats_det_t.get("tendencias") if isinstance(stats_det_t, dict) else None
    features_t = resultado_detallado.get("features_usadas", {})

    # Variables de trabajo
    data_ready = False

    # Caso 1: Datos ya estructurados en 'tendencias'
    if tendencias_obj and isinstance(tendencias_obj, dict):
        try:
            t_h = tendencias_obj.get("home", {})
            win_rate_h = float(t_h.get("win_rate", 0.5)) * 100
            win_rate_s_h = float(t_h.get("win_rate_season", 0.5)) * 100
            record_s_h = t_h.get("season_record", "0-0")
            racha_h = int(t_h.get("racha", 0))

            t_a = tendencias_obj.get("away", {})
            win_rate_a = float(t_a.get("win_rate", 0.5)) * 100
            win_rate_s_a = float(t_a.get("win_rate_season", 0.5)) * 100
            record_s_a = t_a.get("season_record", "0-0")
            racha_a = int(t_a.get("racha", 0))
            data_ready = True
        except Exception:
            data_ready = False

    # Caso 2: Retrocompatibilidad con features_usadas
    if not data_ready and features_t:
        try:
            win_rate_h = float(features_t.get("home_win_rate_10", 0.5)) * 100
            win_rate_s_h = float(features_t.get("home_win_rate_season", 0.5)) * 100
            record_s_h = features_t.get("home_season_record", "0-0")
            racha_h = int(features_t.get("home_racha", 0))

            win_rate_a = float(features_t.get("away_win_rate_10", 0.5)) * 100
            win_rate_s_a = float(features_t.get("away_win_rate_season", 0.5)) * 100
            record_s_a = features_t.get("away_season_record", "0-0")
            racha_a = int(features_t.get("away_racha", 0))
            data_ready = True
        except Exception:
            data_ready = False

    if data_ready:
        st.markdown("---")
        st.markdown("### 📈 Tendencias y Momentum")
        st.markdown(
            render_tendencias_html(
                home_team,
                away_team,
                win_rate_h,
                racha_h,
                win_rate_s_h,
                record_s_h,
                win_rate_a,
                racha_a,
                win_rate_s_a,
                record_s_a,
            ),
            unsafe_allow_html=True,
        )

    # Estadísticas detalladas
    stats_det = resultado_detallado.get("stats_detalladas", {})
    if stats_det and (
        stats_det.get("home_pitcher")
        or stats_det.get("away_pitcher")
        or stats_det.get("home_batters")
        or stats_det.get("away_batters")
    ):
        st.markdown("## Estadísticas Detalladas")
        st.markdown("### Lanzadores Iniciales")

        home_pitcher_stats = stats_det.get("home_pitcher")
        away_pitcher_stats = stats_det.get("away_pitcher")
        st.markdown(
            render_lanzadores_html(
                home_team,
                away_team,
                home_pitcher,
                away_pitcher,
                home_pitcher_stats,
                away_pitcher_stats,
            ),
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### Top 3 Bateadores")

        bcol1, bcol2 = st.columns(2)

        with bcol1:
            home_batters = stats_det.get("home_batters", [])
            if home_batters and isinstance(home_batters, list):
                st.markdown(
                    get_team_logo_html(home_team, 40) + f"- {home_team}",
                    unsafe_allow_html=True,
                )
                for i, batter in enumerate(home_batters[:3], 1):
                    if batter and isinstance(batter, dict):
                        nombre_batter = batter.get("nombre", "N/A")
                        if nombre_batter and nombre_batter != "N/A":
                            with st.expander(f"#{i} - {nombre_batter}", expanded=(i == 1)):
                                k1, k2, k3, k4 = st.columns(4)
                                with k1:
                                    st.metric("OPS", f"{batter.get('OPS', 0):.3f}")
                                with k2:
                                    st.metric("BA", f"{batter.get('BA', 0):.3f}")
                                with k3:
                                    st.metric("HR", int(batter.get("HR", 0)))
                                with k4:
                                    st.metric("RBI", int(batter.get("RBI", 0)))
            else:
                st.info(f"ℹ️ No hay datos de bateadores disponibles para {home_team}")

        with bcol2:
            away_batters = stats_det.get("away_batters", [])
            if away_batters and isinstance(away_batters, list):
                st.markdown(
                    get_team_logo_html(away_team, 40) + f"- {away_team}",
                    unsafe_allow_html=True,
                )
                for i, batter in enumerate(away_batters[:3], 1):
                    if batter and isinstance(batter, dict):
                        nombre_batter = batter.get("nombre", "N/A")
                        if nombre_batter and nombre_batter != "N/A":
                            with st.expander(f"#{i} - {nombre_batter}", expanded=(i == 1)):
                                k1, k2, k3, k4 = st.columns(4)
                                with k1:
                                    st.metric("OPS", f"{batter.get('OPS', 0):.3f}")
                                with k2:
                                    st.metric("BA", f"{batter.get('BA', 0):.3f}")
                                with k3:
                                    st.metric("HR", int(batter.get("HR", 0)))
                                with k4:
                                    st.metric("RBI", int(batter.get("RBI", 0)))
            else:
                st.info(f"ℹ️ No hay datos de bateadores disponibles para {away_team}")


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.image("https://www.mlbstatic.com/team-logos/league-on-dark/1.svg", width=140)

    st.markdown(
        """
    <div style="text-align: center; margin: 1rem 0;">
        <h2 style="margin: 0; color: #ffffff !important; font-size: 1.5rem; font-weight: 800; letter-spacing: 0.5px;">MLB Predictor</h2>
        <span class="badge badge-pro">PRO V4.0</span>
        <span class="badge badge-live">LIVE</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        '<div class="sidebar-section-title">Explorar</div>',
        unsafe_allow_html=True,
    )
    pagina = st.radio(
        "Selecciona una sección:",
        [
            "📅 Partidos de Hoy",
            "📊 Comparación & Historial",
            "🔮 Power Rankings",
            "📈 Rendimiento del Modelo",
            "🧠 Acerca del Modelo",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown(
        '<div class="sidebar-section-title">Tema Visual</div>',
        unsafe_allow_html=True,
    )
    st.selectbox(
        "Selecciona un tema:",
        ["🌙 Oscuro", "☀️ Claro"],
        key="theme_selector",
        label_visibility="collapsed",
    )

# ============================================================================
# PÁGINA: PREDICCIÓN MANUAL - COMPLETAMENTE CORREGIDA
# ============================================================================

if pagina == "⚾ Predicción Manual":
    st.markdown(
        '<div class="main-title">Predicción Manual de Partidos</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Análisis profundo con estadísticas en tiempo real</div>',
        unsafe_allow_html=True,
    )

    if not api_ok:
        st.error("La API no está disponible. Por favor inicia el servidor.")
        st.stop()

    st.markdown("### Configurar Partido")

    col1, col2 = st.columns(2)

    team_options = [f"{code} - {EQUIPOS_MLB[code]['nombre']}" for code in sorted(EQUIPOS_MLB.keys())]

    with col1:
        st.markdown("#### Equipo Local")
        home_sel = st.selectbox("Selecciona equipo local", team_options, key="home_sel_manual")
        home_team = home_sel.split(" - ")[0]
        st.markdown(
            f'<div style="text-align: center; padding: 10px;">{get_team_logo_html(home_team, 100)}</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("#### Equipo Visitante")
        away_sel = st.selectbox("Selecciona equipo visitante", team_options, key="away_sel_manual")
        away_team = away_sel.split(" - ")[0]
        st.markdown(
            f'<div style="text-align: center; padding: 10px;">{get_team_logo_html(away_team, 100)}</div>',
            unsafe_allow_html=True,
        )

    with st.form("prediction_form_manual"):
        col1, col2 = st.columns(2)

        with col1:
            home_pitcher = st.text_input(
                "Lanzador Abridor Local",
                placeholder="Ej: Gerrit Cole, Sandy Alcantara...",
                help="Nombre completo del lanzador",
            )

        with col2:
            away_pitcher = st.text_input(
                "Lanzador Abridor Visitante",
                placeholder="Ej: Spencer Strider, Shane Bieber...",
                help="Nombre completo del lanzador",
            )

        col1, col2 = st.columns([3, 1])
        with col1:
            year = st.number_input(
                "Temporada para estadísticas",
                min_value=2020,
                max_value=2026,
                value=2026,
                help="Año de las estadísticas a utilizar",
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("Realizar Predicción", use_container_width=True, type="primary")

    if submit:
        if not home_pitcher or not away_pitcher:
            st.error("Por favor ingresa los nombres de ambos lanzadores")
        elif home_team == away_team:
            st.error("Los equipos deben ser diferentes")
        else:
            with st.spinner(f"Analizando {home_team} vs {away_team}... Esto puede tardar varios segundos"):
                try:
                    # Llamada a la API
                    response = requests.post(
                        f"{API_URL}/predict/detailed",
                        json={
                            "home_team": home_team,
                            "away_team": away_team,
                            "home_pitcher": home_pitcher,
                            "away_pitcher": away_pitcher,
                            "year": year,
                        },
                        timeout=120,
                    )

                    if response.status_code == 200:
                        resultado = response.json()
                        st.success("¡Predicción completada exitosamente!")

                        # FIX: Extraer y normalizar probabilidades
                        ganador = resultado.get("ganador", home_team)
                        prob_home_raw = resultado.get("prob_home", 0.5)
                        prob_away_raw = resultado.get("prob_away", 0.5)
                        confianza_raw = resultado.get("confianza", 0.5)

                        # Normalizar a rango 0-1
                        prob_home = normalizar_probabilidad(prob_home_raw)
                        prob_away = normalizar_probabilidad(prob_away_raw)
                        confianza = normalizar_probabilidad(confianza_raw)

                        prob_ganador = prob_home if ganador == home_team else prob_away

                        # Renderizar ganador
                        team_name = get_team_display_name(ganador)
                        team_logo = get_team_logo_html(ganador, 80)

                        st.markdown(
                            f"""
                        <div class="winner-box">
                            <div class="winner-title">Ganador Predicho</div>
                            <div style="margin: 1.5rem 0;">
                                {team_logo}
                                <span class="winner-team">{team_name}</span>
                            </div>
                            <div style="font-size: 2rem; font-weight: 700;">
                                {prob_ganador * 100:.1f}% de probabilidad
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                        # Métricas principales
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.markdown(f"**{home_team}**")
                            st.metric(
                                "Probabilidad",
                                f"{prob_home * 100:.1f}%",
                                delta=f"{(prob_home - 0.5) * 100:+.1f}% vs 50%",
                            )

                        with col2:
                            st.markdown(f"**{away_team}**")
                            st.metric(
                                "Probabilidad",
                                f"{prob_away * 100:.1f}%",
                                delta=f"{(prob_away - 0.5) * 100:+.1f}% vs 50%",
                            )

                        with col3:
                            conf_val = max(prob_home, prob_away)
                            if conf_val > 0.70:
                                conf_label = "MUY ALTA"
                                conf_class = "confidence-muy-alta"
                            elif conf_val > 0.60:
                                conf_label = "ALTA"
                                conf_class = "confidence-alta"
                            elif conf_val > 0.55:
                                conf_label = "MODERADA"
                                conf_class = "confidence-moderada"
                            else:
                                conf_label = "BAJA"
                                conf_class = "confidence-baja"

                            st.metric("Confianza", f"{conf_val * 100:.1f}%")
                            st.markdown(
                                f'<p class="{conf_class}" style="text-align: center;">{conf_label}</p>',
                                unsafe_allow_html=True,
                            )

                        # Gráficos
                        st.markdown("---")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.plotly_chart(
                                crear_grafico_probabilidades(prob_home, prob_away, home_team, away_team),
                                use_container_width=True,
                            )

                        with col2:
                            st.plotly_chart(
                                crear_gauge_confianza(conf_val),
                                use_container_width=True,
                            )

                        # FIX: Super Features - Validar existencia
                        st.markdown("---")
                        st.markdown("### Super Features: Análisis Dinámico de Matchups")

                        features = resultado.get("features_usadas", {})

                        if not features or not any("super_" in k for k in features):
                            st.info("ℹ️ Las Super Features no están disponibles en esta respuesta de la API")
                        else:
                            col1, col2, col3 = st.columns(3)

                            # 1. Neutralización
                            neut = features.get("super_neutralizacion_whip_ops", 0)
                            ventaja_n = home_team if neut < 0 else away_team
                            pitcher_n = home_pitcher if neut < 0 else away_pitcher
                            rival_n = away_team if neut < 0 else home_team

                            with col1:
                                st.markdown(
                                    f"""
                                <div class="super-feature">
                                    <h4>Neutralización WHIP vs OPS</h4>
                                    <div class="super-feature-advantage">Ventaja: {ventaja_n}</div>
                                    <p style="font-size: 0.925rem; line-height: 1.5; color: #334155; margin-top: 0.75rem;">
                                        <strong>{pitcher_n}</strong> tiene un WHIP que neutraliza efectivamente
                                        el OPS del lineup de <strong>{rival_n}</strong>, limitando su producción ofensiva.
                                    </p>
                                    <div style="margin-top: 0.75rem; font-size: 0.775rem; color: #64748b; font-weight: 600;">
                                        Impacto: <strong>{abs(neut):.4f}</strong>
                                    </div>
                                </div>
                                """,
                                    unsafe_allow_html=True,
                                )

                            # 2. Resistencia
                            res = features.get("super_resistencia_era_ops", 0)
                            ventaja_r = home_team if res < 0 else away_team
                            pitcher_r = home_pitcher if res < 0 else away_pitcher

                            with col2:
                                st.markdown(
                                    f"""
                                <div class="super-feature">
                                    <h4>Resistencia ERA vs Poder</h4>
                                    <div class="super-feature-advantage">Ventaja: {ventaja_r}</div>
                                    <p style="font-size: 0.925rem; line-height: 1.5; color: #334155; margin-top: 0.75rem;">
                                        <strong>{pitcher_r}</strong> demuestra mayor resistencia ante bateadores
                                        de poder, manteniendo su ERA bajo presión ofensiva.
                                    </p>
                                    <div style="margin-top: 0.75rem; font-size: 0.775rem; color: #64748b; font-weight: 600;">
                                        Impacto: <strong>{abs(res):.4f}</strong>
                                    </div>
                                </div>
                                """,
                                    unsafe_allow_html=True,
                                )

                            # 3. Muro Bullpen
                            muro = features.get("super_muro_bullpen", 0)
                            ventaja_m = home_team if muro < 0 else away_team

                            with col3:
                                st.markdown(
                                    f"""
                                <div class="super-feature">
                                    <h4>Muro del Bullpen</h4>
                                    <div class="super-feature-advantage">Ventaja: {ventaja_m}</div>
                                    <p style="font-size: 0.925rem; line-height: 1.5; color: #334155; margin-top: 0.75rem;">
                                        El bullpen de <strong>{ventaja_m}</strong> es más efectivo
                                        contra los mejores bateadores rivales en innings críticos.
                                    </p>
                                    <div style="margin-top: 0.75rem; font-size: 0.775rem; color: #64748b; font-weight: 600;">
                                        Impacto: <strong>{abs(muro):.4f}</strong>
                                    </div>
                                </div>
                                """,
                                    unsafe_allow_html=True,
                                )

                            # --- SECCIÓN DE TENDENCIAS Y MOMENTUM ---
                            stats_det_t = resultado.get("stats_detalladas", {})
                            tendencias_obj = stats_det_t.get("tendencias") if isinstance(stats_det_t, dict) else None
                            features_t = resultado.get("features_usadas", {})

                            # Variables de trabajo
                            data_ready = False

                            # Caso 1: Datos ya estructurados en 'tendencias'
                            if tendencias_obj and isinstance(tendencias_obj, dict):
                                try:
                                    t_h = tendencias_obj.get("home", {})
                                    win_rate_h = float(t_h.get("win_rate", 0.5)) * 100
                                    win_rate_s_h = float(t_h.get("win_rate_season", 0.5)) * 100
                                    record_s_h = t_h.get("season_record", "0-0")
                                    racha_h = int(t_h.get("racha", 0))

                                    t_a = tendencias_obj.get("away", {})
                                    win_rate_a = float(t_a.get("win_rate", 0.5)) * 100
                                    win_rate_s_a = float(t_a.get("win_rate_season", 0.5)) * 100
                                    record_s_a = t_a.get("season_record", "0-0")
                                    racha_a = int(t_a.get("racha", 0))
                                    data_ready = True
                                except Exception:
                                    data_ready = False

                            # Caso 2: Retrocompatibilidad con features_usadas
                            if not data_ready and features_t:
                                try:
                                    win_rate_h = float(features_t.get("home_win_rate_10", 0.5)) * 100
                                    win_rate_s_h = float(features_t.get("home_win_rate_season", 0.5)) * 100
                                    record_s_h = features_t.get("home_season_record", "0-0")
                                    racha_h = int(features_t.get("home_racha", 0))

                                    win_rate_a = float(features_t.get("away_win_rate_10", 0.5)) * 100
                                    win_rate_s_a = float(features_t.get("away_win_rate_season", 0.5)) * 100
                                    record_s_a = features_t.get("away_season_record", "0-0")
                                    racha_a = int(features_t.get("away_racha", 0))
                                    data_ready = True
                                except Exception:
                                    data_ready = False

                            if data_ready:
                                st.markdown("---")
                                st.markdown("### 📈 Tendencias y Momentum")
                                st.markdown(
                                    render_tendencias_html(
                                        home_team,
                                        away_team,
                                        win_rate_h,
                                        racha_h,
                                        win_rate_s_h,
                                        record_s_h,
                                        win_rate_a,
                                        racha_a,
                                        win_rate_s_a,
                                        record_s_a,
                                    ),
                                    unsafe_allow_html=True,
                                )

                        # FIX: Estadísticas detalladas - Validación completa
                        stats_det = resultado.get("stats_detalladas", {})

                        if stats_det and (
                            stats_det.get("home_pitcher")
                            or stats_det.get("away_pitcher")
                            or stats_det.get("home_batters")
                            or stats_det.get("away_batters")
                        ):
                            st.markdown("---")
                            st.markdown("## Estadísticas Detalladas")

                            st.markdown("### Lanzadores Iniciales")

                            home_pitcher_stats = stats_det.get("home_pitcher")
                            away_pitcher_stats = stats_det.get("away_pitcher")
                            st.markdown(
                                render_lanzadores_html(
                                    home_team,
                                    away_team,
                                    home_pitcher,
                                    away_pitcher,
                                    home_pitcher_stats,
                                    away_pitcher_stats,
                                ),
                                unsafe_allow_html=True,
                            )

                            st.markdown("---")
                            st.markdown("### Top 3 Bateadores")

                            col1, col2 = st.columns(2)

                            with col1:
                                home_batters = stats_det.get("home_batters", [])
                                if home_batters and isinstance(home_batters, list) and len(home_batters) > 0:
                                    st.markdown(
                                        get_team_logo_html(home_team, 40) + f"- {home_team}",
                                        unsafe_allow_html=True,
                                    )
                                    for i, batter in enumerate(home_batters[:3], 1):
                                        if batter and isinstance(batter, dict):
                                            nombre_batter = batter.get("nombre", "N/A")
                                            if nombre_batter and nombre_batter != "N/A":
                                                with st.expander(
                                                    f"#{i} - {nombre_batter}",
                                                    expanded=(i == 1),
                                                ):
                                                    (
                                                        subcol1,
                                                        subcol2,
                                                        subcol3,
                                                        subcol4,
                                                    ) = st.columns(4)
                                                    with subcol1:
                                                        st.metric(
                                                            "OPS",
                                                            f"{batter.get('OPS', 0):.3f}",
                                                        )
                                                    with subcol2:
                                                        st.metric(
                                                            "BA",
                                                            f"{batter.get('BA', 0):.3f}",
                                                        )
                                                    with subcol3:
                                                        st.metric(
                                                            "HR",
                                                            int(batter.get("HR", 0)),
                                                        )
                                                    with subcol4:
                                                        st.metric(
                                                            "RBI",
                                                            int(batter.get("RBI", 0)),
                                                        )
                                else:
                                    st.info(f"ℹ️ No hay datos de bateadores disponibles para {home_team}")

                            with col2:
                                away_batters = stats_det.get("away_batters", [])
                                if away_batters and isinstance(away_batters, list) and len(away_batters) > 0:
                                    st.markdown(
                                        get_team_logo_html(away_team, 40) + f"- {away_team}",
                                        unsafe_allow_html=True,
                                    )
                                    for i, batter in enumerate(away_batters[:3], 1):
                                        if batter and isinstance(batter, dict):
                                            nombre_batter = batter.get("nombre", "N/A")
                                            if nombre_batter and nombre_batter != "N/A":
                                                with st.expander(
                                                    f"#{i} - {nombre_batter}",
                                                    expanded=(i == 1),
                                                ):
                                                    (
                                                        subcol1,
                                                        subcol2,
                                                        subcol3,
                                                        subcol4,
                                                    ) = st.columns(4)
                                                    with subcol1:
                                                        st.metric(
                                                            "OPS",
                                                            f"{batter.get('OPS', 0):.3f}",
                                                        )
                                                    with subcol2:
                                                        st.metric(
                                                            "BA",
                                                            f"{batter.get('BA', 0):.3f}",
                                                        )
                                                    with subcol3:
                                                        st.metric(
                                                            "HR",
                                                            int(batter.get("HR", 0)),
                                                        )
                                                    with subcol4:
                                                        st.metric(
                                                            "RBI",
                                                            int(batter.get("RBI", 0)),
                                                        )
                                else:
                                    st.info(f"ℹ️ No hay datos de bateadores disponibles para {away_team}")

                        # Descargar reporte
                        st.markdown("---")
                        result_json = json.dumps(resultado, indent=2)
                        st.download_button(
                            label="📥 Descargar Reporte Técnico (JSON)",
                            data=result_json,
                            file_name=f"prediccion_{home_team}_vs_{away_team}_{year}.json",
                            mime="application/json",
                        )
                    else:
                        error = response.json()
                        st.error(f"❌ Error: {error.get('detail', 'Error desconocido')}")
                        st.info("💡 Verifica que los nombres de los lanzadores o el año de busqueda sean correctos.")

                except requests.exceptions.Timeout:
                    st.error("⏱️ Timeout: La predicción tardó más de 2 minutos")
                except Exception as e:
                    st.error("❌ Error al procesar la predicción")
                    st.exception(e)

# ============================================================================
# PÁGINA: PARTIDOS DE HOY
# ============================================================================

elif pagina == "📅 Partidos de Hoy":
    st.markdown(
        '<div class="main-title">Partidos y Predicciones del Día</div>',
        unsafe_allow_html=True,
    )

    if not api_ok:
        st.markdown(render_custom_alert("warning", "Servicio no responde", "La API no respondió al health check a tiempo. Intentando cargar datos igualmente..."), unsafe_allow_html=True)

    try:
        response_partidos = requests.get(f"{API_URL}/games/today", timeout=30)
        response_predicciones = requests.get(f"{API_URL}/predictions/today", timeout=30)

        partidos = response_partidos.json() if response_partidos.status_code == 200 else []
        predicciones = response_predicciones.json() if response_predicciones.status_code == 200 else []

        # Combinar datos
        if partidos and predicciones:
            pred_dict = {p["game_id"]: p for p in predicciones}
            for partido in partidos:
                if partido["game_id"] in pred_dict:
                    partido.update(pred_dict[partido["game_id"]])

        # Mostrar y operar siempre con referencia horaria USA (ET).
        ahora_et = datetime.now(ZoneInfo("America/New_York"))
        fecha_hoy = ahora_et.strftime("%Y-%m-%d")
        hora_et = ahora_et.strftime("%H:%M")
        fechas_api = sorted({p.get("fecha") for p in partidos if p.get("fecha")})

        st.caption(
            f"Referencia horaria MLB (ET): {fecha_hoy} {hora_et}. "
            "Si accedes desde otra zona horaria, la jornada visible puede parecer de 'ayer'."
        )

        partidos_hoy = [p for p in partidos if p.get("fecha") == fecha_hoy]
        fecha_mostrada = fecha_hoy
        diff_dias = 0

        if partidos_hoy:
            partidos = partidos_hoy
        elif fechas_api:
            fecha_max = max(fechas_api)
            # Solo permitir fallback a 'ayer'
            from datetime import datetime

            diff_dias = (
                datetime.strptime(fecha_hoy, "%Y-%m-%d").date() - datetime.strptime(fecha_max, "%Y-%m-%d").date()
            ).days

            if diff_dias == 1:
                fecha_mostrada = fecha_max
                partidos = [p for p in partidos if p.get("fecha") == fecha_mostrada]
            else:
                partidos = []
        else:
            partidos = []

        # Renderizar panel de estado de la cartelera del día (estilo Imagen 3)
        col_c1, col_c2, col_c3 = st.columns(3)

        with col_c1:
            st.markdown(
                f"""<div class="stats-card-mini">
<h4>Fecha Actual</h4>
<div class="stats-card-val value-blue">{fecha_hoy}</div>
<div class="stats-card-desc">Referencia horaria MLB (ET)</div>
</div>""",
                unsafe_allow_html=True,
            )

        with col_c2:
            fecha_api_disp = fechas_api[0] if (fechas_api and len(fechas_api) > 0) else "N/A"
            st.markdown(
                f"""<div class="stats-card-mini">
<h4>Fecha API</h4>
<div class="stats-card-val value-purple">{fecha_api_disp}</div>
<div class="stats-card-desc">Última cartelera sincronizada</div>
</div>""",
                unsafe_allow_html=True,
            )

        with col_c3:
            num_juegos = len(partidos)
            color_num = "#10b981" if num_juegos > 0 else "#ef4444"
            bottom_desc = "Partidos programados hoy" if (fecha_mostrada == fecha_hoy and num_juegos > 0) else ("Partidos de la jornada anterior" if num_juegos > 0 else "No hay cartelera programada")
            st.markdown(
                f"""<div class="stats-card-mini">
<h4>Número de Partidos</h4>
<div class="stats-card-val" style="color: {color_num};">{num_juegos} Partidos</div>
<div class="stats-card-desc">{bottom_desc}</div>
</div>""",
                unsafe_allow_html=True,
            )

        # Mostrar alerta de fallback sólo si estamos mostrando el día anterior
        if diff_dias == 1 and len(partidos) > 0:
            st.markdown(
                render_custom_alert(
                    "warning",
                    "Cartelera no disponible para hoy",
                    f"No hay cartelera programada para hoy en ET ({fecha_hoy}). Mostrando la última fecha disponible: {fecha_mostrada}.",
                ),
                unsafe_allow_html=True,
            )

        if not partidos:
            st.markdown(
                """
            <div style="text-align: center; padding: 4rem 2rem; color: #64748b;">
                <div style="font-size: 5rem; margin-bottom: 1rem;"></div>
                <h2>No hay partidos programados para hoy</h2>
                <p style="font-size: 1.1rem; margin: 1rem 0;">
                    Los partidos aún no han sido scrapeados o no hay juegos programados
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            if st.button("Buscar Partidos Manualmente", type="primary", use_container_width=True):
                exito, mensaje = ejecutar_scraper_manual()
                if exito:
                    st.success(mensaje)
                    st.rerun()
                else:
                    st.warning(mensaje)
        else:
            if "detalles_partidos_hoy" not in st.session_state:
                st.session_state.detalles_partidos_hoy = {}

            for partido in partidos:
                with st.container():
                    home_logo = get_team_logo_html(partido["home_team"], 50)
                    away_logo = get_team_logo_html(partido["away_team"], 50)

                    # Intentar cargar detalles y features para la barra de momentum
                    features = {}
                    detalles_str = partido.get("detalles")
                    if detalles_str:
                        try:
                            import json
                            detalles_dict = json.loads(detalles_str) if isinstance(detalles_str, str) else detalles_str
                            features = detalles_dict.get("features_usadas", {})
                        except Exception:
                            pass

                    # Formatear momentum de Away/Visitante
                    away_w_rate = features.get("away_win_rate_season", 0)
                    away_record = features.get("away_season_record", "N/A")
                    away_l10_rate = features.get("away_win_rate_10", 0.5)
                    away_l10_wins = int(away_l10_rate * 10)
                    away_l10_losses = 10 - away_l10_wins
                    away_l10 = f"{away_l10_wins}-{away_l10_losses}"
                    away_racha_val = features.get("away_racha", 0)
                    if away_racha_val > 0:
                        away_racha = f"🔥 {away_racha_val}G"
                        away_racha_class = "racha-win"
                    elif away_racha_val < 0:
                        away_racha = f"❄️ {abs(away_racha_val)}P"
                        away_racha_class = "racha-loss"
                    else:
                        away_racha = "—"
                        away_racha_class = ""

                    # Formatear momentum de Home/Local
                    home_w_rate = features.get("home_win_rate_season", 0)
                    home_record = features.get("home_season_record", "N/A")
                    home_l10_rate = features.get("home_win_rate_10", 0.5)
                    home_l10_wins = int(home_l10_rate * 10)
                    home_l10_losses = 10 - home_l10_wins
                    home_l10 = f"{home_l10_wins}-{home_l10_losses}"
                    home_racha_val = features.get("home_racha", 0)
                    if home_racha_val > 0:
                        home_racha = f"🔥 {home_racha_val}G"
                        home_racha_class = "racha-win"
                    elif home_racha_val < 0:
                        home_racha = f"❄️ {abs(home_racha_val)}P"
                        home_racha_class = "racha-loss"
                    else:
                        home_racha = "—"
                        home_racha_class = ""

                    # Lanzadores y ERA
                    home_pitcher_txt = (partido.get("home_pitcher") or "Por confirmar").strip()
                    away_pitcher_txt = (partido.get("away_pitcher") or "Por confirmar").strip()

                    home_starter_era = features.get("home_starter_ERA")
                    away_starter_era = features.get("away_starter_ERA")

                    home_pitcher_line = f"{home_pitcher_txt}" + (f" · ERA {home_starter_era:.2f}" if home_starter_era is not None else "")
                    away_pitcher_line = f"{away_pitcher_txt}" + (f" · ERA {away_starter_era:.2f}" if away_starter_era is not None else "")

                    # Probabilidades
                    prob_home_raw = partido.get("prob_home", 0.5)
                    prob_away_raw = partido.get("prob_away", 0.5)

                    prob_home = normalizar_probabilidad(prob_home_raw)
                    prob_away = normalizar_probabilidad(prob_away_raw)

                    # Determinar Favorito vs Underdog
                    if prob_home >= prob_away:
                        home_fav_label = "HOME · FAVORITO"
                        home_fav_class = "status-favorite"
                        away_fav_label = "AWAY · MENOS FAVORITO"
                        away_fav_class = "status-underdog"

                        home_prob_class = "prob-fav"
                        away_prob_class = "prob-und"

                        edge = (prob_home - prob_away) * 100
                    else:
                        home_fav_label = "HOME · MENOS FAVORITO"
                        home_fav_class = "status-underdog"
                        away_fav_label = "AWAY · FAVORITO"
                        away_fav_class = "status-favorite"

                        home_prob_class = "prob-und"
                        away_prob_class = "prob-fav"

                        edge = (prob_away - prob_home) * 100

                    if "prediccion" in partido:
                        pred_team = get_team_display_name(partido["prediccion"])
                        confianza = partido.get("confianza", "N/A")

                        # Racha styles
                        away_racha_style = f'class="racha-pill {away_racha_class}"' if away_racha_class else 'style="color:#64748b"'
                        home_racha_style = f'class="racha-pill {home_racha_class}"' if home_racha_class else 'style="color:#64748b"'

                        away_logo_url = EQUIPOS_MLB.get(partido["away_team"], {}).get("logo", "")
                        home_logo_url = EQUIPOS_MLB.get(partido["home_team"], {}).get("logo", "")

                        # Estilos para barra de progreso de probabilidad
                        away_bar_opacity = "opacity: 1.0;" if prob_away >= prob_home else "opacity: 0.35;"
                        home_bar_opacity = "opacity: 1.0;" if prob_home >= prob_away else "opacity: 0.35;"
                        away_bar_style = f"width:{prob_away * 100:.0f}%; {away_bar_opacity}"
                        home_bar_style = f"width:{prob_home * 100:.0f}%; {home_bar_opacity}"

                        # Render del scoreboard premium sin indentación al inicio de línea
                        matchup_html = f"""<div class="mlb-matchup-container">
<div class="mlb-matchup-top-header">
<div class="mlb-matchup-pred-badge">
PREDICCIÓN: <b>{pred_team}</b>
</div>
<div class="mlb-matchup-conf-pill">
CONFIANZA: <b>{confianza}</b>
</div>
</div>
<div class="mlb-matchup-grid">
<div class="mlb-matchup-team">
<img class="bg-logo" src="{away_logo_url}" alt="">
<div class="mlb-matchup-logo-container">{away_logo}</div>
<div class="mlb-matchup-status-label {away_fav_class}">{away_fav_label}</div>
<div class="mlb-matchup-team-name">{partido["away_team"]}</div>
<div class="mlb-matchup-pitcher-name">{away_pitcher_line}</div>
<div class="mlb-matchup-prob-value {away_prob_class}">{prob_away * 100:.0f}%</div>
<div class="mlb-matchup-prob-label">Probabilidad de ganar</div>
<div class="mlb-p-bar"><div class="mlb-p-bar-fill" style="{away_bar_style}"></div></div>
</div>
<div class="mlb-matchup-center">
<div class="mlb-matchup-vs">VS</div>
<div class="mlb-matchup-edge-box">
<span class="mlb-matchup-edge-value">+{edge:.1f}%</span>
<span class="mlb-matchup-edge-label">Ventaja Modelo</span>
</div>
<div class="mlb-matchup-stadium">Ventaja local</div>
</div>
<div class="mlb-matchup-team">
<img class="bg-logo" src="{home_logo_url}" alt="">
<div class="mlb-matchup-logo-container">{home_logo}</div>
<div class="mlb-matchup-status-label {home_fav_class}">{home_fav_label}</div>
<div class="mlb-matchup-team-name">{partido["home_team"]}</div>
<div class="mlb-matchup-pitcher-name">{home_pitcher_line}</div>
<div class="mlb-matchup-prob-value {home_prob_class}">{prob_home * 100:.0f}%</div>
<div class="mlb-matchup-prob-label">Probabilidad de ganar</div>
<div class="mlb-p-bar"><div class="mlb-p-bar-fill" style="{home_bar_style}"></div></div>
</div>
</div>
<div class="mlb-momentum-bar">
<div class="mlb-momentum-column column-away">
<span class="mlb-momentum-team-badge">{partido["away_team"]}</span>
<div class="mlb-momentum-stat">
<span class="mlb-momentum-stat-label">W% Season</span>
<span class="mlb-momentum-stat-value">{away_w_rate * 100:.0f}%</span>
</div>
<div class="mlb-momentum-stat">
<span class="mlb-momentum-stat-label">Récord</span>
<span class="mlb-momentum-stat-value">{away_record}</span>
</div>
<div class="mlb-momentum-stat">
<span class="mlb-momentum-stat-label">Récord L10</span>
<span class="mlb-momentum-stat-value">{away_l10}</span>
</div>
<div class="mlb-momentum-stat">
<span class="mlb-momentum-stat-label">Racha</span>
<span {away_racha_style}>{away_racha}</span>
</div>
</div>
<div class="mlb-momentum-column column-home">
<div class="mlb-momentum-stat">
<span class="mlb-momentum-stat-label">W% Season</span>
<span class="mlb-momentum-stat-value">{home_w_rate * 100:.0f}%</span>
</div>
<div class="mlb-momentum-stat">
<span class="mlb-momentum-stat-label">Récord</span>
<span class="mlb-momentum-stat-value">{home_record}</span>
</div>
<div class="mlb-momentum-stat">
<span class="mlb-momentum-stat-label">Récord L10</span>
<span class="mlb-momentum-stat-value">{home_l10}</span>
</div>
<div class="mlb-momentum-stat">
<span class="mlb-momentum-stat-label">Racha</span>
<span {home_racha_style}>{home_racha}</span>
</div>
<span class="mlb-momentum-team-badge">{partido["home_team"]}</span>
</div>
</div>
</div>"""
                        st.markdown(matchup_html, unsafe_allow_html=True)

                        game_id = partido.get(
                            "game_id",
                            f"{partido.get('fecha', '')}_{partido['away_team']}_{partido['home_team']}",
                        )

                        with st.expander(
                            "Ver Análisis Detallado & Desglose de Pitching/Bateo",
                            expanded=False,
                        ):
                            home_pitcher = partido.get("home_pitcher", "")
                            away_pitcher = partido.get("away_pitcher", "")
                            year_detalle = int(str(partido.get("fecha", fecha_hoy))[:4])

                            if not home_pitcher or not away_pitcher:
                                st.warning(
                                    "No hay lanzadores definidos para este partido. No se puede generar el análisis detallado."
                                )
                            else:
                                if game_id not in st.session_state.detalles_partidos_hoy:
                                    if st.button(
                                        "Cargar análisis detallado",
                                        key=f"load_detail_{game_id}",
                                        use_container_width=True,
                                    ):
                                        with st.spinner(
                                            f"Generando análisis detallado para {partido['away_team']} @ {partido['home_team']}..."
                                        ):
                                            ok_detalle, payload = obtener_prediccion_detallada_partido(
                                                partido["home_team"],
                                                partido["away_team"],
                                                home_pitcher,
                                                away_pitcher,
                                                year_detalle,
                                                partido.get("fecha", fecha_hoy),
                                            )
                                        if ok_detalle:
                                            st.session_state.detalles_partidos_hoy[game_id] = payload
                                            st.rerun()
                                        else:
                                            st.error(f"No se pudo cargar el análisis detallado: {payload}")
                                            st.caption(
                                                "Tip: si el error persiste, intenta borrar la caché de la app "
                                                "con el menú ⋮ → Clear cache, o recarga la página."
                                            )
                                else:
                                    renderizar_analisis_detallado_partido(
                                        st.session_state.detalles_partidos_hoy[game_id],
                                        partido["home_team"],
                                        partido["away_team"],
                                        home_pitcher,
                                        away_pitcher,
                                    )

                    st.markdown("---")
    except Exception as e:
        st.error(f"❌ Error obteniendo datos: {str(e)}")

# ============================================================================
# PÁGINA: COMPARACIÓN & HISTORIAL
# ============================================================================

elif pagina == "📊 Comparación & Historial":
    st.markdown(
        '<div class="main-title"> Comparación de Predicciones vs Resultados</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Analiza el rendimiento histórico del modelo</div>',
        unsafe_allow_html=True,
    )

    if not api_ok:
        st.error("❌ La API no está disponible")
        st.stop()

    estado_fechas = {}
    try:
        response_estado = requests.get(f"{API_URL}/status/dates", timeout=10)
        if response_estado.status_code == 200:
            estado_fechas = response_estado.json()
    except Exception:
        estado_fechas = {}

    compare_latest = estado_fechas.get("compare_latest")
    predictions_latest = estado_fechas.get("predictions_latest")
    # Usar la fecha de predicciones como default si compare_latest es viejo (>3 días)
    default_compare_date = datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)
    fecha_ref = predictions_latest or compare_latest
    if fecha_ref:
        try:
            default_compare_date = datetime.strptime(fecha_ref, "%Y-%m-%d").date()
        except ValueError:
            pass

    if "fecha_analizar" not in st.session_state:
        st.session_state.fecha_analizar = default_compare_date

    st.markdown("### 📅 Selecciona una Fecha")

    if predictions_latest:
        st.caption(
            f"Última fecha con predicciones: {predictions_latest}"
            + (f" | Última fecha con resultados reales: {compare_latest}" if compare_latest else "")
        )

    col1, col2, col3 = st.columns([3, 1, 1], vertical_alignment="bottom")

    with col1:
        fecha_seleccionada = st.date_input(
            "Fecha",
            value=st.session_state.fecha_analizar,
            max_value=datetime.now().date(),
        )

    with col2:
        if st.button("📅 Analizar Fecha", type="primary", use_container_width=True):
            st.session_state.fecha_analizar = fecha_seleccionada
            st.session_state.pop("stats_30d", None)

    with col3:
        if st.button("📈 Ver Estadísticas", use_container_width=True):
            try:
                response = requests.get(f"{API_URL}/stats/accuracy?dias=30")
                if response.status_code == 200:
                    st.session_state.stats_30d = response.json()
                else:
                    st.session_state.stats_30d = None
            except Exception as e:
                st.error(f"Error obteniendo estadísticas: {e}")
                st.session_state.stats_30d = None

    _stats_30d = st.session_state.get("stats_30d")
    if isinstance(_stats_30d, dict):
        _s = _stats_30d
        _periodo = _s.get("periodo", "Últimos 30 días")
        _total = _s.get("total", 0)
        _aciertos = _s.get("aciertos", 0)
        _acc = _s.get("accuracy_general", 0)
        _por_conf = _s.get("por_confianza", {})

        # 1. Caja General (Estadísticas Últimos 30 Días)
        st.markdown(
            f"""<div class="rendimiento-container">
<div class="rendimiento-header">
<div style="display: flex; align-items: center; gap: 12px;">
<div style="font-size: 2.2rem; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.15));">📈</div>
<div>
<div class="rendimiento-title">Estadísticas — {_periodo}</div>
<div class="rendimiento-subtitle">Rendimiento histórico acumulado del modelo</div>
</div>
</div>
</div>
<div class="rendimiento-stats">
<div class="rendimiento-stat-box">
<span class="rendimiento-stat-val">{_total}</span>
<span class="rendimiento-stat-lbl">Total Partidos</span>
</div>
<div class="rendimiento-stat-box">
<span class="rendimiento-stat-val" style="color: #10b981;">{_aciertos}</span>
<span class="rendimiento-stat-lbl">Aciertos</span>
</div>
<div class="rendimiento-stat-box">
<span class="rendimiento-stat-val" style="color: #3b82f6;">{_acc:.1f}%</span>
<span class="rendimiento-stat-lbl">Accuracy General</span>
</div>
</div>
</div>""",
            unsafe_allow_html=True,
        )

        # 2. Caja de Desglose por Nivel de Confianza
        if _por_conf:
            # Crear las cajas para cada nivel de confianza ordenadas
            boxes_html = ""
            _iconos = {
                "MUY ALTA": "🔴",
                "ALTA": "🟡",
                "MODERADA": "🟢",
                "BAJA": "🟣",
                "BAJA (Partido muy parejo)": "🟣",
                "BAJA (PARTIDO MUY PAREJO)": "🟣",
            }
            _colores = {
                "MUY ALTA": "#ef4444",
                "ALTA": "#eab308",
                "MODERADA": "#10b981",
                "BAJA": "#b388ff",
                "BAJA (Partido muy parejo)": "#b388ff",
                "BAJA (PARTIDO MUY PAREJO)": "#b388ff",
            }

            # Ordenar claves de manera robusta: MUY ALTA, ALTA, MODERADA, BAJA
            ordered_keys = []
            for k in _por_conf.keys():
                if k.upper() == "MUY ALTA":
                    ordered_keys.append(k)
            for k in _por_conf.keys():
                if k.upper() == "ALTA":
                    ordered_keys.append(k)
            for k in _por_conf.keys():
                if k.upper() == "MODERADA":
                    ordered_keys.append(k)
            for k in _por_conf.keys():
                if "BAJA" in k.upper():
                    ordered_keys.append(k)
            for k in _por_conf.keys():
                if k not in ordered_keys:
                    ordered_keys.append(k)

            for _nivel in ordered_keys:
                _datos = _por_conf[_nivel]
                _icono = _iconos.get(_nivel, "⚪")
                _color = _colores.get(_nivel, "#64748b")
                if _icono == "⚪" and "BAJA" in _nivel.upper():
                    _icono = "🟣"
                    _color = "#b388ff"
                _acc_nivel = _datos.get('accuracy', 0)
                _aciertos_nivel = _datos.get('aciertos', 0)
                _total_nivel = _datos.get('total', 0)

                boxes_html += f"""
<div class="rendimiento-stat-box" style="min-width: 140px;">
<span class="rendimiento-stat-val" style="color: {_color};">{_acc_nivel:.1f}%</span>
<span class="rendimiento-stat-lbl" style="font-weight: 800; font-size: 0.65rem;">{_icono} {_nivel}</span>
<span class="rendimiento-stat-lbl" style="text-transform: none; font-size: 0.55rem; color: #64748b; margin-top: 2px;">{_aciertos_nivel}/{_total_nivel} aciertos</span>
</div>"""

            st.markdown(
                f"""<div class="rendimiento-container" style="border-left: 6px solid #eab308 !important;">
<div class="rendimiento-header">
<div style="display: flex; align-items: center; gap: 12px;">
<div style="font-size: 2.2rem; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.15));">🎯</div>
<div>
<div class="rendimiento-title">Nivel de Confianza</div>
<div class="rendimiento-subtitle">Desglose de precisión por seguridad</div>
</div>
</div>
</div>
<div class="rendimiento-stats">
{boxes_html}
</div>
</div>""",
                unsafe_allow_html=True,
            )
        st.divider()

    if "fecha_analizar" in st.session_state:
        fecha_str = st.session_state.fecha_analizar.strftime("%Y-%m-%d")

        with st.spinner(f"Cargando datos para {fecha_str}..."):
            try:
                response = requests.get(f"{API_URL}/compare/{fecha_str}")
                comparacion = response.json() if response.status_code == 200 else None
            except Exception as e:
                comparacion = None
                st.error(f"Error obteniendo comparación: {e}")

        if comparacion and comparacion["partidos"]:
            partidos = comparacion["partidos"]
            stats = comparacion["estadisticas"]
            solo_predicciones = stats.get("solo_predicciones", False)

            if solo_predicciones:
                if theme == "Oscuro":
                    bg_color = "#050c1a"
                    border_style = "border: 1px solid #1e3f6a; border-left: 4px solid #00c8ff;"
                    text_color = "#d8eef8"
                    title_color = "#00c8ff"
                    hr_color = "#1e3f6a"
                    link_color = "#00c8ff"
                else:
                    bg_color = "#eff6ff"
                    border_style = "border-left: 4px solid #3b82f6;"
                    text_color = "#1e40af"
                    title_color = "#1e3a8a"
                    hr_color = "#bfdbfe"
                    link_color = "#1d4ed8"

                st.markdown(
                    f"""
                    <div style="background-color: {bg_color}; {border_style} padding: 1.25rem; border-radius: 0.5rem; margin-bottom: 2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); font-family: 'Inter', sans-serif;">
                        <div style="display: flex; align-items: center; margin-bottom: 0.75rem;">
                            <span style="font-size: 1.5rem; margin-right: 0.75rem;">📅</span>
                            <span style="color: {title_color}; font-weight: 700; font-size: 1.15rem; font-family: 'Outfit', sans-serif;">
                                Resultados Reales Aún No Disponibles
                            </span>
                        </div>
                        <p style="color: {text_color}; font-size: 0.975rem; line-height: 1.6; margin: 0;">
                            Los resultados reales para la fecha <b>{fecha_str}</b> aún no han sido registrados en la base de datos o los partidos están en desarrollo/programados para jugarse hoy.
                        </p>
                        <hr style="border: 0; border-top: 1px solid {hr_color}; margin: 1rem 0;">
                        <p style="color: {link_color}; font-size: 0.925rem; line-height: 1.5; margin: 0; font-weight: 500;">
                            💡 <b>¿Quieres ver la cartelera y predicciones de hoy?</b><br>
                            Dirígete a la sección <b>📅 Partidos de Hoy</b> en el menú lateral izquierdo para ver lineups oficiales y análisis detallados en tiempo real.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                errores = stats["total"] - stats["aciertos"]
                st.markdown(
                    f"""<div class="rendimiento-container">
<div class="rendimiento-header">
<div style="display: flex; align-items: center; gap: 12px;">
<div style="font-size: 2.2rem; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.15));">⚾</div>
<div>
<div class="rendimiento-title">Rendimiento del Día</div>
<div class="rendimiento-subtitle">Métricas de precisión y aciertos del modelo</div>
</div>
</div>
</div>
<div class="rendimiento-stats">
<div class="rendimiento-stat-box">
<span class="rendimiento-stat-val">{stats["total"]}</span>
<span class="rendimiento-stat-lbl">Total Partidos</span>
</div>
<div class="rendimiento-stat-box">
<span class="rendimiento-stat-val" style="color: #10b981;">{stats["aciertos"]}</span>
<span class="rendimiento-stat-lbl">Aciertos</span>
</div>
<div class="rendimiento-stat-box">
<span class="rendimiento-stat-val" style="color: #3b82f6;">{stats["accuracy"]:.1f}%</span>
<span class="rendimiento-stat-lbl">Accuracy</span>
</div>
<div class="rendimiento-stat-box">
<span class="rendimiento-stat-val" style="color: #ef4444;">{errores}</span>
<span class="rendimiento-stat-lbl">Errores</span>
</div>
</div>
</div>""",
                    unsafe_allow_html=True,
                )

                st.markdown("### Resultados Detallados")

                # Grid de Tarjetas estilo MLB
                num_cols = 2
                for i in range(0, len(partidos), num_cols):
                    grid_cols = st.columns(num_cols)
                    for j in range(num_cols):
                        if i + j < len(partidos):
                            partido = partidos[i + j]
                            with grid_cols[j]:
                                acierto = partido.get("acierto", 0)
                                _ht = partido["home_team"]
                                _at = partido["away_team"]
                                _hn = get_team_display_name(_ht)
                                _an = get_team_display_name(_at)
                                _h_logo = get_team_logo_html(_ht, 32)
                                _a_logo = get_team_logo_html(_at, 32)
                                _score_a = partido.get("score_away", "-")
                                _score_h = partido.get("score_home", "-")

                                tipo_pred = partido.get("tipo") or ""
                                es_fallback = "FALLBACK" in str(tipo_pred).upper()

                                if es_fallback:
                                    _badge_icon = "⚠️"
                                    _badge_text = "PRECAUCIÓN: SIN DATOS COMPLETOS"
                                    _badge_color = "#fef3c7"  # Amarillo/Ámbar suave
                                    _text_color = "#b45309"   # Marrón/Naranja oscuro
                                else:
                                    _badge_icon = "✅" if acierto else "❌"
                                    _badge_text = "ACIERTO" if acierto else "ERROR"
                                    _badge_color = "#dcfce7" if acierto else "#fee2e2"
                                    _text_color = "#166534" if acierto else "#991b1b"

                                # Renderizado de Tarjeta
                                st.markdown(
                                    f"""
<div class="mlb-scoreboard-card">
    <div class="mlb-card-header">
        <span>FINAL</span>
        <span>{partido.get("fecha", "")}</span>
    </div>
    <div class="mlb-card-body">
        <div class="mlb-team-row">
            <div class="mlb-team-info">
                {_a_logo}
                <div class="mlb-team-name">{_an}</div>
            </div>
            <div class="mlb-score">{_score_a}</div>
        </div>
        <div class="mlb-team-row">
            <div class="mlb-team-info">
                {_h_logo}
                <div class="mlb-team-name">{_hn}</div>
            </div>
            <div class="mlb-score">{_score_h}</div>
        </div>
    </div>
    <div class="mlb-card-footer">
        <div class="mlb-prediction-badge" style="background:{_badge_color}; color:{_text_color};">
            {_badge_icon} <b>{_badge_text}</b>
        </div>
        <div style="font-size: 0.75rem; color: #64748b;">
            Predicción: <b>{get_team_display_name(partido.get("prediccion", ""))}</b>
        </div>
    </div>
</div>
""",
                                    unsafe_allow_html=True,
                                )
                                # Botón de detalles opcional
                                with st.expander("🔍 Ver Análisis Detallado", expanded=False):
                                    prob_h = normalizar_probabilidad(partido.get("prob_home", 0.5))
                                    prob_a = normalizar_probabilidad(partido.get("prob_away", 0.5))

                                    # Confianza
                                    confianza = partido.get("confianza", "N/A")
                                    color_conf = {
                                        "MUY ALTA": "#ef4444",
                                        "ALTA": "#f59e0b",
                                        "MODERADA": "#22c55e",
                                        "BAJA": "#64748b",
                                    }.get(confianza, "#64748b")

                                    st.markdown(
                                        f"**Confianza:** <span style='color:{color_conf}; font-weight:bold;'>{confianza}</span>",
                                        unsafe_allow_html=True,
                                    )

                                    col_p1, col_p2 = st.columns(2)
                                    with col_p1:
                                        st.metric(
                                            label=f"Prob. {get_team_display_name(partido.get('away_team', 'Visitante'))}",
                                            value=f"{prob_a * 100:.1f}%"
                                        )
                                    with col_p2:
                                        st.metric(
                                            label=f"Prob. {get_team_display_name(partido.get('home_team', 'Home'))}",
                                            value=f"{prob_h * 100:.1f}%"
                                        )
                                    if "stats_detalladas" in partido:
                                        st.json(partido["stats_detalladas"], expanded=False)
        else:
            st.info(f" No hay datos disponibles para {fecha_str}")

# ============================================================================
# PÁGINA: POWER RANKINGS
# ============================================================================

elif pagina == "🔮 Power Rankings":
    st.markdown(
        '<div class="main-title">Power Rankings (Sistema ELO)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Clasificación dinámica de fuerza y probabilidad relativa de equipos</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🏆 Clasificación de Fuerza de Equipos (ELO)")

    legend_html = """<div class="elo-legend-container" style="background: rgba(0, 200, 255, 0.04); border: 1px solid rgba(0, 200, 255, 0.15); border-radius: 12px; padding: 16px; margin-bottom: 20px; font-family: 'Inter', sans-serif;">
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
<span style="font-size: 1.3rem;">📚</span>
<h4 style="margin: 0; color: #00c8ff; font-weight: 700; font-size: 1.05rem;">Leyenda y Funcionamiento del Sistema ELO</h4>
</div>
<p style="margin: 0 0 12px 0; font-size: 0.85rem; color: #d8eef8; line-height: 1.45;">
El <strong>Sistema ELO</strong> es un método matemático de clasificación que calcula la fuerza competitiva relativa de cada equipo de forma dinámica.
Todos los equipos inician con una base neutral de <strong>1500 puntos</strong>. Tras cada partido, el equipo ganador sustrae puntos al perdedor según la expectativa previa de victoria (vencer a un rival más fuerte otorga más puntos). Se incluye una ventaja de localía de <strong>+24 puntos</strong> para el equipo de casa en la evaluación de cada encuentro.
</p>
<div style="display: flex; gap: 8px; flex-wrap: wrap; font-size: 0.75rem; font-weight: 700;">
<span style="background: rgba(0, 230, 118, 0.12); color: #00e676; border: 1px solid rgba(0, 230, 118, 0.25); padding: 4px 8px; border-radius: 6px;">🔥 Elite (&ge; 1580)</span>
<span style="background: rgba(0, 200, 255, 0.12); color: #00c8ff; border: 1px solid rgba(0, 200, 255, 0.25); padding: 4px 8px; border-radius: 6px;">💪 Fuerte (1520 - 1579)</span>
<span style="background: rgba(251, 191, 36, 0.12); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.25); padding: 4px 8px; border-radius: 6px;">📊 Competitivo (1480 - 1519)</span>
<span style="background: rgba(244, 63, 94, 0.12); color: #f43f5e; border: 1px solid rgba(244, 63, 94, 0.25); padding: 4px 8px; border-radius: 6px;">⚠️ En Desarrollo (1420 - 1479)</span>
<span style="background: rgba(148, 163, 184, 0.12); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.25); padding: 4px 8px; border-radius: 6px;">📉 En Reconstrucción (&lt; 1420)</span>
</div>
</div>"""
    st.markdown("".join([line.strip() for line in legend_html.splitlines()]), unsafe_allow_html=True)

    elo_list = get_elo_power_rankings()
    if elo_list:
        elo_table_html = render_power_rankings_table_html(elo_list)
        st.markdown(elo_table_html, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No se pudieron cargar los Power Rankings ELO.")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📚 ¿Cómo interpretar la clasificación ELO y sus niveles?"):
        st.markdown("""
        El **Sistema ELO** es un método matemático para calcular la fuerza competitiva relativa de los equipos:

        * **Puntuación Base (1500):** Es el punto de partida neutral. Un equipo con ELO de 1500 tiene un rendimiento equilibrado.
        * **Cálculo Dinámico:** Tras cada partido, el equipo ganador toma puntos del perdedor. La cantidad de puntos ganados depende de la expectativa del partido.
        * **Calidad de Oposición:** Si un equipo de bajo nivel vence a uno de nivel alto (sorpresa), la transferencia de puntos es mucho mayor.
        * **Ventaja de Localía (+24 pts):** Al evaluar un partido, al equipo de casa se le suman artificialmente 24 puntos de ELO para representar la ventaja estadística de jugar en su estadio.

        #### 🎖️ Niveles de Clasificación (Tiers):
        * 🔥 **Elite (≥ 1580):** Contendientes indiscutibles al campeonato. Rendimiento dominante y rachas de victorias constantes.
        * 💪 **Fuerte (1520 - 1579):** Equipos muy sólidos con altas probabilidades de clasificar a postemporada.
        * 📊 **Competitivo (1480 - 1519):** Equipos de nivel medio, capaces de pelear por puestos de comodín (*Wild Card*).
        * ⚠️ **En Desarrollo (1420 - 1479):** Equipos en transición con problemas de consistencia o rachas negativas.
        * 📉 **En Reconstrucción (< 1420):** Equipos en la parte baja de la tabla, enfocados en desarrollar talento joven.
        """)

# ============================================================================
# PÁGINA: RENDIMIENTO DEL MODELO
# ============================================================================

elif pagina == "📈 Rendimiento del Modelo":
    st.markdown(
        '<div class="main-title">Rendimiento del Modelo</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Análisis visual del rendimiento histórico del modelo</div>',
        unsafe_allow_html=True,
    )

    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Consulta a la base de datos
            df_dash = pd.read_sql(
                """
                SELECT
                    r.fecha as Fecha,
                    r.home_team as Home,
                    r.away_team as Away,
                    p.prediccion as Prediccion,
                    p.prob_home as Prob_Home,
                    p.prob_away as Prob_Away,
                    CASE
                        WHEN r.ganador = 1 THEN r.home_team
                        ELSE r.away_team
                    END as Resultado_Real,
                    CASE
                        WHEN (r.ganador = 1 AND p.prediccion = r.home_team) OR
                             (r.ganador = 0 AND p.prediccion = r.away_team)
                        THEN '✅ Acertado'
                        ELSE '❌ Error'
                    END as Estado,
                    CASE
                        WHEN (r.ganador = 1 AND p.prediccion = r.home_team) OR
                             (r.ganador = 0 AND p.prediccion = r.away_team)
                        THEN 1
                        ELSE 0
                    END as Acierto_Num
                FROM historico_real r
                INNER JOIN predicciones_historico p
                    ON r.fecha = p.fecha
                    AND r.home_team = p.home_team
                    AND r.away_team = p.away_team
                ORDER BY r.fecha DESC
                """,
                conn,
            )

            if not df_dash.empty:
                # Calcular confianza numérica como el máximo de las dos probabilidades
                df_dash["Prob_Home"] = (
                    pd.to_numeric(df_dash["Prob_Home"], errors="coerce").fillna(0).apply(normalizar_probabilidad)
                )
                df_dash["Prob_Away"] = (
                    pd.to_numeric(df_dash["Prob_Away"], errors="coerce").fillna(0).apply(normalizar_probabilidad)
                )
                df_dash["Confianza"] = df_dash[["Prob_Home", "Prob_Away"]].max(axis=1)

                df_dash["Fecha_Original"] = pd.to_datetime(df_dash["Fecha"])
                df_dash["Fecha"] = df_dash["Fecha_Original"].dt.date

                st.markdown("### 📅 Filtro de Fechas")
                min_date = df_dash["Fecha"].min()
                max_date = df_dash["Fecha"].max()

                # Por defecto mostramos los últimos 14 días
                default_start = max_date - timedelta(days=14) if max_date - min_date > timedelta(days=14) else min_date

                col_filter1, col_filter2 = st.columns([1, 2])
                with col_filter1:
                    date_range = st.date_input(
                        "Selecciona el rango:", value=(default_start, max_date), min_value=min_date, max_value=max_date
                    )

                if len(date_range) == 2:
                    start_date, end_date = date_range
                    mask = (df_dash["Fecha"] >= start_date) & (df_dash["Fecha"] <= end_date)
                    df_filtered = df_dash.loc[mask].copy()

                    # Detectar cambio de filtro de fecha para reiniciar página
                    if "last_date_range" not in st.session_state or st.session_state.last_date_range != date_range:
                        st.session_state.last_date_range = date_range
                        st.session_state.resultados_page = 1

                    st.markdown("---")
                    st.markdown("### 🏟️ Resultados por Partido")

                    # Crear DataFrame formateado para mostrar al usuario
                    df_display = df_filtered.copy()
                    df_display["Juego"] = "🏟️ " + df_display["Away"] + " @ " + df_display["Home"]

                    # Confianza ya fue normalizada al cargar los datos
                    confianza_norm = df_display["Confianza"]
                    df_display["Predicción"] = (
                        "⚾ " + df_display["Prediccion"] + " | " + (confianza_norm * 100).round(1).astype(str) + "%"
                    )

                    df_final = df_display[["Fecha", "Juego", "Predicción", "Resultado_Real", "Estado"]].copy()
                    df_final.columns = ["Fecha", "Encuentro", "Predicción del Modelo", "Ganador Real", "Estado"]

                    # Paginación (18 resultados por lote)
                    items_per_page = 18
                    total_items = len(df_final)
                    import math
                    total_pages = max(1, math.ceil(total_items / items_per_page))

                    # Inicializar página en session state
                    if "resultados_page" not in st.session_state:
                        st.session_state.resultados_page = 1

                    # Asegurar que la página esté dentro de los límites válidos
                    if st.session_state.resultados_page > total_pages:
                        st.session_state.resultados_page = total_pages
                    if st.session_state.resultados_page < 1:
                        st.session_state.resultados_page = 1

                    # Obtener lote de datos actual
                    start_idx = (st.session_state.resultados_page - 1) * items_per_page
                    end_idx = min(start_idx + items_per_page, total_items)

                    df_page = df_final.iloc[start_idx:end_idx].copy()

                    # Renderizar tabla para la página actual
                    table_html = render_resultados_table_html(df_page)
                    st.markdown(table_html, unsafe_allow_html=True)

                    # Controles de paginación
                    st.markdown("<br>", unsafe_allow_html=True)
                    col_pag1, col_pag2, col_pag3 = st.columns([1, 2, 1])

                    # Obtener colores e información del tema para la paginación
                    theme_act = "Oscuro" if "theme_selector" in st.session_state and "🌙 Oscuro" in st.session_state.theme_selector else "Claro"
                    text_color_pag = "#90cce8" if theme_act == "Oscuro" else "#64748b"

                    with col_pag1:
                        if st.session_state.resultados_page > 1:
                            if st.button("⬅️ Anterior", key="btn_prev_page", use_container_width=True):
                                st.session_state.resultados_page -= 1
                                st.rerun()
                        else:
                            st.button("⬅️ Anterior", key="btn_prev_page_disabled", disabled=True, use_container_width=True)

                    with col_pag2:
                        st.markdown(
                            f"<div style='text-align: center; font-weight: bold; padding: 6px; color: {text_color_pag}; font-family: sans-serif; font-size: 0.85rem;'>"
                            f"Página {st.session_state.resultados_page} de {total_pages} (Mostrando {start_idx+1}-{end_idx} de {total_items} resultados)"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                    with col_pag3:
                        if st.session_state.resultados_page < total_pages:
                            if st.button("Siguiente ➡️", key="btn_next_page", use_container_width=True):
                                st.session_state.resultados_page += 1
                                st.rerun()
                        else:
                            st.button("Siguiente ➡️", key="btn_next_page_disabled", disabled=True, use_container_width=True)

                    st.markdown("---")
                    st.markdown("### 📈 Histórico de Tasa de Aciertos")

                    col_agrup, _ = st.columns([1, 2])
                    with col_agrup:
                        agrupacion = st.radio("Agrupar métricas por:", ["Día", "Semana", "Mes"], horizontal=True)

                    df_trend = df_filtered.copy()

                    if agrupacion == "Día":
                        df_grouped = (
                            df_trend.groupby("Fecha")
                            .agg(Total=("Acierto_Num", "count"), Aciertos=("Acierto_Num", "sum"))
                            .reset_index()
                        )
                        df_grouped["Tasa de Aciertos (%)"] = (df_grouped["Aciertos"] / df_grouped["Total"]) * 100
                        x_col = "Fecha"
                    elif agrupacion == "Semana":
                        df_trend["Semana"] = df_trend["Fecha_Original"].dt.isocalendar().week
                        df_grouped = (
                            df_trend.groupby("Semana")
                            .agg(Total=("Acierto_Num", "count"), Aciertos=("Acierto_Num", "sum"))
                            .reset_index()
                        )
                        df_grouped["Tasa de Aciertos (%)"] = (df_grouped["Aciertos"] / df_grouped["Total"]) * 100
                        df_grouped["Semana_Label"] = "Semana " + df_grouped["Semana"].astype(str)
                        x_col = "Semana_Label"
                    else:  # Mes
                        df_trend["Mes_Num"] = df_trend["Fecha_Original"].dt.month
                        df_grouped = (
                            df_trend.groupby("Mes_Num")
                            .agg(Total=("Acierto_Num", "count"), Aciertos=("Acierto_Num", "sum"))
                            .reset_index()
                        )
                        df_grouped["Tasa de Aciertos (%)"] = (df_grouped["Aciertos"] / df_grouped["Total"]) * 100
                        meses = {
                            1: "Enero",
                            2: "Febrero",
                            3: "Marzo",
                            4: "Abril",
                            5: "Mayo",
                            6: "Junio",
                            7: "Julio",
                            8: "Agosto",
                            9: "Septiembre",
                            10: "Octubre",
                            11: "Noviembre",
                            12: "Diciembre",
                        }
                        df_grouped["Mes_Label"] = df_grouped["Mes_Num"].map(meses)
                        x_col = "Mes_Label"

                    # Crear gráfico de línea
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=df_grouped[x_col],
                            y=df_grouped["Tasa de Aciertos (%)"],
                            mode="lines+markers",
                            name="Tasa de Aciertos",
                            line=dict(color="#3b82f6", width=4, shape="spline"),
                            marker=dict(size=12, color="#1e40af", line=dict(color="white", width=2)),
                            fill="tozeroy",
                            fillcolor="rgba(59, 130, 246, 0.1)",
                            hovertemplate="<b>%{x}</b><br>Tasa de Aciertos: %{y:.1f}%<br>Partidos: %{customdata[0]}<extra></extra>",
                            customdata=df_grouped[["Total"]],
                        )
                    )

                    # Línea de referencia (50% de aciertos)
                    fig.add_hline(y=50, line_dash="dash", line_color="red", line_width=1, annotation_text="50% (Azar)")

                    fig.update_layout(
                        title={
                            "text": f"Evolución de la Tasa de Aciertos (por {agrupacion.lower()})",
                            "font": {"size": 20, "weight": "bold"},
                            "x": 0.5,
                        },
                        xaxis_title=agrupacion,
                        yaxis_title="Tasa de Aciertos (%)",
                        yaxis=dict(
                            range=[
                                max(0, df_grouped["Tasa de Aciertos (%)"].min() - 10),
                                min(105, df_grouped["Tasa de Aciertos (%)"].max() + 10),
                            ]
                        ),
                        hovermode="x unified",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(gridcolor="rgba(0,0,0,0.05)", showgrid=True),
                        yaxis_gridcolor="rgba(0,0,0,0.05)",
                        height=450,
                        margin=dict(l=20, r=20, t=60, b=20),
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # --- NUEVAS MÉTRICAS ---
                    st.markdown("---")

                    col_m1, col_m2 = st.columns(2)

                    with col_m1:
                        st.markdown("### 🎯 Aciertos por Nivel de Confianza")

                        df_conf = df_filtered.copy()
                        df_conf["Conf_Pct"] = df_conf["Confianza"]

                        # Crear los buckets
                        def categorize_conf(val):
                            val_pct = val * 100 if val <= 1.0 else val
                            if val_pct < 55:
                                return "Baja (< 55%)"
                            elif val_pct < 65:
                                return "Media (55% - 65%)"
                            elif val_pct < 75:
                                return "Alta (65% - 75%)"
                            else:
                                return "Muy Alta (≥ 75%)"

                        df_conf["Nivel"] = df_conf["Conf_Pct"].apply(categorize_conf)

                        df_conf_grouped = (
                            df_conf.groupby("Nivel")
                            .agg(Total=("Acierto_Num", "count"), Aciertos=("Acierto_Num", "sum"))
                            .reset_index()
                        )

                        # Evitar division by zero
                        df_conf_grouped["Tasa (%)"] = df_conf_grouped.apply(
                            lambda row: (row["Aciertos"] / row["Total"] * 100) if row["Total"] > 0 else 0, axis=1
                        )

                        # Ordenar las categorías lógicamente
                        cat_order = {
                            "Baja (< 55%)": 0,
                            "Media (55% - 65%)": 1,
                            "Alta (65% - 75%)": 2,
                            "Muy Alta (≥ 75%)": 3
                        }
                        df_conf_grouped["Order"] = df_conf_grouped["Nivel"].map(cat_order)
                        df_conf_grouped = df_conf_grouped.sort_values("Order")

                        colors_map = {
                            "Baja (< 55%)": "#94a3b8",
                            "Media (55% - 65%)": "#fbbf24",
                            "Alta (65% - 75%)": "#3b82f6",
                            "Muy Alta (≥ 75%)": "#10b981"
                        }
                        bar_colors = [colors_map.get(n, "#94a3b8") for n in df_conf_grouped["Nivel"]]

                        fig_conf = go.Figure()
                        fig_conf.add_trace(
                            go.Bar(
                                x=df_conf_grouped["Nivel"],
                                y=df_conf_grouped["Tasa (%)"],
                                text=df_conf_grouped["Tasa (%)"].round(1).astype(str) + "%",
                                textposition="inside",
                                marker_color=bar_colors,
                                hovertemplate="<b>%{x}</b><br>Aciertos: %{customdata[0]} de %{customdata[1]}<extra></extra>",
                                customdata=df_conf_grouped[["Aciertos", "Total"]],
                            )
                        )

                        theme_act = "Oscuro" if "theme_selector" in st.session_state and "🌙 Oscuro" in st.session_state.theme_selector else "Claro"
                        grid_color = "rgba(255, 255, 255, 0.07)" if theme_act == "Oscuro" else "rgba(0, 0, 0, 0.07)"
                        text_color = "#90cce8" if theme_act == "Oscuro" else "#1e293b"

                        fig_conf.update_layout(
                            yaxis_title="Tasa de Aciertos (%)",
                            yaxis=dict(
                                range=[0, 115],
                                gridcolor=grid_color,
                                showgrid=True,
                                tickfont=dict(color=text_color, size=9)
                            ),
                            xaxis=dict(
                                tickfont=dict(color=text_color, size=9)
                            ),
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=10, r=10, t=10, b=10),
                            height=280,
                            font=dict(family="'Inter', sans-serif", color=text_color),
                        )
                        st.plotly_chart(fig_conf, use_container_width=True)

                    with col_m2:
                        st.markdown("### 🏟️ Precisión por Equipo")

                        # Analizar la tasa de acierto cada vez que juega un equipo
                        df_home = df_filtered[["Home", "Acierto_Num"]].rename(columns={"Home": "Equipo"})
                        df_away = df_filtered[["Away", "Acierto_Num"]].rename(columns={"Away": "Equipo"})
                        df_teams = pd.concat([df_home, df_away])

                        df_team_acc = (
                            df_teams.groupby("Equipo")
                            .agg(Total=("Acierto_Num", "count"), Aciertos=("Acierto_Num", "sum"))
                            .reset_index()
                        )

                        # Filtrar equipos con al menos 2 juegos en el rango para no sesgar si hay muchos juegos
                        min_games = 2 if len(df_filtered) > 10 else 1
                        df_team_acc = df_team_acc[df_team_acc["Total"] >= min_games]

                        df_team_acc["Tasa (%)"] = (df_team_acc["Aciertos"] / df_team_acc["Total"]) * 100

                        # Seleccionar top 5 y peores 5
                        df_team_acc = df_team_acc.sort_values("Tasa (%)", ascending=False)

                        if len(df_team_acc) > 10:
                            top_teams = df_team_acc.head(5)
                            bottom_teams = df_team_acc.tail(5)
                            df_show = pd.concat([top_teams, bottom_teams]).sort_values("Tasa (%)", ascending=True)
                        else:
                            df_show = df_team_acc.sort_values("Tasa (%)", ascending=True)

                        colors = [
                            "#ef4444" if x < 50 else ("#10b981" if x >= 60 else "#3b82f6") for x in df_show["Tasa (%)"]
                        ]

                        fig_team = go.Figure()
                        fig_team.add_trace(
                            go.Bar(
                                y=df_show["Equipo"],
                                x=df_show["Tasa (%)"],
                                orientation="h",
                                text=df_show["Tasa (%)"].round(1).astype(str) + "% (" + df_show["Total"].astype(str) + "J)",
                                textposition="inside",
                                marker_color=colors,
                                hovertemplate="<b>%{y}</b><br>Tasa: %{x:.1f}%<br>Partidos: %{customdata[0]}<extra></extra>",
                                customdata=df_show[["Total"]],
                            )
                        )

                        fig_team.update_layout(
                            xaxis_title="Tasa de Aciertos (%)",
                            xaxis=dict(
                                range=[0, 115],
                                gridcolor=grid_color,
                                showgrid=True,
                                tickfont=dict(color=text_color, size=9)
                            ),
                            yaxis=dict(
                                tickfont=dict(color=text_color, size=9)
                            ),
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=10, r=10, t=10, b=10),
                            height=280,
                            font=dict(family="'Inter', sans-serif", color=text_color),
                        )
                        st.plotly_chart(fig_team, use_container_width=True)
                else:
                    st.info("ℹ️ Por favor, selecciona un rango de fechas válido (fecha inicio y fecha fin).")

            else:
                st.info("ℹ️ No hay datos históricos disponibles en la base de datos para mostrar los dashboards.")
    except Exception as e:
        st.error(f"❌ Error al cargar los datos para los dashboards: {str(e)}")

    # ============================================================================
    # PÁGINA: ACERCA DEL MODELO
    # ============================================================================

elif pagina == "🧠 Acerca del Modelo":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(
            "https://www.mlbstatic.com/team-logos/league-on-dark/1.svg",
            use_container_width=True,
        )

    st.markdown('<div class="main-title">MLB Predictor Pro V4.0</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Sistema Híbrido de Predicción con Machine Learning (Optuna & MLB API)</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Estado del Sistema")
    render_system_status_panel(api_ok, api_data)

    st.markdown("### Estadísticas Totales del Motor")
    render_motor_lifetime_panel()

    st.markdown(
        """
    <div class="stats-card">
        <h3> ¿Qué es MLB Predictor Pro?</h3>
        <p style="font-size: 1.1rem; line-height: 1.8;">
            Sistema avanzado de predicción de partidos MLB que utiliza <strong>XGBoost</strong>
            y <strong>web scraping en tiempo real</strong> para analizar más de 26 características
            estadísticas por partido, incluyendo matchups específicos de lanzadores vs bateadores.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        <div class="stats-card">
            <h4> Características V4.0</h4>
            <ul style="font-size: 1.05rem; line-height: 1.8;">
                <li><strong>Modelo Híbrido</strong>: Temporal + Ingestión en vivo</li>
                <li><strong>Super Features</strong>: Matchups directos y coeficientes dinámicos</li>
                <li><strong>Optimización Bayesiana</strong>: Optuna para ajuste de hiperparámetros</li>
                <li><strong>Reentrenamiento Incremental</strong>: Automatizado e inteligente</li>
                <li><strong>API FastAPI</strong>: Endpoints RESTful de alto rendimiento</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="stats-card">
            <h4> Fuentes de Datos</h4>
            <ul style="font-size: 1.05rem; line-height: 1.8;">
                <li><strong>MLB Stats API Oficial</strong>: Ingesta digital ultra rápida de cartelera y rosters</li>
                <li><strong>Histórico 2020-2026</strong>: Más de 10,000 partidos</li>
                <li><strong>Actualización Automática</strong>: GitHub Actions</li>
                <li><strong>Base de Datos SQLite</strong>: Almacenamiento local estructurado</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("### Super Features Explicadas")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        <div class="stats-card" style="border-left-color: #667eea;">
            <h4> Neutralización</h4>
            <p style="font-size: 1rem; line-height: 1.6;">
                Mide cómo el <strong>WHIP del lanzador</strong> neutraliza el
                <strong>OPS del lineup rival</strong>, limitando su producción ofensiva.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="stats-card" style="border-left-color: #f093fb;">
            <h4> Resistencia</h4>
            <p style="font-size: 1rem; line-height: 1.6;">
                Evalúa la capacidad del lanzador de mantener su <strong>ERA bajo presión</strong>
                contra bateadores de poder (alto OPS).
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="stats-card" style="border-left-color: #a8edea;">
            <h4> Muro Bullpen</h4>
            <p style="font-size: 1rem; line-height: 1.6;">
                Analiza la efectividad del <strong>bullpen</strong> contra los
                <strong>mejores bateadores rivales</strong> en innings críticos.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("### Stack Tecnológico")

    tech_badges = [
        "Python 3.10+",
        "XGBoost",
        "Scikit-learn",
        "FastAPI",
        "Streamlit",
        "Pandas",
        "NumPy",
        "Plotly",
        "SQLite",
        "GitHub Actions",
        "Uvicorn",
    ]

    badges_html = " ".join(
        [
            f'<span class="badge" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin: 0.25rem;">{tech}</span>'
            for tech in tech_badges
        ]
    )
    st.markdown(
        f'<div style="text-align: center; margin: 2rem 0;">{badges_html}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Equipos de la Liga")

    cols = st.columns(6)
    teams_sorted = sorted(EQUIPOS_MLB.items(), key=lambda x: x[1]["nombre"])

    for i, (code, info) in enumerate(teams_sorted):
        with cols[i % 6]:
            st.image(info["logo"], width=60)
            st.caption(f"**{code}**")

    st.markdown("---")
    st.info("""
    ⚠️ Este sistema es una herramienta estadística para análisis y entretenimiento.
    El béisbol es inherentemente impredecible. No utilices estas predicciones para apuestas deportivas.
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #64748b; padding: 2rem 0;">
    <p style="margin: 0; font-size: 1rem;">
        <strong>MLB Predictor Pro V4.0</strong> | 2026
    </p>
    <p style="margin: 0.5rem 0; font-size: 0.9rem;">
        Datos: API Oficial de la MLB (statsapi.mlb.com) | Powered by XGBoost, Optuna & FastAPI
    </p>
    <p style="margin: 0.5rem 0; font-size: 0.85rem;">
        🔄 Actualización automática vía GitHub Actions
    </p>
</div>
""",
    unsafe_allow_html=True,
)
