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
    from mlb_config import DB_PATH, TEAM_CODE_TO_NAME
except Exception as e:
    TEAM_CODE_TO_NAME = {}
    DB_PATH = "./data/mlb_reentrenamiento.db"
    st.warning(f"Config fallback due to error: {e}")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

st.set_page_config(
    page_title="MLB Predictor Pro V3.5",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API Configuration
API_URL = os.getenv("API_URL") or st.secrets.get("API_URL", "http://localhost:8000")

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

st.markdown(
    """
<style>
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.75rem;
        border-radius: 1rem;
        color: white;
        margin: 1rem 0;
        min-height: 200px;
        box-shadow: 0 8px 25px rgba(118, 75, 162, 0.3);
        border: 2px solid rgba(255,255,255,0.2);
    }

    .super-feature h4 {
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .super-feature-advantage {
        font-size: 1.25rem;
        font-weight: 700;
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        display: inline-block;
        margin: 0.5rem 0;
    }

    .game-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-top: 4px solid var(--primary-blue);
        transition: transform 0.2s;
    }

    .game-card:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
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
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================


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
                error_output = (
                    scrape_result.stderr or scrape_result.stdout or ""
                ).strip()
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

            error_output = (
                predict_result.stderr or predict_result.stdout or ""
            ).strip()
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
    entorno = (
        "Producción" if (os.getenv("API_URL") or st.secrets.get("API_URL")) else "Local"
    )

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
            total_predicciones = conn.execute(
                "SELECT COUNT(*) FROM predicciones_historico"
            ).fetchone()[0]

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
        tasa_exito = (
            round((aciertos / total_validados) * 100, 2) if total_validados else 0.0
        )

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
            st.markdown(
                "**Cobertura de validación** (partidos con resultado real vs. totales)"
            )
            cobertura = validados / total if total > 0 else 0
            st.progress(
                cobertura, text=f"{validados:,} de {total:,} ({cobertura * 100:.1f}%)"
            )
        with col_g2:
            st.markdown("**Tasa de acierto** (predicciones correctas sobre validadas)")
            tasa_frac = aciertos / validados if validados > 0 else 0
            color_text = (
                "verde"
                if tasa_frac >= 0.55
                else ("amarillo" if tasa_frac >= 0.50 else "rojo")
            )
            _ = color_text  # solo para referencia semántica
            st.progress(
                tasa_frac, text=f"{aciertos:,} correctas de {validados:,} ({tasa:.1f}%)"
            )

    if stats["error"]:
        st.warning(
            f"No se pudieron cargar estadísticas históricas del motor: {stats['error']}"
        )


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

    fig = go.Figure()

    colors = [
        "#3b82f6" if prob_home > prob_away else "#94a3b8",
        "#ef4444" if prob_away > prob_home else "#94a3b8",
    ]

    fig.add_trace(
        go.Bar(
            x=[prob_home * 100, prob_away * 100],
            y=[f"{home_team}", f"{away_team}"],
            orientation="h",
            marker={"color": colors, "line": {"color": "white", "width": 3}},
            text=[f"{prob_home * 100:.1f}%", f"{prob_away * 100:.1f}%"],
            textposition="auto",
            textfont={"size": 18, "color": "white", "family": "Arial Black"},
            hovertemplate="<b>%{y}</b><br>Probabilidad: %{x:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": "Probabilidades de Victoria",
            "font": {"size": 24, "weight": "bold"},
        },
        xaxis_title="Probabilidad (%)",
        height=350,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"size": 14, "family": "Arial"},
        xaxis={"range": [0, 100], "gridcolor": "rgba(0,0,0,0.1)"},
        yaxis={"gridcolor": "rgba(0,0,0,0)"},
    )

    return fig


def crear_gauge_confianza(confianza):
    """Crea gauge de nivel de confianza - CORREGIDO"""
    # Normalizar confianza
    confianza = normalizar_probabilidad(confianza)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confianza * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": "Nivel de Confianza",
                "font": {"size": 24, "weight": "bold"},
            },
            number={"suffix": "%", "font": {"size": 48, "weight": "bold"}},
            gauge={
                "axis": {"range": [None, 100], "tickwidth": 2, "tickcolor": "#1e40af"},
                "bar": {"color": "#3b82f6", "thickness": 0.8},
                "bgcolor": "white",
                "borderwidth": 3,
                "bordercolor": "#cbd5e1",
                "steps": [
                    {"range": [0, 55], "color": "#fee2e2"},
                    {"range": [55, 60], "color": "#fed7aa"},
                    {"range": [60, 70], "color": "#d9f99d"},
                    {"range": [70, 100], "color": "#bbf7d0"},
                ],
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
        font={"color": "#334155", "family": "Arial"},
    )

    return fig


def obtener_prediccion_detallada_partido(
    home_team, away_team, home_pitcher, away_pitcher, year, fecha=None
):
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
                "fecha": fecha
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


def renderizar_analisis_detallado_partido(
    resultado_detallado, home_team, away_team, home_pitcher, away_pitcher
):
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
    year_solicitado = resultado_detallado.get(
        "year_solicitado", resultado_detallado.get("year_usado_home", 2026)
    )
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
            detalle_home = (
                razon_home
                if razon_home
                else f"No había muestra suficiente/estable en {year_solicitado}"
            )
            st.info(
                f"📌 **{home_team} - {home_pitcher}**: Se usaron estadísticas de la temporada "
                f"**{year_home}** para mejorar la estabilidad del análisis. "
                f"Motivo: {detalle_home}."
                f" IP consideradas: **{ip_home}**."
            )

        # Mostrar info del lanzador visitante si tiene fallback
        if hubo_fallback_away:
            ip_away = resultado_detallado.get("ip_away", 0)
            detalle_away = (
                razon_away
                if razon_away
                else f"No había muestra suficiente/estable en {year_solicitado}"
            )
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
                <p style="font-size: 1.1rem; line-height: 1.6; margin-top: 1rem;">
                    <strong>{pitcher_n}</strong> tiene un WHIP que neutraliza efectivamente
                    el OPS del lineup de <strong>{rival_n}</strong>, limitando su producción ofensiva.
                </p>
                <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.9;">
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
                <p style="font-size: 1.1rem; line-height: 1.6; margin-top: 1rem;">
                    <strong>{pitcher_r}</strong> demuestra mayor resistencia ante bateadores
                    de poder, manteniendo su ERA bajo presión ofensiva.
                </p>
                <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.9;">
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
                <p style="font-size: 1.1rem; line-height: 1.6; margin-top: 1rem;">
                    El bullpen de <strong>{ventaja_m}</strong> es más efectivo
                    contra los mejores bateadores rivales en innings críticos.
                </p>
                <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.9;">
                    Impacto: <strong>{abs(muro):.4f}</strong>
                </div>
            </div>
            """,
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

        pcol1, pcol2 = st.columns(2)

        with pcol1:
            home_pitcher_stats = stats_det.get("home_pitcher")
            if home_pitcher_stats and isinstance(home_pitcher_stats, dict):
                nombre_real = home_pitcher_stats.get("nombre", home_pitcher)
                st.markdown(
                    get_team_logo_html(home_team, 40) + f" {home_team} - {nombre_real}",
                    unsafe_allow_html=True,
                )
                s1, s2, s3 = st.columns(3)
                with s1:
                    st.metric("ERA", f"{home_pitcher_stats.get('ERA', 0):.2f}")
                with s2:
                    st.metric("WHIP", f"{home_pitcher_stats.get('WHIP', 0):.3f}")
                with s3:
                    st.metric("SO9", f"{home_pitcher_stats.get('SO9', 0):.2f}")
            else:
                st.warning(f"⚠️ No se encontraron estadísticas para {home_pitcher}")

        with pcol2:
            away_pitcher_stats = stats_det.get("away_pitcher")
            if away_pitcher_stats and isinstance(away_pitcher_stats, dict):
                nombre_real = away_pitcher_stats.get("nombre", away_pitcher)
                st.markdown(
                    get_team_logo_html(away_team, 40) + f" {away_team} - {nombre_real}",
                    unsafe_allow_html=True,
                )
                s1, s2, s3 = st.columns(3)
                with s1:
                    st.metric("ERA", f"{away_pitcher_stats.get('ERA', 0):.2f}")
                with s2:
                    st.metric("WHIP", f"{away_pitcher_stats.get('WHIP', 0):.3f}")
                with s3:
                    st.metric("SO9", f"{away_pitcher_stats.get('SO9', 0):.2f}")
            else:
                st.warning(f"⚠️ No se encontraron estadísticas para {away_pitcher}")

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
                            with st.expander(
                                f"#{i} - {nombre_batter}", expanded=(i == 1)
                            ):
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
                            with st.expander(
                                f"#{i} - {nombre_batter}", expanded=(i == 1)
                            ):
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
        <h2 style="margin: 0;">MLB Predictor</h2>
        <span class="badge badge-pro">PRO V3.5</span>
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
            "⚾ Predicción Manual",
            "🧠 Acerca del Modelo",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")

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

    team_options = [
        f"{code} - {EQUIPOS_MLB[code]['nombre']}" for code in sorted(EQUIPOS_MLB.keys())
    ]

    with col1:
        st.markdown("#### Equipo Local")
        home_sel = st.selectbox(
            "Selecciona equipo local", team_options, key="home_sel_manual"
        )
        home_team = home_sel.split(" - ")[0]
        st.markdown(
            f'<div style="text-align: center; padding: 10px;">{get_team_logo_html(home_team, 100)}</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("#### Equipo Visitante")
        away_sel = st.selectbox(
            "Selecciona equipo visitante", team_options, key="away_sel_manual"
        )
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
            submit = st.form_submit_button(
                "Realizar Predicción", use_container_width=True, type="primary"
            )

    if submit:
        if not home_pitcher or not away_pitcher:
            st.error("Por favor ingresa los nombres de ambos lanzadores")
        elif home_team == away_team:
            st.error("Los equipos deben ser diferentes")
        else:
            with st.spinner(
                f"Analizando {home_team} vs {away_team}... Esto puede tardar varios segundos"
            ):
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
                                crear_grafico_probabilidades(
                                    prob_home, prob_away, home_team, away_team
                                ),
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
                            st.info(
                                "ℹ️ Las Super Features no están disponibles en esta respuesta de la API"
                            )
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
                                    <p style="font-size: 1.1rem; line-height: 1.6; margin-top: 1rem;">
                                        <strong>{pitcher_n}</strong> tiene un WHIP que neutraliza efectivamente
                                        el OPS del lineup de <strong>{rival_n}</strong>, limitando su producción ofensiva.
                                    </p>
                                    <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.9;">
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
                                    <p style="font-size: 1.1rem; line-height: 1.6; margin-top: 1rem;">
                                        <strong>{pitcher_r}</strong> demuestra mayor resistencia ante bateadores
                                        de poder, manteniendo su ERA bajo presión ofensiva.
                                    </p>
                                    <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.9;">
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
                                    <p style="font-size: 1.1rem; line-height: 1.6; margin-top: 1rem;">
                                        El bullpen de <strong>{ventaja_m}</strong> es más efectivo
                                        contra los mejores bateadores rivales en innings críticos.
                                    </p>
                                    <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.9;">
                                        Impacto: <strong>{abs(muro):.4f}</strong>
                                    </div>
                                </div>
                                """,
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

                            col1, col2 = st.columns(2)

                            with col1:
                                home_pitcher_stats = stats_det.get("home_pitcher")
                                if home_pitcher_stats and isinstance(
                                    home_pitcher_stats, dict
                                ):
                                    pitcher_nombre_real = home_pitcher_stats.get(
                                        "nombre", home_pitcher
                                    )

                                    st.markdown(
                                        get_team_logo_html(home_team, 40)
                                        + f" {home_team} - {pitcher_nombre_real}",
                                        unsafe_allow_html=True,
                                    )

                                    subcol1, subcol2, subcol3 = st.columns(3)
                                    with subcol1:
                                        st.metric(
                                            "ERA",
                                            f"{home_pitcher_stats.get('ERA', 0):.2f}",
                                        )
                                    with subcol2:
                                        st.metric(
                                            "WHIP",
                                            f"{home_pitcher_stats.get('WHIP', 0):.3f}",
                                        )
                                    with subcol3:
                                        st.metric(
                                            "SO9",
                                            f"{home_pitcher_stats.get('SO9', 0):.2f}",
                                        )
                                else:
                                    st.warning(
                                        f"⚠️ No se encontraron estadísticas para {home_pitcher}"
                                    )

                            with col2:
                                away_pitcher_stats = stats_det.get("away_pitcher")
                                if away_pitcher_stats and isinstance(
                                    away_pitcher_stats, dict
                                ):
                                    pitcher_nombre_real = away_pitcher_stats.get(
                                        "nombre", away_pitcher
                                    )

                                    st.markdown(
                                        get_team_logo_html(away_team, 40)
                                        + f" {away_team} - {pitcher_nombre_real}",
                                        unsafe_allow_html=True,
                                    )

                                    subcol1, subcol2, subcol3 = st.columns(3)
                                    with subcol1:
                                        st.metric(
                                            "ERA",
                                            f"{away_pitcher_stats.get('ERA', 0):.2f}",
                                        )
                                    with subcol2:
                                        st.metric(
                                            "WHIP",
                                            f"{away_pitcher_stats.get('WHIP', 0):.3f}",
                                        )
                                    with subcol3:
                                        st.metric(
                                            "SO9",
                                            f"{away_pitcher_stats.get('SO9', 0):.2f}",
                                        )
                                else:
                                    st.warning(
                                        f" No se encontraron estadísticas para {away_pitcher}"
                                    )

                            st.markdown("---")
                            st.markdown("### Top 3 Bateadores")

                            col1, col2 = st.columns(2)

                            with col1:
                                home_batters = stats_det.get("home_batters", [])
                                if (
                                    home_batters
                                    and isinstance(home_batters, list)
                                    and len(home_batters) > 0
                                ):
                                    st.markdown(
                                        get_team_logo_html(home_team, 40)
                                        + f"- {home_team}",
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
                                    st.info(
                                        f"ℹ️ No hay datos de bateadores disponibles para {home_team}"
                                    )

                            with col2:
                                away_batters = stats_det.get("away_batters", [])
                                if (
                                    away_batters
                                    and isinstance(away_batters, list)
                                    and len(away_batters) > 0
                                ):
                                    st.markdown(
                                        get_team_logo_html(away_team, 40)
                                        + f"- {away_team}",
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
                                    st.info(
                                        f"ℹ️ No hay datos de bateadores disponibles para {away_team}"
                                    )

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
                        st.error(
                            f"❌ Error: {error.get('detail', 'Error desconocido')}"
                        )
                        st.info(
                            "💡 Verifica que los nombres de los lanzadores o el año de busqueda sean correctos."
                        )

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
    st.markdown(
        '<div class="subtitle">Predicciones automáticas basadas en lineups scrapeados</div>',
        unsafe_allow_html=True,
    )

    if not api_ok:
        st.warning(
            "La API no respondio al health check a tiempo. Intentando cargar datos igualmente..."
        )

    try:
        response_partidos = requests.get(f"{API_URL}/games/today", timeout=30)
        response_predicciones = requests.get(f"{API_URL}/predictions/today", timeout=30)

        partidos = (
            response_partidos.json() if response_partidos.status_code == 200 else []
        )
        predicciones = (
            response_predicciones.json()
            if response_predicciones.status_code == 200
            else []
        )

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

        if fechas_api:
            if len(fechas_api) == 1:
                st.info(
                    f"Fecha cargada por la API: {fechas_api[0]} | Fecha actual: {fecha_hoy}"
                )
            else:
                st.info(
                    f"Fechas cargadas por la API: {', '.join(fechas_api)} | Fecha actual: {fecha_hoy}"
                )

        partidos_hoy = [p for p in partidos if p.get("fecha") == fecha_hoy]
        fecha_mostrada = fecha_hoy

        if partidos_hoy:
            partidos = partidos_hoy
        elif fechas_api:
            fecha_mostrada = max(fechas_api)
            partidos = [p for p in partidos if p.get("fecha") == fecha_mostrada]
            st.warning(
                f"No hay cartelera para hoy en ET ({fecha_hoy}). "
                f"Mostrando la última fecha disponible: {fecha_mostrada}."
            )
        else:
            partidos = []

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

            if st.button(
                "Buscar Partidos Manualmente", type="primary", use_container_width=True
            ):
                exito, mensaje = ejecutar_scraper_manual()
                if exito:
                    st.success(mensaje)
                    st.rerun()
                else:
                    st.warning(mensaje)
        else:
            etiqueta_fecha = (
                f"hoy ({fecha_mostrada}, ET)"
                if fecha_mostrada == fecha_hoy
                else f"{fecha_mostrada} (ET)"
            )
            st.success(f"Se encontraron {len(partidos)} partidos para {etiqueta_fecha}")

            if "detalles_partidos_hoy" not in st.session_state:
                st.session_state.detalles_partidos_hoy = {}

            for partido in partidos:
                with st.container():
                    home_logo = get_team_logo_html(partido["home_team"], 50)
                    away_logo = get_team_logo_html(partido["away_team"], 50)
                    home_pitcher_txt = (partido.get("home_pitcher") or "").strip()
                    away_pitcher_txt = (partido.get("away_pitcher") or "").strip()
                    if not home_pitcher_txt or not away_pitcher_txt:
                        pitcher_line = "Lanzadores por confirmar"
                    else:
                        pitcher_line = f"{away_pitcher_txt} vs {home_pitcher_txt}"

                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(
                            f"""
                        <div class="game-card">
                            <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                                {away_logo}{partido["away_team"]} @ {home_logo}{partido["home_team"]}
                            </div>
                            <div style="color: #64748b; margin: 0.5rem 0;">
                                {pitcher_line}
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                        if "prediccion" in partido:
                            prob = (
                                partido.get("prob_home", 0)
                                if partido["prediccion"] == partido["home_team"]
                                else partido.get("prob_away", 0)
                            )
                            prob_normalizada = normalizar_probabilidad(prob)

                    with col2:
                        if "prediccion" in partido:
                            prob = (
                                partido.get("prob_home", 0)
                                if partido["prediccion"] == partido["home_team"]
                                else partido.get("prob_away", 0)
                            )
                            prob_normalizada = normalizar_probabilidad(prob)

                            st.metric(
                                "Confianza",
                                f"{prob_normalizada * 100:.1f}%",
                                delta=partido.get("confianza", "N/A"),
                            )
                        else:
                            st.warning("⏳ Pendiente")

                    if "prediccion" in partido:
                        pred_team = get_team_display_name(partido["prediccion"])
                        prob = (
                            partido.get("prob_home", 0)
                            if partido["prediccion"] == partido["home_team"]
                            else partido.get("prob_away", 0)
                        )
                        prob_normalizada = normalizar_probabilidad(prob)

                        game_id = partido.get(
                            "game_id",
                            f"{partido.get('fecha', '')}_{partido['away_team']}_{partido['home_team']}",
                        )

                        with st.expander(
                            f"Predicho: {pred_team} ({prob_normalizada * 100:.1f}%) - Ver análisis detallado",
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
                                if (
                                    game_id
                                    not in st.session_state.detalles_partidos_hoy
                                ):
                                    if st.button(
                                        "Cargar análisis detallado",
                                        key=f"load_detail_{game_id}",
                                        use_container_width=True,
                                    ):
                                        with st.spinner(
                                            f"Generando análisis detallado para {partido['away_team']} @ {partido['home_team']}..."
                                        ):
                                            ok_detalle, payload = (
                                                obtener_prediccion_detallada_partido(
                                                    partido["home_team"],
                                                    partido["away_team"],
                                                    home_pitcher,
                                                    away_pitcher,
                                                    year_detalle,
                                                    partido.get("fecha", fecha_hoy)
                                                )
                                            )
                                        if ok_detalle:
                                            st.session_state.detalles_partidos_hoy[
                                                game_id
                                            ] = payload
                                            st.rerun()
                                        else:
                                            st.error(
                                                f"No se pudo cargar el análisis detallado: {payload}"
                                            )
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

        st.markdown(f"### 📈 Estadísticas — {_periodo}")
        _c1, _c2, _c3 = st.columns(3)
        _c1.metric("Total Partidos", _total)
        _c2.metric("Aciertos", _aciertos)
        _c3.metric("Accuracy General", f"{_acc:.1f}%")

        if _por_conf:
            st.markdown("#### Desglose por Nivel de Confianza")
            _iconos = {"ALTA": "🟡", "MODERADA": "🟢", "BAJA": "🔵", "MUY ALTA": "🔴"}
            _conf_cols = st.columns(len(_por_conf))
            for _i, (_nivel, _datos) in enumerate(_por_conf.items()):
                with _conf_cols[_i]:
                    _icono = _iconos.get(_nivel, "⚪")
                    st.metric(
                        label=f"{_icono} {_nivel}",
                        value=f"{_datos.get('accuracy', 0):.1f}%",
                        delta=f"{_datos.get('aciertos', 0)}/{_datos.get('total', 0)} aciertos",
                        delta_color="off",
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
                st.info("📋 Predicciones del día — Los resultados reales aún no están disponibles.")
                st.markdown("### Predicciones")
                col1 = st.columns(1)[0]
                with col1:
                    st.metric("Total Partidos", stats["total"])
            else:
                st.markdown("### Rendimiento del Día")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Partidos", stats["total"])
                with col2:
                    st.metric("Aciertos", stats["aciertos"])
                with col3:
                    st.metric("Accuracy", f"{stats['accuracy']:.1f}%")
                with col4:
                    errores = stats["total"] - stats["aciertos"]
                    st.metric("Errores", errores)

            st.markdown("### Resultados Detallados" if not solo_predicciones else "### Partidos")

            for partido in partidos:
                acierto = partido.get("acierto", 0)
                _ht = partido["home_team"]
                _at = partido["away_team"]
                _hn = get_team_display_name(_ht)
                _an = get_team_display_name(_at)
                _h_logo = get_team_logo_html(_ht, 28)
                _a_logo = get_team_logo_html(_at, 28)
                _score_a = partido["score_away"]
                _score_h = partido["score_home"]
                _badge = "✅" if acierto else "❌"

                _titulo_exp = (
                    f"{_an} @ {_hn}"
                    if solo_predicciones or _score_a is None
                    else f"{_badge}  {_an} @ {_hn}  —  {_score_a}-{_score_h}"
                )
                with st.expander(_titulo_exp, expanded=False):
                    prob_h = partido.get("prob_home", 0)
                    prob_a = partido.get("prob_away", 0)
                    prob_h_norm = normalizar_probabilidad(prob_h)
                    prob_a_norm = normalizar_probabilidad(prob_a)
                    prediccion_code = partido.get("prediccion", "")
                    prediccion_nombre = (
                        get_team_display_name(prediccion_code)
                        if prediccion_code
                        else "N/A"
                    )
                    pred_logo = (
                        get_team_logo_html(prediccion_code, 22)
                        if prediccion_code
                        else ""
                    )

                    ganador_real_code = _ht if partido["ganador_real"] == 1 else _at
                    ganador_real_nombre = get_team_display_name(ganador_real_code)
                    ganador_real_logo = get_team_logo_html(ganador_real_code, 22)

                    confianza = partido.get("confianza", "N/A")
                    color_conf = {
                        "MUY ALTA": "#ef4444",
                        "ALTA": "#f59e0b",
                        "MODERADA": "#22c55e",
                        "BAJA": "#64748b",
                    }.get(confianza, "#64748b")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown(
                            f"""
<div style="background:rgba(59,130,246,0.06);border-radius:10px;padding:14px 16px;">
  <div style="font-size:0.75rem;font-weight:700;color:#3b82f6;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;">🤖 Predicción</div>
  <div style="display:flex;align-items:center;gap:6px;font-size:1rem;font-weight:600;margin-bottom:10px;">
    {pred_logo}{prediccion_nombre}
  </div>
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
    <span style="display:flex;align-items:center;gap:4px;font-size:0.85rem;">{get_team_logo_html(_ht, 16)}{_hn}</span>
    <span style="font-weight:700;font-size:0.95rem;">{prob_h_norm * 100:.1f}%</span>
  </div>
  <div style="background:#e2e8f0;border-radius:4px;height:6px;margin-bottom:8px;">
    <div style="background:#3b82f6;width:{prob_h_norm * 100:.1f}%;height:6px;border-radius:4px;"></div>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
    <span style="display:flex;align-items:center;gap:4px;font-size:0.85rem;">{get_team_logo_html(_at, 16)}{_an}</span>
    <span style="font-weight:700;font-size:0.95rem;">{prob_a_norm * 100:.1f}%</span>
  </div>
  <div style="background:#e2e8f0;border-radius:4px;height:6px;">
    <div style="background:#8b5cf6;width:{prob_a_norm * 100:.1f}%;height:6px;border-radius:4px;"></div>
  </div>
</div>""",
                            unsafe_allow_html=True,
                        )

                    with col2:
                        st.markdown(
                            f"""
<div style="background:rgba(16,185,129,0.06);border-radius:10px;padding:14px 16px;">
  <div style="font-size:0.75rem;font-weight:700;color:#10b981;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;">⚾ Resultado Real</div>
  <div style="display:flex;align-items:center;gap:6px;font-size:1rem;font-weight:600;margin-bottom:12px;">
    {ganador_real_logo}{ganador_real_nombre}
  </div>
  <div style="display:flex;align-items:center;justify-content:center;gap:16px;margin-top:8px;">
    <div style="text-align:center;">
      <div style="font-size:0.75rem;color:#64748b;">{_an}</div>
      <div style="font-size:2rem;font-weight:800;line-height:1;">{_score_a}</div>
    </div>
    <div style="font-size:1rem;color:#64748b;font-weight:600;">—</div>
    <div style="text-align:center;">
      <div style="font-size:0.75rem;color:#64748b;">{_hn}</div>
      <div style="font-size:2rem;font-weight:800;line-height:1;">{_score_h}</div>
    </div>
  </div>
</div>""",
                            unsafe_allow_html=True,
                        )

                    with col3:
                        res_bg = (
                            "rgba(34,197,94,0.08)"
                            if acierto
                            else "rgba(239,68,68,0.08)"
                        )
                        res_color = "#16a34a" if acierto else "#dc2626"
                        res_text = "ACIERTO ✅" if acierto else "ERROR ❌"
                        st.markdown(
                            f"""
<div style="background:{res_bg};border-radius:10px;padding:14px 16px;text-align:center;">
  <div style="font-size:0.75rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;">✓ Verificación</div>
  <div style="font-size:1.3rem;font-weight:800;color:{res_color};margin-bottom:14px;">{res_text}</div>
  <div style="background:rgba(0,0,0,0.05);border-radius:6px;padding:6px 10px;display:inline-block;">
    <span style="font-size:0.75rem;color:#64748b;">Confianza: </span>
    <span style="font-size:0.85rem;font-weight:700;color:{color_conf};">{confianza}</span>
  </div>
</div>""",
                            unsafe_allow_html=True,
                        )
        else:
            st.info(f" No hay datos disponibles para {fecha_str}")

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

    st.markdown(
        '<div class="main-title">MLB Predictor Pro V3.5</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">Sistema Híbrido de Predicción con Machine Learning</div>',
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
            <h4> Características V3.5</h4>
            <ul style="font-size: 1.05rem; line-height: 1.8;">
                <li><strong>Modelo Híbrido</strong>: Temporal + Scraping en vivo</li>
                <li><strong>Super Features</strong>: Análisis de matchups directos</li>
                <li><strong>XGBoost Optimizado</strong>: GridSearchCV para hiperparámetros</li>
                <li><strong>Reentrenamiento Incremental</strong>: Se actualiza cada 150 juegos</li>
                <li><strong>API FastAPI</strong>: Endpoints RESTful para integración</li>
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
                <li><strong>Baseball-Reference</strong>: Stats de jugadores</li>
                <li><strong>Histórico 2020-2026</strong>: Más de 10,000 partidos</li>
                <li><strong>Scraping Diario</strong>: Lineups y lanzadores</li>
                <li><strong>Actualización Automática</strong>: GitHub Actions</li>
                <li><strong>Base de Datos SQLite</strong>: Almacenamiento local</li>
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
        "BeautifulSoup",
        "Cloudscraper",
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
        <strong>MLB Predictor Pro V3.5</strong> | 2026
    </p>
    <p style="margin: 0.5rem 0; font-size: 0.9rem;">
        Datos: Baseball-Reference | Powered by XGBoost & FastAPI
    </p>
    <p style="margin: 0.5rem 0; font-size: 0.85rem;">
        🔄 Actualización automática vía GitHub Actions
    </p>
</div>
""",
    unsafe_allow_html=True,
)
