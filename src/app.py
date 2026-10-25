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
    import mlb_live
    from mlb_config import DB_PATH, TEAM_CODE_TO_NAME, get_team_code
except Exception as e:
    TEAM_CODE_TO_NAME = {}
    DB_PATH = "./data/mlb_reentrenamiento.db"
    def get_team_code(name):
        return name
    st.warning(f"Config fallback due to error: {e}")
from vistas_sistema import render_salud_modelo, render_simulacion_playoffs

try:  # la base de datos vive en un dataset de Hugging Face (Streamlit Cloud no la trae en el repo)
    from mlb_datos_remotos import asegurar_db

    asegurar_db()
except Exception as _e:  # noqa: BLE001
    print(f"Base de datos remota no disponible: {_e}")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

st.set_page_config(
    page_title="MLB Predictor",
    page_icon=os.path.join(ASSETS_DIR, "favicon.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)



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
# TEMA VISUAL (src/assets/theme.css + src/assets/logo.svg)
# ============================================================================


def leer_asset(nombre):
    """Lee un archivo estático de la carpeta assets (CSS, SVG)."""
    with open(os.path.join(ASSETS_DIR, nombre), encoding="utf-8") as f:
        return f.read()


# Una sola línea: el markdown de Streamlit trataría las líneas indentadas del SVG como código.
LOGO_SVG = " ".join(line.strip() for line in leer_asset("logo.svg").splitlines())
st.markdown(f"<style>{leer_asset('theme.css')}</style>", unsafe_allow_html=True)

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
    elif tipo == "warning":
        card_class = "alert-card-warning"
        title_class = "alert-card-warning-title"
        msg_class = "alert-card-warning-message"
    elif tipo == "success":
        card_class = "alert-card-success"
        title_class = "alert-card-success-title"
        msg_class = "alert-card-success-message"
    else:
        card_class = "alert-card-info"
        title_class = "alert-card-info-title"
        msg_class = "alert-card-info-message"

    html = f"""<div class="{card_class}">
<div class="{title_class}">
<b>{titulo}</b>
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

    div_names = {
        "ALE": "AL Este", "ALC": "AL Central", "ALO": "AL Oeste",
        "NLE": "NL Este", "NLC": "NL Central", "NLO": "NL Oeste"
    }

    table_html = """<table class="mlb-elo-table">
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
        team_info = EQUIPOS_MLB.get(code, {"nombre": code, "logo": "", "color": "#ff6a00"})
        logo_url = team_info.get("logo", "")
        team_name = team_info.get("nombre", code)

        div_code = DIVISION_MAP.get(code, "MLB")
        div_desc = div_names.get(div_code, div_code)

        if elo >= 1580:
            tier_class = "elo-tier-elite"
            tier_lbl = "Elite"
        elif elo >= 1520:
            tier_class = "elo-tier-strong"
            tier_lbl = "Fuerte"
        elif elo >= 1480:
            tier_class = "elo-tier-competitive"
            tier_lbl = "Competitivo"
        elif elo >= 1420:
            tier_class = "elo-tier-developing"
            tier_lbl = "En Desarrollo"
        else:
            tier_class = "elo-tier-rebuilding"
            tier_lbl = "En Reconstrucción"

        pct = max(5.0, min(100.0, (elo - min_elo) / elo_range * 100.0))

        if elo >= 1580:
            bar_color = "linear-gradient(90deg, #3ddc84, #9df5c3)"
        elif elo >= 1520:
            bar_color = "linear-gradient(90deg, #7a3300, #ff6a00)"
        elif elo >= 1480:
            bar_color = "linear-gradient(90deg, #b37b00, #ffb000)"
        else:
            bar_color = "linear-gradient(90deg, #4a4a4a, #8a8a8a)"

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

    # Definir variables de estilo según el tema
    bar_color = "#ff6a00"
    html = '<div style="overflow-x: auto;"><table class="mlb-results-table">'
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
                    html += '<td><span class="table-badge table-badge-success">Acertado</span></td>'
                else:
                    html += '<td><span class="table-badge table-badge-error">Error</span></td>'

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
                    html += f'<td><div style="display: flex; align-items: center; gap: 8px;">{away_logo} <b>{away}</b> <span style="color: #8a8a8a; font-size: 0.8rem;">@</span> {home_logo} <b>{home}</b></div></td>'
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
        with st.spinner("Actualizando partidos y predicciones del día..."):
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
                    return False, f"El scraper falló: {error_output[-300:]}"
                return False, "El scraper no devolvió datos para hoy"

            predict_result = subprocess.run(
                [sys.executable, predict_path],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=script_dir,
                env=env,
            )
            if predict_result.returncode == 0:
                return True, "Partidos y predicciones actualizados exitosamente"

            error_output = (predict_result.stderr or predict_result.stdout or "").strip()
            if error_output:
                return False, f"Las predicciones fallaron: {error_output[-300:]}"
            return (
                False,
                "Se actualizaron los partidos, pero falló la generación de predicciones",
            )
    except subprocess.TimeoutExpired:
        return False, "Timeout: El proceso tardó más de 5 minutos"
    except Exception as e:
        return False, f"Error: {str(e)}"


def render_system_status_panel(api_ok, api_data):
    """Muestra el estado operativo dentro de la sección informativa."""
    entorno = "Producción" if (os.getenv("API_URL") or has_secrets_api) else "Local"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
        <div class="stats-card">
            <h4>API Backend</h4>
            <p style="font-size: 1.05rem; color: {"#3ddc84" if api_ok else "#ff3b3b"}; font-weight: 800;">{"Conectada" if api_ok else "No disponible"}</p>
            <p style="color: #8a8a8a; margin: 0;">Salud general del servicio</p>
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
            <p style="font-size: 1.05rem; color: {"#3ddc84" if modelo_ok else "#ff3b3b"}; font-weight: 800;">{"Cargado" if modelo_ok else "No disponible"}</p>
            <p style="color: #8a8a8a; margin: 0;">Artefacto predictivo principal</p>
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
            <p style="font-size: 1.05rem; color: {"#3ddc84" if db_ok else "#ffb000"}; font-weight: 800;">{entorno}</p>
            <p style="color: #8a8a8a; margin: 0;">Base de datos {"disponible" if db_ok else "requiere revisión"}</p>
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
            <p style="font-size: 1.35rem; color: #ff6a00; font-weight: 900;">{total:,}</p>
            <p style="color: #8a8a8a; margin: 0;">Generadas por el motor desde el inicio</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="stats-card">
            <h4>Con Resultado Real</h4>
            <p style="font-size: 1.35rem; color: #f2f2f2; font-weight: 900;">{validados:,}</p>
            <p style="color: #8a8a8a; margin: 0;">Partidos jugados y con resultado conocido</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="stats-card">
            <h4>Tasa de Acierto</h4>
            <p style="font-size: 1.35rem; color: #3ddc84; font-weight: 900;">{tasa:.1f}%</p>
            <p style="color: #8a8a8a; margin: 0;">{aciertos:,} aciertos de {validados:,} validados</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
        <div class="stats-card">
            <h4>Pendientes de Validar</h4>
            <p style="font-size: 1.35rem; color: #ffb000; font-weight: 900;">{pendientes:,}</p>
            <p style="color: #8a8a8a; margin: 0;">Partidos sin resultado registrado aún</p>
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


    colors = [
        "#ff6a00" if prob_home > prob_away else "#262626",
        "#ff6a00" if prob_away > prob_home else "#262626",
    ]
    text_color = "#f2f2f2"
    grid_color = "rgba(38, 38, 38, 0.5)"
    line_color = "#0f0f0f"

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[prob_home * 100, prob_away * 100],
            y=[f"{home_team}", f"{away_team}"],
            orientation="h",
            marker={"color": colors, "line": {"color": line_color, "width": 3}},
            text=[f"{prob_home * 100:.1f}%", f"{prob_away * 100:.1f}%"],
            textposition="auto",
            textfont={"size": 18, "color": "white", "family": "Big Shoulders Display"},
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
        font={"size": 14, "family": "Manrope", "color": text_color},
        xaxis={"range": [0, 100], "gridcolor": grid_color, "tickfont": {"color": text_color}, "title_font": {"color": text_color}},
        yaxis={"gridcolor": "rgba(0,0,0,0)", "tickfont": {"color": text_color}},
    )

    return fig


def crear_gauge_confianza(confianza):
    """Crea gauge de nivel de confianza - CORREGIDO"""
    # Normalizar confianza
    confianza = normalizar_probabilidad(confianza)


    axis_color = "#ff6a00"
    bar_color = "#ff6a00"
    bg_gauge = "#070707"
    border_color = "#262626"
    text_color = "#f2f2f2"
    steps = [
        {"range": [0, 55], "color": "rgba(255, 59, 59, 0.15)"},
        {"range": [55, 60], "color": "rgba(255, 176, 0, 0.15)"},
        {"range": [60, 70], "color": "rgba(255, 106, 0, 0.15)"},
        {"range": [70, 100], "color": "rgba(61, 220, 132, 0.25)"},
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
                    "line": {"color": "#ff3b3b", "width": 6},
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
        font={"color": text_color, "family": "Manrope"},
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
            return f'<span style="background:#1b4d31;color:#fff;padding:2px 10px;border-radius:12px;font-weight:700">{racha}G</span>'
        elif racha < 0:
            return f'<span style="background:#5c1b1b;color:#fff;padding:2px 10px;border-radius:12px;font-weight:700">{abs(racha)}P</span>'
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


    if elo_h >= 1580:
        bar_color_h = "linear-gradient(90deg, #3ddc84, #9df5c3)"
    elif elo_h >= 1520:
        bar_color_h = "linear-gradient(90deg, #7a3300, #ff6a00)"
    elif elo_h >= 1480:
        bar_color_h = "linear-gradient(90deg, #b37b00, #ffb000)"
    else:
        bar_color_h = "linear-gradient(90deg, #4a4a4a, #8a8a8a)"

    if elo_a >= 1580:
        bar_color_a = "linear-gradient(90deg, #3ddc84, #9df5c3)"
    elif elo_a >= 1520:
        bar_color_a = "linear-gradient(90deg, #7a3300, #ff6a00)"
    elif elo_a >= 1480:
        bar_color_a = "linear-gradient(90deg, #b37b00, #ffb000)"
    else:
        bar_color_a = "linear-gradient(90deg, #4a4a4a, #8a8a8a)"

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
                    <div style="font-size:1.1rem;font-weight:700;color:{"#3ddc84" if win_rate_s_h >= 50 else "#ff3b3b"}">{win_rate_s_h:.0f}%</div>
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
                    <div style="font-size:1.1rem;font-weight:700;color:#ff6a00">{elo_h:.1f}</div>
                </div>
                <div style="text-align:right">
                    <div style="font-size:0.65rem;opacity:0.7;margin-bottom:4px;text-align:right;">FUERZA</div>
                    <div style="background:{"#070707"} !important; border-radius:4px !important; height:8px !important; width:180px !important; overflow:hidden !important; display:inline-block !important; vertical-align:middle !important; border:1px solid {"rgba(255,255,255,0.15)"} !important;">
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
                    <div style="font-size:1.1rem;font-weight:700;color:{"#3ddc84" if win_rate_s_a >= 50 else "#ff3b3b"}">{win_rate_s_a:.0f}%</div>
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
                    <div style="font-size:1.1rem;font-weight:700;color:#ff6a00">{elo_a:.1f}</div>
                </div>
                <div style="text-align:right">
                    <div style="font-size:0.65rem;opacity:0.7;margin-bottom:4px;text-align:right;">FUERZA</div>
                    <div style="background:{"#070707"} !important; border-radius:4px !important; height:8px !important; width:180px !important; overflow:hidden !important; display:inline-block !important; vertical-align:middle !important; border:1px solid {"rgba(255,255,255,0.15)"} !important;">
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
        color_era_h = "#3ddc84" if era_h < 3.5 else ("#ff3b3b" if era_h > 4.5 else "#ffb000")
        # Color del WHIP
        color_whip_h = "#3ddc84" if whip_h < 1.20 else ("#ff3b3b" if whip_h > 1.40 else "#ffb000")

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
                    <div style="font-size:1.3rem;font-weight:700;color:#3ddc84">{so9_h:.2f}</div>
                </div>
            </div>
        """
    else:
        inner_html_h = f"""
            <div style="text-align:center;padding:1rem;color:#8a8a8a;font-size:0.8rem;font-style:italic">
                Sin estadísticas para {home_pitcher}
            </div>
        """
        nombre_h = home_pitcher

    # 2. Away Pitcher Card
    if away_pitcher_stats and isinstance(away_pitcher_stats, dict):
        nombre_a = away_pitcher_stats.get("nombre", away_pitcher)
        era_a = away_pitcher_stats.get("ERA", 0.0)
        whip_a = away_pitcher_stats.get("WHIP", 0.0)
        so9_a = away_pitcher_stats.get("SO9", 0.0)

        color_era_a = "#3ddc84" if era_a < 3.5 else ("#ff3b3b" if era_a > 4.5 else "#ffb000")
        color_whip_a = "#3ddc84" if whip_a < 1.20 else ("#ff3b3b" if whip_a > 1.40 else "#ffb000")

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
                    <div style="font-size:1.3rem;font-weight:700;color:#3ddc84">{so9_a:.2f}</div>
                </div>
            </div>
        """
    else:
        inner_html_a = f"""
            <div style="text-align:center;padding:1rem;color:#8a8a8a;font-size:0.8rem;font-style:italic">
                Sin estadísticas para {away_pitcher}
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
        st.markdown("### Información sobre los datos utilizados")

        # Mostrar info del lanzador local si tiene fallback
        if hubo_fallback_home:
            ip_home = resultado_detallado.get("ip_home", 0)
            detalle_home = razon_home if razon_home else f"No había muestra suficiente/estable en {year_solicitado}"
            st.info(
                f"**{home_team} - {home_pitcher}**: Se usaron estadísticas de la temporada "
                f"**{year_home}** para mejorar la estabilidad del análisis. "
                f"Motivo: {detalle_home}."
                f" IP consideradas: **{ip_home}**."
            )

        # Mostrar info del lanzador visitante si tiene fallback
        if hubo_fallback_away:
            ip_away = resultado_detallado.get("ip_away", 0)
            detalle_away = razon_away if razon_away else f"No había muestra suficiente/estable en {year_solicitado}"
            st.info(
                f"**{away_team} - {away_pitcher}**: Se usaron estadísticas de la temporada "
                f"**{year_away}** para mejorar la estabilidad del análisis. "
                f"Motivo: {detalle_away}."
                f" IP consideradas: **{ip_away}**."
            )

        st.markdown(
            "*Este ajuste es normal al inicio de temporada: cuando la muestra de innings "
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
        st.info("Las Super Features no están disponibles en esta respuesta de la API")
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
                <p style="font-size: 0.925rem; line-height: 1.5; color: #c9c9c9; margin-top: 0.75rem;">
                    <strong>{pitcher_n}</strong> tiene un WHIP que neutraliza efectivamente
                    el OPS del lineup de <strong>{rival_n}</strong>, limitando su producción ofensiva.
                </p>
                <div style="margin-top: 0.75rem; font-size: 0.775rem; color: #8a8a8a; font-weight: 600;">
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
                <p style="font-size: 0.925rem; line-height: 1.5; color: #c9c9c9; margin-top: 0.75rem;">
                    <strong>{pitcher_r}</strong> demuestra mayor resistencia ante bateadores
                    de poder, manteniendo su ERA bajo presión ofensiva.
                </p>
                <div style="margin-top: 0.75rem; font-size: 0.775rem; color: #8a8a8a; font-weight: 600;">
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
                <p style="font-size: 0.925rem; line-height: 1.5; color: #c9c9c9; margin-top: 0.75rem;">
                    El bullpen de <strong>{ventaja_m}</strong> es más efectivo
                    contra los mejores bateadores rivales en innings críticos.
                </p>
                <div style="margin-top: 0.75rem; font-size: 0.775rem; color: #8a8a8a; font-weight: 600;">
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
        st.markdown("### Tendencias y Momentum")
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
                st.info(f"No hay datos de bateadores disponibles para {home_team}")

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
                st.info(f"No hay datos de bateadores disponibles para {away_team}")



# ============================================================================
# SEGUIMIENTO DE PARTIDOS (MLB Stats API): en vivo, qué pasó y repetición
# ============================================================================


@st.cache_data(ttl=3600, show_spinner=False)
def resolver_game_pk(fecha, home_team, away_team):
    """Id oficial (gamePk) de un partido del proyecto; None si la MLB no lo lista."""
    try:
        return mlb_live.buscar_game_pk(fecha, home_team, away_team)
    except Exception:
        return None


@st.cache_data(ttl=20, show_spinner=False)
def resumen_partido_vivo(game_pk):
    return mlb_live.resumir_feed(mlb_live.obtener_feed(game_pk))


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def resumen_partido_final(game_pk):
    return mlb_live.resumir_feed(mlb_live.obtener_feed(game_pk))


@st.cache_data(ttl=120, show_spinner=False)
def cartelera_oficial(fecha):
    try:
        return mlb_live.cartelera_del_dia(fecha)
    except Exception:
        return []


def obtener_resumen_partido(game_pk, terminado=False):
    """Caché larga para partidos terminados y corta (20 s) para los que pueden estar en curso."""
    try:
        if terminado:
            return resumen_partido_final(game_pk)
        return resumen_partido_vivo(game_pk)
    except Exception:
        return None


def parsear_features(partido):
    """Variables que usó el modelo (campo `detalles` de la predicción)."""
    detalles = partido.get("detalles")
    if not detalles:
        return {}
    try:
        datos = json.loads(detalles) if isinstance(detalles, str) else detalles
        return datos.get("features_usadas", {}) or {}
    except Exception:
        return {}


def _ordinal(n):
    return mlb_live.ORDINALES.get(n, f"{n}ª")


def html_bases(bases):
    def rombo(cx, cy, ocupada):
        color = "#ff6a00" if ocupada else "#3a3a3a"
        relleno = "#ff6a00" if ocupada else "none"
        return (
            f'<rect x="{cx - 9}" y="{cy - 9}" width="18" height="18" transform="rotate(45 {cx} {cy})" '
            f'fill="{relleno}" stroke="{color}" stroke-width="2"/>'
        )

    return (
        '<svg viewBox="0 0 60 60" aria-label="Bases">'
        + rombo(44, 32, bases.get("primera"))
        + rombo(30, 18, bases.get("segunda"))
        + rombo(16, 32, bases.get("tercera"))
        + "</svg>"
    )


def html_linescore(resumen, hasta_entrada=None, en_vivo=False):
    """Tabla de carreras por entrada. Con `hasta_entrada` recorta la repetición."""
    entradas = resumen.get("entradas", [])
    n_cols = max(resumen.get("entradas_programadas", 9) or 9, len(entradas))
    actual = resumen["situacion"].get("entrada") if en_vivo else None
    por_num = {e["num"]: e for e in entradas}

    def fila(lado, code):
        celdas, total = [], 0
        for n in range(1, n_cols + 1):
            e = por_num.get(n)
            visible = e is not None and (hasta_entrada is None or n <= hasta_entrada)
            carreras = e.get(lado) if visible else None
            if carreras is not None:
                total += carreras
            clase = ' class="now"' if actual == n else ""
            celdas.append(f"<td{clase}>{'' if carreras is None else carreras}</td>")
        if hasta_entrada is None:
            r = resumen["marcador"].get(lado)
            h = resumen["marcador"]["hits"].get(lado, 0)
            e_ = resumen["marcador"]["errores"].get(lado, 0)
            tot = f'<td class="tot tot-r">{"" if r is None else r}</td><td class="tot">{h}</td><td class="tot">{e_}</td>'
        else:
            tot = f'<td class="tot tot-r">{total}</td><td class="tot"></td><td class="tot"></td>'
        ganador = ' class="ganador"' if resumen.get("ganador") == lado and hasta_entrada is None else ""
        return f"<tr{ganador}><td>{code}</td>{''.join(celdas)}{tot}</tr>"

    cab = "".join(f"<th>{n}</th>" for n in range(1, n_cols + 1))
    return (
        '<table class="linescore"><thead><tr><th></th>'
        + cab
        + '<th class="tot">R</th><th class="tot">H</th><th class="tot">E</th></tr></thead><tbody>'
        + fila("away", resumen["away"]["code"])
        + fila("home", resumen["home"]["code"])
        + "</tbody></table>"
    )


def html_situacion(resumen):
    s = resumen["situacion"]
    outs = "".join(f'<i class="{"on" if i < (s.get("outs") or 0) else ""}"></i>' for i in range(3))
    mitad = "Alta" if s.get("mitad") == "alta" else "Baja"
    entrada = _ordinal(s.get("entrada") or 0)
    return (
        '<div class="situacion">'
        f'<div class="bases">{html_bases(s.get("bases", {}))}</div>'
        f'<div><div class="inning">{mitad} {entrada}<small>{s.get("estado_entrada", "")}</small></div>'
        f'<div class="count">Cuenta <b>{s.get("bolas", 0)}-{s.get("strikes", 0)}</b> &nbsp; Outs <span class="outs">{outs}</span></div></div>'
        f'<div class="duelo">Lanza <b>{s.get("lanzador") or "-"}</b><br>Batea <b>{s.get("bateador") or "-"}</b></div>'
        "</div>"
        + (f'<div class="ultima-jugada"><b>Última jugada</b> {s["ultima_jugada"]}</div>' if s.get("ultima_jugada") else "")
    )


def html_jugadas(jugadas, decisiva=None):
    if not jugadas:
        return '<p class="stats-card-desc">Sin jugadas registradas.</p>'
    items = []
    for j in jugadas:
        clases = []
        if j.get("anota"):
            clases.append("anota")
        if decisiva is not None and j.get("indice") == decisiva.get("indice"):
            clases.append("decisiva")
        flecha = "&#9650;" if j["mitad"] == "alta" else "&#9660;"
        items.append(
            f'<li class="{" ".join(clases)}"><span class="cuando">{flecha} {_ordinal(j["entrada"])}</span>'
            f'<span>{j["descripcion"]}</span><span class="marcador">{j["away"]}-{j["home"]}</span></li>'
        )
    return f'<ul class="play-list">{"".join(items)}</ul>'


def html_factores(factores, prediccion, rival):
    items = []
    for f in factores:
        if f["favorece"] == prediccion:
            chip = f'<span class="chip a-favor">favorece {prediccion}</span>'
        elif f["favorece"] == rival:
            chip = f'<span class="chip en-contra">favorece {rival}</span>'
        else:
            chip = '<span class="chip">neutro</span>'
        items.append(f'<li><span class="factor">{f["factor"]}</span><span>{f["texto"]}</span>{chip}</li>')
    return f'<ul class="reason-list">{"".join(items)}</ul>'


def html_abridores(resumen):
    def linea(lado):
        a = resumen["box"][lado].get("abridor")
        code = resumen[lado]["code"]
        if not a:
            return f'<div class="pitcher-line"><b>{code}</b>Sin datos del abridor</div>'
        return (
            f'<div class="pitcher-line"><b>{a["nombre"]} ({code})</b>'
            f'{a["ip"]} IP, {a["h"]} H, {a["r"]} C ({a["er"]} limpias), {a["bb"]} BB, {a["k"]} K, {a["lanzamientos"]} lanzamientos</div>'
        )

    return f'<div class="pitchers-box">{linea("away")}{linea("home")}</div>'


def html_abridores_previos(probables, resumen):
    return (
        '<div class="pitchers-box" style="padding: 0.9rem 1.25rem;">'
        f'<div class="pitcher-line"><b>{resumen["away"]["code"]}</b>Abre {probables.get("away") or "por confirmar"}</div>'
        f'<div class="pitcher-line"><b>{resumen["home"]["code"]}</b>Abre {probables.get("home") or "por confirmar"}</div>'
        "</div>"
    )


def render_por_que(partido, features):
    """Por qué el modelo eligió a ese equipo, a partir de las variables guardadas."""
    prediccion = partido.get("prediccion")
    home, away = partido["home_team"], partido["away_team"]
    if not prediccion:
        st.caption("Este partido no tiene predicción del modelo.")
        return
    if not features:
        st.caption("No se guardaron las variables del modelo para este partido, así que no se puede reconstruir el razonamiento.")
        return
    rival = away if prediccion == home else home
    factores = mlb_live.explicar_prediccion(
        prediccion, home, away, features, partido.get("home_pitcher") or "", partido.get("away_pitcher") or ""
    )
    prob = normalizar_probabilidad(partido.get("prob_home" if prediccion == home else "prob_away", 0.5)) * 100
    confianza = (partido.get("confianza") or "sin dato").lower()
    st.markdown(
        f'<div class="analisis"><p>El modelo dio <b>{prob:.1f}%</b> a <b>{prediccion}</b> con confianza {confianza}. '
        f"{mlb_live.resumen_factores(factores, prediccion, rival)}</p></div>" + html_factores(factores, prediccion, rival),
        unsafe_allow_html=True,
    )


def render_que_paso(partido, resumen, features):
    """Análisis del resultado frente a la predicción, con box score y jugadas de anotación."""
    prediccion = partido.get("prediccion")
    home, away = partido["home_team"], partido["away_team"]
    if not prediccion:
        st.caption("Sin predicción del modelo para contrastar.")
        st.markdown(html_abridores(resumen), unsafe_allow_html=True)
        st.markdown(html_jugadas(resumen["jugadas_anotacion"]), unsafe_allow_html=True)
        return
    analisis = mlb_live.analizar_resultado(prediccion, home, away, resumen, features)
    if analisis["acierto"] is None:
        st.caption(analisis["titulo"])
        return
    chip = '<span class="chip ok">acierto</span>' if analisis["acierto"] else '<span class="chip fallo">fallo</span>'
    claves = "".join(f'<div>{c["etiqueta"]}<b>{c["valor"]}</b></div>' for c in analisis["claves"])
    parrafos = "".join(f"<p>{p}</p>" for p in analisis["parrafos"])
    st.markdown(
        f'<div class="analisis"><div class="analisis-titulo">{analisis["titulo"]} {chip}</div>{parrafos}'
        f'<div class="claves">{claves}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(html_abridores(resumen), unsafe_allow_html=True)
    st.markdown('<div class="replay-half">Jugadas de anotación (la decisiva, en naranja)</div>', unsafe_allow_html=True)
    st.markdown(html_jugadas(resumen["jugadas_anotacion"], analisis.get("momento")), unsafe_allow_html=True)


def render_repeticion(resumen, key):
    """Simulador del partido: recorre entrada a entrada el marcador y las jugadas."""
    por_entrada = resumen.get("jugadas_por_entrada") or {}
    if not por_entrada:
        st.caption("Todavía no hay jugadas para reproducir.")
        return
    ultima = max(por_entrada)
    n = st.slider("Entrada", 1, ultima, ultima, key=f"rep_{key}") if ultima > 1 else 1
    jugadas_hasta = [j for e, mitades in por_entrada.items() if e <= n for j in mitades["alta"] + mitades["baja"]]
    away_r = jugadas_hasta[-1]["away"] if jugadas_hasta else 0
    home_r = jugadas_hasta[-1]["home"] if jugadas_hasta else 0
    st.markdown(
        '<div class="replay-score">'
        f'<span class="eq">{resumen["away"]["code"]}</span><span class="num {"lidera" if away_r > home_r else ""}">{away_r}</span>'
        '<span class="eq">-</span>'
        f'<span class="num {"lidera" if home_r > away_r else ""}">{home_r}</span><span class="eq">{resumen["home"]["code"]}</span>'
        "</div>" + html_linescore(resumen, hasta_entrada=n),
        unsafe_allow_html=True,
    )
    mitades = por_entrada.get(n, {"alta": [], "baja": []})
    for nombre, lista in (("alta", mitades["alta"]), ("baja", mitades["baja"])):
        quien = resumen["away"]["code"] if nombre == "alta" else resumen["home"]["code"]
        st.markdown(f'<div class="replay-half">{nombre.capitalize()} de la {_ordinal(n)}, batea {quien}</div>', unsafe_allow_html=True)
        if not lista and nombre == "baja" and n == ultima and resumen.get("estado") == "final":
            st.markdown('<p class="stats-card-desc">No se jugó: el local ya ganaba al cerrar la alta.</p>', unsafe_allow_html=True)
        else:
            st.markdown(html_jugadas(lista), unsafe_allow_html=True)


def _render_panel_partido(resumen, partido, features, key):
    estado = resumen["estado"]
    prediccion = partido.get("prediccion")
    home, away = partido["home_team"], partido["away_team"]
    if estado == "en_vivo":
        badge = '<span class="live-badge">EN VIVO</span>'
    elif estado == "final":
        badge = '<span class="final-badge">FINAL</span>'
    else:
        badge = '<span class="previo-badge">PROGRAMADO</span>'
    lugar = " ".join(x for x in (resumen.get("lugar"), f"({resumen['clima']})" if resumen.get("clima") else "") if x)
    cabecera = (
        '<div class="live-panel"><div class="live-panel-head">'
        f"{badge}<span>{resumen.get('hora_et', '')} ET</span><span>{lugar}</span></div>"
    )

    if estado == "previo":
        cuerpo = (
            '<div class="live-pulse">Primer lanzamiento a las '
            f"<b>{(resumen.get('hora_et') or '')[-5:]}</b> hora del Este.</div>"
            + html_abridores_previos(resumen.get("probables", {}), resumen)
        )
        st.markdown(cabecera + cuerpo + "</div>", unsafe_allow_html=True)
        if prediccion and features:
            with st.expander(f"Por qué el modelo elige a {prediccion}"):
                render_por_que(partido, features)
        return

    pulso = ""
    if estado == "en_vivo" and prediccion:
        pulso = f'<div class="live-pulse">{mlb_live.pulso_en_vivo(prediccion, home, away, resumen)}</div>'
    cuerpo = pulso + html_linescore(resumen, en_vivo=(estado == "en_vivo"))
    if estado == "en_vivo":
        cuerpo += html_situacion(resumen)
    st.markdown(cabecera + cuerpo + "</div>", unsafe_allow_html=True)

    if estado == "en_vivo":
        with st.expander("Jugadas de anotación hasta ahora"):
            st.markdown(html_jugadas(resumen["jugadas_anotacion"]), unsafe_allow_html=True)
        if prediccion and features:
            with st.expander(f"Por qué el modelo eligió a {prediccion}"):
                render_por_que(partido, features)
        return

    tab_paso, tab_rep, tab_por = st.tabs(["Qué pasó", "Repetición", "Por qué lo predijimos"])
    with tab_paso:
        render_que_paso(partido, resumen, features)
    with tab_rep:
        render_repeticion(resumen, key)
    with tab_por:
        render_por_que(partido, features)


def render_seguimiento_partido(partido, features, key, terminado=False):
    """Panel bajo la tarjeta del partido: en vivo (se refresca solo), programado o final."""
    game_pk = partido.get("game_pk") or resolver_game_pk(partido.get("fecha"), partido["home_team"], partido["away_team"])
    if not game_pk:
        st.caption("No se encontró este partido en la cartelera oficial de la MLB.")
        return
    resumen = obtener_resumen_partido(game_pk, terminado=terminado)
    if not resumen:
        st.caption("La MLB no respondió a tiempo; el seguimiento del partido no está disponible ahora mismo.")
        return

    if resumen["estado"] == "en_vivo":

        @st.fragment(run_every="30s")
        def _panel_en_vivo():
            actual = obtener_resumen_partido(game_pk) or resumen
            _render_panel_partido(actual, partido, features, key)

        _panel_en_vivo()
    else:
        _render_panel_partido(resumen, partido, features, key)


def render_tarjeta_oficial(p):
    """Tarjeta compacta para un partido de la cartelera oficial sin predicción del modelo."""
    a_logo = get_team_logo_html(p["away_team"], 32)
    h_logo = get_team_logo_html(p["home_team"], 32)
    estado = {"en_vivo": "EN VIVO", "final": "FINAL", "previo": "PROGRAMADO"}.get(p["estado"], p["estado_detalle"])
    st.markdown(
        f"""<div class="mlb-scoreboard-card">
<div class="mlb-card-header"><span>{estado}</span><span>{p.get("serie") or ""} {p.get("hora_et", "")} ET</span></div>
<div class="mlb-card-body">
<div class="mlb-team-row"><div class="mlb-team-info">{a_logo}<div class="mlb-team-name">{get_team_display_name(p["away_team"])}</div></div><div class="mlb-team-record">{p.get("away_pitcher") or ""}</div></div>
<div class="mlb-team-row"><div class="mlb-team-info">{h_logo}<div class="mlb-team-name">{get_team_display_name(p["home_team"])}</div></div><div class="mlb-team-record">{p.get("home_pitcher") or ""}</div></div>
</div></div>""",
        unsafe_allow_html=True,
    )


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    estado_api = (
        '<span class="live">API conectada</span>' if api_ok else "<span>API sin conexión</span>"
    )
    st.markdown(
        f"""
    <div class="brand">
        {LOGO_SVG}
        <div class="brand-name">MLB<span>Predictor</span></div>
    </div>
    <div class="brand-meta">
        <span>Modelo V4.0</span>
        {estado_api}
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        '<div class="sidebar-section-title">Secciones</div>',
        unsafe_allow_html=True,
    )
    pagina = st.radio(
        "Selecciona una sección:",
        [
            "Partidos de hoy",
            "Historial y precisión",
            "Power Rankings",
            "Rendimiento",
            "Playoffs",
            "Acerca del modelo",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        """
    <div class="sidebar-foot">
        <b>Datos</b>: API oficial de la MLB<br>
        <b>Actualización</b>: automática, cada día<br>
        <b>Uso</b>: análisis estadístico, no asesoría de apuestas
    </div>
    """,
        unsafe_allow_html=True,
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
                            st.info("Las Super Features no están disponibles en esta respuesta de la API")
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
                                    <p style="font-size: 0.925rem; line-height: 1.5; color: #c9c9c9; margin-top: 0.75rem;">
                                        <strong>{pitcher_n}</strong> tiene un WHIP que neutraliza efectivamente
                                        el OPS del lineup de <strong>{rival_n}</strong>, limitando su producción ofensiva.
                                    </p>
                                    <div style="margin-top: 0.75rem; font-size: 0.775rem; color: #8a8a8a; font-weight: 600;">
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
                                    <p style="font-size: 0.925rem; line-height: 1.5; color: #c9c9c9; margin-top: 0.75rem;">
                                        <strong>{pitcher_r}</strong> demuestra mayor resistencia ante bateadores
                                        de poder, manteniendo su ERA bajo presión ofensiva.
                                    </p>
                                    <div style="margin-top: 0.75rem; font-size: 0.775rem; color: #8a8a8a; font-weight: 600;">
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
                                    <p style="font-size: 0.925rem; line-height: 1.5; color: #c9c9c9; margin-top: 0.75rem;">
                                        El bullpen de <strong>{ventaja_m}</strong> es más efectivo
                                        contra los mejores bateadores rivales en innings críticos.
                                    </p>
                                    <div style="margin-top: 0.75rem; font-size: 0.775rem; color: #8a8a8a; font-weight: 600;">
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
                                st.markdown("### Tendencias y Momentum")
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
                                    st.info(f"No hay datos de bateadores disponibles para {home_team}")

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
                                    st.info(f"No hay datos de bateadores disponibles para {away_team}")

                        # Descargar reporte
                        st.markdown("---")
                        result_json = json.dumps(resultado, indent=2)
                        st.download_button(
                            label="Descargar Reporte Técnico (JSON)",
                            data=result_json,
                            file_name=f"prediccion_{home_team}_vs_{away_team}_{year}.json",
                            mime="application/json",
                        )
                    else:
                        error = response.json()
                        st.error(f"Error: {error.get('detail', 'Error desconocido')}")
                        st.info("Verifica que los nombres de los lanzadores o el año de busqueda sean correctos.")

                except requests.exceptions.Timeout:
                    st.error("Timeout: La predicción tardó más de 2 minutos")
                except Exception as e:
                    st.error("Error al procesar la predicción")
                    st.exception(e)

# ============================================================================
# PÁGINA: PARTIDOS DE HOY
# ============================================================================

elif pagina == "Partidos de hoy":
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
            color_num = "#3ddc84" if num_juegos > 0 else "#ff3b3b"
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
            <div style="text-align: center; padding: 4rem 2rem; color: #8a8a8a;">
                <h2>No hay partidos programados para hoy</h2>
                <p style="font-size: 1.1rem; margin: 1rem 0;">
                    Los partidos aún no han sido scrapeados o no hay juegos programados
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            oficiales = cartelera_oficial(fecha_hoy)
            titulo_oficial = "Cartelera oficial de la MLB para hoy"
            if not oficiales:
                manana = (ahora_et + timedelta(days=1)).strftime("%Y-%m-%d")
                oficiales = cartelera_oficial(manana)
                titulo_oficial = f"Próximos partidos oficiales ({manana})"
            if oficiales:
                st.markdown(f"### {titulo_oficial}")
                st.caption(
                    "Según la MLB Stats API. Estos partidos aún no tienen predicción del modelo; "
                    "el seguimiento en vivo y el resumen sí están disponibles."
                )
                for p in oficiales:
                    render_tarjeta_oficial(p)
                    render_seguimiento_partido(p, {}, key=f"of_{p['game_pk']}")

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
                        away_racha = f"{away_racha_val}G"
                        away_racha_class = "racha-win"
                    elif away_racha_val < 0:
                        away_racha = f"{abs(away_racha_val)}P"
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
                        home_racha = f"{home_racha_val}G"
                        home_racha_class = "racha-win"
                    elif home_racha_val < 0:
                        home_racha = f"{abs(home_racha_val)}P"
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
                        away_racha_style = f'class="racha-pill {away_racha_class}"' if away_racha_class else 'style="color:#8a8a8a"'
                        home_racha_style = f'class="racha-pill {home_racha_class}"' if home_racha_class else 'style="color:#8a8a8a"'

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
                        render_seguimiento_partido(
                            partido, features, key=f"hoy_{partido.get('game_id', '')}_{partido['home_team']}"
                        )

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
        st.error(f"Error obteniendo datos: {str(e)}")

# ============================================================================
# PÁGINA: COMPARACIÓN & HISTORIAL
# ============================================================================

elif pagina == "Historial y precisión":
    st.markdown(
        '<div class="main-title"> Comparación de Predicciones vs Resultados</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Analiza el rendimiento histórico del modelo</div>',
        unsafe_allow_html=True,
    )

    if not api_ok:
        st.error("La API no está disponible")
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

    st.markdown("### Selecciona una Fecha")

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
        if st.button("Analizar Fecha", type="primary", use_container_width=True):
            st.session_state.fecha_analizar = fecha_seleccionada
            st.session_state.pop("stats_30d", None)

    with col3:
        if st.button("Ver Estadísticas", use_container_width=True):
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
<span class="rendimiento-stat-val" style="color: #3ddc84;">{_aciertos}</span>
<span class="rendimiento-stat-lbl">Aciertos</span>
</div>
<div class="rendimiento-stat-box">
<span class="rendimiento-stat-val" style="color: #ff6a00;">{_acc:.1f}%</span>
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
            _colores = {
                "MUY ALTA": "#ff6a00",
                "ALTA": "#ffb000",
                "MODERADA": "#f2f2f2",
                "BAJA": "#8a8a8a",
                "BAJA (Partido muy parejo)": "#8a8a8a",
                "BAJA (PARTIDO MUY PAREJO)": "#8a8a8a",
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
                _color = _colores.get(_nivel, "#8a8a8a")
                _acc_nivel = _datos.get('accuracy', 0)
                _aciertos_nivel = _datos.get('aciertos', 0)
                _total_nivel = _datos.get('total', 0)

                boxes_html += f"""
<div class="rendimiento-stat-box" style="min-width: 140px;">
<span class="rendimiento-stat-val" style="color: {_color};">{_acc_nivel:.1f}%</span>
<span class="rendimiento-stat-lbl" style="font-weight: 800; font-size: 0.65rem;">{_nivel}</span>
<span class="rendimiento-stat-lbl" style="text-transform: none; font-size: 0.55rem; color: #8a8a8a; margin-top: 2px;">{_aciertos_nivel}/{_total_nivel} aciertos</span>
</div>"""

            st.markdown(
                f"""<div class="rendimiento-container" style="border-left: 6px solid #ffb000 !important;">
<div class="rendimiento-header">
<div style="display: flex; align-items: center; gap: 12px;">
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
                bg_color = "#070707"
                border_style = "border: 1px solid #262626; border-left: 4px solid #ff6a00;"
                text_color = "#f2f2f2"
                title_color = "#ff6a00"
                hr_color = "#262626"
                link_color = "#ff6a00"

                st.markdown(
                    f"""
                    <div style="background-color: {bg_color}; {border_style} padding: 1.25rem; border-radius: 0.5rem; margin-bottom: 2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                        <div style="display: flex; align-items: center; margin-bottom: 0.75rem;">
                            <span style="color: {title_color}; font-weight: 700; font-size: 1.15rem;">
                                Resultados Reales Aún No Disponibles
                            </span>
                        </div>
                        <p style="color: {text_color}; font-size: 0.975rem; line-height: 1.6; margin: 0;">
                            Los resultados reales para la fecha <b>{fecha_str}</b> aún no han sido registrados en la base de datos o los partidos están en desarrollo/programados para jugarse hoy.
                        </p>
                        <hr style="border: 0; border-top: 1px solid {hr_color}; margin: 1rem 0;">
                        <p style="color: {link_color}; font-size: 0.925rem; line-height: 1.5; margin: 0; font-weight: 500;">
                            <b>¿Quieres ver la cartelera y predicciones de hoy?</b><br>
                            Dirígete a la sección <b>Partidos de Hoy</b> en el menú lateral izquierdo para ver lineups oficiales y análisis detallados en tiempo real.
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
<span class="rendimiento-stat-val" style="color: #3ddc84;">{stats["aciertos"]}</span>
<span class="rendimiento-stat-lbl">Aciertos</span>
</div>
<div class="rendimiento-stat-box">
<span class="rendimiento-stat-val" style="color: #ff6a00;">{stats["accuracy"]:.1f}%</span>
<span class="rendimiento-stat-lbl">Accuracy</span>
</div>
<div class="rendimiento-stat-box">
<span class="rendimiento-stat-val" style="color: #ff3b3b;">{errores}</span>
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
                                    _badge_text = "PRECAUCIÓN: SIN DATOS COMPLETOS"
                                    _badge_color = "rgba(255, 176, 0, 0.15)"  # Amarillo/Ámbar suave
                                    _text_color = "#ffb000"   # Marrón/Naranja oscuro
                                else:
                                    _badge_text = "ACIERTO" if acierto else "ERROR"
                                    _badge_color = "rgba(61, 220, 132, 0.15)" if acierto else "rgba(255, 59, 59, 0.15)"
                                    _text_color = "#3ddc84" if acierto else "#ff3b3b"

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
            <b>{_badge_text}</b>
        </div>
        <div style="font-size: 0.75rem; color: #8a8a8a;">
            Predicción: <b>{get_team_display_name(partido.get("prediccion", ""))}</b>
        </div>
    </div>
</div>
""",
                                    unsafe_allow_html=True,
                                )
                                # Botón de detalles opcional
                                with st.expander("Ver análisis del partido", expanded=False):
                                    prob_h = normalizar_probabilidad(partido.get("prob_home", 0.5))
                                    prob_a = normalizar_probabilidad(partido.get("prob_away", 0.5))
                                    confianza = partido.get("confianza", "N/A")
                                    color_conf = {
                                        "MUY ALTA": "#ff6a00",
                                        "ALTA": "#ffb000",
                                        "MODERADA": "#f2f2f2",
                                        "BAJA": "#8a8a8a",
                                    }.get(confianza, "#8a8a8a")
                                    st.markdown(
                                        f"**Confianza:** <span style='color:{color_conf}; font-weight:bold;'>{confianza}</span>"
                                        f" &nbsp; {get_team_display_name(_at)} {prob_a * 100:.1f}% &nbsp; "
                                        f"{get_team_display_name(_ht)} {prob_h * 100:.1f}%",
                                        unsafe_allow_html=True,
                                    )
                                    features_partido = parsear_features(partido)
                                    render_seguimiento_partido(
                                        partido,
                                        features_partido,
                                        key=f"hist_{partido.get('game_id', '')}_{_ht}",
                                        terminado=partido.get("score_home") is not None,
                                    )
        else:
            st.info(f" No hay datos disponibles para {fecha_str}")

# ============================================================================
# PÁGINA: POWER RANKINGS
# ============================================================================

elif pagina == "Power Rankings":
    st.markdown(
        '<div class="main-title">Power Rankings (Sistema ELO)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Clasificación dinámica de fuerza y probabilidad relativa de equipos</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Clasificación de Fuerza de Equipos (ELO)")

    legend_html = """<div class="elo-legend-container" style="background: rgba(255, 106, 0, 0.04); border: 1px solid rgba(255, 106, 0, 0.15); border-radius: 12px; padding: 16px; margin-bottom: 20px;">
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
<h4 style="margin: 0; color: #ff6a00; font-weight: 700; font-size: 1.05rem;">Leyenda y Funcionamiento del Sistema ELO</h4>
</div>
<p style="margin: 0 0 12px 0; font-size: 0.85rem; color: #f2f2f2; line-height: 1.45;">
El <strong>Sistema ELO</strong> es un método matemático de clasificación que calcula la fuerza competitiva relativa de cada equipo de forma dinámica.
Todos los equipos inician con una base neutral de <strong>1500 puntos</strong>. Tras cada partido, el equipo ganador sustrae puntos al perdedor según la expectativa previa de victoria (vencer a un rival más fuerte otorga más puntos). Se incluye una ventaja de localía de <strong>+24 puntos</strong> para el equipo de casa en la evaluación de cada encuentro.
</p>
<div style="display: flex; gap: 8px; flex-wrap: wrap; font-size: 0.75rem; font-weight: 700;">
<span style="background: rgba(61, 220, 132, 0.12); color: #3ddc84; border: 1px solid rgba(61, 220, 132, 0.25); padding: 4px 8px; border-radius: 6px;">Elite (&ge; 1580)</span>
<span style="background: rgba(255, 106, 0, 0.12); color: #ff6a00; border: 1px solid rgba(255, 106, 0, 0.25); padding: 4px 8px; border-radius: 6px;">Fuerte (1520 - 1579)</span>
<span style="background: rgba(255, 176, 0, 0.12); color: #ffb000; border: 1px solid rgba(255, 176, 0, 0.25); padding: 4px 8px; border-radius: 6px;">Competitivo (1480 - 1519)</span>
<span style="background: rgba(255, 59, 59, 0.12); color: #ff3b3b; border: 1px solid rgba(255, 59, 59, 0.25); padding: 4px 8px; border-radius: 6px;">En Desarrollo (1420 - 1479)</span>
<span style="background: rgba(138, 138, 138, 0.12); color: #8a8a8a; border: 1px solid rgba(138, 138, 138, 0.25); padding: 4px 8px; border-radius: 6px;">En Reconstrucción (&lt; 1420)</span>
</div>
</div>"""
    st.markdown("".join([line.strip() for line in legend_html.splitlines()]), unsafe_allow_html=True)

    elo_list = get_elo_power_rankings()
    if elo_list:
        elo_table_html = render_power_rankings_table_html(elo_list)
        st.markdown(elo_table_html, unsafe_allow_html=True)
    else:
        st.warning("No se pudieron cargar los Power Rankings ELO.")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("¿Cómo interpretar la clasificación ELO y sus niveles?"):
        st.markdown("""
        El **Sistema ELO** es un método matemático para calcular la fuerza competitiva relativa de los equipos:

        * **Puntuación Base (1500):** Es el punto de partida neutral. Un equipo con ELO de 1500 tiene un rendimiento equilibrado.
        * **Cálculo Dinámico:** Tras cada partido, el equipo ganador toma puntos del perdedor. La cantidad de puntos ganados depende de la expectativa del partido.
        * **Calidad de Oposición:** Si un equipo de bajo nivel vence a uno de nivel alto (sorpresa), la transferencia de puntos es mucho mayor.
        * **Ventaja de Localía (+24 pts):** Al evaluar un partido, al equipo de casa se le suman artificialmente 24 puntos de ELO para representar la ventaja estadística de jugar en su estadio.

        #### Niveles de Clasificación (Tiers):
        * **Elite (≥ 1580):** Contendientes indiscutibles al campeonato. Rendimiento dominante y rachas de victorias constantes.
        * **Fuerte (1520 - 1579):** Equipos muy sólidos con altas probabilidades de clasificar a postemporada.
        * **Competitivo (1480 - 1519):** Equipos de nivel medio, capaces de pelear por puestos de comodín (*Wild Card*).
        * **En Desarrollo (1420 - 1479):** Equipos en transición con problemas de consistencia o rachas negativas.
        * **En Reconstrucción (< 1420):** Equipos en la parte baja de la tabla, enfocados en desarrollar talento joven.
        """)

# ============================================================================
# PÁGINA: RENDIMIENTO DEL MODELO
# ============================================================================

elif pagina == "Rendimiento":
    st.markdown(
        '<div class="main-title">Rendimiento del Modelo</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Análisis visual del rendimiento histórico del modelo</div>',
        unsafe_allow_html=True,
    )
    render_salud_modelo(API_URL, render_custom_alert)
    st.markdown("---")

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

                st.markdown("### Filtro de Fechas")
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
                    st.markdown("### Resultados por Partido")

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
                    text_color_pag = "#a3a3a3"

                    with col_pag1:
                        if st.session_state.resultados_page > 1:
                            if st.button("Anterior", key="btn_prev_page", use_container_width=True):
                                st.session_state.resultados_page -= 1
                                st.rerun()
                        else:
                            st.button("Anterior", key="btn_prev_page_disabled", disabled=True, use_container_width=True)

                    with col_pag2:
                        st.markdown(
                            f"<div style='text-align: center; font-weight: bold; padding: 6px; color: {text_color_pag}; font-size: 0.85rem;'>"
                            f"Página {st.session_state.resultados_page} de {total_pages} (Mostrando {start_idx+1}-{end_idx} de {total_items} resultados)"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                    with col_pag3:
                        if st.session_state.resultados_page < total_pages:
                            if st.button("Siguiente", key="btn_next_page", use_container_width=True):
                                st.session_state.resultados_page += 1
                                st.rerun()
                        else:
                            st.button("Siguiente", key="btn_next_page_disabled", disabled=True, use_container_width=True)

                    st.markdown("---")
                    st.markdown("### Histórico de Tasa de Aciertos")

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
                            line=dict(color="#ff6a00", width=4, shape="spline"),
                            marker=dict(size=12, color="#ff6a00", line=dict(color="white", width=2)),
                            fill="tozeroy",
                            fillcolor="rgba(255, 106, 0, 0.1)",
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
                        st.markdown("### Aciertos por Nivel de Confianza")

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
                            "Baja (< 55%)": "#4a4a4a",
                            "Media (55% - 65%)": "#8a8a8a",
                            "Alta (65% - 75%)": "#ffb000",
                            "Muy Alta (≥ 75%)": "#ff6a00"
                        }
                        bar_colors = [colors_map.get(n, "#8a8a8a") for n in df_conf_grouped["Nivel"]]

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

                        grid_color = "rgba(255, 255, 255, 0.07)"
                        text_color = "#a3a3a3"

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
                        st.markdown("### Precisión por Equipo")

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
                            "#ff3b3b" if x < 50 else ("#3ddc84" if x >= 60 else "#ffb000") for x in df_show["Tasa (%)"]
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
                    st.info("Por favor, selecciona un rango de fechas válido (fecha inicio y fecha fin).")

            else:
                st.info("No hay datos históricos disponibles en la base de datos para mostrar los dashboards.")
    except Exception as e:
        st.error(f"Error al cargar los datos para los dashboards: {str(e)}")

    # ============================================================================
    # PÁGINA: ACERCA DEL MODELO
    # ============================================================================

elif pagina == "Playoffs":
    render_simulacion_playoffs(API_URL, render_custom_alert, get_team_logo_html)

elif pagina == "Acerca del modelo":
    st.markdown(f'<div class="about-hero">{LOGO_SVG}</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">MLB Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Predicción de partidos de la MLB con machine learning, entrenado con datos oficiales de la liga y reentrenado cada día.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Estado del Sistema")
    render_system_status_panel(api_ok, api_data)

    st.markdown("### Estadísticas Totales del Motor")
    render_motor_lifetime_panel()

    st.markdown(
        """
    <div class="stats-card">
        <h3>¿Qué es MLB Predictor?</h3>
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
        <div class="stats-card" style="border-left-color: #ff6a00;">
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
        <div class="stats-card" style="border-left-color: #ffb000;">
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
        <div class="stats-card" style="border-left-color: #3ddc84;">
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
            f'<span class="badge">{tech}</span>'
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
    Este sistema es una herramienta estadística para análisis y entretenimiento.
    El béisbol es inherentemente impredecible. No utilices estas predicciones para apuestas deportivas.
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
<div class="site-footer">
    <div><b>MLB Predictor</b> 2026. Modelo V4.0.</div>
    <div>Datos de la API oficial de la MLB. Actualización automática diaria.</div>
    <div>Herramienta estadística y de entretenimiento. No es asesoría de apuestas.</div>
</div>
""",
    unsafe_allow_html=True,
)
