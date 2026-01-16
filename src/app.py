"""
MLB Game Predictor V3.5 - VERSIÓN DEFINITIVA
Combina lo mejor de V3.0 (detalles) + nueva estructura profesional
Ejecutar: streamlit run app.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
import sys
import subprocess
import time
from PIL import Image
import unicodedata
import re

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

st.set_page_config(
    page_title="MLB Predictor V3.5",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Logos MLB oficiales
EQUIPOS_MLB = {
    'ARI': {'nombre': 'Arizona Diamondbacks', 'logo': 'https://www.mlbstatic.com/team-logos/109.svg', 'color': '#A71930'},
    'ATL': {'nombre': 'Atlanta Braves', 'logo': 'https://www.mlbstatic.com/team-logos/144.svg', 'color': '#CE1141'},
    'BAL': {'nombre': 'Baltimore Orioles', 'logo': 'https://www.mlbstatic.com/team-logos/110.svg', 'color': '#DF4601'},
    'BOS': {'nombre': 'Boston Red Sox', 'logo': 'https://www.mlbstatic.com/team-logos/111.svg', 'color': '#BD3039'},
    'CHC': {'nombre': 'Chicago Cubs', 'logo': 'https://www.mlbstatic.com/team-logos/112.svg', 'color': '#0E3386'},
    'CHW': {'nombre': 'Chicago White Sox', 'logo': 'https://www.mlbstatic.com/team-logos/145.svg', 'color': '#27251F'},
    'CIN': {'nombre': 'Cincinnati Reds', 'logo': 'https://www.mlbstatic.com/team-logos/113.svg', 'color': '#C6011F'},
    'CLE': {'nombre': 'Cleveland Guardians', 'logo': 'https://www.mlbstatic.com/team-logos/114.svg', 'color': '#0C2340'},
    'COL': {'nombre': 'Colorado Rockies', 'logo': 'https://www.mlbstatic.com/team-logos/115.svg', 'color': '#33006F'},
    'DET': {'nombre': 'Detroit Tigers', 'logo': 'https://www.mlbstatic.com/team-logos/116.svg', 'color': '#0C2340'},
    'HOU': {'nombre': 'Houston Astros', 'logo': 'https://www.mlbstatic.com/team-logos/117.svg', 'color': '#002D62'},
    'KCR': {'nombre': 'Kansas City Royals', 'logo': 'https://www.mlbstatic.com/team-logos/118.svg', 'color': '#004687'},
    'LAA': {'nombre': 'Los Angeles Angels', 'logo': 'https://www.mlbstatic.com/team-logos/108.svg', 'color': '#BA0021'},
    'LAD': {'nombre': 'Los Angeles Dodgers', 'logo': 'https://www.mlbstatic.com/team-logos/119.svg', 'color': '#005A9C'},
    'MIA': {'nombre': 'Miami Marlins', 'logo': 'https://www.mlbstatic.com/team-logos/146.svg', 'color': '#00A3E0'},
    'MIL': {'nombre': 'Milwaukee Brewers', 'logo': 'https://www.mlbstatic.com/team-logos/158.svg', 'color': '#12284B'},
    'MIN': {'nombre': 'Minnesota Twins', 'logo': 'https://www.mlbstatic.com/team-logos/142.svg', 'color': '#002B5C'},
    'NYM': {'nombre': 'New York Mets', 'logo': 'https://www.mlbstatic.com/team-logos/121.svg', 'color': '#002D72'},
    'NYY': {'nombre': 'New York Yankees', 'logo': 'https://www.mlbstatic.com/team-logos/147.svg', 'color': '#003087'},
    'OAK': {'nombre': 'Oakland Athletics', 'logo': 'https://www.mlbstatic.com/team-logos/133.svg', 'color': '#003831'},
    'PHI': {'nombre': 'Philadelphia Phillies', 'logo': 'https://www.mlbstatic.com/team-logos/143.svg', 'color': '#E81828'},
    'PIT': {'nombre': 'Pittsburgh Pirates', 'logo': 'https://www.mlbstatic.com/team-logos/134.svg', 'color': '#27251F'},
    'SDP': {'nombre': 'San Diego Padres', 'logo': 'https://www.mlbstatic.com/team-logos/135.svg', 'color': '#2F241D'},
    'SEA': {'nombre': 'Seattle Mariners', 'logo': 'https://www.mlbstatic.com/team-logos/136.svg', 'color': '#0C2C56'},
    'SFG': {'nombre': 'San Francisco Giants', 'logo': 'https://www.mlbstatic.com/team-logos/137.svg', 'color': '#FD5A1E'},
    'STL': {'nombre': 'St. Louis Cardinals', 'logo': 'https://www.mlbstatic.com/team-logos/138.svg', 'color': '#C41E3A'},
    'TBR': {'nombre': 'Tampa Bay Rays', 'logo': 'https://www.mlbstatic.com/team-logos/139.svg', 'color': '#092C5C'},
    'TEX': {'nombre': 'Texas Rangers', 'logo': 'https://www.mlbstatic.com/team-logos/140.svg', 'color': '#003278'},
    'TOR': {'nombre': 'Toronto Blue Jays', 'logo': 'https://www.mlbstatic.com/team-logos/141.svg', 'color': '#134A8E'},
    'WSN': {'nombre': 'Washington Nationals', 'logo': 'https://www.mlbstatic.com/team-logos/120.svg', 'color': '#AB0003'}
}

# ============================================================================
# ESTILOS CSS (Combinación de ambas versiones)
# ============================================================================

st.markdown("""
<style>
    /* Header Principal */
    .main-header {
        font-size: 3.2rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.3rem;
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Badges */
    .v3-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 2rem;
        font-size: 0.85rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(118, 75, 162, 0.3);
        display: inline-block;
    }

    /* Caja de Ganador */
    .winner-box {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        color: white;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Tarjetas de Lanzadores */
    .pitcher-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 0.75rem;
        color: #1e293b;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 8px solid #1f77b4;
    }
    
    .pitcher-name {
        font-size: 1.4rem;
        font-weight: bold;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }

    /* Super Features */
    .super-feature-box {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 1.2rem;
        border-radius: 0.8rem;
        color: white;
        margin: 0.5rem 0;
        min-height: 160px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Niveles de Confianza */
    .confidence-high { color: #10b981; font-weight: 900; font-size: 2.5rem; }
    .confidence-medium { color: #f59e0b; font-weight: 900; font-size: 2.5rem; }
    .confidence-low { color: #ef4444; font-weight: 900; font-size: 2.5rem; }

    /* Partido Card */
    .game-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-top: 4px solid #3b82f6;
        transition: transform 0.2s;
    }
    
    .game-card:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def normalizar_texto(texto):
    """Normaliza texto para comparaciones (igual que tu API)"""
    if not texto:
        return ""
    texto = str(texto).strip()
    texto = "".join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    texto = re.sub(r'[^a-zA-Z0-9\s]', '', texto)
    return texto

@st.cache_data(ttl=60)
def verificar_api_salud():
    """Verifica el estado de la API"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None

def get_team_logo_html(team_code, size=40):
    """Genera HTML para logo de equipo"""
    if team_code in EQUIPOS_MLB:
        logo_url = EQUIPOS_MLB[team_code]['logo']
        return f'<img src="{logo_url}" width="{size}" style="vertical-align: middle; margin-right: 10px; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));">'
    return ''

def get_team_display_name(team_code):
    """Obtiene nombre completo del equipo"""
    if team_code in EQUIPOS_MLB:
        return EQUIPOS_MLB[team_code]['nombre']
    return team_code

def ejecutar_scraper_manual():
    """Ejecuta el scraper diario manualmente"""
    try:
        with st.spinner("🔍 Buscando partidos del día..."):
            result = subprocess.run(
                [sys.executable, "mlb_daily_scraper.py"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                return True, "✅ Scraping completado exitosamente"
            else:
                return False, "⚠️ No se encontraron partidos para hoy"
    except subprocess.TimeoutExpired:
        return False, "❌ Timeout: El proceso tardó demasiado"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def hacer_prediccion_detallada(home_team, away_team, home_pitcher, away_pitcher, year):
    """Llama a /predict/detailed de tu API"""
    data = {
        "home_team": home_team,
        "away_team": away_team,
        "home_pitcher": home_pitcher,
        "away_pitcher": away_pitcher,
        "year": year
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict/detailed",
            json=data,
            timeout=120
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            error = response.json()
            return False, error.get('detail', 'Error desconocido')
    except requests.exceptions.Timeout:
        return False, "Timeout: La predicción tardó demasiado"
    except Exception as e:
        return False, f"Error: {str(e)}"

def crear_grafico_probabilidades(prob_home, prob_away, home_team, away_team):
    """Crea gráfico de barras de probabilidades"""
    home_name = get_team_display_name(home_team)
    away_name = get_team_display_name(away_team)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[prob_home * 100, prob_away * 100],
        y=[f'{home_team} (Local)', f'{away_team} (Visitante)'],
        orientation='h',
        marker=dict(
            color=['#3b82f6', '#ef4444'],
            line=dict(color='white', width=2)
        ),
        text=[f'{prob_home*100:.1f}%', f'{prob_away*100:.1f}%'],
        textposition='auto',
        textfont=dict(size=16, color='white', family='Arial Black')
    ))
    
    fig.update_layout(
        title="Probabilidades de Victoria",
        xaxis_title="Probabilidad (%)",
        yaxis_title="",
        height=300,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        xaxis=dict(range=[0, 100])
    )
    
    return fig

def crear_gauge_confianza(confianza):
    """Crea gauge de nivel de confianza"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confianza * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Nivel de Confianza", 'font': {'size': 24}},
        number={'suffix': "%", 'font': {'size': 36}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 55], 'color': '#fee2e2'},
                {'range': [55, 60], 'color': '#fed7aa'},
                {'range': [60, 70], 'color': '#d9f99d'},
                {'range': [70, 100], 'color': '#bbf7d0'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "gray", 'family': "Arial"}
    )
    
    return fig

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.image("https://www.mlbstatic.com/team-logos/league-on-dark/1.svg", width=120)
    st.title("MLB Predictor")
    st.markdown('''
    <div style="text-align: center; margin: 0.5rem 0;">
        <span class="v3-badge">MODEL V3.5</span>
        <span class="badge-live" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 0.4rem 1.2rem; border-radius: 2rem; font-size: 0.85rem; font-weight: bold; box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4); display: inline-block; margin-left: 0.5rem;">LIVE</span>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Estado del Sistema")
    api_ok, api_data = verificar_api_salud()
    
    if api_ok:
        st.success("✅ API Conectada")
        st.success("✅ Modelo V3.5 Cargado")
        st.info("🎯 Super Features Activas")
    else:
        st.error("❌ API No Disponible")
        st.warning("Inicia la API:")
        st.code("python api_hybrid_v3.py", language="python")
    
    st.markdown("---")
    
    st.subheader("🧭 Navegación")
    pagina = st.radio(
        "Ir a:",
        ["🎯 Predicción Manual", "📅 Partidos de Hoy", "📊 Comparación & Historial", "ℹ️ Acerca de"],
        label_visibility="collapsed"
    )

# ============================================================================
# PÁGINA: PREDICCIÓN MANUAL
# ============================================================================

if pagina == "🎯 Predicción Manual":
    st.markdown('<div class="main-header">⚾ MLB Game Predictor</div>', unsafe_allow_html=True)
    
    if not api_ok:
        st.error("❌ La API V3 no está disponible")
        st.stop()

    st.subheader("📋 Datos del Partido")
    col_input1, col_input2 = st.columns(2)

    team_options = [f"{code} - {EQUIPOS_MLB[code]['nombre']}" for code in sorted(EQUIPOS_MLB.keys())]

    with col_input1:
        st.markdown("#### 🏠 Equipo Local")
        home_sel = st.selectbox("Selecciona equipo local", team_options, key="home_display_v3")
        home_team = home_sel.split(" - ")[0]
        st.markdown(f'<div style="text-align: center; padding:10px;">{get_team_logo_html(home_team, 80)}</div>', unsafe_allow_html=True)

    with col_input2:
        st.markdown("#### ✈️ Equipo Visitante")
        away_sel = st.selectbox("Selecciona equipo visitante", team_options, key="away_display_v3")
        away_team = away_sel.split(" - ")[0]
        st.markdown(f'<div style="text-align: center; padding:10px;">{get_team_logo_html(away_team, 80)}</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            home_pitcher = st.text_input("Lanzador Local", placeholder="Ej: Gerrit Cole, Bello...", key="hp_input_v3")
        with col_p2:
            away_pitcher = st.text_input("Lanzador Visitante", placeholder="Ej: Webb, Bieber...", key="ap_input_v3")
    
        year = st.number_input("Temporada", min_value=2020, max_value=2030, value=2026)
        submit_button = st.form_submit_button("🚀 Realizar Predicción", use_container_width=True, type="primary")

    if submit_button:
        if not home_pitcher or not away_pitcher:
            st.error("❌ Por favor ingresa los nombres de ambos lanzadores")
        elif home_team == away_team:
            st.error("❌ Los equipos deben ser diferentes")
        else:
            h_clean = normalizar_texto(home_pitcher)
            a_clean = normalizar_texto(away_pitcher)
            
            with st.spinner(f"🔍 Analizando: {home_team} vs {away_team}..."):
                try:
                    exito, resultado = hacer_prediccion_detallada(
                        home_team, away_team, h_clean, a_clean, year
                    )
                    
                    if not exito or not resultado.get('stats_detalladas'):
                        st.error("❌ Lanzador/es no encontrado/s. Verifique el nombre e intente de nuevo")
                        st.stop()
                    
                    st.success("✅ Predicción realizada exitosamente!")
                    
                    st.markdown("---")
                    st.markdown("## 🏆 Resultado")
                    
                    ganador = resultado.get('ganador')
                    prob_home = resultado.get('prob_home', 0)
                    prob_away = resultado.get('prob_away', 0)
                    confianza = resultado.get('confianza', 0)
                    
                    ganador_nombre = get_team_display_name(ganador)
                    ganador_logo = get_team_logo_html(ganador, 60)
                    
                    st.markdown(f"""
                    <div class="winner-box">
                        <h1 style="margin:0; font-size: 2.5rem;">GANADOR PREDICHO</h1>
                        <div style="margin:1rem 0;">
                            {ganador_logo}
                            <h2 style="display:inline; margin:0; font-size: 3rem; vertical-align: middle;">{ganador_nombre}</h2>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(
                            get_team_logo_html(home_team, 30) + f"**Probabilidad de {home_team}**",
                            unsafe_allow_html=True
                        )
                        st.metric("", f"{prob_home:.1f}%", delta=f"{(prob_home-50):+.1f}% vs 50%")
                    
                    with col2:
                        st.markdown(
                            get_team_logo_html(away_team, 30) + f"**Probabilidad de {away_team}**",
                            unsafe_allow_html=True
                        )
                        st.metric("", f"{prob_away:.1f}%", delta=f"{(prob_away-50):+.1f}% vs 50%")
                    
                    with col3:
                        if confianza > 0.70:
                            conf_class = "confidence-high"
                            conf_text = "MUY ALTA"
                        elif confianza > 0.60:
                            conf_class = "confidence-medium"
                            conf_text = "ALTA"
                        elif confianza > 0.55:
                            conf_class = "confidence-medium"
                            conf_text = "MODERADA"
                        else:
                            conf_class = "confidence-low"
                            conf_text = "BAJA"
                        
                        st.metric("Confianza", f"{confianza*100:.1f}%")
                        st.markdown(f'<p class="{conf_class}">{conf_text}</p>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.plotly_chart(
                            crear_grafico_probabilidades(prob_home, prob_away, home_team, away_team),
                            use_container_width=True
                        )
                    
                    with col2:
                        st.plotly_chart(
                            crear_gauge_confianza(confianza),
                            use_container_width=True
                        )
                    
                    # SUPER FEATURES
                    features = resultado.get('features_usadas', {})
                    stats_det = resultado.get('stats_detalladas', {})
                    
                    name_h = stats_det.get('home_pitcher', {}).get('nombre', home_pitcher) if stats_det.get('home_pitcher') else home_pitcher
                    name_a = stats_det.get('away_pitcher', {}).get('nombre', away_pitcher) if stats_det.get('away_pitcher') else away_pitcher

                    st.markdown("---")
                    st.markdown("## 🚀 Super Features: Análisis Dinámico")
                    st.info("Comparativa de dominio directo basada en el cruce de estadísticas de lanzadores vs. bateadores.")

                    col1, col2, col3 = st.columns(3)

                    # 1. Neutralización
                    neut = features.get('super_neutralizacion_whip_ops', 0)
                    ventaja_n = home_team if neut < 0 else away_team
                    pitcher_n = name_h if neut < 0 else name_a
                    rival_n = away_team if neut < 0 else home_team
                    pct_n = abs(neut) * 100

                    with col1:
                        st.markdown(f"""
                        <div class="super-feature-box">
                            <h4>🛡️ Neutralización</h4>
                            <p style="font-size: 1.1rem; color: #fff; font-weight: bold;">Ventaja {ventaja_n}</p>
                            <p style="font-size: 0.9rem;"><b>{pitcher_n}</b> controla el bateo de {rival_n} con una eficiencia del <b>{pct_n:.1f}%</b> superior al promedio.</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # 2. Resistencia
                    res = features.get('super_resistencia_era_ops', 0)
                    ventaja_r = home_team if res < 0 else away_team
                    pitcher_r = name_h if res < 0 else name_a
                    pct_r = min(abs(res) * 10, 100.0)

                    with col2:
                        st.markdown(f"""
                        <div class="super-feature-box">
                            <h4>💪 Resistencia</h4>
                            <p style="font-size: 1.1rem; color: #fff; font-weight: bold;">Ventaja {ventaja_r}</p>
                            <p style="font-size: 0.9rem;"><b>{pitcher_r}</b> es un <b>{pct_r:.1f}%</b> más sólido ante el poder (OPS) de los bateadores rivales.</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # 3. Muro Bullpen
                    muro = features.get('super_muro_bullpen', 0)
                    ventaja_m = home_team if muro < 0 else away_team
                    pct_m = abs(muro) * 100

                    with col3:
                        st.markdown(f"""
                        <div class="super-feature-box">
                            <h4>🧱 Muro Bullpen</h4>
                            <p style="font-size: 1.1rem; color: #fff; font-weight: bold;">Ventaja {ventaja_m}</p>
                            <p style="font-size: 0.9rem;">El relevo de {ventaja_m} domina a los mejores bateadores rivales en un <b>{pct_m:.1f}%</b>.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # ESTADÍSTICAS DETALLADAS
                    if stats_det:
                        st.markdown("---")
                        st.markdown("## 📊 Estadísticas Detalladas")
                        
                        st.markdown("### 🎯 Lanzadores Iniciales")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            home_pitcher_stats = stats_det.get('home_pitcher')
                            if home_pitcher_stats:
                                st.markdown(
                                    get_team_logo_html(home_team, 40) + 
                                    f"🏠 {home_team} - {home_pitcher_stats.get('nombre', home_pitcher)}",
                                    unsafe_allow_html=True
                                )
                                
                                subcol1, subcol2, subcol3 = st.columns(3)
                                with subcol1:
                                    st.metric("ERA", f"{home_pitcher_stats.get('ERA', 0):.2f}")
                                with subcol2:
                                    st.metric("WHIP", f"{home_pitcher_stats.get('WHIP', 0):.3f}")
                                with subcol3:
                                    st.metric("SO9", f"{home_pitcher_stats.get('SO9', 0):.2f}")
                            else:
                                st.warning(f"⚠️ No se encontró {home_pitcher}")
                        
                        with col2:
                            away_pitcher_stats = stats_det.get('away_pitcher')
                            if away_pitcher_stats:
                                st.markdown(
                                    get_team_logo_html(away_team, 40) + 
                                    f"✈️ {away_team} - {away_pitcher_stats.get('nombre', away_pitcher)}",
                                    unsafe_allow_html=True
                                )
                                
                                subcol1, subcol2, subcol3 = st.columns(3)
                                with subcol1:
                                    st.metric("ERA", f"{away_pitcher_stats.get('ERA', 0):.2f}")
                                with subcol2:
                                    st.metric("WHIP", f"{away_pitcher_stats.get('WHIP', 0):.3f}")
                                with subcol3:
                                    st.metric("SO9", f"{away_pitcher_stats.get('SO9', 0):.2f}")
                            else:
                                st.warning(f"⚠️ No se encontró {away_pitcher}")
                        
                        st.markdown("---")
                        st.markdown("### 🔥 Top 3 Bateadores")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            home_batters = stats_det.get('home_batters', [])
                            if home_batters:
                                st.markdown(
                                    get_team_logo_html(home_team, 40) + f"🏠 {home_team}",
                                    unsafe_allow_html=True
                                )
                                for i, batter in enumerate(home_batters, 1):
                                    with st.expander(f"#{i} - {batter.get('nombre', 'N/A')}", expanded=(i==1)):
                                        subcol1, subcol2, subcol3, subcol4 = st.columns(4)
                                        with subcol1:
                                            st.metric("OPS", f"{batter.get('OPS', 0):.3f}")
                                        with subcol2:
                                            st.metric("BA", f"{batter.get('BA', 0):.3f}")
                                        with subcol3:
                                            st.metric("HR", int(batter.get('HR', 0)))
                                        with subcol4:
                                            st.metric("RBI", int(batter.get('RBI', 0)))
                            else:
                                st.warning(f"⚠️ No se encontraron bateadores")
                        
                        with col2:
                            away_batters = stats_det.get('away_batters', [])
                            if away_batters:
                                st.markdown(
                                    get_team_logo_html(away_team, 40) + f"✈️ {away_team}",
                                    unsafe_allow_html=True
                                )
                                for i, batter in enumerate(away_batters, 1):
                                    with st.expander(f"#{i} - {batter.get('nombre', 'N/A')}", expanded=(i==1)):
                                        subcol1, subcol2, subcol3, subcol4 = st.columns(4)
                                        with subcol1:
                                            st.metric("OPS", f"{batter.get('OPS', 0):.3f}")
                                        with subcol2:
                                            st.metric("BA", f"{batter.get('BA', 0):.3f}")
                                        with subcol3:
                                            st.metric("HR", int(batter.get('HR', 0)))
                                        with subcol4:
                                            st.metric("RBI", int(batter.get('RBI', 0)))
                            else:
                                st.warning(f"⚠️ No se encontraron bateadores")
                    
                    # Mensaje si se usó año diferente
                    if resultado.get('mensaje'):
                        st.info(f"ℹ️ {resultado.get('mensaje')}")
                    
                    # Botón para descargar JSON
                    st.markdown("---")
                    result_json = json.dumps(resultado, indent=2)
                    st.download_button(
                        label="📥 Descargar Reporte Técnico (JSON)",
                        data=result_json,
                        file_name=f"prediccion_{home_team}_{away_team}_{year}.json",
                        mime="application/json",
                    )
                except Exception as e:
                    st.error("❌ Error al procesar la predicción. Verifique los datos e intente nuevamente.")
                    st.exception(e)

# ============================================================================
# PÁGINA: PARTIDOS DE HOY
# ============================================================================

elif pagina == "📅 Partidos de Hoy":
    st.markdown('<div class="main-header">📅 Partidos y Predicciones del Día</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predicciones automáticas basadas en lineups scrapeados</div>', unsafe_allow_html=True)
    
    if not api_ok:
        st.error("❌ La API no está disponible")
        st.stop()
    
    # Intentar obtener datos de la API
    try:
        response_partidos = requests.get(f"{API_URL}/games/today", timeout=10)
        response_predicciones = requests.get(f"{API_URL}/predictions/today", timeout=10)
        
        partidos = response_partidos.json() if response_partidos.status_code == 200 else []
        predicciones = response_predicciones.json() if response_predicciones.status_code == 200 else []
        
        # Combinar datos
        if partidos and predicciones:
            pred_dict = {p['game_id']: p for p in predicciones}
            for partido in partidos:
                if partido['game_id'] in pred_dict:
                    partido.update(pred_dict[partido['game_id']])
        
        if not partidos:
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; color: #64748b;">
                <div style="font-size: 5rem; margin-bottom: 1rem;">⚾</div>
                <h2>No hay partidos programados para hoy</h2>
                <p style="font-size: 1.1rem; margin: 1rem 0;">
                    Los partidos aún no han sido scrapeados o no hay juegos programados
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Botón para ejecutar scraper manualmente
            if st.button("🔄 Buscar Partidos Manualmente", type="primary", use_container_width=True):
                exito, mensaje = ejecutar_scraper_manual()
                if exito:
                    st.success(mensaje)
                    st.rerun()
                else:
                    st.warning(mensaje)
        else:
            st.success(f"✅ Se encontraron {len(partidos)} partidos para hoy")
            
            # Mostrar cada partido
            for partido in partidos:
                with st.container():
                    home_logo = get_team_logo_html(partido['home_team'], 50)
                    away_logo = get_team_logo_html(partido['away_team'], 50)
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="game-card">
                            <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                                {away_logo}{partido['away_team']} @ {home_logo}{partido['home_team']}
                            </div>
                            <div style="color: #64748b; margin: 0.5rem 0;">
                                🎯 {partido.get('away_pitcher', 'TBD')} vs {partido.get('home_pitcher', 'TBD')}
                            </div>
                        """, unsafe_allow_html=True)
                        
                        if 'prediccion' in partido:
                            pred_team = get_team_display_name(partido['prediccion'])
                            prob = partido.get('prob_home', 0) if partido['prediccion'] == partido['home_team'] else partido.get('prob_away', 0)
                            
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); padding: 0.75rem; border-radius: 0.5rem; margin: 0.5rem 0; font-weight: 600;">
                                🏆 Predicho: <strong>{pred_team}</strong> ({prob*100:.1f}%)
                            </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col2:
                        if 'prediccion' in partido:
                            prob = partido.get('prob_home', 0) if partido['prediccion'] == partido['home_team'] else partido.get('prob_away', 0)
                            
                            st.metric(
                                "Confianza",
                                f"{prob*100:.1f}%",
                                delta=partido.get('confianza', 'N/A')
                            )
                        else:
                            st.warning("⏳ Pendiente")
                    
                    st.markdown("---")
    except Exception as e:
        st.error(f"❌ Error obteniendo datos: {str(e)}")

# ============================================================================
# PÁGINA: COMPARACIÓN & HISTORIAL
# ============================================================================

elif pagina == "📊 Comparación & Historial":
    st.markdown('<div class="main-header">📊 Comparación de Predicciones vs Resultados</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analiza el rendimiento histórico del modelo</div>', unsafe_allow_html=True)
    
    if not api_ok:
        st.error("❌ La API no está disponible")
        st.stop()
    
    # Selector de fecha
    st.markdown("### 📅 Selecciona una Fecha")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        fecha_seleccionada = st.date_input(
            "Fecha",
            value=datetime.now() - timedelta(days=1),
            max_value=datetime.now()
        )
    
    with col2:
        if st.button("🔍 Analizar Fecha", type="primary", use_container_width=True):
            st.session_state.fecha_analizar = fecha_seleccionada
    
    with col3:
        if st.button("📊 Ver Estadísticas", use_container_width=True):
            try:
                response = requests.get(f"{API_URL}/stats/accuracy?dias=30")
                if response.status_code == 200:
                    stats = response.json()
                    st.json(stats)
            except:
                st.error("Error obteniendo estadísticas")
    
    # Mostrar comparación si hay fecha seleccionada
    if 'fecha_analizar' in st.session_state:
        fecha_str = st.session_state.fecha_analizar.strftime("%Y-%m-%d")
        
        with st.spinner(f"Cargando datos para {fecha_str}..."):
            try:
                response = requests.get(f"{API_URL}/compare/{fecha_str}")
                comparacion = response.json() if response.status_code == 200 else None
            except:
                comparacion = None
        
        if comparacion and comparacion['partidos']:
            partidos = comparacion['partidos']
            stats = comparacion['estadisticas']
            
            # Métricas generales
            st.markdown("### 📈 Rendimiento del Día")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Partidos", stats['total'])
            with col2:
                st.metric("Aciertos", stats['aciertos'])
            with col3:
                st.metric("Accuracy", f"{stats['accuracy']:.1f}%")
            with col4:
                errores = stats['total'] - stats['aciertos']
                st.metric("Errores", errores)
            
            # Tabla de resultados
            st.markdown("### 🎯 Resultados Detallados")
            
            for partido in partidos:
                acierto = partido.get('acierto', 0)
                
                with st.expander(
                    f"{'✅' if acierto else '❌'} {partido['away_team']} @ {partido['home_team']} - {partido['score_away']}-{partido['score_home']}",
                    expanded=False
                ):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**🎯 Predicción**")
                        st.write(f"Ganador: {partido.get('prediccion', 'N/A')}")
                        st.write(f"Prob Home: {partido.get('prob_home', 0)*100:.1f}%")
                        st.write(f"Prob Away: {partido.get('prob_away', 0)*100:.1f}%")
                    
                    with col2:
                        st.markdown("**📊 Resultado Real**")
                        ganador_real = partido['home_team'] if partido['ganador_real'] == 1 else partido['away_team']
                        st.write(f"Ganador: {ganador_real}")
                        st.write(f"Score: {partido['score_away']}-{partido['score_home']}")
                    
                    with col3:
                        st.markdown("**✓ Verificación**")
                        if acierto:
                            st.success("✅ ACIERTO")
                        else:
                            st.error("❌ ERROR")
                        st.write(f"Confianza: {partido.get('confianza', 'N/A')}")
        else:
            st.info(f"📭 No hay datos disponibles para {fecha_str}")

# ============================================================================
# PÁGINA: ACERCA DE
# ============================================================================

elif pagina == "ℹ️ Acerca de":
    col1, col2, col3 = st.columns([1, 0.8, 1])
    with col2:
        st.image("https://www.mlbstatic.com/team-logos/league-on-dark/1.svg", use_container_width=True)
    
    st.markdown('<div class="main-header">Sobre el Modelo V3.5</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Sistema Híbrido de Predicción Estadística</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background: white; padding: 1.75rem; border-radius: 1rem; box-shadow: 0 4px 20px rgba(0,0,0,0.08); border-left: 5px solid #3b82f6; margin: 1rem 0;">
        <h3>🎯 ¿Qué es MLB Game Predictor?</h3>
        <p style="font-size: 1.1rem; line-height: 1.8;">
            Sistema de predicción de partidos MLB usando <b>Machine Learning con Modelo Híbrido</b> 
            entrenado con datos históricos y en tiempo real para analizar estadísticas de equipos y jugadores,
            enfocándose en la ventaja competitiva de los <b>matchups directos</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 1.75rem; border-radius: 1rem; box-shadow: 0 4px 20px rgba(0,0,0,0.08); margin: 1rem 0;">
            <h4>🚀 Novedades V3.5</h4>
            <ul style="font-size: 1.05rem; line-height: 1.8;">
                <li><b>XGBoost Optimizado</b>: 20+ años de datos</li>
                <li><b>SelectKBest Features</b>: 26 variables clave</li>
                <li><b>Dynamic Scraping</b>: Baseball-Reference en vivo</li>
                <li><b>Modelo Híbrido</b>: Temporal + Real-time</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: white; padding: 1.75rem; border-radius: 1rem; box-shadow: 0 4px 20px rgba(0,0,0,0.08); margin: 1rem 0;">
            <h4>📊 Características</h4>
            <ul style="font-size: 1.05rem; line-height: 1.8;">
                <li>Análisis de lanzadores y bateadores</li>
                <li>Logos oficiales MLB</li>
                <li>Métricas de confianza por partido</li>
                <li>Interfaz profesional optimizada</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🎯 Super Features Explicadas")
    sf1, sf2, sf3 = st.columns(3)
    
    with sf1:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 0.75rem; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin: 0.5rem 0;">
            <b>🛡️ Neutralización</b><br>
            <small>Cruza el WHIP del lanzador con el OPS rival.</small>
        </div>
        """, unsafe_allow_html=True)
    
    with sf2:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 0.75rem; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin: 0.5rem 0;">
            <b>💪 Resistencia</b><br>
            <small>Entradas de calidad frente al poder del lineup.</small>
        </div>
        """, unsafe_allow_html=True)
    
    with sf3:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 0.75rem; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin: 0.5rem 0;">
            <b>🧱 Muro Bullpen</b><br>
            <small>Efectividad del relevo vs bateadores de cierre.</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🛠️ Stack Tecnológico")
    tech_badges = ["FastAPI", "XGBoost", "Scikit-learn", "Streamlit", "BeautifulSoup", "Plotly"]
    badges_html = " ".join([
        f'<span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.5rem 1rem; border-radius: 2rem; font-size: 0.875rem; font-weight: 700; margin: 0.25rem; display: inline-block;">{tech}</span>' 
        for tech in tech_badges
    ])
    st.markdown(f'<div style="text-align: center; margin: 2rem 0;">{badges_html}</div>', unsafe_allow_html=True)

    st.info("⚠️ **Disclaimer**: Este sistema es una herramienta estadística para análisis. El béisbol es impredecible, ¡disfruta el juego!")

    # Equipos MLB
    st.markdown("### ⚾ Equipos de la MLB")
    cols = st.columns(6)
    teams_sorted = sorted(EQUIPOS_MLB.items(), key=lambda x: x[1]['nombre'])
    
    for i, (code, info) in enumerate(teams_sorted):
        with cols[i % 6]:
            st.image(info['logo'], width=60)
            st.caption(f"**{code}**")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    f"<center><p style='color: #666;'>MLB Predictor V3.5 | 2026 | Datos: Baseball-Reference</p></center>", 
    unsafe_allow_html=True
)