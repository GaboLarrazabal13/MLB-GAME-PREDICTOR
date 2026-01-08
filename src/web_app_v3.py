"""
MLB Game Predictor - Web App V3 con Streamlit
MODELO V3: Super Features + Bullpen + 60.70% Accuracy
Ejecutar: streamlit run web_app_v3.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import unicodedata
import re
from PIL import Image

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, "logo.png")

try:
    favicon = Image.open(logo_path)
except FileNotFoundError:
    favicon = "⚾"

st.set_page_config(
    page_title="MLB Predictor V3",
    page_icon=favicon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DATOS
# ============================================================================

API_URL = st.secrets.get("API_URL", "https://mlb-game-predictor.onrender.com")

EQUIPOS_MLB = {
    'ARI': {'nombre': 'Arizona Diamondbacks', 'logo': 'https://www.mlbstatic.com/team-logos/109.svg'},
    'ATL': {'nombre': 'Atlanta Braves', 'logo': 'https://www.mlbstatic.com/team-logos/144.svg'},
    'BAL': {'nombre': 'Baltimore Orioles', 'logo': 'https://www.mlbstatic.com/team-logos/110.svg'},
    'BOS': {'nombre': 'Boston Red Sox', 'logo': 'https://www.mlbstatic.com/team-logos/111.svg'},
    'CHC': {'nombre': 'Chicago Cubs', 'logo': 'https://www.mlbstatic.com/team-logos/112.svg'},
    'CHW': {'nombre': 'Chicago White Sox', 'logo': 'https://www.mlbstatic.com/team-logos/145.svg'},
    'CIN': {'nombre': 'Cincinnati Reds', 'logo': 'https://www.mlbstatic.com/team-logos/113.svg'},
    'CLE': {'nombre': 'Cleveland Guardians', 'logo': 'https://www.mlbstatic.com/team-logos/114.svg'},
    'COL': {'nombre': 'Colorado Rockies', 'logo': 'https://www.mlbstatic.com/team-logos/115.svg'},
    'DET': {'nombre': 'Detroit Tigers', 'logo': 'https://www.mlbstatic.com/team-logos/116.svg'},
    'HOU': {'nombre': 'Houston Astros', 'logo': 'https://www.mlbstatic.com/team-logos/117.svg'},
    'KCR': {'nombre': 'Kansas City Royals', 'logo': 'https://www.mlbstatic.com/team-logos/118.svg'},
    'LAA': {'nombre': 'Los Angeles Angels', 'logo': 'https://www.mlbstatic.com/team-logos/108.svg'},
    'LAD': {'nombre': 'Los Angeles Dodgers', 'logo': 'https://www.mlbstatic.com/team-logos/119.svg'},
    'MIA': {'nombre': 'Miami Marlins', 'logo': 'https://www.mlbstatic.com/team-logos/146.svg'},
    'MIL': {'nombre': 'Milwaukee Brewers', 'logo': 'https://www.mlbstatic.com/team-logos/158.svg'},
    'MIN': {'nombre': 'Minnesota Twins', 'logo': 'https://www.mlbstatic.com/team-logos/142.svg'},
    'NYM': {'nombre': 'New York Mets', 'logo': 'https://www.mlbstatic.com/team-logos/121.svg'},
    'NYY': {'nombre': 'New York Yankees', 'logo': 'https://www.mlbstatic.com/team-logos/147.svg'},
    'OAK': {'nombre': 'Oakland Athletics', 'logo': 'https://www.mlbstatic.com/team-logos/133.svg'},
    'PHI': {'nombre': 'Philadelphia Phillies', 'logo': 'https://www.mlbstatic.com/team-logos/143.svg'},
    'PIT': {'nombre': 'Pittsburgh Pirates', 'logo': 'https://www.mlbstatic.com/team-logos/134.svg'},
    'SDP': {'nombre': 'San Diego Padres', 'logo': 'https://www.mlbstatic.com/team-logos/135.svg'},
    'SEA': {'nombre': 'Seattle Mariners', 'logo': 'https://www.mlbstatic.com/team-logos/136.svg'},
    'SFG': {'nombre': 'San Francisco Giants', 'logo': 'https://www.mlbstatic.com/team-logos/137.svg'},
    'STL': {'nombre': 'St. Louis Cardinals', 'logo': 'https://www.mlbstatic.com/team-logos/138.svg'},
    'TBR': {'nombre': 'Tampa Bay Rays', 'logo': 'https://www.mlbstatic.com/team-logos/139.svg'},
    'TEX': {'nombre': 'Texas Rangers', 'logo': 'https://www.mlbstatic.com/team-logos/140.svg'},
    'TOR': {'nombre': 'Toronto Blue Jays', 'logo': 'https://www.mlbstatic.com/team-logos/141.svg'},
    'WSN': {'nombre': 'Washington Nationals', 'logo': 'https://www.mlbstatic.com/team-logos/120.svg'}
}

EQUIPOS_CODES = list(EQUIPOS_MLB.keys())

# ============================================================================
# ESTILOS CSS
# ============================================================================

st.markdown("""
<style>
    /* Estilo Global y Header */
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
    
    /* Badges y Etiquetas */
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

    /* Caja de Predicción Ganadora (Rescatado de V2) */
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

    /* Tarjetas de Pitchers (Mejorado de V2) */
    .pitcher-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 0.75rem;
        color: #1e293b;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 8px solid #1f77b4; /* Color distintivo */
    }
    
    .pitcher-name {
        font-size: 1.4rem;
        font-weight: bold;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }

    /* Filas de Bateadores (Rescatado de V2) */
    .batter-row {
        background: #ffffff;
        padding: 0.8rem;
        border-radius: 0.6rem;
        margin: 0.5rem 0;
        border-left: 5px solid #6366f1;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Super Features (Estilo V3 mejorado) */
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
    .confidence-high { color: #10b981; font-weight: 900; font-size: 3rem; }
    .confidence-medium { color: #f59e0b; font-weight: 900; font-size: 3rem; }
    .confidence-low { color: #ef4444; font-weight: 900; font-size: 3rem; }

    /* Logos de equipos */
    .team-logo-inline {
        vertical-align: middle;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================
def normalizar_texto(texto):
    if not texto:
        return ""
    # 1. Convertimos a string y quitamos espacios extra a los lados
    texto = str(texto).strip()
    # 2. Quitamos acentos (Ej: Rodón -> Rodon)
    texto = "".join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    # 3. Mantenemos letras, números y ESPACIOS (importante para nombres completos)
    # Eliminamos cualquier otro carácter especial extraño
    texto = re.sub(r'[^a-zA-Z0-9\s]', '', texto)
    
    return texto

def get_team_logo_html(team_code, size=30):
    if team_code in EQUIPOS_MLB:
        logo_url = EQUIPOS_MLB[team_code]['logo']
        return f'<img src="{logo_url}" width="{size}" style="vertical-align: middle; margin-right: 8px;">'
    return ''

def get_team_display_name(team_code):
    if team_code in EQUIPOS_MLB:
        return EQUIPOS_MLB[team_code]['nombre']
    return team_code

@st.cache_data(ttl=300)
def verificar_api():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('model_loaded', False), data
        return False, None
    except:
        return False, None

@st.cache_data(ttl=300)
def obtener_info_modelo() -> dict | None:
    try:
        response = requests.get(f"{API_URL}/info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def hacer_prediccion_detallada(home_team, away_team, home_pitcher, away_pitcher, year):
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
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
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

def guardar_prediccion_local(home_team, away_team, home_pitcher, away_pitcher, year, resultado):
    if 'historial' not in st.session_state:
        st.session_state.historial = []
    
    # Validación de seguridad: Si resultado no es lo esperado, cancelamos el guardado
    if not isinstance(resultado, dict):
        return

    # Extraer estadísticas de forma segura con .get() y validación de tipos
    stats = resultado.get('stats_detalladas', {})
    if stats is None: stats = {}

    # Si el nombre del lanzador no se encontró en la API, usamos el nombre original del input
    h_info = stats.get('home_pitcher', {})
    h_real_name = h_info.get('nombre', home_pitcher) if isinstance(h_info, dict) else home_pitcher
    
    a_info = stats.get('away_pitcher', {})
    a_real_name = a_info.get('nombre', away_pitcher) if isinstance(a_info, dict) else away_pitcher
    
    prediccion = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'home_team': home_team,
        'away_team': away_team,
        'home_pitcher': h_real_name,
        'away_pitcher': a_real_name,
        'year': year,
        'ganador': resultado.get('ganador'),
        'prob_home': resultado.get('prob_home'),
        'prob_away': resultado.get('prob_away'),
        'confianza': resultado.get('confianza')
    }
    
    st.session_state.historial.insert(0, prediccion)
    if len(st.session_state.historial) > 50:
        st.session_state.historial = st.session_state.historial[:50]

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.image("https://www.mlbstatic.com/team-logos/league-on-dark/1.svg", width=120)
    st.title("MLB Predictor")
    st.markdown('<span class="v3-badge">MODEL V3.0</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Estado del Sistema")
    
    api_ok, api_data = verificar_api()
    
    if api_ok:
        st.success("✅ API Conectada")
        st.success("✅ Modelo V3 Cargado")
        
        st.info(" Accuracy: 60.70%")
        
        info = obtener_info_modelo()
        if info:
            st.markdown("---")
            st.subheader("Información del Modelo")
            st.metric("Versión", "3.0.0")
            st.caption(" Super Features")
            st.caption(" Bullpen Tracking")
            st.caption(" XGBoost Optimizado")
    else:
        st.error("❌ API No Disponible")
        st.warning("Inicia la API V3:")
        st.code("python api_hybrid_v3.py")
    
    st.markdown("---")
    
    st.subheader("Configuración")
    nueva_url = st.text_input("URL de la API", value=API_URL)
    if nueva_url != API_URL:
        if st.button("Actualizar URL"):
            API_URL = nueva_url
            st.rerun()
    
    st.markdown("---")
    
    st.subheader(" Navegación")
    pagina = st.radio(
        "Ir a:",
        [" Predictor", " Historial", " Acerca de"],
        label_visibility="collapsed"
    )

# ============================================================================
# PÁGINA PRINCIPAL - PREDICTOR
# ============================================================================

if pagina == " Predictor":
    
    st.markdown('<div class="main-header"> MLB Game Predictor</div>', unsafe_allow_html=True)
    
    if not api_ok:
        st.error(" La API V3 no está disponible")
        st.stop()

    # --- PASO 1: SELECCIÓN DE EQUIPOS (Fuera del form para actualización instantánea) ---
    st.subheader(" Datos del Partido")
    col_input1, col_input2 = st.columns(2)

    # Mantenemos home_team y away_team como nombres de variables principales
    home_team_options = [f"{code} - {EQUIPOS_MLB[code]['nombre']}" for code in EQUIPOS_CODES]

    with col_input1:
        st.markdown("####  Equipo Local")
        # Usamos un nombre temporal para el selectbox pero asignamos el valor a tu variable original
        home_sel = st.selectbox("Selecciona equipo local", home_team_options, key="home_display_v3")
        home_team = home_sel.split(" - ")[0] 
    
        # Muestra el logo inmediatamente al cambiar el selectbox
        st.markdown(f'<div style="text-align: center; padding:10px;">{get_team_logo_html(home_team, 80)}</div>', unsafe_allow_html=True)

    with col_input2:
        st.markdown("####  Equipo Visitante")
        away_sel = st.selectbox("Selecciona equipo visitante", home_team_options, key="away_display_v3")
        away_team = away_sel.split(" - ")[0]
    
        # Muestra el logo inmediatamente al cambiar el selectbox
        st.markdown(f'<div style="text-align: center; padding:10px;">{get_team_logo_html(away_team, 80)}</div>', unsafe_allow_html=True)

    # --- PASO 2: LANZADORES (Dentro del form para no disparar la API al escribir) ---
    with st.form("prediction_form"):
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            # Usamos tu variable original: home_pitcher
            home_pitcher = st.text_input("Lanzador Local", placeholder="Ej: Bello, Kershaw...", key="hp_input_v3")
        with col_p2:
            # Usamos tu variable original: away_pitcher
            away_pitcher = st.text_input("Lanzador Visitante", placeholder="Ej: Webb, Cole...", key="ap_input_v3")
    
        year = st.number_input("Temporada", min_value=2020, max_value=2030, value=2026)
        submit_button = st.form_submit_button(" Realizar Predicción", use_container_width=True, type="primary")

    # --- PASO 3: PROCESAMIENTO ---
    # --- PASO 3: PROCESAMIENTO ---
    if submit_button:
        if not home_pitcher or not away_pitcher:
            st.error(" Por favor ingresa los nombres de ambos lanzadores")
        elif home_team == away_team:
            st.error(" Los equipos deben ser diferentes")
        else:
            # Llamamos a la función ya corregida
            h_clean = normalizar_texto(home_pitcher)
            a_clean = normalizar_texto(away_pitcher)
            
            with st.spinner(f"Analizando: {home_team} vs {away_team}..."):
                try:
                    exito, resultado = hacer_prediccion_detallada(
                        home_team, away_team, h_clean, a_clean, year
                    )
                    
                    # VALIDACIÓN CRÍTICA: Si no hay stats detalladas, es porque el scraper no halló al lanzador
                    if not exito or not resultado.get('stats_detalladas'):
                        st.error("Verifique el nombre del lanzador e intente de nuevo")
                        st.stop()
                        
                    guardar_prediccion_local(home_team, away_team, home_pitcher, away_pitcher, year, resultado)
                    
                    st.success("Predicción realizada exitosamente!")
                    
                    st.markdown("---")
                    st.markdown("## Resultado")
                    
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
                        st.metric("", f"{prob_home*100:.1f}%", delta=f"{(prob_home-0.5)*100:+.1f}% vs 50%")
                    
                    with col2:
                        st.markdown(
                            get_team_logo_html(away_team, 30) + f"**Probabilidad de {away_team}**",
                            unsafe_allow_html=True
                        )
                        st.metric("", f"{prob_away*100:.1f}%", delta=f"{(prob_away-0.5)*100:+.1f}% vs 50%")
                    
                    with col3:
                        if confianza > 0.70:
                            conf_class = "confidence-high"
                            conf_emoji = ""
                            conf_text = "MUY ALTA"
                        elif confianza > 0.60:
                            conf_class = "confidence-medium"
                            conf_emoji = ""
                            conf_text = "ALTA"
                        elif confianza > 0.55:
                            conf_class = "confidence-medium"
                            conf_emoji = ""
                            conf_text = "MODERADA"
                        else:
                            conf_class = "confidence-low"
                            conf_emoji = ""
                            conf_text = "BAJA"
                        
                        st.metric("Confianza", f"{confianza*100:.1f}%")
                        st.markdown(f'<p class="{conf_class}">{conf_emoji} {conf_text}</p>', unsafe_allow_html=True)
                    
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
                    
                    # --- LÓGICA DE PROCESAMIENTO PARA WEB APP ---
                    # Extraemos los datos necesarios del objeto 'resultado'
                    stats_det = resultado.get('stats_detalladas', {})
                    features = resultado.get('features_usadas', {}) # Asegúrate que tu API devuelva esto

                    # Nombres reales de los lanzadores para las tarjetas
                    name_h = stats_det.get('home_pitcher', {}).get('nombre', f"Abridor {home_team}")
                    name_a = stats_det.get('away_pitcher', {}).get('nombre', f"Abridor {away_team}")

                    st.markdown("---")
                    st.markdown("##  Super Features: Análisis Dinámico")
                    st.info("Comparativa de dominio directo basada en el cruce de estadísticas de lanzadores vs. bateadores.")

                    col1, col2, col3 = st.columns(3)

                    # 1. Neutralización (WHIP vs OPS)
                    neut = features.get('super_neutralizacion_whip_ops', 0)
                    ventaja_n = home_team if neut < 0 else away_team
                    pitcher_n = name_h if neut < 0 else name_a
                    rival_n = away_team if neut < 0 else home_team
                    # Calculamos un % de impacto relativo (ajusta la escala según los valores de tu modelo)
                    pct_n = abs(neut) * 100 

                    with col1:
                        st.markdown(f"""
                        <div class="super-feature-box">
                            <h4> Neutralización</h4>
                            <p style="font-size: 1.1rem; color: #1f77b4; font-weight: bold;">Ventaja {ventaja_n}</p>
                            <p style="font-size: 0.9rem;"><b>{pitcher_n}</b> controla el bateo de {rival_n} con una eficiencia del <b>{pct_n:.1f}%</b> superior al promedio.</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # 2. Resistencia (ERA vs OPS)
                    res = features.get('super_resistencia_era_ops', 0)
                    ventaja_r = home_team if res < 0 else away_team
                    pitcher_r = name_h if res < 0 else name_a
                    rival_r = away_team if res < 0 else home_team

                    # SOLUCIÓN: Usamos logaritmo o un factor de escala menor para normalizar
                    pct_r = min(abs(res) * 10, 100.0) # Opción A: Escala x10 y tope en 100%

                    with col2:
                        st.markdown(f"""
                        <div class="super-feature-box">
                            <h4> Resistencia</h4>
                            <p style="font-size: 1.1rem; color: #1f77b4; font-weight: bold;">Ventaja {ventaja_r}</p>
                            <p style="font-size: 0.9rem;"><b>{pitcher_r}</b> es un <b>{pct_r:.1f}%</b> más sólido ante el poder (OPS) de los bateadores de {rival_r}.</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # 3. Muro Bullpen
                    muro = features.get('super_muro_bullpen', 0)
                    ventaja_m = home_team if muro < 0 else away_team
                    rival_m = away_team if muro < 0 else home_team
                    pct_m = abs(muro) * 100

                    with col3:
                        st.markdown(f"""
                        <div class="super-feature-box">
                            <h4> Muro Bullpen</h4>
                            <p style="font-size: 1.1rem; color: #1f77b4; font-weight: bold;">Ventaja {ventaja_m}</p>
                            <p style="font-size: 0.9rem;">El relevo de {ventaja_m} domina a los mejores bateadores de {rival_m} en un <b>{pct_m:.1f}%</b>.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # ESTADÍSTICAS DETALLADAS
                    stats_detalladas = resultado.get('stats_detalladas', {})
                    
                    if stats_detalladas:
                        st.markdown("---")
                        st.markdown("##  Estadísticas Detalladas")
                        
                        st.markdown("###  Lanzadores Iniciales")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            home_pitcher_stats = stats_detalladas.get('home_pitcher')
                            if home_pitcher_stats:
                                st.markdown(
                                    get_team_logo_html(home_team, 40) + 
                                    f"  {home_team} - {home_pitcher_stats.get('nombre', home_pitcher)}",
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
                            away_pitcher_stats = stats_detalladas.get('away_pitcher')
                            if away_pitcher_stats:
                                st.markdown(
                                    get_team_logo_html(away_team, 40) + 
                                    f"  {away_team} - {away_pitcher_stats.get('nombre', away_pitcher)}",
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
                                st.warning(f" No se encontró {away_pitcher}")
                        
                        st.markdown("---")
                        st.markdown("### Top 3 Bateadores")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            home_batters = stats_detalladas.get('home_batters', [])
                            if home_batters:
                                st.markdown(
                                    get_team_logo_html(home_team, 40) + f"   {home_team}",
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
                                st.warning(f" No se encontraron bateadores")
                        
                        with col2:
                            away_batters = stats_detalladas.get('away_batters', [])
                            if away_batters:
                                st.markdown(
                                    get_team_logo_html(away_team, 40) + f"   {away_team}",
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
                                st.warning(f" No se encontraron bateadores")
                    
                    if isinstance(resultado, dict) and resultado.get('mensaje'):
                        st.info(f"{resultado.get('mensaje')}")
                    
                    # BOTÓN PARA DESCARGAR JSON DE RESULTADOS
                    st.markdown("---")
                    result_json = json.dumps(resultado, indent=2)
                    st.download_button(
                        label="📥 Descargar Reporte Técnico (JSON)",
                        data=result_json,
                        file_name=f"prediccion_{home_team}_{away_team}_{year}.json",
                        mime="application/json",
                    )
                except Exception as e:
                    # Captura cualquier otro error inesperado y muestra el mismo mensaje
                    st.error("Lanzador/es no encontrado/s. Verifique el nombre del **Lanzador** y **Equipo** e intente de nuevo, Probabilidades calculadas con datos generales del equipo.")
                    st.stop()
# ============================================================================
# PÁGINA: HISTORIAL
# ============================================================================
elif pagina == " Historial":

    # Logo centrado
    col_l1, col_l2, col_l3 = st.columns([1, 0.8, 1])
    with col_l2:
        st.image("https://www.mlbstatic.com/team-logos/league-on-dark/1.svg", use_container_width=True)

    st.markdown('<div class="main-header"> Historial de Predicciones</div>', unsafe_allow_html=True)
    
    if 'historial' not in st.session_state or not st.session_state.historial:
        st.info("Aún no hay predicciones en esta sesión. ¡Realiza una en el Predictor!")
    else:
        if st.button("Clear History"):
            st.session_state.historial = []
            st.rerun()

        for idx, p in enumerate(st.session_state.historial):
            # El título del expander usa los nombres de los equipos y el año
            with st.expander(f" {p['timestamp']} — {p['home_team']} vs {p['away_team']} ({p['year']})"):
                col1, col2, col3 = st.columns(3)
                
                # Columna 1: Equipo Ganador con logo si es posible
                ganador_display = get_team_display_name(p['ganador'])
                col1.metric("Ganador", ganador_display)
                
                # Columna 2: Nivel de confianza
                col2.metric("Confianza", f"{p['confianza']*100:.1f}%")
                
                # Columna 3: Nombres REALES de los lanzadores (ya formateados en el paso 1)
                col3.markdown("**Duelo de Lanzadores:**")
                col3.write(f" {p['home_pitcher']}")
                col3.write(f" {p['away_pitcher']}")

# ============================================================================
# PÁGINA: ACERCA DE
# ============================================================================
elif pagina == " Acerca de":
    # 1. Header y Logo Centrado con CSS
    st.markdown("""
        <style>
        .about-container {
            text-align: center;
            padding: 2rem 0;
        }
        .info-card {
            padding: 1.5rem;
            border-radius: 0.8rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            border: 1px solid #e2e8f0;
            margin-bottom: 1.5rem;
            height: 100%;
        }
        .tech-badge {
            background-color: #f1f5f9;
            color: #475569;
            padding: 0.2rem 0.6rem;
            border-radius: 0.5rem;
            font-size: 0.8rem;
            font-weight: bold;
            display: inline-block;
            margin: 2px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Logo centrado
    col_l1, col_l2, col_l3 = st.columns([1, 0.8, 1])
    with col_l2:
        st.image("https://www.mlbstatic.com/team-logos/league-on-dark/1.svg", use_container_width=True)
    
    st.markdown('<div class="main-header">Sobre el Modelo V3</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Sistema Híbrido de Predicción Estadística</div>', unsafe_allow_html=True)

    # 2. Introducción en una columna ancha
    st.markdown("""
    <div class="info-card">
        <h3>¿Qué es MLB Game Predictor Hybrid?</h3>
        <p>MLB Game Predictor es un sistema de predicción de partidos de béisbol que utiliza 
        <b>Machine Learning con Modelo Híbrido</b> entrenado con datos Historicos y datos en Tiempo Real para analizar estadísticas de equipos y jugadores, 
        enfocándose en la ventaja competitiva generada por los enfrentamientos directos (Matchups).</p>
    </div>
    """, unsafe_allow_html=True)

    # 3. Dos columnas para Innovaciones y Características
    col_inf1, col_inf2 = st.columns(2)
    
    with col_inf1:
        st.markdown("""
        <div class="info-card">
            <h4>¿Qué hay de nuevo en la V3?</h4>
            <ul>
                <li><b>XGBoost Optimizado:</b> Entrenamiento con más de 20 años de datos.</li>
                <li><b>SelectKBest Features:</b> 26 variables con mayor impacto real.</li>
                <li><b>Dynamic Scraping:</b> Datos en tiempo real de Baseball-Reference.</li>
                <li><b>Modelo Híbrido:</b> Combinación de técnicas de ML para mayor precisión.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_inf2:
        st.markdown("""
        <div class="info-card">
            <h4>Características Principales</h4>
            <ul>
                <li>Análisis detallado de lanzadores y bateadores.</li>
                <li>Logos oficiales y visualización avanzada.</li>
                <li>Métricas de confianza por partido.</li>
                <li>Interfaz profesional optimizada.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # 4. Super Features en 3 columnas (aprovechando el ancho)
    st.markdown("### Super Features Explicadas")
    sf1, sf2, sf3 = st.columns(3)
    
    with sf1:
        st.markdown('<div class="info-card"><b>Neutralización</b><br><small>Cruza el WHIP del lanzador con el OPS rival.</small></div>', unsafe_allow_html=True)
    with sf2:
        st.markdown('<div class="info-card"><b>Resistencia</b><br><small>Entradas de calidad frente al poder del lineup.</small></div>', unsafe_allow_html=True)
    with sf3:
        st.markdown('<div class="info-card"><b>Muro Bullpen</b><br><small>Efectividad del relevo vs bateadores de cierre.</small></div>', unsafe_allow_html=True)

    # 5. Tecnologías (con badges limpios)
    st.markdown("### Stack Tecnológico")
    st.markdown("""
        <span class="tech-badge">FastAPI</span> <span class="tech-badge">XGBoost</span> 
        <span class="tech-badge">Scikit-learn</span> <span class="tech-badge">Streamlit</span> 
        <span class="tech-badge">BeautifulSoup</span> <span class="tech-badge">Plotly</span>
    """, unsafe_allow_html=True)

    st.info("Nota: Este sistema es una herramienta estadística. El béisbol es impredecible, ¡disfruta el juego!")

    # 6. Cuadrícula de Equipos (revisada)
    st.markdown("---")
    st.markdown("### Equipos de la MLB")
    
    cols_per_row = 6 # Aumentamos a 6 para que se vea más organizado
    teams_sorted = sorted(EQUIPOS_MLB.items(), key=lambda x: x[1]['nombre'])
    
    for i in range(0, len(teams_sorted), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(teams_sorted):
                code, info = teams_sorted[i + j]
                with col:
                    st.image(info['logo'], width=60)
                    st.caption(f"**{code}**")
# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    f"<center><p style='color: #666;'>MLB Predictor V3.0 | 2026 | Datos: Baseball-Reference</p></center>", 
    unsafe_allow_html=True
)
