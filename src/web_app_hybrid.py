"""
MLB Game Predictor - Web App con Streamlit (MODELO HÍBRIDO)
Ejecutar: streamlit run web_app_hybrid.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="MLB Game Predictor - Hybrid Model",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DATOS Y CONFIGURACIÓN
# ============================================================================

# Al inicio del archivo, después de los imports
import os

# Configurar URL de la API
if 'STREAMLIT_SHARING' in os.environ:
    # En producción (Streamlit Cloud)
    # Aquí tendrías que desplegar también la API en otro servicio
    # Por ahora, usamos el scraping directo sin API
    USE_API = False
else:
    # En desarrollo local
    API_URL = st.secrets.get("API_URL", "https://mlb-game-predictor-v2.streamlit.app/")
    USE_API = True

# # URL de la API HÍBRIDA (cambiar según donde esté desplegada)
# API_URL = st.secrets.get("API_URL", "http://localhost:8001")

# Mapeo de equipos
EQUIPOS_MLB = {
    'ARI': '🔶 Arizona Diamondbacks',
    'ATL': '🔴 Atlanta Braves',
    'BAL': '🟠 Baltimore Orioles',
    'BOS': '🔴 Boston Red Sox',
    'CHC': '🔵 Chicago Cubs',
    'CHW': '⚫ Chicago White Sox',
    'CIN': '🔴 Cincinnati Reds',
    'CLE': '🔴 Cleveland Guardians',
    'COL': '🟣 Colorado Rockies',
    'DET': '🔵 Detroit Tigers',
    'HOU': '🟠 Houston Astros',
    'KCR': '🔵 Kansas City Royals',
    'LAA': '🔴 Los Angeles Angels',
    'LAD': '🔵 Los Angeles Dodgers',
    'MIA': '🔵 Miami Marlins',
    'MIL': '🟡 Milwaukee Brewers',
    'MIN': '🔴 Minnesota Twins',
    'NYM': '🔵 New York Mets',
    'NYY': '🔵 New York Yankees',
    'OAK': '🟢 Oakland Athletics',
    'PHI': '🔴 Philadelphia Phillies',
    'PIT': '🟡 Pittsburgh Pirates',
    'SDP': '🟤 San Diego Padres',
    'SEA': '⚪ Seattle Mariners',
    'SFG': '🟠 San Francisco Giants',
    'STL': '🔴 St. Louis Cardinals',
    'TBR': '🔵 Tampa Bay Rays',
    'TEX': '🔴 Texas Rangers',
    'TOR': '🔵 Toronto Blue Jays',
    'WSN': '🔴 Washington Nationals'
}

# Códigos para selector
EQUIPOS_CODES = list(EQUIPOS_MLB.keys())
EQUIPOS_NAMES = list(EQUIPOS_MLB.values())

# ============================================================================
# ESTILOS CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .winner-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    .stat-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .pitcher-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        color: white;
        margin: 1rem 0;
    }
    .batter-row {
        background: #f8f9fa;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3b82f6;
    }
    .confidence-high {
        color: #10b981;
        font-weight: bold;
    }
    .confidence-medium {
        color: #f59e0b;
        font-weight: bold;
    }
    .confidence-low {
        color: #ef4444;
        font-weight: bold;
    }
    .hybrid-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

@st.cache_data(ttl=300)  # Cache por 5 minutos
def verificar_api():
    """Verifica que la API esté disponible"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('model_loaded', False), data
        return False, None
    except:
        return False, None


@st.cache_data(ttl=300)
def obtener_info_modelo():
    """Obtiene información del modelo"""
    try:
        response = requests.get(f"{API_URL}/info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def hacer_prediccion_detallada(home_team, away_team, home_pitcher, away_pitcher, year):
    """Realiza la predicción con estadísticas detalladas"""
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
        font=dict(size=14)
    )
    
    fig.update_layout(xaxis=dict(range=[0, 100]))
    
    return fig


def crear_gauge_confianza(confianza):
    """Crea gauge de confianza"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confianza * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Nivel de Confianza", 'font': {'size': 24}},
        number={'suffix': "%", 'font': {'size': 40}},
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
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig


def mostrar_stats_lanzador(pitcher_stats, titulo):
    """Muestra las estadísticas de un lanzador"""
    if pitcher_stats:
        st.markdown(f"### {titulo}")
        st.markdown(f"**Nombre:** {pitcher_stats.get('nombre', 'N/A')}")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("ERA", f"{pitcher_stats.get('ERA', 0):.2f}")
        with col2:
            st.metric("WHIP", f"{pitcher_stats.get('WHIP', 0):.3f}")
        with col3:
            st.metric("H9", f"{pitcher_stats.get('H9', 0):.2f}")
        with col4:
            st.metric("Victorias", int(pitcher_stats.get('W', 0)))
        with col5:
            st.metric("Derrotas", int(pitcher_stats.get('L', 0)))
    else:
        st.warning(f"⚠️ {titulo}: No se encontraron estadísticas")


def mostrar_stats_bateadores(batters_list, titulo):
    """Muestra las estadísticas de los top 3 bateadores"""
    if batters_list and len(batters_list) > 0:
        st.markdown(f"### {titulo}")
        
        for i, batter in enumerate(batters_list, 1):
            with st.expander(f"#{i} - {batter.get('nombre', 'N/A')}", expanded=(i==1)):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("BA", f"{batter.get('BA', 0):.3f}")
                    st.metric("OBP", f"{batter.get('OBP', 0):.3f}")
                
                with col2:
                    st.metric("SLG", f"{batter.get('SLG', 0):.3f}")
                    st.metric("OPS", f"{batter.get('OPS', 0):.3f}")
                
                with col3:
                    st.metric("HR", int(batter.get('HR', 0)))
                    st.metric("RBI", int(batter.get('RBI', 0)))
                
                with col4:
                    st.metric("R", int(batter.get('R', 0)))
                    st.metric("AB", int(batter.get('AB', 0)))
    else:
        st.warning(f"⚠️ {titulo}: No se encontraron estadísticas")


def crear_comparacion_lanzadores(home_pitcher, away_pitcher):
    """Crea gráfico de comparación de lanzadores"""
    if not home_pitcher or not away_pitcher:
        return None
    
    categorias = ['ERA', 'WHIP', 'H9']
    home_vals = [
        home_pitcher.get('ERA', 0),
        home_pitcher.get('WHIP', 0),
        home_pitcher.get('H9', 0)
    ]
    away_vals = [
        away_pitcher.get('ERA', 0),
        away_pitcher.get('WHIP', 0),
        away_pitcher.get('H9', 0)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Lanzador Local',
        x=categorias,
        y=home_vals,
        marker_color='#3b82f6'
    ))
    
    fig.add_trace(go.Bar(
        name='Lanzador Visitante',
        x=categorias,
        y=away_vals,
        marker_color='#ef4444'
    ))
    
    fig.update_layout(
        title="Comparación de Lanzadores",
        xaxis_title="Estadística",
        yaxis_title="Valor",
        barmode='group',
        height=400,
        showlegend=True
    )
    
    return fig


def guardar_prediccion_local(home_team, away_team, home_pitcher, away_pitcher, year, resultado):
    """Guarda la predicción en session_state"""
    if 'historial' not in st.session_state:
        st.session_state.historial = []
    
    prediccion = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'home_team': home_team,
        'away_team': away_team,
        'home_pitcher': home_pitcher,
        'away_pitcher': away_pitcher,
        'year': year,
        'ganador': resultado.get('ganador'),
        'prob_home': resultado.get('prob_home'),
        'prob_away': resultado.get('prob_away'),
        'confianza': resultado.get('confianza')
    }
    
    st.session_state.historial.insert(0, prediccion)
    
    # Mantener solo las últimas 50
    if len(st.session_state.historial) > 50:
        st.session_state.historial = st.session_state.historial[:50]


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/baseball.png", width=80)
    st.title(" MLB Predictor")
    st.markdown('<span class="hybrid-badge">🔬MODEL V2.0</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Estado de la API
    st.subheader("🔌 Estado del Sistema")
    
    api_ok, api_data = verificar_api()
    
    if api_ok:
        st.success("✅ API Conectada")
        st.success("✅ Modelo Cargado")
        
        # Indicar tipo de modelo
        model_type = api_data.get('model_type', 'unknown')
        if model_type == 'hybrid_optimized':
            st.info("🔬 Modelo Híbrido Optimizado")
        
        # Info del modelo
        info = obtener_info_modelo()
        if info:
            st.markdown("---")
            st.subheader("📊 Información del Modelo")
            st.metric("Modelo", info.get('nombre', 'N/A'))
            st.metric("Accuracy", f"{info.get('accuracy', 0)*100:.2f}%")
            st.metric("ROC-AUC", f"{info.get('roc_auc', 0):.4f}")
            st.metric("Features", info.get('n_features', 0))
            
            # Info adicional del modelo híbrido
            if 'validacion_temporal' in info:
                if info['validacion_temporal']:
                    st.caption("✅ Validación Temporal Activada")
    else:
        st.error("❌ API No Disponible")
        st.warning("Verifica que la API esté corriendo:")
        st.code("python api_hybrid.py")
    
    st.markdown("---")
    
    # Configuración
    st.subheader("⚙️ Configuración")
    
    nueva_url = st.text_input("URL de la API", value=API_URL)
    if nueva_url != API_URL:
        if st.button("Actualizar URL"):
            API_URL = nueva_url
            st.rerun()
    
    st.markdown("---")
    
    # Navegación
    st.subheader("📱 Navegación")
    pagina = st.radio(
        "Ir a:",
        ["🎯 Predictor", "📜 Historial", "ℹ️ Acerca de"],
        label_visibility="collapsed"
    )

# ============================================================================
# PÁGINA PRINCIPAL - PREDICTOR
# ============================================================================

if pagina == "🎯 Predictor":
    
    # Header
    st.markdown('<div class="main-header">⚾ MLB Game Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predicciones con Modelo Híbrido</div>', unsafe_allow_html=True)
    
    if not api_ok:
        st.error("⚠️ La API no está disponible. Por favor, inicia la API primero.")
        st.code("python api_hybrid.py", language="bash")
        st.stop()
    
    # Formulario de predicción
    with st.form("prediction_form"):
        st.subheader("📝 Datos del Partido")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏠 Equipo Local")
            home_team_display = st.selectbox(
                "Selecciona equipo local",
                EQUIPOS_NAMES,
                key="home_display"
            )
            home_team = EQUIPOS_CODES[EQUIPOS_NAMES.index(home_team_display)]
            
            home_pitcher = st.text_input(
                "Lanzador Local",
                placeholder="Ej: Bello, Kershaw, Cole...",
                key="home_pitcher"
            )
        
        with col2:
            st.markdown("#### ✈️ Equipo Visitante")
            away_team_display = st.selectbox(
                "Selecciona equipo visitante",
                EQUIPOS_NAMES,
                key="away_display"
            )
            away_team = EQUIPOS_CODES[EQUIPOS_NAMES.index(away_team_display)]
            
            away_pitcher = st.text_input(
                "Lanzador Visitante",
                placeholder="Ej: Webb, Cole, Ohtani...",
                key="away_pitcher"
            )
        
        year = st.number_input("Temporada", min_value=2020, max_value=2030, value=2025)
        
        st.markdown("---")
        
        submit_button = st.form_submit_button(
            " Realizar Predicción",
            use_container_width=True,
            type="primary"
        )
    
    # Procesar predicción
    if submit_button:
        
        if not home_pitcher or not away_pitcher:
            st.error("⚠️ Por favor ingresa los nombres de ambos lanzadores")
        elif home_team == away_team:
            st.error("⚠️ Los equipos deben ser diferentes")
        else:
            with st.spinner("🔄 Analizando datos y realizando predicción... Esto puede tomar 10-30 segundos..."):
                exito, resultado = hacer_prediccion_detallada(
                    home_team, away_team, home_pitcher, away_pitcher, year
                )
            
            if exito:
                # Guardar en historial
                guardar_prediccion_local(home_team, away_team, home_pitcher, away_pitcher, year, resultado)
                
                st.success("✅ Predicción realizada exitosamente!")
                
                # Mostrar resultado
                st.markdown("---")
                st.markdown("## 🎯 Resultado de la Predicción")
                
                ganador = resultado.get('ganador')
                prob_home = resultado.get('prob_home', 0)
                prob_away = resultado.get('prob_away', 0)
                confianza = resultado.get('confianza', 0)
                
                # Ganador destacado
                ganador_nombre = EQUIPOS_MLB.get(ganador, ganador)
                st.markdown(f"""
                <div class="winner-box">
                    <h1 style="margin:0; font-size: 2.5rem;">🏆 GANADOR PREDICHO</h1>
                    <h2 style="margin:0.5rem 0 0 0; font-size: 3rem;">{ganador_nombre}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Métricas
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        f"Probabilidad {home_team}",
                        f"{prob_home*100:.1f}%",
                        delta=f"{(prob_home-0.5)*100:+.1f}% vs 50%"
                    )
                
                with col2:
                    st.metric(
                        f"Probabilidad {away_team}",
                        f"{prob_away*100:.1f}%",
                        delta=f"{(prob_away-0.5)*100:+.1f}% vs 50%"
                    )
                
                with col3:
                    if confianza > 0.70:
                        conf_class = "confidence-high"
                        conf_emoji = "🔥"
                        conf_text = "MUY ALTA"
                    elif confianza > 0.60:
                        conf_class = "confidence-medium"
                        conf_emoji = "👍"
                        conf_text = "ALTA"
                    elif confianza > 0.55:
                        conf_class = "confidence-medium"
                        conf_emoji = "🤔"
                        conf_text = "MODERADA"
                    else:
                        conf_class = "confidence-low"
                        conf_emoji = "🤷"
                        conf_text = "BAJA"
                    
                    st.metric(
                        "Confianza",
                        f"{confianza*100:.1f}%"
                    )
                    st.markdown(f'<p class="{conf_class}">{conf_emoji} {conf_text}</p>', unsafe_allow_html=True)
                
                # Gráficos de probabilidades
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
                
                # ============================================================
                # ESTADÍSTICAS DETALLADAS
                # ============================================================
                
                stats_detalladas = resultado.get('stats_detalladas', {})
                
                if stats_detalladas:
                    st.markdown("---")
                    st.markdown("## 📊 Estadísticas Detalladas de Jugadores")
                    
                    # LANZADORES
                    st.markdown("###  Lanzadores Iniciales")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        home_pitcher_stats = stats_detalladas.get('home_pitcher')
                        if home_pitcher_stats:
                            st.markdown(f"#### 🏠 {home_team} - {home_pitcher_stats.get('nombre', home_pitcher)}")
                            
                            subcol1, subcol2, subcol3 = st.columns(3)
                            with subcol1:
                                era_val = home_pitcher_stats.get('ERA', 0)
                                st.metric("ERA", f"{era_val:.2f}", 
                                         delta=f"{4.0-era_val:.2f}" if era_val < 4.0 else f"{era_val-4.0:.2f}",
                                         delta_color="inverse")
                            with subcol2:
                                whip_val = home_pitcher_stats.get('WHIP', 0)
                                st.metric("WHIP", f"{whip_val:.3f}",
                                         delta=f"{1.3-whip_val:.2f}" if whip_val < 1.3 else f"{whip_val-1.3:.2f}",
                                         delta_color="inverse")
                            with subcol3:
                                st.metric("H9", f"{home_pitcher_stats.get('H9', 0):.2f}")
                            
                            subcol1, subcol2 = st.columns(2)
                            with subcol1:
                                st.metric("Victorias", int(home_pitcher_stats.get('W', 0)))
                            with subcol2:
                                st.metric("Derrotas", int(home_pitcher_stats.get('L', 0)))
                        else:
                            st.warning(f"⚠️ No se encontraron datos del lanzador {home_pitcher}")
                    
                    with col2:
                        away_pitcher_stats = stats_detalladas.get('away_pitcher')
                        if away_pitcher_stats:
                            st.markdown(f"#### ✈️ {away_team} - {away_pitcher_stats.get('nombre', away_pitcher)}")
                            
                            subcol1, subcol2, subcol3 = st.columns(3)
                            with subcol1:
                                era_val = away_pitcher_stats.get('ERA', 0)
                                st.metric("ERA", f"{era_val:.2f}",
                                         delta=f"{4.0-era_val:.2f}" if era_val < 4.0 else f"{era_val-4.0:.2f}",
                                         delta_color="inverse")
                            with subcol2:
                                whip_val = away_pitcher_stats.get('WHIP', 0)
                                st.metric("WHIP", f"{whip_val:.3f}",
                                         delta=f"{1.3-whip_val:.2f}" if whip_val < 1.3 else f"{whip_val-1.3:.2f}",
                                         delta_color="inverse")
                            with subcol3:
                                st.metric("H9", f"{away_pitcher_stats.get('H9', 0):.2f}")
                            
                            subcol1, subcol2 = st.columns(2)
                            with subcol1:
                                st.metric("Victorias", int(away_pitcher_stats.get('W', 0)))
                            with subcol2:
                                st.metric("Derrotas", int(away_pitcher_stats.get('L', 0)))
                        else:
                            st.warning(f"⚠️ No se encontraron datos del lanzador {away_pitcher}")
                    
                    # Gráfico comparativo de lanzadores
                    if stats_detalladas.get('home_pitcher') and stats_detalladas.get('away_pitcher'):
                        st.plotly_chart(
                            crear_comparacion_lanzadores(
                                stats_detalladas.get('home_pitcher'),
                                stats_detalladas.get('away_pitcher')
                            ),
                            use_container_width=True
                        )
                    
                    # BATEADORES
                    st.markdown("---")
                    st.markdown("###  Top 3 Bateadores")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        home_batters = stats_detalladas.get('home_batters', [])
                        if home_batters:
                            st.markdown(f"#### 🏠 {home_team}")
                            for i, batter in enumerate(home_batters, 1):
                                with st.expander(f"#{i} - {batter.get('nombre', 'N/A')}", expanded=(i==1)):
                                    subcol1, subcol2, subcol3, subcol4 = st.columns(4)
                                    
                                    with subcol1:
                                        st.metric("BA", f"{batter.get('BA', 0):.3f}")
                                        st.metric("OBP", f"{batter.get('OBP', 0):.3f}")
                                    
                                    with subcol2:
                                        st.metric("SLG", f"{batter.get('SLG', 0):.3f}")
                                        st.metric("OPS", f"{batter.get('OPS', 0):.3f}")
                                    
                                    with subcol3:
                                        st.metric("HR", int(batter.get('HR', 0)))
                                        st.metric("RBI", int(batter.get('RBI', 0)))
                                    
                                    with subcol4:
                                        st.metric("R", int(batter.get('R', 0)))
                                        st.metric("AB", int(batter.get('AB', 0)))
                        else:
                            st.warning(f"⚠️ No se encontraron bateadores del {home_team}")
                    
                    with col2:
                        away_batters = stats_detalladas.get('away_batters', [])
                        if away_batters:
                            st.markdown(f"#### ✈️ {away_team}")
                            for i, batter in enumerate(away_batters, 1):
                                with st.expander(f"#{i} - {batter.get('nombre', 'N/A')}", expanded=(i==1)):
                                    subcol1, subcol2, subcol3, subcol4 = st.columns(4)
                                    
                                    with subcol1:
                                        st.metric("BA", f"{batter.get('BA', 0):.3f}")
                                        st.metric("OBP", f"{batter.get('OBP', 0):.3f}")
                                    
                                    with subcol2:
                                        st.metric("SLG", f"{batter.get('SLG', 0):.3f}")
                                        st.metric("OPS", f"{batter.get('OPS', 0):.3f}")
                                    
                                    with subcol3:
                                        st.metric("HR", int(batter.get('HR', 0)))
                                        st.metric("RBI", int(batter.get('RBI', 0)))
                                    
                                    with subcol4:
                                        st.metric("R", int(batter.get('R', 0)))
                                        st.metric("AB", int(batter.get('AB', 0)))
                        else:
                            st.warning(f"⚠️ No se encontraron bateadores del {away_team}")
                
                # Mensaje adicional
                if resultado.get('mensaje'):
                    st.info(f"ℹ️ {resultado.get('mensaje')}")
                
                # Botón de descargar
                result_json = json.dumps(resultado, indent=2)
                st.download_button(
                    "📥 Descargar Resultado (JSON)",
                    result_json,
                    file_name=f"prediccion_{home_team}_vs_{away_team}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
            else:
                st.error(f"❌ Error en la predicción: {resultado}")

# ============================================================================
# PÁGINA - HISTORIAL
# ============================================================================

elif pagina == "📜 Historial":
    st.title("📜 Historial de Predicciones")
    
    if 'historial' not in st.session_state or len(st.session_state.historial) == 0:
        st.info("📋 No hay predicciones en el historial aún")
    else:
        st.success(f"✅ {len(st.session_state.historial)} predicciones guardadas")
        
        # Convertir a DataFrame
        df = pd.DataFrame(st.session_state.historial)
        
        # Mostrar tabla
        st.dataframe(
            df.style.format({
                'prob_home': '{:.1%}',
                'prob_away': '{:.1%}',
                'confianza': '{:.1%}'
            }),
            use_container_width=True
        )
        
        # Estadísticas
        st.markdown("---")
        st.subheader("📊 Estadísticas del Historial")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Predicciones", len(df))
        
        with col2:
            conf_promedio = df['confianza'].mean()
            st.metric("Confianza Promedio", f"{conf_promedio*100:.1f}%")
        
        with col3:
            ganador_mas_comun = df['ganador'].mode()[0] if len(df) > 0 else "N/A"
            st.metric("Equipo Más Predicho", ganador_mas_comun)
        
        # Botón limpiar historial
        if st.button("🗑️ Limpiar Historial", type="secondary"):
            st.session_state.historial = []
            st.rerun()

# ============================================================================
# PÁGINA - ACERCA DE
# ============================================================================

elif pagina == "ℹ️ Acerca de":
    st.title("ℹ️ Acerca de MLB Game Predictor - Hybrid Model")
    
    st.markdown("""
    ## 🏟️ ¿Qué es MLB Game Predictor Hybrid?
    
    MLB Game Predictor es un sistema de predicción de partidos de béisbol que utiliza 
    **Machine Learning con Modelo Híbrido** para analizar estadísticas de equipos y jugadores.
    
    ### 🔬 Modelo Híbrido
    
    Este modelo combina **dos tipos de features** para mayor precisión:
    
    1. **Features Temporales** (del CSV):
       - Últimos 10 partidos de cada equipo
       - Últimas 5 aperturas de cada lanzador
       - Historial de enfrentamientos (H2H)
       - Rachas actuales y momentum
       - Diferencial de carreras reciente
    
    2. **Features de Scraping** (en tiempo real):
       - Estadísticas actualizadas de equipos
       - Stats detalladas de lanzadores iniciales
       - Top 3 bateadores por OBP
       - Métricas avanzadas (ERA, WHIP, OPS, SLG)
    
    ### 🎯 Características
    
    - ✅ **Predicciones en tiempo real** basadas en estadísticas actualizadas
    - ✅ **Análisis detallado de jugadores clave** (lanzadores iniciales y top 3 bateadores)
    - ✅ **Interfaz intuitiva** y fácil de usar
    - ✅ **Historial de predicciones** para seguimiento
    - ✅ **Múltiples visualizaciones** de resultados
    - ✅ **Estadísticas completas** de cada jugador clave
    - ✅ **Modelo híbrido optimizado** con XGBoost
    
    ### 📊 Modelo de Machine Learning
    
    El modelo utiliza:
    - **XGBoost / Random Forest** optimizados con GridSearchCV
    - **~55 features híbridas** (temporales + scraping)
    - **Accuracy de ~63-67%** en datos de prueba
    - **Validación temporal** (TimeSeriesSplit)
    - **Optimización de hiperparámetros**
    
    ### 🔑 Features Principales
    
    **Temporales:**
    - Últimos 10 partidos por equipo
    - Últimas 5 aperturas por lanzador
    - Historial H2H (últimos 10 enfrentamientos)
    - Rachas y momentum
    
    **Scraping:**
    - **Pitching:** ERA, WHIP, H9, W, L
    - **Batting:** BA, OBP, SLG, OPS, RBI, R, HR
    - **Jugadores Clave:** Top 3 bateadores (por OBP) + Lanzador inicial
    
    **Derivadas:**
    - Diferencias entre equipos
    - Comparaciones de lanzadores
    - Ratios y ventajas
    
    ### 🛠️ Tecnologías
    
    - **Backend:** FastAPI + scikit-learn + XGBoost
    - **Scraping:** cloudscraper + BeautifulSoup
    - **Frontend:** Streamlit
    - **Visualización:** Plotly
    - **Data:** Baseball-Reference.com
    - **Optimización:** GridSearchCV + TimeSeriesSplit
    
    ### 🚀 Ventajas del Modelo Híbrido
    
    1. **Mayor precisión**: Combina contexto temporal + stats actuales
    2. **Adaptabilidad**: Se ajusta a la forma reciente de los equipos
    3. **Profundidad**: Analiza tanto el equipo completo como jugadores clave
    4. **Robustez**: Validación temporal previene overfitting
    5. **Escalabilidad**: Procesa 3000+ partidos sin colapsar
    
    ### 👨‍💻 Desarrollo
    
    Desarrollado como proyecto avanzado de Machine Learning aplicado a deportes.
    Evolución del modelo original con mejoras en features, validación y optimización.
    
    ### 📝 Nota
    
    Las predicciones son estimaciones basadas en datos históricos y estadísticas actuales.
    No garantizan resultados futuros. El modelo híbrido mejora la precisión pero 
    el béisbol sigue siendo un deporte impredecible.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🚀 Cómo usar
    
    1. **Inicia la API híbrida** en una terminal:
       ```bash
       python api_hybrid.py
       ```
       O con uvicorn:
       ```bash
       uvicorn api_hybrid:app --reload
       ```
    
    2. **Inicia esta web app** en otra terminal:
       ```bash
       streamlit run web_app_hybrid.py
       ```
    
    3. **Selecciona los equipos** y lanzadores
    
    4. **¡Haz la predicción!**
    
    5. **Revisa las estadísticas detalladas** de lanzadores y bateadores
    
    ### 📈 Diferencias con el Modelo Original
    
    | Característica | Modelo Original | Modelo Híbrido |
    |----------------|----------------|----------------|
    | **Features** | 37 (solo scraping) | 55+ (temporal + scraping) |
    | **Validación** | KFold estándar | TimeSeriesSplit temporal |
    | **Algoritmos** | RF + GBM | RF + GBM + XGBoost |
    | **Optimización** | Parámetros fijos | GridSearchCV automático |
    | **Accuracy** | ~60-64% | ~63-67% |
    | **Partidos procesables** | ~600 | 3000+ |
    | **Scraping** | Todos los partidos | Solo partidos recientes |
    | **Velocidad** | 2-3 horas | 1-1.5 horas |
    
    ### 🎓 Aprendizajes Clave
    
    - Las **features temporales** (últimos 10 partidos) son altamente predictivas
    - La **validación temporal** es crucial para evitar "ver el futuro"
    - El **scraping inteligente** (solo partidos recientes) optimiza tiempos
    - Los **modelos ensemble** mejoran la robustez
    - Las **features derivadas** facilitan el aprendizaje del modelo
    """)