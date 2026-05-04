import os
import re
import sqlite3
import unicodedata
import warnings

import pandas as pd
import xgboost as xgb

# Importamos la lógica de extracción del archivo de entrenamiento
from train_model_hybrid_actions import extraer_features_hibridas

warnings.filterwarnings('ignore')

# CONFIGURACIÓN DE RUTAS
MODELO_PATH = './models/modelo_mlb_v3.5.json'
DB_PATH = './data/mlb_reentrenamiento.db'

# MAPEO PARA NOMBRES COMPLETOS (Solo para Output Visual) - Mantenido Intacto
CODE_TO_FULL_NAME = {
    'Arizona D\'Backs': 'ARI', 'Atlanta Braves': 'ATL', 'Baltimore Orioles': 'BAL',
    'Boston Red Sox': 'BOS', 'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CHW',
    'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE', 'Colorado Rockies': 'COL',
    'Detroit Tigers': 'DET', 'Houston Astros': 'HOU', 'Kansas City Royals': 'KCR',
    'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD', 'Miami Marlins': 'MIA',
    'Milwaukee Brewers': 'MIL', 'Minnesota Twins': 'MIN', 'New York Mets': 'NYM',
    'New York Yankees': 'NYY', 'Oakland Athletics': 'OAK', 'Philadelphia Phillies': 'PHI',
    'Pittsburgh Pirates': 'PIT', 'San Diego Padres': 'SDP', 'San Francisco Giants': 'SFG',
    'Seattle Mariners': 'SEA', 'St. Louis Cardinals': 'STL', 'Tampa Bay Rays': 'TBR',
    'Texas Rangers': 'TEX', 'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSN',
    'Arizona Diamondbacks': 'ARI', 'Atlanta Braves': 'ATL',
    'Baltimore Orioles': 'BAL', 'Boston Red Sox': 'BOS',
    'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CHW',
    'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE',
    'Colorado Rockies': 'COL', 'Detroit Tigers': 'DET',
    'Houston Astros': 'HOU', 'Kansas City Royals': 'KCR',
    'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD',
    'Miami Marlins': 'MIA', 'Milwaukee Brewers': 'MIL',
    'Minnesota Twins': 'MIN', 'New York Mets': 'NYM',
    'New York Yankees': 'NYY', 'Oakland Athletics': 'OAK',
    'Philadelphia Phillies': 'PHI', 'Pittsburgh Pirates': 'PIT',
    'San Diego Padres': 'SDP', 'Seattle Mariners': 'SEA',
    'San Francisco Giants': 'SFG', 'St. Louis Cardinals': 'STL',
    'Tampa Bay Rays': 'TBR', 'Texas Rangers': 'TEX',
    'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSN',
    'Diamondbacks': 'ARI', 'Braves': 'ATL', 'Orioles': 'BAL',
    'Red Sox': 'BOS', 'Cubs': 'CHC', 'White Sox': 'CHW',
    'Reds': 'CIN', 'Guardians': 'CLE', 'Rockies': 'COL',
    'Tigers': 'DET', 'Astros': 'HOU', 'Royals': 'KCR',
    'Angels': 'LAA', 'Dodgers': 'LAD', 'Marlins': 'MIA',
    'Brewers': 'MIL', 'Twins': 'MIN', 'Mets': 'NYM',
    'Yankees': 'NYY', 'Athletics': 'OAK', 'Phillies': 'PHI',
    'Pirates': 'PIT', 'Padres': 'SDP', 'Mariners': 'SEA',
    'Giants': 'SFG', 'Cardinals': 'STL', 'Rays': 'TBR',
    'Rangers': 'TEX', 'Blue Jays': 'TOR', 'Nationals': 'WSN',
    'ARI': 'ARI', 'ATL': 'ATL', 'BAL': 'BAL', 'BOS': 'BOS',
    'CHC': 'CHC', 'CHW': 'CHW', 'CIN': 'CIN', 'CLE': 'CLE',
    'COL': 'COL', 'DET': 'DET', 'HOU': 'HOU', 'KCR': 'KCR',
    'LAA': 'LAA', 'LAD': 'LAD', 'MIA': 'MIA', 'MIL': 'MIL',
    'MIN': 'MIN', 'NYM': 'NYM', 'NYY': 'NYY', 'OAK': 'OAK',
    'PHI': 'PHI', 'PIT': 'PIT', 'SDP': 'SDP', 'SEA': 'SEA',
    'SFG': 'SFG', 'STL': 'STL', 'TBR': 'TBR', 'TEX': 'TEX',
    'TOR': 'TOR', 'WSN': 'WSN'
}

# ============================================================================
# FUNCIONES DE SOPORTE (Respetadas)
# ============================================================================
def normalizar_texto(texto):
    """Limpia nombres de lanzadores para que coincidan con el scraper"""
    if not texto: return ""
    texto = str(texto).lower()
    texto = "".join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    texto = re.sub(r'[^a-z0-9]', '', texto)
    return texto

def obtener_nivel_confianza(prob_pct):
    confianza = max(prob_pct, 100 - prob_pct) / 100
    if confianza > 0.70: return "MUY ALTA"
    if confianza > 0.60: return "ALTA"
    if confianza > 0.55: return "MODERADA"
    return "BAJA (Partido muy parejo)"

# ============================================================================
# MOTOR DE PREDICCIÓN
# ============================================================================
def predecir_juego(home_team, away_team, home_pitcher, away_pitcher, year=2026, modo_auto=False):
    if not os.path.exists(MODELO_PATH):
        print(f"❌ Error: No existe el modelo en {MODELO_PATH}")
        return None

    model = xgb.Booster()
    model.load_model(MODELO_PATH)
    expected_features = model.feature_names

    p_home_clean = normalizar_texto(home_pitcher)
    p_away_clean = normalizar_texto(away_pitcher)

    row_data = {
        'home_team': home_team,
        'away_team': away_team,
        'home_pitcher': home_pitcher,
        'away_pitcher': away_pitcher,
        'home_pitcher_clean': p_home_clean,
        'away_pitcher_clean': p_away_clean,
        'year': year,
        'fecha': pd.Timestamp.now().strftime('%Y-%m-%d')
    }

    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Manejo preventivo si la tabla no existe aún
            try:
                df_historico = pd.read_sql("SELECT * FROM historico_real", conn)
            except:
                df_historico = pd.DataFrame()

        # 3. EXTRACCIÓN HÍBRIDA (Usa tu lógica de train_model_hybrid_actions)
        features_dict = extraer_features_hibridas(
            row_data,
            df_historico=df_historico,
            hacer_scraping=True,
            session_cache={}
        )

        if not features_dict:
            return None

        X_df = pd.DataFrame([features_dict])

        # 4. CÁLCULO DE SUPER FEATURES (Mantenido Intacto)
        s_neut = (features_dict.get('home_starter_WHIP', 1.3) * features_dict.get('away_team_OPS', 0.75)) - \
                 (features_dict.get('away_starter_WHIP', 1.3) * features_dict.get('home_team_OPS', 0.75))

        s_res = (features_dict.get('home_starter_ERA', 4.0) / (features_dict.get('away_team_OPS', 0.75) + 0.01)) - \
                (features_dict.get('away_starter_ERA', 4.0) / (features_dict.get('home_team_OPS', 0.75) + 0.01))

        s_muro = (features_dict.get('home_bullpen_WHIP', 1.3) * features_dict.get('away_best_OPS', 0.85)) - \
                 (features_dict.get('away_bullpen_WHIP', 1.3) * features_dict.get('home_best_OPS', 0.85))

        X_df['super_neutralizacion_whip_ops'] = s_neut
        X_df['super_resistencia_era_ops'] = s_res
        X_df['super_muro_bullpen'] = s_muro

        # 5. PREDICCIÓN
        X_df = X_df.reindex(columns=expected_features, fill_value=0)
        dmatrix = xgb.DMatrix(X_df)
        prob_home = model.predict(dmatrix)[0]

        prob_home_pct = round(float(prob_home) * 100, 2)
        prob_away_pct = round(100 - prob_home_pct, 2)
        conf_label = obtener_nivel_confianza(prob_home_pct)

        # 6. REPORTE
        ganador_code = home_team if prob_home > 0.5 else away_team
        ganador_full = ganador_code
        for full, code in CODE_TO_FULL_NAME.items():
            if code == ganador_code and len(full) > 4:
                ganador_full = full
                break

        # 7. OUTPUT VISUAL ENRIQUECIDO
        if not modo_auto:
            print("\n" + "="*75)
            print("   ⚾ MLB PREDICTOR V3.5 - ANÁLISIS ESTADÍSTICO")
            print("="*75)

            # Nombres reales (Punto 2)
            h_p_name = features_dict.get('home_pitcher_name_real', home_pitcher)
            a_p_name = features_dict.get('away_pitcher_name_real', away_pitcher)

            print(f" Encuentro: {home_team} vs {away_team}")
            print(f" Temporada: {year} | Scraping: Baseball-Reference")

            print("\n📊 COMPARATIVA DE EQUIPOS:")
            print(f" 🏠  {home_team}: OPS: {features_dict.get('home_team_OPS', 0):.3f} | Bullpen WHIP: {features_dict.get('home_bullpen_WHIP', 0):.3f}")
            print(f" ✈️   {away_team}: OPS: {features_dict.get('away_team_OPS', 0):.3f} | Bullpen WHIP: {features_dict.get('away_bullpen_WHIP', 0):.3f}")

            print("\n👤 LANZADORES ABRIDORES:")
            print(f" 🏠 {h_p_name}: ERA: {features_dict.get('home_starter_ERA', 0):.2f} | WHIP: {features_dict.get('home_starter_WHIP', 0):.3f}")
            print(f" ✈️  {a_p_name}: ERA: {features_dict.get('away_starter_ERA', 0):.2f} | WHIP: {features_dict.get('away_starter_WHIP', 0):.3f}")

            # Análisis de Bullpen (Punto 3)
            print("\n🧱 ANÁLISIS DE BULLPEN (Muro):")
            print(f" 🏠 {home_team}: ERA: {features_dict.get('home_bullpen_ERA', 0):.3f} | WHIP: {features_dict.get('home_bullpen_WHIP', 0):.3f}")
            print(f" ✈️  {away_team}: ERA: {features_dict.get('away_bullpen_ERA', 0):.3f} |  WHIP: {features_dict.get('away_bullpen_WHIP', 0):.3f}")

            d_era = features_dict.get('diff_bullpen_ERA', 0)
            print(f" 📊 Diferencial ERA: {d_era:+.2f}")

            # Tabla de Bateadores (Punto 1)
            print("\n🔥 TOP 3 BATEADORES ANALIZADOS:")
            for label, team_key in [("🏠 " + home_team, 'home_top_3_batters_details'), ("✈️  " + away_team, 'away_top_3_batters_details')]:
                print(f"\n {label}:")
                print(f" {'Nombre':<22} | {'BA':<5} | {'OBP':<5} | {'SLG':<5} | {'OPS':<5} | {'HR':<3} | {'RBI'}")
                print("-" * 75)
                for b in features_dict.get(team_key, []):
                    # Usamos .get() con valores por defecto para evitar errores si falta alguna estadística
                    nombre = b.get('n', 'Desconocido')
                    ba = b.get('ba', 0)
                    obp = b.get('obp', 0)
                    slg = b.get('slg', 0)
                    ops = b.get('ops', b.get('o', 0)) # Mantiene compatibilidad con tu 'o' anterior
                    hr = b.get('hr', 0)
                    rbi = b.get('rbi', 0)
                    print(f" {nombre:<22} | {ba:.3f} | {obp:.3f} | {slg:.3f} | {ops:.3f} | {int(hr):<3} | {int(rbi)}")

            print("\n" + "="*75)
            print(f" 🏆 GANADOR PREDICHO: {ganador_full}")
            print("="*75)
            print(f" Probabilidades: {home_team} {prob_home_pct}% | {away_team} {prob_away_pct}%")
            print(f" Confianza: {conf_label}")

            print("\n🚀 DIAGNÓSTICO DE SUPER FEATURES:")
            n_v = home_team if s_neut < 0 else away_team
            print(f" 🛡️ Neutralización: {s_neut:.4f} (Ventaja {n_v})")
            r_v = home_team if s_res < 0 else away_team
            print(f" 📉 Resistencia:    {s_res:.4f} (Ventaja {r_v})")
            m_v = home_team if s_muro < 0 else away_team
            print(f" 🧱 Muro Bullpen:   {s_muro:.4f} (Ventaja {m_v})")
            print("="*75 + "\n")

        # 8. GUARDADO EN DB (Lógica original intacta)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS predicciones_historico 
                           (fecha TEXT, home_team TEXT, away_team TEXT, home_pitcher TEXT, 
                            away_pitcher TEXT, prob_home REAL, prob_away REAL, 
                            prediccion TEXT, tipo TEXT)''')

            db_data = {
                'fecha': row_data['fecha'],
                'home_team': home_team,
                'away_team': away_team,
                'home_pitcher': home_pitcher,
                'away_pitcher': away_pitcher,
                'prob_home': prob_home_pct,
                'prob_away': prob_away_pct,
                'prediccion': ganador_code,
                'tipo': 'AUTOMATICO' if modo_auto else 'MANUAL'
            }
            pd.DataFrame([db_data]).to_sql('predicciones_historico', conn, if_exists='append', index=False)

        return db_data

    except Exception as e:
        print(f"❌ Error crítico en motor: {e}")
        return None

def ejecutar_flujo_diario():
    print("🚀 Iniciando Motor de Predicción Diario...")
    if not os.path.exists(DB_PATH):
        print("❌ Error: Base de datos no encontrada.")
        return

    with sqlite3.connect(DB_PATH) as conn:
        try:
            df_hoy = pd.read_sql("SELECT * FROM historico_partidos WHERE fecha = date('now')", conn)
        except:
            print("📭 Tabla 'historico_partidos' no encontrada.")
            return

    if df_hoy.empty:
        print("📭 No hay juegos registrados para hoy.")
        return

    for _, row in df_hoy.iterrows():
        predecir_juego(
            row['home_team'], row['away_team'],
            row['home_pitcher'], row['away_pitcher'],
            year=row.get('year', 2026), modo_auto=True
        )

if __name__ == "__main__":
    ejecutar_flujo_diario()



# CODE_TO_FULL_NAME = {
#     'Arizona D\'Backs': 'ARI', 'Atlanta Braves': 'ATL', 'Baltimore Orioles': 'BAL',
#     'Boston Red Sox': 'BOS', 'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CHW',
#     'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE', 'Colorado Rockies': 'COL',
#     'Detroit Tigers': 'DET', 'Houston Astros': 'HOU', 'Kansas City Royals': 'KCR',
#     'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD', 'Miami Marlins': 'MIA',
#     'Milwaukee Brewers': 'MIL', 'Minnesota Twins': 'MIN', 'New York Mets': 'NYM',
#     'New York Yankees': 'NYY', 'Oakland Athletics': 'OAK', 'Philadelphia Phillies': 'PHI',
#     'Pittsburgh Pirates': 'PIT', 'San Diego Padres': 'SDP', 'San Francisco Giants': 'SFG',
#     'Seattle Mariners': 'SEA', 'St. Louis Cardinals': 'STL', 'Tampa Bay Rays': 'TBR',
#     'Texas Rangers': 'TEX', 'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSN',
#     'Arizona Diamondbacks': 'ARI', 'Atlanta Braves': 'ATL',
#     'Baltimore Orioles': 'BAL', 'Boston Red Sox': 'BOS',
#     'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CHW',
#     'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE',
#     'Colorado Rockies': 'COL', 'Detroit Tigers': 'DET',
#     'Houston Astros': 'HOU', 'Kansas City Royals': 'KCR',
#     'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD',
#     'Miami Marlins': 'MIA', 'Milwaukee Brewers': 'MIL',
#     'Minnesota Twins': 'MIN', 'New York Mets': 'NYM',
#     'New York Yankees': 'NYY', 'Oakland Athletics': 'OAK',
#     'Philadelphia Phillies': 'PHI', 'Pittsburgh Pirates': 'PIT',
#     'San Diego Padres': 'SDP', 'Seattle Mariners': 'SEA',
#     'San Francisco Giants': 'SFG', 'St. Louis Cardinals': 'STL',
#     'Tampa Bay Rays': 'TBR', 'Texas Rangers': 'TEX',
#     'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSN',
#     # Nombres cortos
#     'Diamondbacks': 'ARI', 'Braves': 'ATL', 'Orioles': 'BAL',
#     'Red Sox': 'BOS', 'Cubs': 'CHC', 'White Sox': 'CHW',
#     'Reds': 'CIN', 'Guardians': 'CLE', 'Rockies': 'COL',
#     'Tigers': 'DET', 'Astros': 'HOU', 'Royals': 'KCR',
#     'Angels': 'LAA', 'Dodgers': 'LAD', 'Marlins': 'MIA',
#     'Brewers': 'MIL', 'Twins': 'MIN', 'Mets': 'NYM',
#     'Yankees': 'NYY', 'Athletics': 'OAK', 'Phillies': 'PHI',
#     'Pirates': 'PIT', 'Padres': 'SDP', 'Mariners': 'SEA',
#     'Giants': 'SFG', 'Cardinals': 'STL', 'Rays': 'TBR',
#     'Rangers': 'TEX', 'Blue Jays': 'TOR', 'Nationals': 'WSN',
#     # Códigos
#     'ARI': 'ARI', 'ATL': 'ATL', 'BAL': 'BAL', 'BOS': 'BOS',
#     'CHC': 'CHC', 'CHW': 'CHW', 'CIN': 'CIN', 'CLE': 'CLE',
#     'COL': 'COL', 'DET': 'DET', 'HOU': 'HOU', 'KCR': 'KCR',
#     'LAA': 'LAA', 'LAD': 'LAD', 'MIA': 'MIA', 'MIL': 'MIL',
#     'MIN': 'MIN', 'NYM': 'NYM', 'NYY': 'NYY', 'OAK': 'OAK',
#     'PHI': 'PHI', 'PIT': 'PIT', 'SDP': 'SDP', 'SEA': 'SEA',
#     'SFG': 'SFG', 'STL': 'STL', 'TBR': 'TBR', 'TEX': 'TEX',
#     'TOR': 'TOR', 'WSN': 'WSN'
# }



