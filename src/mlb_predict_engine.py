"""
Motor de Predicción MLB V3.5 - REFACTORIZADO
Usa módulos centralizados para evitar duplicación de código
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import xgboost as xgb
import warnings
import sys

# Importar módulos centralizados
from mlb_config import (
    MODELO_PATH, DB_PATH,
    get_team_code, get_team_name
)
from mlb_feature_engineering import (
    calcular_super_features,
    detectar_outliers,
    calcular_estadisticas_agregadas
)

# Importar extracción de features del módulo de entrenamiento
from train_model_hybrid_actions import extraer_features_hibridas, normalizar_texto

warnings.filterwarnings('ignore')


# ============================================================================
# FUNCIONES DE SOPORTE
# ============================================================================

def obtener_nivel_confianza(prob_pct):
    """Determina el nivel de confianza basado en la probabilidad"""
    confianza = max(prob_pct, 100 - prob_pct) / 100
    if confianza > 0.70:
        return "MUY ALTA"
    if confianza > 0.60:
        return "ALTA"
    if confianza > 0.55:
        return "MODERADA"
    return "BAJA (Partido muy parejo)"


# ============================================================================
# MOTOR DE PREDICCIÓN
# ============================================================================

def predecir_juego(home_team, away_team, home_pitcher, away_pitcher, year=2026, modo_auto=False):
    """
    Predice el resultado de un juego de MLB
    
    Args:
        home_team: Código del equipo local
        away_team: Código del equipo visitante
        home_pitcher: Nombre del lanzador abridor local
        away_pitcher: Nombre del lanzador abridor visitante
        year: Año para scraping de estadísticas
        modo_auto: Si es True, suprime algunos prints
        
    Returns:
        Dict con resultado de la predicción o None si hay error
    """
    # 1. Validar que existe el modelo
    if not os.path.exists(MODELO_PATH):
        print(f"❌ Error: No existe el modelo en {MODELO_PATH}")
        return None

    # 2. Cargar modelo
    try:
        model = xgb.Booster()
        model.load_model(MODELO_PATH)
        expected_features = model.feature_names
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return None
    
    # 3. Normalizar nombres de lanzadores
    p_home_clean = normalizar_texto(home_pitcher)
    p_away_clean = normalizar_texto(away_pitcher)

    # 4. Preparar datos del partido
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
        # 5. Cargar histórico de partidos
        with sqlite3.connect(DB_PATH) as conn:
            try:
                df_historico = pd.read_sql("SELECT * FROM historico_real", conn)
            except:
                df_historico = pd.DataFrame()
        
        # 6. Extracción de features híbrida (temporal + scraping)
        features_dict = extraer_features_hibridas(
            row_data, 
            df_historico=df_historico, 
            hacer_scraping=True, 
            session_cache={}
        )
        
        if not features_dict:
            print("❌ No se pudieron extraer las features necesarias")
            return None

        # 7. Aplicar super features usando módulo centralizado
        features_dict = calcular_super_features(features_dict)
        
        # 8. Validar datos extraídos
        warnings_data = detectar_outliers(features_dict)
        if warnings_data and not modo_auto:
            print("\n⚠️ Advertencias de datos:")
            for w in warnings_data:
                print(f"  {w}")

        # 9. Preparar DataFrame para predicción
        X_df = pd.DataFrame([features_dict])
        X_df = X_df.reindex(columns=expected_features, fill_value=0)
        
        # 10. Realizar predicción
        dmatrix = xgb.DMatrix(X_df)
        prob_home = model.predict(dmatrix)[0]
        
        prob_home_pct = round(float(prob_home) * 100, 2)
        prob_away_pct = round(100 - prob_home_pct, 2)
        conf_label = obtener_nivel_confianza(prob_home_pct)
        
        # 11. Determinar ganador
        ganador_code = home_team if prob_home > 0.5 else away_team
        ganador_full = get_team_name(ganador_code)

        # 12. Calcular estadísticas agregadas para análisis adicional
        stats_agregadas = calcular_estadisticas_agregadas(features_dict)

        # 13. OUTPUT VISUAL ENRIQUECIDO
        if not modo_auto:
            print("\n" + "="*75)
            print("   ⚾ MLB PREDICTOR V3.5 - ANÁLISIS ESTADÍSTICO")
            print("="*75)
            
            # Nombres reales de lanzadores
            h_p_name = features_dict.get('home_pitcher_name_real', home_pitcher)
            a_p_name = features_dict.get('away_pitcher_name_real', away_pitcher)

            print(f" Encuentro: {home_team} vs {away_team}")
            print(f" Temporada: {year} | Scraping: Baseball-Reference")
            
            print(f"\n📊 COMPARATIVA DE EQUIPOS:")
            print(f" 🏠  {home_team}: OPS: {features_dict.get('home_team_OPS', 0):.3f} | Bullpen WHIP: {features_dict.get('home_bullpen_WHIP', 0):.3f}")
            print(f" ✈️  {away_team}: OPS: {features_dict.get('away_team_OPS', 0):.3f} | Bullpen WHIP: {features_dict.get('away_bullpen_WHIP', 0):.3f}")

            print(f"\n👤 LANZADORES ABRIDORES:")
            print(f" 🏠 {h_p_name}: ERA: {features_dict.get('home_starter_ERA', 0):.2f} | WHIP: {features_dict.get('home_starter_WHIP', 0):.3f} | SO9: {features_dict.get('home_starter_SO9', 0):.2f}")
            print(f" ✈️  {a_p_name}: ERA: {features_dict.get('away_starter_ERA', 0):.2f} | WHIP: {features_dict.get('away_starter_WHIP', 0):.3f} | SO9: {features_dict.get('away_starter_SO9', 0):.2f}")

            # Análisis de Bullpen
            print(f"\n🧱 ANÁLISIS DE BULLPEN:")
            print(f" 🏠 {home_team}: ERA: {features_dict.get('home_bullpen_ERA', 0):.3f} | WHIP: {features_dict.get('home_bullpen_WHIP', 0):.3f}")
            print(f" ✈️  {away_team}: ERA: {features_dict.get('away_bullpen_ERA', 0):.3f} | WHIP: {features_dict.get('away_bullpen_WHIP', 0):.3f}")

            d_era = features_dict.get('diff_bullpen_ERA', 0)
            print(f" 📊 Diferencial ERA: {d_era:+.2f}")

            # Tabla de Bateadores
            print(f"\n🔥 TOP 3 BATEADORES ANALIZADOS:")
            for label, team_key in [("🏠 " + home_team, 'home_top_3_batters_details'), 
                                    ("✈️  " + away_team, 'away_top_3_batters_details')]:
                print(f"\n {label}:")
                print(f" {'Nombre':<22} | {'BA':<5} | {'OBP':<5} | {'SLG':<5} | {'OPS':<5} | {'HR':<3} | {'RBI'}")
                print("-" * 75)
                for b in features_dict.get(team_key, []):
                    nombre = b.get('n', 'Desconocido')
                    ba = b.get('ba', 0)
                    obp = b.get('obp', 0)
                    slg = b.get('slg', 0)
                    ops = b.get('ops', b.get('o', 0))
                    hr = b.get('hr', 0)
                    rbi = b.get('rbi', 0)
                    print(f" {nombre:<22} | {ba:.3f} | {obp:.3f} | {slg:.3f} | {ops:.3f} | {int(hr):<3} | {int(rbi)}")

            # Tendencias recientes
            print(f"\n📈 TENDENCIAS RECIENTES (Últimos 10 juegos):")
            print(f" 🏠 {home_team}: Win Rate: {features_dict.get('home_win_rate_10', 0.5):.1%} | Racha: {features_dict.get('home_racha', 0):+d}")
            print(f" ✈️  {away_team}: Win Rate: {features_dict.get('away_win_rate_10', 0.5):.1%} | Racha: {features_dict.get('away_racha', 0):+d}")

            print("\n" + "="*75)
            print(f" 🏆 GANADOR PREDICHO: {ganador_full}")
            print("="*75)
            print(f" Probabilidades: {home_team} {prob_home_pct}% | {away_team} {prob_away_pct}%")
            print(f" Confianza: {conf_label}")

            # Diagnóstico de super features
            print("\n🚀 DIAGNÓSTICO DE SUPER FEATURES:")
            s_neut = features_dict.get('super_neutralizacion_whip_ops', 0)
            s_res = features_dict.get('super_resistencia_era_ops', 0)
            s_muro = features_dict.get('super_muro_bullpen', 0)
            
            n_v = home_team if s_neut < 0 else away_team
            print(f" 🛡️ Neutralización: {s_neut:.4f} (Ventaja {n_v})")
            r_v = home_team if s_res < 0 else away_team
            print(f" 📉 Resistencia:    {s_res:.4f} (Ventaja {r_v})")
            m_v = home_team if s_muro < 0 else away_team
            print(f" 🧱 Muro Bullpen:   {s_muro:.4f} (Ventaja {m_v})")
            
            # Análisis agregado
            print("\n💡 ANÁLISIS COMPUESTO:")
            print(f" Ventaja Pitcheo: {stats_agregadas.get('pitching_advantage', 0):+.3f}")
            print(f" Ventaja Bateo:   {stats_agregadas.get('batting_advantage', 0):+.3f}")
            print(f" Ventaja Momentum: {stats_agregadas.get('momentum_advantage', 0):+.3f}")
            print(f" Score Compuesto: {stats_agregadas.get('composite_advantage', 0):+.3f}")
            
            print("="*75 + "\n")

        # 14. Guardar predicción en base de datos
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS predicciones_historico 
                           (fecha TEXT, home_team TEXT, away_team TEXT, home_pitcher TEXT, 
                            away_pitcher TEXT, prob_home REAL, prob_away REAL, 
                            prediccion TEXT, confianza TEXT, tipo TEXT)''')
            
            db_data = {
                'fecha': row_data['fecha'],
                'home_team': home_team,
                'away_team': away_team,
                'home_pitcher': home_pitcher,
                'away_pitcher': away_pitcher,
                'prob_home': prob_home_pct,
                'prob_away': prob_away_pct,
                'prediccion': ganador_code,
                'confianza': conf_label,
                'tipo': 'AUTOMATICO' if modo_auto else 'MANUAL'
            }
            pd.DataFrame([db_data]).to_sql('predicciones_historico', conn, if_exists='append', index=False)
        
        return db_data 

    except Exception as e:
        print(f"❌ Error crítico en motor: {e}")
        import traceback
        traceback.print_exc()
        return None


def ejecutar_flujo_diario():
    """Ejecuta predicciones automáticas para los juegos del día"""
    print("🚀 Iniciando Motor de Predicción Diario...")
    
    if not os.path.exists(DB_PATH):
        print("❌ Error: Base de datos no encontrada.")
        return

    with sqlite3.connect(DB_PATH) as conn:
        try:
            df_hoy = pd.read_sql("SELECT * FROM historico_partidos WHERE fecha = date('now')", conn)
        except:
            print("🔭 Tabla 'historico_partidos' no encontrada.")
            return
    
    if df_hoy.empty:
        print("🔭 No hay juegos registrados para hoy.")
        return

    print(f"📅 Se encontraron {len(df_hoy)} juegos para hoy\n")
    
    resultados = []
    for idx, row in df_hoy.iterrows():
        print(f"Procesando juego {idx+1}/{len(df_hoy)}: {row['away_team']} @ {row['home_team']}")
        
        resultado = predecir_juego(
            row['home_team'], row['away_team'], 
            row['home_pitcher'], row['away_pitcher'], 
            year=row.get('year', 2026), 
            modo_auto=True
        )
        
        if resultado:
            resultados.append(resultado)
            print(f"✅ Predicción: {resultado['prediccion']} (Confianza: {resultado['confianza']})\n")
        else:
            print(f"❌ Error en predicción\n")
    
    print(f"\n✅ Proceso completado: {len(resultados)}/{len(df_hoy)} predicciones exitosas")


if __name__ == "__main__":
    ejecutar_flujo_diario()