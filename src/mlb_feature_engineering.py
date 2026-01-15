"""
Módulo de Ingeniería de Features para MLB Predictor
Centraliza el cálculo de features para garantizar consistencia
entre entrenamiento y predicción
"""

import pandas as pd
import numpy as np


def calcular_super_features(features_dict):
    """
    Calcula las "super features" derivadas a partir de features base.
    Debe usarse tanto en entrenamiento como en predicción.
    
    Args:
        features_dict: Diccionario con features base
        
    Returns:
        Diccionario actualizado con super features
    """
    # Valores por defecto seguros
    defaults = {
        'home_starter_WHIP': 1.3,
        'away_starter_WHIP': 1.3,
        'home_team_OPS': 0.75,
        'away_team_OPS': 0.75,
        'home_starter_ERA': 4.0,
        'away_starter_ERA': 4.0,
        'home_bullpen_WHIP': 1.3,
        'away_bullpen_WHIP': 1.3,
        'home_best_OPS': 0.85,
        'away_best_OPS': 0.85
    }
    
    # Extraer valores con fallback a defaults
    def get_safe(key):
        return features_dict.get(key, defaults.get(key, 0))
    
    # Super Feature 1: Neutralización WHIP vs OPS
    # Mide cómo el WHIP del pitcher neutraliza el OPS del equipo contrario
    home_whip = get_safe('home_starter_WHIP')
    away_whip = get_safe('away_starter_WHIP')
    home_ops = get_safe('home_team_OPS')
    away_ops = get_safe('away_team_OPS')
    
    features_dict['super_neutralizacion_whip_ops'] = (
        (home_whip * away_ops) - (away_whip * home_ops)
    )
    
    # Super Feature 2: Resistencia ERA vs OPS
    # Mide la capacidad del pitcher de resistir la ofensiva rival
    home_era = get_safe('home_starter_ERA')
    away_era = get_safe('away_starter_ERA')
    
    features_dict['super_resistencia_era_ops'] = (
        (home_era / (away_ops + 0.01)) - (away_era / (home_ops + 0.01))
    )
    
    # Super Feature 3: Muro del Bullpen
    # Mide efectividad del bullpen contra mejores bateadores rivales
    home_bull_whip = get_safe('home_bullpen_WHIP')
    away_bull_whip = get_safe('away_bullpen_WHIP')
    home_best_ops = get_safe('home_best_OPS')
    away_best_ops = get_safe('away_best_OPS')
    
    features_dict['super_muro_bullpen'] = (
        (home_bull_whip * away_best_ops) - (away_bull_whip * home_best_ops)
    )
    
    return features_dict


def validar_features(features_dict, expected_features=None):
    """
    Valida que un diccionario de features tenga los campos esperados
    
    Args:
        features_dict: Diccionario con features
        expected_features: Lista de features esperadas (opcional)
        
    Returns:
        Tuple (is_valid, missing_features, extra_features)
    """
    if expected_features is None:
        return True, [], []
    
    current_features = set(features_dict.keys())
    expected_set = set(expected_features)
    
    missing = expected_set - current_features
    extra = current_features - expected_set
    
    is_valid = len(missing) == 0
    
    return is_valid, list(missing), list(extra)


def normalizar_features_dataframe(df, expected_features):
    """
    Asegura que un DataFrame tenga exactamente las columnas esperadas
    en el orden correcto, rellenando con 0 las faltantes
    
    Args:
        df: DataFrame con features
        expected_features: Lista ordenada de features esperadas
        
    Returns:
        DataFrame normalizado
    """
    return df.reindex(columns=expected_features, fill_value=0)


def calcular_estadisticas_agregadas(features_dict):
    """
    Calcula estadísticas agregadas útiles para análisis
    
    Args:
        features_dict: Diccionario con features
        
    Returns:
        Diccionario con estadísticas adicionales
    """
    stats = {}
    
    # Ventaja general de pitcheo
    home_pitch_quality = (
        features_dict.get('home_starter_ERA', 4.0) +
        features_dict.get('home_bullpen_ERA', 4.0)
    ) / 2
    
    away_pitch_quality = (
        features_dict.get('away_starter_ERA', 4.0) +
        features_dict.get('away_bullpen_ERA', 4.0)
    ) / 2
    
    stats['pitching_advantage'] = away_pitch_quality - home_pitch_quality
    
    # Ventaja general de bateo
    stats['batting_advantage'] = (
        features_dict.get('home_team_OPS', 0.75) -
        features_dict.get('away_team_OPS', 0.75)
    )
    
    # Ventaja de momentum
    stats['momentum_advantage'] = (
        features_dict.get('home_win_rate_10', 0.5) -
        features_dict.get('away_win_rate_10', 0.5)
    )
    
    # Score compuesto
    stats['composite_advantage'] = (
        stats['pitching_advantage'] * 0.4 +
        stats['batting_advantage'] * 0.4 +
        stats['momentum_advantage'] * 0.2
    )
    
    return stats


def detectar_outliers(features_dict, thresholds=None):
    """
    Detecta valores atípicos que podrían indicar errores de scraping
    
    Args:
        features_dict: Diccionario con features
        thresholds: Dict con umbrales personalizados
        
    Returns:
        Lista de warnings sobre valores sospechosos
    """
    if thresholds is None:
        thresholds = {
            'ERA': (0.5, 10.0),
            'WHIP': (0.5, 3.0),
            'OPS': (0.4, 1.2),
            'win_rate': (0.0, 1.0),
            'racha': (-15, 15)
        }
    
    warnings = []
    
    # Validar ERAs
    for prefix in ['home_starter', 'away_starter', 'home_bullpen', 'away_bullpen']:
        era_key = f'{prefix}_ERA'
        if era_key in features_dict:
            era = features_dict[era_key]
            if era < thresholds['ERA'][0] or era > thresholds['ERA'][1]:
                warnings.append(f"⚠️ {era_key} fuera de rango: {era:.2f}")
    
    # Validar WHIPs
    for prefix in ['home_starter', 'away_starter', 'home_bullpen', 'away_bullpen']:
        whip_key = f'{prefix}_WHIP'
        if whip_key in features_dict:
            whip = features_dict[whip_key]
            if whip < thresholds['WHIP'][0] or whip > thresholds['WHIP'][1]:
                warnings.append(f"⚠️ {whip_key} fuera de rango: {whip:.3f}")
    
    # Validar OPS
    for key in ['home_team_OPS', 'away_team_OPS', 'home_best_OPS', 'away_best_OPS']:
        if key in features_dict:
            ops = features_dict[key]
            if ops < thresholds['OPS'][0] or ops > thresholds['OPS'][1]:
                warnings.append(f"⚠️ {key} fuera de rango: {ops:.3f}")
    
    return warnings


def generar_feature_importance_report(model, feature_names, top_n=20):
    """
    Genera un reporte de importancia de features
    
    Args:
        model: Modelo XGBoost entrenado
        feature_names: Lista de nombres de features
        top_n: Número de features top a mostrar
        
    Returns:
        DataFrame con ranking de importancia
    """
    try:
        importance = model.get_booster().get_score(importance_type='weight')
        
        # Convertir a DataFrame
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ]).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    except Exception as e:
        print(f"Error generando reporte de importancia: {e}")
        return pd.DataFrame()


def crear_features_interaccion(features_dict):
    """
    Crea features de interacción adicionales que podrían mejorar el modelo
    
    Args:
        features_dict: Diccionario con features base
        
    Returns:
        Diccionario actualizado con features de interacción
    """
    # Interacción pitcheo-bateo
    if 'home_starter_ERA' in features_dict and 'away_best_OPS' in features_dict:
        features_dict['home_pitch_vs_away_bat'] = (
            features_dict['home_starter_ERA'] * features_dict['away_best_OPS']
        )
    
    if 'away_starter_ERA' in features_dict and 'home_best_OPS' in features_dict:
        features_dict['away_pitch_vs_home_bat'] = (
            features_dict['away_starter_ERA'] * features_dict['home_best_OPS']
        )
    
    # Interacción racha-calidad
    if 'home_racha' in features_dict and 'home_team_OPS' in features_dict:
        features_dict['home_momentum_quality'] = (
            features_dict['home_racha'] * features_dict['home_team_OPS']
        )
    
    if 'away_racha' in features_dict and 'away_team_OPS' in features_dict:
        features_dict['away_momentum_quality'] = (
            features_dict['away_racha'] * features_dict['away_team_OPS']
        )
    
    return features_dict


if __name__ == "__main__":
    # Test de funciones
    test_features = {
        'home_starter_ERA': 3.2,
        'away_starter_ERA': 4.1,
        'home_starter_WHIP': 1.1,
        'away_starter_WHIP': 1.35,
        'home_team_OPS': 0.78,
        'away_team_OPS': 0.72,
        'home_bullpen_ERA': 3.8,
        'away_bullpen_ERA': 4.2,
        'home_bullpen_WHIP': 1.25,
        'away_bullpen_WHIP': 1.40,
        'home_best_OPS': 0.92,
        'away_best_OPS': 0.87,
        'home_win_rate_10': 0.65,
        'away_win_rate_10': 0.55,
        'home_racha': 3,
        'away_racha': -2
    }
    
    print("🧪 Testing Feature Engineering...")
    
    # Calcular super features
    result = calcular_super_features(test_features.copy())
    print("\n✅ Super Features:")
    for key in ['super_neutralizacion_whip_ops', 'super_resistencia_era_ops', 'super_muro_bullpen']:
        print(f"  {key}: {result.get(key, 0):.4f}")
    
    # Detectar outliers
    warnings = detectar_outliers(test_features)
    if warnings:
        print("\n⚠️ Warnings detectados:")
        for w in warnings:
            print(f"  {w}")
    else:
        print("\n✅ No se detectaron outliers")
    
    # Estadísticas agregadas
    stats = calcular_estadisticas_agregadas(test_features)
    print("\n📊 Estadísticas Agregadas:")
    for key, val in stats.items():
        print(f"  {key}: {val:.4f}")