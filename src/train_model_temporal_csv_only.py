"""
Script de Entrenamiento ML para Predicción MLB - Versión con Features Temporales
Usa SOLO datos del CSV (sin scraping)
Calcula tendencias de últimos 10 partidos por equipo
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# FUNCIONES DE FEATURE ENGINEERING TEMPORAL
# ============================================================================

def calcular_tendencias_equipo(df, team, fecha_limite, ventana=10):
    """
    Calcula tendencias de un equipo en sus últimos N partidos
    
    Args:
        df: DataFrame con todos los partidos
        team: Código del equipo
        fecha_limite: Fecha del partido actual (no incluir este)
        ventana: Número de partidos históricos a considerar
    
    Returns:
        dict con tendencias calculadas
    """
    # Convertir fecha_limite a datetime si es string
    if isinstance(fecha_limite, str):
        fecha_limite = pd.to_datetime(fecha_limite)
    
    # Filtrar partidos anteriores donde el equipo jugó
    partidos_equipo = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (pd.to_datetime(df['fecha']) < fecha_limite)
    ].copy()
    
    # Ordenar por fecha descendente y tomar últimos N
    partidos_equipo = partidos_equipo.sort_values('fecha', ascending=False).head(ventana)
    
    # Si no hay suficientes partidos históricos
    if len(partidos_equipo) == 0:
        return {
            'victorias_recientes': 0.5,  # Neutral
            'carreras_anotadas_avg': 4.5,  # Liga promedio
            'carreras_recibidas_avg': 4.5,
            'racha_actual': 0,
            'partidos_disponibles': 0,
            'diferencial_carreras': 0
        }
    
    # Calcular victorias
    victorias = 0
    carreras_anotadas = []
    carreras_recibidas = []
    racha_actual = 0
    ultimo_resultado = None
    
    for idx, partido in partidos_equipo.iterrows():
        # Determinar si jugó local o visitante
        es_local = (partido['home_team'] == team)
        
        # Determinar si ganó
        if es_local:
            gano = (partido['ganador'] == 1)
            carreras_anotadas.append(partido['score_home'])
            carreras_recibidas.append(partido['score_away'])
        else:
            gano = (partido['ganador'] == 0)
            carreras_anotadas.append(partido['score_away'])
            carreras_recibidas.append(partido['score_home'])
        
        if gano:
            victorias += 1
        
        # Calcular racha (secuencia de victorias/derrotas)
        if ultimo_resultado is None:
            ultimo_resultado = gano
            racha_actual = 1 if gano else -1
        elif ultimo_resultado == gano:
            if gano:
                racha_actual += 1
            else:
                racha_actual -= 1
        else:
            ultimo_resultado = gano
            racha_actual = 1 if gano else -1
    
    # Calcular promedios
    n_partidos = len(partidos_equipo)
    tasa_victorias = victorias / n_partidos if n_partidos > 0 else 0.5
    avg_anotadas = np.mean(carreras_anotadas) if carreras_anotadas else 4.5
    avg_recibidas = np.mean(carreras_recibidas) if carreras_recibidas else 4.5
    diferencial = avg_anotadas - avg_recibidas
    
    return {
        'victorias_recientes': tasa_victorias,
        'carreras_anotadas_avg': avg_anotadas,
        'carreras_recibidas_avg': avg_recibidas,
        'racha_actual': racha_actual,
        'partidos_disponibles': n_partidos,
        'diferencial_carreras': diferencial
    }


def calcular_tendencias_lanzador(df, equipo, lanzador, fecha_limite, ventana=5):
    """
    Calcula tendencias de un lanzador en sus últimas aperturas
    
    Args:
        df: DataFrame con todos los partidos
        equipo: Código del equipo
        lanzador: Nombre del lanzador
        fecha_limite: Fecha del partido actual
        ventana: Número de aperturas a considerar
    
    Returns:
        dict con tendencias del lanzador
    """
    if isinstance(fecha_limite, str):
        fecha_limite = pd.to_datetime(fecha_limite)
    
    # Filtrar partidos donde este lanzador abrió para este equipo
    partidos_lanzador = df[
        (
            ((df['home_team'] == equipo) & (df['home_pitcher'] == lanzador)) |
            ((df['away_team'] == equipo) & (df['away_pitcher'] == lanzador))
        ) &
        (pd.to_datetime(df['fecha']) < fecha_limite)
    ].copy()
    
    partidos_lanzador = partidos_lanzador.sort_values('fecha', ascending=False).head(ventana)
    
    if len(partidos_lanzador) == 0:
        return {
            'victorias_lanzador': 0.5,
            'carreras_permitidas_avg': 4.5,
            'aperturas_recientes': 0
        }
    
    victorias = 0
    carreras_permitidas = []
    
    for idx, partido in partidos_lanzador.iterrows():
        es_local = (partido['home_team'] == equipo)
        
        if es_local:
            gano = (partido['ganador'] == 1)
            carreras_permitidas.append(partido['score_away'])
        else:
            gano = (partido['ganador'] == 0)
            carreras_permitidas.append(partido['score_home'])
        
        if gano:
            victorias += 1
    
    n_aperturas = len(partidos_lanzador)
    tasa_victorias = victorias / n_aperturas if n_aperturas > 0 else 0.5
    avg_permitidas = np.mean(carreras_permitidas) if carreras_permitidas else 4.5
    
    return {
        'victorias_lanzador': tasa_victorias,
        'carreras_permitidas_avg': avg_permitidas,
        'aperturas_recientes': n_aperturas
    }


def calcular_historial_enfrentamientos(df, team1, team2, fecha_limite, ventana=10):
    """
    Calcula historial de enfrentamientos entre dos equipos
    
    Args:
        df: DataFrame con todos los partidos
        team1: Equipo de referencia (típicamente local)
        team2: Equipo oponente
        fecha_limite: Fecha del partido actual
        ventana: Número de enfrentamientos a considerar
    
    Returns:
        dict con historial
    """
    if isinstance(fecha_limite, str):
        fecha_limite = pd.to_datetime(fecha_limite)
    
    # Filtrar enfrentamientos entre estos equipos
    enfrentamientos = df[
        (
            ((df['home_team'] == team1) & (df['away_team'] == team2)) |
            ((df['home_team'] == team2) & (df['away_team'] == team1))
        ) &
        (pd.to_datetime(df['fecha']) < fecha_limite)
    ].copy()
    
    enfrentamientos = enfrentamientos.sort_values('fecha', ascending=False).head(ventana)
    
    if len(enfrentamientos) == 0:
        return {
            'victorias_team1': 0.5,
            'enfrentamientos_previos': 0,
            'carreras_avg_team1': 4.5,
            'carreras_avg_team2': 4.5
        }
    
    victorias_team1 = 0
    carreras_team1 = []
    carreras_team2 = []
    
    for idx, partido in enfrentamientos.iterrows():
        team1_es_local = (partido['home_team'] == team1)
        
        if team1_es_local:
            gano_team1 = (partido['ganador'] == 1)
            carreras_team1.append(partido['score_home'])
            carreras_team2.append(partido['score_away'])
        else:
            gano_team1 = (partido['ganador'] == 0)
            carreras_team1.append(partido['score_away'])
            carreras_team2.append(partido['score_home'])
        
        if gano_team1:
            victorias_team1 += 1
    
    n_enfrentamientos = len(enfrentamientos)
    tasa_victorias = victorias_team1 / n_enfrentamientos if n_enfrentamientos > 0 else 0.5
    
    return {
        'victorias_team1': tasa_victorias,
        'enfrentamientos_previos': n_enfrentamientos,
        'carreras_avg_team1': np.mean(carreras_team1) if carreras_team1 else 4.5,
        'carreras_avg_team2': np.mean(carreras_team2) if carreras_team2 else 4.5
    }


def extraer_features_temporales(df, idx, ventana_equipo=10, ventana_lanzador=5, ventana_h2h=10):
    """
    Extrae features temporales para un partido específico
    
    Args:
        df: DataFrame completo
        idx: Índice del partido
        ventana_equipo: Partidos recientes a considerar por equipo
        ventana_lanzador: Aperturas recientes del lanzador
        ventana_h2h: Enfrentamientos head-to-head a considerar
    
    Returns:
        dict con features calculadas
    """
    row = df.iloc[idx]
    
    # Tendencias de equipos
    tend_home = calcular_tendencias_equipo(df, row['home_team'], row['fecha'], ventana_equipo)
    tend_away = calcular_tendencias_equipo(df, row['away_team'], row['fecha'], ventana_equipo)
    
    # Tendencias de lanzadores
    tend_pitcher_home = calcular_tendencias_lanzador(
        df, row['home_team'], row['home_pitcher'], row['fecha'], ventana_lanzador
    )
    tend_pitcher_away = calcular_tendencias_lanzador(
        df, row['away_team'], row['away_pitcher'], row['fecha'], ventana_lanzador
    )
    
    # Historial de enfrentamientos
    h2h = calcular_historial_enfrentamientos(
        df, row['home_team'], row['away_team'], row['fecha'], ventana_h2h
    )
    
    # Compilar features
    features = {
        # Tendencias equipo local
        'home_victorias_L10': tend_home['victorias_recientes'],
        'home_runs_anotadas_L10': tend_home['carreras_anotadas_avg'],
        'home_runs_recibidas_L10': tend_home['carreras_recibidas_avg'],
        'home_racha': tend_home['racha_actual'],
        'home_run_diff_L10': tend_home['diferencial_carreras'],
        
        # Tendencias equipo visitante
        'away_victorias_L10': tend_away['victorias_recientes'],
        'away_runs_anotadas_L10': tend_away['carreras_anotadas_avg'],
        'away_runs_recibidas_L10': tend_away['carreras_recibidas_avg'],
        'away_racha': tend_away['racha_actual'],
        'away_run_diff_L10': tend_away['diferencial_carreras'],
        
        # Tendencias lanzador local
        'home_pitcher_victorias_L5': tend_pitcher_home['victorias_lanzador'],
        'home_pitcher_runs_permitidas_L5': tend_pitcher_home['carreras_permitidas_avg'],
        
        # Tendencias lanzador visitante
        'away_pitcher_victorias_L5': tend_pitcher_away['victorias_lanzador'],
        'away_pitcher_runs_permitidas_L5': tend_pitcher_away['carreras_permitidas_avg'],
        
        # Head-to-head
        'h2h_home_win_rate': h2h['victorias_team1'],
        'h2h_home_runs_avg': h2h['carreras_avg_team1'],
        'h2h_away_runs_avg': h2h['carreras_avg_team2'],
        
        # Features derivadas (comparativas)
        'victorias_diff': tend_home['victorias_recientes'] - tend_away['victorias_recientes'],
        'run_diff_diff': tend_home['diferencial_carreras'] - tend_away['diferencial_carreras'],
        'pitcher_win_rate_diff': tend_pitcher_home['victorias_lanzador'] - tend_pitcher_away['victorias_lanzador'],
        'pitcher_runs_allowed_diff': tend_pitcher_away['carreras_permitidas_avg'] - tend_pitcher_home['carreras_permitidas_avg'],
        'racha_diff': tend_home['racha_actual'] - tend_away['racha_actual'],
    }
    
    return features


# ============================================================================
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ============================================================================

def entrenar_modelo_temporal(csv_path='./data/processed/datos_ml_ready.csv', 
                            test_size=0.2,
                            ventana_equipo=10,
                            ventana_lanzador=5,
                            ventana_h2h=10,
                            min_partidos_historicos=20):
    """
    Entrena el modelo usando features temporales del CSV
    
    Args:
        csv_path: Ruta al CSV con datos históricos
        test_size: Proporción de datos para testing
        ventana_equipo: Partidos recientes a considerar por equipo
        ventana_lanzador: Aperturas recientes del lanzador
        ventana_h2h: Enfrentamientos head-to-head
        min_partidos_historicos: Mínimo de partidos previos para incluir
    """
    
    print("\n" + "="*80)
    print(" ENTRENAMIENTO MODELO ML - PREDICTOR MLB (FEATURES TEMPORALES)")
    print("="*80)
    
    # Cargar datos
    print(f"\n📂 Cargando datos desde: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Asegurar que fecha sea datetime
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # Ordenar por fecha (importante para cálculos temporales)
    df = df.sort_values('fecha').reset_index(drop=True)
    
    print(f"   Total de partidos: {len(df)}")
    print(f"   Rango de fechas: {df['fecha'].min()} a {df['fecha'].max()}")
    
    # Verificar distribución
    print(f"\n📊 Distribución de ganadores:")
    print(f"   Victorias locales (1): {(df['ganador'] == 1).sum()} ({(df['ganador'] == 1).sum()/len(df)*100:.1f}%)")
    print(f"   Victorias visitantes (0): {(df['ganador'] == 0).sum()} ({(df['ganador'] == 0).sum()/len(df)*100:.1f}%)")
    
    # Extraer features temporales
    print(f"\n🔄 Extrayendo features temporales...")
    print(f"   Ventana equipo: últimos {ventana_equipo} partidos")
    print(f"   Ventana lanzador: últimas {ventana_lanzador} aperturas")
    print(f"   Ventana H2H: últimos {ventana_h2h} enfrentamientos")
    print(f"   Mínimo partidos históricos: {min_partidos_historicos}")
    
    features_list = []
    labels = []
    indices_validos = []
    
    for idx in range(len(df)):
        if (idx + 1) % 100 == 0:
            print(f"   Progreso: {idx+1}/{len(df)} partidos procesados...")
        
        # Verificar que haya suficiente historia
        fecha_actual = df.iloc[idx]['fecha']
        partidos_previos = df[pd.to_datetime(df['fecha']) < fecha_actual]
        
        if len(partidos_previos) < min_partidos_historicos:
            continue
        
        try:
            features = extraer_features_temporales(
                df, idx, 
                ventana_equipo=ventana_equipo,
                ventana_lanzador=ventana_lanzador,
                ventana_h2h=ventana_h2h
            )
            
            features_list.append(features)
            labels.append(df.iloc[idx]['ganador'])
            indices_validos.append(idx)
            
        except Exception as e:
            print(f"   ⚠️  Error en partido {idx}: {e}")
            continue
    
    print(f"\n   ✅ Features extraídas: {len(features_list)} partidos")
    print(f"   ⚠️  Partidos excluidos por falta de historia: {len(df) - len(features_list)}")
    
    # Crear DataFrames
    X = pd.DataFrame(features_list)
    y = np.array(labels)
    
    # Manejar valores faltantes
    X = X.fillna(X.mean())
    
    print(f"\n📊 Shape final de datos:")
    print(f"   Features (X): {X.shape}")
    print(f"   Labels (y): {y.shape}")
    print(f"\n   Columnas de features ({len(X.columns)}):")
    for i, col in enumerate(X.columns, 1):
        print(f"   {i:2d}. {col}")
    
    # Estadísticas descriptivas
    print(f"\n📈 Estadísticas de features clave:")
    print(f"   Tasa victorias local L10: {X['home_victorias_L10'].mean():.3f} ± {X['home_victorias_L10'].std():.3f}")
    print(f"   Tasa victorias visitante L10: {X['away_victorias_L10'].mean():.3f} ± {X['away_victorias_L10'].std():.3f}")
    print(f"   Diff carreras local L10: {X['home_run_diff_L10'].mean():.3f} ± {X['home_run_diff_L10'].std():.3f}")
    print(f"   Tasa victorias H2H local: {X['h2h_home_win_rate'].mean():.3f} ± {X['h2h_home_win_rate'].std():.3f}")
    
    # Split train/test
    # Para datos temporales, es mejor hacer split cronológico
    # pero aquí usaremos random para comparabilidad
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\n📊 División de datos:")
    print(f"   Train: {len(X_train)} partidos ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Test:  {len(X_test)} partidos ({len(X_test)/len(X)*100:.1f}%)")
    
    # Escalar features
    print(f"\n⚙️  Escalando features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar múltiples modelos
    print(f"\n" + "="*80)
    print(" ENTRENANDO MODELOS")
    print("="*80)
    
    modelos = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    }
    
    resultados = {}
    
    for nombre, modelo in modelos.items():
        print(f"\n🤖 Entrenando {nombre}...")
        
        modelo.fit(X_train_scaled, y_train)
        
        y_pred = modelo.predict(X_test_scaled)
        y_proba = modelo.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        cv_scores = cross_val_score(modelo, X_train_scaled, y_train, cv=5)
        
        resultados[nombre] = {
            'modelo': modelo,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        print(f"   Accuracy: {accuracy*100:.2f}%")
        print(f"   ROC-AUC: {roc_auc:.4f}")
        print(f"   CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Seleccionar mejor modelo
    mejor_modelo_nombre = max(resultados.items(), key=lambda x: x[1]['accuracy'])[0]
    mejor_modelo_data = resultados[mejor_modelo_nombre]
    mejor_modelo = mejor_modelo_data['modelo']
    
    print(f"\n" + "="*80)
    print(f" MEJOR MODELO: {mejor_modelo_nombre}")
    print("="*80)
    
    print(f"\n📊 MÉTRICAS DEL MODELO:")
    print(f"   Accuracy: {mejor_modelo_data['accuracy']*100:.2f}%")
    print(f"   ROC-AUC: {mejor_modelo_data['roc_auc']:.4f}")
    print(f"   CV Score: {mejor_modelo_data['cv_mean']:.4f} (+/- {mejor_modelo_data['cv_std']:.4f})")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, mejor_modelo_data['y_pred'], 
                                target_names=['Away Win', 'Home Win']))
    
    print(f"\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, mejor_modelo_data['y_pred'])
    print(f"                 Predicted")
    print(f"                 Away  Home")
    print(f"   Actual Away   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"   Actual Home   {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Feature importance
    if hasattr(mejor_modelo, 'feature_importances_'):
        print(f"\n🔝 Top 15 Features más importantes:")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': mejor_modelo.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance.head(15).iterrows():
            print(f"   {row['feature']:40s} {row['importance']:.4f}")
    
    # Guardar modelo
    print(f"\n💾 Guardando modelo temporal...")
    
    with open('./models/mlb_model_temporal.pkl', 'wb') as f:
        pickle.dump(mejor_modelo, f)
    
    with open('./models/mlb_scaler_temporal.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('./models/mlb_feature_names_temporal.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    model_info = {
        'nombre': mejor_modelo_nombre,
        'accuracy': mejor_modelo_data['accuracy'],
        'roc_auc': mejor_modelo_data['roc_auc'],
        'cv_mean': mejor_modelo_data['cv_mean'],
        'cv_std': mejor_modelo_data['cv_std'],
        'n_features': len(X.columns),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'ventana_equipo': ventana_equipo,
        'ventana_lanzador': ventana_lanzador,
        'ventana_h2h': ventana_h2h
    }
    
    with open('./models/mlb_model_info_temporal.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"   ✅ mlb_model_temporal.pkl")
    print(f"   ✅ mlb_scaler_temporal.pkl")
    print(f"   ✅ mlb_feature_names_temporal.pkl")
    print(f"   ✅ mlb_model_info_temporal.pkl")
    
    print(f"\n" + "="*80)
    print(" ✅ ENTRENAMIENTO COMPLETADO")
    print("="*80)
    
    return mejor_modelo, scaler, list(X.columns), model_info


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    
    modelo, scaler, feature_names, info = entrenar_modelo_temporal(
        csv_path='./data/processed/datos_ml_ready.csv',
        test_size=0.2,
        ventana_equipo=10,      # Últimos 10 partidos por equipo
        ventana_lanzador=5,     # Últimas 5 aperturas del lanzador
        ventana_h2h=10,         # Últimos 10 enfrentamientos directos
        min_partidos_historicos=20  # Mínimo de historia necesaria
    )
    
    print("\n Modelo temporal listo para predicciones en vivo!")