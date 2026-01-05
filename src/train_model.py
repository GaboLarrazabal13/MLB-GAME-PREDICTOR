"""
Script de Entrenamiento del Modelo ML para Predicción de Partidos MLB
Usa datos históricos con stats de equipos y lanzadores iniciales
Features basadas en EDA: ERA, WHIP, H9 (pitching) y BA, OBP, RBI, R (batting)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import cloudscraper
from bs4 import BeautifulSoup
import time
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# FUNCIONES DE SCRAPING (para extraer features de partidos históricos)
# ============================================================================

def obtener_html(url):
    scraper = cloudscraper.create_scraper()
    try:
        response = scraper.get(url, timeout=15)
        # AGREGA ESTO PARA DEBUGEAR:
        if response.status_code != 200:
            print(f"⚠️ Error {response.status_code} al acceder a {url}")
        
        response.encoding = 'utf-8'
        return response.text if response.status_code == 200 else None
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return None


def limpiar_dataframe(df):
    """Limpia el DataFrame eliminando filas no deseadas"""
    if df is None or len(df) == 0:
        return df
    
    if 'Rk' in df.columns:
        df = df.drop('Rk', axis=1)
    
    name_col = df.columns[0]
    df = df.dropna(subset=[name_col])
    df = df[~df[name_col].astype(str).str.contains(
        r'Team Totals|Rank in|^\s*$', 
        case=False, na=False, regex=True
    )]
    return df.reset_index(drop=True)


def scrape_player_stats(team_code, year=2025):
    """Extrae estadísticas de jugadores de un equipo"""
    url = f"https://www.baseball-reference.com/teams/{team_code}/{year}.shtml"
    
    html = obtener_html(url)
    if not html:
        return None, None
    
    soup = BeautifulSoup(html, 'html.parser')
    
    batting_table = soup.find('table', {'id': 'players_standard_batting'})
    pitching_table = soup.find('table', {'id': 'players_standard_pitching'})
    
    batting_df = pitching_df = None
    
    if batting_table:
        try:
            batting_df = pd.read_html(str(batting_table))[0]
            batting_df = limpiar_dataframe(batting_df)
        except:
            pass
    
    if pitching_table:
        try:
            pitching_df = pd.read_html(str(pitching_table))[0]
            pitching_df = limpiar_dataframe(pitching_df)
        except:
            pass
    
    return batting_df, pitching_df


def safe_float(val):
    """Convierte valores a float de forma segura"""
    try:
        return float(val)
    except:
        return 0.0


def encontrar_lanzador(pitching_df, nombre_lanzador):
    """Busca un lanzador y extrae sus stats: ERA, WHIP, H9"""
    if pitching_df is None or len(pitching_df) == 0:
        return None
    
    nombre_busqueda = nombre_lanzador.lower().strip()
    name_col = pitching_df.columns[0]
    
    # Búsqueda flexible por coincidencia
    mask = pitching_df[name_col].astype(str).str.lower().str.contains(nombre_busqueda, na=False)
    
    if mask.sum() == 0:
        # No se encontró - retornar None, el código usará fallback
        return None
    
    lanzador = pitching_df[mask].iloc[0]
    
    return {
        'ERA': safe_float(lanzador.get('ERA', 0)),
        'WHIP': safe_float(lanzador.get('WHIP', 0)),
        'H9': safe_float(lanzador.get('H9', 0)),
        'W': safe_float(lanzador.get('W', 0)),
        'L': safe_float(lanzador.get('L', 0)),
        'IP': safe_float(lanzador.get('IP', 0))
    }


def encontrar_mejor_bateador(batting_df):
    """
    Encuentra los 3 mejores bateadores según OBP
    Filtro: deben tener más turnos al bate (AB) que la mediana del equipo
    Stats: BA, OBP, RBI, R
    """
    if batting_df is None or len(batting_df) == 0:
        return None
    
    # Verificar que existan las columnas necesarias
    if 'OBP' not in batting_df.columns or 'AB' not in batting_df.columns:
        return None
    
    # Convertir a numérico
    batting_df['OBP'] = pd.to_numeric(batting_df['OBP'], errors='coerce')
    batting_df['AB'] = pd.to_numeric(batting_df['AB'], errors='coerce')
    batting_df = batting_df.dropna(subset=['OBP', 'AB'])
    
    if len(batting_df) == 0:
        return None
    
    # Calcular mediana de turnos al bate del equipo
    mediana_ab = batting_df['AB'].median()
    
    # Filtrar: solo jugadores con AB > mediana
    batting_filtrado = batting_df[batting_df['AB'] > mediana_ab].copy()
    
    if len(batting_filtrado) == 0:
        batting_filtrado = batting_df  # Usar todos si no hay suficientes
    
    # Ordenar por OBP y tomar top 3
    batting_filtrado = batting_filtrado.sort_values('OBP', ascending=False)
    top_3 = batting_filtrado.head(3)
    
    # Calcular promedios de los top 3
    stats_promedio = {
        'BA': 0,
        'OBP': 0,
        'RBI': 0,
        'R': 0
    }
    
    count = 0
    for idx, bateador in top_3.iterrows():
        ba = safe_float(bateador.get('BA', 0))
        obp = safe_float(bateador.get('OBP', 0))
        rbi = safe_float(bateador.get('RBI', 0))
        r = safe_float(bateador.get('R', 0))
        
        stats_promedio['BA'] += ba
        stats_promedio['OBP'] += obp
        stats_promedio['RBI'] += rbi
        stats_promedio['R'] += r
        count += 1
    
    # Promediar
    if count > 0:
        for key in stats_promedio:
            stats_promedio[key] /= count
    
    return stats_promedio


def calcular_stats_equipo(batting_df, pitching_df):
    """Calcula estadísticas agregadas del equipo"""
    stats = {}
    
    # Stats de batting
    if batting_df is not None and len(batting_df) > 0:
        for col in ['BA', 'OBP', 'RBI', 'R']:
            if col in batting_df.columns:
                batting_df[col] = pd.to_numeric(batting_df[col], errors='coerce')
                stats[f'team_{col}_mean'] = batting_df[col].mean()
    
    # Stats de pitching
    if pitching_df is not None and len(pitching_df) > 0:
        for col in ['ERA', 'WHIP', 'H9']:
            if col in pitching_df.columns:
                pitching_df[col] = pd.to_numeric(pitching_df[col], errors='coerce')
                stats[f'team_{col}_mean'] = pitching_df[col].mean()
    
    return stats


def extraer_features_partido(row, verbose=False):
    """Extrae features de un partido usando scraping"""
    
    if verbose and int(row.name) % 20 == 0:
        print(f"  [{row.name}] {row['home_team']} vs {row['away_team']}")
    
    # Obtener stats de ambos equipos
    batting1, pitching1 = scrape_player_stats(row['home_team'], row['year'])
    batting2, pitching2 = scrape_player_stats(row['away_team'], row['year'])
    
    if batting1 is None or batting2 is None:
        return None
    
    # Stats del equipo
    stats_team1 = calcular_stats_equipo(batting1, pitching1)
    stats_team2 = calcular_stats_equipo(batting2, pitching2)
    
    # Lanzadores
    pitcher1_stats = encontrar_lanzador(pitching1, row['home_pitcher'])
    pitcher2_stats = encontrar_lanzador(pitching2, row['away_pitcher'])
    
    # Mejores bateadores
    best_batter1 = encontrar_mejor_bateador(batting1)
    best_batter2 = encontrar_mejor_bateador(batting2)
    
    # Crear vector de features
    features = {}
    
    # Features del equipo local
    for key, val in stats_team1.items():
        features[f'home_{key}'] = val
    
    # Features del equipo visitante
    for key, val in stats_team2.items():
        features[f'away_{key}'] = val
    
    # Features de lanzadores
    if pitcher1_stats:
        features['home_pitcher_ERA'] = pitcher1_stats['ERA']
        features['home_pitcher_WHIP'] = pitcher1_stats['WHIP']
        features['home_pitcher_H9'] = pitcher1_stats['H9']
        features['home_pitcher_W'] = pitcher1_stats['W']
        features['home_pitcher_L'] = pitcher1_stats['L']
    else:
        features.update({
            'home_pitcher_ERA': 0, 'home_pitcher_WHIP': 0, 
            'home_pitcher_H9': 0, 'home_pitcher_W': 0, 'home_pitcher_L': 0
        })
    
    if pitcher2_stats:
        features['away_pitcher_ERA'] = pitcher2_stats['ERA']
        features['away_pitcher_WHIP'] = pitcher2_stats['WHIP']
        features['away_pitcher_H9'] = pitcher2_stats['H9']
        features['away_pitcher_W'] = pitcher2_stats['W']
        features['away_pitcher_L'] = pitcher2_stats['L']
    else:
        features.update({
            'away_pitcher_ERA': 0, 'away_pitcher_WHIP': 0,
            'away_pitcher_H9': 0, 'away_pitcher_W': 0, 'away_pitcher_L': 0
        })
    
    # Features de mejores bateadores
    if best_batter1:
        features['home_best_BA'] = best_batter1['BA']
        features['home_best_OBP'] = best_batter1['OBP']
        features['home_best_RBI'] = best_batter1['RBI']
        features['home_best_R'] = best_batter1['R']
    else:
        features.update({
            'home_best_BA': 0, 'home_best_OBP': 0,
            'home_best_RBI': 0, 'home_best_R': 0
        })
    
    if best_batter2:
        features['away_best_BA'] = best_batter2['BA']
        features['away_best_OBP'] = best_batter2['OBP']
        features['away_best_RBI'] = best_batter2['RBI']
        features['away_best_R'] = best_batter2['R']
    else:
        features.update({
            'away_best_BA': 0, 'away_best_OBP': 0,
            'away_best_RBI': 0, 'away_best_R': 0
        })
    
    # Features derivadas (diferencias críticas según EDA)
    features['pitcher_ERA_diff'] = features['away_pitcher_ERA'] - features['home_pitcher_ERA']
    features['pitcher_WHIP_diff'] = features['away_pitcher_WHIP'] - features['home_pitcher_WHIP']
    features['pitcher_H9_diff'] = features['away_pitcher_H9'] - features['home_pitcher_H9']
    features['team_BA_diff'] = features['home_team_BA_mean'] - features['away_team_BA_mean']
    features['team_OBP_diff'] = features['home_team_OBP_mean'] - features['away_team_OBP_mean']
    
    return features


# ============================================================================
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ============================================================================

def entrenar_modelo(csv_path='./data/processed/datos_ml_ready.csv', test_size=0.2, 
                   usar_cache=True, cache_path='./cache/features_cache.pkl'):
    """
    Entrena el modelo de predicción de partidos MLB
    
    Args:
        csv_path: Ruta al CSV con datos históricos
        test_size: Proporción de datos para testing
        usar_cache: Si True, intenta cargar features de cache
        cache_path: Ruta al archivo de cache
    """
    
    print("\n" + "="*70)
    print(" ENTRENAMIENTO DEL MODELO ML - PREDICTOR MLB")
    print("="*70)
    
    # Cargar datos
    print(f"\n📂 Cargando datos desde: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Total de partidos: {len(df)}")
    print(f"   Columnas: {list(df.columns)}")
    
    # Verificar distribución de clases
    print(f"\n📊 Distribución de ganadores:")
    print(f"   Victorias locales (1): {(df['ganador'] == 1).sum()} ({(df['ganador'] == 1).sum()/len(df)*100:.1f}%)")
    print(f"   Victorias visitantes (0): {(df['ganador'] == 0).sum()} ({(df['ganador'] == 0).sum()/len(df)*100:.1f}%)")
    
    # Intentar cargar cache de features
    if usar_cache:
        try:
            print(f"\n💾 Intentando cargar features desde cache...")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                X = cache_data['X']
                y = cache_data['y']
            print(f"   ✅ Cache cargado: {X.shape}")
        except FileNotFoundError:
            print(f"   ⚠️  No se encontró cache, extrayendo features...")
            usar_cache = False
    
    # Extraer features si no hay cache
    if not usar_cache:
        print(f"\n🔄 Extrayendo features de {len(df)} partidos...")
        print("   (Esto puede tardar varios minutos...)")
        
        features_list = []
        labels = []
        partidos_fallidos = 0
        
        for idx, row in df.iterrows():
            if (idx + 1) % 10 == 0:
                print(f"   Progreso: {idx+1}/{len(df)} partidos procesados...")
            
            features = extraer_features_partido(row, verbose=False)
            
            if features:
                features_list.append(features)
                labels.append(row['ganador'])
            else:
                partidos_fallidos += 1
            
            time.sleep(1.5)  # Ser amigable con el servidor
        
        print(f"\n   ✅ Features extraídas: {len(features_list)} partidos")
        print(f"   ⚠️  Partidos fallidos: {partidos_fallidos}")
        
        # Crear DataFrames
        X = pd.DataFrame(features_list)
        y = np.array(labels)
        
        # Guardar cache
        print(f"\n💾 Guardando features en cache...")
        with open(cache_path, 'wb') as f:
            pickle.dump({'X': X, 'y': y}, f)
        print(f"   ✅ Cache guardado: {cache_path}")
    
    # Manejar valores faltantes
    X = X.fillna(0)
    
    print(f"\n📊 Shape final de datos:")
    print(f"   Features (X): {X.shape}")
    print(f"   Labels (y): {y.shape}")
    print(f"\n   Columnas de features ({len(X.columns)}):")
    for i, col in enumerate(X.columns, 1):
        print(f"   {i:2d}. {col}")
    
    # Split train/test
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
    
    # Entrenar múltiples modelos y comparar
    print(f"\n" + "="*70)
    print(" ENTRENANDO MODELOS")
    print("="*70)
    
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
        print(f"\n Entrenando {nombre}...")
        
        # Entrenar
        modelo.fit(X_train_scaled, y_train)
        
        # Predicciones
        y_pred = modelo.predict(X_test_scaled)
        y_proba = modelo.predict_proba(X_test_scaled)[:, 1]
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        # Cross-validation
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
    
    print(f"\n" + "="*70)
    print(f" MEJOR MODELO: {mejor_modelo_nombre}")
    print("="*70)
    
    # Reporte detallado del mejor modelo
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
    
    # Feature importance (si aplica)
    if hasattr(mejor_modelo, 'feature_importances_'):
        print(f"\n🔝 Top 15 Features más importantes:")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': mejor_modelo.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance.head(15).iterrows():
            print(f"   {row['feature']:30s} {row['importance']:.4f}")
    
    # Guardar modelo
    print(f"\n💾 Guardando modelo...")
    
    with open('./models/mlb_model.pkl', 'wb') as f:
        pickle.dump(mejor_modelo, f)
    
    with open('./models/mlb_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('./models/mlb_feature_names.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    # Guardar info del modelo
    model_info = {
        'nombre': mejor_modelo_nombre,
        'accuracy': mejor_modelo_data['accuracy'],
        'roc_auc': mejor_modelo_data['roc_auc'],
        'cv_mean': mejor_modelo_data['cv_mean'],
        'cv_std': mejor_modelo_data['cv_std'],
        'n_features': len(X.columns),
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
    
    with open('./models/mlb_model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"   ✅ mlb_model.pkl")
    print(f"   ✅ mlb_scaler.pkl")
    print(f"   ✅ mlb_feature_names.pkl")
    print(f"   ✅ mlb_model_info.pkl")
    
    print(f"\n" + "="*70)
    print(" ✅ ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"\nAhora puedes usar el modelo para hacer predicciones:")
    print(f"   python predict_game.py BOS NYY Bello Cole")
    
    return mejor_modelo, scaler, list(X.columns), model_info


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    
    # Entrenar el modelo
    modelo, scaler, feature_names, info = entrenar_modelo(
        csv_path='./data/processed/datos_ml_ready.csv',
        test_size=0.2,
        usar_cache=True  # Cambiar a False para re-extraer features
    )
    
    print("\n Modelo listo para usar!")