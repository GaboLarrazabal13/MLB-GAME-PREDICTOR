"""
Utilidades adicionales para el Sistema MLB Predictor V3.5
Funciones helper para análisis, reportes y mantenimiento
"""

import pandas as pd
import sqlite3
import os
from datetime import datetime, timedelta
from mlb_config import DB_PATH, MODELO_PATH, CACHE_PATH, get_team_name


# ============================================================================
# ANÁLISIS DE RENDIMIENTO DEL MODELO
# ============================================================================

def analizar_accuracy_historico(dias=30):
    """
    Analiza el accuracy del modelo en predicciones recientes
    
    Args:
        dias: Número de días hacia atrás para analizar
        
    Returns:
        DataFrame con estadísticas de rendimiento
    """
    if not os.path.exists(DB_PATH):
        print("❌ Base de datos no encontrada")
        return None
    
    fecha_limite = datetime.now() - timedelta(days=dias)
    
    with sqlite3.connect(DB_PATH) as conn:
        # Obtener predicciones
        query_pred = f"""
            SELECT * FROM predicciones_historico 
            WHERE fecha >= '{fecha_limite.strftime('%Y-%m-%d')}'
            ORDER BY fecha DESC
        """
        df_pred = pd.read_sql(query_pred, conn)
        
        # Obtener resultados reales
        query_real = f"""
            SELECT * FROM historico_real 
            WHERE fecha >= '{fecha_limite.strftime('%Y-%m-%d')}'
        """
        df_real = pd.read_sql(query_real, conn)
    
    if df_pred.empty or df_real.empty:
        print(f"⚠️ No hay datos suficientes para los últimos {dias} días")
        return None
    
    # Merge para comparar
    df_pred['match_key'] = df_pred['fecha'] + '_' + df_pred['home_team'] + '_' + df_pred['away_team']
    df_real['match_key'] = df_real['fecha'].astype(str) + '_' + df_real['home_team'] + '_' + df_real['away_team']
    
    merged = df_pred.merge(df_real[['match_key', 'ganador']], on='match_key', how='inner')
    
    if merged.empty:
        print("⚠️ No se pudieron emparejar predicciones con resultados")
        return None
    
    # Calcular aciertos
    merged['acierto'] = merged.apply(
        lambda row: (row['prediccion'] == row['home_team'] and row['ganador'] == 1) or
                   (row['prediccion'] == row['away_team'] and row['ganador'] == 0),
        axis=1
    )
    
    # Estadísticas generales
    total = len(merged)
    aciertos = merged['acierto'].sum()
    accuracy = (aciertos / total * 100) if total > 0 else 0
    
    # Estadísticas por confianza
    stats_confianza = merged.groupby('confianza').agg({
        'acierto': ['count', 'sum', 'mean']
    }).round(3)
    
    print("\n" + "="*60)
    print(f"📊 ANÁLISIS DE RENDIMIENTO - Últimos {dias} días")
    print("="*60)
    print(f"Total de predicciones: {total}")
    print(f"Aciertos: {aciertos}")
    print(f"Accuracy General: {accuracy:.2f}%")
    print("\n📈 Accuracy por Nivel de Confianza:")
    print(stats_confianza)
    print("="*60 + "\n")
    
    return merged


def generar_reporte_equipos(equipo_code, ultimos_n=20):
    """
    Genera un reporte del rendimiento del modelo para un equipo específico
    
    Args:
        equipo_code: Código del equipo (ej: 'NYY')
        ultimos_n: Número de partidos recientes a analizar
    """
    if not os.path.exists(DB_PATH):
        print("❌ Base de datos no encontrada")
        return
    
    with sqlite3.connect(DB_PATH) as conn:
        query = f"""
            SELECT * FROM predicciones_historico 
            WHERE home_team = '{equipo_code}' OR away_team = '{equipo_code}'
            ORDER BY fecha DESC
            LIMIT {ultimos_n}
        """
        df = pd.read_sql(query, conn)
    
    if df.empty:
        print(f"⚠️ No hay predicciones para {equipo_code}")
        return
    
    team_name = get_team_name(equipo_code)
    
    print("\n" + "="*60)
    print(f"📊 REPORTE DE PREDICCIONES - {team_name} ({equipo_code})")
    print("="*60)
    
    # Estadísticas como local y visitante
    como_local = df[df['home_team'] == equipo_code]
    como_visitante = df[df['away_team'] == equipo_code]
    
    predicho_ganar_local = (como_local['prediccion'] == equipo_code).sum()
    predicho_ganar_visitante = (como_visitante['prediccion'] == equipo_code).sum()
    
    print(f"\nComo Local: {len(como_local)} juegos")
    print(f"  Predicho ganar: {predicho_ganar_local} ({predicho_ganar_local/len(como_local)*100:.1f}%)")
    
    print(f"\nComo Visitante: {len(como_visitante)} juegos")
    print(f"  Predicho ganar: {predicho_ganar_visitante} ({predicho_ganar_visitante/len(como_visitante)*100:.1f}%)")
    
    # Probabilidad promedio
    prob_promedio_local = como_local['prob_home'].mean() if len(como_local) > 0 else 0
    prob_promedio_visitante = (100 - como_visitante['prob_home']).mean() if len(como_visitante) > 0 else 0
    
    print(f"\nProbabilidad promedio como local: {prob_promedio_local:.1f}%")
    print(f"Probabilidad promedio como visitante: {prob_promedio_visitante:.1f}%")
    
    print("\n📋 Últimas 5 predicciones:")
    print("-" * 60)
    for _, row in df.head(5).iterrows():
        es_local = row['home_team'] == equipo_code
        rival = row['away_team'] if es_local else row['home_team']
        prob = row['prob_home'] if es_local else row['prob_away']
        print(f"{row['fecha']}: vs {rival} - Pred: {row['prediccion']} ({prob:.1f}%) [{row['confianza']}]")
    
    print("="*60 + "\n")


# ============================================================================
# MANTENIMIENTO DE BASE DE DATOS
# ============================================================================

def limpiar_cache():
    """Elimina el caché de features para forzar re-scraping"""
    if os.path.exists(CACHE_PATH):
        backup_path = CACHE_PATH + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.rename(CACHE_PATH, backup_path)
        print(f"✅ Caché respaldado en: {backup_path}")
        print("✅ Caché limpiado. El próximo entrenamiento volverá a scrapear datos.")
    else:
        print("⚠️ No hay caché para limpiar")


def compactar_base_datos():
    """Compacta y optimiza la base de datos SQLite"""
    if not os.path.exists(DB_PATH):
        print("❌ Base de datos no encontrada")
        return
    
    with sqlite3.connect(DB_PATH) as conn:
        # Obtener tamaño antes
        cursor = conn.cursor()
        cursor.execute("PRAGMA page_count")
        page_count_before = cursor.fetchone()[0]
        cursor.execute("PRAGMA page_size")
        page_size = cursor.fetchone()[0]
        size_before = page_count_before * page_size / 1024 / 1024  # MB
        
        # Compactar
        conn.execute("VACUUM")
        
        # Obtener tamaño después
        cursor.execute("PRAGMA page_count")
        page_count_after = cursor.fetchone()[0]
        size_after = page_count_after * page_size / 1024 / 1024  # MB
        
        print(f"✅ Base de datos compactada")
        print(f"   Antes: {size_before:.2f} MB")
        print(f"   Después: {size_after:.2f} MB")
        print(f"   Ahorro: {size_before - size_after:.2f} MB")


def verificar_integridad_db():
    """Verifica la integridad de la base de datos"""
    if not os.path.exists(DB_PATH):
        print("❌ Base de datos no encontrada")
        return
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        # Verificar integridad
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()[0]
        
        if result == "ok":
            print("✅ Integridad de base de datos: OK")
        else:
            print(f"❌ Problemas de integridad: {result}")
        
        # Listar tablas
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tablas = cursor.fetchall()
        
        print("\n📋 Tablas en la base de datos:")
        for tabla in tablas:
            cursor.execute(f"SELECT COUNT(*) FROM {tabla[0]}")
            count = cursor.fetchone()[0]
            print(f"   {tabla[0]}: {count} registros")


# ============================================================================
# EXPORTACIÓN DE DATOS
# ============================================================================

def exportar_predicciones_csv(output_path='predicciones_export.csv', dias=30):
    """
    Exporta predicciones recientes a CSV
    
    Args:
        output_path: Ruta del archivo CSV de salida
        dias: Número de días hacia atrás para exportar
    """
    if not os.path.exists(DB_PATH):
        print("❌ Base de datos no encontrada")
        return
    
    fecha_limite = datetime.now() - timedelta(days=dias)
    
    with sqlite3.connect(DB_PATH) as conn:
        query = f"""
            SELECT * FROM predicciones_historico 
            WHERE fecha >= '{fecha_limite.strftime('%Y-%m-%d')}'
            ORDER BY fecha DESC
        """
        df = pd.read_sql(query, conn)
    
    if df.empty:
        print(f"⚠️ No hay predicciones en los últimos {dias} días")
        return
    
    df.to_csv(output_path, index=False)
    print(f"✅ {len(df)} predicciones exportadas a: {output_path}")


def exportar_resultados_csv(output_path='resultados_export.csv', dias=30):
    """
    Exporta resultados reales a CSV
    
    Args:
        output_path: Ruta del archivo CSV de salida
        dias: Número de días hacia atrás para exportar
    """
    if not os.path.exists(DB_PATH):
        print("❌ Base de datos no encontrada")
        return
    
    fecha_limite = datetime.now() - timedelta(days=dias)
    
    with sqlite3.connect(DB_PATH) as conn:
        query = f"""
            SELECT * FROM historico_real 
            WHERE fecha >= '{fecha_limite.strftime('%Y-%m-%d')}'
            ORDER BY fecha DESC
        """
        df = pd.read_sql(query, conn)
    
    if df.empty:
        print(f"⚠️ No hay resultados en los últimos {dias} días")
        return
    
    df.to_csv(output_path, index=False)
    print(f"✅ {len(df)} resultados exportados a: {output_path}")


# ============================================================================
# MONITOREO DEL MODELO
# ============================================================================

def verificar_estado_modelo():
    """Verifica el estado del modelo y archivos necesarios"""
    print("\n" + "="*60)
    print("🔍 VERIFICACIÓN DEL SISTEMA MLB PREDICTOR")
    print("="*60)
    
    # Modelo
    if os.path.exists(MODELO_PATH):
        size = os.path.getsize(MODELO_PATH) / 1024  # KB
        mtime = datetime.fromtimestamp(os.path.getmtime(MODELO_PATH))
        print(f"✅ Modelo: {MODELO_PATH}")
        print(f"   Tamaño: {size:.2f} KB")
        print(f"   Última modificación: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"❌ Modelo NO encontrado: {MODELO_PATH}")
    
    # Base de datos
    if os.path.exists(DB_PATH):
        size = os.path.getsize(DB_PATH) / 1024 / 1024  # MB
        print(f"\n✅ Base de datos: {DB_PATH}")
        print(f"   Tamaño: {size:.2f} MB")
    else:
        print(f"\n❌ Base de datos NO encontrada: {DB_PATH}")
    
    # Caché
    if os.path.exists(CACHE_PATH):
        size = os.path.getsize(CACHE_PATH) / 1024 / 1024  # MB
        mtime = datetime.fromtimestamp(os.path.getmtime(CACHE_PATH))
        print(f"\n✅ Caché: {CACHE_PATH}")
        print(f"   Tamaño: {size:.2f} MB")
        print(f"   Última actualización: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"\n⚠️ Caché NO encontrado: {CACHE_PATH}")
    
    print("="*60 + "\n")


# ============================================================================
# EJECUCIÓN DE UTILIDADES
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python mlb_utils.py <comando> [argumentos]")
        print("\nComandos disponibles:")
        print("  accuracy [dias]           - Analizar accuracy del modelo")
        print("  equipo <codigo> [n]       - Reporte de un equipo")
        print("  limpiar_cache             - Limpiar caché de features")
        print("  compactar                 - Compactar base de datos")
        print("  verificar                 - Verificar integridad DB")
        print("  exportar_pred [dias]      - Exportar predicciones a CSV")
        print("  exportar_real [dias]      - Exportar resultados a CSV")
        print("  estado                    - Verificar estado del sistema")
        sys.exit(0)
    
    comando = sys.argv[1].lower()
    
    if comando == 'accuracy':
        dias = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        analizar_accuracy_historico(dias)
    
    elif comando == 'equipo':
        if len(sys.argv) < 3:
            print("❌ Debes especificar el código del equipo")
        else:
            codigo = sys.argv[2].upper()
            n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
            generar_reporte_equipos(codigo, n)
    
    elif comando == 'limpiar_cache':
        limpiar_cache()
    
    elif comando == 'compactar':
        compactar_base_datos()
    
    elif comando == 'verificar':
        verificar_integridad_db()
    
    elif comando == 'exportar_pred':
        dias = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        exportar_predicciones_csv(dias=dias)
    
    elif comando == 'exportar_real':
        dias = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        exportar_resultados_csv(dias=dias)
    
    elif comando == 'estado':
        verificar_estado_modelo()
    
    else:
        print(f"❌ Comando desconocido: {comando}")