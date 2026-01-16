"""
Actualizador de Resultados Reales MLB - REFACTORIZADO
Ejecuta al día siguiente a las 5 AM para capturar resultados finales
Versión optimizada para GitHub Actions
"""

import cloudscraper
import sqlite3
import pandas as pd
from bs4 import BeautifulSoup
import re
import time
import os
import sys
from datetime import datetime, timedelta

# Importar configuración centralizada
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_config import DB_PATH, get_team_code, SCRAPING_CONFIG

# ============================================================================
# FUNCIONES DE SOPORTE
# ============================================================================

def obtener_html(url, max_retries=None):
    """Obtiene HTML con reintentos"""
    if max_retries is None:
        max_retries = SCRAPING_CONFIG['max_retries']
    
    scraper = cloudscraper.create_scraper()
    
    for intento in range(max_retries):
        try:
            response = scraper.get(url, timeout=SCRAPING_CONFIG['timeout'])
            if response.status_code == 200:
                response.encoding = 'utf-8'
                return response.text
            time.sleep(2 ** intento)
        except Exception as e:
            if intento == max_retries - 1:
                print(f"       Error final obteniendo {url}: {e}")
            time.sleep(2 ** intento)
    
    return None


def obtener_fechas_ayer():
    """Calcula las fechas de ayer para la ejecución automática a las 5 AM"""
    ayer = datetime.now() - timedelta(days=1)
    
    # Formato para Baseball-Reference
    fecha_bref = ayer.strftime("%A, %B %-d, %Y" if os.name != 'nt' else "%A, %B %#d, %Y")
    
    # Formato para base de datos
    fecha_db = ayer.strftime("%Y-%m-%d")
    
    return fecha_bref, fecha_db, ayer.year


# ============================================================================
# PROCESO PRINCIPAL
# ============================================================================

def actualizar_resultados_reales():
    """
    Actualiza los resultados reales de los partidos de ayer
    Retorna True si procesó datos, False si no
    """
    fecha_bref, fecha_db, year_val = obtener_fechas_ayer()

    print(f"\n{'='*70}")
    print(f"🕐 Actualizando resultados reales para: {fecha_bref}")
    print(f"{'='*70}")
    
    url_schedule = f"https://www.baseball-reference.com/leagues/majors/{year_val}-schedule.shtml"
    html = obtener_html(url_schedule)
    
    if not html:
        print("❌ Error de conexión con Baseball-Reference.")
        return False

    soup = BeautifulSoup(html, 'html.parser')
    header = soup.find('h3', string=fecha_bref)
    
    if not header:
        print(f"⚠️ No hay juegos registrados para la fecha {fecha_bref} aún.")
        return False

    data_resultados = []
    cursor = header.find_next_sibling()
    partidos_procesados = 0
    
    while cursor and cursor.name == 'p' and 'game' in cursor.get('class', []):
        try:
            links = cursor.find_all('a')
            if len(links) >= 3:
                away_team_full = links[0].text.strip()
                home_team_full = links[1].text.strip()
                
                # Convertir a códigos
                away_team = get_team_code(away_team_full) or away_team_full
                home_team = get_team_code(home_team_full) or home_team_full
                
                # Extraer scores del texto
                texto_juego = cursor.get_text()
                scores = re.findall(r'\((\d+)\)', texto_juego)
                
                if len(scores) >= 2:
                    score_away = int(scores[0])
                    score_home = int(scores[1])
                    ganador = 1 if score_home > score_away else 0
                    
                    # Crear game_id consistente
                    game_id = f"{fecha_db}_{home_team}_{away_team}"
                    
                    data_resultados.append({
                        'game_id': game_id,
                        'fecha': fecha_db,
                        'year': year_val,
                        'home_team': home_team,
                        'away_team': away_team,
                        'score_home': score_home,
                        'score_away': score_away,
                        'ganador': ganador
                    })
                    
                    partidos_procesados += 1
                    print(f"✅ {away_team} @ {home_team}: {score_away}-{score_home}")
                else:
                    print(f"⚠️ Partido sin scores finales: {away_team} @ {home_team}")
                    
        except Exception as e:
            print(f"⚠️ Error procesando partido: {e}")
            import traceback
            traceback.print_exc()
        
        cursor = cursor.find_next_sibling()

    if not data_resultados:
        print("\n⚠️ No se encontraron resultados finales para procesar.")
        return False

    # Procesar y guardar resultados
    print(f"\n{'='*70}")
    print(f"💾 Procesando {len(data_resultados)} resultados...")
    
    with sqlite3.connect(DB_PATH) as conn:
        df_res = pd.DataFrame(data_resultados)
        
        # Obtener lanzadores desde lineup_ini usando game_id
        query_pitchers = f"""
            SELECT game_id, team, player as pitcher 
            FROM lineup_ini 
            WHERE fecha='{fecha_db}' AND [order]='P'
        """
        df_p = pd.read_sql(query_pitchers, conn)
        
        if df_p.empty:
            print("⚠️ No se encontraron lineups previos. Guardando sin datos de lanzadores.")
            # Podemos guardar de todas formas pero marcando que faltan pitchers
            df_res['home_pitcher'] = None
            df_res['away_pitcher'] = None
        else:
            # Merge para Home Pitcher
            df_final = df_res.merge(
                df_p[df_p['team'].isin(df_res['home_team'])], 
                left_on=['game_id', 'home_team'], 
                right_on=['game_id', 'team'], 
                how='left'
            )
            df_final = df_final.rename(columns={'pitcher': 'home_pitcher'}).drop(columns=['team'], errors='ignore')
            
            # Merge para Away Pitcher
            df_final = df_final.merge(
                df_p[df_p['team'].isin(df_res['away_team'])], 
                left_on=['game_id', 'away_team'], 
                right_on=['game_id', 'team'], 
                how='left'
            )
            df_final = df_final.rename(columns={'pitcher': 'away_pitcher'}).drop(columns=['team'], errors='ignore')
            
            df_res = df_final

        # Crear tabla si no existe
        conn.execute('''CREATE TABLE IF NOT EXISTS historico_real 
                       (game_id TEXT PRIMARY KEY, home_team TEXT, away_team TEXT, 
                        home_pitcher TEXT, away_pitcher TEXT, ganador INTEGER, 
                        year INTEGER, fecha TEXT, score_home INTEGER, score_away INTEGER)''')
        
        # Columnas finales
        columnas_finales = [
            'game_id', 'home_team', 'away_team', 'home_pitcher', 'away_pitcher', 
            'ganador', 'year', 'fecha', 'score_home', 'score_away'
        ]
        
        # Preparar export
        df_export = df_res[columnas_finales].copy()
        
        # Mostrar cuántos tienen pitchers
        con_pitchers = df_export[df_export['home_pitcher'].notna() & df_export['away_pitcher'].notna()]
        sin_pitchers = df_export[df_export['home_pitcher'].isna() | df_export['away_pitcher'].isna()]
        
        print(f"\n📊 Estadísticas:")
        print(f"   - Partidos con lanzadores: {len(con_pitchers)}")
        print(f"   - Partidos sin lanzadores: {len(sin_pitchers)}")
        
        if len(sin_pitchers) > 0:
            print(f"\n⚠️ Partidos sin datos de lanzadores:")
            for _, row in sin_pitchers.iterrows():
                print(f"      {row['away_team']} @ {row['home_team']}")
        
        # Guardar todos (INSERT OR REPLACE para evitar duplicados)
        for _, row in df_export.iterrows():
            conn.execute('''INSERT OR REPLACE INTO historico_real VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        tuple(row))
        
        conn.commit()
        
        print(f"\n✅ Se han guardado {len(df_export)} resultados reales en 'historico_real'")
        print(f"{'='*70}\n")
        
        return True


def verificar_juegos_pendientes():
    """Verifica si hay juegos que fueron capturados pero no tienen resultado"""
    with sqlite3.connect(DB_PATH) as conn:
        query = """
            SELECT hp.game_id, hp.fecha, hp.home_team, hp.away_team
            FROM historico_partidos hp
            LEFT JOIN historico_real hr ON hp.game_id = hr.game_id
            WHERE hr.game_id IS NULL
            ORDER BY hp.fecha DESC
            LIMIT 20
        """
        df_pendientes = pd.read_sql(query, conn)
        
        if not df_pendientes.empty:
            print(f"\n📋 Partidos pendientes de resultado ({len(df_pendientes)}):")
            for _, row in df_pendientes.iterrows():
                print(f"   {row['fecha']}: {row['away_team']} @ {row['home_team']}")
        else:
            print("\n✅ No hay partidos pendientes de resultado")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Actualizador de resultados reales MLB')
    parser.add_argument('--verificar', action='store_true', help='Verificar juegos pendientes')
    args = parser.parse_args()
    
    if args.verificar:
        verificar_juegos_pendientes()
    else:
        resultado = actualizar_resultados_reales()
        
        # Verificar pendientes después de actualizar
        if resultado:
            print("\n" + "="*70)
            verificar_juegos_pendientes()
        
        # Retornar código de salida para GitHub Actions
        sys.exit(0 if resultado else 1)


