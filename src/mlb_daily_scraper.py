"""
Scraper Diario de Partidos MLB - REFACTORIZADO
Ejecuta a las 10 AM y 1 PM para capturar lineups del día
Versión optimizada para GitHub Actions
"""

import cloudscraper
import sqlite3
import pandas as pd
from bs4 import BeautifulSoup, Comment
import time
import re
import unicodedata
from io import StringIO
from datetime import datetime
import sys
import os

# Importar configuración centralizada
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_config import DB_PATH, get_team_code, SCRAPING_CONFIG

# ============================================================================
# FUNCIONES DE SOPORTE Y FORMATEO
# ============================================================================

def normalizar_texto(texto):
    """Normaliza texto para comparaciones"""
    if not texto:
        return ""
    texto = str(texto).lower()
    texto = "".join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    texto = re.sub(r'[^a-z0-9]', '', texto)
    return texto


def safe_float(val):
    """Convierte a float de forma segura"""
    try:
        if pd.isna(val) or val == '-':
            return 0.0
        return float(val)
    except:
        return 0.0


def limpiar_dataframe(df):
    """Limpia dataframes de Baseball-Reference"""
    if df is None or len(df) == 0:
        return df
    name_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    df = df[~df[name_col].astype(str).str.contains(
        r'Team Totals|Rank in|^\s*$', 
        case=False, na=False, regex=True
    )]
    return df.reset_index(drop=True)


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


def obtener_fechas_ejecucion():
    """Obtiene fechas formateadas para scraping"""
    ahora = datetime.now()
    
    # Formato para Baseball-Reference (ej: "Monday, April 1, 2024")
    fecha_bref = ahora.strftime("%A, %B %-d, %Y" if os.name != 'nt' else "%A, %B %#d, %Y")
    
    # Formato para base de datos
    fecha_db = ahora.strftime("%Y-%m-%d")
    
    return fecha_bref, fecha_db, ahora.year


# ============================================================================
# EXTRACCIÓN DE DATOS
# ============================================================================

def scrape_player_stats(team_path, year):
    """Extrae estadísticas de pitcheo de un equipo"""
    match = re.search(r'/teams/([^/]+)/', team_path)
    team_code = match.group(1) if match else team_path
    
    url = f"https://www.baseball-reference.com/teams/{team_code}/{year}.shtml"
    html = obtener_html(url)
    
    if not html:
        return None
    
    soup = BeautifulSoup(html, 'html.parser')
    
    def buscar_tabla(table_id):
        """Busca tabla incluso si está en comentarios HTML"""
        tab = soup.find('table', {'id': table_id})
        if not tab:
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for c in comments:
                if f'id="{table_id}"' in c:
                    return BeautifulSoup(str(c), 'html.parser').find('table')
        return tab

    pitching_table = buscar_tabla('players_standard_pitching')
    
    if pitching_table:
        try:
            return pd.read_html(StringIO(str(pitching_table)))[0]
        except:
            pass
    
    return None


def encontrar_lanzador(pitching_df, nombre_lanzador):
    """Encuentra estadísticas de un lanzador específico"""
    if pitching_df is None or not nombre_lanzador:
        return None
    
    pitching_df = limpiar_dataframe(pitching_df)
    busqueda = normalizar_texto(nombre_lanzador)
    name_col = 'Name' if 'Name' in pitching_df.columns else pitching_df.columns[1]
    
    for _, fila in pitching_df.iterrows():
        nombre_tabla_limpio = normalizar_texto(str(fila[name_col]))
        if busqueda in nombre_tabla_limpio or nombre_tabla_limpio in busqueda:
            return {
                'ERA': safe_float(fila.get('ERA', 0)),
                'WHIP': safe_float(fila.get('WHIP', 0)),
                'H9': safe_float(fila.get('H9', 0)),
                'SO9': safe_float(fila.get('SO9', 0)),
                'W': safe_float(fila.get('W', 0)),
                'L': safe_float(fila.get('L', 0))
            }
    
    return None


def extraer_lineups_completos(box_url):
    """Extrae lineups completos de la página del boxscore"""
    html = obtener_html(f"https://www.baseball-reference.com{box_url}")
    
    if not html:
        return [], [], None, None
    
    soup = BeautifulSoup(html, 'html.parser')
    
    def obtener_datos(div_id):
        """Extrae bateadores y lanzador de un div de lineup"""
        div = soup.find('div', id=div_id)
        
        if not div:
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for c in comments:
                if f'id="{div_id}"' in c:
                    div = BeautifulSoup(str(c), 'html.parser').find('div', id=div_id)
                    break
        
        if div:
            links = div.find_all('a')
            bateadores = [a.get_text(strip=True) for a in links[:-1]]
            lanzador = links[-1].get_text(strip=True) if links else None
            return bateadores, lanzador
        
        return [], None

    b_away, p_away = obtener_datos('lineups_1')
    b_home, p_home = obtener_datos('lineups_2')
    
    return b_away, b_home, p_away, p_home


# ============================================================================
# EJECUCIÓN DIARIA
# ============================================================================

def ejecutar_pipeline_diario():
    """
    Pipeline principal para scraping diario
    Retorna True si encontró datos, False si no
    """
    fecha_bref, fecha_db, year_val = obtener_fechas_ejecucion()
    url_schedule = f"https://www.baseball-reference.com/leagues/majors/{year_val}-schedule.shtml"
    
    print(f"\n{'='*70}")
    print(f"📅 Ejecutando scraping automático para: {fecha_bref}")
    print(f"{'='*70}")
    
    html = obtener_html(url_schedule)
    
    if not html:
        print("❌ No se pudo conectar a Baseball-Reference")
        return False
    
    soup = BeautifulSoup(html, 'html.parser')
    header = soup.find('h3', string=fecha_bref)
    
    if not header:
        print(f"⚠️ No se encontraron partidos listados para hoy ({fecha_bref}).")
        print("   (Los lineups podrían no estar publicados aún)")
        return False

    data_partidos = []
    data_lineups = []
    cursor = header.find_next_sibling()
    partidos_procesados = 0
    
    while cursor and cursor.name == 'p' and 'game' in cursor.get('class', []):
        try:
            links = cursor.find_all('a')
            if len(links) >= 3:
                # Extraer información básica
                away_team_full = links[0].text.strip()
                away_path = links[0]['href']
                home_team_full = links[1].text.strip()
                home_path = links[1]['href']
                box_link = cursor.find('em').find('a')['href']
                
                # Convertir nombres completos a códigos
                away_team = get_team_code(away_team_full) or away_team_full
                home_team = get_team_code(home_team_full) or home_team_full
                
                print(f"\n⚾ Procesando: {away_team} @ {home_team}")
                
                # Extraer lineups
                b_away, b_home, p_away, p_home = extraer_lineups_completos(box_link)
                
                if not p_away or not p_home:
                    print(f"   ⚠️ Lineups no disponibles aún para este partido")
                    cursor = cursor.find_next_sibling()
                    continue
                
                # Extraer stats de lanzadores
                print(f"   🔍 Buscando stats de {p_away}...")
                s_away = encontrar_lanzador(scrape_player_stats(away_path, year_val), p_away)
                
                time.sleep(SCRAPING_CONFIG['min_delay'])
                
                print(f"   🔍 Buscando stats de {p_home}...")
                s_home = encontrar_lanzador(scrape_player_stats(home_path, year_val), p_home)
                
                # Crear game_id unificado
                game_id = f"{fecha_db}_{home_team}_{away_team}"
                
                # Guardar datos del partido
                data_partidos.append({
                    'game_id': game_id,
                    'box_score_url': box_link,  # Guardamos también la URL por si acaso
                    'fecha': fecha_db,
                    'year': year_val,
                    'away_team': away_team,
                    'home_team': home_team,
                    'away_pitcher': p_away,
                    'home_pitcher': p_home,
                    'away_starter_ERA': s_away['ERA'] if s_away else 0.0,
                    'away_starter_WHIP': s_away['WHIP'] if s_away else 0.0,
                    'away_starter_H9': s_away['H9'] if s_away else 0.0,
                    'away_starter_SO9': s_away['SO9'] if s_away else 0.0,
                    'away_starter_W': s_away['W'] if s_away else 0,
                    'away_starter_L': s_away['L'] if s_away else 0,
                    'home_starter_ERA': s_home['ERA'] if s_home else 0.0,
                    'home_starter_WHIP': s_home['WHIP'] if s_home else 0.0,
                    'home_starter_H9': s_home['H9'] if s_home else 0.0,
                    'home_starter_SO9': s_home['SO9'] if s_home else 0.0,
                    'home_starter_W': s_home['W'] if s_home else 0,
                    'home_starter_L': s_home['L'] if s_home else 0
                })

                # Guardar lineups
                for i, bat in enumerate(b_away):
                    data_lineups.append({
                        'fecha': fecha_db,
                        'game_id': game_id,
                        'team': away_team,
                        'order': str(i+1),
                        'player': bat
                    })
                
                data_lineups.append({
                    'fecha': fecha_db,
                    'game_id': game_id,
                    'team': away_team,
                    'order': 'P',
                    'player': p_away
                })
                
                for i, bat in enumerate(b_home):
                    data_lineups.append({
                        'fecha': fecha_db,
                        'game_id': game_id,
                        'team': home_team,
                        'order': str(i+1),
                        'player': bat
                    })
                
                data_lineups.append({
                    'fecha': fecha_db,
                    'game_id': game_id,
                    'team': home_team,
                    'order': 'P',
                    'player': p_home
                })
                
                partidos_procesados += 1
                print(f"   ✅ Partido procesado exitosamente")
                
                time.sleep(SCRAPING_CONFIG['min_delay'])
                
        except Exception as e:
            print(f"  ⚠️ Error en partido: {e}")
            import traceback
            traceback.print_exc()
        
        cursor = cursor.find_next_sibling()

    # Guardar en base de datos
    if data_partidos:
        print(f"\n{'='*70}")
        print(f"💾 Guardando {len(data_partidos)} partidos en la base de datos...")
        
        with sqlite3.connect(DB_PATH) as conn:
            # Crear tablas si no existen
            conn.execute('''CREATE TABLE IF NOT EXISTS historico_partidos 
                           (game_id TEXT PRIMARY KEY, box_score_url TEXT, fecha TEXT, year INTEGER,
                            away_team TEXT, home_team TEXT, away_pitcher TEXT, home_pitcher TEXT,
                            away_starter_ERA REAL, away_starter_WHIP REAL, away_starter_H9 REAL,
                            away_starter_SO9 REAL, away_starter_W INTEGER, away_starter_L INTEGER,
                            home_starter_ERA REAL, home_starter_WHIP REAL, home_starter_H9 REAL,
                            home_starter_SO9 REAL, home_starter_W INTEGER, home_starter_L INTEGER)''')
            
            conn.execute('''CREATE TABLE IF NOT EXISTS lineup_ini 
                           (fecha TEXT, game_id TEXT, team TEXT, [order] TEXT, player TEXT)''')
            
            # Guardar partidos (con INSERT OR REPLACE para evitar duplicados)
            df_partidos = pd.DataFrame(data_partidos)
            for _, row in df_partidos.iterrows():
                conn.execute('''INSERT OR REPLACE INTO historico_partidos VALUES 
                               (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                            tuple(row))
            
            # Guardar lineups
            if data_lineups:
                pd.DataFrame(data_lineups).to_sql('lineup_ini', conn, if_exists='append', index=False)
            
            conn.commit()
        
        print(f"✅ Proceso finalizado exitosamente")
        print(f"   - Partidos guardados: {len(data_partidos)}")
        print(f"   - Lineups guardados: {len(data_lineups)} jugadores")
        print(f"{'='*70}\n")
        
        return True
    else:
        print("\n⚠️ No hubo datos nuevos para procesar.")
        return False


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Scraper diario de partidos MLB')
    parser.add_argument('--retry', action='store_true', help='Modo reintentar si no hay datos')
    args = parser.parse_args()
    
    resultado = ejecutar_pipeline_diario()
    
    # Retornar código de salida para GitHub Actions
    sys.exit(0 if resultado else 1)


# import cloudscraper
# import sqlite3
# import pandas as pd
# from bs4 import BeautifulSoup, Comment
# import time
# import re
# import unicodedata
# from io import StringIO
# from datetime import datetime

# # ============================================================================
# # FUNCIONES DE SOPORTE Y FORMATEO
# # ============================================================================
# def normalizar_texto(texto):
#     if not texto: return ""
#     texto = str(texto).lower()
#     texto = "".join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
#     texto = re.sub(r'[^a-z0-9]', '', texto)
#     return texto

# def safe_float(val):
#     try:
#         if pd.isna(val) or val == '-': return 0.0
#         return float(val)
#     except: return 0.0

# def limpiar_dataframe(df):
#     if df is None or len(df) == 0: return df
#     name_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
#     df = df[~df[name_col].astype(str).str.contains(r'Team Totals|Rank in|^\s*$', case=False, na=False, regex=True)]
#     return df.reset_index(drop=True)

# def obtener_html(url):
#     scraper = cloudscraper.create_scraper()
#     try:
#         response = scraper.get(url, timeout=15)
#         if response.status_code == 200:
#             response.encoding = 'utf-8'
#             return response.text
#     except: return None
#     return None

# def obtener_fechas_ejecucion():
#     ahora = datetime.now()
#     fecha_bref = ahora.strftime("%A, %B %-d, %Y").replace(" 0", " ") 
#     fecha_db = ahora.strftime("%Y-%m-%d")
#     return fecha_bref, fecha_db, ahora.year

# # ============================================================================
# # EXTRACCIÓN DE DATOS
# # ============================================================================
# def scrape_player_stats(team_path, year):
#     match = re.search(r'/teams/([^/]+)/', team_path)
#     team_code = match.group(1) if match else team_path
#     url = f"https://www.baseball-reference.com/teams/{team_code}/{year}.shtml"
#     html = obtener_html(url)
#     if not html: return None
    
#     soup = BeautifulSoup(html, 'html.parser')
#     def buscar_tabla(table_id):
#         tab = soup.find('table', {'id': table_id})
#         if not tab:
#             comments = soup.find_all(string=lambda text: isinstance(text, Comment))
#             for c in comments:
#                 if f'id="{table_id}"' in c:
#                     return BeautifulSoup(str(c), 'html.parser').find('table')
#         return tab

#     pitching_table = buscar_tabla('players_standard_pitching')
#     if pitching_table:
#         try:
#             return pd.read_html(StringIO(str(pitching_table)))[0]
#         except: pass
#     return None

# def encontrar_lanzador(pitching_df, nombre_lanzador):
#     if pitching_df is None or not nombre_lanzador: return None
#     pitching_df = limpiar_dataframe(pitching_df)
#     busqueda = normalizar_texto(nombre_lanzador)
#     name_col = 'Name' if 'Name' in pitching_df.columns else pitching_df.columns[1]
    
#     for _, fila in pitching_df.iterrows():
#         nombre_tabla_limpio = normalizar_texto(str(fila[name_col]))
#         if busqueda in nombre_tabla_limpio or nombre_tabla_limpio in busqueda:
#             return {
#                 'ERA': safe_float(fila.get('ERA', 0)),
#                 'WHIP': safe_float(fila.get('WHIP', 0)),
#                 'H9': safe_float(fila.get('H9', 0)),
#                 'SO9': safe_float(fila.get('SO9', 0)),
#                 'W': safe_float(fila.get('W', 0)),
#                 'L': safe_float(fila.get('L', 0))
#             }
#     return None

# def extraer_lineups_completos(box_url):
#     html = obtener_html(f"https://www.baseball-reference.com{box_url}")
#     if not html: return [], [], None, None
#     soup = BeautifulSoup(html, 'html.parser')
    
#     def obtener_datos(div_id):
#         div = soup.find('div', id=div_id)
#         if not div:
#             comments = soup.find_all(string=lambda text: isinstance(text, Comment))
#             for c in comments:
#                 if f'id="{div_id}"' in c:
#                     div = BeautifulSoup(str(c), 'html.parser').find('div', id=div_id)
#                     break
#         if div:
#             links = div.find_all('a')
#             bateadores = [a.get_text(strip=True) for a in links[:-1]]
#             lanzador = links[-1].get_text(strip=True) if links else None
#             return bateadores, lanzador
#         return [], None

#     b_away, p_away = obtener_datos('lineups_1')
#     b_home, p_home = obtener_datos('lineups_2')
#     return b_away, b_home, p_away, p_home


# # ============================================================================
# # EJECUCIÓN DIARIA
# # ============================================================================
# def ejecutar_pipeline_diario():
#     fecha_bref, fecha_db, year_val = obtener_fechas_ejecucion()
#     url_schedule = f"https://www.baseball-reference.com/leagues/majors/{year_val}-schedule.shtml"
    
#     print(f"📅 Ejecutando scraping automático para: {fecha_bref}")
#     html = obtener_html(url_schedule)
#     if not html: return
    
#     soup = BeautifulSoup(html, 'html.parser')
#     header = soup.find('h3', string=fecha_bref)
    
#     if not header:
#         print(f"⚠️ No se encontraron partidos listados para hoy ({fecha_bref}).")
#         return

#     data_partidos = []
#     data_lineups = []
#     cursor = header.find_next_sibling()
    
#     while cursor and cursor.name == 'p' and 'game' in cursor.get('class', []):
#         try:
#             links = cursor.find_all('a')
#             if len(links) >= 3:
#                 away_team, away_path = links[0].text.strip(), links[0]['href']
#                 home_team, home_path = links[1].text.strip(), links[1]['href']
#                 box_link = cursor.find('em').find('a')['href']
                
#                 print(f" ⚾ Procesando: {away_team} @ {home_team}")
                
#                 b_away, b_home, p_away, p_home = extraer_lineups_completos(box_link)
#                 s_away = encontrar_lanzador(scrape_player_stats(away_path, year_val), p_away)
#                 s_home = encontrar_lanzador(scrape_player_stats(home_path, year_val), p_home)
                
#                 # MODIFICACIÓN: Se incluye game_id en historico_partidos
#                 data_partidos.append({
#                     'game_id': box_link, # ID UNIFICADO
#                     'fecha': fecha_db,
#                     'year': year_val,
#                     'away_team': away_team,
#                     'home_team': home_team,
#                     'away_pitcher': p_away,
#                     'home_pitcher': p_home,
#                     'away_starter_ERA': s_away['ERA'] if s_away else 0.0,
#                     'away_starter_WHIP': s_away['WHIP'] if s_away else 0.0,
#                     'away_starter_H9': s_away['H9'] if s_away else 0.0,
#                     'away_starter_SO9': s_away['SO9'] if s_away else 0.0,
#                     'away_starter_W': s_away['W'] if s_away else 0,
#                     'away_starter_L': s_away['L'] if s_away else 0,
#                     'home_starter_ERA': s_home['ERA'] if s_home else 0.0,
#                     'home_starter_WHIP': s_home['WHIP'] if s_home else 0.0,
#                     'home_starter_H9': s_home['H9'] if s_home else 0.0,
#                     'home_starter_SO9': s_home['SO9'] if s_home else 0.0,
#                     'home_starter_W': s_home['W'] if s_home else 0,
#                     'home_starter_L': s_home['L'] if s_home else 0
#                 })

#                 for i, bat in enumerate(b_away):
#                     data_lineups.append({'fecha': fecha_db, 'game_id': box_link, 'team': away_team, 'order': str(i+1), 'player': bat})
#                 data_lineups.append({'fecha': fecha_db, 'game_id': box_link, 'team': away_team, 'order': 'P', 'player': p_away})
                
#                 for i, bat in enumerate(b_home):
#                     data_lineups.append({'fecha': fecha_db, 'game_id': box_link, 'team': home_team, 'order': str(i+1), 'player': bat})
#                 data_lineups.append({'fecha': fecha_db, 'game_id': box_link, 'team': home_team, 'order': 'P', 'player': p_home})
                
#                 time.sleep(2)
#         except Exception as e:
#             print(f"  ⚠️ Error en partido: {e}")
#         cursor = cursor.find_next_sibling()

#     if data_partidos:
#         with sqlite3.connect('../data/mlb_reentrenamiento.db') as conn:
#             # Guardamos datos
#             pd.DataFrame(data_partidos).to_sql('historico_partidos', conn, if_exists='append', index=False)
#             if data_lineups:
#                 pd.DataFrame(data_lineups).to_sql('lineup_ini', conn, if_exists='append', index=False)
#         print(f"\n✅ Proceso finalizado. game_id sincronizado en ambas tablas.")
#     else:
#         print("No hubo datos nuevos para procesar.")

# if __name__ == "__main__":
#     ejecutar_pipeline_diario()