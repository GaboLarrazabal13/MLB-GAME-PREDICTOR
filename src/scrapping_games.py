import pandas as pd
import cloudscraper
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta

# --- Mapeo de Nombres Completos de Equipo a Códigos de 3 Letras (MLB) ---
# Se define fuera de la función para mayor eficiencia y claridad.
TEAM_CODES = {
    'Arizona Diamondbacks': 'ARI',
    'Atlanta Braves': 'ATL',
    'Baltimore Orioles': 'BAL',
    'Boston Red Sox': 'BOS',
    'Chicago Cubs': 'CHC',
    'Chicago White Sox': 'CHW',
    'Cincinnati Reds': 'CIN',
    'Cleveland Guardians': 'CLE',  # Usamos Guardians, si el dato es de 2025
    'Colorado Rockies': 'COL',
    'Detroit Tigers': 'DET',
    'Houston Astros': 'HOU',
    'Kansas City Royals': 'KCR',
    'Los Angeles Angels': 'LAA',
    'Los Angeles Dodgers': 'LAD',
    'Miami Marlins': 'MIA',
    'Milwaukee Brewers': 'MIL',
    'Minnesota Twins': 'MIN',
    'New York Mets': 'NYM',
    'New York Yankees': 'NYY',
    'Oakland Athletics': 'OAK',
    'Philadelphia Phillies': 'PHI',
    'Pittsburgh Pirates': 'PIT',
    'San Diego Padres': 'SDP',
    'Seattle Mariners': 'SEA',
    'San Francisco Giants': 'SFG',
    'St. Louis Cardinals': 'STL',
    'Tampa Bay Rays': 'TBR',
    'Texas Rangers': 'TEX',
    'Toronto Blue Jays': 'TOR',
    'Washington Nationals': 'WSN',
    # Para equipos que puedan aparecer con nombre abreviado en BR (aunque raro en resúmenes)
    'Diamondbacks': 'ARI',
    'Braves': 'ATL',
    'Orioles': 'BAL',
    'Red Sox': 'BOS',
    'Cubs': 'CHC',
    'White Sox': 'CHW',
    'Reds': 'CIN',
    'Guardians': 'CLE',
    'Rockies': 'COL',
    'Tigers': 'DET',
    'Astros': 'HOU',
    'Royals': 'KCR',
    'Angels': 'LAA',
    'Dodgers': 'LAD',
    'Marlins': 'MIA',
    'Brewers': 'MIL',
    'Twins': 'MIN',
    'Mets': 'NYM',
    'Yankees': 'NYY',
    'Athletics': 'OAK',
    'Phillies': 'PHI',
    'Pirates': 'PIT',
    'Padres': 'SDP',
    'Mariners': 'SEA',
    'Giants': 'SFG',
    'Cardinals': 'STL',
    'Rays': 'TBR',
    'Rangers': 'TEX',
    'Blue Jays': 'TOR',
    'Nationals': 'WSN'
}

def obtener_codigo_equipo(nombre_completo):
    """Devuelve el código de 3 letras basado en el nombre del equipo."""
    return TEAM_CODES.get(nombre_completo, 'UNKNOWN')


def extraer_datos_dia(fecha):
    """
    Extrae los resultados de los juegos de béisbol para una fecha específica,
    aplicando el renombre y la adición de columnas de códigos de equipo.
    """
    
    url = f'https://www.baseball-reference.com/boxes/index.fcgi?date={fecha}' 
    scraper = cloudscraper.create_scraper()
    
    try:
        print(f"📡 Accediendo a los resultados del {fecha}...")
        response = scraper.get(url, timeout=15)
        if response.status_code != 200: 
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        bloques_juego = soup.find_all('div', class_=re.compile(r'game_summary nohover'))
        
        resultados = []

        for bloque in bloques_juego:
            # --- 1. EXTRAER EQUIPOS Y SCORES ---
            tabla_equipos = bloque.find('table', class_='teams')
            if not tabla_equipos: continue 
                
            filas = tabla_equipos.find_all('tr')
            
            # Fila 0: Visitante | Fila 1: Local
            equipo_v_name = filas[0].find('a').get_text(strip=True)
            score_v_tag = filas[0].find('td', class_='right')
            score_v = score_v_tag.get_text(strip=True) if score_v_tag else 'N/A'
            
            equipo_l_name = filas[1].find('a').get_text(strip=True)
            score_l_tag = filas[1].find('td', class_='right')
            score_l = score_l_tag.get_text(strip=True) if score_l_tag else 'N/A'


            # --- 2. EXTRACCIÓN DE LANZADORES (Omitido por brevedad, se mantiene igual) ---
            tablas = bloque.find_all('table')
            tabla_pitchers = tablas[1] if len(tablas) > 1 else None
            
            w_pitcher, l_pitcher, s_pitcher = "N/A", "N/A", "N/A"
            if tabla_pitchers:
                for fila_p in tabla_pitchers.find_all('tr'):
                    celdas = fila_p.find_all('td')
                    if len(celdas) >= 2:
                        rol = celdas[0].get_text(strip=True)
                        nombre_raw = celdas[1].get_text(strip=True)
                        nombre_limpio = re.sub(r'\s*\([^)]*\)', '', nombre_raw)
                        
                        if rol == 'W': w_pitcher = nombre_limpio
                        elif rol == 'L': l_pitcher = nombre_limpio
                        elif rol == 'S': s_pitcher = nombre_limpio

            # --- 3. CÁLCULO DE 'ganador' ---
            ganador_col = None
            try:
                r_v_clean = re.sub(r'[\s\S]*?(\d+)$', r'\1', score_v).strip()
                r_l_clean = re.sub(r'[\s\S]*?(\d+)$', r'\1', score_l).strip()
                
                r_v_int = int(r_v_clean)
                r_l_int = int(r_l_clean)
                
                if r_l_int > r_v_int:
                    ganador_col = 1
                elif r_v_int > r_l_int:
                    ganador_col = 0
            except (ValueError, TypeError):
                pass
            
            # --- 4. APLICACIÓN DE NUEVAS REGLAS DE COLUMNAS ---
            resultados.append({
                'date': fecha,
                # NUEVAS COLUMNAS DE CÓDIGO
                'away_team': obtener_codigo_equipo(equipo_v_name), 
                'a_team_name': equipo_v_name, # Columna renombrada
                'R_A': score_v,             
                # NUEVAS COLUMNAS DE CÓDIGO
                'home_team': obtener_codigo_equipo(equipo_l_name), 
                'h_team_name': equipo_l_name, # Columna renombrada
                'R_H': score_l,             
                'Ganador_Pitcher': w_pitcher, 
                'Perdedor_Pitcher': l_pitcher,
                'Salvado_Pitcher': s_pitcher,
                'ganador': ganador_col 
            })

        # Aplicamos el reordenamiento final para asegurar el orden:
        df_temp = pd.DataFrame(resultados)
        
        columnas_ordenadas = [
            'date', 'away_team', 'a_team_name', 'R_A', 
            'home_team', 'h_team_name', 'R_H', 
            'Ganador_Pitcher', 'Perdedor_Pitcher', 'Salvado_Pitcher', 'ganador'
        ]
        
        if not df_temp.empty:
            return df_temp[columnas_ordenadas]
        return df_temp


    except Exception as e:
        print(f"⚠️ Error general al procesar {fecha}: {e}")
        return None

def extraer_rango_fechas(fecha_inicio_str, fecha_fin_str):
    """
    Itera sobre un rango de fechas y consolida los resultados en un solo DataFrame.
    """
    
    fecha_inicio = datetime.strptime(fecha_inicio_str, '%Y-%m-%d')
    fecha_fin = datetime.strptime(fecha_fin_str, '%Y-%m-%d')
    
    delta = timedelta(days=1)
    fecha_actual = fecha_inicio
    
    df_final = pd.DataFrame()
    
    print(f"⌛ Iniciando extracción de {fecha_inicio_str} a {fecha_fin_str}...")
    
    while fecha_actual <= fecha_fin:
        fecha_str = fecha_actual.strftime('%Y-%m-%d')
        
        df_dia = extraer_datos_dia(fecha_str)
        
        if df_dia is not None and not df_dia.empty:
            df_final = pd.concat([df_final, df_dia], ignore_index=True)
            
        fecha_actual += delta
        
    return df_final


# --- PRUEBA DE FUNCIONAMIENTO ---

FECHA_PRUEBA_INICIO = "2025-03-27" 
FECHA_PRUEBA_FIN = "2025-09-28"

df_resultados = extraer_rango_fechas(FECHA_PRUEBA_INICIO, FECHA_PRUEBA_FIN)
df_resultados['Salvado_Pitcher'] = df_resultados['Salvado_Pitcher'].replace('N/A', 'NotApplyed')


if not df_resultados.empty:
    print("\n✅ Extracción de datos completada y unificada.")
    print(f"Datos  Guardados en csv: './data/mlb_game_results_{FECHA_PRUEBA_INICIO}_to_{FECHA_PRUEBA_FIN}.csv'")
    df_resultados.to_csv(f'./data/mlb_game_results_{FECHA_PRUEBA_INICIO}_to_{FECHA_PRUEBA_FIN}.csv', index=False)
    print("--- 5 Primeras Filas del DataFrame Final (Con Códigos y Nombres Nuevos) ---")
    print(df_resultados.head().to_string())
    print(f"\nNúmero total de juegos extraídos: {len(df_resultados)}")
    print("\nColumnas del DataFrame (Confirmando el nuevo orden):")
    print(df_resultados.columns.tolist())
else:
    print("\n❌ No se pudieron extraer datos en el rango especificado.")