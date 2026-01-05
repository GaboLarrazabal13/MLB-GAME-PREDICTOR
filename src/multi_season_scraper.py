import pandas as pd
import cloudscraper
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta

# --- Mapeo de Nombres Completos de Equipo a Códigos de 3 Letras (MLB) ---
TEAM_CODES = {
    'Arizona Diamondbacks': 'ARI',
    'Atlanta Braves': 'ATL',
    'Baltimore Orioles': 'BAL',
    'Boston Red Sox': 'BOS',
    'Chicago Cubs': 'CHC',
    'Chicago White Sox': 'CHW',
    'Cincinnati Reds': 'CIN',
    'Cleveland Guardians': 'CLE',
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
    # Nombres abreviados
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


            # --- 2. EXTRACCIÓN DE LANZADORES ---
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
                'away_team': obtener_codigo_equipo(equipo_v_name), 
                'a_team_name': equipo_v_name,
                'R_A': score_v,             
                'home_team': obtener_codigo_equipo(equipo_l_name), 
                'h_team_name': equipo_l_name,
                'R_H': score_l,             
                'Ganador_Pitcher': w_pitcher, 
                'Perdedor_Pitcher': l_pitcher,
                'Salvado_Pitcher': s_pitcher,
                'ganador': ganador_col 
            })

        # Reordenamiento final
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
        print(f"⚠️ Error al procesar {fecha}: {e}")
        return None


def extraer_rango_fechas(fecha_inicio_str, fecha_fin_str, mostrar_progreso=True):
    """
    Itera sobre un rango de fechas y consolida los resultados en un solo DataFrame.
    """
    
    fecha_inicio = datetime.strptime(fecha_inicio_str, '%Y-%m-%d')
    fecha_fin = datetime.strptime(fecha_fin_str, '%Y-%m-%d')
    
    delta = timedelta(days=1)
    fecha_actual = fecha_inicio
    
    df_final = pd.DataFrame()
    
    total_dias = (fecha_fin - fecha_inicio).days + 1
    dias_procesados = 0
    
    if mostrar_progreso:
        print(f"⌛ Extrayendo de {fecha_inicio_str} a {fecha_fin_str} ({total_dias} días)...")
    
    while fecha_actual <= fecha_fin:
        fecha_str = fecha_actual.strftime('%Y-%m-%d')
        
        df_dia = extraer_datos_dia(fecha_str)
        
        if df_dia is not None and not df_dia.empty:
            df_final = pd.concat([df_final, df_dia], ignore_index=True)
        
        dias_procesados += 1
        
        # Mostrar progreso cada 10 días
        if mostrar_progreso and dias_procesados % 10 == 0:
            progreso = (dias_procesados / total_dias) * 100
            print(f"   📊 Progreso: {dias_procesados}/{total_dias} días ({progreso:.1f}%)")
            
        fecha_actual += delta
    
    if mostrar_progreso:
        print(f"   ✅ Completado: {dias_procesados}/{total_dias} días procesados")
        
    return df_final


def solicitar_temporadas():
    """
    Solicita al usuario la cantidad de temporadas y sus rangos de fechas.
    
    Returns:
        list: Lista de diccionarios con información de cada temporada
    """
    print("\n" + "="*70)
    print(" DESCARGA DE TEMPORADAS MLB")
    print("="*70)
    
    # Solicitar cantidad de temporadas
    while True:
        try:
            num_temporadas = int(input("\n📋 ¿Cuántas temporadas deseas descargar? (1-10): "))
            if 1 <= num_temporadas <= 10:
                break
            print("   ❌ Por favor, ingresa un número entre 1 y 10")
        except ValueError:
            print("   ❌ Por favor, ingresa un número válido")
    
    temporadas = []
    
    # Solicitar información de cada temporada
    for i in range(num_temporadas):
        print(f"\n{'─'*70}")
        print(f"📅 TEMPORADA {i+1} de {num_temporadas}")
        print(f"{'─'*70}")
        
        # Año de la temporada
        while True:
            try:
                año = int(input(f"\nAño para la temporada {i+1}: "))
                if 2000 <= año <= 2030:
                    break
                print("   ❌ Por favor, ingresa un año entre 2000 y 2030")
            except ValueError:
                print("   ❌ Por favor, ingresa un año válido")
        
        # Fecha de inicio
        while True:
            fecha_inicio = input(f"Fecha de inicio de temporada {año} (YYYY-MM-DD): ").strip()
            try:
                datetime.strptime(fecha_inicio, '%Y-%m-%d')
                break
            except ValueError:
                print("   ❌ Formato incorrecto. Usa YYYY-MM-DD (ejemplo: 2023-03-27)")
        
        # Fecha de fin
        while True:
            fecha_fin = input(f"Fecha de fin de temporada {año} (YYYY-MM-DD): ").strip()
            try:
                fecha_fin_dt = datetime.strptime(fecha_fin, '%Y-%m-%d')
                fecha_inicio_dt = datetime.strptime(fecha_inicio, '%Y-%m-%d')
                
                if fecha_fin_dt >= fecha_inicio_dt:
                    break
                print("   ❌ La fecha de fin debe ser posterior o igual a la fecha de inicio")
            except ValueError:
                print("   ❌ Formato incorrecto. Usa YYYY-MM-DD (ejemplo: 2023-10-01)")
        
        temporadas.append({
            'año': año,
            'fecha_inicio': fecha_inicio,
            'fecha_fin': fecha_fin
        })
        
        print(f"   ✅ Temporada {año} configurada: {fecha_inicio} al {fecha_fin}")
    
    return temporadas


def procesar_temporadas(temporadas):
    """
    Procesa múltiples temporadas y las combina en un solo DataFrame.
    
    Args:
        temporadas: Lista de diccionarios con información de temporadas
    
    Returns:
        tuple: (DataFrame combinado, lista de años)
    """
    print("\n" + "="*70)
    print(" PROCESAMIENTO DE TEMPORADAS")
    print("="*70)
    
    todos_los_datos = []
    años_procesados = []
    
    for i, temp in enumerate(temporadas, 1):
        print(f"\n🔄 Procesando Temporada {i}/{len(temporadas)}: {temp['año']}")
        print(f"   Rango: {temp['fecha_inicio']} → {temp['fecha_fin']}")
        
        df_temporada = extraer_rango_fechas(
            temp['fecha_inicio'], 
            temp['fecha_fin'],
            mostrar_progreso=True
        )
        
        if not df_temporada.empty:
            # Reemplazar N/A en Salvado_Pitcher
            df_temporada['Salvado_Pitcher'] = df_temporada['Salvado_Pitcher'].replace('N/A', 'NotApplyed')
            
            todos_los_datos.append(df_temporada)
            años_procesados.append(str(temp['año']))
            
            print(f"   ✅ {len(df_temporada)} juegos extraídos de temporada {temp['año']}")
        else:
            print(f"   ⚠️  No se obtuvieron datos para temporada {temp['año']}")
    
    # Combinar todas las temporadas
    if todos_los_datos:
        df_combinado = pd.concat(todos_los_datos, ignore_index=True)
        return df_combinado, años_procesados
    else:
        return pd.DataFrame(), []


def guardar_resultados(df, años):
    """
    Guarda el DataFrame en un archivo CSV con nombre descriptivo.
    
    Args:
        df: DataFrame con los datos
        años: Lista de años procesados
    """
    if df.empty:
        print("\n❌ No hay datos para guardar")
        return
    
    # Crear nombre de archivo
    años_str = "_".join(años)
    nombre_archivo = f'./data/raw/resultados_béisbol_season_{años_str}.csv'
    
    # Guardar
    df.to_csv(nombre_archivo, index=False)
    
    print("\n" + "="*70)
    print(" ✅ EXTRACCIÓN COMPLETADA")
    print("="*70)
    print(f"\n📁 Archivo guardado: {nombre_archivo}")
    print(f"📊 Total de juegos: {len(df)}")
    print(f"📅 Temporadas incluidas: {', '.join(años)}")
    
    print("\n📋 Primeras 5 filas:")
    print(df.head().to_string())
    
    print(f"\n📊 Resumen por temporada:")
    if 'date' in df.columns:
        df['year'] = pd.to_datetime(df['date']).dt.year
        resumen = df.groupby('year').size()
        for año, cantidad in resumen.items():
            print(f"   {año}: {cantidad} juegos")
    
    print(f"\n✅ Columnas del DataFrame:")
    print(f"   {df.columns.tolist()}")


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print(" DESCARGADOR DE TEMPORADAS MLB")
    print(" Extrae resultados de partidos de Baseball Reference")
    print("="*70)
    
    # Solicitar información de temporadas
    temporadas = solicitar_temporadas()
    
    # Confirmar antes de procesar
    print("\n" + "="*70)
    print(" RESUMEN DE TEMPORADAS A DESCARGAR")
    print("="*70)
    
    for i, temp in enumerate(temporadas, 1):
        dias = (datetime.strptime(temp['fecha_fin'], '%Y-%m-%d') - 
                datetime.strptime(temp['fecha_inicio'], '%Y-%m-%d')).days + 1
        print(f"\n{i}. Temporada {temp['año']}")
        print(f"   Desde: {temp['fecha_inicio']}")
        print(f"   Hasta: {temp['fecha_fin']}")
        print(f"   Días a procesar: {dias}")
    
    confirmar = input("\n¿Deseas continuar con la descarga? (s/n): ").strip().lower()
    
    if confirmar != 's':
        print("\n Descarga cancelada por el usuario")
        exit()
    
    # Procesar temporadas
    df_final, años = procesar_temporadas(temporadas)
    
    # Guardar resultados
    if not df_final.empty:
        guardar_resultados(df_final, años)
    else:
        print("\n No se pudieron extraer datos de ninguna temporada")