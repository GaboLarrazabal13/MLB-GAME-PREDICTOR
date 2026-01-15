import pandas as pd
import sqlite3
import os
import time
import re

# CONFIGURACIÓN DE RUTAS
CSV_PATH = './data/processed/datos_ml_ready.csv'
DB_PATH = './data/mlb_reentrenamiento.db'

def limpiar_nombre_equipo(nombre):
    """Asegura que el código del equipo sea de 3 letras limpias."""
    nombre = str(nombre).strip().upper()
    # Mapeo rápido de excepciones comunes si las hay en tu CSV
    mapeo_especial = {'SDG': 'SDP', 'SFO': 'SFG', 'KCA': 'KCR', 'ANA': 'LAA', 'LAN': 'LAD', 'NYA': 'NYY', 'NYN': 'NYM', 'TBA': 'TBR'}
    nombre = mapeo_especial.get(nombre, nombre)
    return re.sub(r'[^A-Z]', '', nombre)[:3]

def inicializar_db():
    """Crea la estructura de tablas si la DB es nueva."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS historico_real (
            game_id TEXT PRIMARY KEY,
            home_team TEXT,
            away_team TEXT,
            home_pitcher TEXT,
            away_pitcher TEXT,
            ganador INTEGER,
            year INTEGER,
            fecha TEXT
        )
    ''')
    # Tabla de control para evitar re-entrenar lo mismo
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS control_entrenamiento (
            game_id TEXT PRIMARY KEY
        )
    ''')
    conn.commit()
    conn.close()

def importar_y_transformar():
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: No se encontró el archivo en {CSV_PATH}")
        return

    print(f"📖 Cargando datos desde {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # 1. Limpieza y Formateo
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['year'] = df['fecha'].dt.year
    
    print("🛠️ Generando Game IDs oficiales (Formato: TEAMYYYYMMDD0)...")
    
    def generar_id(row):
        team_code = limpiar_nombre_equipo(row['home_team'])
        fecha_str = row['fecha'].strftime('%Y%m%d')
        return f"{team_code}{fecha_str}0"

    df['game_id'] = df.apply(generar_id, axis=1)

    # 2. Conectar e Insertar
    inicializar_db()
    conn = sqlite3.connect(DB_PATH)
    
    print(f"🚀 Intentando integrar {len(df)} juegos a la base de datos...")
    
    registros_nuevos = 0
    errores = 0

    for i, row in df.iterrows():
        try:
            # Insertar en la tabla de juegos (Ignorar si ya existe el ID)
            conn.execute('''
                INSERT OR IGNORE INTO historico_real 
                (game_id, home_team, away_team, home_pitcher, away_pitcher, ganador, year, fecha)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['game_id'], row['home_team'], row['away_team'], 
                row['home_pitcher'], row['away_pitcher'], 
                int(row['ganador']), int(row['year']), row['fecha'].strftime('%Y-%m-%d')
            ))
            registros_nuevos += 1
        except Exception as e:
            errores += 1

        # Lógica de bloques y feedback visual
        if (i + 1) % 500 == 0:
            conn.commit()
            print(f"   > Procesados {i + 1} de {len(df)}...")

    conn.commit()

    # --- VERIFICACIÓN DE INSERCIÓN FINAL ---
    print("\n🔍 VERIFICACIÓN DE TABLA HISTORICO_REAL:")
    conteo_final = conn.execute("SELECT COUNT(*) FROM historico_real").fetchone()[0]
    por_año = pd.read_sql("SELECT year, COUNT(*) as total FROM historico_real GROUP BY year", conn)
    print(f"Total de filas reales en DB: {conteo_final}")
    print(por_año.to_string(index=False))
    # ---------------------------------------

    conn.close()

    print("\n" + "="*50)
    print(f"✅ PROCESO FINALIZADO")
    print(f"📊 Juegos procesados: {len(df)}")
    print(f"📦 Registros en DB: {registros_nuevos}")
    print(f"⚠️  Duplicados ignorados o errores: {len(df) - registros_nuevos}")
    print("="*50)
    print("💡 NOTA: Los juegos NO se marcaron en 'control_entrenamiento'.")
    print("Esto permitirá que tu script principal los procese y haga el scraping")
    print("necesario para obtener las 32 variables de la v3.5.")

if __name__ == "__main__":
    importar_y_transformar()

# import pandas as pd
# import sqlite3
# import os
# import time
# import re

# # CONFIGURACIÓN DE RUTAS
# CSV_PATH = './data/processed/datos_ml_ready.csv'
# DB_PATH = './data/mlb_reentrenamiento.db'

# def limpiar_nombre_equipo(nombre):
#     """Asegura que el código del equipo sea de 3 letras limpias."""
#     nombre = str(nombre).strip().upper()
#     # Mapeo rápido de excepciones comunes si las hay en tu CSV
#     mapeo_especial = {'SDG': 'SDP', 'SFO': 'SFG', 'KCA': 'KCR', 'ANA': 'LAA', 'LAN': 'LAD', 'NYA': 'NYY', 'NYN': 'NYM', 'TBA': 'TBR'}
#     nombre = mapeo_especial.get(nombre, nombre)
#     return re.sub(r'[^A-Z]', '', nombre)[:3]

# def inicializar_db():
#     """Crea la estructura de tablas si la DB es nueva."""
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     # Tabla principal de juegos
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS juegos (
#             game_id TEXT PRIMARY KEY,
#             home_team TEXT,
#             away_team TEXT,
#             home_pitcher TEXT,
#             away_pitcher TEXT,
#             ganador INTEGER,
#             year INTEGER,
#             fecha TEXT
#         )
#     ''')
#     # Tabla de control para evitar re-entrenar lo mismo
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS control_entrenamiento (
#             game_id TEXT PRIMARY KEY
#         )
#     ''')
#     conn.commit()
#     conn.close()

# def importar_y_transformar():
#     if not os.path.exists(CSV_PATH):
#         print(f"❌ Error: No se encontró el archivo en {CSV_PATH}")
#         return

#     print(f"📖 Cargando datos desde {CSV_PATH}...")
#     df = pd.read_csv(CSV_PATH)
    
#     # 1. Limpieza y Formateo
#     df['fecha'] = pd.to_datetime(df['fecha'])
#     df['year'] = df['fecha'].dt.year
    
#     print("🛠️ Generando Game IDs oficiales (Formato: TEAMYYYYMMDD0)...")
    
#     def generar_id(row):
#         team_code = limpiar_nombre_equipo(row['home_team'])
#         fecha_str = row['fecha'].strftime('%Y%m%d')
#         return f"{team_code}{fecha_str}0"

#     df['game_id'] = df.apply(generar_id, axis=1)

#     # 2. Conectar e Insertar
#     inicializar_db()
#     conn = sqlite3.connect(DB_PATH)
    
#     print(f"🚀 Intentando integrar {len(df)} juegos a la base de datos...")
    
#     registros_nuevos = 0
#     errores = 0

#     for i, row in df.iterrows():
#         try:
#             # Insertar en la tabla de juegos (Ignorar si ya existe el ID)
#             conn.execute('''
#                 INSERT OR IGNORE INTO juegos 
#                 (game_id, home_team, away_team, home_pitcher, away_pitcher, ganador, year, fecha)
#                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)
#             ''', (
#                 row['game_id'], row['home_team'], row['away_team'], 
#                 row['home_pitcher'], row['away_pitcher'], 
#                 int(row['ganador']), int(row['year']), row['fecha'].strftime('%Y-%m-%d')
#             ))
#             registros_nuevos += 1
#         except Exception as e:
#             errores += 1

#         # Lógica de bloques y feedback visual
#         if (i + 1) % 500 == 0:
#             conn.commit()
#             print(f"   > Procesados {i + 1} de {len(df)}...")

#     conn.commit()
#     conn.close()

#     print("\n" + "="*50)
#     print(f"✅ PROCESO FINALIZADO")
#     print(f"📊 Juegos procesados: {len(df)}")
#     print(f"📦 Registros en DB: {registros_nuevos}")
#     print(f"⚠️  Duplicados ignorados o errores: {len(df) - registros_nuevos}")
#     print("="*50)
#     print("💡 NOTA: Los juegos NO se marcaron en 'control_entrenamiento'.")
#     print("Esto permitirá que tu script principal los procese y haga el scraping")
#     print("necesario para obtener las 32 variables de la v3.5.")

# if __name__ == "__main__":
#     importar_y_transformar()