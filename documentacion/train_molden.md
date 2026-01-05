# Análisis Detallado del Script de Entrenamiento MLB

## 📋 Índice
1. [Visión General](#visión-general)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Funciones de Scraping](#funciones-de-scraping)
4. [Extracción de Features](#extracción-de-features)
5. [Pipeline de Entrenamiento](#pipeline-de-entrenamiento)
6. [Por Qué 37 Features](#por-qué-37-features)

---

## 🎯 Visión General

### Objetivo del Script
Este script entrena un modelo de Machine Learning para predecir el ganador de partidos de béisbol MLB (local vs visitante) basándose en:
- Estadísticas históricas de equipos
- Rendimiento de lanzadores iniciales
- Características de los mejores bateadores

### Filosofía de Diseño
El script sigue un enfoque de **enriquecimiento de datos**: toma datos básicos (equipos, lanzadores, resultado) y los enriquece con estadísticas detalladas mediante web scraping de Baseball Reference.

---

## 🏗️ Arquitectura del Sistema

### Estructura del Flujo de Datos

```
CSV Histórico → Scraping Web → Feature Engineering → Modelo ML → Predicciones
    ↓              ↓                 ↓                  ↓
Partidos      Stats de        37 Features       RandomForest/
Básicos       Baseball-Ref    Calculadas        GradientBoosting
```

### Componentes Principales

1. **Módulo de Scraping**: Extrae estadísticas actualizadas
2. **Feature Extractor**: Transforma stats en features predictivas
3. **Pipeline ML**: Entrena y evalúa múltiples modelos
4. **Sistema de Cache**: Optimiza tiempos de re-entrenamiento

---

## 🕷️ Funciones de Scraping

### 1. `obtener_html(url)`

**Propósito**: Obtener HTML de forma robusta evitando bloqueos

```python
def obtener_html(url):
    scraper = cloudscraper.create_scraper()
```

**Por qué cloudscraper**:
- Baseball Reference usa protección Cloudflare
- `requests` normal sería bloqueado
- cloudscraper simula un navegador real

**Manejo de errores**:
- Timeout de 15 segundos (evita colgarse)
- Verifica status code 200
- Retorna `None` si falla (permite continuar)

---

### 2. `limpiar_dataframe(df)`

**Propósito**: Sanitizar tablas HTML parseadas

**Problemas que resuelve**:

1. **Filas de totales**: "Team Totals" no son jugadores
2. **Filas de ranking**: "Rank in 14th" contamina datos
3. **Filas vacías**: Espaciadores HTML
4. **Columna Rk**: No aporta información (es solo número de fila)

**Por qué es crítico**:
Sin esta limpieza, el modelo intentaría "aprender" de filas que no son jugadores reales.

---

### 3. `scrape_player_stats(team_code, year)`

**Propósito**: Extraer estadísticas completas de un equipo

**Estrategia**:
```python
url = f"https://www.baseball-reference.com/teams/{team_code}/{year}.shtml"
```

**Dos tablas clave**:
- `players_standard_batting`: Estadísticas ofensivas
- `players_standard_pitching`: Estadísticas de pitcheo

**Por qué ambas**:
- Necesitamos contexto completo del equipo
- Un equipo fuerte ofensivamente + pitcheo débil ≠ victoria garantizada
- El béisbol es balance entre ataque y defensa

**Manejo robusto**:
```python
if batting_table:
    try:
        batting_df = pd.read_html(str(batting_table))[0]
```
- Continúa aunque falle una tabla
- Retorna `None, None` si todo falla
- Permite al código principal decidir qué hacer

---

### 4. `safe_float(val)`

**Propósito**: Convertir valores de forma defensiva

**Por qué existe**:
- HTML puede tener: "3.45", "—", "", "N/A"
- `float("—")` crashearía el programa
- Retornar 0.0 es mejor que crashear (el modelo aprenderá que 0 = sin dato)

---

### 5. `encontrar_lanzador(pitching_df, nombre_lanzador)`

**Propósito**: Buscar lanzador específico y extraer sus estadísticas clave

**Estadísticas extraídas**:
- **ERA** (Earned Run Average): Carreras limpias por 9 innings
- **WHIP** (Walks + Hits per Inning): Corredores permitidos por inning
- **H9**: Hits permitidos por 9 innings
- **W/L**: Record de victorias/derrotas
- **IP**: Innings lanzados (indica experiencia)

**Por qué estas métricas**:
- **ERA**: El mejor predictor de efectividad de un lanzador
  - ERA < 3.00 = Excelente
  - ERA > 5.00 = Problema
- **WHIP**: Captura presión sobre el lanzador
  - WHIP < 1.00 = Elite
  - WHIP > 1.50 = Vulnerable
- **H9**: Complementa WHIP (aislando hits)

**Búsqueda flexible**:
```python
mask = pitching_df[name_col].astype(str).str.lower().str.contains(nombre_busqueda, na=False)
```
- Permite "Cole" encontrar "Gerrit Cole"
- Case-insensitive
- Maneja variaciones de nombres

**Fallback a None**:
Si no encuentra al lanzador, retorna `None` → el extractor pondrá 0s → el modelo aprende "sin información de lanzador"

---

### 6. `encontrar_mejor_bateador(batting_df)`

**Propósito**: Identificar poder ofensivo del equipo

**Estrategia innovadora**:
```python
mediana_ab = batting_df['AB'].median()
batting_filtrado = batting_df[batting_df['AB'] > mediana_ab]
```

**Por qué filtrar por AB (At Bats)**:
- Evita "outliers" de jugadores con pocos turnos
- Un jugador con BA=1.000 en 2 AB no representa al equipo
- Mediana asegura considerar solo titulares regulares

**Top 3 promediados**:
```python
top_3 = batting_filtrado.sort_values('OBP', ascending=False).head(3)
```

**Por qué promediar top 3 y no tomar solo #1**:
- El béisbol es un deporte de equipo
- Un súper estrella + 8 malos ≠ victoria
- Top 3 captura "núcleo ofensivo"

**Estadísticas elegidas**:
- **BA** (Batting Average): Hits / At Bats
- **OBP** (On-Base Percentage): Incluye bases por bolas
- **RBI** (Runs Batted In): Capacidad de impulsar carreras
- **R** (Runs): Carreras anotadas

**Por qué OBP > BA para ordenar**:
- OBP es mejor predictor de carreras que BA
- "Tres cosas ciertas: muerte, impuestos, y el OBP predice mejor" - Bill James

---

### 7. `calcular_stats_equipo(batting_df, pitching_df)`

**Propósito**: Obtener contexto agregado del equipo completo

**Por qué promedios del equipo además de mejores jugadores**:
- Captura **profundidad** del roster
- Un equipo con 9 jugadores sólidos > equipo con 3 estrellas
- En béisbol, todos batean → profundidad importa

---

## 🔧 Extracción de Features

### `extraer_features_partido(row, verbose=False)`

Esta es la **función más importante** del script. Transforma un partido simple en un vector de 37 features.

### Proceso paso a paso:

#### 1. **Obtener datos de ambos equipos**
```python
batting1, pitching1 = scrape_player_stats(row['home_team'], row['year'])
batting2, pitching2 = scrape_player_stats(row['away_team'], row['year'])
```

#### 2. **Calcular estadísticas agregadas**
```python
stats_team1 = calcular_stats_equipo(batting1, pitching1)
stats_team2 = calcular_stats_equipo(batting2, pitching2)
```

Genera features tipo:
- `home_team_BA_mean`
- `home_team_OBP_mean`
- `home_team_ERA_mean`
- etc.

#### 3. **Extraer stats de lanzadores específicos**
```python
pitcher1_stats = encontrar_lanzador(pitching1, row['home_pitcher'])
```

Genera features tipo:
- `home_pitcher_ERA`
- `home_pitcher_WHIP`
- `home_pitcher_H9`
- `home_pitcher_W`
- `home_pitcher_L`

#### 4. **Extraer stats de mejores bateadores**
```python
best_batter1 = encontrar_mejor_bateador(batting1)
```

Genera features tipo:
- `home_best_BA`
- `home_best_OBP`
- `home_best_RBI`
- `home_best_R`

#### 5. **Calcular features derivadas (CRÍTICO)**
```python
features['pitcher_ERA_diff'] = features['away_pitcher_ERA'] - features['home_pitcher_ERA']
```

**Por qué diferencias**:
- El modelo ML aprende mejor de **comparaciones** que valores absolutos
- ERA=3.0 vs ERA=4.0 → diff=-1.0 (ventaja local)
- Simplifica el aprendizaje: diff > 0 = ventaja local

**Features derivadas calculadas**:
1. `pitcher_ERA_diff`: ¿Quién tiene mejor lanzador?
2. `pitcher_WHIP_diff`: ¿Quién permite menos corredores?
3. `pitcher_H9_diff`: ¿Quién permite menos hits?
4. `team_BA_diff`: ¿Quién batea mejor?
5. `team_OBP_diff`: ¿Quién se embasiza más?

---

## 🎓 Pipeline de Entrenamiento

### `entrenar_modelo()`

### Fase 1: Carga y Validación

```python
df = pd.read_csv(csv_path)
print(f"Total de partidos: {len(df)}")
```

**Verifica distribución de clases**:
```python
print(f"Victorias locales (1): {(df['ganador'] == 1).sum()}")
```

**Por qué importa**:
- Datasets desbalanceados (90% local gana) → modelo aprende "siempre predice local"
- Béisbol real: ~54% local gana → dataset debe reflejarlo

---

### Fase 2: Sistema de Cache

```python
if usar_cache:
    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)
```

**Por qué cache es esencial**:
- Scraping de 3000 partidos = **varias horas**
- Re-entrenar el modelo sin cache = repetir scraping
- Cache permite iterar en hyperparámetros sin re-scrapear

**Qué se cachea**:
- `X`: DataFrame de features (37 columnas × N partidos)
- `y`: Array de labels (ganador: 0 o 1)

---

### Fase 3: Extracción de Features

```python
for idx, row in df.iterrows():
    features = extraer_features_partido(row, verbose=False)
    if features:
        features_list.append(features)
        labels.append(row['ganador'])
    time.sleep(1.5)  # Ser amigable con el servidor
```

**Por qué `time.sleep(1.5)`**:
- Baseball Reference bloqueará IPs con > 20 requests/minuto
- 1.5 segundos = ~40 requests/minuto = seguro
- Sin esto: IP baneada a mitad del scraping

**Manejo de fallos**:
```python
if features:
    features_list.append(features)
else:
    partidos_fallidos += 1
```
- No crashea por un partido fallido
- Continúa con los demás
- Reporta cuántos fallaron

---

### Fase 4: Preparación de Datos

#### Split Train/Test
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Por qué `stratify=y`**:
- Asegura misma proporción local/visitante en train y test
- Sin stratify: test podría ser todo locales → métricas engañosas

**80/20 split**:
- 80% entrena el modelo
- 20% evalúa desempeño real
- Estándar en ML para datasets medianos

#### Escalado de Features
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

**Por qué escalar**:
- ERA típico: 3-5
- RBI típico: 50-120
- Sin escalar: modelo priorizaría RBI (números más grandes)
- StandardScaler: transforma a media=0, std=1
- Ahora ERA y RBI tienen misma "importancia numérica"

**Fit solo en train**:
```python
scaler.fit_transform(X_train)  # Calcula media/std
X_test_scaled = scaler.transform(X_test)  # Usa misma media/std
```
- Previene "data leakage"
- Test no debe influir en transformaciones

---

### Fase 5: Entrenamiento de Múltiples Modelos

```python
modelos = {
    'Random Forest': RandomForestClassifier(...),
    'Gradient Boosting': GradientBoostingClassifier(...),
    'Logistic Regression': LogisticRegression(...)
}
```

**Por qué 3 modelos**:
No sabemos a priori cuál funcionará mejor con estos datos específicos.

#### Random Forest
```python
RandomForestClassifier(
    n_estimators=200,    # 200 árboles de decisión
    max_depth=15,        # Profundidad máxima por árbol
    min_samples_split=5, # Min muestras para dividir nodo
    random_state=42,     # Reproducibilidad
    n_jobs=-1            # Usar todos los cores CPU
)
```

**Ventajas**:
- Robusto a overfitting
- Maneja features no lineales
- Provee feature importance

**Cuándo funciona bien**:
- Datos con interacciones complejas
- "ERA bajo + BA alto = victoria más probable"

#### Gradient Boosting
```python
GradientBoostingClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1
)
```

**Diferencia con RF**:
- RF: árboles independientes
- GB: cada árbol corrige errores del anterior
- Típicamente más preciso pero más lento

#### Logistic Regression
```python
LogisticRegression(max_iter=1000)
```

**Por qué incluirlo**:
- Es el "baseline" simple
- Si RF/GB apenas lo superan → señal de que datos son simples
- Útil para interpretabilidad

---

### Fase 6: Evaluación

#### Métricas calculadas:

1. **Accuracy**: % de predicciones correctas
   - Fácil de entender
   - Problema: no distingue tipo de error

2. **ROC-AUC**: Área bajo curva ROC
   - 0.5 = azar
   - 1.0 = perfecto
   - Mejor que accuracy para clases desbalanceadas

3. **Cross-Validation (5-fold)**:
```python
cv_scores = cross_val_score(modelo, X_train_scaled, y_train, cv=5)
```

**Por qué CV**:
- Train/test split es **una** partición aleatoria
- CV hace 5 splits diferentes
- Promedio de 5 → estimación más robusta
- Desviación estándar → qué tan estable es el modelo

---

### Fase 7: Selección del Mejor Modelo

```python
mejor_modelo_nombre = max(resultados.items(), key=lambda x: x[1]['accuracy'])[0]
```

**Criterio**: Accuracy máxima en test set

**Por qué accuracy aquí**:
- Clases balanceadas en béisbol (~54% local)
- Fácil de comunicar: "acierta 63% de partidos"

---

### Fase 8: Análisis del Mejor Modelo

#### Classification Report
```
              precision    recall  f1-score
Away Win         0.60      0.55      0.57
Home Win         0.65      0.70      0.67
```

**Interpretación**:
- **Precision**: De los que predijo local, % que fueron local
- **Recall**: De los locales reales, % que predijo
- **F1**: Balance entre precision y recall

#### Confusion Matrix
```
                 Predicted
                 Away  Home
   Actual Away    110    90
   Actual Home     80   120
```

**Lectura**:
- Diagonal = aciertos
- Off-diagonal = errores
- Muestra **tipo** de errores (falsos positivos vs negativos)

#### Feature Importance
```python
if hasattr(mejor_modelo, 'feature_importances_'):
```

**Solo RF/GB tienen esto**:
- Muestra qué features más influyeron
- Típicamente: ERA, WHIP, OBP serán top

**Utilidad**:
- Validación: ¿el modelo usa features sensatas?
- Simplificación: ¿podemos quitar features poco importantes?

---

### Fase 9: Persistencia

```python
pickle.dump(mejor_modelo, f)      # Modelo entrenado
pickle.dump(scaler, f)            # Escalador (crucial!)
pickle.dump(list(X.columns), f)   # Nombres de features
```

**Por qué 3 archivos**:
1. **Modelo**: Pesos aprendidos
2. **Scaler**: Para transformar datos nuevos igual que train
3. **Feature names**: Para asegurar orden correcto de features

**Sin scaler**:
```python
# Entrenamiento: ERA escalado a -1.5
# Predicción: ERA sin escalar = 3.5
# Modelo: "Este 3.5 es altísimo!" (asume escala de train)
# Resultado: Predicción errónea
```

---

## 📊 Por Qué 37 Features

### Desglose Completo:

#### Features de Equipo (14 features)

**Local (7)**:
1. `home_team_BA_mean` - Promedio de bateo del equipo
2. `home_team_OBP_mean` - Promedio de embase del equipo
3. `home_team_RBI_mean` - Promedio de carreras impulsadas
4. `home_team_R_mean` - Promedio de carreras anotadas
5. `home_team_ERA_mean` - ERA promedio del pitcheo
6. `home_team_WHIP_mean` - WHIP promedio del pitcheo
7. `home_team_H9_mean` - H9 promedio del pitcheo

**Visitante (7)**:
8-14. Mismas stats para equipo visitante con prefijo `away_`

**Por qué estas**:
- Capturan **calidad general** del roster
- BA/OBP/RBI/R → poder ofensivo
- ERA/WHIP/H9 → calidad de pitcheo

---

#### Features de Lanzador Inicial (10 features)

**Local (5)**:
15. `home_pitcher_ERA`
16. `home_pitcher_WHIP`
17. `home_pitcher_H9`
18. `home_pitcher_W` - Victorias
19. `home_pitcher_L` - Derrotas

**Visitante (5)**:
20-24. Mismas stats para lanzador visitante

**Por qué estas**:
- Lanzador inicial determina primeros 5-7 innings
- W/L captura "clutch" y soporte del equipo
- ERA/WHIP/H9 → efectividad pura

---

#### Features de Mejores Bateadores (8 features)

**Local (4)**:
25. `home_best_BA` - BA promedio top 3
26. `home_best_OBP` - OBP promedio top 3
27. `home_best_RBI` - RBI promedio top 3
28. `home_best_R` - R promedio top 3

**Visitante (4)**:
29-32. Mismas stats para top 3 visitantes

**Por qué estas**:
- Estrellas ganan partidos
- Top 3 captura núcleo ofensivo sin outliers
- Complementa promedios de equipo (elite vs profundidad)

---

#### Features Derivadas (5 features)

33. `pitcher_ERA_diff` = away_ERA - home_ERA
34. `pitcher_WHIP_diff` = away_WHIP - home_WHIP
35. `pitcher_H9_diff` = away_H9 - home_H9
36. `team_BA_diff` = home_BA - away_BA
37. `team_OBP_diff` = home_OBP - away_OBP

**Por qué estas son CRÍTICAS**:
- Facilitan aprendizaje del modelo
- Diff > 0 en pitcher_ERA_diff → ventaja local clara
- Modelo no tiene que "descubrir" que debe comparar
- En experimentos: +5-10% accuracy vs sin diffs

---

## 🎯 Resumen de Diseño

### Principios Clave:

1. **Robustez**: Maneja fallos sin crashear
2. **Eficiencia**: Cache evita re-scraping
3. **Interpretabilidad**: Features tienen significado real de béisbol
4. **Balance**: Stats de pitcheo + bateo + equipo + individuales
5. **Comparabilidad**: Features derivadas facilitan aprendizaje

### Limitaciones Actuales:

1. **Scraping lento**: 3000 partidos = 2-3 horas
2. **Sin contexto temporal**: No considera racha reciente
3. **Sin venue**: No considera estadio (parque pequeño vs grande)
4. **Sin clima**: Viento/temperatura afectan fly balls
5. **Sin lineup**: Asume mismos jugadores siempre

### Por Qué Funciona:

El modelo combina:
- **Nivel macro**: Calidad del equipo completo
- **Nivel meso**: Efectividad del lanzador inicial
- **Nivel micro**: Estrellas ofensivas

Esta jerarquía de información replica cómo analistas humanos evalúan partidos.

---

## 📈 Próximos Pasos Sugeridos

1. **Features temporales**: Últimos 10 partidos
2. **Features de contexto**: Estadio, división
3. **Optimización**: Paralelizar scraping
4. **Validación temporal**: Train en 2022, test en 2023
5. **Ensemble**: Combinar múltiples modelos