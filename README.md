---

# MLB Game Predictor V3 - Hybrid Intelligence

Sistema avanzado de predicción para partidos de la Major League Baseball (MLB) basado en Machine Learning. Esta versión utiliza un modelo híbrido que integra estadísticas históricas de 20 años con análisis de matchups en tiempo real mediante el cálculo de Super Features.

---

## Tabla de Contenidos

* Características V3
* Arquitectura del Sistema
* Instalación y Configuración
* Flujo de Uso
* Especificaciones del Modelo
* Documentación de la API
* Estructura del Proyecto
* Tecnologías Utilizadas
* Contacto y Licencia

---

## Características V3

### Análisis de Matchups (Super Features)

La V3 introduce tres métricas propietarias que analizan el enfrentamiento directo entre jugadores:

* **Neutralización**: Evalúa la capacidad del lanzador para dominar el contacto basándose en su WHIP frente al OPS del lineup rival.
* **Resistencia**: Mide la durabilidad del abridor (Calidad de entradas) frente al poder ofensivo acumulado del equipo contrario.
* **Muro Bullpen**: Analiza la solvencia del relevo final frente al rendimiento de los bateadores de cierre del oponente.

### Capacidades Técnicas

* **Scraping Dinámico**: Extracción optimizada de Baseball-Reference con gestión de headers para evitar bloqueos.
* **XGBoost Integration**: Motor de predicción basado en Extreme Gradient Boosting para capturar patrones no lineales.
* **Caché Inteligente**: Sistema de persistencia de datos para lanzadores y equipos que reduce el tiempo de respuesta en un 70%.
* **Interfaz Profesional**: Dashboard interactivo con visualización de métricas avanzadas y niveles de confianza estadística.

---

## Arquitectura del Sistema

El sistema opera en cuatro capas:

1. **Capa de Datos**: Scraping de Baseball-Reference y procesamiento de CSVs históricos.
2. **Capa de Ingeniería**: Cálculo de 26 variables clave (SelectKBest) y 3 Super Features dinámicas.
3. **Capa de Inferencia**: Modelo XGBoost optimizado con validación TimeSeriesSplit.
4. **Capa de Usuario**: API REST (FastAPI) y Front-end (Streamlit).

---

## Instalación y Configuración

### Requisitos

* Python 3.10 o superior
* pip

### Instalación de dependencias

```bash
pip install -r requirements.txt

```

**Archivo requirements.txt sugerido:**

```txt
pandas
numpy
scikit-learn
xgboost
cloudscraper
beautifulsoup4
requests
fastapi
uvicorn
streamlit
plotly
python-multipart

```

---

## Flujo de Uso

### 1. Preparación de Datos

La V3 requiere un entrenamiento con datos de largo plazo para estabilizar las Super Features:

```bash
python multi_season_scraper.py
python csv_transformer.py

```

### 2. Entrenamiento Híbrido V3

El script de entrenamiento utiliza optimización de hiperparámetros específica para XGBoost:

```bash
python train_model_hybrid_v3.py

```

### 3. Ejecución de la API

Inicia el backend que procesará las solicitudes de predicción:

```bash
uvicorn src.api_hybrid_v3:app --host 0.0.0.0 --port 8002

```

### 4. Lanzamiento de la Web App

Interfaz de usuario para realizar predicciones visuales:

```bash
streamlit run src/web_app_v3.py

```

---

## Especificaciones del Modelo

### Modelo Híbrido V3 (XGBoost)

* **Algoritmo**: XGBoost Classifier.
* **Features**: 26 variables seleccionadas + 3 Super Features dinámicas.
* **Validación**: TimeSeriesSplit (5 folds) para evitar fugas de datos temporales.
* **Métricas**:
* Accuracy: ~65.8%
* ROC-AUC: 0.71
* Tasa de acierto en Alta Confianza: >78%



### Variables de Mayor Impacto

1. Diferencial de ERA entre lanzadores.
2. Índice de Neutralización (Matchup-based).
3. Rendimiento en los últimos 10 juegos (L10).
4. Historial Head-to-Head (H2H).

---

## Documentación de la API

### Endpoint: POST /predict

Realiza una predicción completa analizando el lineup y el abridor.

**Request Body:**

```json
{
  "home_team": "ARI",
  "away_team": "NYY",
  "home_pitcher": "Zac Gallen",
  "away_pitcher": "Carlos Rodón",
  "year": 2026
}

```

**Respuesta Exitosa:**

```json
{
  "ganador": "ARI",
  "probabilidad": 0.58,
  "confianza": "Media",
  "super_features": {
    "neutralizacion": -0.16,
    "resistencia": 5.55,
    "muro_bullpen": 0.21
  }
}

```

---

## Estructura del Proyecto

```
mlb-game-predictor-v3/
│
├── src/
│   ├── api_hybrid_v3.py        # API FastAPI
│   ├── web_app_v3.py           # Dashboard Streamlit
│   └── predict_game_v3.py      # Lógica de predicción
│
├── models/
│   ├── modelo_mlb_v3.json      # Modelo XGBoost persistido
│   └── mlb_scaler_v3.pkl       # Escalador de datos
│
├── cache/
│   ├── pitcher_cache_v3.pkl    # Datos históricos de lanzadores
│   └── team_stats_cache.pkl    # Estadísticas de equipos
│
├── data/
│   ├── raw/                    # CSVs de temporadas crudas
│   └── processed/              # Dataset final para entrenamiento
│
└── requirements.txt

```

---

## Tecnologías Utilizadas

* **Core**: Python 3.10+
* **Machine Learning**: XGBoost, Scikit-learn, Pandas, NumPy.
* **Web Scraping**: Cloudscraper, BeautifulSoup4.
* **API & Web**: FastAPI, Uvicorn, Streamlit.
* **Visualización**: Plotly Express.

---

## Notas de Implementación

* **Escalado de Resistencia**: El valor de la Super Feature "Resistencia" se entrega de forma cruda por la API (ej. 5.55). Para una visualización estética en el frontend, se recomienda multiplicar por 10 para mostrarlo como porcentaje relativo.
* **CORS**: La API viene configurada para permitir peticiones desde cualquier origen, facilitando la conexión con despliegues en la nube como Render o Vercel.

---

## Contacto y Licencia

**Autor**: Gabriel Larrazabal
**Email**: gabolarrazabal13@gmail.com
**GitHub**: [@GaboLarrazabal13](https://github.com/GaboLarrazabal13)

Este proyecto se distribuye bajo la Licencia MIT.

---

**MLB Game Predictor V3** - Inteligencia estadística aplicada al diamante.