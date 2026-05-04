---

# Guía de Despliegue: API MLB Predictor Hybrid V3

## Tabla de Contenidos

1. Requisitos Previos
2. Estructura del Proyecto V3
3. Despliegue Local
4. Pruebas de la API Híbrida
5. Despliegue en la Nube (Render)
6. Troubleshooting

---

## 1. REQUISITOS PREVIOS

### Verificar archivos del modelo:

```bash
# El Modelo V3 utiliza formato JSON para XGBoost y un pickle para la caché
ls models/
# Debe mostrar:
# modelo_mlb_v3.json

```

### Verificar caché de features:

La V3 utiliza un sistema de caché para acelerar el scraping:

```bash
ls cache/
# Debe mostrar:
# features_hibridas_v3_cache.pkl

```

### Dependencias necesarias:

```bash
pip install fastapi uvicorn pandas numpy xgboost cloudscraper beautifulsoup4 scikit-learn

```

---

## 2. ESTRUCTURA DEL PROYECTO V3

Para que la API funcione correctamente, respeta la siguiente jerarquía de carpetas:

```
tu-proyecto/
│
├── src/
│   ├── api_hybrid_v3.py        <-- Archivo principal de la API
│   ├── predict_game_v3.py      <-- Lógica de predicción y scraping
│   └── web_app_v3.py           <-- Interfaz de usuario
├── models/
│   └── modelo_mlb_v3.json      <-- Modelo XGBoost
├── cache/
│   └── features_hibridas_v3_cache.pkl
└── requirements.txt

```

---

## 3. DESPLIEGUE LOCAL

### Paso 1: Configurar rutas en api_hybrid_v3.py

Asegúrate de que la carga del modelo apunte a la carpeta superior si ejecutas desde `src`:

```python
# Corrección de ruta para el modelo
MODEL_PATH = "../models/modelo_mlb_v3.json"

```

### Paso 2: Iniciar el servidor

Desde la carpeta `src`, ejecuta:

```bash
uvicorn api_hybrid_v3:app --reload --host 0.0.0.0 --port 8000

```

### Paso 3: Verificación de arranque

Deberías ver en la consola:

```
INFO: Loading Hybrid Model V3 (XGBoost)...
INFO: Model loaded successfully.
INFO: Uvicorn running on http://localhost:8000

```

---

## 4. PRUEBAS DE LA API HÍBRIDA

### Endpoint de Predicción (POST /predict)

La V3 requiere los nombres de los equipos y los lanzadores para calcular las Super Features en tiempo real.

**Ejemplo de JSON para enviar:**

```json
{
  "home_team": "ARI",
  "away_team": "NYY",
  "home_pitcher": "Zac Gallen",
  "away_pitcher": "Carlos Rodón",
  "year": 2026
}

```

**Respuesta esperada (Esquema V3):**

```json
{
  "ganador": "ARI",
  "probabilidad": 0.584,
  "confianza": "Media",
  "stats_detalladas": { ... },
  "features_usadas": {
    "super_neutralizacion_whip_ops": -0.16,
    "super_resistencia_era_ops": 5.55,
    "super_muro_bullpen": 0.21
  }
}

```

---

## 5. DESPLIEGUE EN LA NUBE (RENDER)

### Paso 1: requirements.txt actualizado

Crea el archivo con estas versiones específicas para evitar conflictos con XGBoost:

```txt
fastapi
uvicorn
pandas
numpy
xgboost
scikit-learn
cloudscraper
beautifulsoup4
requests
python-multipart

```

### Paso 2: Configuración en Render

1. Crea un "Web Service" conectado a tu repositorio.
2. **Build Command**: `pip install -r requirements.txt`
3. **Start Command**: `uvicorn src.api_hybrid_v3:app --host 0.0.0.0 --port $PORT`
4. **Environment Variables**:
* `PYTHON_VERSION`: 3.10.0 (mínimo)



---

## 6. TROUBLESHOOTING

### Error: "No module named src"

Si ejecutas uvicorn desde fuera de la carpeta `src`, usa el punto como separador: `uvicorn src.api_hybrid_v3:app`.

### Valor de Resistencia (555.5%)

Este es un comportamiento esperado en la V3 cuando hay una diferencia amplia de ERA/OPS. Para corregir la visualización, aplica un factor de escala (abs(val) * 10) en el código del frontend, no en la API.

### Error de Scraping (403 Forbidden)

Baseball-Reference puede bloquear peticiones frecuentes. La V3 incluye `cloudscraper` para mitigar esto, pero si persiste, asegúrate de que la caché en la carpeta `cache/` tenga permisos de escritura.

---

**Resumen de Endpoints V3:**

* `GET /`: Estado del sistema.
* `GET /health`: Verificación de carga de modelo y archivos.
* `POST /predict`: Predicción híbrida con análisis de Matchups.
