# 🚀 Guía Completa: Despliegue de API MLB Predictor

## 📋 Tabla de Contenidos

1. [Requisitos Previos](#requisitos-previos)
2. [Despliegue Local](#despliegue-local)
3. [Pruebas de la API](#pruebas-de-la-api)
4. [Despliegue en la Nube](#despliegue-en-la-nube)
5. [Troubleshooting](#troubleshooting)

---

## 1️⃣ REQUISITOS PREVIOS

### ✅ Verificar que tienes:

```bash
# 1. Modelo entrenado
ls models/
# Debe mostrar:
# mlb_model.pkl
# mlb_scaler.pkl
# mlb_feature_names.pkl
# mlb_model_info.pkl

# 2. Dependencias instaladas
pip list | grep fastapi
pip list | grep uvicorn

# Si no están instaladas:
pip install fastapi uvicorn pydantic
```

### 📁 Estructura necesaria:

```
tu-proyecto/
│
├── api.py                    ← Tu archivo API
├── models/
│   ├── mlb_model.pkl
│   ├── mlb_scaler.pkl
│   ├── mlb_feature_names.pkl
│   └── mlb_model_info.pkl
└── requirements.txt
```

---

## 2️⃣ DESPLIEGUE LOCAL

### Paso 1: Verificar el archivo API

Asegúrate que `api.py` tiene las rutas correctas:

```python
# En api.py, busca estas líneas:
with open('mlb_model.pkl', 'rb') as f:  # ❌ Sin ./models/

# Y cámbialas a:
with open('./models/mlb_model.pkl', 'rb') as f:  # ✅ Con ./models/
```

### Paso 2: Iniciar la API

**Opción A: Usando Python directamente**

```bash
cd /ruta/a/tu/proyecto
python api.py
```

**Opción B: Usando Uvicorn (Recomendado)**

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Parámetros:
- `--reload`: Recarga automática al cambiar código
- `--host 0.0.0.0`: Accesible desde cualquier IP
- `--port 8000`: Puerto 8000

### Paso 3: Verificar que funciona

Deberías ver:

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
✅ Modelo cargado exitosamente
   Modelo: Random Forest
   Accuracy: 65.43%
INFO:     Application startup complete.
```

### Paso 4: Acceder a la documentación

Abre tu navegador y ve a:

**Swagger UI (Interactiva):**
```
http://localhost:8000/docs
```

**ReDoc (Documentación):**
```
http://localhost:8000/redoc
```

---

## 3️⃣ PRUEBAS DE LA API

### 🌐 Método 1: Desde el navegador (Swagger UI)

1. Abre http://localhost:8000/docs
2. Verás todos los endpoints disponibles
3. Click en `POST /predict`
4. Click en "Try it out"
5. Edita el JSON:

```json
{
  "home_team": "BOS",
  "away_team": "NYY",
  "home_pitcher": "Bello",
  "away_pitcher": "Cole",
  "year": 2025
}
```

6. Click en "Execute"
7. Verás la respuesta:

```json
{
  "ganador": "BOS",
  "prob_home": 0.623,
  "prob_away": 0.377,
  "confianza": 0.623,
  "year_usado": 2025,
  "mensaje": null
}
```

### 💻 Método 2: Usando cURL (Terminal)

```bash
# Prueba básica - Raíz
curl http://localhost:8000/

# Health check
curl http://localhost:8000/health

# Info del modelo
curl http://localhost:8000/info

# Predicción
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "LAD",
    "away_team": "SFG",
    "home_pitcher": "Kershaw",
    "away_pitcher": "Webb",
    "year": 2025
  }'
```

### 🐍 Método 3: Usando Python

Crea un archivo `test_api.py`:

```python
import requests
import json

# URL de la API
API_URL = "http://localhost:8000"

# Test 1: Health check
print("🔍 Test 1: Health Check")
response = requests.get(f"{API_URL}/health")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}\n")

# Test 2: Info del modelo
print("📊 Test 2: Información del Modelo")
response = requests.get(f"{API_URL}/info")
print(json.dumps(response.json(), indent=2))
print()

# Test 3: Predicción
print("🎯 Test 3: Predicción de Partido")
prediccion_data = {
    "home_team": "BOS",
    "away_team": "NYY",
    "home_pitcher": "Bello",
    "away_pitcher": "Cole",
    "year": 2025
}

response = requests.post(
    f"{API_URL}/predict",
    json=prediccion_data
)

if response.status_code == 200:
    resultado = response.json()
    print(f"✅ Predicción exitosa!")
    print(f"Ganador: {resultado['ganador']}")
    print(f"Probabilidad Local: {resultado['prob_home']*100:.1f}%")
    print(f"Probabilidad Visitante: {resultado['prob_away']*100:.1f}%")
    print(f"Confianza: {resultado['confianza']*100:.1f}%")
else:
    print(f"❌ Error: {response.status_code}")
    print(response.json())
```

Ejecutar:

```bash
python test_api.py
```

### 📱 Método 4: Usando Postman

1. Descarga Postman: https://www.postman.com/downloads/
2. Crear nueva request
3. Método: `POST`
4. URL: `http://localhost:8000/predict`
5. Headers: 
   - Key: `Content-Type`
   - Value: `application/json`
6. Body → raw → JSON:

```json
{
  "home_team": "LAD",
  "away_team": "SEA",
  "home_pitcher": "Yamamoto",
  "away_pitcher": "Gilbert",
  "year": 2026
}
```

7. Click "Send"

---

## 4️⃣ DESPLIEGUE EN LA NUBE

### 🌩️ Opción A: Render (GRATIS y FÁCIL)

#### Paso 1: Preparar archivos

**1. Crear `requirements.txt`:**

```bash
# En la raíz del proyecto
pip freeze > requirements.txt
```

O crear manualmente:

```txt
fastapi==0.95.0
uvicorn==0.21.0
pydantic==1.10.0
pandas==1.5.0
numpy==1.23.0
scikit-learn==1.2.0
cloudscraper==1.2.71
beautifulsoup4==4.11.0
requests==2.28.0
```

**2. Crear archivo `render.yaml`:**

```yaml
services:
  - type: web
    name: mlb-predictor-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn api:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.10.0
```

#### Paso 2: Subir a GitHub

```bash
# Inicializar git (si no lo has hecho)
git init
git add .
git commit -m "Initial commit - MLB Predictor API"

# Crear repositorio en GitHub y conectar
git remote add origin https://github.com/tu-usuario/mlb-predictor.git
git push -u origin main
```

#### Paso 3: Desplegar en Render

1. Ve a https://render.com/
2. Sign up / Login
3. Click "New +" → "Web Service"
4. Conecta tu repositorio de GitHub
5. Configuración:
   - **Name**: mlb-predictor-api
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`
6. Click "Create Web Service"
7. Espera 5-10 minutos...
8. Tu API estará en: `https://mlb-predictor-api.onrender.com`

#### Paso 4: Probar API en la nube

```bash
# Health check
curl https://mlb-predictor-api.onrender.com/health

# Predicción
curl -X POST "https://mlb-predictor-api.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "BOS",
    "away_team": "NYY",
    "home_pitcher": "Bello",
    "away_pitcher": "Cole",
    "year": 2025
  }'
```

### 🚀 Opción B: Railway (ALTERNATIVA)

1. Ve a https://railway.app/
2. Sign up con GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Selecciona tu repo
5. Railway detecta automáticamente Python
6. Añade variable de entorno: `PORT=8000`
7. Deploy automático

### ☁️ Opción C: Heroku (CLÁSICO)

**1. Instalar Heroku CLI:**

```bash
# Mac
brew tap heroku/brew && brew install heroku

# Windows
# Descargar de https://devcenter.heroku.com/articles/heroku-cli
```

**2. Crear `Procfile`:**

```
web: uvicorn api:app --host 0.0.0.0 --port $PORT
```

**3. Crear `runtime.txt`:**

```
python-3.10.0
```

**4. Desplegar:**

```bash
# Login
heroku login

# Crear app
heroku create mlb-predictor-api

# Deploy
git push heroku main

# Abrir
heroku open
```

---

## 5️⃣ TROUBLESHOOTING

### ❌ Problema: "Address already in use"

**Solución:**

```bash
# Ver qué está usando el puerto 8000
lsof -i :8000

# Matar el proceso
kill -9 <PID>

# O usar otro puerto
uvicorn api:app --port 8001
```

### ❌ Problema: "Module not found"

**Solución:**

```bash
# Verificar que estás en el entorno virtual
which python

# Instalar dependencias
pip install -r requirements.txt

# Verificar
pip list
```

### ❌ Problema: "Modelo no encontrado"

**Solución:**

```python
# En api.py, cambiar rutas relativas:
with open('./models/mlb_model.pkl', 'rb') as f:
    model = pickle.load(f)
```

### ❌ Problema: API muy lenta

**Causa:** Scraping en tiempo real

**Solución:**
- Usar cache
- Pre-calcular features
- Implementar Redis para cache

### ❌ Problema: CORS errors en frontend

**Solución:**

Ya está configurado en `api.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🎓 RESUMEN DE COMANDOS

### Desarrollo Local:

```bash
# Iniciar API
uvicorn api:app --reload

# Probar
curl http://localhost:8000/health
```

### Producción:

```bash
# Sin auto-reload, más workers
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Monitoreo:

```bash
# Ver logs en Render
render logs --tail

# Ver logs en Heroku
heroku logs --tail
```

---

## 📚 PRÓXIMOS PASOS

1. ✅ **Añadir autenticación** (API Keys)
2. ✅ **Implementar rate limiting** (limitar requests)
3. ✅ **Agregar logging** (guardar requests)
4. ✅ **Crear dashboard** (Streamlit/React)
5. ✅ **Monitoreo** (Sentry para errores)

---

## 🔗 ENLACES ÚTILES

- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Render**: https://render.com/
- **Railway**: https://railway.app/
- **Heroku**: https://www.heroku.com/
- **Postman**: https://www.postman.com/

---

**¡Tu API está lista! 🎉⚾**