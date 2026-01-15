# 🚀 Guía de Despliegue - MLB Predictor V3.5

Esta guía explica cómo configurar y desplegar el sistema completo de MLB Predictor usando GitHub Actions.

## 📋 Índice

1. [Estructura del Proyecto](#estructura-del-proyecto)
2. [Configuración Inicial](#configuración-inicial)
3. [GitHub Actions Setup](#github-actions-setup)
4. [Despliegue de la API](#despliegue-de-la-api)
5. [Monitoreo y Mantenimiento](#monitoreo-y-mantenimiento)

---

## 📁 Estructura del Proyecto

```
mlb-game-predictor/
├── .github/
│   └── workflows/
│       └── mlb_predictor.yml        # Automatización GitHub Actions
├── src/
│   ├── mlb_config.py                # Configuración centralizada
│   ├── mlb_feature_engineering.py   # Ingeniería de features
│   ├── train_model_hybrid_actions.py # Entrenamiento
│   ├── mlb_predict_engine.py        # Motor de predicción
│   ├── mlb_manual_interface.py      # CLI manual
│   ├── mlb_daily_scraper.py         # Scraping diario
│   ├── mlb_update_real_results.py   # Actualización de resultados
│   ├── mlb_utils.py                 # Utilidades
│   └── api.py                       # API FastAPI
├── models/
│   └── modelo_mlb_v3.5.json         # Modelo entrenado 
├── data/
│   └── mlb_reentrenamiento.db       # Base de datos SQLite
├── cache/
│   └── features_hibridas_v3.5_cache.pkl  # Caché
├── requirements.txt                  # Dependencias Python
├── README.md                         # Documentación principal
└── DEPLOYMENT.md                     # Esta guía

```

---

## ⚙️ Configuración Inicial

### 1. Preparar el Repositorio

```bash
# Clonar el repositorio
git clone <tu-repo-url>
cd mlb-game-predictor

# Crear estructura de carpetas
mkdir -p models data cache

# Instalar dependencias localmente (para testing)
pip install -r requirements.txt
```

### 2. Inicializar la Base de Datos

```bash
cd src
python -c "
import sqlite3
from mlb_config import DB_PATH

with sqlite3.connect(DB_PATH) as conn:
    # Tabla de partidos del día
    conn.execute('''CREATE TABLE IF NOT EXISTS historico_partidos 
                   (game_id TEXT PRIMARY KEY, box_score_url TEXT, fecha TEXT, year INTEGER,
                    away_team TEXT, home_team TEXT, away_pitcher TEXT, home_pitcher TEXT,
                    away_starter_ERA REAL, away_starter_WHIP REAL, away_starter_H9 REAL,
                    away_starter_SO9 REAL, away_starter_W INTEGER, away_starter_L INTEGER,
                    home_starter_ERA REAL, home_starter_WHIP REAL, home_starter_H9 REAL,
                    home_starter_SO9 REAL, home_starter_W INTEGER, home_starter_L INTEGER)''')
    
    # Tabla de lineups
    conn.execute('''CREATE TABLE IF NOT EXISTS lineup_ini 
                   (fecha TEXT, game_id TEXT, team TEXT, [order] TEXT, player TEXT)''')
    
    # Tabla de resultados reales
    conn.execute('''CREATE TABLE IF NOT EXISTS historico_real 
                   (game_id TEXT PRIMARY KEY, home_team TEXT, away_team TEXT, 
                    home_pitcher TEXT, away_pitcher TEXT, ganador INTEGER, 
                    year INTEGER, fecha TEXT, score_home INTEGER, score_away INTEGER)''')
    
    # Tabla de predicciones
    conn.execute('''CREATE TABLE IF NOT EXISTS predicciones_historico 
                   (fecha TEXT, home_team TEXT, away_team TEXT, home_pitcher TEXT, 
                    away_pitcher TEXT, prob_home REAL, prob_away REAL, 
                    prediccion TEXT, confianza TEXT, tipo TEXT)''')
    
    # Tabla de control de entrenamiento
    conn.execute('''CREATE TABLE IF NOT EXISTS control_entrenamiento 
                   (game_id TEXT PRIMARY KEY)''')
    
    conn.commit()
    print('✅ Base de datos inicializada')
"
```

### 3. Entrenamiento Inicial del Modelo

**IMPORTANTE:** Antes de activar GitHub Actions, necesitas un modelo pre-entrenado.

```bash
# Opción A: Si tienes datos históricos
# Importa tus datos históricos a historico_real
python train_model_hybrid_actions.py

# Opción B: Entrenamiento mínimo con datos de prueba
# (Necesitarás al menos 500 partidos para un modelo básico)
```

### 4. Validar Configuración

```bash
# Verificar que todo está correcto
python mlb_config.py

# Deberías ver:
# ✅ Configuración validada correctamente
# 📁 Modelo: .../models/modelo_mlb_v3.5.json
#    Existe: ✅ SÍ
# 📁 DB: .../data/mlb_reentrenamiento.db
#    Existe: ✅ SÍ
```

---

## 🤖 GitHub Actions Setup

### 1. Preparar el Repositorio para Actions

```bash
# Asegurar que los archivos de datos están en .gitignore
echo "cache/*.pkl.bak" >> .gitignore
echo "*.pyc" >> .gitignore
echo "__pycache__/" >> .gitignore

# Pero INCLUIMOS los archivos principales
git add models/modelo_mlb_v3.5.json
git add data/mlb_reentrenamiento.db
git add .github/workflows/mlb_predictor.yml
git commit -m "🚀 Configuración inicial para GitHub Actions"
git push
```

### 2. Configurar Permisos en GitHub

1. Ve a tu repositorio en GitHub
2. Settings → Actions → General
3. Workflow permissions:
   - ✅ **Read and write permissions**
   - ✅ **Allow GitHub Actions to create and approve pull requests**

### 3. Secrets (Opcional)

Si necesitas configuraciones privadas:

```
Settings → Secrets and variables → Actions → New repository secret
```

---

## 🔄 Flujo de Trabajo Automático

### Horarios de Ejecución (EST)

| Hora | Job | Descripción |
|------|-----|-------------|
| 10:00 AM | Scraping Diario | Captura lineups del día |
| 1:00 PM | Scraping (Reintento) | Si 10 AM falló |
| 5:00 AM | Actualizar Resultados | Resultados del día anterior |
| 6:00 AM | Reentrenamiento | Si hay 150+ juegos nuevos |

### Flujo Completo

```
DÍA 1 - 10 AM
  ↓
[Scraping] → Captura partidos del día
  ↓
[Predicción] → Genera predicciones automáticas
  ↓
Commit y Push a GitHub

DÍA 2 - 5 AM
  ↓
[Actualizar Resultados] → Captura scores finales
  ↓
Commit y Push a GitHub

DÍA 2 - 6 AM
  ↓
[Verificar] → ¿Hay 150+ juegos nuevos?
  ↓
  SÍ → [Reentrenar Modelo] → ¿Mejora accuracy?
                                ↓
                               SÍ → Actualizar modelo
                                ↓
                               NO → Mantener modelo anterior
```

---

## 🌐 Despliegue de la API

### Opción 1: Render.com (Recomendado - Free Tier)

```bash
# 1. Crear render.yaml en la raíz del proyecto
cat > render.yaml << 'EOF'
services:
  - type: web
    name: mlb-predictor-api
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "cd src && uvicorn api:app --host 0.0.0.0 --port $PORT"
    envVars:
      - key: PYTHON_VERSION
        value: 3.10.0
EOF

# 2. Conectar GitHub a Render
# Ve a render.com → New Web Service → Connect Repository
```

### Opción 2: Railway.app

```bash
# 1. Crear Procfile
echo "web: cd src && uvicorn api:app --host 0.0.0.0 --port \$PORT" > Procfile

# 2. Conectar en railway.app
# New Project → Deploy from GitHub
```

### Opción 3: Local (Para desarrollo)

```bash
cd src
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# API disponible en: http://localhost:8000
# Documentación: http://localhost:8000/docs
```

---

## 📊 Monitoreo y Mantenimiento

### Verificar Jobs de GitHub Actions

```bash
# Ver logs en:
# GitHub → Actions → Seleccionar workflow run
```

### Monitorear Accuracy

```bash
# Endpoint de API
curl http://tu-api.com/stats/accuracy?dias=30

# O desde Python
python mlb_utils.py accuracy 30
```

### Limpiar Caché (Si hay problemas)

```bash
python mlb_utils.py limpiar_cache
```

### Compactar Base de Datos

```bash
# Cada 6 meses aproximadamente
python mlb_utils.py compactar
```

---

## 🐛 Solución de Problemas

### GitHub Actions no se ejecuta

**Problema:** El workflow no corre automáticamente

**Solución:**
1. Verifica permisos en Settings → Actions
2. Asegúrate de que el workflow está en `.github/workflows/`
3. Verifica sintaxis YAML en [yamllint.com](https://www.yamllint.com/)

### Scraping falla constantemente

**Problema:** Rate limiting de Baseball-Reference

**Solución:**
```python
# En mlb_config.py, aumentar delays
SCRAPING_CONFIG = {
    'min_delay': 4,  # Aumentar de 2 a 4
    'max_delay': 8,  # Aumentar de 4 a 8
}
```

### Modelo no mejora

**Problema:** Reentrenamiento no actualiza el modelo

**Solución:**
```bash
# Verificar juegos pendientes
python -c "
import sqlite3
from mlb_config import DB_PATH
with sqlite3.connect(DB_PATH) as conn:
    cursor = conn.execute('''
        SELECT COUNT(*) FROM historico_real 
        WHERE game_id NOT IN (SELECT game_id FROM control_entrenamiento)
    ''')
    print(f'Juegos pendientes: {cursor.fetchone()[0]}')
"
```

### API retorna errores 500

**Problema:** Rutas de archivos incorrectas

**Solución:**
```bash
# Verificar configuración
cd src
python mlb_config.py

# Asegúrate de que modelo y DB existen
```

---

## 📈 Mejoras Futuras

1. **Caché de Redis** para features scrapeadas
2. **Websockets** para predicciones en tiempo real
3. **Análisis avanzado** con Plotly/Dash
4. **Notificaciones** por email/Telegram
5. **A/B Testing** de diferentes modelos

---

## 🆘 Soporte

- **Issues**: Abre un issue en GitHub
- **Documentación**: Ver [README.md](README.md)
- **API Docs**: `http://tu-api.com/docs`

---

**Última actualización:** Enero 2026  
**Versión:** 3.5v