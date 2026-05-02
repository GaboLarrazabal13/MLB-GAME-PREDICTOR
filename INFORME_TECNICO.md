# MLB Game Predictor — Informe Técnico de Proyecto
**Versión:** V3.5.2 | **Autor:** Gabriel Larrazabal | **Repositorio:** [GaboLarrazabal13/MLB-GAME-PREDICTOR](https://github.com/GaboLarrazabal13/MLB-GAME-PREDICTOR)

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Arquitectura del Sistema](#2-arquitectura-del-sistema)
3. [Pipeline de Datos](#3-pipeline-de-datos)
4. [Motor de Machine Learning](#4-motor-de-machine-learning)
5. [Ingeniería de Features](#5-ingeniería-de-features)
6. [API REST y Frontend](#6-api-rest-y-frontend)
7. [Infraestructura y DevOps](#7-infraestructura-y-devops)
8. [Calidad y Testing](#8-calidad-y-testing)
9. [Evolución y Cambios Clave Recientes](#9-evolución-y-cambios-clave-recientes)
10. [Métricas de Rendimiento](#10-métricas-de-rendimiento)
11. [Stack Tecnológico](#11-stack-tecnológico)

---

## 1. Resumen Ejecutivo

MLB Game Predictor es un **sistema de predicción de partidos de béisbol de la Major League Baseball (MLB)** basado en Machine Learning. El sistema integra un pipeline de datos completamente automatizado, un motor de predicción con features avanzadas y una interfaz web interactiva, todo orquestado mediante GitHub Actions para operar de forma autónoma cada día durante la temporada.

### Logros principales

| Métrica | Valor |
|---|---|
| Precisión del modelo (con scraping completo) | **~74–80%** |
| Juegos predichos en base de datos | **363+** |
| Predicciones de baja calidad (sin scraping) | **0** |
| Features totales por partido | **38** (temporales + scraping + super features) |
| Cobertura de equipos MLB | **30 / 30** |
| Versión del modelo | XGBoost V3.5 |
| Ejecución automatizada diaria | GitHub Actions (3 triggers UTC) |

> El sistema está diseñado con un principio fundamental: **"Scrape-or-Nothing"** — las predicciones solo se persisten en la base de datos histórica cuando cuentan con el set completo de features obtenidas mediante live scraping, garantizando así la integridad de las métricas de rendimiento.

---

## 2. Arquitectura del Sistema

El proyecto sigue una arquitectura en capas, completamente desacoplada:

```
┌─────────────────────────────────────────────────────────────┐
│                    CAPA DE PRESENTACIÓN                     │
│          Streamlit App (app.py) · Puerto 8501               │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP REST
┌────────────────────────▼────────────────────────────────────┐
│                     CAPA DE API                             │
│        FastAPI (api.py) V3.5.2 · Puerto 8000                │
│  Rate Limiting · CORS · Cache TTL · Validación Pydantic     │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  CAPA DE LÓGICA DE NEGOCIO                  │
│  mlb_predict_engine.py · train_model_hybrid_actions.py      │
│  mlb_feature_engineering.py · mlb_daily_scraper.py          │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    CAPA DE DATOS                            │
│     SQLite (mlb_reentrenamiento.db) · Pickle Cache          │
│     Tablas: historico_partidos · predicciones_historico      │
│             historico_real · sync_control                   │
└─────────────────────────────────────────────────────────────┘
```

### Componentes principales

| Módulo | Responsabilidad |
|---|---|
| `mlb_config.py` | Configuración centralizada: rutas, mapeos de equipos, parámetros del modelo |
| `mlb_daily_scraper.py` | Scraping diario de Baseball-Reference: lineups, pitcheros, estadísticas |
| `mlb_schedule_utils.py` | Parsing del calendario de MLB, detección robusta de secciones de fecha |
| `mlb_feature_engineering.py` | Cálculo de Super Features y validación de datos |
| `train_model_hybrid_actions.py` | Entrenamiento XGBoost con datos híbridos (histórico + scraping en tiempo real) |
| `mlb_predict_engine.py` | Motor de predicción: genera y persiste predicciones con el conjunto completo de features |
| `mlb_update_real_results.py` | Actualización de resultados reales para el cálculo de accuracy |
| `api.py` | API REST FastAPI con caché, rate limiting y endpoints completos |
| `app.py` | Dashboard Streamlit con logos oficiales MLB, gráficas Plotly y análisis detallado |

---

## 3. Pipeline de Datos

El pipeline opera **completamente de forma autónoma** mediante tres disparadores programados en GitHub Actions (horario UTC):

```
16:30 UTC → mlb_update_real_results.py   (actualiza resultados del día anterior)
17:30 UTC → mlb_daily_scraper.py         (scraping del día actual: lineups + stats)
18:30 UTC → mlb_predict_engine.py        (genera y guarda predicciones del día)
```

### Flujo de scraping diario

```
Baseball-Reference (/schedule/MLB/2026.shtml)
         │
         ▼
 mlb_schedule_utils.py
 ┌─────────────────────────────────────────────┐
 │ 1. Detecta sección de fecha correcta        │
 │ 2. Soporta headers: "May 1" y "Today's Games"│
 │ 3. Valida que la sección coincida con fecha │
 │    objetivo antes de procesar               │
 └─────────────────────────────────────────────┘
         │
         ▼
 mlb_daily_scraper.py
 ┌─────────────────────────────────────────────┐
 │ 1. Extrae lineup confirmado de cada equipo  │
 │ 2. Extrae estadísticas del abridor          │
 │    (ERA, WHIP, K/9, SO9)                    │
 │ 3. Extrae estadísticas de bullpen           │
 │    (ERA, WHIP por equipo)                   │
 │ 4. Extrae OPS y BA del equipo y top batters │
 │ 5. Guarda en historico_partidos (SQLite)    │
 └─────────────────────────────────────────────┘
         │
         ▼
 mlb_predict_engine.py
 ┌─────────────────────────────────────────────┐
 │ 1. Calcula features temporales (win_rate,   │
 │    rachas, runs_avg) desde histórico        │
 │ 2. Une con features de scraping             │
 │ 3. Calcula Super Features derivadas         │
 │ 4. Genera predicción con XGBoost V3.5       │
 │ 5. Guarda SOLO si scraping fue exitoso      │
 └─────────────────────────────────────────────┘
```

### Principio de integridad de datos

El sistema implementa una regla estricta de calidad:

- ✅ **Con scraping completo** → Se guarda en `predicciones_historico`
- ❌ **Sin scraping (fallback)** → Se devuelve al usuario pero **NO se persiste** en la base de datos

Esto garantiza que las métricas de accuracy reflejen únicamente el rendimiento real del modelo con datos completos.

---

## 4. Motor de Machine Learning

### Algoritmo

El sistema utiliza **XGBoost (Extreme Gradient Boosting)**, optimizado mediante Grid Search con validación cruzada de 3 folds.

```python
# Parámetros de búsqueda (mlb_config.py)
MODEL_CONFIG = {
    "test_size": 0.20,
    "cv_folds": 3,
    "param_grid": {
        "n_estimators": [200, 300, 400],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05],
        "gamma": [0.1, 0.2],
    },
}
```

### Estrategia de entrenamiento híbrido

El modelo se entrena con una **estrategia híbrida inteligente** que combina dos fuentes de datos:

| Fuente | Temporadas | Features disponibles |
|---|---|---|
| CSV históricos | 2022–2024 | Solo features temporales (win rates, rachas) |
| Base de datos + Scraping | 2026 (temporada actual) | Set completo: temporales + scraping + super features |

Esta arquitectura permite entrenar sobre un volumen grande de datos históricos mientras se beneficia de las features avanzadas de la temporada en curso.

### Reentrenamiento incremental automático

El modelo se reentrena automáticamente al alcanzar hitos de partidos disputados:

```
Hitos de reentrenamiento: 486 → 972 → 1,458 → 1,944 → 2,430 partidos
```

El sistema evalúa el hito actual en cada ejecución y lanza el reentrenamiento solo cuando corresponde, evitando cargas computacionales innecesarias.

---

## 5. Ingeniería de Features

El sistema utiliza **38 features** organizadas en tres capas de abstracción:

### Layer 1 — Features Temporales (9 features)
Calculadas a partir del historial de partidos en SQLite. No requieren scraping.

| Feature | Descripción |
|---|---|
| `home_win_rate_10` | Tasa de victorias del local en los últimos 10 partidos |
| `home_racha` | Racha actual (positiva = victorias consecutivas) |
| `home_runs_avg` | Promedio de carreras anotadas en los últimos 10 partidos |
| `home_runs_diff` | Diferencial de carreras (anotadas - recibidas) |
| *(análogos para away)* | |
| `year` | Temporada (para ajuste por cohorte) |

### Layer 2 — Features de Scraping (26 features)
Obtenidas en tiempo real de Baseball-Reference.

| Categoría | Features |
|---|---|
| **Equipo (ofensiva)** | `home_team_OPS`, `away_team_OPS`, `diff_team_BA`, `diff_team_OPS` |
| **Abridor** | `home_starter_ERA`, `home_starter_WHIP`, `home_starter_SO9`, `diff_starter_ERA`, `diff_starter_WHIP`, `diff_starter_SO9` |
| **Mejores bateadores** | `home_best_OPS`, `away_best_OPS`, `diff_best_OPS`, `diff_best_BA`, `diff_best_HR` |
| **Bullpen** | `home_bullpen_ERA`, `away_bullpen_ERA`, `home_bullpen_WHIP`, `away_bullpen_WHIP`, `diff_bullpen_ERA`, `diff_bullpen_WHIP` |
| **Anclas** | `anchor_pitching_level`, `anchor_offensive_level` |

### Layer 3 — Super Features (3 features derivadas)
Métricas compuestas diseñadas para capturar interacciones clave entre pitcheo y bateo:

```python
# Super Feature 1: Neutralización WHIP vs OPS
# Mide cómo el WHIP del abridor neutraliza el OPS del equipo rival
super_neutralizacion_whip_ops = (home_WHIP × away_OPS) − (away_WHIP × home_OPS)

# Super Feature 2: Resistencia ERA vs OPS
# Capacidad del abridor de resistir la ofensiva contraria
super_resistencia_era_ops = (home_ERA / (away_OPS + 0.01)) − (away_ERA / (home_OPS + 0.01))

# Super Feature 3: Muro del Bullpen
# Efectividad del bullpen frente a los mejores bateadores rivales
super_muro_bullpen = (home_bullpen_WHIP × away_best_OPS) − (away_bullpen_WHIP × home_best_OPS)
```

Estas super features capturan relaciones no lineales entre variables que el modelo difícilmente aprendería de forma aislada, y han sido el principal factor de mejora de la precisión desde la versión V3.0.

### Validación y detección de outliers

El módulo `mlb_feature_engineering.py` incluye un sistema automático de detección de valores atípicos con umbrales específicos por estadística (ERA, WHIP, OPS, win_rate) que alerta sobre posibles errores de scraping antes de generar predicciones.

---

## 6. API REST y Frontend

### API FastAPI (V3.5.2)

La API está construida con **FastAPI** y expone endpoints para todos los casos de uso del sistema:

```
GET  /                          → Información general de la API
GET  /health                    → Health check para Docker/monitoreo
GET  /games/today               → Partidos del día actual
GET  /predictions/today         → Predicciones del día actual
POST /predict                   → Predicción manual (equipos + pitcheros)
POST /predict/detailed          → Análisis detallado con caché TTL=1h
GET  /compare/{fecha}           → Comparación predicciones vs resultados reales
GET  /results                   → Historial de resultados con accuracy
GET  /stats/accuracy            → Estadísticas globales de rendimiento
```

**Características de producción:**
- **Rate Limiting** por IP+ruta: 180 req/min globales, 30 req/min en endpoints de predicción
- **CORS** configurable por variable de entorno `ALLOWED_ORIGINS`
- **Caché TTL** de 1 hora para análisis detallados (evita scraping redundante)
- **Validación Pydantic V2** en todos los modelos de entrada/salida
- **Health check** con endpoint dedicado `/health`

### Dashboard Streamlit

El frontend (`app.py`) es una aplicación web interactiva con:

- **Logos oficiales MLB** desde CDN `mlbstatic.com` para los 30 equipos
- **Visualizaciones Plotly** interactivas (gráficas de accuracy por fecha, distribución de confianza)
- **Análisis detallado por partido**: estadísticas del abridor, bullpen y top bateadores comparados visualmente
- **Historial de predicciones** con filtros por fecha y equipo
- **Indicadores de confianza**: BAJA / MODERADA / ALTA / MUY ALTA según la probabilidad predicha

---

## 7. Infraestructura y DevOps

### Docker — Imagen multi-stage optimizada

El proyecto incluye un `Dockerfile` de **dos stages** optimizado para producción:

```dockerfile
# Stage 1: Builder — compila dependencias con gcc/g++
FROM python:3.12-slim as builder
# Crea entorno virtual aislado

# Stage 2: Production — imagen final mínima y segura
FROM python:3.12-slim as production
# Usuario no-root (UID 1000) por seguridad
# Health check integrado
# Expone puerto 8000
```

`docker-compose.yml` orquesta dos servicios con red interna compartida:
- **`mlb-api`** (FastAPI, puerto 8000) con health check
- **`mlb-streamlit`** (Streamlit, puerto 8501) con dependencia condicional sobre la API

### GitHub Actions — Automatización completa

El archivo `.github/workflows/mlb_predictor.yml` (926 líneas) implementa una automatización sofisticada con múltiples jobs:

| Job | Trigger | Descripción |
|---|---|---|
| `update_results` | `cron: 16:30 UTC` | Actualiza resultados reales del día anterior |
| `scrape_pipeline` | `cron: 17:30 UTC` | Scraping + predicción del día en curso |
| `post_scrape_validate` | `cron: 18:30 UTC` | Validación de datos y reentrenamiento si hay nuevo hito |
| `backfill_range` | Manual | Rellena un rango de fechas históricas |
| `manual_all_pipeline` | Manual | Ejecuta el pipeline completo secuencial |
| `predict` | Manual | Genera predicciones para fecha específica |
| `train` | Manual | Fuerza reentrenamiento del modelo |

**Estrategias de robustez en el workflow:**
- `continue-on-error: true` en el paso de scraping (no bloquea el pipeline)
- Reintento automático de scraping si la fecha de la cartelera no coincide
- Push con hasta 3 reintentos (`git push` con rebase `--no-rebase -X ours`)
- Cache de dependencias pip y datos/modelos entre ejecuciones

### CI Pipeline (ci.yml)

Un pipeline de integración continua separado se ejecuta en cada push a `main`:

| Job | Herramienta | Descripción |
|---|---|---|
| `lint` | **Ruff** | Linting y formato automático del código Python |
| `test` | **pytest + pytest-cov** | Suite de tests unitarios con cobertura de código (umbral: 60%) |
| `build-docker` | **Docker** | Construcción y health check de la imagen de producción |
| `security` | **Safety** | Auditoría de vulnerabilidades en dependencias |

> Los jobs de CI están configurados para **omitirse en commits automáticos** del bot (scraping, resultados, reentrenamiento) mediante filtros sobre el mensaje del commit, evitando ciclos de CI innecesarios.

---

## 8. Calidad y Testing

### Suite de tests

```
tests/
├── conftest.py          → Fixtures compartidos (DB en memoria, cliente HTTP de prueba)
├── test_api.py          → Tests de integración de endpoints FastAPI
├── test_config.py       → Tests de configuración y mapeo de equipos
└── test_utils.py        → Tests unitarios de funciones auxiliares
```

### Calidad de código

| Herramienta | Propósito | Configuración |
|---|---|---|
| **Ruff** | Linting + formato | `ruff.toml` en raíz del proyecto |
| **mypy** | Type checking estático | `--ignore-missing-imports` |
| **pytest** | Tests unitarios e integración | `pytest.ini` |
| **pytest-cov** | Cobertura de código | Umbral mínimo: 60% |
| **Safety** | Vulnerabilidades en dependencias | Ejecutado en cada push |

### Integridad de datos

El sistema implementa varios mecanismos para garantizar la integridad del histórico:

1. **`guardar_db=False`** en todas las rutas de fallback: ninguna predicción sin scraping se persiste.
2. **Validación de fecha** en el scraper: se rechaza cualquier sección de calendario que no coincida con la fecha objetivo.
3. **Detección de outliers** en features: valores de ERA, WHIP u OPS fuera de rangos históricos generan advertencias antes de guardar.
4. **Tabla `sync_control`**: registra la fuente y timestamp de cada sincronización de datos.

---

## 9. Evolución y Cambios Clave Recientes

### Fix: Parsing robusto del calendario de Baseball-Reference
**Problema:** El scraper fallaba cuando Baseball-Reference mostraba "Today's Games" en lugar de una fecha formal como "Thursday, May 1". Esto provocaba que el pipeline completo no ingiriera datos, generando un hueco en la base de datos.

**Solución:** Se actualizó `mlb_schedule_utils.py` para detectar ambos formatos mediante expresión regular y atributo HTML `is_today`, y `mlb_daily_scraper.py` para procesar secciones marcadas como día actual aunque la fecha parseada sea `None`.

```python
# Caso especial: "Today's Games"
if "today's games" in texto.lower() and target_date:
    return target_date
```

### Fix: Principio Scrape-or-Nothing para el histórico
**Problema:** Cuando el scraping fallaba, el sistema generaba predicciones de emergencia con features incompletas y las guardaba en `predicciones_historico`, contaminando las métricas de accuracy. Esto causó una caída artificial del 74% al 33% de acierto.

**Solución:** Se añadió el parámetro `guardar_db=False` en todas las rutas de fallback (`api.py` y `mlb_predict_engine.py`). Las predicciones estimadas se devuelven al usuario para no romper la UX, pero no se persisten en la base de datos.

### Fix: Compatibilidad con GitHub Actions Node.js 24
**Problema:** Los actions `actions/setup-python` y `actions/cache` en versiones antiguas producían warnings de deprecación con el runtime Node.js 24 usado por GitHub.

**Solución:** Actualización masiva en todo el workflow (14 ocurrencias) de `setup-python@v4 → v5` y `cache@v3 → v4`.

### Fix: Corrección de lint (Ruff)
Se corrigieron 10 errores de estilo en `api.py`, `mlb_predict_engine.py`, `mlb_daily_scraper.py` y `mlb_schedule_utils.py` para que el pipeline de CI pase limpiamente en cada push.

---

## 10. Métricas de Rendimiento

### Accuracy por periodo

| Periodo | Accuracy | Condición |
|---|---|---|
| Con datos completos (scraping activo) | **74–80%** | Set completo de 38 features |
| Con datos parciales (sin scraping) | ~33% | Solo features temporales (9 features) |
| Diferencia de calidad | **~40–47 pp** | Evidencia del valor del live scraping |

> Esta diferencia de ~45 puntos porcentuales demuestra cuantitativamente el impacto de las features de scraping y las super features sobre el rendimiento del modelo. Es la razón principal por la que el sistema implementa el principio Scrape-or-Nothing.

### Distribución de predicciones en DB

| Tipo | Cantidad | % del total |
|---|---|---|
| AUTOMÁTICO (pipeline diario) | 360 | 99.2% |
| MANUAL | 3 | 0.8% |
| **Total** | **363** | **100%** |

---

## 11. Stack Tecnológico

| Categoría | Tecnología | Versión |
|---|---|---|
| **Lenguaje** | Python | 3.12 |
| **ML Core** | XGBoost | 3.1.2 |
| **Data Processing** | Pandas, NumPy | ≥2.3, ≥2.4 |
| **ML Utilities** | Scikit-learn | ≥1.8 |
| **Web Scraping** | CloudScraper, BeautifulSoup4 | ≥1.2, ≥4.11 |
| **API Backend** | FastAPI + Uvicorn | ≥0.127, ≥0.40 |
| **Validación** | Pydantic V2 | ≥2.12 |
| **Frontend** | Streamlit | 1.52.2 |
| **Gráficas** | Plotly | 6.5.0 |
| **Base de Datos** | SQLite | (stdlib) |
| **Contenedores** | Docker + Docker Compose | multi-stage |
| **CI/CD** | GitHub Actions | — |
| **Linting** | Ruff | ≥0.1 |
| **Type Checking** | mypy | ≥1.8 |
| **Testing** | pytest + pytest-cov | ≥8.0 |
| **Seguridad** | Safety | — |

---

## Conclusión

MLB Game Predictor es un sistema de producción real, no un prototipo académico. Ingiere datos de forma autónoma cada día, mantiene un histórico íntegro con más de 363 predicciones verificadas, y ha demostrado una precisión del **74–80%** cuando opera con el set completo de features.

La arquitectura está diseñada para escalar: el pipeline de datos es independiente del modelo de ML, el modelo puede reentrenarse sin interrumpir el servicio y el frontend consume la API de forma desacoplada. Todos los componentes están dockerizados, testeados, con linting automático y desplegados mediante CI/CD.

El proyecto representa una integración completa del ciclo de vida de un sistema de ML: **recolección de datos → ingeniería de features → entrenamiento → predicción → evaluación → retroalimentación**, operando de forma completamente autónoma y con salvaguardas de calidad de datos robustas.

---

*Informe generado el 2 de mayo de 2026 | MLB Predictor V3.5.2*
