# ⚾ MLB Game Predictor V4.0

> Sistema de Machine Learning de grado de producción para predicción de partidos de la Major League Baseball (MLB), con pipeline de datos automatizado (100% digital vía la API Oficial de la MLB), API REST, dashboard interactivo, y optimización bayesiana de hiperparámetros vía Optuna.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.127+-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.1.2-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-multi--stage-blue.svg)](https://www.docker.com/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-black.svg)](https://github.com/features/actions)
[![Accuracy](https://img.shields.io/badge/Accuracy-77--82%25-brightgreen.svg)](#métricas-de-rendimiento)

[![MLB Predictor Logo](src/logo.png)](https://mlb-game-predictor-live.streamlit.app/)

> **Acceso de producción:** Los enlaces de API no se publican en este repositorio por seguridad operativa.

---

## Tabla de Contenidos

1. [Acerca de este proyecto](#acerca-de-este-proyecto)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Pipeline de Datos Automatizado](#pipeline-de-datos-automatizado)
4. [Motor de Machine Learning](#motor-de-machine-learning)
5. [Ingeniería de Features](#ingeniería-de-features)
6. [API REST y Dashboard](#api-rest-y-dashboard)
7. [Infraestructura y DevOps](#infraestructura-y-devops)
8. [Ciclo de Vida MLOps](#ciclo-de-vida-mlops)
9. [Instalación y Uso](#instalación-y-uso)
10. [Base de Datos](#base-de-datos)
11. [Métricas de Rendimiento](#métricas-de-rendimiento)
12. [Stack Tecnológico](#stack-tecnológico)
13. [Créditos](#créditos)

---

## Acerca de este proyecto

MLB Game Predictor es un sistema de predicción de partidos MLB de **grado de producción**. Ingiere datos diariamente de forma autónoma utilizando la **API de Estadísticas Oficial de la MLB**, entrena un modelo XGBoost optimizado bayesianamente mediante **Optuna**, genera predicciones y mide su propio rendimiento — todo sin intervención manual.

| Métrica | Valor |
|---|---|
| Precisión (con API de la MLB completo) | **~77–82%** (Exactitud en temporada regular) |
| Juegos predichos en DB | **523+** partidos reales con features |
| Predicciones de baja calidad guardadas | **0** |
| Features por partido | **38** (temporales + estadísticas de pitcheo + super features) |
| Equipos cubiertos | **30/30 MLB** |
| Modelo | XGBoost V4.0 con Optuna (Ajuste Bayesiano) |
| Automatización | GitHub Actions · 3 triggers diarios UTC (Ejecución optimizada en < 10s) |

El principio central de diseño es **"API-or-Nothing"**: las predicciones únicamente se persisten en el histórico cuando cuentan con el set completo de features obtenidas via la API oficial, garantizando que las métricas de accuracy reflejen el rendimiento real del modelo.

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────┐
│              CAPA DE PRESENTACIÓN                       │
│        Streamlit Dashboard (app.py) · :8501             │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP REST
┌───────────────────────▼─────────────────────────────────┐
│                  CAPA DE API                            │
│     FastAPI V3.5.2 (api.py) · :8000                     │
│  Rate Limiting · CORS · Cache TTL · Pydantic V2         │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│             CAPA DE LÓGICA DE NEGOCIO                   │
│  mlb_predict_engine · train_model_hybrid_actions        │
│  mlb_feature_engineering · mlb_daily_scraper            │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                  CAPA DE DATOS                          │
│  SQLite (mlb_reentrenamiento.db) · Pickle Cache         │
│  historico_partidos · predicciones_historico            │
│  historico_real · sync_control                          │
└─────────────────────────────────────────────────────────┘
```

### Módulos principales

| Módulo | Responsabilidad |
|---|---|
| `mlb_config.py` | Configuración centralizada: rutas, mapeos de equipos, hiperparámetros |
| `mlb_daily_scraper.py` | Ingestión diaria oficial API-First: consulta la API de Estadísticas de la MLB de forma directa y autónoma |
| `mlb_feature_engineering.py` | Super Features derivadas, validación y detección de outliers |
| `train_model_hybrid_actions.py` | Entrenamiento XGBoost híbrido con optimización bayesiana (Optuna) |
| `mlb_predict_engine.py` | Motor de predicción con control de calidad `guardar_db` |
| `mlb_update_real_results.py` | Actualización de resultados reales para cálculo de accuracy |
| `api.py` | API REST FastAPI: predicciones, historial, estadísticas, health check |
| `app.py` | Dashboard Streamlit con logos MLB, gráficas Plotly y análisis detallado |

---

## Pipeline de Datos Automatizado

El pipeline opera **de forma completamente autónoma** mediante 3 triggers diarios en GitHub Actions. Utilizando la **API de Estadísticas de la MLB (MLB Stats API)** como fuente de datos única y definitiva, el job de ingesta y generación de predicciones se completa de forma extremadamente eficiente en **menos de 10 segundos** de forma digital y sin riesgo de bloqueos.

```
16:30 UTC → mlb_update_real_results.py   # Resultados del día anterior
17:30 UTC → mlb_daily_scraper.py         # Ingestión digital del día actual (API Oficial)
18:30 UTC → mlb_predict_engine.py        # Predicciones del día
```

### Flujo de Ingestión Digital (API-First)

El sistema consulta directamente la API oficial pública `statsapi.mlb.com` para obtener el calendario y abridores probables de la jornada, mapeando automáticamente los nombres y estadísticas de temporada de los abridores (ERA, WHIP, H9, SO9, W, L).

Para garantizar la **compatibilidad del 100% de los scripts y bases de datos aguas abajo (downstream)**:
* Los nombres de equipos oficiales de la API se estandarizan a los códigos de 3 letras del proyecto (ej: `Cincinnati Reds` -> `CIN`, `Los Angeles Angels` -> `LAA`).
* Se generan dinámicamente enlaces relativos sintéticos en la columna `box_score_url` (ej: `/boxes/PHI/PHI202605200.shtml`), manteniendo total compatibilidad retroactiva con la base de datos sin necesidad de reescribir esquemas o scripts downstream.

```
                  ┌─────────────────────────────────┐
                  │      Ejecución del Pipeline     │
                  └────────────────┬────────────────┘
                                   │
                  ┌────────────────▼────────────────┐
         └───────────────┬─────────────┘   └─┬───────────────────────────┘
                         │                   │
                         └─────────┬─────────┘
                                   │ Estandarización de códigos y URLs
                  ┌────────────────▼────────────────┐
                  │      mlb_daily_scraper.py       │
                  │   · Datos de abridor/bullpen    │
                  │   · Guardado en DB              │
                  └────────────────┬────────────────┘
                                   │
                  ┌────────────────▼────────────────┐
                  │      mlb_predict_engine.py      │
                  │   · Unión con win rates/rachas  │
                  │   · Cálculo de Super Features   │
                  │   · Predicción XGBoost V3.5     │
                  │   · Persistencia en histórico   │
                  └─────────────────────────────────┘
```

**Integridad garantizada:** si la obtención de datos falla por completo, la API puede devolver una estimación al usuario, pero **nunca persiste** en `predicciones_historico` para no contaminar las métricas de accuracy.

---

## Motor de Machine Learning

### Algoritmo: XGBoost con Ajuste Bayesiano (Optuna)

En lugar de una búsqueda en rejilla estática (`GridSearchCV`), la versión 4.0 implementa una optimización bayesiana automática mediante **Optuna** (35 trials) que optimiza dinámicamente el Accuracy utilizando una validación cruzada estratificada de 3 splits (`StratifiedKFold`):

```python
# Hiperparámetros ajustados dinámicamente por el estudio de Optuna:
params = {
    "n_estimators": trial.suggest_int("n_estimators", 150, 450),
    "max_depth": trial.suggest_int("max_depth", 3, 9),
    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
    "gamma": trial.suggest_float("gamma", 0.0, 0.5),
    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
}
```

Esta optimización dinámica ha mejorado la precisión del modelo en más de un **3.5% directo en test** frente al GridSearchCV heredado, reduciendo drásticamente el *overfitting*.

### Estrategia de entrenamiento híbrido

| Fuente | Temporadas | Features disponibles |
|---|---|---|
| CSV históricos | 2022–2024 | Features temporales (9) |
| DB + API oficial en vivo | 2026 | Set completo: 38 features |

### Reentrenamiento incremental automático

El modelo se reentrena automáticamente al alcanzar hitos de partidos en la temporada actual:

```
Hitos: 486 → 972 → 1,458 → 1,944 → 2,430 partidos
```

El workflow evalúa el hito en cada ejecución y lanza el reentrenamiento solo si corresponde. El modelo anterior se preserva en backup y solo se reemplaza si el nuevo supera el accuracy.

---

## Ingeniería de Features

El sistema utiliza **38 features** en tres capas:

### Layer 1 — Features Temporales (9)
Calculadas desde el historial de la DB sin necesidad de scraping.

| Feature | Descripción |
|---|---|
| `home_win_rate_10` | Tasa de victorias local, últimos 10 partidos |
| `home_racha` | Racha actual (positivo = victorias consecutivas) |
| `home_runs_avg` | Promedio de carreras anotadas |
| `home_runs_diff` | Diferencial carreras anotadas - recibidas |
| *(análogos para `away_`)* | |
| `year` | Temporada (ajuste por cohorte) |

### Layer 2 — Features de la API (26)
Obtenidas en tiempo real de la API oficial de la MLB.

| Categoría | Features clave |
|---|---|
| Equipo (ofensiva) | `home_team_OPS`, `away_team_OPS`, `diff_team_BA`, `diff_team_OPS` |
| Abridor | `home_starter_ERA`, `home_starter_WHIP`, `home_starter_SO9`, `diff_starter_ERA` |
| Top bateadores | `home_best_OPS`, `away_best_OPS`, `diff_best_OPS`, `diff_best_HR` |
| Bullpen | `home_bullpen_ERA`, `away_bullpen_ERA`, `home_bullpen_WHIP`, `diff_bullpen_ERA` |
| Anclas | `anchor_pitching_level`, `anchor_offensive_level` |

### Layer 3 — Super Features (3)
Métricas compuestas que capturan interacciones no lineales entre pitcheo y bateo:

```
super_neutralizacion_whip_ops = (home_WHIP × away_OPS) − (away_WHIP × home_OPS)
super_resistencia_era_ops     = (home_ERA / (away_OPS + 0.01)) − (away_ERA / (home_OPS + 0.01))
super_muro_bullpen            = (home_bullpen_WHIP × away_best_OPS) − (away_bullpen_WHIP × home_best_OPS)
```

Estas super features son el principal factor de mejora de precisión respecto a versiones anteriores, añadiendo ~15–20 puntos porcentuales de accuracy.

---

## API REST y Dashboard

### Endpoints FastAPI (V3.5.2)

```
GET  /health              → Health check (Docker / monitoreo)
GET  /games/today         → Partidos del día actual
GET  /predictions/today   → Predicciones del día actual
POST /predict             → Predicción manual (equipos + pitcheros)
POST /predict/detailed    → Análisis detallado con caché TTL=1h
GET  /compare/{fecha}     → Predicciones vs resultados reales
GET  /results             → Historial con accuracy calculado
GET  /stats/accuracy      → Estadísticas globales de rendimiento
```

**Características de producción:**
- Rate Limiting por IP+ruta: 180 req/min globales, 30 req/min en predicciones
- CORS configurable via variable de entorno `ALLOWED_ORIGINS`
- Caché de análisis detallados con TTL de 1 hora
- Validación Pydantic V2 en todos los modelos de entrada/salida

### Dashboard Streamlit

- Logos oficiales MLB desde CDN `mlbstatic.com` para los 30 equipos
- Gráficas Plotly interactivas: accuracy por fecha, distribución de confianza
- Análisis detallado por partido: abridor, bullpen y top bateadores comparados
- Indicadores de confianza: BAJA / MODERADA / ALTA / MUY ALTA

### Ejemplo de salida de predicción

```
===========================================================================
   ⚾ MLB PREDICTOR V3.5 - ANÁLISIS ESTADÍSTICO
===========================================================================
 Encuentro: NYY vs BOS
 Temporada: 2026 | Ingesta: API de la MLB

📊 COMPARATIVA DE EQUIPOS:
 🏠  NYY: OPS: 0.782 | Bullpen WHIP: 1.234
 ✈️  BOS: OPS: 0.758 | Bullpen WHIP: 1.301

👤 LANZADORES ABRIDORES:
 🏠 Gerrit Cole:    ERA: 3.12 | WHIP: 1.089 | SO9: 10.2
 ✈️  Tanner Houck:  ERA: 3.89 | WHIP: 1.234 | SO9: 8.7

🧱 ANÁLISIS DE BULLPEN:
 🏠 NYY: ERA: 3.456 | WHIP: 1.234
 ✈️  BOS: ERA: 3.891 | WHIP: 1.301

📈 TENDENCIAS (Últimos 10 partidos):
 🏠 NYY: Win Rate: 70.0% | Racha: +5
 ✈️  BOS: Win Rate: 40.0% | Racha: -3

===========================================================================
 🏆 GANADOR PREDICHO: New York Yankees
 Probabilidades: NYY 68.5% | BOS 31.5% | Confianza: ALTA
===========================================================================

🚀 SUPER FEATURES:
 🛡️ Neutralización WHIP/OPS: -0.0234  (ventaja NYY)
 📉 Resistencia ERA/OPS:      -0.1456  (ventaja NYY)
 🧱 Muro Bullpen:             -0.0891  (ventaja NYY)
```

---

## Infraestructura y DevOps

### Docker — Imagen multi-stage

```dockerfile
# Stage 1: Builder — compila dependencias (gcc/g++)
FROM python:3.12-slim as builder

# Stage 2: Production — imagen mínima, usuario no-root
FROM python:3.12-slim as production
# · Usuario UID 1000 (seguridad)
# · Health check integrado cada 30s
# · Expone puerto 8000
```

`docker-compose.yml` orquesta dos servicios en red interna:
- **`mlb-api`** (FastAPI, :8000) con health check
- **`mlb-streamlit`** (Streamlit, :8501) con `depends_on: service_healthy`

### GitHub Actions — 7 jobs automatizados

| Job | Trigger | Descripción |
|---|---|---|
| `update_results` | cron 16:30 UTC | Actualiza resultados reales |
| `scrape_pipeline` | cron 17:30 UTC | Ingesta + predicción del día |
| `post_scrape_validate` | cron 18:30 UTC | Validación + reentrenamiento si hay hito |
| `backfill_range` | Manual | Rellena rango de fechas históricas |
| `manual_all_pipeline` | Manual | Pipeline completo secuencial |
| `predict` | Manual | Predicciones para fecha específica |
| `train` | Manual | Fuerza reentrenamiento del modelo |

**Estrategias de robustez:**
- `continue-on-error: true` en el paso de scraping
- Reintento automático si hay desfase de fecha en la cartelera
- Push con hasta 3 reintentos y merge automático `--no-rebase -X ours`
- Cache de dependencias pip y artefactos entre ejecuciones

### CI Pipeline (ci.yml)

| Job | Herramienta | Descripción |
|---|---|---|
| `lint` | Ruff | Linting y formato automático |
| `test` | pytest + pytest-cov | Tests con cobertura mínima del 60% |
| `build-docker` | Docker | Build + health check de imagen de producción |
| `security` | Safety | Auditoría de vulnerabilidades en dependencias |
Los jobs de CI se omiten automáticamente en commits del bot (scraping, resultados, reentrenamiento) para evitar ciclos innecesarios.

---

## Ciclo de Vida MLOps

El sistema implementa un ciclo de MLOps (Machine Learning Operations) continuo, auto-regulado y diseñado para mitigar la degradación del modelo (concept drift) a lo largo de la temporada.

### 1. Ingestión Continua (Data Drift Monitoring)
El pipeline diario ingiere nuevos datos. La integración de la **MLB Stats API** elimina el data drift y las interrupciones operativas al eliminar la dependencia de cambios estructurales en el HTML origen. Si ocurren fallos de red o de parsing en la API, la capa de validación (`post_scrape_validation`) y los mecanismos de Auto-Heal (`mlb_mass_healer.py`) aseguran que no queden "huecos" temporales que puedan corromper las secuencias o rachas de los equipos.

### 2. Entrenamiento e Hitos Automáticos
No requiere intervención manual para el reentrenamiento. El sistema monitorea el volumen de la base de datos y detona el entrenamiento de un nuevo modelo (Challenger) al cruzar hitos predefinidos (e.g., 486, 972 partidos).

### 3. Shadow Testing y Validación
- Cuando se entrena un modelo **Challenger**, se evalúa utilizando validación cruzada y optimización bayesiana con **Optuna** (35 trials).
- Se compara frente al modelo **Champion** actual usando el conjunto de test más reciente.
- Solo si el **Challenger** mejora el accuracy global sin caer en sobreajuste, reemplaza al Champion. En caso de error o degradación, el sistema retiene la versión previa (Backup fallback).

### 4. Monitoreo de Rendimiento (Feedback Loop)
Todos los días a las 10:00 UTC, el job `update_real_results` busca el resultado de los partidos predecidos el día anterior. 
Este feedback se escribe en `historico_real`, permitiendo que el dashboard de Streamlit actualice la métrica de **Accuracy Global** en tiempo real. Esto provee un termómetro inmediato de la salud del modelo en producción.

### 5. Inferencia en Producción (Serving)
El modelo ganador se expone mediante FastAPI, protegiendo los cálculos costosos con cachés y evitando fallos mediante validaciones asíncronas tipo Pydantic V2.

---

## Instalación y Uso

### Requisitos

- Python 3.12+
- Docker (opcional, recomendado para producción)

### Instalación local

```bash
git clone https://github.com/GaboLarrazabal13/MLB-GAME-PREDICTOR.git
cd MLB-GAME-PREDICTOR
pip install -r requirements.txt

# Verificar configuración
python src/mlb_config.py
```

### Ejecución con Docker

```bash
# Construir y levantar API + Dashboard
docker-compose up --build

# API disponible en http://localhost:8000
# Dashboard en   http://localhost:8501
```

### Ejecución local

```bash
# API FastAPI
cd src && uvicorn api:app --reload --port 8000

# Dashboard Streamlit (en otra terminal)
cd src && streamlit run app.py
```

### Comandos del pipeline

```bash
# Scraping del día
cd src && python mlb_daily_scraper.py

# Generar predicciones
cd src && TARGET_DATE=2026-05-02 python mlb_predict_engine.py

# Actualizar resultados reales
cd src && python mlb_update_real_results.py

# Reentrenar modelo
cd src && python train_model_hybrid_actions.py
```

### Predicción manual (CLI)

```bash
# Modo interactivo
python src/mlb_manual_interface.py

# Modo rápido por argumentos
python src/mlb_manual_interface.py NYY BOS "Gerrit Cole" "Tanner Houck" 2026
```

### Configuración avanzada

Todos los parámetros se centralizan en `src/mlb_config.py`:

```python
# Hiperparámetros del modelo (Espacio de búsqueda de Optuna)
MODEL_CONFIG = {
    "test_size": 0.20, "cv_folds": 3,
    "optuna_search_space": {
        "n_estimators_range": (150, 450),
        "max_depth_range": (3, 9),
        "learning_rate_range": (0.005, 0.1),
        "gamma_range": (0.0, 0.5),
        "subsample_range": (0.6, 1.0),
        "colsample_bytree_range": (0.6, 1.0),
        "min_child_weight_range": (1, 8),
        "reg_alpha_range": (1e-8, 5.0),
        "reg_lambda_range": (1e-8, 5.0),
    },
}

# Rate limiting del scraper
SCRAPING_CONFIG = {
    "max_retries": 3, "timeout": 15,
    "min_delay": 2, "max_delay": 4,
    "bloque_size": 150, "pausa_entre_bloques": 45,
}
```

### Solución de problemas comunes

| Error | Causa | Solución |
|---|---|---|
| `No se pudieron extraer features` | Nombre del lanzador no encontrado | Usar nombre completo exacto como aparece en la API de la MLB |
| `Rate limit (429) detectado` | Demasiadas peticiones | El sistema espera automáticamente; aumentar `pausa_entre_bloques` |
| Accuracy baja repentinamente | Predicciones sin API en histórico | Verificar que `guardar_db=False` en rutas de fallback |
| Fallo en la MLB Stats API | Indisponibilidad o datos de lanzadores incompletos | El sistema reintenta automáticamente hasta 3 veces de forma transparente |

---

## Base de Datos

### `historico_partidos` — Partidos scrapeados

| Columna | Tipo | Descripción |
|---|---|---|
| `fecha` | TEXT | Fecha del partido (YYYY-MM-DD) |
| `home_team` | TEXT | Código equipo local (ej: NYY) |
| `away_team` | TEXT | Código equipo visitante |
| `home_pitcher` | TEXT | Abridor local confirmado |
| `away_pitcher` | TEXT | Abridor visitante confirmado |
| `year` | INTEGER | Temporada |

### `predicciones_historico` — Predicciones generadas

| Columna | Tipo | Descripción |
|---|---|---|
| `fecha` | TEXT | Fecha de la predicción |
| `prob_home` | REAL | Probabilidad del local (0–100) |
| `prob_away` | REAL | Probabilidad del visitante (0–100) |
| `prediccion` | TEXT | Equipo predicho como ganador |
| `confianza` | TEXT | MUY ALTA / ALTA / MODERADA / BAJA |
| `tipo` | TEXT | AUTOMATICO o MANUAL |
| `detalles` | TEXT | JSON con features y stats completas |

### `historico_real` — Resultados reales

| Columna | Tipo | Descripción |
|---|---|---|
| `score_home` | INTEGER | Carreras equipo local |
| `score_away` | INTEGER | Carreras equipo visitante |
| `ganador` | INTEGER | 1 = local ganó, 0 = visitante ganó |

---

## Métricas de Rendimiento

| Condición | Accuracy | Features usadas |
|---|---|---|
| Con features de la API | **74–80%** | 38 features (todas las capas) |
| Sin features (solo básicas) | ~33% | 9 features (solo temporales) |
| **Diferencia** | **~45 pp** | Impacto cuantificado de las features de la API |

Esta diferencia de ~45 puntos porcentuales demuestra el valor de las features avanzadas y las super features. Es la razón fundamental del principio API-or-Nothing implementado en el sistema.

### Distribución actual del histórico

| Tipo | Cantidad | % |
|---|---|---|
| AUTOMÁTICO (pipeline diario) | 520 | 99.4% |
| MANUAL | 3 | 0.6% |
| **Total** | **523+** | |

---

## Stack Tecnológico

| Categoría | Tecnología | Versión |
|---|---|---|
| Lenguaje | Python | 3.12 |
| ML Core | XGBoost | 3.1.2 |
| Data Processing | Pandas, NumPy | ≥2.3, ≥2.4 |
| ML Utilities | Scikit-learn | ≥1.8 |
| Hyperparameter Tuning | Optuna | ≥3.6.0 |
| Conexión API | Requests | ≥2.32 |
| API Backend | FastAPI + Uvicorn | ≥0.127, ≥0.40 |
| Validación | Pydantic V2 | ≥2.12 |
| Frontend | Streamlit | 1.52.2 |
| Gráficas | Plotly | 6.5.0 |
| Base de Datos | SQLite | stdlib |
| Contenedores | Docker + Docker Compose | multi-stage |
| CI/CD | GitHub Actions | — |
| Linting | Ruff | ≥0.1 |
| Type Checking | mypy | ≥1.8 |
| Testing | pytest + pytest-cov | ≥8.0 |
| Seguridad | Safety | — |

---

## Créditos

- **Datos:** API Oficial de la MLB (statsapi.mlb.com)
- **ML Framework:** [XGBoost](https://xgboost.readthedocs.io/)

---

## Disclaimer

Este sistema es para fines educativos y de entretenimiento. No debe usarse para apuestas deportivas. Las predicciones son estimaciones basadas en datos históricos y estadísticos, y no garantizan resultados futuros.

---

**Autor:** Gabriel Larrazabal | **Versión:** V4.0.0 | **Última actualización:** Mayo 2026

*MLB Game Predictor V4.0 — Inteligencia estadística aplicada al diamante.*