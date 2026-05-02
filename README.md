# ⚾ MLB Game Predictor V3.5

> Sistema de Machine Learning para predicción de partidos de la Major League Baseball (MLB), con pipeline de datos completamente automatizado, API REST, dashboard interactivo y orquestación mediante GitHub Actions.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.127+-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.1.2-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-multi--stage-blue.svg)](https://www.docker.com/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-black.svg)](https://github.com/features/actions)
[![Accuracy](https://img.shields.io/badge/Accuracy-74--80%25-brightgreen.svg)](#métricas-de-rendimiento)

![MLB Predictor Logo](src/logo.png)

> **Acceso de producción:** Los enlaces de API y frontend no se publican en este repositorio por seguridad operativa.

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Pipeline de Datos Automatizado](#pipeline-de-datos-automatizado)
4. [Motor de Machine Learning](#motor-de-machine-learning)
5. [Ingeniería de Features](#ingeniería-de-features)
6. [API REST y Dashboard](#api-rest-y-dashboard)
7. [Infraestructura y DevOps](#infraestructura-y-devops)
8. [Instalación y Uso](#instalación-y-uso)
9. [Base de Datos](#base-de-datos)
10. [Métricas de Rendimiento](#métricas-de-rendimiento)
11. [Stack Tecnológico](#stack-tecnológico)
12. [Créditos](#créditos)

---

## Resumen Ejecutivo

MLB Game Predictor es un sistema de predicción de partidos MLB de **grado de producción**. Ingiere datos diariamente de forma autónoma, entrena un modelo XGBoost con 38 features avanzadas, genera predicciones y mide su propio rendimiento — todo sin intervención manual.

| Métrica | Valor |
|---|---|
| Precisión (con scraping completo) | **~74–80%** |
| Juegos predichos en DB | **363+** |
| Predicciones de baja calidad guardadas | **0** |
| Features por partido | **38** (temporales + scraping + super features) |
| Equipos cubiertos | **30/30 MLB** |
| Modelo | XGBoost V3.5 con GridSearchCV |
| Automatización | GitHub Actions · 3 triggers diarios UTC |

El principio central de diseño es **"Scrape-or-Nothing"**: las predicciones únicamente se persisten en el histórico cuando cuentan con el set completo de features obtenidas via live scraping, garantizando que las métricas de accuracy reflejen el rendimiento real del modelo.

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
| `mlb_daily_scraper.py` | Scraping diario de Baseball-Reference: lineups, abridores, bullpen |
| `mlb_schedule_utils.py` | Parsing robusto del calendario MLB (soporta múltiples formatos de fecha) |
| `mlb_feature_engineering.py` | Super Features derivadas, validación y detección de outliers |
| `train_model_hybrid_actions.py` | Entrenamiento XGBoost híbrido con GridSearchCV |
| `mlb_predict_engine.py` | Motor de predicción con control de calidad `guardar_db` |
| `mlb_update_real_results.py` | Actualización de resultados reales para cálculo de accuracy |
| `api.py` | API REST FastAPI: predicciones, historial, estadísticas, health check |
| `app.py` | Dashboard Streamlit con logos MLB, gráficas Plotly y análisis detallado |

---

## Pipeline de Datos Automatizado

El pipeline opera **de forma completamente autónoma** mediante 3 triggers diarios en GitHub Actions:

```
16:30 UTC → mlb_update_real_results.py   # Resultados del día anterior
17:30 UTC → mlb_daily_scraper.py         # Scraping del día actual
18:30 UTC → mlb_predict_engine.py        # Predicciones del día
```

### Flujo de scraping

```
Baseball-Reference → mlb_schedule_utils.py (parsing de fecha)
        ↓
mlb_daily_scraper.py
  · Lineup confirmado por equipo
  · Stats del abridor (ERA, WHIP, K/9)
  · Stats del bullpen (ERA, WHIP)
  · OPS y BA del equipo + top bateadores
  · Guardado en historico_partidos
        ↓
mlb_predict_engine.py
  · Features temporales (win rates, rachas) desde DB
  · Union con features de scraping
  · Calculo de Super Features
  · Prediccion XGBoost V3.5
  · Guardado SOLO si scraping fue exitoso
```

**Integridad garantizada:** si el scraping falla, la API puede devolver una estimación al usuario, pero **nunca persiste** en `predicciones_historico` para no contaminar las métricas.

---

## Motor de Machine Learning

### Algoritmo: XGBoost con búsqueda de hiperparámetros

```python
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

| Fuente | Temporadas | Features disponibles |
|---|---|---|
| CSV históricos | 2022–2024 | Features temporales (9) |
| DB + Scraping vivo | 2026 | Set completo: 38 features |

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

### Layer 2 — Features de Scraping (26)
Obtenidas en tiempo real de Baseball-Reference.

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
 Temporada: 2026 | Scraping: Baseball-Reference

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
| `scrape_pipeline` | cron 17:30 UTC | Scraping + predicción del día |
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
# Hiperparámetros del modelo
MODEL_CONFIG = {
    "test_size": 0.20, "cv_folds": 3,
    "param_grid": {
        "n_estimators": [200, 300, 400],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05],
        "gamma": [0.1, 0.2],
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
| `No se pudieron extraer features` | Nombre del lanzador no encontrado | Usar nombre completo exacto como aparece en Baseball-Reference |
| `Rate limit (429) detectado` | Demasiadas peticiones | El sistema espera automáticamente; aumentar `pausa_entre_bloques` |
| Accuracy baja repentinamente | Predicciones sin scraping en histórico | Verificar que `guardar_db=False` en rutas de fallback |

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
| Con scraping completo | **74–80%** | 38 features (todas las capas) |
| Sin scraping (fallback) | ~33% | 9 features (solo temporales) |
| **Diferencia** | **~45 pp** | Impacto cuantificado del live scraping |

Esta diferencia de ~45 puntos porcentuales demuestra el valor de las features de scraping y las super features. Es la razón fundamental del principio Scrape-or-Nothing implementado en el sistema.

### Distribución actual del histórico

| Tipo | Cantidad | % |
|---|---|---|
| AUTOMÁTICO (pipeline diario) | 360 | 99.2% |
| MANUAL | 3 | 0.8% |
| **Total** | **363+** | |

---

## Stack Tecnológico

| Categoría | Tecnología | Versión |
|---|---|---|
| Lenguaje | Python | 3.12 |
| ML Core | XGBoost | 3.1.2 |
| Data Processing | Pandas, NumPy | ≥2.3, ≥2.4 |
| ML Utilities | Scikit-learn | ≥1.8 |
| Web Scraping | CloudScraper, BeautifulSoup4 | ≥1.2, ≥4.11 |
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

- **Datos:** [Baseball-Reference](https://www.baseball-reference.com)
- **ML Framework:** [XGBoost](https://xgboost.readthedocs.io/)
- **Web Scraping:** [cloudscraper](https://github.com/VeNoMouS/cloudscraper)

---

## Disclaimer

Este sistema es para fines educativos y de entretenimiento. No debe usarse para apuestas deportivas. Las predicciones son estimaciones basadas en datos históricos y estadísticos, y no garantizan resultados futuros.

---

**Autor:** Gabriel Larrazabal | **Versión:** V3.5.2 | **Última actualización:** Mayo 2026

*MLB Game Predictor V3.5 — Inteligencia estadística aplicada al diamante.*