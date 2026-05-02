"""
API FastAPI para MLB Predictor V3.5 - VERSIÓN CORREGIDA
Endpoints para predicciones manuales, automáticas y análisis de rendimiento
"""

import asyncio
import os
import sqlite3
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from threading import Lock

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Importar módulos del proyecto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_config import DB_PATH, TEAM_CODE_TO_NAME, get_team_code, get_team_name
from mlb_predict_engine import predecir_juego

# ============================================================================
# CONFIGURACIÓN DE LA API
# ============================================================================

app = FastAPI(
    title="MLB Predictor API",
    description="API para predicciones de partidos MLB usando Machine Learning",
    version="3.5.2",
)

# CORS restringido por variable de entorno (coma separada).
# Ejemplo: ALLOWED_ORIGINS="https://tu-frontend.com,https://www.tu-frontend.com"
allowed_origins_env = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
)
ALLOWED_ORIGINS = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting básico por IP+ruta para evitar abuso.
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "180"))
RATE_LIMIT_PREDICT_MAX_REQUESTS = int(
    os.getenv("RATE_LIMIT_PREDICT_MAX_REQUESTS", "30")
)
DETAILED_ANALYSIS_TIMEOUT_SECONDS = int(
    os.getenv("DETAILED_ANALYSIS_TIMEOUT_SECONDS", "120")
)
DETAILED_CACHE_TTL_SECONDS = int(os.getenv("DETAILED_CACHE_TTL_SECONDS", "3600"))
_rate_limit_store: dict[str, deque[float]] = defaultdict(deque)
_rate_limit_lock = Lock()
_detailed_prediction_cache: dict[tuple[str, str, str, str, int], tuple[float, dict]] = {}
_detailed_prediction_cache_lock = Lock()


def _get_client_ip(request: Request) -> str:
    x_forwarded_for = request.headers.get("x-forwarded-for", "")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _get_detailed_cache_key(request: "PrediccionRequest") -> tuple[str, str, str, str, int]:
    return (
        request.home_team,
        request.away_team,
        request.home_pitcher,
        request.away_pitcher,
        int(request.year or 2026),
    )


def _get_cached_detailed_prediction(cache_key):
    now = time.time()
    with _detailed_prediction_cache_lock:
        cached = _detailed_prediction_cache.get(cache_key)
        if not cached:
            return None

        cached_at, payload = cached
        if now - cached_at > DETAILED_CACHE_TTL_SECONDS:
            _detailed_prediction_cache.pop(cache_key, None)
            return None

        return payload


def _set_cached_detailed_prediction(cache_key, payload):
    with _detailed_prediction_cache_lock:
        _detailed_prediction_cache[cache_key] = (time.time(), payload)


def _crear_payload_degradado_rapido(request: "PrediccionRequest", motivo: str | None = None):
    home_code = get_team_code(request.home_team)
    away_code = get_team_code(request.away_team)

    if not home_code or not away_code:
        raise HTTPException(
            status_code=504,
            detail=(
                "El analisis detallado no pudo completarse y el modo rapido "
                "no se pudo inicializar."
            ),
        )

    try:
        resultado = predecir_juego(
            home_team=home_code,
            away_team=away_code,
            home_pitcher=request.home_pitcher,
            away_pitcher=request.away_pitcher,
            year=int(request.year or 2026),
            modo_auto=True,
            guardar_db=False,
            hacer_scraping=False,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                "El analisis detallado no pudo completarse y el modo rapido "
                "tambien fallo."
            ),
        ) from exc

    if not resultado:
        raise HTTPException(
            status_code=504,
            detail=(
                "El analisis detallado no pudo completarse y el modo rapido "
                "no devolvio resultado."
            ),
        )

    prob_home_decimal = (
        resultado["prob_home"] / 100.0
        if resultado["prob_home"] > 1
        else resultado["prob_home"]
    )
    prob_away_decimal = (
        resultado["prob_away"] / 100.0
        if resultado["prob_away"] > 1
        else resultado["prob_away"]
    )

    mensaje_base = (
        "Se devolvio analisis rapido porque el scraping detallado no estuvo "
        "disponible dentro de los limites de la solicitud."
    )
    if motivo:
        mensaje_base = f"{mensaje_base} Motivo: {motivo}"

    return {
        "ganador": resultado["prediccion"],
        "prob_home": prob_home_decimal,
        "prob_away": prob_away_decimal,
        "confianza": max(prob_home_decimal, prob_away_decimal),
        "year_solicitado": request.year,
        "year_usado_home": request.year,
        "year_usado_away": request.year,
        "razon_fallback_home": None,
        "razon_fallback_away": None,
        "ip_home": 0,
        "ip_away": 0,
        "features_usadas": {},
        "stats_detalladas": {
            "home_pitcher": {},
            "away_pitcher": {},
            "home_batters": [],
            "away_batters": [],
        },
        "modo_degradado": True,
        "mensaje": mensaje_base,
    }


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    path = request.url.path

    # Excluir endpoints de metadata/health del límite.
    if path in {"/", "/health", "/docs", "/redoc", "/openapi.json"}:
        return await call_next(request)

    client_ip = _get_client_ip(request)
    max_requests = (
        RATE_LIMIT_PREDICT_MAX_REQUESTS
        if path.startswith("/predict")
        else RATE_LIMIT_MAX_REQUESTS
    )

    now = time.time()
    key = f"{client_ip}:{path}"

    with _rate_limit_lock:
        bucket = _rate_limit_store[key]
        cutoff = now - RATE_LIMIT_WINDOW_SECONDS
        while bucket and bucket[0] < cutoff:
            bucket.popleft()

        if len(bucket) >= max_requests:
            retry_after = int(max(1, RATE_LIMIT_WINDOW_SECONDS - (now - bucket[0])))
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={"Retry-After": str(retry_after)},
            )

        bucket.append(now)

    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(max_requests)
    return response


# ============================================================================
# MODELOS DE DATOS (Pydantic)
# ============================================================================


class PrediccionRequest(BaseModel):
    """Modelo para solicitud de predicción manual"""

    home_team: str = Field(..., description="Código del equipo local (ej: NYY)")
    away_team: str = Field(..., description="Código del equipo visitante (ej: BOS)")
    home_pitcher: str = Field(..., description="Nombre del lanzador local")
    away_pitcher: str = Field(..., description="Nombre del lanzador visitante")
    year: int | None = Field(2026, description="Año para scraping de stats")
    fecha: str | None = Field(None, description="Fecha del partido para buscar en caché")

    class Config:
        schema_extra = {
            "example": {
                "home_team": "NYY",
                "away_team": "BOS",
                "home_pitcher": "Gerrit Cole",
                "away_pitcher": "Tanner Houck",
                "year": 2024,
            }
        }


class PrediccionResponse(BaseModel):
    """Modelo para respuesta de predicción"""

    success: bool
    fecha: str
    home_team: str
    away_team: str
    home_pitcher: str
    away_pitcher: str
    prob_home: float
    prob_away: float
    prediccion: str
    confianza: str
    detalles: dict | None = None


class PartidoHoy(BaseModel):
    """Modelo para partido del día"""

    game_id: str
    fecha: str
    home_team: str
    away_team: str
    home_pitcher: str
    away_pitcher: str
    prediccion: str | None = None
    prob_home: float | None = None
    prob_away: float | None = None
    confianza: str | None = None


class ResultadoReal(BaseModel):
    """Modelo para resultado real"""

    game_id: str
    fecha: str
    home_team: str
    away_team: str
    score_home: int
    score_away: int
    ganador: int
    prediccion: str | None = None
    acierto: bool | None = None


def _obtener_fecha_publicada(conn, dataset, table_name):
    """Obtiene la fecha más nueva entre sync_control y la tabla real."""
    candidatos = []

    try:
        filas = conn.execute(
            "SELECT fecha FROM sync_control WHERE dataset = ?",
            (dataset,),
        ).fetchall()
        candidatos.extend([fecha for (fecha,) in filas if fecha])
    except Exception:
        pass

    try:
        fecha_tabla = conn.execute(f"SELECT MAX(fecha) FROM {table_name}").fetchone()[0]
        if fecha_tabla:
            candidatos.append(fecha_tabla)
    except Exception:
        pass

    return max(candidatos) if candidatos else None


def _obtener_estado_fechas(conn):
    """Resume las fechas más recientes disponibles por dataset."""
    compare_latest = None

    try:
        compare_latest = conn.execute(
            """
            SELECT MAX(r.fecha)
            FROM historico_real r
            INNER JOIN predicciones_historico p
                ON r.fecha = p.fecha
                AND r.home_team = p.home_team
                AND r.away_team = p.away_team
            """
        ).fetchone()[0]
    except Exception:
        compare_latest = None

    return {
        "games_latest": _obtener_fecha_publicada(
            conn, "games_today", "historico_partidos"
        ),
        "predictions_latest": _obtener_fecha_publicada(
            conn, "predictions_today", "predicciones_historico"
        ),
        "results_latest": _obtener_fecha_publicada(
            conn, "results_today", "historico_real"
        ),
        "compare_latest": compare_latest,
    }


def _backfill_predicciones_fecha(fecha: str) -> int:
    """Genera predicciones faltantes para una fecha usando datos ya guardados en DB.

    Prioriza `historico_partidos` (fuente original de predicción del día) y, si no hay
    filas para la fecha, intenta con `historico_real` como recuperación.
    """
    with sqlite3.connect(DB_PATH) as conn:
        df_juegos = pd.read_sql(
            """
            SELECT fecha, home_team, away_team, home_pitcher, away_pitcher, COALESCE(year, 2026) AS year
            FROM historico_partidos
            WHERE fecha = ?
            """,
            conn,
            params=[fecha],
        )

        if df_juegos.empty:
            df_juegos = pd.read_sql(
                """
                SELECT fecha, home_team, away_team, home_pitcher, away_pitcher, COALESCE(year, 2026) AS year
                FROM historico_real
                WHERE fecha = ?
                """,
                conn,
                params=[fecha],
            )

    if df_juegos.empty:
        return 0

    generadas = 0
    for _, row in df_juegos.iterrows():
        try:
            res = predecir_juego(
                row["home_team"],
                row["away_team"],
                row.get("home_pitcher", ""),
                row.get("away_pitcher", ""),
                year=int(row.get("year", 2026)),
                modo_auto=True,
                fecha_partido=fecha,
                hacer_scraping=False,  # solo features temporales para respuesta rápida
                guardar_db=True,       # GUARDAR en DB para que la comparación sea persistente
            )
            if res:
                generadas += 1
        except Exception:
            continue

    if generadas > 0:
        run_source = (
            os.getenv("RUN_SOURCE", "api_backfill").strip().lower() or "api_backfill"
        )
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO sync_control (dataset, source, fecha, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(dataset, source)
                DO UPDATE SET fecha = excluded.fecha,
                              updated_at = CURRENT_TIMESTAMP
                """,
                ("predictions_today", run_source, fecha),
            )
            conn.commit()

    return generadas


# ============================================================================
# ENDPOINTS - INFORMACIÓN GENERAL
# ============================================================================


@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "MLB Predictor API",
        "docs": "/docs",
        "endpoints": {
            "prediccion_manual": "/predict",
            "prediccion_detallada": "/predict/detailed",
            "partidos_hoy": "/games/today",
            "predicciones_hoy": "/predictions/today",
            "resultados": "/results",
            "comparacion": "/compare/{fecha}",
            "accuracy": "/stats/accuracy",
            "equipos": "/teams",
        },
    }


@app.get("/health")
async def health_check():
    """Verifica el estado de la API y recursos"""
    import os

    from mlb_config import DB_PATH, MODELO_PATH

    modelo_exists = os.path.exists(MODELO_PATH)
    db_exists = os.path.exists(DB_PATH)

    return {
        "status": "healthy" if (modelo_exists and db_exists) else "degraded",
        "modelo_disponible": modelo_exists,
        "base_datos_disponible": db_exists,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/teams", response_model=list[dict[str, str]])
async def listar_equipos():
    """Lista todos los equipos MLB disponibles"""
    equipos = [
        {"codigo": code, "nombre": name} for code, name in TEAM_CODE_TO_NAME.items()
    ]
    return sorted(equipos, key=lambda x: x["nombre"])


@app.get("/status/dates")
async def obtener_estado_fechas():
    """Expone las fechas más recientes disponibles para juegos, predicciones y comparación."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return _obtener_estado_fechas(conn)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error obteniendo estado de fechas: {str(e)}"
        ) from e


# ============================================================================
# ENDPOINTS - PREDICCIONES
# ============================================================================


@app.post("/predict", response_model=PrediccionResponse)
async def crear_prediccion_manual(request: PrediccionRequest):
    """
    Crea una predicción manual para un partido específico
    Incluye scraping de estadísticas en tiempo real
    """
    try:
        # Validar códigos de equipos
        home_code = get_team_code(request.home_team)
        away_code = get_team_code(request.away_team)

        if not home_code:
            raise HTTPException(
                status_code=400, detail=f"Equipo local no válido: {request.home_team}"
            )
        if not away_code:
            raise HTTPException(
                status_code=400,
                detail=f"Equipo visitante no válido: {request.away_team}",
            )
        if home_code == away_code:
            raise HTTPException(
                status_code=400, detail="Los equipos no pueden ser iguales"
            )

        # Realizar predicción
        resultado = predecir_juego(
            home_team=home_code,
            away_team=away_code,
            home_pitcher=request.home_pitcher,
            away_pitcher=request.away_pitcher,
            year=int(request.year or 2026),
            modo_auto=True,
        )

        if not resultado:
            raise HTTPException(
                status_code=500,
                detail="No se pudo completar la predicción. Verifica los nombres de los lanzadores.",
            )

        return PrediccionResponse(
            success=True,
            fecha=resultado["fecha"],
            home_team=resultado["home_team"],
            away_team=resultado["away_team"],
            home_pitcher=resultado["home_pitcher"],
            away_pitcher=resultado["away_pitcher"],
            prob_home=resultado["prob_home"],
            prob_away=resultado["prob_away"],
            prediccion=resultado["prediccion"],
            confianza=resultado["confianza"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}") from e


def _crear_prediccion_detallada_sync(request: PrediccionRequest):
    """
    Crea predicción detallada reutilizando el mismo pipeline de Predicción Manual.
    """
    try:
        # Validar códigos de equipos
        home_code = get_team_code(request.home_team)
        away_code = get_team_code(request.away_team)

        if not home_code:
            raise HTTPException(
                status_code=400, detail=f"Equipo local no válido: {request.home_team}"
            )
        if not away_code:
            raise HTTPException(
                status_code=400,
                detail=f"Equipo visitante no válido: {request.away_team}",
            )
        if home_code == away_code:
            raise HTTPException(
                status_code=400, detail="Los equipos no pueden ser iguales"
            )

        # Buscar en caché primero si tenemos fecha
        import json
        if request.fecha:
            with sqlite3.connect(DB_PATH) as conn:
                try:
                    query = """
                        SELECT prob_home, prob_away, prediccion, confianza, detalles
                        FROM predicciones_historico
                        WHERE fecha = ? AND home_team = ? AND away_team = ?
                        LIMIT 1
                    """
                    row = conn.execute(query, (request.fecha, home_code, away_code)).fetchone()
                    if row and row[4]:  # Si existe el JSON de detalles
                        print(f"✅ Caché HIT para análisis detallado: {away_code} @ {home_code}")
                        prob_home = row[0]
                        prob_away = row[1]
                        prediccion_code = row[2]
                        confianza_label = row[3]
                        
                        # Manejar formato de porcentajes (algunos podrían estar como 60.0 en lugar de 0.6)
                        prob_home_decimal = prob_home / 100.0 if prob_home > 1.0 else prob_home
                        prob_away_decimal = prob_away / 100.0 if prob_away > 1.0 else prob_away

                        detalles_json = row[4]
                        detalles = json.loads(detalles_json)
                        
                        stats_detalladas = detalles.get("stats_detalladas") or {
                            "home_pitcher": {},
                            "away_pitcher": {},
                            "home_batters": [],
                            "away_batters": [],
                        }
                        features_usadas = detalles.get("features_usadas") or {}
                        year_usado = detalles.get("year_usado", request.year)
                        
                        prob_max = max(prob_home_decimal, prob_away_decimal)
                        if prob_max >= 0.65:
                            confianza_decimal = 0.85
                        elif prob_max >= 0.58:
                            confianza_decimal = 0.70
                        else:
                            confianza_decimal = 0.55

                        return {
                            "ganador": prediccion_code,
                            "prob_home": prob_home_decimal,
                            "prob_away": prob_away_decimal,
                            "confianza": confianza_decimal,
                            "year_solicitado": request.year,
                            "year_usado_home": year_usado,
                            "year_usado_away": year_usado,
                            "razon_fallback_home": None,
                            "razon_fallback_away": None,
                            "ip_home": stats_detalladas.get("home_pitcher", {}).get("IP", 0),
                            "ip_away": stats_detalladas.get("away_pitcher", {}).get("IP", 0),
                            "features_usadas": features_usadas,
                            "stats_detalladas": stats_detalladas,
                        }
                except Exception as db_e:
                    print(f"⚠️ Error leyendo caché de BD: {db_e}")

        # Reusar el mismo motor de Predicción Manual para evitar divergencias de lógica.
        print(f"⚠️ Caché MISS (o sin fecha) para {away_code} @ {home_code}. Scrapeando...")
        resultado = predecir_juego(
            home_team=home_code,
            away_team=away_code,
            home_pitcher=request.home_pitcher,
            away_pitcher=request.away_pitcher,
            year=int(request.year or 2026),
            modo_auto=True,
            guardar_db=False,
            hacer_scraping=True,
        )

        if not resultado:
            raise HTTPException(
                status_code=500, detail="No se pudo completar la predicción"
            )

        detalles = resultado.get("detalles") or {}
        stats_detalladas = detalles.get("stats_detalladas") or {
            "home_pitcher": {},
            "away_pitcher": {},
            "home_batters": [],
            "away_batters": [],
        }
        features_usadas = detalles.get("features_usadas") or {}
        year_usado = detalles.get("year_usado", request.year)

        # CORRECCIÓN: Calcular confianza correctamente (es una probabilidad 0-1, no porcentaje)
        prob_home_decimal = (
            resultado["prob_home"] / 100.0
            if resultado["prob_home"] > 1
            else resultado["prob_home"]
        )
        prob_away_decimal = (
            resultado["prob_away"] / 100.0
            if resultado["prob_away"] > 1
            else resultado["prob_away"]
        )
        confianza_decimal = max(prob_home_decimal, prob_away_decimal)

        # Construir respuesta detallada
        return {
            "ganador": resultado["prediccion"],
            "prob_home": prob_home_decimal,
            "prob_away": prob_away_decimal,
            "confianza": confianza_decimal,  # CORREGIDO: valor entre 0-1
            "year_solicitado": request.year,
            "year_usado_home": year_usado,
            "year_usado_away": year_usado,
            "razon_fallback_home": None,
            "razon_fallback_away": None,
            "ip_home": stats_detalladas.get("home_pitcher", {}).get("IP", 0),
            "ip_away": stats_detalladas.get("away_pitcher", {}).get("IP", 0),
            "features_usadas": features_usadas,
            "stats_detalladas": stats_detalladas,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}") from e


@app.post("/predict/detailed")
async def crear_prediccion_detallada(request: PrediccionRequest):
    cache_key = _get_detailed_cache_key(request)
    cached_payload = _get_cached_detailed_prediction(cache_key)
    if cached_payload is not None:
        return cached_payload

    try:
        payload = await asyncio.wait_for(
            asyncio.to_thread(_crear_prediccion_detallada_sync, request),
            timeout=DETAILED_ANALYSIS_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        payload = await asyncio.to_thread(
            _crear_payload_degradado_rapido,
            request,
            "timeout en analisis detallado",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno en analisis detallado: {str(exc)}",
        ) from exc

    _set_cached_detailed_prediction(cache_key, payload)
    return payload


@app.post("/predict/detailed/debug")
async def debug_prediccion_detallada(request: PrediccionRequest):
    """Diagnóstico del análisis detallado con tiempos por subpaso."""
    home_code = get_team_code(request.home_team)
    away_code = get_team_code(request.away_team)

    if not home_code:
        raise HTTPException(
            status_code=400, detail=f"Equipo local no válido: {request.home_team}"
        )
    if not away_code:
        raise HTTPException(
            status_code=400,
            detail=f"Equipo visitante no válido: {request.away_team}",
        )
    if home_code == away_code:
        raise HTTPException(status_code=400, detail="Los equipos no pueden ser iguales")

    started_at = time.perf_counter()
    resultado = await asyncio.to_thread(
        predecir_juego,
        home_team=home_code,
        away_team=away_code,
        home_pitcher=request.home_pitcher,
        away_pitcher=request.away_pitcher,
        year=int(request.year or 2026),
        modo_auto=True,
        guardar_db=False,
        hacer_scraping=True,
        debug=True,
    )
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)

    if not resultado:
        return {
            "success": False,
            "elapsed_ms": elapsed_ms,
            "error": "predecir_juego devolvió None",
            "debug": {},
        }

    debug_data = resultado.get("_debug", {}) if isinstance(resultado, dict) else {}
    detalles = resultado.get("detalles", {}) if isinstance(resultado, dict) else {}
    stats = detalles.get("stats_detalladas", {}) if isinstance(detalles, dict) else {}
    features = detalles.get("features_usadas", {}) if isinstance(detalles, dict) else {}

    home_pitcher_stats = stats.get("home_pitcher", {}) if isinstance(stats, dict) else {}
    away_pitcher_stats = stats.get("away_pitcher", {}) if isinstance(stats, dict) else {}
    home_batters = stats.get("home_batters", []) if isinstance(stats, dict) else []
    away_batters = stats.get("away_batters", []) if isinstance(stats, dict) else []

    failed_stage = None
    for stage in debug_data.get("stages", []):
        if not stage.get("ok", True):
            failed_stage = stage
            break

    return {
        "success": bool(resultado) and not bool(resultado.get("error")),
        "elapsed_ms": elapsed_ms,
        "error": resultado.get("error"),
        "failed_stage": failed_stage,
        "summary": {
            "features_count": len(features) if isinstance(features, dict) else 0,
            "home_pitcher_has_stats": bool(home_pitcher_stats),
            "away_pitcher_has_stats": bool(away_pitcher_stats),
            "home_batters_count": len(home_batters) if isinstance(home_batters, list) else 0,
            "away_batters_count": len(away_batters) if isinstance(away_batters, list) else 0,
            "prediccion": resultado.get("prediccion"),
            "prob_home": resultado.get("prob_home"),
            "prob_away": resultado.get("prob_away"),
        },
        "debug": debug_data,
    }


@app.get("/games/today", response_model=list[PartidoHoy])
async def obtener_partidos_hoy():
    """Obtiene los partidos de la jornada mas reciente en la base de datos"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            fecha_objetivo = conn.execute(
                "SELECT MAX(fecha) FROM historico_partidos"
            ).fetchone()[0]

            if not fecha_objetivo:
                return []

            query = """
                SELECT game_id, fecha, home_team, away_team, home_pitcher, away_pitcher
                FROM historico_partidos
                WHERE fecha = ?
                ORDER BY home_team
            """
            df = pd.read_sql(query, conn, params=[fecha_objetivo])

        if df.empty:
            return []

        return [
            PartidoHoy(
                game_id=row["game_id"],
                fecha=row["fecha"],
                home_team=row["home_team"],
                away_team=row["away_team"],
                home_pitcher=row["home_pitcher"],
                away_pitcher=row["away_pitcher"],
            )
            for _, row in df.iterrows()
        ]

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error obteniendo partidos: {str(e)}"
        ) from e


@app.get("/predictions/today", response_model=list[dict])
async def obtener_predicciones_hoy():
    """Obtiene las predicciones de la jornada mas reciente en la base de datos"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            fecha_objetivo = conn.execute(
                "SELECT MAX(fecha) FROM predicciones_historico"
            ).fetchone()[0]

            if not fecha_objetivo:
                return []

            query = """
                SELECT p.*, hp.game_id
                FROM predicciones_historico p
                LEFT JOIN historico_partidos hp
                    ON p.fecha = hp.fecha
                    AND p.home_team = hp.home_team
                    AND p.away_team = hp.away_team
                WHERE p.fecha = ?
                ORDER BY p.prob_home DESC
            """
            df = pd.read_sql(query, conn, params=[fecha_objetivo])

        if df.empty:
            return []

        return df.to_dict("records")

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error obteniendo predicciones: {str(e)}"
        ) from e


@app.get("/predictions/latest", response_model=list[dict])
async def obtener_ultimas_predicciones(
    limit: int = Query(
        20, ge=1, le=100, description="Número de predicciones a retornar"
    ),
):
    """Obtiene las últimas N predicciones generadas"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            query = f"""
                SELECT * FROM predicciones_historico
                ORDER BY fecha DESC, home_team
                LIMIT {limit}
            """
            df = pd.read_sql(query, conn)

        return df.to_dict("records")

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error obteniendo predicciones: {str(e)}"
        ) from e


# ============================================================================
# ENDPOINTS - RESULTADOS Y COMPARACIÓN
# ============================================================================


@app.get("/results/latest", response_model=list[ResultadoReal])
async def obtener_ultimos_resultados(
    limit: int = Query(20, ge=1, le=100, description="Número de resultados a retornar"),
):
    """Obtiene los últimos resultados reales"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            query = f"""
                SELECT * FROM historico_real
                ORDER BY fecha DESC, home_team
                LIMIT {limit}
            """
            df = pd.read_sql(query, conn)

        if df.empty:
            return []

        def _safe_int(value, default=0):
            return int(value) if pd.notna(value) else default

        return [
            ResultadoReal(
                game_id=row["game_id"],
                fecha=row["fecha"],
                home_team=row["home_team"],
                away_team=row["away_team"],
                score_home=_safe_int(row["score_home"]),
                score_away=_safe_int(row["score_away"]),
                ganador=_safe_int(row["ganador"]),
            )
            for _, row in df.iterrows()
        ]

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error obteniendo resultados: {str(e)}"
        ) from e


@app.get("/compare/{fecha}")
async def comparar_predicciones_resultados(fecha: str):
    """Compara predicciones con resultados reales de una fecha"""
    try:
        # Validar formato de fecha
        try:
            datetime.strptime(fecha, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Formato de fecha inválido. Use YYYY-MM-DD"
            ) from None

        query = """
                SELECT
                    r.game_id,
                    r.fecha,
                    r.home_team,
                    r.away_team,
                    r.home_pitcher,
                    r.away_pitcher,
                    r.score_home,
                    r.score_away,
                    r.ganador as ganador_real,
                    p.prediccion,
                    p.prob_home,
                    p.prob_away,
                    p.confianza,
                    CASE
                        WHEN (r.ganador = 1 AND p.prediccion = r.home_team) OR
                             (r.ganador = 0 AND p.prediccion = r.away_team)
                        THEN 1
                        ELSE 0
                    END as acierto
                FROM historico_real r
                LEFT JOIN predicciones_historico p
                    ON r.fecha = p.fecha
                    AND r.home_team = p.home_team
                    AND r.away_team = p.away_team
                WHERE r.fecha = ?
                ORDER BY r.home_team
            """

        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql(query, conn, params=[fecha])

        # Si no hay resultados para la fecha consultada, intentar backfill automático.
        # Usamos <= por desfase horario entre servidor (UTC) y la fecha esperada por usuario.
        if df.empty:
            fecha_dt = datetime.strptime(fecha, "%Y-%m-%d")
            if fecha_dt.date() <= datetime.now().date():
                try:
                    from mlb_update_real_results import (
                        _formatear_fecha_bref_desde_db,
                        actualizar_resultados_reales_en_fecha,
                    )

                    actualizado = actualizar_resultados_reales_en_fecha(
                        _formatear_fecha_bref_desde_db(fecha), fecha, fecha_dt.year
                    )

                    if actualizado:
                        with sqlite3.connect(DB_PATH) as conn:
                            df = pd.read_sql(query, conn, params=[fecha])
                except Exception:
                    # Si el backfill falla, mantenemos respuesta vacía sin romper el endpoint.
                    pass

        # Si hay resultados reales pero faltan predicciones, intentar completar predicciones
        # usando los datos ya almacenados para esa fecha.
        if not df.empty and "prediccion" in df.columns:
            faltantes = int(df["prediccion"].isna().sum())
            if faltantes > 0:
                try:
                    generadas = _backfill_predicciones_fecha(fecha)
                    if generadas > 0:
                        with sqlite3.connect(DB_PATH) as conn:
                            df = pd.read_sql(query, conn, params=[fecha])
                except Exception:
                    pass

        if df.empty:
            # No hay resultados reales: devolver predicciones solas si existen
            query_pred_only = """
                SELECT
                    hp.game_id,
                    hp.fecha,
                    hp.home_team,
                    hp.away_team,
                    hp.home_pitcher,
                    hp.away_pitcher,
                    NULL as score_home,
                    NULL as score_away,
                    NULL as ganador_real,
                    p.prediccion,
                    p.prob_home,
                    p.prob_away,
                    p.confianza,
                    NULL as acierto
                FROM predicciones_historico p
                LEFT JOIN historico_partidos hp
                    ON p.fecha = hp.fecha
                    AND p.home_team = hp.home_team
                    AND p.away_team = hp.away_team
                WHERE p.fecha = ?
                ORDER BY p.prob_home DESC
            """
            with sqlite3.connect(DB_PATH) as conn:
                df = pd.read_sql(query_pred_only, conn, params=[fecha])

            if df.empty:
                return {
                    "fecha": fecha,
                    "partidos": [],
                    "estadisticas": {"total": 0, "aciertos": 0, "accuracy": 0.0, "solo_predicciones": True},
                }

            df_json = df.astype(object).where(pd.notna(df), None)
            return {
                "fecha": fecha,
                "partidos": df_json.to_dict("records"),
                "estadisticas": {
                    "total": len(df),
                    "aciertos": 0,
                    "accuracy": 0.0,
                    "solo_predicciones": True,
                },
            }

        # Calcular estadísticas
        total = len(df)
        aciertos = df["acierto"].fillna(0).sum() if "acierto" in df.columns else 0
        accuracy = (aciertos / total * 100) if total > 0 else 0.0

        # Evitar NaN en la respuesta JSON (FastAPI falla al serializarlos).
        df_json = df.astype(object).where(pd.notna(df), None)

        por_confianza = {}
        if "confianza" in df_json.columns and "acierto" in df_json.columns:
            df_conf = df_json.dropna(subset=["confianza"]).copy()
            if not df_conf.empty:
                por_confianza = (
                    df_conf.groupby("confianza")["acierto"]
                    .agg(["count", "sum", "mean"])
                    .to_dict("index")
                )

        partidos = df_json.to_dict("records")

        return {
            "fecha": fecha,
            "partidos": partidos,
            "estadisticas": {
                "total": int(total),
                "aciertos": int(aciertos),
                "accuracy": round(accuracy, 2),
                "por_confianza": por_confianza,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error en comparación: {str(e)}"
        ) from e


# ============================================================================
# ENDPOINTS - ESTADÍSTICAS Y ANÁLISIS
# ============================================================================


@app.get("/stats/accuracy")
async def obtener_estadisticas_accuracy(
    dias: int = Query(30, ge=1, le=365, description="Número de días hacia atrás"),
):
    """Obtiene estadísticas de accuracy del modelo"""
    try:
        fecha_limite = (datetime.now() - timedelta(days=dias)).strftime("%Y-%m-%d")

        with sqlite3.connect(DB_PATH) as conn:
            query = f"""
                SELECT
                    r.fecha,
                    r.home_team,
                    r.away_team,
                    r.ganador as ganador_real,
                    p.prediccion,
                    p.confianza,
                    CASE
                        WHEN (r.ganador = 1 AND p.prediccion = r.home_team) OR
                             (r.ganador = 0 AND p.prediccion = r.away_team)
                        THEN 1
                        ELSE 0
                    END as acierto
                FROM historico_real r
                INNER JOIN predicciones_historico p
                    ON r.fecha = p.fecha
                    AND r.home_team = p.home_team
                    AND r.away_team = p.away_team
                WHERE r.fecha >= '{fecha_limite}'
            """
            df = pd.read_sql(query, conn)

        if df.empty:
            return {
                "periodo": f"Últimos {dias} días",
                "total": 0,
                "aciertos": 0,
                "accuracy_general": 0.0,
                "por_confianza": {},
            }

        total = len(df)
        aciertos = df["acierto"].sum()
        accuracy = (aciertos / total * 100) if total > 0 else 0.0

        # Estadísticas por nivel de confianza
        por_confianza = {}
        if "confianza" in df.columns:
            for conf in df["confianza"].unique():
                df_conf = df[df["confianza"] == conf]
                por_confianza[conf] = {
                    "total": len(df_conf),
                    "aciertos": int(df_conf["acierto"].sum()),
                    "accuracy": round(df_conf["acierto"].mean() * 100, 2),
                }

        return {
            "periodo": f"Últimos {dias} días",
            "total": int(total),
            "aciertos": int(aciertos),
            "accuracy_general": round(accuracy, 2),
            "por_confianza": por_confianza,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error calculando accuracy: {str(e)}"
        ) from e


@app.get("/stats/team/{team_code}")
async def obtener_estadisticas_equipo(
    team_code: str,
    ultimos_n: int = Query(20, ge=1, le=100, description="Número de partidos"),
):
    """Obtiene estadísticas de predicciones para un equipo específico"""
    try:
        # Validar equipo
        if not get_team_name(team_code):
            raise HTTPException(
                status_code=400, detail=f"Código de equipo no válido: {team_code}"
            )

        with sqlite3.connect(DB_PATH) as conn:
            query = f"""
                SELECT * FROM predicciones_historico
                WHERE home_team = '{team_code}' OR away_team = '{team_code}'
                ORDER BY fecha DESC
                LIMIT {ultimos_n}
            """
            df = pd.read_sql(query, conn)

        if df.empty:
            return {
                "equipo": team_code,
                "nombre": get_team_name(team_code),
                "predicciones": [],
            }

        return {
            "equipo": team_code,
            "nombre": get_team_name(team_code),
            "total_predicciones": len(df),
            "predicciones": df.to_dict("records"),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error obteniendo stats del equipo: {str(e)}"
        ) from e


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
