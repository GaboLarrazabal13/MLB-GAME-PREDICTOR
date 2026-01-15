"""
API FastAPI para MLB Predictor V3.5
Endpoints para predicciones manuales, automáticas y análisis de rendimiento
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import sqlite3
import pandas as pd
import os
import sys

# Importar módulos del proyecto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_config import DB_PATH, get_team_code, get_team_name, TEAM_CODE_TO_NAME
from mlb_predict_engine import predecir_juego
from mlb_utils import analizar_accuracy_historico, generar_reporte_equipos

# ============================================================================
# CONFIGURACIÓN DE LA API
# ============================================================================

app = FastAPI(
    title="MLB Predictor API",
    description="API para predicciones de partidos MLB usando Machine Learning",
    version="3.5.0"
)

# CORS para permitir acceso desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica tus dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODELOS DE DATOS (Pydantic)
# ============================================================================

class PrediccionRequest(BaseModel):
    """Modelo para solicitud de predicción manual"""
    home_team: str = Field(..., description="Código del equipo local (ej: NYY)")
    away_team: str = Field(..., description="Código del equipo visitante (ej: BOS)")
    home_pitcher: str = Field(..., description="Nombre del lanzador local")
    away_pitcher: str = Field(..., description="Nombre del lanzador visitante")
    year: Optional[int] = Field(2026, description="Año para scraping de stats")
    
    class Config:
        schema_extra = {
            "example": {
                "home_team": "NYY",
                "away_team": "BOS",
                "home_pitcher": "Gerrit Cole",
                "away_pitcher": "Tanner Houck",
                "year": 2024
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
    detalles: Optional[Dict] = None


class PartidoHoy(BaseModel):
    """Modelo para partido del día"""
    game_id: str
    fecha: str
    home_team: str
    away_team: str
    home_pitcher: str
    away_pitcher: str
    prediccion: Optional[str] = None
    prob_home: Optional[float] = None
    prob_away: Optional[float] = None
    confianza: Optional[str] = None


class ResultadoReal(BaseModel):
    """Modelo para resultado real"""
    game_id: str
    fecha: str
    home_team: str
    away_team: str
    score_home: int
    score_away: int
    ganador: int
    prediccion: Optional[str] = None
    acierto: Optional[bool] = None


# ============================================================================
# ENDPOINTS - INFORMACIÓN GENERAL
# ============================================================================

@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "MLB Predictor API V3.5",
        "docs": "/docs",
        "endpoints": {
            "prediccion_manual": "/predict",
            "partidos_hoy": "/games/today",
            "predicciones_hoy": "/predictions/today",
            "resultados": "/results",
            "comparacion": "/compare/{fecha}",
            "accuracy": "/stats/accuracy",
            "equipos": "/teams"
        }
    }


@app.get("/health")
async def health_check():
    """Verifica el estado de la API y recursos"""
    import os
    from mlb_config import MODELO_PATH, DB_PATH
    
    modelo_exists = os.path.exists(MODELO_PATH)
    db_exists = os.path.exists(DB_PATH)
    
    return {
        "status": "healthy" if (modelo_exists and db_exists) else "degraded",
        "modelo_disponible": modelo_exists,
        "base_datos_disponible": db_exists,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/teams", response_model=List[Dict[str, str]])
async def listar_equipos():
    """Lista todos los equipos MLB disponibles"""
    equipos = [
        {"codigo": code, "nombre": name}
        for code, name in TEAM_CODE_TO_NAME.items()
    ]
    return sorted(equipos, key=lambda x: x['nombre'])


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
            raise HTTPException(status_code=400, detail=f"Equipo local no válido: {request.home_team}")
        if not away_code:
            raise HTTPException(status_code=400, detail=f"Equipo visitante no válido: {request.away_team}")
        if home_code == away_code:
            raise HTTPException(status_code=400, detail="Los equipos no pueden ser iguales")
        
        # Realizar predicción
        resultado = predecir_juego(
            home_team=home_code,
            away_team=away_code,
            home_pitcher=request.home_pitcher,
            away_pitcher=request.away_pitcher,
            year=request.year,
            modo_auto=True  # Modo silencioso para API
        )
        
        if not resultado:
            raise HTTPException(
                status_code=500, 
                detail="No se pudo completar la predicción. Verifica los nombres de los lanzadores."
            )
        
        return PrediccionResponse(
            success=True,
            fecha=resultado['fecha'],
            home_team=resultado['home_team'],
            away_team=resultado['away_team'],
            home_pitcher=resultado['home_pitcher'],
            away_pitcher=resultado['away_pitcher'],
            prob_home=resultado['prob_home'],
            prob_away=resultado['prob_away'],
            prediccion=resultado['prediccion'],
            confianza=resultado['confianza']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@app.get("/games/today", response_model=List[PartidoHoy])
async def obtener_partidos_hoy():
    """Obtiene los partidos programados para hoy"""
    try:
        fecha_hoy = datetime.now().strftime("%Y-%m-%d")
        
        with sqlite3.connect(DB_PATH) as conn:
            query = f"""
                SELECT game_id, fecha, home_team, away_team, home_pitcher, away_pitcher
                FROM historico_partidos
                WHERE fecha = '{fecha_hoy}'
                ORDER BY home_team
            """
            df = pd.read_sql(query, conn)
        
        if df.empty:
            return []
        
        return [
            PartidoHoy(
                game_id=row['game_id'],
                fecha=row['fecha'],
                home_team=row['home_team'],
                away_team=row['away_team'],
                home_pitcher=row['home_pitcher'],
                away_pitcher=row['away_pitcher']
            )
            for _, row in df.iterrows()
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo partidos: {str(e)}")


@app.get("/predictions/today", response_model=List[Dict])
async def obtener_predicciones_hoy():
    """Obtiene las predicciones generadas para hoy"""
    try:
        fecha_hoy = datetime.now().strftime("%Y-%m-%d")
        
        with sqlite3.connect(DB_PATH) as conn:
            query = f"""
                SELECT p.*, hp.game_id
                FROM predicciones_historico p
                LEFT JOIN historico_partidos hp 
                    ON p.fecha = hp.fecha 
                    AND p.home_team = hp.home_team 
                    AND p.away_team = hp.away_team
                WHERE p.fecha = '{fecha_hoy}'
                ORDER BY p.prob_home DESC
            """
            df = pd.read_sql(query, conn)
        
        if df.empty:
            return []
        
        return df.to_dict('records')
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo predicciones: {str(e)}")


@app.get("/predictions/latest", response_model=List[Dict])
async def obtener_ultimas_predicciones(
    limit: int = Query(20, ge=1, le=100, description="Número de predicciones a retornar")
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
        
        return df.to_dict('records')
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo predicciones: {str(e)}")


# ============================================================================
# ENDPOINTS - RESULTADOS Y COMPARACIÓN
# ============================================================================

@app.get("/results/latest", response_model=List[ResultadoReal])
async def obtener_ultimos_resultados(
    limit: int = Query(20, ge=1, le=100, description="Número de resultados a retornar")
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
        
        return [
            ResultadoReal(
                game_id=row['game_id'],
                fecha=row['fecha'],
                home_team=row['home_team'],
                away_team=row['away_team'],
                score_home=int(row['score_home']),
                score_away=int(row['score_away']),
                ganador=int(row['ganador'])
            )
            for _, row in df.iterrows()
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo resultados: {str(e)}")


@app.get("/compare/{fecha}")
async def comparar_predicciones_resultados(fecha: str):
    """
    Compara las predicciones con los resultados reales de una fecha específica
    Formato de fecha: YYYY-MM-DD
    """
    try:
        # Validar formato de fecha
        try:
            datetime.strptime(fecha, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Formato de fecha inválido. Use YYYY-MM-DD")
        
        with sqlite3.connect(DB_PATH) as conn:
            query = f"""
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
                WHERE r.fecha = '{fecha}'
                ORDER BY r.home_team
            """
            df = pd.read_sql(query, conn)
        
        if df.empty:
            return {
                "fecha": fecha,
                "partidos": [],
                "estadisticas": {
                    "total": 0,
                    "aciertos": 0,
                    "accuracy": 0.0
                }
            }
        
        # Calcular estadísticas
        total = len(df)
        aciertos = df['acierto'].sum() if 'acierto' in df.columns else 0
        accuracy = (aciertos / total * 100) if total > 0 else 0.0
        
        partidos = df.to_dict('records')
        
        return {
            "fecha": fecha,
            "partidos": partidos,
            "estadisticas": {
                "total": int(total),
                "aciertos": int(aciertos),
                "accuracy": round(accuracy, 2),
                "por_confianza": df.groupby('confianza')['acierto'].agg(['count', 'sum', 'mean']).to_dict('index') if 'confianza' in df.columns else {}
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en comparación: {str(e)}")


# ============================================================================
# ENDPOINTS - ESTADÍSTICAS Y ANÁLISIS
# ============================================================================

@app.get("/stats/accuracy")
async def obtener_estadisticas_accuracy(
    dias: int = Query(30, ge=1, le=365, description="Número de días hacia atrás")
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
                "por_confianza": {}
            }
        
        total = len(df)
        aciertos = df['acierto'].sum()
        accuracy = (aciertos / total * 100) if total > 0 else 0.0
        
        # Estadísticas por nivel de confianza
        por_confianza = {}
        if 'confianza' in df.columns:
            for conf in df['confianza'].unique():
                df_conf = df[df['confianza'] == conf]
                por_confianza[conf] = {
                    "total": len(df_conf),
                    "aciertos": int(df_conf['acierto'].sum()),
                    "accuracy": round(df_conf['acierto'].mean() * 100, 2)
                }
        
        return {
            "periodo": f"Últimos {dias} días",
            "total": int(total),
            "aciertos": int(aciertos),
            "accuracy_general": round(accuracy, 2),
            "por_confianza": por_confianza
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculando accuracy: {str(e)}")


@app.get("/stats/team/{team_code}")
async def obtener_estadisticas_equipo(
    team_code: str,
    ultimos_n: int = Query(20, ge=1, le=100, description="Número de partidos")
):
    """Obtiene estadísticas de predicciones para un equipo específico"""
    try:
        # Validar equipo
        if not get_team_name(team_code):
            raise HTTPException(status_code=400, detail=f"Código de equipo no válido: {team_code}")
        
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
                "predicciones": []
            }
        
        return {
            "equipo": team_code,
            "nombre": get_team_name(team_code),
            "total_predicciones": len(df),
            "predicciones": df.to_dict('records')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo stats del equipo: {str(e)}")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)