---

# Guía de Usuario: MLB Game Predictor V3 - Web App

Esta guía detalla la instalación, configuración y uso de la interfaz gráfica desarrollada en Streamlit para el Modelo Híbrido V3.

---

## 1. INSTALACIÓN

### Paso 1: Dependencias de la V3

Asegúrate de tener las bibliotecas necesarias para el renderizado de las Super Features y la conexión con el modelo XGBoost:

```bash
pip install streamlit plotly pandas requests

```

### Paso 2: Configuración del Tema (Design V3)

Para que la aplicación mantenga la estética profesional de la V3, crea el archivo de configuración:

**Ruta:** `.streamlit/config.toml`

```toml
[theme]
base = "light"
primaryColor = "#1f77b4"
backgroundColor = "#f8f9fa"
secondaryBackgroundColor = "#ffffff"
textColor = "#1e293b"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false

```

### Paso 3: Estructura de Archivos V3

```
mlb-game-predictor-v3/
│
├── .streamlit/
│   └── config.toml
│
├── src/
│   ├── web_app_v3.py         <- Interfaz principal
│   ├── api_hybrid_v3.py      <- Backend de predicción
│   └── predict_game_v3.py    <- Lógica híbrida
│
├── models/
│   └── modelo_mlb_v3.json
│
└── requirements.txt

```

---

## 2. CÓMO USAR LA APLICACIÓN

### Ejecución Local Híbrida

Para que la Web App funcione, el motor de la API V3 debe estar activo simultáneamente.

**Terminal 1 (Backend API):**

```bash
uvicorn src.api_hybrid_v3:app --host 0.0.0.0 --port 8000

```

**Terminal 2 (Frontend Streamlit):**

```bash
streamlit run src/web_app_v3.py

```

Acceso automático en: `http://localhost:8501`

---

## 3. CARACTERÍSTICAS DE LA INTERFAZ V3

### Panel Predictor Inteligente

* Selectores dinámicos: Carga automática de los 30 equipos de la MLB.
* Buscador de lanzadores: Autocompletado basado en la caché de la V3.
* Botón de predicción: Dispara el scraping híbrido y el cálculo de matchups.

### Visualización de Super Features

La V3 incluye tres tarjetas exclusivas de análisis:

* Neutralización (Matchup de contacto).
* Resistencia (Efectividad vs Poder).
* Muro Bullpen (Solidez del cierre).

### Estadísticas Detalladas

* Comparativa de Pitchers: Tarjetas con ERA, WHIP y SO9.
* Lineup Analysis: Visualización del Top 3 de bateadores (AVG, OPS, HR) por cada equipo.

---

## 4. DESPLIEGUE EN PRODUCCIÓN

### Opción A: Streamlit Community Cloud (Recomendado)

1. Sube el código a GitHub (incluyendo la carpeta models/ y cache/).
2. En Streamlit Cloud, selecciona el repositorio.
3. Configura el "Main file path" como: `src/web_app_v3.py`.
4. Añade en "Advanced Settings" la variable de entorno:
`API_URL = "https://tu-api-v3-en-render.com"`

### Opción B: Docker (Contenedor Unificado)

Puedes crear un archivo `Dockerfile` para correr ambos servicios:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["sh", "-c", "uvicorn src.api_hybrid_v3:app --port 8000 & streamlit run src/web_app_v3.py --server.port 8501"]

```

---

## 5. SOLUCIÓN DE PROBLEMAS (TROUBLESHOOTING)

### El valor de Resistencia aparece como 555.5%

Causa: Es el valor crudo de la API V3.
Solución: En `web_app_v3.py`, busca la variable de visualización y aplica `abs(valor) * 10` para normalizar la escala a porcentaje estándar.

### Los logos de los equipos no cargan

Causa: Cambio en las URLs de la API de MLB o falta de conexión.
Solución: Verifica que el diccionario `EQUIPOS_MLB` en tu código utilice las rutas de `mlbstatic.com`.

### Error: Connection Refused (Port 8000)

Causa: La Web App no encuentra la API activa.
Solución: Asegúrate de que `api_hybrid_v3.py` esté corriendo antes de intentar una predicción en la web.

---

## 6. PRÓXIMOS PASOS PARA LA V4

* Implementación de modo oscuro automático.
* Gráficos de probabilidad histórica por entrada.
* Filtro de predicción por estadio (efecto de altitud y dimensiones).
* Integración de clima en tiempo real.

---

**MLB Predictor V3 - Documentación de Interfaz**
Desarrollado para análisis avanzado de datos de béisbol.