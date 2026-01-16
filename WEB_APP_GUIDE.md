# 🌐 MLB Predictor Pro V3.5 - Guía de la Web App

Web application profesional para predicciones MLB en tiempo real con análisis avanzado.

## 📋 Tabla de Contenidos

1. [Instalación](#instalación)
2. [Configuración](#configuración)
3. [Ejecución](#ejecución)
4. [Funcionalidades](#funcionalidades)
5. [Despliegue](#despliegue)

---

## 🚀 Instalación

### Prerequisitos

```bash
# Python 3.10+
python --version

# Instalar dependencias
pip install -r requirements.txt
```

### Estructura de Archivos

```
mlb-game-predictor/
├── src/
│   ├── app.py                    # 🌐 Web App Principal
│   ├── api.py                    # 🔌 API FastAPI
│   ├── mlb_config.py             # ⚙️ Configuración
│   └── ...                       # Otros módulos
├── .streamlit/
│   ├── config.toml               # Configuración Streamlit
│   └── secrets.toml              # Secrets (crear desde .example)
├── data/
│   └── mlb_reentrenamiento.db    # Base de datos
├── models/
│   └── modelo_mlb_v3.5.json      # Modelo entrenado
└── requirements.txt
```

---

## ⚙️ Configuración

### 1. Crear archivo de secrets

```bash
cd .streamlit
cp secrets.toml.example secrets.toml
```

### 2. Editar `.streamlit/secrets.toml`

```toml
# Para desarrollo local
API_URL = "http://localhost:8000"

# Para producción
# API_URL = "https://tu-api-produccion.com"
```

### 3. Verificar configuración

```bash
python -c "import streamlit as st; print('✅ Streamlit instalado correctamente')"
```

---

## 🎮 Ejecución

### Modo Desarrollo (Local)

#### Paso 1: Iniciar la API

```bash
cd src
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

#### Paso 2: Iniciar la Web App (en otra terminal)

```bash
cd src
streamlit run app.py
```

La app estará disponible en: **http://localhost:8501**

### Modo Producción

```bash
# API
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4

# Web App
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

---

## 🎯 Funcionalidades

### 1. 🎯 Predicción Manual

**Descripción**: Crea predicciones personalizadas para cualquier partido.

**Características**:
- ✅ Selección visual de equipos con logos
- ✅ Input de lanzadores abridores
- ✅ Scraping en tiempo real de Baseball-Reference
- ✅ Análisis de Super Features
- ✅ Gráficos interactivos de probabilidades
- ✅ Descarga de reporte técnico en JSON

**Cómo usar**:
1. Selecciona equipo local y visitante
2. Ingresa nombres de lanzadores (ej: "Gerrit Cole")
3. Selecciona temporada (2020-2026)
4. Click en "🚀 Realizar Predicción"
5. Espera 30-120 segundos (scraping en vivo)
6. Analiza resultados detallados

**Tips**:
- Usa nombres completos: "Sandy Alcantara" ✅ no "S. Alcantara" ❌
- Verifica que el lanzador haya jugado en la temporada seleccionada
- El scraping puede tardar si Baseball-Reference está lento

---

### 2. 📅 Partidos de Hoy

**Descripción**: Visualiza predicciones automáticas para los partidos del día.

**Características**:
- ✅ Lista de partidos scrapeados automáticamente
- ✅ Predicciones pre-calculadas por GitHub Actions
- ✅ Botón de scraping manual si no hay datos
- ✅ Nivel de confianza por partido
- ✅ Acceso rápido a análisis detallado

**Flujo automático**:
```
10:00 AM → GitHub Actions scrapea partidos del día
10:05 AM → API genera predicciones automáticas
         → Disponibles en esta sección
```

**Si no hay partidos**:
1. Click en "🔄 Buscar Partidos Manualmente"
2. El sistema ejecutará `mlb_daily_scraper.py`
3. Si hay juegos, se mostrarán automáticamente
4. Si no, verás mensaje amigable

**Estados posibles**:
- ✅ **Partidos encontrados**: Muestra cards con predicciones
- ⏳ **Predicción pendiente**: Partido scrapeado pero sin predicción
- 📭 **Sin partidos**: Día libre de MLB o aún no publicados

---

### 3. 📊 Comparación & Historial

**Descripción**: Analiza el rendimiento histórico del modelo.

**Características**:
- ✅ Selector de fechas con calendario
- ✅ Comparación predicción vs resultado real
- ✅ Accuracy por fecha
- ✅ Tabla detallada de aciertos/errores
- ✅ Estadísticas agregadas (30 días)

**Cómo usar**:
1. Selecciona una fecha (generalmente día anterior)
2. Click en "🔍 Analizar Fecha"
3. Revisa tabla de comparación
4. Verifica aciertos (✅) y errores (❌)

**Métricas mostradas**:
- **Total Partidos**: Juegos de ese día
- **Aciertos**: Predicciones correctas
- **Accuracy**: Porcentaje de aciertos
- **Errores**: Predicciones incorrectas

**Detalle por partido**:
```
✅ NYY @ BOS - 5-3
   Predicción: NYY (65.2%)
   Real: NYY ganó
   Confianza: ALTA
   Resultado: ✅ ACIERTO
```

---

### 4. ℹ️ Acerca del Modelo

**Descripción**: Información técnica del sistema.

**Contenido**:
- 🎯 Descripción general
- 🚀 Características V3.5
- 📊 Fuentes de datos
- 🎯 Explicación de Super Features
- 🛠️ Stack tecnológico
- ⚾ Lista de equipos MLB

---

## 🎨 Características Visuales

### Diseño Responsivo
- ✅ Optimizado para desktop y tablet
- ✅ Sidebar colapsable
- ✅ Cards con hover effects
- ✅ Gradientes modernos

### Temas de Color
- **Primary Blue**: #3b82f6 (Predicciones, botones)
- **Success Green**: #10b981 (Aciertos)
- **Danger Red**: #ef4444 (Errores)
- **Warning Yellow**: #f59e0b (Moderada confianza)

### Logos Oficiales MLB
- Cargados desde mlbstatic.com
- Filtro drop-shadow para profundidad
- Tamaños adaptativos según contexto

---

## 🚀 Despliegue

### Streamlit Cloud (Recomendado)

1. **Fork el repositorio**
2. **Conecta a Streamlit Cloud**: https://streamlit.io/cloud
3. **Configurar**:
   - Main file: `src/app.py`
   - Python version: 3.10
4. **Agregar Secrets**:
   ```toml
   API_URL = "https://tu-api.onrender.com"
   ```
5. **Deploy** ✅

### Heroku

```bash
# Crear Procfile
echo "web: cd src && streamlit run app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Deploy
heroku create mlb-predictor-app
git push heroku main
```

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t mlb-predictor-web .
docker run -p 8501:8501 mlb-predictor-web
```

---

## 🐛 Solución de Problemas

### Error: "API No Disponible"

**Causa**: La API no está corriendo o URL incorrecta

**Solución**:
```bash
# Verifica que la API esté corriendo
curl http://localhost:8000/health

# Si no responde, inicia la API
cd src
uvicorn api:app --reload
```

### Error: "Predicción tardó demasiado"

**Causa**: Timeout de scraping (>2 minutos)

**Solución**:
- Verifica conexión a Baseball-Reference
- Intenta con otro lanzador
- Revisa logs del servidor API

### No se muestran partidos del día

**Causa**: GitHub Actions no ejecutó o falló

**Solución**:
1. Click en "🔄 Buscar Partidos Manualmente"
2. Espera 30-60 segundos
3. Si dice "no hay partidos", verifica la fecha en Baseball-Reference

### Logos no cargan

**Causa**: mlbstatic.com inaccesible

**Solución**:
- Verifica conexión a internet
- Los logos son opcionales, la funcionalidad sigue operando

---

## 📊 Rendimiento

### Tiempos Esperados

| Operación | Tiempo | Descripción |
|-----------|--------|-------------|
| Carga inicial | 2-3s | Primera carga de la app |
| Predicción manual | 30-120s | Incluye scraping en vivo |
| Partidos del día | <1s | Datos pre-cargados |
| Comparación | 2-5s | Query a base de datos |

### Optimizaciones

```python
# Cache de datos
@st.cache_data(ttl=300)  # 5 minutos
def obtener_partidos_hoy():
    ...

# Cache de configuración
@st.cache_resource
def cargar_modelo():
    ...
```

---

## 🔐 Seguridad

### Secrets Management

❌ **NUNCA** commits:
- `.streamlit/secrets.toml`
- Credenciales de base de datos
- API keys

✅ **SÍ** commits:
- `.streamlit/secrets.toml.example`
- Configuración pública

### Variables de Entorno

```bash
# Desarrollo
export API_URL="http://localhost:8000"

# Producción
export API_URL="https://api-produccion.com"
```

---

## 📈 Mejoras Futuras

### Roadmap

- [ ] 🔔 Notificaciones push de predicciones
- [ ] 📱 Versión mobile optimizada
- [ ] 🎮 Modo oscuro
- [ ] 📊 Dashboard de analytics avanzado
- [ ] 🤖 Chatbot con IA para análisis
- [ ] 🔄 WebSockets para updates en vivo
- [ ] 📥 Export a PDF de reportes
- [ ] 🏆 Leaderboard de equipos

---

## 📞 Soporte

- **Issues**: GitHub Issues
- **Documentación**: Este archivo
- **API Docs**: http://localhost:8000/docs

---

## 📄 Licencia

MIT License - Ver LICENSE file

---

**Última actualización**: Enero 2026  
**Versión**: 3.5 Professional  
**Autor**: [Tu Nombre]