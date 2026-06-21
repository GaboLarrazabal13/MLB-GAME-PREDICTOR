# =============================================================================
# MLB Predictor V3.5 - Multi-stage Dockerfile
# Optimizado para producción con Python 3.12
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Instalar dependencias y compilar
# -----------------------------------------------------------------------------
FROM python:3.12-slim as builder

WORKDIR /app

# Instalar dependencias del sistema para compilaciones
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Crear entorno virtual
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copiar solo requirements para cache de dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Production - Imagen final optimizada
# -----------------------------------------------------------------------------
FROM python:3.12-slim as production

# Crear usuario no-root por seguridad
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Instalar libgomp1 requerida por LightGBM (no incluida en python:slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar entorno virtual del builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copiar código de la aplicación
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup models/ ./models/
COPY --chown=appuser:appgroup data/ ./data/
COPY --chown=appuser:appgroup cache/ ./cache/
COPY --chown=appuser:appgroup .streamlit/ .streamlit/

# Copiar archivos de configuración
COPY --chown=appuser:appgroup requirements.txt .
COPY --chown=appuser:appgroup .streamlit/config.toml .
COPY --chown=appuser:appgroup start.sh .

# Crear directorios necesarios y dar permisos
RUN mkdir -p /app/logs && \
    chmod +x /app/start.sh && \
    chown -R appuser:appgroup /app/logs /app/start.sh

# Cambiar a usuario no-root
USER appuser

# -----------------------------------------------------------------------------
# Configuración de puerto y health check
# -----------------------------------------------------------------------------
EXPOSE 8000 7860

# Health check para Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# -----------------------------------------------------------------------------
# Comandos de inicio
# -----------------------------------------------------------------------------
# Ejecuta el script de inicio que levanta la API y el frontend Streamlit
CMD ["/bin/bash", "/app/start.sh"]
