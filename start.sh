#!/bin/bash
# =============================================================================
# MLB Predictor - Hugging Face / Docker Space Startup Script
# Inicia la API de FastAPI en segundo plano y el Dashboard Streamlit en primer plano
# =============================================================================

# Asegurar que existe el directorio de logs
mkdir -p /app/logs

# Iniciar la API en segundo plano
echo "⚾ [1/3] Iniciando la API de FastAPI en localhost:8000..."
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 > /app/logs/api.log 2>&1 &
API_PID=$!

# Esperar a que la API esté respondiendo
echo "⚾ [2/3] Esperando a que la API de predicción esté lista..."
for i in $(seq 1 30); do
    if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" > /dev/null 2>&1; then
        echo "✅ API de predicción levantada con éxito en PID: $API_PID (${i}s)"
        break
    fi
    sleep 1
done

# Iniciar Streamlit expuesto en el puerto requerido por Hugging Face (por defecto 7860)
PORT=${PORT:-7860}
echo "⚾ [3/3] Iniciando Dashboard Streamlit en el puerto $PORT..."
exec python -m streamlit run src/app.py --server.port=$PORT --server.address=0.0.0.0
