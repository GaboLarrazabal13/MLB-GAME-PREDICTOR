# Configuración de APIs - MLB Game Predictor

## 🌐 Puertos Asignados

### API Original (Modelo Básico)
- **Puerto**: 8000
- **Archivo**: `api.py`
- **URL**: `http://localhost:8000`
- **Iniciar**:
  ```bash
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
  ```

### API Híbrida (Modelo Optimizado)
- **Puerto**: 8001
- **Archivo**: `api_hybrid.py`
- **URL**: `http://localhost:8001`
- **Iniciar**:
  ```bash
  uvicorn api_hybrid:app --host 0.0.0.0 --port 8001 --reload
  ```

## 🚀 Iniciar Ambas APIs Simultáneamente

### Opción 1: Dos Terminales

**Terminal 1 - API Original:**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - API Híbrida:**
```bash
uvicorn api_hybrid:app --host 0.0.0.0 --port 8001 --reload
```

### Opción 2: Script de Inicio (crear `start_apis.sh`)

```bash
#!/bin/bash

# Iniciar API original en background
echo "🚀 Iniciando API Original en puerto 8000..."
uvicorn api:app --host 0.0.0.0 --port 8000 --reload &
PID1=$!

# Esperar 2 segundos
sleep 2

# Iniciar API híbrida en background
echo "🚀 Iniciando API Híbrida en puerto 8001..."
uvicorn api_hybrid:app --host 0.0.0.0 --port 8001 --reload &
PID2=$!

echo ""
echo "✅ APIs iniciadas:"
echo "   API Original: http://localhost:8000"
echo "   API Híbrida: http://localhost:8001"
echo ""
echo "Para detener ambas APIs, presiona Ctrl+C"

# Esperar a que terminen
wait $PID1 $PID2
```

## 🌐 Web Apps

### Web App Original
- **Puerto**: 8501 (default Streamlit)
- **Archivo**: `web_app.py`
- **API URL**: `http://localhost:8000`
- **Iniciar**:
  ```bash
  streamlit run web_app.py
  ```

### Web App Híbrida
- **Puerto**: 8502
- **Archivo**: `web_app_hybrid.py`
- **API URL**: `http://localhost:8001`
- **Iniciar**:
  ```bash
  streamlit run web_app_hybrid.py --server.port 8502
  ```

## 🔄 Iniciar Sistema Completo

### Sistema Original Completo
```bash
# Terminal 1
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2
streamlit run web_app.py
```

**Acceder**: `http://localhost:8501`

### Sistema Híbrido Completo
```bash
# Terminal 1
uvicorn api_hybrid:app --host 0.0.0.0 --port 8001 --reload

# Terminal 2
streamlit run web_app_hybrid.py --server.port 8502
```

**Acceder**: `http://localhost:8502`

### Ambos Sistemas Simultáneamente
```bash
# Terminal 1 - API Original
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 - API Híbrida
uvicorn api_hybrid:app --host 0.0.0.0 --port 8001 --reload

# Terminal 3 - Web App Original
streamlit run web_app.py

# Terminal 4 - Web App Híbrida
streamlit run web_app_hybrid.py --server.port 8502
```

## 📝 Actualizar URL en Web Apps

Si despliegas en producción, actualiza la URL de la API en cada web app:

**En `web_app.py`:**
```python
API_URL = st.secrets.get("API_URL", "http://localhost:8000")
```

**En `web_app_hybrid.py`:**
```python
API_URL = st.secrets.get("API_URL", "http://localhost:8001")
```

## 🌍 Deployment en Streamlit Cloud

### Para Web App Original:
1. En Streamlit Cloud, ve a Settings → Secrets
2. Añade:
   ```toml
   API_URL = "https://tu-api-original.herokuapp.com"
   ```

### Para Web App Híbrida:
1. En Streamlit Cloud, ve a Settings → Secrets
2. Añade:
   ```toml
   API_URL = "https://tu-api-hibrida.herokuapp.com"
   ```

## 🐛 Troubleshooting

### Error: "Address already in use"
Si el puerto está ocupado:
```bash
# En Linux/Mac
lsof -ti:8000 | xargs kill -9
lsof -ti:8001 | xargs kill -9

# En Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### APIs no se conectan
Verifica que las URLs coincidan:
- Web App Original debe apuntar a puerto 8000
- Web App Híbrida debe apuntar a puerto 8001