# 🌐 MLB Game Predictor - Web App con Streamlit

## 📦 Instalación

### Paso 1: Instalar dependencias adicionales

```bash
pip install streamlit plotly
```

O actualizar `requirements.txt`:

```txt
# Añadir estas líneas al final
streamlit==1.28.0
plotly==5.17.0
```

Luego:

```bash
pip install -r requirements.txt
```

### Paso 2: Crear estructura de carpetas

```bash
mkdir .streamlit
```

### Paso 3: Guardar archivos

**Estructura final:**

```
mlb-game-predictor/
│
├── .streamlit/
│   └── config.toml          ← Configuración de Streamlit
│
├── web_app.py               ← App principal
├── client.py                ← Cliente de terminal
├── api.py                   ← API REST
├── models/
│   └── ...
└── requirements.txt
```

---

## 🚀 CÓMO USAR

### Opción 1: Ejecución Local Completa

**Terminal 1 - API:**
```bash
cd mlb-game-predictor
uvicorn api:app --reload
```

**Terminal 2 - Web App:**
```bash
streamlit run web_app.py
```

Tu navegador se abrirá automáticamente en:
```
http://localhost:8501
```

---

### Opción 2: Solo Web App (sin API local)

Si tu API está en la nube:

```bash
# Editar web_app.py, línea 32:
API_URL = "https://tu-api.onrender.com"

# Ejecutar
streamlit run web_app.py
```

---

## 📱 CARACTERÍSTICAS DE LA WEB APP

### 1️⃣ Página Principal - Predictor

- ✅ Selectores dropdown para equipos
- ✅ Inputs para lanzadores
- ✅ Validación de datos
- ✅ Gráficos interactivos con Plotly
- ✅ Gauge de confianza
- ✅ Descarga de resultados en JSON

### 2️⃣ Historial

- ✅ Tabla de todas las predicciones
- ✅ Estadísticas agregadas
- ✅ Limpiar historial

### 3️⃣ Acerca de

- ✅ Información del proyecto
- ✅ Instrucciones de uso
- ✅ Detalles técnicos

---

## 🎨 PERSONALIZACIÓN

### Cambiar colores (en web_app.py):

```python
# Línea ~30, en theme config
primaryColor = "#FF0000"  # Rojo
backgroundColor = "#000000"  # Negro
```

### Cambiar puerto:

```bash
streamlit run web_app.py --server.port 8502
```

### Modo oscuro:

En `.streamlit/config.toml`:

```toml
[theme]
base = "dark"
primaryColor = "#ff4b4b"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
textColor = "#fafafa"
```

---

## 🌐 DESPLIEGUE EN LA NUBE

### Opción 1: Streamlit Cloud (GRATIS)

1. Sube tu código a GitHub
2. Ve a https://share.streamlit.io/
3. Conecta tu repo
4. ¡Deploy automático!

**Configuración para Streamlit Cloud:**

Crear archivo `secrets.toml` (no subir a Git):

```toml
API_URL = "https://tu-api.onrender.com"
```

### Opción 2: Heroku

```bash
# Crear Procfile
echo "web: streamlit run web_app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Deploy
heroku create mlb-predictor-web
git push heroku main
```

---

## 🎯 EJEMPLOS DE USO

### Usar la app:

1. **Selecciona equipos** de los dropdowns
2. **Ingresa lanzadores**: "Bello", "Cole", etc.
3. **Click en "Realizar Predicción"**
4. **Espera 10-30 segundos** (scraping en tiempo real)
5. **¡Ve el resultado!**

### Capturas de pantalla:

```
🏟️ MLB Game Predictor
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 Datos del Partido
┌─────────────────────┬─────────────────────┐
│  🏠 Equipo Local    │  ✈️ Equipo Visitante │
│  🔵 Boston Red Sox  │  🔵 New York Yankees│
│  Lanzador: Bello    │  Lanzador: Cole     │
└─────────────────────┴─────────────────────┘

             🔮 Realizar Predicción

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 Resultado de la Predicción

┌────────────────────────────────────────────────┐
│         🏆 GANADOR PREDICHO                    │
│      🔵 New York Yankees                       │
└────────────────────────────────────────────────┘

┌──────────────┬──────────────┬──────────────┐
│ Prob. BOS    │ Prob. NYY    │  Confianza   │
│   37.7%      │   62.3%      │    62.3%     │
│              │              │  👍 ALTA     │
└──────────────┴──────────────┴──────────────┘

[Gráfico de barras]  [Gauge de confianza]
```

---

## 🐛 TROUBLESHOOTING

### Error: "API No Disponible"

**Solución:**
```bash
# Verifica que la API esté corriendo
curl http://localhost:8000/health

# Si no responde, iníciala:
uvicorn api:app --reload
```

### Error: "Module not found: streamlit"

**Solución:**
```bash
pip install streamlit plotly
```

### La app se ve mal en móvil

**Solución:** Streamlit no es responsive por defecto. Considera usar:
- CSS custom en `st.markdown()`
- O crear una versión mobile-first

### Predicción muy lenta

**Causa:** Scraping en tiempo real

**Soluciones:**
1. Usar cache en la API
2. Pre-calcular features
3. Implementar cola de trabajos (Celery)

---

## 📊 MEJORAS FUTURAS

- [ ] Comparar múltiples partidos a la vez
- [ ] Gráficos de tendencias históricas
- [ ] Exportar a Excel
- [ ] Notificaciones por email
- [ ] Integración con calendario MLB
- [ ] Modo oscuro/claro
- [ ] Autenticación de usuarios
- [ ] Base de datos para historial persistente

---

## 🎉 ¡Listo!

Tu web app está completa y lista para usar. Disfruta prediciendo partidos de MLB con una interfaz visual moderna y profesional.

**Comandos rápidos:**

```bash
# Terminal 1
uvicorn api:app --reload

# Terminal 2
streamlit run web_app.py
```

**URLs:**
- Web App: http://localhost:8501
- API: http://localhost:8000
- Docs API: http://localhost:8000/docs

---

**¡Que gane el mejor equipo! ⚾🎉**