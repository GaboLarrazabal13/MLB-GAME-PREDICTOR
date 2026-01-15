# ⚾ MLB Predictor V3.5 - Sistema de Predicción de Partidos MLB

Sistema de machine learning para predecir resultados de partidos de béisbol de las Grandes Ligas (MLB) usando XGBoost y web scraping de Baseball-Reference.

## 🎯 Características Principales

- **Predicción híbrida**: Combina features temporales (tendencias) con stats actuales scrapeadas
- **Entrenamiento incremental**: Actualiza el modelo con nuevos datos sin reentrenar desde cero
- **Web scraping inteligente**: Extrae estadísticas de equipos, lanzadores y bateadores de Baseball-Reference
- **Sistema de caché**: Evita re-scraping innecesario y respeta rate limits
- **Análisis profundo**: Incluye bullpen, super features y análisis compuesto
- **Interfaz CLI amigable**: Modo interactivo y modo rápido por línea de comandos

## 📁 Estructura del Proyecto

```
mlb_predictor_v3.5/
├── mlb_config.py                    # Configuración centralizada
├── mlb_feature_engineering.py       # Cálculo de features
├── train_model_hybrid_actions.py    # Entrenamiento del modelo
├── mlb_predict_engine.py            # Motor de predicción
├── mlb_manual_interface.py          # Interfaz CLI
├── mlb_utils.py                     # Utilidades y mantenimiento
├── models/
│   ├── modelo_mlb_v3.5.json        # Modelo entrenado
│   └── modelo_mlb_v3.5_backup.json # Backup automático
├── data/
│   └── mlb_reentrenamiento.db      # Base de datos SQLite
└── cache/
    └── features_hibridas_v3.5_cache.pkl  # Caché de scraping
```

## 🚀 Instalación

### Requisitos

- Python 3.8+
- Dependencias:

```bash
pip install pandas numpy scikit-learn xgboost cloudscraper beautifulsoup4 lxml
```

### Configuración Inicial

1. Clona el repositorio:
```bash
git clone <tu-repo>
cd mlb_predictor_v3.5
```

2. Crea las carpetas necesarias (se crean automáticamente al ejecutar):
```bash
python mlb_config.py
```

3. Verifica el estado del sistema:
```bash
python mlb_utils.py estado
```

## 📖 Uso

### 1. Entrenamiento del Modelo

Para entrenar o actualizar el modelo con nuevos datos:

```bash
python train_model_hybrid_actions.py
```

**Características del entrenamiento:**
- Procesa datos en bloques de 150 juegos con pausas de 45s entre bloques
- Usa GridSearchCV para optimizar hiperparámetros
- Solo actualiza el modelo si mejora el accuracy
- Crea backups automáticos

**Variables importantes:**
- `bloque_size`: Número de juegos por bloque (default: 150)
- `pausa_entre_bloques`: Segundos de pausa entre bloques (default: 45)

### 2. Predicción Manual (Modo Interactivo)

```bash
python mlb_manual_interface.py
```

El modo interactivo te guiará paso a paso:

```
🏠 Equipo Local (Nombre/Código): Yankees
✅ Equipo local: New York Yankees (NYY)

✈️ Equipo Visitante (Nombre/Código): Red Sox
✅ Equipo visitante: Boston Red Sox (BOS)

👤 Lanzador abridor de NYY: Gerrit Cole
👤 Lanzador abridor de BOS: Tanner Houck
📅 Año para el scraping de stats (Enter=2026): 2024
```

### 3. Predicción Rápida (Línea de Comandos)

```bash
python mlb_manual_interface.py NYY BOS "Gerrit Cole" "Tanner Houck" 2024
```

### 4. Predicción Automática Diaria

```bash
python mlb_predict_engine.py
```

Lee los partidos del día desde la base de datos y genera predicciones automáticas.

## 🛠️ Utilidades de Mantenimiento

### Analizar Rendimiento del Modelo

```bash
# Últimos 30 días (default)
python mlb_utils.py accuracy

# Últimos 60 días
python mlb_utils.py accuracy 60
```

Muestra:
- Accuracy general
- Accuracy por nivel de confianza
- Comparación con resultados reales

### Reporte de Equipo

```bash
# Últimas 20 predicciones de Yankees
python mlb_utils.py equipo NYY

# Últimas 50 predicciones
python mlb_utils.py equipo NYY 50
```

### Limpiar Caché

```bash
python mlb_utils.py limpiar_cache
```

Elimina el caché de features para forzar re-scraping en el próximo entrenamiento.

### Compactar Base de Datos

```bash
python mlb_utils.py compactar
```

Optimiza y reduce el tamaño de la base de datos SQLite.

### Verificar Integridad

```bash
python mlb_utils.py verificar
```

Verifica que la base de datos no tenga corrupciones.

### Exportar Datos

```bash
# Exportar predicciones de los últimos 30 días
python mlb_utils.py exportar_pred 30

# Exportar resultados reales
python mlb_utils.py exportar_real 30
```

## 🧠 Características del Modelo

### Features Temporales (Tendencias)

- `home_win_rate_10`: % de victorias en últimos 10 juegos
- `home_racha`: Racha actual (positiva/negativa)
- `home_runs_avg`: Promedio de carreras anotadas
- `home_runs_diff`: Diferencial de carreras
- (Mismo conjunto para away)

### Features de Scraping

**Equipos:**
- OPS promedio del equipo
- ERA promedio del pitcheo
- Bateo promedio (BA)

**Lanzadores Abridores:**
- ERA, WHIP, SO9 (ponches por 9 innings)
- Récord W-L

**Mejores Bateadores (Top 3):**
- OPS, BA, HR, RBI

**Bullpen (Top 3 relevistas):**
- ERA promedio
- WHIP promedio

### Super Features

Features derivadas que capturan interacciones complejas:

1. **Neutralización WHIP vs OPS**: Mide cómo el WHIP del pitcher neutraliza el OPS rival
2. **Resistencia ERA vs OPS**: Capacidad del pitcher de resistir ofensiva rival
3. **Muro del Bullpen**: Efectividad del bullpen contra mejores bateadores

## 📊 Formato de Salida

### Ejemplo de Predicción

```
===========================================================================
   ⚾ MLB PREDICTOR V3.5 - ANÁLISIS ESTADÍSTICO
===========================================================================
 Encuentro: NYY vs BOS
 Temporada: 2024 | Scraping: Baseball-Reference

📊 COMPARATIVA DE EQUIPOS:
 🏠  NYY: OPS: 0.782 | Bullpen WHIP: 1.234
 ✈️  BOS: OPS: 0.758 | Bullpen WHIP: 1.301

👤 LANZADORES ABRIDORES:
 🏠 Gerrit Cole: ERA: 3.12 | WHIP: 1.089 | SO9: 10.2
 ✈️  Tanner Houck: ERA: 3.89 | WHIP: 1.234 | SO9: 8.7

🧱 ANÁLISIS DE BULLPEN:
 🏠 NYY: ERA: 3.456 | WHIP: 1.234
 ✈️  BOS: ERA: 3.891 | WHIP: 1.301
 📊 Diferencial ERA: +0.44

🔥 TOP 3 BATEADORES ANALIZADOS:
 🏠 NYY:
 Nombre                 | BA    | OBP   | SLG   | OPS   | HR  | RBI
 ---------------------------------------------------------------------------
 Aaron Judge            | 0.322 | 0.458 | 0.701 | 1.159 | 58  | 144
 Juan Soto              | 0.288 | 0.419 | 0.569 | 0.988 | 41  | 109
 Gleyber Torres         | 0.257 | 0.330 | 0.378 | 0.708 | 15  | 63

📈 TENDENCIAS RECIENTES (Últimos 10 juegos):
 🏠 NYY: Win Rate: 70.0% | Racha: +5
 ✈️  BOS: Win Rate: 40.0% | Racha: -3

===========================================================================
 🏆 GANADOR PREDICHO: New York Yankees
===========================================================================
 Probabilidades: NYY 68.5% | BOS 31.5%
 Confianza: ALTA

🚀 DIAGNÓSTICO DE SUPER FEATURES:
 🛡️ Neutralización: -0.0234 (Ventaja NYY)
 📉 Resistencia:    -0.1456 (Ventaja NYY)
 🧱 Muro Bullpen:   -0.0891 (Ventaja NYY)

💡 ANÁLISIS COMPUESTO:
 Ventaja Pitcheo: +0.234
 Ventaja Bateo:   +0.024
 Ventaja Momentum: +0.300
 Score Compuesto: +0.186
===========================================================================
```

## ⚙️ Configuración Avanzada

### Modificar Hiperparámetros del Modelo

Edita `mlb_config.py`:

```python
MODEL_CONFIG = {
    'test_size': 0.20,
    'random_state': 42,
    'cv_folds': 3,
    'param_grid': {
        'n_estimators': [200, 300, 400],      # Número de árboles
        'max_depth': [4, 6, 8],               # Profundidad máxima
        'learning_rate': [0.01, 0.03, 0.05],  # Tasa de aprendizaje
        'gamma': [0.1, 0.2]                   # Regularización
    }
}
```

### Ajustar Rate Limiting

Edita `mlb_config.py`:

```python
SCRAPING_CONFIG = {
    'max_retries': 3,
    'timeout': 15,
    'min_delay': 2,        # Mínimo delay entre requests
    'max_delay': 4,        # Máximo delay entre requests
    'rate_limit_wait': 10,
    'bloque_size': 150,    # Juegos por bloque
    'pausa_entre_bloques': 45  # Segundos entre bloques
}
```

## 🔧 Solución de Problemas

### Error: "No se pudieron extraer features"

**Causa**: Nombres de lanzadores incorrectos o no encontrados en Baseball-Reference

**Solución**:
- Verifica la ortografía del nombre del lanzador
- Usa el nombre completo (ej: "Gerrit Cole" en lugar de "G. Cole")
- Prueba con un año diferente donde el lanzador tenga datos

### Error: "Rate limit (429) detectado"

**Causa**: Demasiadas peticiones a Baseball-Reference

**Solución**:
- El sistema esperará automáticamente
- Aumenta `pausa_entre_bloques` en la configuración
- Reduce `bloque_size` para hacer pausas más frecuentes

### Modelo no mejora el accuracy

**Posibles causas**:
- Datos insuficientes
- Overfitting
- Features no relevantes

**Solución**:
1. Verifica que tengas al menos 500+ juegos entrenados
2. Ajusta hiperparámetros en `MODEL_CONFIG`
3. Analiza feature importance:

```python
from mlb_feature_engineering import generar_feature_importance_report
import xgboost as xgb

model = xgb.Booster()
model.load_model('models/modelo_mlb_v3.5.json')
report = generar_feature_importance_report(model, model.feature_names)
print(report)
```

## 📝 Estructura de la Base de Datos

### Tabla: `historico_real`

Resultados reales de partidos:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| fecha | TEXT | Fecha del partido (YYYY-MM-DD) |
| home_team | TEXT | Código equipo local |
| away_team | TEXT | Código equipo visitante |
| home_pitcher | TEXT | Lanzador abridor local |
| away_pitcher | TEXT | Lanzador abridor visitante |
| score_home | INTEGER | Carreras equipo local |
| score_away | INTEGER | Carreras equipo visitante |
| ganador | INTEGER | 1=local ganó, 0=visitante ganó |

### Tabla: `predicciones_historico`

Predicciones generadas:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| fecha | TEXT | Fecha de la predicción |
| home_team | TEXT | Código equipo local |
| away_team | TEXT | Código equipo visitante |
| home_pitcher | TEXT | Lanzador abridor local |
| away_pitcher | TEXT | Lanzador abridor visitante |
| prob_home | REAL | Probabilidad del local (0-100) |
| prob_away | REAL | Probabilidad del visitante (0-100) |
| prediccion | TEXT | Código del equipo predicho como ganador |
| confianza | TEXT | Nivel de confianza (MUY ALTA, ALTA, etc) |
| tipo | TEXT | MANUAL o AUTOMATICO |

### Tabla: `control_entrenamiento`

Control de juegos ya procesados:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| game_id | TEXT | ID único del partido (PRIMARY KEY) |

## 🤝 Contribuciones

Las contribuciones son bienvenidas! Por favor:

1. Haz fork del proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-feature`)
3. Commit tus cambios (`git commit -am 'Agrega nueva feature'`)
4. Push a la rama (`git push origin feature/nueva-feature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para detalles.

## ⚠️ Disclaimer

Este sistema es solo para fines educativos y de entretenimiento. No debe usarse para apuestas deportivas. Las predicciones son estimaciones basadas en datos históricos y no garantizan resultados futuros.

## 🙏 Créditos

- Datos: [Baseball-Reference](https://www.baseball-reference.com)
- ML Framework: [XGBoost](https://xgboost.readthedocs.io/)
- Web Scraping: [cloudscraper](https://github.com/VeNoMouS/cloudscraper)

## 📞 Soporte

Para reportar bugs o solicitar features, abre un issue en GitHub.

---

**Versión**: 3.5 (Refactorizada)  
**Última actualización**: Enero 2026  
**Autor**: [Tu nombre]