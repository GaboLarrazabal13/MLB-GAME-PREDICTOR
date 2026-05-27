# ⚾ Guía de Migración a Hugging Face Spaces

Dado que Render en su plan gratuito limita estrictamente los minutos de compilación (*build minutes*), **Hugging Face Spaces** es la solución ideal: es 100% gratuito, ofrece compilación ultrarrápida, no se apaga agresivamente y soporta contenedores **Docker** completos.

Hemos adaptado el código del proyecto para que **ambos servicios (el backend FastAPI y el frontend Streamlit) se ejecuten juntos de forma optimizada dentro del mismo contenedor**, comunicándose internamente en la red local (`127.0.0.1`).

Sigue estos sencillos pasos para tener tu predictor MLB en línea en menos de 5 minutos:

---

## Paso 1: Crear el Space en Hugging Face

1. Ve a [Hugging Face](https://huggingface.co/) e inicia sesión (o regístrate si no tienes cuenta).
2. Haz clic en tu perfil en la esquina superior derecha y selecciona **New Space** (o ve directo a [huggingface.co/new-space](https://huggingface.co/new-space)).
3. Configura el formulario de la siguiente manera:
   - **Space Name:** `mlb-game-predictor` (o el nombre que prefieras).
   - **License:** `mit` (u otra de tu preferencia).
   - **Select the Space SDK:** Selecciona **Docker** 🐳 (¡CRÍTICO! No selecciones Streamlit directamente, ya que necesitamos que Docker levante tanto FastAPI como Streamlit).
   - **Choose a Docker template:** Selecciona **Blank** (vacío).
   - **Space Hardware:** Deja el gratuito por defecto (**CPU basic • 2 vCPU • 16 GB RAM • Free**).
   - **Visibility:** **Public** (para que cualquiera pueda usarlo) o **Private** (solo tú).
4. Haz clic en **Create Space**.

---

## Paso 2: Generar tu Token de Acceso en Hugging Face

Para subir el código a Hugging Face a través de la terminal, necesitarás un Token de Escritura como contraseña:

1. Ve a **Settings** de tu cuenta de Hugging Face (haz clic en tu foto de perfil > **Settings**).
2. En la barra lateral izquierda, selecciona **Access Tokens**.
3. Haz clic en **New token**:
   - **Name:** `mlb-predictor-git`
   - **Permissions:** Selecciona **Write** (Escritura).
4. Haz clic en **Generate a token** y cópialo. (Guárdalo en un lugar seguro, lo usarás como contraseña al hacer push).

---

## Paso 3: Subir el Código desde la Terminal

Abre la terminal en la raíz de tu proyecto local (`c:\Users\gabo_\OneDrive\Escritorio\Proyectos\mlb-game-predictor`) y ejecuta los siguientes comandos:

1. **Añadir el repositorio de Hugging Face como un origen remoto secundario (`hf`):**
   ```bash
   git remote add hf https://huggingface.co/spaces/TU_USUARIO/TU_SPACE_NAME
   ```
   *(Reemplaza `TU_USUARIO` y `TU_SPACE_NAME` con tus datos de Hugging Face. Por ejemplo: `git remote add hf https://huggingface.co/spaces/GaboLarrazabal13/mlb-game-predictor`)*

2. **Subir la rama principal a Hugging Face:**
   ```bash
   git push -f hf main
   ```
   - **Username:** Tu nombre de usuario de Hugging Face.
   - **Password:** Pega el **Access Token** de escritura que copiaste en el Paso 2.

---

## Paso 4: ¡Listo! Monitorea tu Despliegue

1. Vuelve a la página de tu Space en Hugging Face en tu navegador.
2. Verás que el estado cambia a **Building**. Hugging Face compilará automáticamente tu contenedor usando el `Dockerfile` optimizado.
3. Una vez termine, cambiará a **Running** y ¡el Dashboard interactivo de Streamlit aparecerá directamente en pantalla!

---

## ¿Cómo funciona tras bambalinas?

- **Arranque Híbrido (`start.sh`):** Al iniciar, el contenedor ejecuta nuestro script `start.sh` que levanta la API FastAPI en segundo plano en `127.0.0.1:8000` y luego levanta Streamlit en primer plano en el puerto `7860` (que es el puerto que requiere Hugging Face).
- **Comunicación en Red Local:** Streamlit realiza las consultas HTTP a la API directamente a `http://127.0.0.1:8000` dentro del mismo contenedor, garantizando latencias bajísimas de respuesta y eliminando configuraciones complejas de CORS o IPs públicas para el backend.
- **Base de Datos Persistente en Git:** La base de datos SQLite con los resultados re-predecidos de 2026 (`data/mlb_reentrenamiento.db`) está incluida directamente en el código de Git, por lo que la aplicación arranca con todo el histórico listo e inyectado.
