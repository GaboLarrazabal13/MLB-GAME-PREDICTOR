"""
La base de datos de producción vive en un dataset de Hugging Face, no en git.

Guardar un SQLite binario de ~10 MB en git hacía crecer el repositorio con cada actualización
(tres al día en temporada). Ahora:
  - los jobs del workflow la descargan al empezar y la suben al terminar (los runs se ejecutan
    en serie por el grupo de concurrencia, así que no se pisan);
  - el Space y la app de Streamlit la descargan al arrancar si no la tienen.

    python mlb_datos_remotos.py descargar [--forzar]
    python mlb_datos_remotos.py subir "mensaje del commit"

Repositorio: variable HF_DATASET_REPO (por defecto GaboLarrazabal/mlb-game-predictor-data).
Subir requiere HF_TOKEN; descargar no, porque el dataset es público.
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_config import DB_PATH

REPO_POR_DEFECTO = "GaboLarrazabal/mlb-game-predictor-data"
ARCHIVO_REMOTO = "mlb_reentrenamiento.db"


def repo_id():
    return os.getenv("HF_DATASET_REPO", REPO_POR_DEFECTO)


def descargar(destino=DB_PATH, forzar=False):
    """Descarga la base de datos si falta (o siempre con forzar). Devuelve True si quedó en disco."""
    if os.path.exists(destino) and os.path.getsize(destino) > 0 and not forzar:
        return True
    from huggingface_hub import hf_hub_download

    ruta = hf_hub_download(repo_id=repo_id(), filename=ARCHIVO_REMOTO, repo_type="dataset",
                           token=os.getenv("HF_TOKEN") or None, force_download=forzar)
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    shutil.copyfile(ruta, destino)
    print(f"⬇️ Base de datos descargada de {repo_id()} ({os.path.getsize(destino) / 1e6:.1f} MB)")
    return True


def asegurar_db(destino=DB_PATH):
    """Para la API y el dashboard: intenta descargar la base de datos si no existe; nunca lanza."""
    try:
        return descargar(destino)
    except Exception as e:  # noqa: BLE001 - sin red se sigue con lo que haya
        print(f"⚠️ No se pudo descargar la base de datos: {e}")
        return os.path.exists(destino)


def subir(mensaje, origen=DB_PATH):
    from huggingface_hub import HfApi

    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("❌ Falta HF_TOKEN para subir la base de datos")
    HfApi(token=token).upload_file(path_or_fileobj=origen, path_in_repo=ARCHIVO_REMOTO, repo_id=repo_id(),
                                   repo_type="dataset", commit_message=mensaje)
    print(f"⬆️ Base de datos subida a {repo_id()}: {mensaje}")


def main():
    p = argparse.ArgumentParser(description="Sincroniza la base de datos con el dataset de Hugging Face")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("descargar")
    d.add_argument("--forzar", action="store_true")
    s = sub.add_parser("subir")
    s.add_argument("mensaje")
    a = p.parse_args()
    if a.cmd == "descargar":
        descargar(forzar=a.forzar)
    else:
        subir(a.mensaje)


if __name__ == "__main__":
    main()
