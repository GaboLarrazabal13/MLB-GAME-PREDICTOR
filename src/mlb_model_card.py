"""
Ficha del modelo en producción (`models/model_card.json`): versión, fecha de corte de los
datos, features y métricas de la ventana sombra con la que se promovió.

Módulo ligero a propósito: lo importan la API y el motor de predicción, que no deben cargar
Optuna ni el resto del entrenamiento.
"""

import json
import os

from mlb_config import MODEL_CARD_PATH

VERSION_LEGADO = "catboost-20260527-legacy"


def leer_model_card(path=MODEL_CARD_PATH):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def escribir_model_card(card, path=MODEL_CARD_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2, ensure_ascii=False)
        f.write("\n")


def version_modelo_actual(card=None, path=MODEL_CARD_PATH):
    card = card if card is not None else leer_model_card(path)
    return card.get("version", VERSION_LEGADO) if card else VERSION_LEGADO
