"""
Entrada del reentrenamiento champion/challenger.

    python train_model.py                 # evalúa y promueve si procede (respeta la cadencia)
    python train_model.py --forzar        # ignora la cadencia
    python train_model.py --solo-evaluar  # calcula métricas sin tocar el modelo
    python train_model.py --trials 5      # menos trials de Optuna (pruebas)
"""

import argparse
import os
import sqlite3
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_config import DB_PATH
from mlb_training import debe_entrenar, ejecutar_reentrenamiento, leer_model_card

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Reentrenamiento champion/challenger")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--dias-sombra", type=int, default=30)
    parser.add_argument("--forzar", action="store_true", help="Entrena aunque no toque por cadencia")
    parser.add_argument("--solo-evaluar", action="store_true", help="No guarda modelo ni model card")
    args = parser.parse_args()

    if not args.forzar and not args.solo_evaluar:
        with sqlite3.connect(DB_PATH) as conn:
            toca, razon = debe_entrenar(conn, leer_model_card())
        if not toca:
            print(f"⏭️ No toca entrenar: {razon}")
            return
        print(f"▶️ Toca entrenar: {razon}")

    ejecutar_reentrenamiento(n_trials=args.trials, dias_sombra=args.dias_sombra, solo_evaluar=args.solo_evaluar)


if __name__ == "__main__":
    main()
