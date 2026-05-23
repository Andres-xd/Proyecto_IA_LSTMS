import pandas as pd
import numpy as np
from pathlib import Path


# =========================
# CONFIGURACIÓN
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "energia_normalizado.csv"
OUTPUT_X = BASE_DIR / "data" / "processed" / "X_energia.npy"
OUTPUT_Y = BASE_DIR / "data" / "processed" / "y_energia.npy"

# Ventana de entrada: 48 horas (2 días completos de historia)
WINDOW_INPUT = 48

# Ventana de salida: 12 horas de predicción
WINDOW_OUTPUT = 12

TARGET_COL = "Global_active_power"


# =========================
# FUNCIÓN PARA CREAR VENTANAS MULTI-STEP
# =========================
def crear_ventanas_multistep(data, target_index, window_input, window_output):
    """
    Crea ventanas deslizantes con predicción multi-step.
    
    Entrada : 48 horas de historia (todas las features)
    Salida  : 12 horas futuras (solo Global_active_power)
    
    Ejemplo:
      X[0] = datos de hora 0 a hora 47   (48 horas, 7 features)
      y[0] = Global_active_power de hora 48 a hora 59  (12 valores)
    """
    X = []
    y = []
    
    total_window = window_input + window_output
    
    for i in range(len(data) - total_window):
        # Entrada: 48 horas de historia con TODAS las features
        X.append(data[i : i + window_input])
        # Salida: 12 horas futuras, solo la variable objetivo
        y.append(data[i + window_input : i + total_window, target_index])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# =========================
# FUNCIÓN PRINCIPAL
# =========================
def generar_ventanas_energia():
    print("=" * 60)
    print("  VENTANAS DESLIZANTES — Energía (Multi-Step)")
    print("=" * 60)
    
    print(f"\nLeyendo: {INPUT_FILE.name}")
    df = pd.read_csv(INPUT_FILE, parse_dates=["datetime"])

    columnas_features = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]

    df_features = df[columnas_features].copy()
    data = df_features.values
    target_index = columnas_features.index(TARGET_COL)

    print(f"Total de registros (horas): {len(df_features):,}")
    print(f"Features: {columnas_features}")
    print(f"Variable objetivo: {TARGET_COL}")
    print(f"\nConfiguración de ventanas:")
    print(f"  Entrada : {WINDOW_INPUT} horas ({WINDOW_INPUT // 24} días) de historia")
    print(f"  Salida  : {WINDOW_OUTPUT} horas de predicción")
    print(f"  Total   : {WINDOW_INPUT + WINDOW_OUTPUT} horas por ventana")

    print(f"\nGenerando ventanas multi-step...")
    X, y = crear_ventanas_multistep(data, target_index, WINDOW_INPUT, WINDOW_OUTPUT)

    print(f"\nResultado:")
    print(f"  X shape: {X.shape}  (ventanas, {WINDOW_INPUT}h entrada, {len(columnas_features)} features)")
    print(f"  y shape: {y.shape}  (ventanas, {WINDOW_OUTPUT}h predicción)")
    print(f"  Total ventanas generadas: {len(X):,}")

    # Guardar archivos
    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, y)

    print(f"\n X guardado en: {OUTPUT_X.name}")
    print(f" y guardado en: {OUTPUT_Y.name}")

    print(f"\nEjemplo de una muestra:")
    print(f"  X[0] shape: {X[0].shape} (48 horas de historia, 7 features)")
    print(f"  y[0] shape: {y[0].shape} (12 horas de predicción)")
    print(f"  y[0] = {y[0]}")


if __name__ == "__main__":
    generar_ventanas_energia()