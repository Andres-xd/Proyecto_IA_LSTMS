"""
PASO 3 de 4 — VENTANAS DESLIZANTES (SLIDING WINDOW) — ENERGÍA
===============================================================
¿Qué hace este archivo?
    Convierte la serie temporal de energía en pares de (entrada, salida)
    que el modelo LSTM pueda usar para aprender a predecir.

¿Qué diferencia hay con las ventanas de sismos?
    Los datos de energía son minutales (una medición por minuto),
    mientras que los sísmicos eran diarios. Por eso aquí usamos
    una ventana de 60 pasos (= 60 minutos = 1 hora de historia)
    para predecir el minuto siguiente.

    Ejemplo con ventana de 3 pasos (para simplificar):
    ┌──────────────────────────────────────────────────────┐
    │  Minutos  1, 2, 3  →  Predecir minuto  4             │
    │  Minutos  2, 3, 4  →  Predecir minuto  5             │
    │  Minutos  3, 4, 5  →  Predecir minuto  6             │
    │  ...y así sucesivamente...                            │
    └──────────────────────────────────────────────────────┘

¿Qué predecimos?
    La potencia activa global del siguiente minuto (Global_active_power),
    que representa el consumo eléctrico total del hogar.

⚠️ Nota de rendimiento:
    Con ~2 millones de registros y ventana de 60 pasos, se generan
    ~2 millones de ventanas. Este proceso puede tardar varios minutos.

Input  → data/processed/energia_normalizado.csv
Output → data/processed/X_energia.npy  (ventanas de entrada)
         data/processed/y_energia.npy  (valores a predecir)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Rutas de archivos ─────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
ARCHIVO_ENTRADA = BASE_DIR / "data" / "processed" / "energia_normalizado.csv"
ARCHIVO_X       = BASE_DIR / "data" / "processed" / "X_energia.npy"
ARCHIVO_Y       = BASE_DIR / "data" / "processed" / "y_energia.npy"

# ── Parámetros de la ventana ──────────────────────────────────────────────────
TAMANIO_VENTANA  = 60                   # 60 minutos de historia = 1 hora
COLUMNA_OBJETIVO = "Global_active_power"  # variable a predecir

# ── Columnas que entran al modelo ─────────────────────────────────────────────
COLUMNAS_MODELO = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def crear_ventanas(datos, tamanio_ventana, indice_objetivo):
    """
    Genera todos los pares (X, y) deslizando la ventana sobre la serie.

    Parámetros:
        datos           → array 2D con la serie normalizada (filas=minutos, cols=features)
        tamanio_ventana → cuántos minutos atrás usamos para predecir
        indice_objetivo → índice de la columna que queremos predecir

    Retorna:
        X → array 3D de forma (muestras, minutos, features)
        y → array 1D con los valores de consumo a predecir
    """
    entradas  = []
    objetivos = []

    for i in range(len(datos) - tamanio_ventana):
        ventana   = datos[i : i + tamanio_ventana]          # 60 minutos de historia
        siguiente = datos[i + tamanio_ventana, indice_objetivo]  # minuto a predecir

        entradas.append(ventana)
        objetivos.append(siguiente)

    X = np.array(entradas,  dtype=np.float32)
    y = np.array(objetivos, dtype=np.float32)

    return X, y


def generar_ventanas():
    """
    Función principal: carga la serie normalizada de energía y
    genera las ventanas deslizantes para el LSTM.
    """

    print("=" * 55)
    print("  PASO 3/4 — VENTANAS DESLIZANTES — ENERGÍA")
    print("=" * 55)

    # ── Cargar datos normalizados ─────────────────────────────────────────────
    print(f"\n📂 Cargando: {ARCHIVO_ENTRADA.name}")
    print("   (Archivo grande, puede tardar unos segundos...)")
    df = pd.read_csv(ARCHIVO_ENTRADA, parse_dates=["datetime"])

    indice_objetivo = COLUMNAS_MODELO.index(COLUMNA_OBJETIVO)

    print(f"   Registros en la serie   : {len(df):,}")
    print(f"   Features usadas         : {COLUMNAS_MODELO}")
    print(f"   Variable a predecir     : '{COLUMNA_OBJETIVO}' (posición {indice_objetivo})")
    print(f"   Tamaño de ventana       : {TAMANIO_VENTANA} minutos (= 1 hora de historia)")

    # Convertir a array numpy
    datos = df[COLUMNAS_MODELO].values.astype(np.float32)

    # ── Crear ventanas deslizantes ────────────────────────────────────────────
    print(f"\n⚙️  Generando ventanas deslizantes...")
    print(f"   (Con {len(datos):,} registros esto puede tardar varios minutos...)")
    X, y = crear_ventanas(datos, TAMANIO_VENTANA, indice_objetivo)

    # ── Mostrar resultados ────────────────────────────────────────────────────
    print(f"\n  📊 Resultado de las ventanas:")
    print(f"     Total de ventanas generadas : {len(X):,}")
    print(f"     Forma de X (entradas)       : {X.shape}")
    print(f"       → {X.shape[0]:,} muestras, {X.shape[1]} minutos de historia, {X.shape[2]} variables")
    print(f"     Forma de y (objetivos)      : {y.shape}")
    print(f"     Rango de y (normalizado)    : {y.min():.4f} → {y.max():.4f}")

    # ── Guardar arrays ────────────────────────────────────────────────────────
    print(f"\n💾 Guardando archivos (pueden ser grandes, un momento)...")
    np.save(ARCHIVO_X, X)
    np.save(ARCHIVO_Y, y)

    print(f"\n✅ Entradas (X) guardadas en : {ARCHIVO_X.name}")
    print(f"✅ Objetivos (y) guardados en: {ARCHIVO_Y.name}\n")


# ── Punto de entrada ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    generar_ventanas()