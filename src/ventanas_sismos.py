"""
PASO 4 de 5 — VENTANAS DESLIZANTES (SLIDING WINDOW)
=====================================================
¿Qué hace este archivo?
    Convierte la serie temporal en pares de (entrada, salida) que
    el modelo LSTM pueda usar para aprender.

¿Qué es una ventana deslizante?
    Es una técnica para transformar una secuencia en un problema
    de predicción supervisado. Tomamos un bloque de N días como
    entrada (X) y el día siguiente como lo que queremos predecir (y).

    Ejemplo con ventana de 3 días (para simplificar):
    ┌─────────────────────────────────────────────────┐
    │  Días  1, 2, 3  →  Predecir día  4              │
    │  Días  2, 3, 4  →  Predecir día  5              │
    │  Días  3, 4, 5  →  Predecir día  6              │
    │  ...y así sucesivamente hasta el final...        │
    └─────────────────────────────────────────────────┘

    En nuestro caso usamos 14 días de historia para predecir el día 15.

¿Qué predecimos?
    La magnitud promedio del día siguiente (avg_mag).

Input  → data/processed/sismos_normalizado.csv
Output → data/processed/X_sismos.npy  (ventanas de entrada)
         data/processed/y_sismos.npy  (valores a predecir)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Rutas de archivos ─────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
ARCHIVO_ENTRADA = BASE_DIR / "data" / "processed" / "sismos_normalizado.csv"
ARCHIVO_X       = BASE_DIR / "data" / "processed" / "X_sismos.npy"
ARCHIVO_Y       = BASE_DIR / "data" / "processed" / "y_sismos.npy"

# ── Parámetros de la ventana ──────────────────────────────────────────────────
TAMANIO_VENTANA = 14       # cuántos días de historia usamos como entrada
COLUMNA_OBJETIVO = "avg_mag"  # qué variable queremos predecir


def crear_ventanas(datos, tamanio_ventana, indice_objetivo):
    """
    Genera todos los pares (X, y) deslizando la ventana sobre la serie.

    Parámetros:
        datos           → array 2D con la serie normalizada (filas=días, cols=features)
        tamanio_ventana → cuántos días atrás miramos para predecir
        indice_objetivo → índice de la columna que queremos predecir

    Retorna:
        X → array 3D de forma (muestras, días, features)
        y → array 1D con los valores a predecir
    """
    entradas  = []  # lista de ventanas de entrada
    objetivos = []  # lista de valores a predecir

    for i in range(len(datos) - tamanio_ventana):
        # Tomamos tamanio_ventana días como entrada
        ventana = datos[i : i + tamanio_ventana]
        # El día siguiente es lo que queremos predecir
        siguiente = datos[i + tamanio_ventana, indice_objetivo]

        entradas.append(ventana)
        objetivos.append(siguiente)

    X = np.array(entradas,  dtype=np.float32)
    y = np.array(objetivos, dtype=np.float32)

    return X, y


def generar_ventanas():
    """
    Función principal: carga la serie normalizada y genera las
    ventanas deslizantes para entrenamiento del LSTM.
    """

    print("=" * 55)
    print("  PASO 4/5 — VENTANAS DESLIZANTES")
    print("=" * 55)

    # ── Cargar datos normalizados ─────────────────────────────────────────────
    print(f"\n📂 Cargando: {ARCHIVO_ENTRADA.name}")
    df = pd.read_csv(ARCHIVO_ENTRADA, parse_dates=["date"])

    # Separar las columnas de features (sin la columna de fecha)
    columnas_features = [c for c in df.columns if c != "date"]
    indice_objetivo   = columnas_features.index(COLUMNA_OBJETIVO)

    print(f"   Días en la serie    : {len(df):,}")
    print(f"   Features usadas     : {columnas_features}")
    print(f"   Variable a predecir : '{COLUMNA_OBJETIVO}' (posición {indice_objetivo})")
    print(f"   Tamaño de ventana   : {TAMANIO_VENTANA} días")

    # Convertir a array numpy para procesarlo más rápido
    datos = df[columnas_features].values.astype(np.float32)

    # ── Crear ventanas deslizantes ────────────────────────────────────────────
    print("\n⚙️  Generando ventanas deslizantes...")
    X, y = crear_ventanas(datos, TAMANIO_VENTANA, indice_objetivo)

    # ── Mostrar resultados ────────────────────────────────────────────────────
    print(f"\n  📊 Resultado de las ventanas:")
    print(f"     Total de ventanas generadas : {len(X):,}")
    print(f"     Forma de X (entradas)       : {X.shape}")
    print(f"       → {X.shape[0]} muestras, {X.shape[1]} días de historia, {X.shape[2]} variables")
    print(f"     Forma de y (objetivos)      : {y.shape}")
    print(f"       → {y.shape[0]} valores a predecir")
    print(f"     Rango de y (normalizado)    : {y.min():.4f} → {y.max():.4f}")

    # ── Guardar arrays ────────────────────────────────────────────────────────
    np.save(ARCHIVO_X, X)
    np.save(ARCHIVO_Y, y)

    print(f"\n✅ Entradas (X) guardadas en : {ARCHIVO_X.name}")
    print(f"✅ Objetivos (y) guardados en: {ARCHIVO_Y.name}\n")


# ── Punto de entrada ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    generar_ventanas()