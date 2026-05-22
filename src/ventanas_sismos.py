import numpy as np
import pandas as pd
from pathlib import Path

# Rutas
BASE_DIR = Path(__file__).resolve().parent.parent
ARCHIVO_ENTRADA = BASE_DIR / "data" / "processed" / "sismos_normalizado.csv"
ARCHIVO_X = BASE_DIR / "data" / "processed" / "X_sismos.npy"
ARCHIVO_Y = BASE_DIR / "data" / "processed" / "y_sismos.npy"

# Parametros
TAMANIO_VENTANA = 14
COLUMNA_OBJETIVO = "avg_mag"


# Ventanas
def crear_ventanas(datos, tamanio_ventana, indice_objetivo):
    entradas = []
    objetivos = []

    for i in range(len(datos) - tamanio_ventana):
        ventana = datos[i : i + tamanio_ventana]
        siguiente = datos[i + tamanio_ventana, indice_objetivo]
        entradas.append(ventana)
        objetivos.append(siguiente)

    X = np.array(entradas, dtype=np.float32)
    y = np.array(objetivos, dtype=np.float32)
    return X, y


# Flujo
def generar_ventanas():
    print("=" * 55)
    print("  PASO 4/5 - VENTANAS DESLIZANTES")
    print("=" * 55)

    # Carga
    print(f"\n(^_^) Cargando: {ARCHIVO_ENTRADA.name}")
    df = pd.read_csv(ARCHIVO_ENTRADA, parse_dates=["date"])

    # Columnas
    columnas_features = [c for c in df.columns if c != "date"]
    indice_objetivo = columnas_features.index(COLUMNA_OBJETIVO)

    print(f"   Dias en la serie    : {len(df):,}")
    print(f"   Features usadas     : {columnas_features}")
    print(f"   Variable a predecir : '{COLUMNA_OBJETIVO}' (posicion {indice_objetivo})")
    print(f"   Tamano de ventana   : {TAMANIO_VENTANA} dias")

    # Datos
    datos = df[columnas_features].to_numpy(dtype=np.float32)

    # Generacion
    print("\n(o_o) Generando ventanas deslizantes...")
    X, y = crear_ventanas(datos, TAMANIO_VENTANA, indice_objetivo)

    # Resumen
    print("\n  (n_n) Resultado de las ventanas:")
    print(f"     Total de ventanas generadas : {len(X):,}")
    print(f"     Forma de X                  : {X.shape}")
    print(f"     Forma de y                  : {y.shape}")
    print(f"     Rango de y                  : {y.min():.4f} -> {y.max():.4f}")

    # Guardado
    np.save(ARCHIVO_X, X)
    np.save(ARCHIVO_Y, y)

    print(f"\n(^o^) Entradas guardadas en : {ARCHIVO_X.name}")
    print(f"(^.^) Objetivos guardados en: {ARCHIVO_Y.name}\n")


# Inicio
if __name__ == "__main__":
    generar_ventanas()
