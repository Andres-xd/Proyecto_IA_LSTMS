import numpy as np
import pandas as pd
from pathlib import Path

# Rutas
BASE_DIR = Path(__file__).resolve().parent.parent
ARCHIVO_ENTRADA = BASE_DIR / "data" / "processed" / "energia_normalizado.csv"
ARCHIVO_X = BASE_DIR / "data" / "processed" / "X_energia.npy"
ARCHIVO_Y = BASE_DIR / "data" / "processed" / "y_energia.npy"

# Parametros
TAMANIO_VENTANA = 60
COLUMNA_OBJETIVO = "Global_active_power"

# Columnas
COLUMNAS_MODELO = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


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
    print("  PASO 3/4 - VENTANAS DESLIZANTES - ENERGIA")
    print("=" * 55)

    # Carga
    print(f"\n(^_^) Cargando: {ARCHIVO_ENTRADA.name}")
    print("   (Archivo grande, puede tardar unos segundos...)")
    df = pd.read_csv(ARCHIVO_ENTRADA, parse_dates=["datetime"])

    indice_objetivo = COLUMNAS_MODELO.index(COLUMNA_OBJETIVO)

    print(f"   Registros en la serie   : {len(df):,}")
    print(f"   Features usadas         : {COLUMNAS_MODELO}")
    print(f"   Variable a predecir     : '{COLUMNA_OBJETIVO}' (posicion {indice_objetivo})")
    print(f"   Tamano de ventana       : {TAMANIO_VENTANA} minutos")

    # Datos
    datos = df[COLUMNAS_MODELO].to_numpy(dtype=np.float32)

    # Generacion
    print("\n(o_o) Generando ventanas deslizantes...")
    print(f"   (Con {len(datos):,} registros esto puede tardar varios minutos...)")
    X, y = crear_ventanas(datos, TAMANIO_VENTANA, indice_objetivo)

    # Resumen
    print("\n  (n_n) Resultado de las ventanas:")
    print(f"     Total de ventanas generadas : {len(X):,}")
    print(f"     Forma de X                  : {X.shape}")
    print(f"     Forma de y                  : {y.shape}")
    print(f"     Rango de y                  : {y.min():.4f} -> {y.max():.4f}")

    # Guardado
    print("\n(¬_¬) Guardando archivos...")
    np.save(ARCHIVO_X, X)
    np.save(ARCHIVO_Y, y)

    print(f"\n(^o^) Entradas guardadas en : {ARCHIVO_X.name}")
    print(f"(^.^) Objetivos guardados en: {ARCHIVO_Y.name}\n")


# Inicio
if __name__ == "__main__":
    generar_ventanas()
