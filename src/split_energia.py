import numpy as np
from pathlib import Path

# Rutas
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESADO = BASE_DIR / "data" / "processed"
ARCHIVO_X = PROCESADO / "X_energia.npy"
ARCHIVO_Y = PROCESADO / "y_energia.npy"

# Proporciones
PROPORCION_TRAIN = 0.70
PROPORCION_VAL = 0.15


# Flujo
def dividir_datos():
    print("=" * 55)
    print("  PASO 4/4 - DIVISION TRAIN / VAL / TEST - ENERGIA")
    print("=" * 55)

    # Carga
    print(f"\n(^_^) Cargando ventanas desde: {PROCESADO.name}/")
    print("   (Archivos grandes, puede tardar unos segundos...)")
    X = np.load(ARCHIVO_X)
    y = np.load(ARCHIVO_Y)

    total_muestras = len(X)
    print(f"   Total de muestras disponibles: {total_muestras:,}")
    print(f"   Forma de X: {X.shape}")

    # Cortes
    fin_train = int(total_muestras * PROPORCION_TRAIN)
    fin_val = int(total_muestras * (PROPORCION_TRAIN + PROPORCION_VAL))

    # Division
    X_train = X[:fin_train]
    y_train = y[:fin_train]
    X_val = X[fin_train:fin_val]
    y_val = y[fin_train:fin_val]
    X_test = X[fin_val:]
    y_test = y[fin_val:]

    # Resumen
    print("\n  (o_o) Resultado de la division:")
    print(f"  {'Conjunto':<12} {'Muestras':>12} {'Porcentaje':>12}")
    print(f"  {'-' * 38}")
    print(f"  {'Train':<12} {len(X_train):>12,} {len(X_train) / total_muestras * 100:>11.1f}%")
    print(f"  {'Validacion':<12} {len(X_val):>12,} {len(X_val) / total_muestras * 100:>11.1f}%")
    print(f"  {'Test':<12} {len(X_test):>12,} {len(X_test) / total_muestras * 100:>11.1f}%")
    print(f"  {'-' * 38}")
    print(f"  {'TOTAL':<12} {total_muestras:>12,} {'100.0%':>12}")

    # Verificacion
    suma = len(X_train) + len(X_val) + len(X_test)
    if suma == total_muestras:
        print(f"\n  (^.^) Verificacion OK: {len(X_train):,} + {len(X_val):,} + {len(X_test):,} = {suma:,}")
    else:
        print(f"\n  (>_<) Advertencia: la suma no coincide ({suma:,} != {total_muestras:,})")

    # Guardado
    print("\n(¬_¬) Guardando archivos...")

    conjuntos = [
        ("X_train_energia.npy", X_train),
        ("y_train_energia.npy", y_train),
        ("X_val_energia.npy", X_val),
        ("y_val_energia.npy", y_val),
        ("X_test_energia.npy", X_test),
        ("y_test_energia.npy", y_test),
    ]

    for nombre, array in conjuntos:
        np.save(PROCESADO / nombre, array)
        print(f"   (n_n) {nombre:<28} -> forma: {array.shape}")

    print("\n(^o^) Division completada. Archivos guardados en: data/processed/\n")


# Inicio
if __name__ == "__main__":
    dividir_datos()
