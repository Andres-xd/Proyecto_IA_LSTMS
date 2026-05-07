"""
PASO 5 de 5 — DIVISIÓN EN ENTRENAMIENTO / VALIDACIÓN / TEST
=============================================================
¿Qué hace este archivo?
    Divide los datos en tres partes para evaluar el modelo de forma justa:
        - Train (70%)      → el modelo aprende con estos datos
        - Validación (15%) → ajustamos el modelo sin tocarlo directamente
        - Test (15%)       → evaluación final, el modelo nunca los vio

¿Por qué dividir en 3 partes?
    Si solo usáramos una parte para entrenar y otra para evaluar,
    podríamos ajustar el modelo hasta que funcione bien en la evaluación,
    pero estaríamos "haciendo trampa" sin saberlo. El conjunto de test
    es la prueba final, imparcial, que simula datos del mundo real.

⚠️ IMPORTANTE — División temporal (NO aleatoria):
    Como son datos de tiempo, NO se pueden mezclar. Dividimos en orden
    cronológico para que el modelo siempre prediga el futuro, nunca el pasado.

    Ejemplo:
    ┌────────────────────────────────────────────────────────┐
    │  [─────── Train 70% ──────────][─Val 15%─][─Test 15%─]│
    │  Datos más antiguos          ───────────→ Más recientes│
    └────────────────────────────────────────────────────────┘

Input  → data/processed/X_sismos.npy
         data/processed/y_sismos.npy
Output → data/processed/X_train_sismos.npy  y_train_sismos.npy
         data/processed/X_val_sismos.npy    y_val_sismos.npy
         data/processed/X_test_sismos.npy   y_test_sismos.npy
"""

import numpy as np
from pathlib import Path

# ── Rutas de archivos ─────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
PROCESADO = BASE_DIR / "data" / "processed"

ARCHIVO_X = PROCESADO / "X_sismos.npy"
ARCHIVO_Y = PROCESADO / "y_sismos.npy"

# ── Proporciones de división ──────────────────────────────────────────────────
PROPORCION_TRAIN = 0.70   # 70% para entrenar
PROPORCION_VAL   = 0.15   # 15% para validación
# El 15% restante automáticamente se convierte en test


def dividir_datos():
    """
    Función principal: carga las ventanas generadas y las divide
    en tres conjuntos respetando el orden temporal.
    """

    print("=" * 55)
    print("  PASO 5/5 — DIVISIÓN TRAIN / VAL / TEST")
    print("=" * 55)

    # ── Cargar ventanas deslizantes ───────────────────────────────────────────
    print(f"\n📂 Cargando ventanas desde: {PROCESADO.name}/")
    X = np.load(ARCHIVO_X)
    y = np.load(ARCHIVO_Y)

    total_muestras = len(X)
    print(f"   Total de muestras disponibles: {total_muestras:,}")
    print(f"   Forma de X: {X.shape}")

    # ── Calcular índices de corte ─────────────────────────────────────────────
    fin_train = int(total_muestras * PROPORCION_TRAIN)
    fin_val   = int(total_muestras * (PROPORCION_TRAIN + PROPORCION_VAL))

    # ── Dividir cronológicamente ──────────────────────────────────────────────
    X_train = X[:fin_train]
    y_train = y[:fin_train]

    X_val   = X[fin_train:fin_val]
    y_val   = y[fin_train:fin_val]

    X_test  = X[fin_val:]
    y_test  = y[fin_val:]

    # ── Mostrar resultados de la división ─────────────────────────────────────
    print(f"\n  📊 Resultado de la división:")
    print(f"  {'Conjunto':<12} {'Muestras':>10} {'Porcentaje':>12}")
    print(f"  {'-'*36}")
    print(f"  {'Train':<12} {len(X_train):>10,} {len(X_train)/total_muestras*100:>11.1f}%")
    print(f"  {'Validación':<12} {len(X_val):>10,} {len(X_val)/total_muestras*100:>11.1f}%")
    print(f"  {'Test':<12} {len(X_test):>10,} {len(X_test)/total_muestras*100:>11.1f}%")
    print(f"  {'-'*36}")
    print(f"  {'TOTAL':<12} {total_muestras:>10,} {'100.0%':>12}")

    # Verificación: que la suma de los 3 conjuntos sea igual al total
    suma = len(X_train) + len(X_val) + len(X_test)
    if suma == total_muestras:
        print(f"\n  ✔ Verificación OK: {len(X_train)} + {len(X_val)} + {len(X_test)} = {suma}")
    else:
        print(f"\n  ⚠️  Advertencia: la suma no coincide ({suma} ≠ {total_muestras})")

    # ── Guardar los 6 archivos ────────────────────────────────────────────────
    print(f"\n💾 Guardando archivos...")

    conjuntos = [
        ("X_train_sismos.npy", X_train),
        ("y_train_sismos.npy", y_train),
        ("X_val_sismos.npy",   X_val),
        ("y_val_sismos.npy",   y_val),
        ("X_test_sismos.npy",  X_test),
        ("y_test_sismos.npy",  y_test),
    ]

    for nombre, array in conjuntos:
        np.save(PROCESADO / nombre, array)
        print(f"   ✅ {nombre:<25} → forma: {array.shape}")

    print(f"\n✅ División completada. Archivos guardados en: data/processed/\n")


# ── Punto de entrada ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    dividir_datos()