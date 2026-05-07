"""
PASO 4 de 4 — DIVISIÓN EN ENTRENAMIENTO / VALIDACIÓN / TEST — ENERGÍA
=======================================================================
¿Qué hace este archivo?
    Divide las ventanas generadas en tres conjuntos para entrenar
    y evaluar el modelo de forma honesta e imparcial.

        - Train (70%)      → el modelo aprende con estos datos
        - Validación (15%) → monitoreamos si el modelo mejora o se estanca
        - Test (15%)       → evaluación final con datos que el modelo nunca vio

¿Por qué NO dividir aleatoriamente?
    Esto es una serie de tiempo. Si mezclamos los datos al azar,
    podría pasar que el modelo "aprenda" usando datos de enero 2010
    para predecir datos de enero 2008. Eso no tendría sentido en
    el mundo real, donde solo podemos usar el pasado para predecir el futuro.

    División correcta (cronológica):
    ┌─────────────────────────────────────────────────────────────┐
    │  [──────────── Train 70% ──────────────][─Val 15%─][─Test─]│
    │  Dic 2006 ─────────────────────────────────────→ Nov 2010  │
    └─────────────────────────────────────────────────────────────┘

Input  → data/processed/X_energia.npy
         data/processed/y_energia.npy
Output → data/processed/X_train_energia.npy  y_train_energia.npy
         data/processed/X_val_energia.npy    y_val_energia.npy
         data/processed/X_test_energia.npy   y_test_energia.npy
"""

import numpy as np
from pathlib import Path

# ── Rutas de archivos ─────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
PROCESADO = BASE_DIR / "data" / "processed"

ARCHIVO_X = PROCESADO / "X_energia.npy"
ARCHIVO_Y = PROCESADO / "y_energia.npy"

# ── Proporciones de división ──────────────────────────────────────────────────
PROPORCION_TRAIN = 0.70   # 70% para entrenar
PROPORCION_VAL   = 0.15   # 15% para validación
# El 15% restante se convierte automáticamente en test


def dividir_datos():
    """
    Función principal: carga las ventanas de energía y las divide
    en tres conjuntos respetando el orden cronológico.
    """

    print("=" * 55)
    print("  PASO 4/4 — DIVISIÓN TRAIN / VAL / TEST — ENERGÍA")
    print("=" * 55)

    # ── Cargar ventanas ───────────────────────────────────────────────────────
    print(f"\n📂 Cargando ventanas desde: {PROCESADO.name}/")
    print("   (Archivos grandes, puede tardar unos segundos...)")
    X = np.load(ARCHIVO_X)
    y = np.load(ARCHIVO_Y)

    total_muestras = len(X)
    print(f"   Total de muestras disponibles: {total_muestras:,}")
    print(f"   Forma de X: {X.shape}")

    # ── Calcular índices de corte ─────────────────────────────────────────────
    fin_train = int(total_muestras * PROPORCION_TRAIN)
    fin_val   = int(total_muestras * (PROPORCION_TRAIN + PROPORCION_VAL))

    # ── Dividir en orden cronológico ──────────────────────────────────────────
    X_train = X[:fin_train]
    y_train = y[:fin_train]

    X_val   = X[fin_train:fin_val]
    y_val   = y[fin_train:fin_val]

    X_test  = X[fin_val:]
    y_test  = y[fin_val:]

    # ── Mostrar tabla de resultados ───────────────────────────────────────────
    print(f"\n  📊 Resultado de la división:")
    print(f"  {'Conjunto':<12} {'Muestras':>12} {'Porcentaje':>12}")
    print(f"  {'-'*38}")
    print(f"  {'Train':<12} {len(X_train):>12,} {len(X_train)/total_muestras*100:>11.1f}%")
    print(f"  {'Validación':<12} {len(X_val):>12,} {len(X_val)/total_muestras*100:>11.1f}%")
    print(f"  {'Test':<12} {len(X_test):>12,} {len(X_test)/total_muestras*100:>11.1f}%")
    print(f"  {'-'*38}")
    print(f"  {'TOTAL':<12} {total_muestras:>12,} {'100.0%':>12}")

    # ── Verificación de integridad ────────────────────────────────────────────
    suma = len(X_train) + len(X_val) + len(X_test)
    if suma == total_muestras:
        print(f"\n  ✔ Verificación OK: {len(X_train):,} + {len(X_val):,} + {len(X_test):,} = {suma:,}")
    else:
        print(f"\n  ⚠️  Advertencia: la suma no coincide ({suma:,} ≠ {total_muestras:,})")

    # ── Guardar los 6 archivos ────────────────────────────────────────────────
    print(f"\n💾 Guardando archivos...")

    conjuntos = [
        ("X_train_energia.npy", X_train),
        ("y_train_energia.npy", y_train),
        ("X_val_energia.npy",   X_val),
        ("y_val_energia.npy",   y_val),
        ("X_test_energia.npy",  X_test),
        ("y_test_energia.npy",  y_test),
    ]

    for nombre, array in conjuntos:
        np.save(PROCESADO / nombre, array)
        print(f"   ✅ {nombre:<28} → forma: {array.shape}")

    print(f"\n✅ División completada. Archivos guardados en: data/processed/\n")


# ── Punto de entrada ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    dividir_datos()