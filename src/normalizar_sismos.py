"""
PASO 3 de 5 — NORMALIZACIÓN DE LA SERIE SÍSMICA
=================================================
¿Qué hace este archivo?
    Escala todos los valores numéricos entre 0 y 1 para que el
    modelo LSTM pueda aprender de forma más eficiente.

¿Por qué normalizar?
    Las redes neuronales son sensibles a la escala de los datos.
    Por ejemplo: "magnitud 4.5" y "profundidad 120km" tienen escalas
    muy diferentes. Si no normalizamos, el modelo le da más peso
    a la profundidad solo porque su número es más grande.
    Al normalizar entre 0 y 1, todas las variables tienen el mismo peso.

REGLA IMPORTANTE — Evitar "Data Leakage":
    El scaler (normalizador) solo se ENTRENA con los datos de Train.
    Luego se APLICA a todos los datos (train, validación y test).
    Esto evita que el modelo "vea" información del futuro durante
    el entrenamiento, lo que daría métricas artificialmente buenas.

Input  → data/processed/sismos_serie_diaria.csv
Output → data/processed/sismos_normalizado.csv
         data/processed/scaler_sismos.pkl  (para usar en predicciones reales)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# ── Rutas de archivos ─────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
ARCHIVO_ENTRADA = BASE_DIR / "data" / "processed" / "sismos_serie_diaria.csv"
ARCHIVO_SALIDA  = BASE_DIR / "data" / "processed" / "sismos_normalizado.csv"
ARCHIVO_SCALER  = BASE_DIR / "data" / "processed" / "scaler_sismos.pkl"

# ── Columnas que entran al modelo ─────────────────────────────────────────────
# Solo usamos las columnas numéricas que aportan información al LSTM.
# "date" no se normaliza porque es solo referencia temporal.
COLUMNAS_MODELO = ["avg_mag", "max_mag", "avg_depth", "count_sismos", "hay_actividad"]

# Porcentaje de datos que usamos para "entrenar" el normalizador
# (debe coincidir con el TRAIN_RATIO del paso de split)
PORCENTAJE_TRAIN = 0.70


def normalizar_serie():
    """
    Función principal: normaliza los datos entre 0 y 1 usando MinMaxScaler.

    Pasos:
        1. Carga la serie diaria
        2. Separa la columna de fechas de los valores numéricos
        3. Entrena el scaler SOLO con el 70% inicial (datos de entrenamiento)
        4. Aplica la normalización a TODOS los datos
        5. Guarda el resultado y el scaler para uso futuro
    """

    print("=" * 55)
    print("  PASO 3/5 — NORMALIZACIÓN DE LA SERIE SÍSMICA")
    print("=" * 55)

    # ── Cargar datos ──────────────────────────────────────────────────────────
    print(f"\n📂 Cargando: {ARCHIVO_ENTRADA.name}")
    df = pd.read_csv(ARCHIVO_ENTRADA, parse_dates=["date"])
    print(f"   Registros cargados: {len(df):,}")
    print(f"   Columnas a normalizar: {COLUMNAS_MODELO}")

    # ── Separar fechas de valores numéricos ───────────────────────────────────
    fechas = df["date"]
    datos  = df[COLUMNAS_MODELO].copy()

    # ── Mostrar rangos ANTES de normalizar ────────────────────────────────────
    print("\n  📊 Rangos ANTES de normalizar:")
    for col in COLUMNAS_MODELO:
        print(f"     {col:<15} → min: {datos[col].min():.3f}  max: {datos[col].max():.3f}")

    # ── Entrenar el scaler SOLO con datos de train ────────────────────────────
    # Calculamos dónde termina el 70% de los datos (cronológicamente)
    fin_train = int(len(datos) * PORCENTAJE_TRAIN)

    print(f"\n⚙️  Entrenando el normalizador con el {PORCENTAJE_TRAIN*100:.0f}% "
          f"inicial ({fin_train:,} registros)...")
    print(f"   (El restante {(1-PORCENTAJE_TRAIN)*100:.0f}% se normaliza pero NO se usa para entrenar el scaler)")

    scaler = MinMaxScaler()
    scaler.fit(datos.iloc[:fin_train])           # aprende los rangos del train
    datos_normalizados = scaler.transform(datos) # aplica a todos los datos

    # ── Construir DataFrame normalizado ──────────────────────────────────────
    df_normalizado = pd.DataFrame(datos_normalizados, columns=COLUMNAS_MODELO)
    df_normalizado.insert(0, "date", fechas)  # volver a agregar la columna de fecha

    # ── Mostrar rangos DESPUÉS de normalizar ──────────────────────────────────
    print("\n  📊 Rangos DESPUÉS de normalizar (esperado: entre 0 y 1):")
    for col in COLUMNAS_MODELO:
        print(f"     {col:<15} → min: {df_normalizado[col].min():.3f}  "
              f"max: {df_normalizado[col].max():.3f}")

    # ── Guardar resultados ────────────────────────────────────────────────────
    df_normalizado.to_csv(ARCHIVO_SALIDA, index=False)
    joblib.dump(scaler, ARCHIVO_SCALER)

    print(f"\n✅ Datos normalizados guardados en : {ARCHIVO_SALIDA.name}")
    print(f"✅ Scaler guardado en              : {ARCHIVO_SCALER.name}")
    print(f"   (El scaler se necesita después para convertir predicciones al valor real)\n")


# ── Punto de entrada ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    normalizar_serie()