"""
PASO 2 de 4 — NORMALIZACIÓN DEL DATASET DE ENERGÍA
=====================================================
¿Qué hace este archivo?
    Escala todos los valores numéricos entre 0 y 1 para que el
    modelo LSTM pueda aprender de forma más eficiente.

¿Por qué normalizar?
    Las columnas tienen escalas muy distintas:
        - Global_active_power  → valores entre 0 y 11 (kW)
        - Voltage              → valores entre 220 y 255 (V)
        - Sub_metering_3       → valores entre 0 y 30 (Wh)
    Sin normalizar, el modelo le daría más importancia al Voltage
    solo porque sus números son más grandes. Al escalar todo entre
    0 y 1, todas las variables tienen el mismo peso inicial.

⚠️ REGLA IMPORTANTE — Evitar "Data Leakage":
    El normalizador (scaler) se ENTRENA solo con el 70% más antiguo
    de los datos (conjunto de train). Luego se aplica a TODOS los datos.

    ¿Por qué? Si entrenamos el scaler con TODOS los datos, estaríamos
    usando información del futuro para normalizar el pasado. Eso haría
    que el modelo parezca más preciso de lo que realmente es.

Input  → data/processed/energia_limpio.csv
Output → data/processed/energia_normalizado.csv
         data/processed/scaler_energia.pkl
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# ── Rutas de archivos ─────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
ARCHIVO_ENTRADA = BASE_DIR / "data" / "processed" / "energia_limpio.csv"
ARCHIVO_SALIDA  = BASE_DIR / "data" / "processed" / "energia_normalizado.csv"
ARCHIVO_SCALER  = BASE_DIR / "data" / "processed" / "scaler_energia.pkl"

# ── Columnas que entran al modelo ─────────────────────────────────────────────
# Todas las columnas numéricas del dataset de energía.
# 'datetime' no se normaliza, es solo referencia temporal.
COLUMNAS_MODELO = [
    "Global_active_power",    # ← esta es la variable que predecimos
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

# El scaler aprende los rangos solo del 70% más antiguo (= conjunto de train)
PORCENTAJE_TRAIN = 0.70


def normalizar_datos():
    """
    Función principal: normaliza el dataset de energía entre 0 y 1.

    Pasos:
        1. Carga el dataset limpio
        2. Entrena el scaler SOLO con el 70% inicial
        3. Aplica la normalización a todos los datos
        4. Guarda el resultado y el scaler para predicciones futuras
    """

    print("=" * 55)
    print("  PASO 2/4 — NORMALIZACIÓN DEL DATASET DE ENERGÍA")
    print("=" * 55)

    # ── Cargar datos limpios ──────────────────────────────────────────────────
    print(f"\n📂 Cargando: {ARCHIVO_ENTRADA.name}")
    print("   (Archivo grande, puede tardar unos segundos...)")
    df = pd.read_csv(ARCHIVO_ENTRADA, parse_dates=["datetime"])

    # Nos quedamos solo con datetime + columnas del modelo
    df = df[["datetime"] + COLUMNAS_MODELO].copy()
    print(f"   Registros cargados     : {len(df):,}")
    print(f"   Columnas a normalizar  : {len(COLUMNAS_MODELO)}")

    # ── Mostrar rangos ANTES de normalizar ────────────────────────────────────
    print("\n  📊 Rangos ANTES de normalizar:")
    for col in COLUMNAS_MODELO:
        print(f"     {col:<28} → "
              f"min: {df[col].min():>8.3f}  "
              f"max: {df[col].max():>8.3f}")

    # ── Entrenar scaler solo con datos de train ───────────────────────────────
    fin_train = int(len(df) * PORCENTAJE_TRAIN)

    print(f"\n⚙️  Entrenando normalizador con el {PORCENTAJE_TRAIN*100:.0f}% "
          f"inicial ({fin_train:,} registros)...")

    datos = df[COLUMNAS_MODELO]
    scaler = MinMaxScaler()
    scaler.fit(datos.iloc[:fin_train])           # aprende rangos solo del train
    datos_normalizados = scaler.transform(datos) # aplica a todos

    # ── Construir DataFrame normalizado ──────────────────────────────────────
    df_normalizado = df[["datetime"]].copy()
    df_normalizado[COLUMNAS_MODELO] = datos_normalizados

    # ── Mostrar rangos DESPUÉS de normalizar ──────────────────────────────────
    print("\n  📊 Rangos DESPUÉS de normalizar (esperado: aprox. 0 a 1):")
    for col in COLUMNAS_MODELO:
        print(f"     {col:<28} → "
              f"min: {df_normalizado[col].min():>6.3f}  "
              f"max: {df_normalizado[col].max():>6.3f}")

    # Nota: los valores del val/test pueden superar el 1.0 levemente si hay
    # picos que no estaban en el train. Eso es normal y esperado.

    # ── Guardar resultados ────────────────────────────────────────────────────
    df_normalizado.to_csv(ARCHIVO_SALIDA, index=False)
    joblib.dump(scaler, ARCHIVO_SCALER)

    print(f"\n✅ Datos normalizados guardados en : {ARCHIVO_SALIDA.name}")
    print(f"✅ Scaler guardado en              : {ARCHIVO_SCALER.name}")
    print(f"   (El scaler se necesita después para convertir predicciones al valor real)\n")


# ── Punto de entrada ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    normalizar_datos()