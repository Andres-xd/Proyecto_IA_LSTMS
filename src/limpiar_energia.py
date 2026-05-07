"""
PASO 1 de 4 — LIMPIEZA DEL DATASET DE ENERGÍA ELÉCTRICA
=========================================================
¿Qué hace este archivo?
    Lee el dataset crudo de consumo eléctrico de un hogar,
    combina fecha y hora en una sola columna, rellena valores
    faltantes y guarda una versión limpia lista para procesar.

¿De dónde vienen los datos?
    Dataset "Individual Household Electric Power Consumption"
    del repositorio UCI Machine Learning.
    Contiene mediciones minuto a minuto de un hogar en Francia
    desde diciembre 2006 hasta noviembre 2010 (~2 millones de registros).

Columnas del dataset:
    - Date / Time             → Fecha y hora de la medición (minutal)
    - Global_active_power     → Potencia activa total del hogar (kW) ← OBJETIVO
    - Global_reactive_power   → Potencia reactiva total (kW)
    - Voltage                 → Voltaje de la red eléctrica (V)
    - Global_intensity        → Intensidad de corriente (A)
    - Sub_metering_1          → Energía en cocina (Wh)
    - Sub_metering_2          → Energía en lavandería (Wh)
    - Sub_metering_3          → Energía en agua caliente y A/C (Wh)

¿Por qué usar '?' como valor faltante?
    El dataset original codifica los datos faltantes como '?'
    en lugar de dejar la celda vacía. Pandas los convierte a NaN
    automáticamente con na_values=['?'].

Input  → data/raw/household_power_consumption.txt
Output → data/processed/energia_limpio.csv
"""

import pandas as pd
from pathlib import Path

# ── Rutas de archivos ─────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
ARCHIVO_ENTRADA = BASE_DIR / "data" / "raw" / "household_power_consumption.txt"
ARCHIVO_SALIDA  = BASE_DIR / "data" / "processed" / "energia_limpio.csv"

(BASE_DIR / "data" / "processed").mkdir(parents=True, exist_ok=True)

# ── Columnas numéricas del dataset ────────────────────────────────────────────
COLUMNAS_NUMERICAS = [
    "Global_active_power",    # potencia activa total ← la que predecimos
    "Global_reactive_power",  # potencia reactiva (complementa la activa)
    "Voltage",                # voltaje de la red
    "Global_intensity",       # intensidad de corriente
    "Sub_metering_1",         # consumo en cocina
    "Sub_metering_2",         # consumo en lavandería
    "Sub_metering_3",         # consumo en agua caliente / A/C
]


def cargar_dataset():
    """
    Carga el archivo TXT original del dataset de energía.

    El archivo usa ';' como separador (no coma), y '?' para
    indicar valores faltantes, así que se le avisa a pandas.
    """
    print(f"\n📂 Cargando archivo: {ARCHIVO_ENTRADA.name}")
    print(f"   (Este archivo tiene ~2 millones de registros, puede tardar unos segundos...)")

    df = pd.read_csv(
        ARCHIVO_ENTRADA,
        sep=";",           # separador punto y coma
        na_values=["?"],   # '?' significa dato faltante en este dataset
        low_memory=False   # evita advertencias por columnas con tipos mixtos
    )

    print(f"   Registros cargados : {len(df):,}")
    print(f"   Columnas detectadas: {df.columns.tolist()}")
    return df


def combinar_fecha_hora(df):
    """
    Combina las columnas 'Date' y 'Time' en una sola columna 'datetime'.

    El dataset original trae fecha y hora separadas, pero para trabajar
    con series de tiempo necesitamos un solo índice temporal.

    Ejemplo:
        Date='16/12/2006'  +  Time='17:24:00'  →  datetime='2006-12-16 17:24:00'
    """
    print("\n🔄 Combinando fecha y hora en una sola columna 'datetime'...")

    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce"  # fechas inválidas → NaN (las manejamos después)
    )

    # Ya no necesitamos las columnas separadas
    df = df.drop(columns=["Date", "Time"])

    fechas_invalidas = df["datetime"].isna().sum()
    if fechas_invalidas > 0:
        print(f"   ⚠️  Registros con fecha inválida: {fechas_invalidas:,} (serán eliminados)")

    return df


def convertir_a_numerico(df):
    """
    Asegura que todas las columnas de medición sean de tipo numérico.
    Valores que no se puedan convertir quedan como NaN.
    """
    print("\n🔢 Convirtiendo columnas a tipo numérico...")
    for col in COLUMNAS_NUMERICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def manejar_valores_faltantes(df):
    """
    Rellena los valores faltantes usando interpolación temporal.

    ¿Por qué interpolación y no eliminar?
        Con ~2 millones de registros, eliminar filas con NaN podría
        crear huecos en la serie que rompan la continuidad temporal.
        La interpolación estima el valor basándose en los puntos
        cercanos en el tiempo, que es la estrategia más adecuada
        para series de tiempo continuas.

    Después de la interpolación se aplica bfill/ffill por si quedó
    algún NaN al inicio o final de la serie.
    """
    print("\n🔍 Valores faltantes ANTES de rellenar:")
    faltantes = df[COLUMNAS_NUMERICAS].isna().sum()
    total_faltantes = faltantes.sum()
    if total_faltantes > 0:
        for col, cantidad in faltantes[faltantes > 0].items():
            print(f"   {col:<28} : {cantidad:,} faltantes")
    else:
        print("   Sin valores faltantes ✔")

    # Interpolación temporal (requiere que el índice sea datetime)
    print("\n⚙️  Aplicando interpolación temporal...")
    df = df.set_index("datetime")
    df[COLUMNAS_NUMERICAS] = df[COLUMNAS_NUMERICAS].interpolate(method="time")

    # Relleno adicional por si quedaron NaN al inicio o final
    df[COLUMNAS_NUMERICAS] = df[COLUMNAS_NUMERICAS].bfill().ffill()
    df = df.reset_index()

    faltantes_despues = df[COLUMNAS_NUMERICAS].isna().sum().sum()
    print(f"   Valores faltantes DESPUÉS de rellenar: {faltantes_despues:,}")

    return df


def limpiar_datos():
    """
    Función principal: coordina todos los pasos de limpieza
    del dataset de energía eléctrica.
    """

    print("=" * 55)
    print("  PASO 1/4 — LIMPIEZA DEL DATASET DE ENERGÍA")
    print("=" * 55)

    # ── 1. Cargar ─────────────────────────────────────────────────────────────
    df = cargar_dataset()

    # ── 2. Combinar fecha y hora ──────────────────────────────────────────────
    df = combinar_fecha_hora(df)

    # ── 3. Eliminar filas con fecha inválida ──────────────────────────────────
    antes = len(df)
    df = df.dropna(subset=["datetime"])
    eliminados = antes - len(df)
    if eliminados > 0:
        print(f"   Registros eliminados por fecha inválida: {eliminados:,}")

    # ── 4. Convertir columnas a numérico ──────────────────────────────────────
    df = convertir_a_numerico(df)

    # ── 5. Ordenar cronológicamente ───────────────────────────────────────────
    print("\n📅 Ordenando registros por fecha y hora...")
    df = df.sort_values("datetime").reset_index(drop=True)

    # ── 6. Rellenar valores faltantes ─────────────────────────────────────────
    df = manejar_valores_faltantes(df)

    # ── 7. Mostrar resumen final ──────────────────────────────────────────────
    print(f"\n  📊 Resumen del dataset limpio:")
    print(f"     Total de registros : {len(df):,}")
    print(f"     Periodo cubierto   : {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"     Global_active_power: "
          f"min={df['Global_active_power'].min():.3f}  "
          f"max={df['Global_active_power'].max():.3f}  "
          f"promedio={df['Global_active_power'].mean():.3f} kW")

    # ── 8. Guardar ────────────────────────────────────────────────────────────
    df.to_csv(ARCHIVO_SALIDA, index=False)
    print(f"\n✅ Dataset limpio guardado en: {ARCHIVO_SALIDA.name}\n")


# ── Punto de entrada ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    limpiar_datos()