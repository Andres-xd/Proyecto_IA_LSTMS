import pandas as pd
from pathlib import Path


# =========================
# CONFIGURACIÓN DE RUTAS
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "raw" / "household_power_consumption.txt"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "energia_limpio.csv"

# Intervalo de agregación en minutos (15, 30, 60)
# Cambia este valor para probar diferentes agregaciones
INTERVALO_AGREGACION = 60  # 1 hora por defecto


# =========================
# FUNCIÓN PRINCIPAL
# =========================
def limpiar_dataset_energia():
    print("Leyendo dataset de energía...")

    # Leer archivo original
    # sep=';' porque el TXT usa punto y coma como separador
    # na_values='?' porque este dataset suele usar ? para valores faltantes
    df = pd.read_csv(
        INPUT_FILE,
        sep=";",
        na_values=["?"],
        low_memory=False
    )

    print(f"Registros originales: {len(df)}")
    print("Columnas detectadas:")
    print(df.columns.tolist())

    # =========================
    # UNIR FECHA Y HORA
    # =========================
    print("\nConstruyendo columna datetime...")
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce"
    )

    # Eliminar columnas originales de fecha y hora
    df.drop(columns=["Date", "Time"], inplace=True)

    # =========================
    # CONVERTIR VARIABLES A NUMÉRICO
    # =========================
    columnas_numericas = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]

    print("\nConvirtiendo columnas numéricas...")
    for col in columnas_numericas:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # =========================
    # ELIMINAR FILAS CON DATETIME INVÁLIDO
    # =========================
    antes = len(df)
    df = df.dropna(subset=["datetime"])
    despues = len(df)
    print(f"Filas eliminadas por datetime inválido: {antes - despues}")

    # =========================
    # ORDENAR CRONOLÓGICAMENTE
    # =========================
    print("\nOrdenando datos por fecha...")
    df = df.sort_values("datetime").reset_index(drop=True)

    # =========================
    # REVISIÓN DE VALORES FALTANTES
    # =========================
    print("\nValores faltantes por columna antes de imputar:")
    print(df.isna().sum())

    # Interpolación temporal para no romper la serie
    print("\nAplicando interpolación temporal...")
    df = df.set_index("datetime")
    df[columnas_numericas] = df[columnas_numericas].interpolate(method="time")

    # Relleno extra por seguridad si quedó algo al inicio o final
    df[columnas_numericas] = df[columnas_numericas].bfill().ffill()

    df = df.reset_index()

    print("\nValores faltantes por columna después de imputar:")
    print(df.isna().sum())

    # =========================
    # GUARDAR RESULTADO
    # =========================
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nDataset limpio guardado en: {OUTPUT_FILE}")
    print("\nPrimeras filas del dataset limpio:")
    print(df.head())
    
    return df


def agregar_temporalmente(df, minutos=60):
    """
    Agrega datos de energía por intervalos de tiempo.
    
    Parámetros
    ----------
    df : DataFrame
        DataFrame con datos limpios
    minutos : int
        Intervalo de agregación (15, 30, 60)
    """
    OUTPUT_AGG = BASE_DIR / "data" / "processed" / f"energia_agregado_{minutos}min.csv"
    
    print("\n" + "=" * 60)
    print(f"  AGREGACIÓN TEMPORAL — {minutos} minutos")
    print("=" * 60)
    
    registros_originales = len(df)
    print(f"\nRegistros originales (1 min): {registros_originales:,}")
    
    # Establecer datetime como índice
    df = df.set_index("datetime")
    
    # Estrategia de agregación
    agg_dict = {
        "Global_active_power":    "sum",    # Consumo acumulado
        "Global_reactive_power":  "sum",
        "Sub_metering_1":         "sum",
        "Sub_metering_2":         "sum",
        "Sub_metering_3":         "sum",
        "Voltage":                "mean",   # Promedio
        "Global_intensity":       "mean",
    }
    
    print(f"Agregando cada {minutos} minutos...")
    df_agg = df.resample(f"{minutos}min").agg(agg_dict)
    df_agg = df_agg.dropna().reset_index()
    
    registros_agregados = len(df_agg)
    reduccion = (1 - registros_agregados / registros_originales) * 100
    
    print(f"Registros agregados: {registros_agregados:,}")
    print(f"Reducción: {reduccion:.1f}% ({registros_originales / registros_agregados:.1f}x menos datos)")
    
    # Guardar
    df_agg.to_csv(OUTPUT_AGG, index=False)
    print(f"\n Guardado en: {OUTPUT_AGG.name}")
    
    return df_agg


if __name__ == "__main__":
    # Paso 1: Limpiar datos originales
    df_limpio = limpiar_dataset_energia()
    
    # Paso 2: Agregar temporalmente (configurable arriba)
    print("\n" + "=" * 60)
    print("  INICIANDO AGREGACIÓN TEMPORAL")
    print("=" * 60)
    print(f"Intervalo configurado: {INTERVALO_AGREGACION} minutos")
    print("(Cambia INTERVALO_AGREGACION al inicio del script para probar otros valores)")
    
    agregar_temporalmente(df_limpio, minutos=INTERVALO_AGREGACION)
    
    print("\n" + "=" * 60)
    print("  PROCESO COMPLETADO")
    print("=" * 60)
    print(f" energia_limpio.csv → datos minuto a minuto")
    print(f" energia_agregado_{INTERVALO_AGREGACION}min.csv → datos cada {INTERVALO_AGREGACION} min")