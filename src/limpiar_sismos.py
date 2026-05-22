import pandas as pd
from pathlib import Path

# Rutas
BASE_DIR = Path(__file__).resolve().parent.parent
ARCHIVO_ENTRADA = BASE_DIR / "data" / "raw" / "Dataset_Sismología_Nuevo.csv"
ARCHIVO_SALIDA = BASE_DIR / "data" / "processed" / "sismos_limpio.csv"

(BASE_DIR / "data" / "processed").mkdir(parents=True, exist_ok=True)

# Columnas
COLUMNAS_UTILES = ["time", "latitude", "longitude", "depth", "mag"]


# Resumen
def mostrar_estadisticas(df, titulo):
    print(f"\n   {titulo}")
    print(f"     Registros  : {len(df):,}")
    if "mag" in df.columns:
        print(
            f"     Magnitud   : min={df['mag'].min():.1f}  "
            f"max={df['mag'].max():.1f}  "
            f"promedio={df['mag'].mean():.2f}"
        )
    if "depth" in df.columns:
        print(
            f"     Profundidad: min={df['depth'].min():.1f}km  "
            f"max={df['depth'].max():.1f}km"
        )
    if "time" in df.columns:
        print(f"     Periodo    : {df['time'].min().date()} -> {df['time'].max().date()}")


# Flujo
def limpiar_datos():
    print("=" * 55)
    print("  PASO 1/5 - LIMPIEZA DEL DATASET SISMICO")
    print("=" * 55)

    # Carga
    print(f"\n(^_^) Cargando archivo: {ARCHIVO_ENTRADA.name}")
    df = pd.read_csv(ARCHIVO_ENTRADA)
    print(f"   Registros encontrados : {len(df):,}")
    print(f"   Columnas en el archivo: {len(df.columns)}")

    # Seleccion
    print(f"\n(o_o) Seleccionando columnas utiles: {COLUMNAS_UTILES}")
    df = df[COLUMNAS_UTILES].copy()

    # Conversion
    print("\n(._.) Convirtiendo tipos de datos...")
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

    for columna in ["latitude", "longitude", "depth", "mag"]:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")

    # Limpieza
    registros_antes = len(df)
    df = df.dropna()
    registros_eliminados = registros_antes - len(df)
    print(f"\n  Registros eliminados por datos faltantes: {registros_eliminados:,}")

    # Filtros
    print("\n(>_<) Aplicando filtros de calidad...")
    antes = len(df)

    df = df[df["mag"] > 0]
    df = df[df["mag"] <= 10]
    df = df[df["depth"] >= 0]
    df = df[df["depth"] <= 700]
    df = df[df["time"] >= pd.Timestamp("1970-01-01", tz="UTC")]

    eliminados_filtros = antes - len(df)
    print(f"   Registros eliminados por filtros: {eliminados_filtros:,}")

    # Orden
    df = df.sort_values("time").reset_index(drop=True)

    # Resumen
    mostrar_estadisticas(df, "Resultado final despues de la limpieza")

    # Guardado
    df.to_csv(ARCHIVO_SALIDA, index=False)
    print(f"\n(^o^) Archivo limpio guardado en: {ARCHIVO_SALIDA.name}\n")


# Inicio
if __name__ == "__main__":
    limpiar_datos()
