"""
PASO 1 de 5 — LIMPIEZA DEL DATASET SÍSMICO
============================================
¿Qué hace este archivo?
    Lee el dataset crudo de sismos, elimina datos incompletos o
    incorrectos, y guarda una versión limpia lista para procesar.

¿Por qué es importante?
    Los datos crudos suelen tener errores, valores faltantes o registros
    que no tienen sentido (ej: un sismo con profundidad negativa).
    Si no los limpiamos, el modelo aprende patrones incorrectos.

Input  → data/raw/Dataset_Sismología_Nuevo.csv
Output → data/processed/sismos_limpio.csv
"""

import pandas as pd
from pathlib import Path

# ── Rutas de archivos ─────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
ARCHIVO_ENTRADA = BASE_DIR / "data" / "raw" / "Dataset_Sismología_Nuevo.csv"
ARCHIVO_SALIDA  = BASE_DIR / "data" / "processed" / "sismos_limpio.csv"

# Crear carpeta de salida si no existe
(BASE_DIR / "data" / "processed").mkdir(parents=True, exist_ok=True)

# ── Columnas que vamos a usar ─────────────────────────────────────────────────
# El dataset tiene 22 columnas, pero muchas tienen datos incompletos o
# no son útiles para predecir actividad sísmica. Solo usamos estas 5:
COLUMNAS_UTILES = ["time", "latitude", "longitude", "depth", "mag"]
#   time      → Fecha y hora del sismo
#   latitude  → Latitud (posición norte-sur)
#   longitude → Longitud (posición este-oeste)
#   depth     → Profundidad del sismo en kilómetros
#   mag       → Magnitud del sismo (escala Richter)


def mostrar_estadisticas(df, titulo):
    """Muestra un resumen básico del DataFrame para saber cómo está."""
    print(f"\n   {titulo}")
    print(f"     Registros  : {len(df):,}")
    if "mag" in df.columns:
        print(f"     Magnitud   : min={df['mag'].min():.1f}  "
              f"max={df['mag'].max():.1f}  "
              f"promedio={df['mag'].mean():.2f}")
    if "depth" in df.columns:
        print(f"     Profundidad: min={df['depth'].min():.1f}km  "
              f"max={df['depth'].max():.1f}km")
    if "time" in df.columns:
        print(f"     Periodo    : {df['time'].min().date()} → {df['time'].max().date()}")


def limpiar_datos():
    """
    Función principal: lee, limpia y guarda el dataset sísmico.

    Pasos:
        1. Carga el CSV original
        2. Selecciona solo las columnas útiles
        3. Convierte los tipos de datos correctamente
        4. Elimina registros con valores faltantes (NaN)
        5. Aplica filtros de calidad para quitar datos inválidos
        6. Ordena por fecha y guarda el resultado
    """

    print("=" * 55)
    print("  PASO 1/5 — LIMPIEZA DEL DATASET SÍSMICO")
    print("=" * 55)

    # ── 1. Cargar dataset crudo ───────────────────────────────────────────────
    print(f"\n Cargando archivo: {ARCHIVO_ENTRADA.name}")
    df = pd.read_csv(ARCHIVO_ENTRADA)
    print(f"   Registros encontrados : {len(df):,}")
    print(f"   Columnas en el archivo: {len(df.columns)}")

    # ── 2. Seleccionar solo columnas útiles ───────────────────────────────────
    print(f"\n Seleccionando columnas útiles: {COLUMNAS_UTILES}")
    df = df[COLUMNAS_UTILES].copy()

    # ── 3. Convertir tipos de datos ───────────────────────────────────────────
    # "errors='coerce'" convierte valores inválidos a NaN (los manejamos después)
    print("\n Convirtiendo tipos de datos...")
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

    for columna in ["latitude", "longitude", "depth", "mag"]:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")

    # ── 4. Eliminar registros con valores faltantes ───────────────────────────
    registros_antes = len(df)
    df = df.dropna()
    registros_eliminados = registros_antes - len(df)
    print(f"\n  Registros eliminados por datos faltantes: {registros_eliminados:,}")

    # ── 5. Filtros de calidad ─────────────────────────────────────────────────
    # Eliminamos registros que físicamente no tienen sentido:
    print("\n Aplicando filtros de calidad... one moment please")

    antes = len(df)

    # Magnitud debe ser positiva y no puede superar 10 (máximo físico posible)
    df = df[df["mag"] > 0]
    df = df[df["mag"] <= 10]

    # Profundidad no puede ser negativa (y sismos reales no pasan de 700km)
    df = df[df["depth"] >= 0]
    df = df[df["depth"] <= 700]

    # Solo datos desde 1970 (antes los registros son menos confiables)
    df = df[df["time"] >= pd.Timestamp("1970-01-01", tz="UTC")]

    eliminados_filtros = antes - len(df)
    print(f"   Registros eliminados por filtros: {eliminados_filtros:,}")

    # ── 6. Ordenar por fecha y guardar ────────────────────────────────────────
    df = df.sort_values("time").reset_index(drop=True)

    mostrar_estadisticas(df, "Resultado final después de la limpieza")

    df.to_csv(ARCHIVO_SALIDA, index=False)
    print(f"\n Archivo limpio (con mister musculo) guardado en: {ARCHIVO_SALIDA.name}\n")
    


# ── Punto de entrada ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    limpiar_datos()