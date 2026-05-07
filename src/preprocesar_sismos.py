"""
PASO 2 de 5 — CONSTRUCCIÓN DE LA SERIE TEMPORAL DIARIA
========================================================
¿Qué hace este archivo?
    Toma los sismos individuales y los agrupa por día, creando
    una serie continua día a día sin huecos.

¿Por qué agrupamos por día?
    Un LSTM trabaja con secuencias de tiempo ordenadas. En el dataset
    original puede haber varios sismos en el mismo día, o días donde
    no hubo ninguno. Necesitamos UN solo registro por día para tener
    una secuencia uniforme.

¿Qué pasa con los días sin sismos?
    No los borramos (habría huecos en la secuencia). En cambio,
    creamos una columna "hay_actividad" que vale 1 si ese día tuvo
    sismos y 0 si no. Para los valores de magnitud y profundidad,
    usamos el último valor conocido (forward fill).

Input  → data/processed/sismos_limpio.csv
Output → data/processed/sismos_serie_diaria.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Rutas de archivos ─────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
ARCHIVO_ENTRADA = BASE_DIR / "data" / "processed" / "sismos_limpio.csv"
ARCHIVO_SALIDA  = BASE_DIR / "data" / "processed" / "sismos_serie_diaria.csv"


def agregar_por_dia(df):
    """
    Agrupa los sismos individuales por fecha y calcula métricas diarias.

    Por cada día calcula:
        - avg_mag      → Magnitud promedio de todos los sismos del día
        - max_mag      → Magnitud del sismo más fuerte del día
        - avg_depth    → Profundidad promedio de los sismos del día
        - count_sismos → Cuántos sismos ocurrieron ese día
    """
    df["date"] = df["time"].dt.date
    df["date"] = pd.to_datetime(df["date"])

    serie_diaria = df.groupby("date").agg(
        avg_mag      = ("mag",   "mean"),   # promedio de magnitudes
        max_mag      = ("mag",   "max"),    # magnitud más alta del día
        avg_depth    = ("depth", "mean"),   # profundidad promedio
        count_sismos = ("mag",   "count")   # total de sismos registrados
    ).reset_index()

    return serie_diaria


def completar_dias_faltantes(serie):
    """
    Rellena los días donde no hubo sismos para que la secuencia
    no tenga huecos (requisito del LSTM).

    Estrategia:
        - count_sismos = 0  para días sin actividad
        - hay_actividad = 0 para marcar días sin sismos (columna binaria)
        - avg_mag, max_mag, avg_depth → se propaga el último valor conocido
          (forward fill) para no meter ceros que distorsionen el modelo
    """
    # Crear rango completo de fechas desde el primer hasta el último día
    rango_completo = pd.date_range(
        start=serie["date"].min(),
        end=serie["date"].max(),
        freq="D"  # frecuencia diaria
    )

    # Reindexar para incluir todos los días (los que no tenían datos quedan NaN)
    serie = serie.set_index("date").reindex(rango_completo)
    serie.index.name = "date"

    # Marcar días con y sin actividad ANTES de rellenar
    serie["hay_actividad"] = serie["count_sismos"].notna().astype(int)

    # Días sin sismos → count = 0 (sí es correcto poner cero aquí)
    serie["count_sismos"] = serie["count_sismos"].fillna(0).astype(int)

    # Para magnitud y profundidad: propagar el último valor conocido
    # (no ponemos cero porque eso indicaría "sismo de magnitud 0", que es incorrecto)
    serie["avg_mag"]   = serie["avg_mag"].ffill().bfill()
    serie["max_mag"]   = serie["max_mag"].ffill().bfill()
    serie["avg_depth"] = serie["avg_depth"].ffill().bfill()

    return serie.reset_index()


def construir_serie_diaria():
    """
    Función principal: construye la serie temporal diaria desde
    los sismos individuales del dataset limpio.
    """

    print("=" * 55)
    print("  PASO 2/5 — SERIE TEMPORAL DIARIA")
    print("=" * 55)

    # ── Cargar datos limpios ──────────────────────────────────────────────────
    print(f"\n📂 Cargando: {ARCHIVO_ENTRADA.name}")
    df = pd.read_csv(ARCHIVO_ENTRADA)
    df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)
    print(f"   Sismos individuales cargados: {len(df):,}")

    # ── Agrupar por día ───────────────────────────────────────────────────────
    print("\n📅 Agrupando sismos por día...")
    serie = agregar_por_dia(df)
    print(f"   Días con al menos un sismo: {len(serie):,}")

    # ── Completar días faltantes ──────────────────────────────────────────────
    print("\n🔗 Completando días sin actividad para evitar huecos...")
    serie = completar_dias_faltantes(serie)

    # ── Mostrar resultados ────────────────────────────────────────────────────
    dias_con_sismos    = serie["hay_actividad"].sum()
    dias_sin_sismos    = (serie["hay_actividad"] == 0).sum()
    porcentaje_activos = serie["hay_actividad"].mean() * 100

    print(f"\n  📊 Resumen de la serie diaria:")
    print(f"     Total de días en la serie : {len(serie):,}")
    print(f"     Días CON actividad sísmica: {dias_con_sismos:,} ({porcentaje_activos:.1f}%)")
    print(f"     Días SIN actividad sísmica: {dias_sin_sismos:,} ({100-porcentaje_activos:.1f}%)")
    print(f"     Periodo cubierto          : {serie['date'].min().date()} → {serie['date'].max().date()}")
    print(f"     Magnitud promedio general : {serie['avg_mag'].mean():.2f}")
    print(f"     Sismos máximos en un día  : {serie['count_sismos'].max():,}")
    print(f"     Columnas generadas        : {serie.columns.tolist()}")

    # ── Guardar ───────────────────────────────────────────────────────────────
    serie.to_csv(ARCHIVO_SALIDA, index=False)
    print(f"\n✅ Serie diaria guardada en: {ARCHIVO_SALIDA.name}\n")


# ── Punto de entrada ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    construir_serie_diaria()