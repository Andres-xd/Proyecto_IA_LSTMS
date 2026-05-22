import pandas as pd
from pathlib import Path

# Rutas
BASE_DIR = Path(__file__).resolve().parent.parent
ARCHIVO_ENTRADA = BASE_DIR / "data" / "processed" / "sismos_limpio.csv"
ARCHIVO_SALIDA = BASE_DIR / "data" / "processed" / "sismos_serie_diaria.csv"


# Agrupacion
def agregar_por_dia(df):
    df["date"] = df["time"].dt.date
    df["date"] = pd.to_datetime(df["date"])

    serie_diaria = (
        df.groupby("date")
        .agg(
            avg_mag=("mag", "mean"),
            max_mag=("mag", "max"),
            avg_depth=("depth", "mean"),
            count_sismos=("mag", "count"),
        )
        .reset_index()
    )

    return serie_diaria


# Complecion
def completar_dias_faltantes(serie):
    rango_completo = pd.date_range(
        start=serie["date"].min(),
        end=serie["date"].max(),
        freq="D",
    )

    serie = serie.set_index("date").reindex(rango_completo)
    serie.index.name = "date"
    serie["hay_actividad"] = serie["count_sismos"].notna().astype(int)
    serie["count_sismos"] = serie["count_sismos"].fillna(0).astype(int)
    serie["avg_mag"] = serie["avg_mag"].ffill().bfill()
    serie["max_mag"] = serie["max_mag"].ffill().bfill()
    serie["avg_depth"] = serie["avg_depth"].ffill().bfill()

    return serie.reset_index()


# Flujo
def construir_serie_diaria():
    print("=" * 55)
    print("  PASO 2/5 - SERIE TEMPORAL DIARIA")
    print("=" * 55)

    # Carga
    print(f"\n(^_^) Cargando: {ARCHIVO_ENTRADA.name}")
    df = pd.read_csv(ARCHIVO_ENTRADA)
    df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)
    print(f"   Sismos individuales cargados: {len(df):,}")

    # Agrupacion
    print("\n(o_o) Agrupando sismos por dia...")
    serie = agregar_por_dia(df)
    print(f"   Dias con al menos un sismo: {len(serie):,}")

    # Complecion
    print("\n(=^.^=) Completando dias sin actividad para evitar huecos...")
    serie = completar_dias_faltantes(serie)

    # Resumen
    dias_con_sismos = serie["hay_actividad"].sum()
    dias_sin_sismos = (serie["hay_actividad"] == 0).sum()
    porcentaje_activos = serie["hay_actividad"].mean() * 100

    print("\n  (n_n) Resumen de la serie diaria:")
    print(f"     Total de dias en la serie : {len(serie):,}")
    print(f"     Dias CON actividad sismica: {dias_con_sismos:,} ({porcentaje_activos:.1f}%)")
    print(f"     Dias SIN actividad sismica: {dias_sin_sismos:,} ({100 - porcentaje_activos:.1f}%)")
    print(f"     Periodo cubierto          : {serie['date'].min().date()} -> {serie['date'].max().date()}")
    print(f"     Magnitud promedio general : {serie['avg_mag'].mean():.2f}")
    print(f"     Sismos maximos en un dia  : {serie['count_sismos'].max():,}")
    print(f"     Columnas generadas        : {serie.columns.tolist()}")

    # Guardado
    serie.to_csv(ARCHIVO_SALIDA, index=False)
    print(f"\n(^o^) Serie diaria guardada en: {ARCHIVO_SALIDA.name}\n")


# Inicio
if __name__ == "__main__":
    construir_serie_diaria()
