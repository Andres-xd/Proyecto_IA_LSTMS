import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# Rutas
BASE_DIR = Path(__file__).resolve().parent.parent
ARCHIVO_ENTRADA = BASE_DIR / "data" / "processed" / "sismos_serie_diaria.csv"
ARCHIVO_SALIDA = BASE_DIR / "data" / "processed" / "sismos_normalizado.csv"
ARCHIVO_SCALER = BASE_DIR / "data" / "processed" / "scaler_sismos.pkl"

# Columnas
COLUMNAS_MODELO = ["avg_mag", "max_mag", "avg_depth", "count_sismos", "hay_actividad"]

# Proporcion
PORCENTAJE_TRAIN = 0.70


# Flujo
def normalizar_serie():
    print("=" * 55)
    print("  PASO 3/5 - NORMALIZACION DE LA SERIE SISMICA")
    print("=" * 55)

    # Carga
    print(f"\n(^_^) Cargando: {ARCHIVO_ENTRADA.name}")
    df = pd.read_csv(ARCHIVO_ENTRADA, parse_dates=["date"])
    print(f"   Registros cargados: {len(df):,}")
    print(f"   Columnas a normalizar: {COLUMNAS_MODELO}")

    # Datos
    fechas = df["date"]
    datos = df[COLUMNAS_MODELO].copy()

    # Rangos
    print("\n  (o_o) Rangos ANTES de normalizar:")
    for col in COLUMNAS_MODELO:
        print(f"     {col:<15} -> min: {datos[col].min():.3f}  max: {datos[col].max():.3f}")

    # Ajuste
    fin_train = int(len(datos) * PORCENTAJE_TRAIN)
    print(
        f"\n(¬_¬) Entrenando el normalizador con el {PORCENTAJE_TRAIN * 100:.0f}% "
        f"inicial ({fin_train:,} registros)..."
    )
    print(
        f"   (El restante {(1 - PORCENTAJE_TRAIN) * 100:.0f}% se normaliza pero no se usa "
        "para entrenar el scaler)"
    )

    scaler = MinMaxScaler()
    scaler.fit(datos.iloc[:fin_train])
    datos_normalizados = scaler.transform(datos)

    # Salida
    df_normalizado = pd.DataFrame(datos_normalizados, columns=COLUMNAS_MODELO)
    df_normalizado.insert(0, "date", fechas)

    # Rangos
    print("\n  (^.^) Rangos DESPUES de normalizar:")
    for col in COLUMNAS_MODELO:
        print(
            f"     {col:<15} -> min: {df_normalizado[col].min():.3f}  "
            f"max: {df_normalizado[col].max():.3f}"
        )

    # Guardado
    df_normalizado.to_csv(ARCHIVO_SALIDA, index=False)
    joblib.dump(scaler, ARCHIVO_SCALER)

    print(f"\n(^o^) Datos normalizados guardados en : {ARCHIVO_SALIDA.name}")
    print(f"(n_n) Scaler guardado en              : {ARCHIVO_SCALER.name}")
    print("   (El scaler se necesita despues para convertir predicciones al valor real)\n")


# Inicio
if __name__ == "__main__":
    normalizar_serie()
