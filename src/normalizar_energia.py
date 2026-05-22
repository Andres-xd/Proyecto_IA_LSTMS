import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

BASE_DIR    = Path(__file__).resolve().parent.parent

# Intervalo de agregación usado (debe coincidir con limpiar_energia.py)
INTERVALO_AGREGACION = 60  # 15, 30, o 60 minutos

INPUT_FILE  = BASE_DIR / "data" / "processed" / f"energia_agregado_{INTERVALO_AGREGACION}min.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "energia_normalizado.csv"
SCALER_FILE = BASE_DIR / "data" / "processed" / "scaler_energia.pkl"

FEATURE_COLS = [
    "Global_active_power", "Global_reactive_power", "Voltage",
    "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"
]

def main():
    print("=" * 55)
    print("  NORMALIZACIÓN — Dataset Energía")
    print("=" * 55)
    print(f"Archivo de entrada: energia_agregado_{INTERVALO_AGREGACION}min.csv")

    df = pd.read_csv(INPUT_FILE, parse_dates=["datetime"])
    df = df[["datetime"] + FEATURE_COLS].copy()
    print(f"Registros leídos : {len(df):,}")
    print(f"Frecuencia       : {INTERVALO_AGREGACION} minutos")

    # ── CORRECCIÓN: fit solo sobre el 80% inicial (train) ────────
    split = int(len(df) * 0.8)
    scaler = MinMaxScaler()
    scaler.fit(df[FEATURE_COLS].iloc[:split])
    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])

    df.to_csv(OUTPUT_FILE, index=False)
    joblib.dump(scaler, SCALER_FILE)

    print(f"Split aplicado   : {split:,} registros para fit del scaler")
    print(f"Rango Global_active_power: [{df['Global_active_power'].min():.3f}, {df['Global_active_power'].max():.3f}]")
    print(f"\n✅ Normalizado guardado en : {OUTPUT_FILE}")
    print(f"✅ Scaler guardado en      : {SCALER_FILE}\n")

if __name__ == "__main__":
    main()