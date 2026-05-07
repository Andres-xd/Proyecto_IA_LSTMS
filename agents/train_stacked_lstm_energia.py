"""
AGENTE 4 — STACKED LSTM — ENERGÍA ELÉCTRICA
=============================================
¿Qué hace este archivo?
    Entrena una versión más potente del LSTM de energía usando
    múltiples capas apiladas para capturar patrones más complejos
    en el consumo eléctrico.

¿Qué patrones adicionales puede aprender el Stacked LSTM?
    Con una sola capa, el modelo aprende relaciones directas
    entre los últimos 60 minutos y el siguiente.

    Con 3 capas apiladas, puede aprender:
        Capa 1 → variaciones minuto a minuto
        Capa 2 → tendencias dentro de la hora
        Capa 3 → comportamiento general del periodo (mañana/tarde/noche)

    Esto es especialmente útil en consumo eléctrico, donde
    los patrones tienen estructura en múltiples escalas de tiempo.

Optimización de memoria (misma estrategia que LSTM simple):
    - Submuestreo 1 de cada 4 en train
    - Dataset personalizado con carga por demanda
    - mmap_mode para los archivos numpy grandes

Input  → data/processed/X_train_energia.npy  y_train_energia.npy
                         X_val_energia.npy    y_val_energia.npy
                         X_test_energia.npy   y_test_energia.npy
Output → models/stacked_lstm_energia.pth
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import math

# ── Reproducibilidad ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "stacked_lstm_energia.pth"

# ── Hiperparámetros ───────────────────────────────────────────────────────────
INPUT_SIZE       = 7
HIDDEN_SIZE      = 128    # más neuronas por la mayor complejidad del modelo
NUM_LAYERS       = 3      # 3 capas apiladas
DROPOUT          = 0.3
BATCH_SIZE       = 512
EPOCHS           = 15
LR               = 0.001
PATIENCE         = 4
STEP_SUBMUESTREO = 4      # 1 de cada 4 registros en train

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset personalizado ─────────────────────────────────────────────────────
class DatasetEnergia(Dataset):
    """
    Dataset con carga eficiente por demanda y soporte de submuestreo.
    Mismo enfoque que en el LSTM simple para mantener la RAM controlada.
    """
    def __init__(self, X, y, step=1):
        self.indices = np.arange(0, len(X), step)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (
            torch.tensor(self.X[i], dtype=torch.float32),
            torch.tensor(self.y[i], dtype=torch.float32).unsqueeze(0)
        )


# ── Definición del modelo ─────────────────────────────────────────────────────
class StackedLSTMEnergia(nn.Module):
    """
    Stacked LSTM con 3 capas para predicción de consumo eléctrico.

    Flujo de datos:
        x (batch, 60 minutos, 7 variables)
        → LSTM capa 1 → LSTM capa 2 → LSTM capa 3
        → Dropout → Linear → predicción
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = INPUT_SIZE,
            hidden_size = HIDDEN_SIZE,
            num_layers  = NUM_LAYERS,   # 3 capas apiladas
            dropout     = DROPOUT,      # dropout entre capas intermedias
            batch_first = True
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.fc      = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1, :])
        return self.fc(out)


# ── Funciones auxiliares ──────────────────────────────────────────────────────
def calcular_metricas(y_real, predicciones):
    """Calcula RMSE, MAE y MAPE dados los valores reales y predichos."""
    rmse = math.sqrt(mean_squared_error(y_real, predicciones))
    mae  = mean_absolute_error(y_real, predicciones)
    mascara = y_real != 0
    mape    = np.mean(np.abs((y_real[mascara] - predicciones[mascara]) / y_real[mascara])) * 100
    return rmse, mae, mape


def cargar_datos():
    """
    Carga datos con mmap (memory-mapped) para los arrays grandes.
    mmap_mode='r' permite leer el archivo en disco como si fuera RAM,
    sin cargarlo completo en memoria.
    """
    print("  Cargando datos...")
    X_train = np.load(DATA_DIR / "X_train_energia.npy", mmap_mode="r")
    X_val   = np.load(DATA_DIR / "X_val_energia.npy",   mmap_mode="r")
    X_test  = np.load(DATA_DIR / "X_test_energia.npy",  mmap_mode="r")
    y_train = np.load(DATA_DIR / "y_train_energia.npy")
    y_val   = np.load(DATA_DIR / "y_val_energia.npy")
    y_test  = np.load(DATA_DIR / "y_test_energia.npy")

    dataset_train = DatasetEnergia(X_train, y_train, step=STEP_SUBMUESTREO)
    dataset_val   = DatasetEnergia(X_val,   y_val,   step=1)

    print(f"  Train original   : {len(X_train):,} muestras")
    print(f"  Train usado      : {len(dataset_train):,} muestras (1 de cada {STEP_SUBMUESTREO})")
    print(f"  Reduccion        : {(1 - len(dataset_train)/len(X_train))*100:.0f}% menos datos")
    print(f"  Val              : {len(X_val):,} muestras")
    print(f"  Test             : {len(X_test):,} muestras")

    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader   = DataLoader(dataset_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, X_test, y_test


# ── Entrenamiento ─────────────────────────────────────────────────────────────
def entrenar():
    print("=" * 55)
    print("  STACKED LSTM — Energia Electrica")
    print(f"  Dispositivo: {DEVICE}")
    print("=" * 55)

    train_loader, val_loader, X_test, y_test = cargar_datos()

    modelo      = StackedLSTMEnergia().to(DEVICE)
    criterio    = nn.MSELoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=LR, weight_decay=1e-5)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, patience=3, factor=0.5)

    total_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)

    print(f"\n  Arquitectura:")
    print(f"    Capas LSTM       : {NUM_LAYERS}  (stacked)")
    print(f"    Neuronas         : {HIDDEN_SIZE}")
    print(f"    Dropout          : {DROPOUT}")
    print(f"    Parametros       : {total_params:,}")
    print(f"    Batch size       : {BATCH_SIZE}")
    print(f"    Submuestreo      : cada {STEP_SUBMUESTREO} registros")
    print(f"    Max epocas       : {EPOCHS}")
    print(f"    Patience         : {PATIENCE}\n")

    mejor_val_loss    = float("inf")
    epocas_sin_mejora = 0
    mejor_estado      = None

    for epoca in range(EPOCHS):

        # — Entrenamiento —
        modelo.train()
        loss_train = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizador.zero_grad()
            loss = criterio(modelo(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            optimizador.step()
            loss_train += loss.item()
        loss_train /= len(train_loader)

        # — Validación —
        modelo.eval()
        loss_val = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                loss_val += criterio(modelo(Xb), yb).item()
        loss_val /= len(val_loader)

        scheduler.step(loss_val)

        indicador = "<-- mejor" if loss_val < mejor_val_loss else ""
        print(f"  Epoca [{epoca+1:2d}/{EPOCHS}]  train={loss_train:.6f}  val={loss_val:.6f}  {indicador}")

        if loss_val < mejor_val_loss:
            mejor_val_loss    = loss_val
            mejor_estado      = {k: v.clone() for k, v in modelo.state_dict().items()}
            epocas_sin_mejora = 0
        else:
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= PATIENCE:
                print(f"\n  Detenido en epoca {epoca+1} — sin mejora por {PATIENCE} epocas seguidas.")
                break

    modelo.load_state_dict(mejor_estado)
    torch.save(modelo.state_dict(), MODEL_PATH)
    print(f"\n  Modelo guardado en: {MODEL_PATH.name}")

    # — Evaluación final —
    print("\n" + "=" * 55)
    print("  RESULTADO FINAL — Conjunto de prueba (test)")
    print("=" * 55)

    modelo.eval()
    dataset_test = DatasetEnergia(X_test, y_test, step=1)
    test_loader  = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    predicciones, y_real = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(DEVICE)
            predicciones.extend(modelo(Xb).cpu().numpy().flatten())
            y_real.extend(yb.numpy().flatten())

    predicciones = np.array(predicciones)
    y_real       = np.array(y_real)

    rmse, mae, mape = calcular_metricas(y_real, predicciones)

    print(f"\n  RMSE : {rmse:.6f}  (error cuadratico medio)")
    print(f"  MAE  : {mae:.6f}  (error absoluto medio)")
    print(f"  MAPE : {mape:.2f}%  (error porcentual medio)")

    print("\n  Muestra — primeras 10 predicciones vs valores reales:")
    print(f"  {'Real':>10}  {'Prediccion':>10}  {'Error':>10}")
    print(f"  {'-'*34}")
    for i in range(min(10, len(predicciones))):
        print(f"  {y_real[i]:>10.4f}  {predicciones[i]:>10.4f}  {abs(y_real[i]-predicciones[i]):>10.4f}")


# ── Punto de entrada ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    entrenar()