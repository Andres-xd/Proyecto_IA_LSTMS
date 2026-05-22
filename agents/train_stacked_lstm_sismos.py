import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

# Semillas
torch.manual_seed(42)
np.random.seed(42)

# Rutas
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "stacked_lstm_sismos.pth"

# Parametros
INPUT_SIZE = 5
HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 100
LR = 0.001
PATIENCE = 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Modelo
class StackedLSTMSismos(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


# Metricas
def calcular_metricas(y_real, predicciones):
    rmse = math.sqrt(mean_squared_error(y_real, predicciones))
    mae = mean_absolute_error(y_real, predicciones)
    mascara = y_real != 0
    mape = np.mean(np.abs((y_real[mascara] - predicciones[mascara]) / y_real[mascara])) * 100
    return rmse, mae, mape


# Datos
def cargar_datos():
    X_train = np.load(DATA_DIR / "X_train_sismos.npy")
    X_val = np.load(DATA_DIR / "X_val_sismos.npy")
    X_test = np.load(DATA_DIR / "X_test_sismos.npy")
    y_train = np.load(DATA_DIR / "y_train_sismos.npy")
    y_val = np.load(DATA_DIR / "y_val_sismos.npy")
    y_test = np.load(DATA_DIR / "y_test_sismos.npy")

    def a_loader(X, y, mezclar=False):
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        return DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=mezclar)

    return (
        a_loader(X_train, y_train, mezclar=False),
        a_loader(X_val, y_val),
        X_test,
        y_test,
    )


# Entrenamiento
def entrenar():
    print("=" * 55)
    print("  STACKED LSTM - Sismologia")
    print(f"  Dispositivo: {DEVICE}")
    print("=" * 55)

    train_loader, val_loader, X_test, y_test = cargar_datos()

    modelo = StackedLSTMSismos().to(DEVICE)
    criterio = nn.MSELoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, patience=5, factor=0.5)

    total_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)

    print("\n  Arquitectura:")
    print(f"    Capas LSTM  : {NUM_LAYERS}")
    print(f"    Neuronas    : {HIDDEN_SIZE}")
    print(f"    Dropout     : {DROPOUT}")
    print(f"    Parametros  : {total_params:,}")
    print(f"    Batch size  : {BATCH_SIZE}")
    print(f"    Max epocas  : {EPOCHS}")
    print(f"    Patience    : {PATIENCE}\n")

    mejor_val_loss = float("inf")
    epocas_sin_mejora = 0
    mejor_estado = None

    for epoca in range(EPOCHS):
        # Ajuste
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

        # Validacion
        modelo.eval()
        loss_val = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                loss_val += criterio(modelo(Xb), yb).item()
        loss_val /= len(val_loader)

        scheduler.step(loss_val)

        indicador = "<-- mejor" if loss_val < mejor_val_loss else ""
        print(f"  Epoca [{epoca + 1:3d}/{EPOCHS}]  train={loss_train:.6f}  val={loss_val:.6f}  {indicador}")

        if loss_val < mejor_val_loss:
            mejor_val_loss = loss_val
            mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
            epocas_sin_mejora = 0
        else:
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= PATIENCE:
                print(f"\n  Detenido en epoca {epoca + 1} por falta de mejora.")
                break

    modelo.load_state_dict(mejor_estado)
    torch.save(modelo.state_dict(), MODEL_PATH)
    print(f"\n  Modelo guardado en: {MODEL_PATH.name}")

    # Evaluacion
    print("\n" + "=" * 55)
    print("  RESULTADO FINAL - Conjunto de prueba")
    print("=" * 55)

    modelo.eval()
    X_te_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        predicciones = modelo(X_te_t).cpu().numpy().flatten()
    y_real = y_test.flatten()

    rmse, mae, mape = calcular_metricas(y_real, predicciones)

    print(f"\n  RMSE : {rmse:.6f}")
    print(f"  MAE  : {mae:.6f}")
    print(f"  MAPE : {mape:.2f}%")

    print("\n  Muestra - primeras 10 predicciones vs valores reales:")
    print(f"  {'Real':>10}  {'Prediccion':>10}  {'Error':>10}")
    print(f"  {'-' * 34}")
    for i in range(min(10, len(predicciones))):
        print(f"  {y_real[i]:>10.4f}  {predicciones[i]:>10.4f}  {abs(y_real[i] - predicciones[i]):>10.4f}")


# Inicio
if __name__ == "__main__":
    entrenar()
