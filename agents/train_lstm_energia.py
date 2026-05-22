import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset

# Semillas
torch.manual_seed(42)
np.random.seed(42)

# Rutas
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "lstm_simple_energia.pth"

# Parametros
INPUT_SIZE = 7
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.2
BATCH_SIZE = 512
EPOCHS = 15
LR = 0.001
PATIENCE = 4
STEP_SUBMUESTREO = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset
class DatasetEnergia(Dataset):
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
            torch.tensor(self.y[i], dtype=torch.float32).unsqueeze(0),
        )


# Modelo
class LSTMSimpleEnergia(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
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
    print("  Cargando datos...")
    X_train = np.load(DATA_DIR / "X_train_energia.npy", mmap_mode="r")
    X_val = np.load(DATA_DIR / "X_val_energia.npy", mmap_mode="r")
    X_test = np.load(DATA_DIR / "X_test_energia.npy", mmap_mode="r")
    y_train = np.load(DATA_DIR / "y_train_energia.npy")
    y_val = np.load(DATA_DIR / "y_val_energia.npy")
    y_test = np.load(DATA_DIR / "y_test_energia.npy")

    dataset_train = DatasetEnergia(X_train, y_train, step=STEP_SUBMUESTREO)
    dataset_val = DatasetEnergia(X_val, y_val, step=1)

    muestras_originales = len(X_train)
    muestras_submuestreadas = len(dataset_train)

    print(f"  Train original   : {muestras_originales:,} muestras")
    print(f"  Train usado      : {muestras_submuestreadas:,} muestras (1 de cada {STEP_SUBMUESTREO})")
    print(f"  Reduccion        : {(1 - muestras_submuestreadas / muestras_originales) * 100:.0f}% menos datos")
    print(f"  Val              : {len(X_val):,} muestras")
    print(f"  Test             : {len(X_test):,} muestras")

    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, X_test, y_test


# Entrenamiento
def entrenar():
    print("=" * 55)
    print("  LSTM SIMPLE - Energia Electrica")
    print(f"  Dispositivo: {DEVICE}")
    print("=" * 55)

    train_loader, val_loader, X_test, y_test = cargar_datos()

    modelo = LSTMSimpleEnergia().to(DEVICE)
    criterio = nn.MSELoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, patience=3, factor=0.5)

    print("\n  Arquitectura:")
    print(f"    Capas LSTM       : {NUM_LAYERS}")
    print(f"    Neuronas         : {HIDDEN_SIZE}")
    print(f"    Batch size       : {BATCH_SIZE}")
    print(f"    Submuestreo      : cada {STEP_SUBMUESTREO} registros")
    print(f"    Max epocas       : {EPOCHS}")
    print(f"    Patience         : {PATIENCE}\n")

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
        print(f"  Epoca [{epoca + 1:2d}/{EPOCHS}]  train={loss_train:.6f}  val={loss_val:.6f}  {indicador}")

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
    dataset_test = DatasetEnergia(X_test, y_test, step=1)
    test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    predicciones, y_real = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(DEVICE)
            predicciones.extend(modelo(Xb).cpu().numpy().flatten())
            y_real.extend(yb.numpy().flatten())

    predicciones = np.array(predicciones)
    y_real = np.array(y_real)

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
