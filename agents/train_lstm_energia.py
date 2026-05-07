"""
AGENTE 3 — LSTM SIMPLE — ENERGÍA ELÉCTRICA
============================================
¿Qué hace este archivo?
    Entrena una red LSTM para predecir el consumo eléctrico del
    siguiente minuto basándose en la última hora de mediciones.

¿Por qué este dataset es diferente al de sismos?
    El dataset de energía tiene ~2 millones de registros.
    Cargar todo en memoria RAM y entrenar directamente causaría
    que el proceso se trabe o se caiga. Por eso aplicamos dos
    estrategias de optimización:

ESTRATEGIA 1 — Submuestreo inteligente:
    En lugar de usar los 1.4 millones de muestras de train,
    tomamos 1 de cada 4 registros (~350,000 muestras).
    ¿Por qué funciona? Los datos son minutales y consecutivos.
    Los patrones de consumo eléctrico cambian lentamente.
    Tomar cada 4to minuto no pierde información relevante
    pero reduce la memoria en un 75%.

ESTRATEGIA 2 — Carga por lotes con generador:
    En vez de convertir todo a tensores de PyTorch de una vez
    (lo que duplica el uso de memoria), usamos un Dataset
    personalizado que convierte cada muestra al momento de
    necesitarla. Esto mantiene el uso de RAM constante
    sin importar el tamaño del dataset.

Input  → data/processed/X_train_energia.npy  y_train_energia.npy
                         X_val_energia.npy    y_val_energia.npy
                         X_test_energia.npy   y_test_energia.npy
Output → models/lstm_simple_energia.pth
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
MODEL_PATH = MODELS_DIR / "lstm_simple_energia.pth"

# ── Hiperparámetros ───────────────────────────────────────────────────────────
INPUT_SIZE    = 7      # variables: active_power, reactive_power, voltage, intensity, sub1, sub2, sub3
HIDDEN_SIZE   = 64     # neuronas LSTM
NUM_LAYERS    = 1      # una sola capa (LSTM simple)
DROPOUT       = 0.2
BATCH_SIZE    = 512    # batch grande para aprovechar mejor el hardware
EPOCHS        = 15     # menos épocas porque hay muchos datos incluso submuestreados
LR            = 0.001
PATIENCE      = 4      # early stopping más agresivo

# Submuestreo: tomamos 1 de cada N registros del conjunto de train
# para reducir el tiempo de entrenamiento sin perder los patrones generales
STEP_SUBMUESTREO = 4   # 1 de cada 4 → ~75% menos datos, ~75% menos tiempo

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset personalizado (carga eficiente en memoria) ────────────────────────
class DatasetEnergia(Dataset):
    """
    Dataset que carga los datos por demanda en lugar de todos de una vez.

    ¿Por qué hacer esto?
        Si hacemos torch.tensor(X_train) directamente con 1.4M registros,
        Python necesita tener X_train (numpy) Y el tensor en memoria al
        mismo tiempo, duplicando el uso de RAM.

        Con este Dataset personalizado, cada muestra se convierte a tensor
        solo cuando el DataLoader la necesita. La memoria RAM se mantiene
        constante y controlada.
    """
    def __init__(self, X, y, step=1):
        # Los índices que vamos a usar (submuestreo si step > 1)
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
class LSTMSimpleEnergia(nn.Module):
    """
    Arquitectura LSTM simple de una sola capa para predicción de energía.

    Flujo de datos:
        x (batch, 60 minutos, 7 variables)
        → LSTM → Dropout → Linear → predicción del minuto siguiente
    """
    def __init__(self):
        super().__init__()
        self.lstm    = nn.LSTM(
            input_size  = INPUT_SIZE,
            hidden_size = HIDDEN_SIZE,
            num_layers  = NUM_LAYERS,
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
    Carga los arrays y construye los DataLoaders.
    El conjunto de train se submuestrea para reducir uso de memoria.
    Val y Test se cargan completos (son más pequeños).
    """
    print("  Cargando datos...")
    X_train = np.load(DATA_DIR / "X_train_energia.npy", mmap_mode="r")  # mmap: no carga todo en RAM
    X_val   = np.load(DATA_DIR / "X_val_energia.npy",   mmap_mode="r")
    X_test  = np.load(DATA_DIR / "X_test_energia.npy",  mmap_mode="r")
    y_train = np.load(DATA_DIR / "y_train_energia.npy")
    y_val   = np.load(DATA_DIR / "y_val_energia.npy")
    y_test  = np.load(DATA_DIR / "y_test_energia.npy")

    # Submuestreo solo en train
    dataset_train = DatasetEnergia(X_train, y_train, step=STEP_SUBMUESTREO)
    dataset_val   = DatasetEnergia(X_val,   y_val,   step=1)

    muestras_originales    = len(X_train)
    muestras_submuestreadas = len(dataset_train)

    print(f"  Train original   : {muestras_originales:,} muestras")
    print(f"  Train usado      : {muestras_submuestreadas:,} muestras (1 de cada {STEP_SUBMUESTREO})")
    print(f"  Reduccion        : {(1 - muestras_submuestreadas/muestras_originales)*100:.0f}% menos datos")
    print(f"  Val              : {len(X_val):,} muestras")
    print(f"  Test             : {len(X_test):,} muestras")

    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader   = DataLoader(dataset_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, X_test, y_test


# ── Entrenamiento ─────────────────────────────────────────────────────────────
def entrenar():
    print("=" * 55)
    print("  LSTM SIMPLE — Energia Electrica")
    print(f"  Dispositivo: {DEVICE}")
    print("=" * 55)

    train_loader, val_loader, X_test, y_test = cargar_datos()

    modelo      = LSTMSimpleEnergia().to(DEVICE)
    criterio    = nn.MSELoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=LR, weight_decay=1e-5)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, patience=3, factor=0.5)

    print(f"\n  Arquitectura:")
    print(f"    Capas LSTM       : {NUM_LAYERS}  (simple = 1 capa)")
    print(f"    Neuronas         : {HIDDEN_SIZE}")
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