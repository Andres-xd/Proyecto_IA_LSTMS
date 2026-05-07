"""
AGENTE 1 — LSTM SIMPLE — SISMOLOGÍA
=====================================
¿Qué hace este archivo?
    Entrena una red neuronal LSTM para predecir la magnitud promedio
    del día siguiente basándose en los últimos 14 días de actividad
    sísmica.

¿Qué es un LSTM simple?
    Es la arquitectura base. Una sola capa LSTM que lee la secuencia
    de días y al final produce una predicción. Es el punto de partida
    antes de probar arquitecturas más complejas.

    Entrada → [14 días × 5 variables] → LSTM → Dropout → Predicción

¿Qué es Early Stopping?
    Es una técnica para evitar que el modelo "memorice" los datos
    de entrenamiento en lugar de aprender patrones generales.
    Si después de N épocas el modelo no mejora en validación,
    se detiene solo y guarda la mejor versión encontrada.

Métricas que reporta:
    - RMSE  → error cuadrático medio (penaliza errores grandes)
    - MAE   → error absoluto medio (más intuitivo, misma unidad)
    - MAPE  → error porcentual medio (qué tan lejos en %)

Input  → data/processed/X_train_sismos.npy  y_train_sismos.npy
                         X_val_sismos.npy    y_val_sismos.npy
                         X_test_sismos.npy   y_test_sismos.npy
Output → models/lstm_simple_sismos.pth
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import math

# ── Reproducibilidad ──────────────────────────────────────────────────────────
# Fijar semillas para que los resultados sean iguales cada vez que se corra
torch.manual_seed(42)
np.random.seed(42)

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "lstm_simple_sismos.pth"

# ── Hiperparámetros ───────────────────────────────────────────────────────────
INPUT_SIZE  = 5      # variables por día: avg_mag, max_mag, avg_depth, count_sismos, hay_actividad
HIDDEN_SIZE = 64     # neuronas en la capa LSTM (capacidad del modelo)
NUM_LAYERS  = 1      # una sola capa LSTM (eso lo hace "simple")
DROPOUT     = 0.2    # apaga el 20% de neuronas al azar para evitar sobreajuste
BATCH_SIZE  = 64     # cuántas muestras procesa el modelo a la vez
EPOCHS      = 100    # máximo de vueltas de entrenamiento
LR          = 0.001  # tasa de aprendizaje (qué tan rápido ajusta sus pesos)
PATIENCE    = 15     # épocas sin mejora antes de parar (early stopping)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Definición del modelo ─────────────────────────────────────────────────────
class LSTMSimpleSismos(nn.Module):
    """
    Arquitectura LSTM simple de una sola capa.

    Flujo de datos:
        x (batch, 14 días, 5 variables)
        → LSTM  → procesa la secuencia temporal
        → Dropout → apaga neuronas al azar (regularización)
        → Linear  → colapsa a un solo número (la predicción)
    """
    def __init__(self):
        super().__init__()
        self.lstm    = nn.LSTM(
            input_size  = INPUT_SIZE,
            hidden_size = HIDDEN_SIZE,
            num_layers  = NUM_LAYERS,
            dropout     = 0.0,          # dropout solo aplica entre capas, con 1 capa va aquí abajo
            batch_first = True          # formato: (batch, tiempo, features)
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.fc      = nn.Linear(HIDDEN_SIZE, 1)   # capa de salida: 1 valor = la predicción

    def forward(self, x):
        out, _ = self.lstm(x)           # procesar toda la secuencia
        out    = self.dropout(out[:, -1, :])  # tomar solo el último paso de tiempo
        return self.fc(out)             # producir la predicción final


# ── Funciones auxiliares ──────────────────────────────────────────────────────
def calcular_metricas(y_real, predicciones):
    """Calcula RMSE, MAE y MAPE dados los valores reales y predichos."""
    rmse = math.sqrt(mean_squared_error(y_real, predicciones))
    mae  = mean_absolute_error(y_real, predicciones)

    # MAPE: evitamos dividir entre cero filtrando valores reales = 0
    mascara = y_real != 0
    mape    = np.mean(np.abs((y_real[mascara] - predicciones[mascara]) / y_real[mascara])) * 100

    return rmse, mae, mape


def cargar_datos():
    """Carga los arrays numpy y los convierte a DataLoaders de PyTorch."""
    X_train = np.load(DATA_DIR / "X_train_sismos.npy")
    X_val   = np.load(DATA_DIR / "X_val_sismos.npy")
    X_test  = np.load(DATA_DIR / "X_test_sismos.npy")
    y_train = np.load(DATA_DIR / "y_train_sismos.npy")
    y_val   = np.load(DATA_DIR / "y_val_sismos.npy")
    y_test  = np.load(DATA_DIR / "y_test_sismos.npy")

    def a_loader(X, y, mezclar=False):
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        return DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=mezclar)

    return (
        a_loader(X_train, y_train, mezclar=False),
        a_loader(X_val,   y_val),
        X_test, y_test
    )


# ── Entrenamiento ─────────────────────────────────────────────────────────────
def entrenar():
    print("=" * 55)
    print("  LSTM SIMPLE — Sismologia")
    print(f"  Dispositivo: {DEVICE}")
    print("=" * 55)

    train_loader, val_loader, X_test, y_test = cargar_datos()

    modelo    = LSTMSimpleSismos().to(DEVICE)
    criterio  = nn.MSELoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=LR, weight_decay=1e-5)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, patience=5, factor=0.5)

    print(f"\n  Arquitectura:")
    print(f"    Capas LSTM  : {NUM_LAYERS}  (simple = 1 capa)")
    print(f"    Neuronas    : {HIDDEN_SIZE}")
    print(f"    Dropout     : {DROPOUT}")
    print(f"    Batch size  : {BATCH_SIZE}")
    print(f"    Max epocas  : {EPOCHS}  (con early stopping, puede parar antes)")
    print(f"    Patience    : {PATIENCE} epocas sin mejora\n")

    mejor_val_loss     = float("inf")
    epocas_sin_mejora  = 0
    mejor_estado       = None

    for epoca in range(EPOCHS):

        # — Fase de entrenamiento —
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

        # — Fase de validación —
        modelo.eval()
        loss_val = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                loss_val += criterio(modelo(Xb), yb).item()
        loss_val /= len(val_loader)

        scheduler.step(loss_val)

        indicador = "<-- mejor" if loss_val < mejor_val_loss else ""
        print(f"  Epoca [{epoca+1:3d}/{EPOCHS}]  train={loss_train:.6f}  val={loss_val:.6f}  {indicador}")

        # — Early stopping —
        if loss_val < mejor_val_loss:
            mejor_val_loss    = loss_val
            mejor_estado      = {k: v.clone() for k, v in modelo.state_dict().items()}
            epocas_sin_mejora = 0
        else:
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= PATIENCE:
                print(f"\n  Detenido en epoca {epoca+1} — sin mejora por {PATIENCE} epocas seguidas.")
                break

    # — Guardar el mejor modelo encontrado —
    modelo.load_state_dict(mejor_estado)
    torch.save(modelo.state_dict(), MODEL_PATH)
    print(f"\n  Modelo guardado en: {MODEL_PATH.name}")

    # — Evaluación final en el conjunto de test —
    print("\n" + "=" * 55)
    print("  RESULTADO FINAL — Conjunto de prueba (test)")
    print("=" * 55)

    modelo.eval()
    X_te_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        predicciones = modelo(X_te_t).cpu().numpy().flatten()
    y_real = y_test.flatten()

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