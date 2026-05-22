import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import math
import json

# ── Reproducibilidad ─────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Función auxiliar para encontrar base del proyecto ─────────────
def _encontrar_base():
    p = Path(__file__).resolve()
    for _ in range(5):
        if (p / "data" / "processed").exists():
            return p
        p = p.parent
    raise FileNotFoundError("No se encontró 'data/processed/' en el árbol")

# ── Rutas ────────────────────────────────────────────────────────
BASE_DIR   = _encontrar_base()
DATA_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "bilstm_sismos.pth"
META_PATH  = MODELS_DIR / "bilstm_sismos.json"

# ── Hiperparámetros ──────────────────────────────────────────────
INPUT_SIZE   = 5   # avg_mag, max_mag, avg_depth, count_sismos, hay_actividad
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.3
BATCH_SIZE   = 32
EPOCHS       = 100
LR           = 0.001
PATIENCE     = 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Modelo Bidirectional LSTM ────────────────────────────────────
class BiLSTMSismos(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE, 
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS, 
            dropout=DROPOUT,
            batch_first=True,
            bidirectional=True  # ← bidireccional
        )
        self.dropout = nn.Dropout(DROPOUT)
        # Salida bidireccional: 2*HIDDEN_SIZE
        self.fc = nn.Linear(HIDDEN_SIZE * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

# ── Entrenamiento ────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  ENTRENAMIENTO BIDIRECTIONAL LSTM — Sismología")
    print(f"  Dispositivo: {DEVICE}")
    print("=" * 60)

    X_train = np.load(DATA_DIR / "X_train_sismos.npy")
    X_val   = np.load(DATA_DIR / "X_val_sismos.npy")
    X_test  = np.load(DATA_DIR / "X_test_sismos.npy")
    y_train = np.load(DATA_DIR / "y_train_sismos.npy")
    y_val   = np.load(DATA_DIR / "y_val_sismos.npy")
    y_test  = np.load(DATA_DIR / "y_test_sismos.npy")

    print(f"X_train : {X_train.shape}")
    print(f"X_val   : {X_val.shape}")
    print(f"X_test  : {X_test.shape}")

    def to_loader(X, y, shuffle=False):
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        return DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = to_loader(X_train, y_train)
    val_loader   = to_loader(X_val,   y_val)

    model     = BiLSTMSismos().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print(f"\nModelo:\n{model}\n")
    print("Iniciando entrenamiento con Early Stopping...\n")

    best_val_loss = float("inf")
    epochs_sin_mejora = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(Xb), yb).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        mejora = "⬇" if val_loss < best_val_loss else " "
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] Train: {train_loss:.6f}  Val: {val_loss:.6f} {mejora}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_sin_mejora = 0
        else:
            epochs_sin_mejora += 1
            if epochs_sin_mejora >= PATIENCE:
                print(f"\n⏹ Early stopping en época {epoch+1}")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_PATH)
    
    # Guardar metadata de arquitectura
    metadata = {
        "arquitectura": "BiLSTM",
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "bidirectional": True
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Mejor modelo guardado en: {MODEL_PATH}")
    print(f"✅ Metadata guardada en: {META_PATH}")

    print("\n" + "=" * 60)
    print("  EVALUACIÓN FINAL — Conjunto de prueba")
    print("=" * 60)

    model.eval()
    X_te_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds = model(X_te_t).cpu().numpy().flatten()
    y_real = y_test.flatten()

    rmse = math.sqrt(mean_squared_error(y_real, preds))
    mae  = mean_absolute_error(y_real, preds)

    mask = y_real != 0
    mape = np.mean(np.abs((y_real[mask] - preds[mask]) / y_real[mask])) * 100

    print(f"RMSE : {rmse:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"MAPE : {mape:.2f}%")

    print("\nPrimeras 10 predicciones vs reales:")
    for i in range(min(10, len(preds))):
        print(f"  Real: {y_real[i]:.4f} | Pred: {preds[i]:.4f} | Error: {abs(y_real[i]-preds[i]):.4f}")

if __name__ == "__main__":
    main()