import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
MODEL_PATH = MODELS_DIR / "cnn_lstm_energia.pth"
META_PATH  = MODELS_DIR / "cnn_lstm_energia.json"

# ── Hiperparámetros ──────────────────────────────────────────────
INPUT_SIZE  = 7
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
BATCH_SIZE  = 256
EPOCHS      = 30
LR          = 0.001
PATIENCE    = 8
DROPOUT     = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Modelo CNN-LSTM ──────────────────────────────────────────────
class CNNLSTMEnergia(nn.Module):
    """
    CNN extrae características locales de la secuencia,
    luego LSTM captura dependencias temporales
    """
    def __init__(self):
        super().__init__()
        
        # CNN 1D para extracción de características locales
        # Input shape: (batch, seq_len, features) → (batch, features, seq_len)
        self.conv1 = nn.Conv1d(
            in_channels=INPUT_SIZE, 
            out_channels=32, 
            kernel_size=3, 
            padding=1
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout_cnn = nn.Dropout(0.2)
        
        # LSTM recibe salida de CNN
        self.lstm = nn.LSTM(
            input_size=32,  # canales de salida de CNN
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT
        )
        self.dropout_lstm = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # Permutamos para CNN: (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        
        # CNN
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout_cnn(x)
        
        # Volver a formato LSTM: (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # LSTM
        out, _ = self.lstm(x)
        out = self.dropout_lstm(out[:, -1, :])
        return self.fc(out)

# ── Entrenamiento ────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  ENTRENAMIENTO CNN-LSTM — Energía Eléctrica")
    print(f"  Dispositivo: {DEVICE}")
    print("=" * 60)

    X_train = np.load(DATA_DIR / "X_train_energia.npy")
    X_val   = np.load(DATA_DIR / "X_val_energia.npy")
    X_test  = np.load(DATA_DIR / "X_test_energia.npy")
    y_train = np.load(DATA_DIR / "y_train_energia.npy")
    y_val   = np.load(DATA_DIR / "y_val_energia.npy")
    y_test  = np.load(DATA_DIR / "y_test_energia.npy")

    print(f"X_train : {X_train.shape}")
    print(f"X_val   : {X_val.shape}")
    print(f"X_test  : {X_test.shape}")

    class NumpyDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return (torch.tensor(self.X[idx], dtype=torch.float32),
                    torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0))

    train_loader = DataLoader(NumpyDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader   = DataLoader(NumpyDataset(X_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = CNNLSTMEnergia().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    print(f"\nModelo:\n{model}\n")
    print(f"Iniciando entrenamiento con Early Stopping (patience={PATIENCE})...\n")

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
        print(f"Epoch [{epoch+1:2d}/{EPOCHS}] Train: {train_loss:.6f}  Val: {val_loss:.6f} {mejora}")

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
        "arquitectura": "CNN-LSTM",
        "input_size": INPUT_SIZE,
        "cnn_channels": 32,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Mejor modelo guardado en: {MODEL_PATH}")
    print(f"✅ Metadata guardada en: {META_PATH}")

    print("\n" + "=" * 60)
    print("  EVALUACIÓN FINAL — Conjunto de prueba")
    print("=" * 60)

    model.eval()
    test_loader = DataLoader(NumpyDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    preds, y_real = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(DEVICE)
            preds.extend(model(Xb).cpu().numpy().flatten())
            y_real.extend(yb.numpy().flatten())
    preds  = np.array(preds)
    y_real = np.array(y_real)

    rmse = math.sqrt(mean_squared_error(y_real, preds))
    mae  = mean_absolute_error(y_real, preds)

    mask = y_real != 0
    mape = np.mean(np.abs((y_real[mask] - preds[mask]) / y_real[mask])) * 100

    print(f"RMSE : {rmse:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"MAPE : {mape:.2f}%")

if __name__ == "__main__":
    main()