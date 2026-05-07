import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import math

torch.manual_seed(42)
np.random.seed(42)

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "model_bilstm_energia.pth"

INPUT_SIZE  = 7
HIDDEN_SIZE = 32       # x2 efectivo = 64 por bidireccional
DROPOUT     = 0.2
BATCH_SIZE  = 256
EPOCHS      = 20
LR          = 0.001
PATIENCE    = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0))

class BiLSTMEnergia(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm    = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, num_layers=2,
                               dropout=DROPOUT, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc      = nn.Linear(HIDDEN_SIZE * 2, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))

def main():
    print("=" * 55)
    print("  ENTRENAMIENTO — Bidirectional LSTM | Energía")
    print(f"  Dispositivo: {DEVICE}")
    print("=" * 55)

    X_train = np.load(DATA_DIR / "X_train_energia.npy")
    X_val   = np.load(DATA_DIR / "X_val_energia.npy")
    X_test  = np.load(DATA_DIR / "X_test_energia.npy")
    y_train = np.load(DATA_DIR / "y_train_energia.npy")
    y_val   = np.load(DATA_DIR / "y_val_energia.npy")
    y_test  = np.load(DATA_DIR / "y_test_energia.npy")

    print(f"X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

    train_loader = DataLoader(NumpyDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader   = DataLoader(NumpyDataset(X_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = BiLSTMEnergia().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    print(f"\nModelo:\n{model}\n")

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
    print(f"\n✅ Modelo guardado en: {MODEL_PATH}")

    print("\n" + "=" * 55)
    print("  EVALUACIÓN FINAL")
    print("=" * 55)

    model.eval()
    test_loader = DataLoader(NumpyDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    preds, y_real = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            preds.extend(model(Xb.to(DEVICE)).cpu().numpy().flatten())
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