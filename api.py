"""
API Flask — Sistema LSTM de Predicción de Series de Tiempo
Corregido: subsampling energía, log de épocas, info de división de datos
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math, threading

app = Flask(__name__)
CORS(app)

# Catálogo de modelos disponibles
CATALOGO_MODELOS = {
    "model_energia":        {"nombre": "LSTM Simple", "arquitectura": "LSTMSimple", "dataset": "energia"},
    "model_sismos":         {"nombre": "LSTM Simple", "arquitectura": "LSTMSimple", "dataset": "sismos"},
    "stacked_lstm_energia": {"nombre": "Stacked LSTM", "arquitectura": "LSTMSimple", "dataset": "energia"},
    "stacked_lstm_sismos":  {"nombre": "Stacked LSTM", "arquitectura": "LSTMSimple", "dataset": "sismos"},
    "bilstm_energia":       {"nombre": "Bidirectional LSTM", "arquitectura": "BiLSTM", "dataset": "energia"},
    "bilstm_sismos":        {"nombre": "Bidirectional LSTM", "arquitectura": "BiLSTM", "dataset": "sismos"},
    "cnn_lstm_energia":     {"nombre": "CNN-LSTM Híbrido", "arquitectura": "CNNLSTM", "dataset": "energia"},
    "cnn_lstm_sismos":      {"nombre": "CNN-LSTM Híbrido", "arquitectura": "CNNLSTM", "dataset": "sismos"},
}

BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Subsampling para prototipo de energía (evita OOM y demoras)
ENERGIA_MAX_TRAIN = 25_000
ENERGIA_MAX_VAL   = 5_000
ENERGIA_MAX_TEST  = 5_000

training_state = {
    "running": False, "epoch": 0, "total_epochs": 0,
    "train_loss": 0.0, "val_loss": 0.0, "best_val_loss": 999.0,
    "done": False, "metrics": {}, "epoch_log": [],
    "error": None, "dataset_info": {}, "pred_table": [], "forecast": [],
}

class LSTMSimple(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=0.2, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc      = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=0.3, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(hidden_size * 2, 1)  # x2 por bidireccional
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))

class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool1d(2)
        self.dropout_cnn = nn.Dropout(0.2)
        self.lstm    = nn.LSTM(32, hidden_size, num_layers=num_layers, dropout=0.3, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, features, seq)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout_cnn(x)
        x = x.permute(0, 2, 1)  # (batch, seq, features)
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))

class NumpyDataset(Dataset):
    def __init__(self, X, y): self.X = X; self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0))

def calcular_metricas(y_real, preds):
    rmse = math.sqrt(mean_squared_error(y_real, preds))
    mae  = mean_absolute_error(y_real, preds)
    mask = y_real != 0
    mape = float(np.mean(np.abs((y_real[mask]-preds[mask])/y_real[mask]))*100) if mask.sum()>0 else 0.0
    return {"rmse": round(rmse,6), "mae": round(mae,6), "mape": round(mape,2)}

def entrenar_modelo(dataset_type, config):
    global training_state
    torch.manual_seed(42); np.random.seed(42)
    try:
        training_state.update({
            "running":True,"done":False,"epoch":0,"error":None,
            "metrics":{},"epoch_log":[],"dataset_info":{},"pred_table":[],"forecast":[],
        })
        epochs     = int(config.get("epochs", 20))
        batch_size = int(config.get("batch_size", 64))
        model_name = config.get("model_name", f"model_{dataset_type}_custom")
        patience   = 10
        training_state["total_epochs"] = epochs

        suffix = "energia" if dataset_type == "energia" else "sismos"
        X_train_full = np.load(DATA_DIR / f"X_train_{suffix}.npy")
        X_val_full   = np.load(DATA_DIR / f"X_val_{suffix}.npy")
        X_test_full  = np.load(DATA_DIR / f"X_test_{suffix}.npy")
        y_train_full = np.load(DATA_DIR / f"y_train_{suffix}.npy")
        y_val_full   = np.load(DATA_DIR / f"y_val_{suffix}.npy")
        y_test_full  = np.load(DATA_DIR / f"y_test_{suffix}.npy")

        if dataset_type == "energia":
            X_train,y_train = X_train_full[:ENERGIA_MAX_TRAIN], y_train_full[:ENERGIA_MAX_TRAIN]
            X_val,  y_val   = X_val_full[:ENERGIA_MAX_VAL],     y_val_full[:ENERGIA_MAX_VAL]
            X_test, y_test  = X_test_full[:ENERGIA_MAX_TEST],   y_test_full[:ENERGIA_MAX_TEST]
            nota = f"(prototipo: {ENERGIA_MAX_TRAIN:,} de {len(X_train_full):,} muestras totales)"
        else:
            X_train,y_train = X_train_full, y_train_full
            X_val,  y_val   = X_val_full,   y_val_full
            X_test, y_test  = X_test_full,  y_test_full
            nota = "(dataset completo)"

        training_state["dataset_info"] = {
            "train": {"muestras":len(X_train),"porcentaje":"70%",
                      "descripcion":"Datos con los que el modelo APRENDE a reconocer patrones temporales.","uso":"Entrenamiento"},
            "val":   {"muestras":len(X_val),"porcentaje":"15%",
                      "descripcion":"Datos que el modelo NO ve durante el entrenamiento. Sirven para detectar si está memorizando en vez de aprender.","uso":"Validación"},
            "test":  {"muestras":len(X_test),"porcentaje":"15%",
                      "descripcion":"Datos completamente nuevos usados SOLO al final para medir qué tan bien predice el modelo en situaciones reales.","uso":"Prueba final"},
            "nota":nota, "window_size":X_train.shape[1], "features":X_train.shape[2],
        }

        input_size  = X_train.shape[2]
        hidden_size = 128 if dataset_type=="sismos" else 64
        
        # Seleccionar arquitectura según configuración
        agent_type = config.get("agent_type", "lstm_simple")
        
        if agent_type == "bilstm":
            model = BiLSTM(input_size, hidden_size)
            arch_name = "Bidirectional LSTM"
        elif agent_type == "cnn_lstm":
            model = CNNLSTM(input_size, hidden_size)
            arch_name = "CNN-LSTM Híbrido"
        elif agent_type == "stacked_lstm":
            model = LSTMSimple(input_size, hidden_size, num_layers=3)
            arch_name = "Stacked LSTM (3 capas)"
        else:  # lstm_simple por defecto
            model = LSTMSimple(input_size, hidden_size)
            arch_name = "LSTM Simple"
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        train_loader = DataLoader(NumpyDataset(X_train,y_train), batch_size=batch_size, shuffle=False)
        val_loader   = DataLoader(NumpyDataset(X_val,  y_val),   batch_size=batch_size, shuffle=False)

        best_val=float("inf"); best_state=None; no_mejora=0
        training_state["epoch_log"] += [
            f"Iniciando — {len(X_train):,} muestras de entrenamiento {nota}",
            f"Arquitectura: {arch_name} | Neuronas ocultas: {hidden_size}",
            "─"*55,
        ]

        for epoch in range(epochs):
            model.train(); tl=0.0
            for Xb,yb in train_loader:
                optimizer.zero_grad()
                loss=criterion(model(Xb),yb)
                loss.backward(); optimizer.step()
                tl+=loss.item()
            tl/=len(train_loader)

            model.eval(); vl=0.0
            with torch.no_grad():
                for Xb,yb in val_loader:
                    vl+=criterion(model(Xb),yb).item()
            vl/=len(val_loader)
            scheduler.step(vl)

            mejora = " ⬇ MEJORA" if vl<best_val else ""
            log_line = f"Época [{epoch+1:3d}/{epochs}]  Train: {tl:.6f}  Val: {vl:.6f}{mejora}"
            training_state.update({"epoch":epoch+1,"train_loss":round(tl,6),"val_loss":round(vl,6),"best_val_loss":round(best_val,6)})
            training_state["epoch_log"].append(log_line)
            if len(training_state["epoch_log"])>100:
                training_state["epoch_log"]=training_state["epoch_log"][-100:]

            if vl<best_val:
                best_val=vl; best_state={k:v.clone() for k,v in model.state_dict().items()}; no_mejora=0
            else:
                no_mejora+=1
                if no_mejora>=patience:
                    training_state["epoch_log"].append(f"⏹  Early stopping — sin mejora por {patience} épocas consecutivas")
                    break

        model.load_state_dict(best_state)
        model_path=MODELS_DIR/f"{model_name}.pth"
        torch.save(model.state_dict(), model_path)
        training_state["epoch_log"]+=[f"─"*55, f"✅ Modelo guardado: {model_name}.pth", "Evaluando sobre conjunto de prueba..."]

        model.eval()
        test_loader=DataLoader(NumpyDataset(X_test,y_test),batch_size=batch_size,shuffle=False)
        preds,y_real_list=[],[]
        with torch.no_grad():
            for Xb,yb in test_loader:
                preds.extend(model(Xb).cpu().numpy().flatten())
                y_real_list.extend(yb.numpy().flatten())
        preds=np.array(preds); y_real=np.array(y_real_list)
        metricas=calcular_metricas(y_real,preds)
        training_state["epoch_log"].append(f"RMSE: {metricas['rmse']}  |  MAE: {metricas['mae']}  |  MAPE: {metricas['mape']}%")

        pred_table=[{"n":i+1,"real":round(float(y_real[i]),4),"pred":round(float(preds[i]),4),"error":round(abs(float(y_real[i])-float(preds[i])),4)} for i in range(min(20,len(preds)))]

        forecast=[]
        if dataset_type=="sismos":
            ventana=X_test[-1].copy()
            with torch.no_grad():
                for d in range(30):
                    inp=torch.tensor(ventana[np.newaxis],dtype=torch.float32)
                    pred=model(inp).item()
                    forecast.append({"dia":d+1,"valor":round(pred,4)})
                    nueva=ventana[-1].copy(); nueva[0]=pred
                    ventana=np.vstack([ventana[1:],nueva])

        training_state.update({"running":False,"done":True,"metrics":metricas,"pred_table":pred_table,"forecast":forecast,"model_saved":f"{model_name}.pth"})

    except Exception as e:
        import traceback
        training_state.update({"running":False,"done":True,"error":str(e),"epoch_log":training_state.get("epoch_log",[])+[f"ERROR: {e}",traceback.format_exc()]})

@app.route("/api/status")
def status(): return jsonify({"ok":True})

@app.route("/api/dataset-info/<dataset_type>")
def dataset_info(dataset_type):
    info = {
        "energia":{"nombre":"Consumo Eléctrico Doméstico","fuente":"UCI ML Repository","registros":"2,075,259","periodo":"Dic 2006 — Nov 2010","frecuencia":"1 medición por minuto","features":["Global_active_power","Global_reactive_power","Voltage","Global_intensity","Sub_metering_1","Sub_metering_2","Sub_metering_3"],"target":"Global_active_power","descripcion":"Mediciones de consumo eléctrico de un hogar en Sceaux, Francia. Permite predecir la demanda energética futura para optimizar la distribución en redes eléctricas inteligentes.","window_size":60,"color":"#F59E0B"},
        "sismos":{"nombre":"Actividad Sísmica — Guatemala","fuente":"USGS Earthquake Catalog","registros":"3,586 eventos (1970–2026)","periodo":"Ene 1970 — Mar 2026","frecuencia":"Serie diaria agregada","features":["avg_mag","max_mag","avg_depth","count_sismos","hay_actividad"],"target":"avg_mag (magnitud promedio diaria)","descripcion":"Registro de actividad sísmica en Centroamérica. Permite analizar patrones sísmicos útiles para sistemas de monitoreo en regiones de alta actividad geológica como Guatemala.","window_size":14,"color":"#EF4444"},
    }
    if dataset_type not in info: return jsonify({"error":"Dataset no encontrado"}),404
    return jsonify(info[dataset_type])

@app.route("/api/models")
def list_models():
    models = []
    if MODELS_DIR.exists():
        for f in MODELS_DIR.glob("*.pth"):
            # Detectar dataset del nombre del archivo
            if "energia" in f.stem:
                dataset_auto = "energia"
            elif "sismos" in f.stem or "seismic" in f.stem:
                dataset_auto = "sismos"
            else:
                dataset_auto = "unknown"
            
            # Buscar en catálogo o usar valores por defecto
            info = CATALOGO_MODELOS.get(f.stem, {
                "nombre": f.stem.replace("_", " ").title(),
                "arquitectura": "LSTMSimple",
                "dataset": dataset_auto
            })
            
            models.append({
                "name": f.stem,
                "file": f.name,
                "size_kb": round(f.stat().st_size / 1024, 1),
                "display_name": info["nombre"],
                "architecture": info["arquitectura"],
                "dataset": info["dataset"]
            })
    return jsonify(models)

@app.route("/api/train",methods=["POST"])
def train():
    if training_state["running"]: return jsonify({"error":"Ya hay un entrenamiento en curso"}),400
    data=request.json; dataset_type=data.get("dataset")
    if dataset_type not in ["energia","sismos"]: return jsonify({"error":"dataset inválido"}),400
    config={"epochs":data.get("epochs",20),"batch_size":data.get("batch_size",64),"model_name":data.get("model_name",f"model_{dataset_type}")}
    threading.Thread(target=entrenar_modelo,args=(dataset_type,config),daemon=True).start()
    return jsonify({"ok":True})

@app.route("/api/train/progress")
def train_progress(): return jsonify(training_state)

@app.route("/api/load-model",methods=["POST"])
def load_and_predict():
    data=request.json; dataset_type=data.get("dataset"); model_name=data.get("model_name")
    model_path=MODELS_DIR/f"{model_name}.pth"
    if not model_path.exists(): return jsonify({"error":f"Modelo {model_name}.pth no encontrado"}),404
    
    # Detectar arquitectura del modelo
    info = CATALOGO_MODELOS.get(model_name, {"nombre": "Modelo custom", "arquitectura": "LSTMSimple"})
    arquitectura = info["arquitectura"]
    
    suffix="energia" if dataset_type=="energia" else "sismos"
    X_train_full=np.load(DATA_DIR/f"X_train_{suffix}.npy")
    X_val_full  =np.load(DATA_DIR/f"X_val_{suffix}.npy")
    X_test_full =np.load(DATA_DIR/f"X_test_{suffix}.npy")
    y_test_full =np.load(DATA_DIR/f"y_test_{suffix}.npy")
    if dataset_type=="energia":
        X_test=X_test_full[:ENERGIA_MAX_TEST]; y_test=y_test_full[:ENERGIA_MAX_TEST]
        n_train=ENERGIA_MAX_TRAIN; n_val=ENERGIA_MAX_VAL; nota=f"(prototipo: {ENERGIA_MAX_TRAIN:,} de {len(X_train_full):,} totales)"
    else:
        X_test=X_test_full; y_test=y_test_full
        n_train=len(X_train_full); n_val=len(X_val_full); nota="(dataset completo)"
    
    input_size=X_test.shape[2]; hidden_size=128 if dataset_type=="sismos" else 64
    
    # Instanciar modelo según arquitectura
    if arquitectura == "BiLSTM":
        model = BiLSTM(input_size, hidden_size)
    elif arquitectura == "CNNLSTM":
        model = CNNLSTM(input_size, hidden_size)
    else:
        model = LSTMSimple(input_size, hidden_size)
    
    model.load_state_dict(torch.load(model_path,map_location="cpu")); model.eval()
    test_loader=DataLoader(NumpyDataset(X_test,y_test),batch_size=64,shuffle=False)
    preds,y_real_list=[],[]
    with torch.no_grad():
        for Xb,yb in test_loader:
            preds.extend(model(Xb).cpu().numpy().flatten()); y_real_list.extend(yb.numpy().flatten())
    preds=np.array(preds); y_real=np.array(y_real_list)
    metricas=calcular_metricas(y_real,preds)
    pred_table=[{"n":i+1,"real":round(float(y_real[i]),4),"pred":round(float(preds[i]),4),"error":round(abs(float(y_real[i])-float(preds[i])),4)} for i in range(min(20,len(preds)))]
    forecast=[]
    if dataset_type=="sismos":
        ventana=X_test[-1].copy()
        with torch.no_grad():
            for d in range(30):
                inp=torch.tensor(ventana[np.newaxis],dtype=torch.float32); pred=model(inp).item()
                forecast.append({"dia":d+1,"valor":round(pred,4)})
                nueva=ventana[-1].copy(); nueva[0]=pred; ventana=np.vstack([ventana[1:],nueva])
    return jsonify({
        "metrics":metricas,"pred_table":pred_table,"forecast":forecast,
        "dataset_info":{"train":{"muestras":n_train,"porcentaje":"70%","descripcion":"Datos con los que el modelo aprendió patrones.","uso":"Entrenamiento"},"val":{"muestras":n_val,"porcentaje":"15%","descripcion":"Datos para detectar sobreajuste durante el entrenamiento.","uso":"Validación"},"test":{"muestras":len(X_test),"porcentaje":"15%","descripcion":"Datos completamente nuevos para medir el rendimiento real.","uso":"Prueba final"},"nota":nota},
        "model_loaded":model_name,
        "architecture": info["nombre"],
        "epoch_log":[f"Modelo '{model_name}' ({info['nombre']}) cargado",f"Evaluado sobre {len(X_test):,} muestras",f"RMSE: {metricas['rmse']}  MAE: {metricas['mae']}  MAPE: {metricas['mape']}%"],
    })

if __name__=="__main__":
    app.run(debug=True,port=5000)