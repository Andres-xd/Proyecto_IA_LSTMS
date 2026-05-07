"""
API FLASK — Sistema LSTM de Predicción de Series de Tiempo
===========================================================
Cómo correr:
    python api.py
    Luego abrir http://localhost:5000

Endpoints:
    GET  /api/status                 → verifica que la API esté viva
    GET  /api/modelos                → lista todos los modelos disponibles
    GET  /api/dataset-info/<tipo>    → info del dataset (sismos o energia)
    POST /api/predecir               → evalúa un modelo y devuelve resultados
    POST /api/entrenar               → entrena un modelo en segundo plano
    GET  /api/entrenar/progreso      → progreso del entrenamiento en curso

Sistema de modelos:
    Cada vez que se entrena un modelo, se guardan DOS archivos:
        modelo.pth  → los pesos de la red neuronal
        modelo.json → la configuración usada (capas, neuronas, dataset, etc.)

    Esto permite cargar CUALQUIER modelo entrenado, sin importar el nombre,
    porque la configuración está guardada junto al archivo.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math, threading, json

app = Flask(__name__)
CORS(app)

BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Límites para el dataset de energía (evita quedarse sin RAM)
ENERGIA_MAX_TEST = 5_000

# ── Info de los datasets ───────────────────────────────────────────────────────
INFO_DATASETS = {
    "sismos": {
        "nombre":     "Actividad Sísmica — Guatemala / Centroamérica",
        "fuente":     "USGS Earthquake Catalog",
        "registros":  "3,586 eventos (1970–2026)",
        "periodo":    "Ene 1970 — Mar 2026",
        "frecuencia": "Serie diaria agregada",
        "features":   ["avg_mag", "max_mag", "avg_depth", "count_sismos", "hay_actividad"],
        "target":     "avg_mag (magnitud promedio diaria)",
        "descripcion": "Registro de actividad sísmica en Centroamérica. Permite analizar patrones útiles para sistemas de monitoreo en regiones de alta actividad geológica como Guatemala.",
        "window_size": 14,
        "color": "#EF4444",
    },
    "energia": {
        "nombre":     "Consumo Eléctrico Doméstico",
        "fuente":     "UCI Machine Learning Repository",
        "registros":  "2,075,259 mediciones",
        "periodo":    "Dic 2006 — Nov 2010",
        "frecuencia": "1 medición por minuto",
        "features":   ["Global_active_power", "Global_reactive_power", "Voltage",
                       "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"],
        "target":     "Global_active_power (potencia activa en kW)",
        "descripcion": "Mediciones de consumo eléctrico de un hogar en Francia. Permite predecir la demanda energética futura para optimizar la distribución en redes eléctricas inteligentes.",
        "window_size": 60,
        "color": "#F59E0B",
    },
}


# ── Modelo LSTM genérico ───────────────────────────────────────────────────────
class ModeloLSTM(nn.Module):
    """
    Un solo modelo que sirve para todas las variantes:
        num_layers=1 → LSTM Simple
        num_layers>1 → Stacked LSTM
    """
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = 0.2 if num_layers > 1 else 0.0,
            batch_first = True,
        )
        self.dropout = nn.Dropout(0.2)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


# ── Dataset eficiente en memoria ──────────────────────────────────────────────
class DatasetNumpy(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0),
        )


# ── Estado del entrenamiento en curso ─────────────────────────────────────────
estado = {
    "corriendo":    False,
    "epoca_actual": 0,
    "total_epocas": 0,
    "loss_train":   0.0,
    "loss_val":     0.0,
    "terminado":    False,
    "metricas":     {},
    "log":          [],
    "error":        None,
    "tabla_pred":   [],
    "pronostico":   [],
    "dataset_info": {},
}


# ── Funciones auxiliares ───────────────────────────────────────────────────────
def calcular_metricas(y_real, preds):
    rmse = math.sqrt(mean_squared_error(y_real, preds))
    mae  = mean_absolute_error(y_real, preds)
    mask = y_real != 0
    mape = float(np.mean(np.abs((y_real[mask] - preds[mask]) / y_real[mask])) * 100) if mask.sum() > 0 else 0.0
    return {"rmse": round(rmse, 6), "mae": round(mae, 6), "mape": round(mape, 2)}


def guardar_metadata_modelo(nombre, dataset, tipo, input_size, hidden_size, num_layers):
    """
    Guarda un JSON con la configuración del modelo junto al .pth.
    Esto permite cargarlo después sin necesidad de recordar los parámetros.
    """
    meta = {
        "nombre":      nombre,
        "dataset":     dataset,
        "tipo":        tipo,
        "input_size":  input_size,
        "hidden_size": hidden_size,
        "num_layers":  num_layers,
    }
    ruta = MODELS_DIR / f"{nombre}.json"
    with open(ruta, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def cargar_metadata_modelo(nombre):
    """
    Lee la configuración del modelo en este orden:
        1. Si existe .json → usar esos valores (más confiable)
        2. Si no hay .json → leer los pesos del .pth y detectar
           la arquitectura real desde las dimensiones de las matrices
    """
    ruta_json = MODELS_DIR / f"{nombre}.json"
    ruta_pth  = MODELS_DIR / f"{nombre}.pth"

    if not ruta_pth.exists():
        raise FileNotFoundError(
            f"No existe '{nombre}.pth' en la carpeta models/. "
            f"Entrénalo primero desde la aplicación o desde agents/."
        )

    # Opción 1: JSON disponible → usarlo directamente
    if ruta_json.exists():
        with open(ruta_json) as f:
            return json.load(f)

    # Opción 2: Sin JSON → leer la arquitectura real desde el .pth
    # Los pesos del LSTM tienen formas predecibles:
    #   weight_ih_l0 → (4 * hidden_size, input_size)
    #   weight_hh_l0 → (4 * hidden_size, hidden_size)
    # Con eso podemos reconstruir hidden_size, input_size y num_layers exactos.
    try:
        pesos = torch.load(ruta_pth, map_location="cpu")

        # Detectar input_size y hidden_size desde la primera capa
        w_ih = pesos.get("lstm.weight_ih_l0")
        w_hh = pesos.get("lstm.weight_hh_l0")

        if w_ih is None or w_hh is None:
            raise ValueError("El archivo .pth no tiene el formato esperado de un ModeloLSTM.")

        input_size  = w_ih.shape[1]            # columnas de weight_ih_l0
        hidden_size = w_hh.shape[1]            # columnas de weight_hh_l0

        # Detectar número de capas contando cuántas capas hay en los pesos
        num_layers = sum(1 for k in pesos if k.startswith("lstm.weight_ih_l"))

        # Inferir dataset e tipo por nombre
        nombre_lower = nombre.lower()
        dataset = "energia" if "energia" in nombre_lower else "sismos"
        tipo    = "Stacked LSTM" if num_layers > 1 else "LSTM Simple"

        return {
            "nombre":      nombre,
            "dataset":     dataset,
            "tipo":        tipo,
            "input_size":  input_size,
            "hidden_size": hidden_size,
            "num_layers":  num_layers,
        }

    except Exception as e:
        raise ValueError(
            f"No se pudo leer la arquitectura de '{nombre}.pth': {e}. "
            f"Intenta reentrenar el modelo desde la aplicación."
        )


def cargar_datos_test(dataset):
    suffix = dataset
    X_test = np.load(DATA_DIR / f"X_test_{suffix}.npy")
    y_test = np.load(DATA_DIR / f"y_test_{suffix}.npy")
    if dataset == "energia":
        X_test = X_test[:ENERGIA_MAX_TEST]
        y_test = y_test[:ENERGIA_MAX_TEST]
    return X_test, y_test


def leer_tamanos_dataset(dataset):
    """Lee los tamaños exactos de cada conjunto desde los archivos .npy."""
    try:
        n_train = len(np.load(DATA_DIR / f"X_train_{dataset}.npy", mmap_mode="r"))
        n_val   = len(np.load(DATA_DIR / f"X_val_{dataset}.npy",   mmap_mode="r"))
        n_test  = len(np.load(DATA_DIR / f"X_test_{dataset}.npy",  mmap_mode="r"))
        return n_train, n_val, n_test
    except Exception:
        return 0, 0, 0


def construir_dataset_info(dataset, n_train, n_val, n_test, nota=""):
    total = (n_train + n_val + n_test) or 1
    return {
        "train": {
            "muestras":    n_train,
            "porcentaje":  f"{n_train/total*100:.1f}%",
            "descripcion": "Datos con los que el modelo aprendió a reconocer patrones.",
            "uso":         "Entrenamiento",
        },
        "val": {
            "muestras":    n_val,
            "porcentaje":  f"{n_val/total*100:.1f}%",
            "descripcion": "Datos que el modelo nunca vio durante el entrenamiento. Sirven para detectar si está memorizando en lugar de aprender.",
            "uso":         "Validación",
        },
        "test": {
            "muestras":    n_test,
            "porcentaje":  f"{n_test/total*100:.1f}%",
            "descripcion": "Datos completamente nuevos usados solo al final para medir qué tan bien predice el modelo en situaciones reales.",
            "uso":         "Prueba final",
        },
        "nota": nota,
    }


def generar_pronostico(modelo, ventana_inicial, pasos=30):
    """Predice N pasos hacia el futuro usando la última ventana conocida."""
    ventana    = ventana_inicial.copy()
    pronostico = []
    with torch.no_grad():
        for paso in range(pasos):
            entrada    = torch.tensor(ventana[np.newaxis], dtype=torch.float32)
            pred       = modelo(entrada).item()
            pronostico.append({"paso": paso + 1, "valor": round(pred, 4)})
            nueva_fila    = ventana[-1].copy()
            nueva_fila[0] = pred
            ventana       = np.vstack([ventana[1:], nueva_fila])
    return pronostico


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.route("/api/status")
def status():
    return jsonify({"ok": True})


@app.route("/api/dataset-info/<dataset_tipo>")
def info_dataset(dataset_tipo):
    if dataset_tipo not in INFO_DATASETS:
        return jsonify({"error": "Usa 'sismos' o 'energia'."}), 404
    return jsonify(INFO_DATASETS[dataset_tipo])


@app.route("/api/modelos")
def listar_modelos():
    """
    Lista todos los archivos .pth en la carpeta models/.
    Lee el .json de cada uno para saber su configuración.
    Si no tiene .json, infiere la configuración por el nombre.
    Así aparecen TODOS los modelos entrenados, sin importar el nombre.
    """
    resultado = []
    if not MODELS_DIR.exists():
        return jsonify([])

    for pth in sorted(MODELS_DIR.glob("*.pth")):
        nombre = pth.stem
        try:
            meta = cargar_metadata_modelo(nombre)
            resultado.append({
                "nombre":      nombre,
                "tipo":        meta.get("tipo", "Modelo"),
                "dataset":     meta.get("dataset", "sismos"),
                "descripcion": meta.get("descripcion", f"Modelo guardado como '{nombre}.pth'"),
                "entrenado":   True,
                "tamano_kb":   round(pth.stat().st_size / 1024, 1),
            })
        except Exception as e:
            # Si algo falla al leer, igual lo incluimos con info básica
            resultado.append({
                "nombre":      nombre,
                "tipo":        "Modelo Custom",
                "dataset":     "sismos" if "sismo" in nombre.lower() else "energia" if "energia" in nombre.lower() else "sismos",
                "descripcion": f"Modelo guardado como '{nombre}.pth'",
                "entrenado":   True,
                "tamano_kb":   round(pth.stat().st_size / 1024, 1),
            })

    return jsonify(resultado)


@app.route("/api/predecir", methods=["POST"])
def predecir():
    """
    Carga un modelo por nombre, lo evalúa y devuelve resultados.
    Body: { "modelo": "nombre_sin_extension" }
    """
    datos  = request.json or {}
    nombre = datos.get("modelo")

    if not nombre:
        return jsonify({"error": "Falta el campo 'modelo'."}), 400

    try:
        meta = cargar_metadata_modelo(nombre)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Error al leer configuración: {str(e)}"}), 500

    # Cargar pesos del modelo
    modelo = ModeloLSTM(
        input_size  = meta["input_size"],
        hidden_size = meta["hidden_size"],
        num_layers  = meta["num_layers"],
    )
    try:
        modelo.load_state_dict(torch.load(MODELS_DIR / f"{nombre}.pth", map_location="cpu"))
    except Exception as e:
        return jsonify({"error": f"No se pudo cargar el archivo .pth: {str(e)}"}), 500
    modelo.eval()

    # Cargar datos de test
    try:
        X_test, y_test = cargar_datos_test(meta["dataset"])
    except FileNotFoundError:
        return jsonify({"error": "No se encontraron los datos procesados. Corre el pipeline primero."}), 500

    # Evaluar
    loader = DataLoader(DatasetNumpy(X_test, y_test), batch_size=256, shuffle=False)
    preds, reales = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            preds.extend(modelo(Xb).cpu().numpy().flatten())
            reales.extend(yb.numpy().flatten())

    preds  = np.array(preds)
    reales = np.array(reales)
    metricas = calcular_metricas(reales, preds)

    tabla_pred = [
        {"n": i+1, "real": round(float(reales[i]),4), "pred": round(float(preds[i]),4),
         "error": round(abs(float(reales[i])-float(preds[i])),4)}
        for i in range(min(20, len(preds)))
    ]

    pronostico = []
    if meta["dataset"] == "sismos":
        pronostico = generar_pronostico(modelo, X_test[-1], pasos=30)

    # Tamaños exactos del dataset
    n_train, n_val, n_test = leer_tamanos_dataset(meta["dataset"])
    dataset_info = construir_dataset_info(
        meta["dataset"], n_train, n_val, len(y_test),
        nota=f"Modelo '{meta['tipo']}' — {meta['dataset']}"
    )

    return jsonify({
        "metricas":    metricas,
        "tabla_pred":  tabla_pred,
        "pronostico":  pronostico,
        "dataset_info": dataset_info,
        "info_modelo": {
            "nombre":  nombre,
            "tipo":    meta["tipo"],
            "dataset": meta["dataset"],
        },
    })


# ── ENTRENAMIENTO EN SEGUNDO PLANO ────────────────────────────────────────────

def _hilo_entrenamiento(dataset, config):
    global estado
    torch.manual_seed(42)
    np.random.seed(42)

    nombre     = config.get("model_name", f"model_{dataset}_custom")
    epocas     = int(config.get("epochs",     20))
    batch_size = int(config.get("batch_size", 64))
    patience   = int(config.get("patience",   10))
    tipo       = config.get("tipo",           "LSTM Simple")
    num_layers = 3 if "stacked" in tipo.lower() else 2
    hidden_size = 128 if dataset == "sismos" else 64

    estado.update({
        "corriendo": True, "terminado": False, "error": None,
        "epoca_actual": 0, "total_epocas": epocas,
        "metricas": {}, "log": [], "tabla_pred": [], "pronostico": [],
    })

    try:
        # Cargar datos
        X_train_full = np.load(DATA_DIR / f"X_train_{dataset}.npy", mmap_mode="r")
        X_val_full   = np.load(DATA_DIR / f"X_val_{dataset}.npy",   mmap_mode="r")
        X_test_full  = np.load(DATA_DIR / f"X_test_{dataset}.npy",  mmap_mode="r")
        y_train_full = np.load(DATA_DIR / f"y_train_{dataset}.npy")
        y_val_full   = np.load(DATA_DIR / f"y_val_{dataset}.npy")
        y_test_full  = np.load(DATA_DIR / f"y_test_{dataset}.npy")

        # Para energía: submuestreo para no saturar RAM
        if dataset == "energia":
            idx = np.arange(0, len(X_train_full), 4)
            X_train = X_train_full[idx[:25_000]]
            y_train = y_train_full[idx[:25_000]]
            X_val   = X_val_full[:5_000]
            y_val   = y_val_full[:5_000]
            X_test  = X_test_full[:ENERGIA_MAX_TEST]
            y_test  = y_test_full[:ENERGIA_MAX_TEST]
        else:
            X_train, y_train = X_train_full, y_train_full
            X_val,   y_val   = X_val_full,   y_val_full
            X_test,  y_test  = X_test_full,  y_test_full

        input_size = X_train.shape[2]

        estado["log"] += [
            f"Dataset: {dataset} | Muestras de entrenamiento: {len(X_train):,}",
            f"Agente: {tipo} | Capas: {num_layers} | Neuronas: {hidden_size} | Épocas máx: {epocas}",
            "─" * 50,
        ]

        modelo      = ModeloLSTM(input_size, hidden_size, num_layers)
        criterio    = nn.MSELoss()
        optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, patience=5, factor=0.5)

        train_loader = DataLoader(DatasetNumpy(X_train, y_train), batch_size=batch_size, shuffle=False)
        val_loader   = DataLoader(DatasetNumpy(X_val,   y_val),   batch_size=batch_size, shuffle=False)

        mejor_loss   = float("inf")
        mejor_estado = None
        sin_mejora   = 0

        for epoca in range(epocas):
            modelo.train()
            loss_train = 0.0
            for Xb, yb in train_loader:
                optimizador.zero_grad()
                loss = criterio(modelo(Xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                optimizador.step()
                loss_train += loss.item()
            loss_train /= len(train_loader)

            modelo.eval()
            loss_val = 0.0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    loss_val += criterio(modelo(Xb), yb).item()
            loss_val /= len(val_loader)

            scheduler.step(loss_val)

            indicador = " <-- mejor" if loss_val < mejor_loss else ""
            estado["log"].append(
                f"Época [{epoca+1:3d}/{epocas}]  train={loss_train:.6f}  val={loss_val:.6f}{indicador}"
            )
            if len(estado["log"]) > 120:
                estado["log"] = estado["log"][-120:]

            estado.update({
                "epoca_actual": epoca + 1,
                "loss_train":   round(loss_train, 6),
                "loss_val":     round(loss_val,   6),
            })

            if loss_val < mejor_loss:
                mejor_loss   = loss_val
                mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
                sin_mejora   = 0
            else:
                sin_mejora += 1
                if sin_mejora >= patience:
                    estado["log"].append(f"Early stopping en época {epoca+1} — sin mejora por {patience} épocas.")
                    break

        # Guardar modelo + metadata
        modelo.load_state_dict(mejor_estado)
        torch.save(modelo.state_dict(), MODELS_DIR / f"{nombre}.pth")
        guardar_metadata_modelo(nombre, dataset, tipo, input_size, hidden_size, num_layers)

        estado["log"] += [
            "─" * 50,
            f"Modelo guardado: {nombre}.pth",
            "Evaluando sobre el conjunto de prueba...",
        ]

        # Evaluación final
        test_loader = DataLoader(DatasetNumpy(X_test, y_test), batch_size=256, shuffle=False)
        preds, reales = [], []
        with torch.no_grad():
            for Xb, yb in test_loader:
                preds.extend(modelo(Xb).cpu().numpy().flatten())
                reales.extend(yb.numpy().flatten())

        preds  = np.array(preds)
        reales = np.array(reales)
        metricas = calcular_metricas(reales, preds)

        tabla_pred = [
            {"n": i+1, "real": round(float(reales[i]),4), "pred": round(float(preds[i]),4),
             "error": round(abs(float(reales[i])-float(preds[i])),4)}
            for i in range(min(20, len(preds)))
        ]

        pronostico = []
        if dataset == "sismos":
            pronostico = generar_pronostico(modelo, X_test[-1], pasos=30)

        n_train_real, n_val_real, _ = leer_tamanos_dataset(dataset)
        dataset_info = construir_dataset_info(
            dataset, n_train_real, n_val_real, len(y_test),
            nota=f"Agente '{tipo}' entrenado con dataset de {dataset}"
        )

        estado["log"].append(
            f"RMSE: {metricas['rmse']}  |  MAE: {metricas['mae']}  |  MAPE: {metricas['mape']}%"
        )
        estado.update({
            "corriendo":    False,
            "terminado":    True,
            "metricas":     metricas,
            "tabla_pred":   tabla_pred,
            "pronostico":   pronostico,
            "dataset_info": dataset_info,
        })

    except Exception as e:
        import traceback
        estado.update({
            "corriendo": False,
            "terminado": True,
            "error":     str(e),
            "log":       estado.get("log", []) + [f"ERROR: {e}", traceback.format_exc()],
        })


@app.route("/api/entrenar", methods=["POST"])
def iniciar_entrenamiento():
    if estado["corriendo"]:
        return jsonify({"error": "Ya hay un entrenamiento en curso. Espera a que termine."}), 400

    datos   = request.json or {}
    dataset = datos.get("dataset")
    if dataset not in ["energia", "sismos"]:
        return jsonify({"error": "El campo 'dataset' debe ser 'sismos' o 'energia'."}), 400

    config = {
        "epochs":     datos.get("epochs",     20),
        "batch_size": datos.get("batch_size", 64),
        "patience":   datos.get("patience",   10),
        "model_name": datos.get("model_name", f"model_{dataset}_custom"),
        "tipo":       datos.get("tipo",       "LSTM Simple"),
    }

    threading.Thread(target=_hilo_entrenamiento, args=(dataset, config), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/entrenar/progreso")
def progreso():
    return jsonify(estado)


# ── Punto de entrada ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  API LSTM — Sistema de Predicción")
    print("  Corriendo en http://localhost:5000")
    print("=" * 55)
    app.run(debug=True, port=5000)