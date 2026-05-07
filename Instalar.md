## Cómo correr el proyecto

### 1. Instalar dependencias Python
pip install torch numpy pandas scikit-learn flask flask-cors joblib

### 2. Agregar los datasets en data/raw/
- household_power_consumption.txt (UCI ML Repository)
- Dataset_Sismología_Nuevo.csv (ya incluido)

### 3. Correr el pipeline de datos
python pipeline.py

### 4. Entrenar modelos (opcional, desde terminal)
python agents/train_lstm_sismos.py

### 5. Levantar la API
python api.py

### 6. Correr el dashboard
cd frontend
npm install
npm start