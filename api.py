import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Cargar la versión activa del modelo
with open('model_registry/active_version.txt', 'r') as f:
    version = f.read().strip()
modelo_path = f'model_registry/model_{version}.pkl'
model_data = joblib.load(modelo_path)
model = model_data['modelo']
features = model_data['features']
print(f"Modelo cargado: {modelo_path}")

# Función de explicación SHAP (opcional, puede ser pesada en producción real)
import shap
explainer = shap.TreeExplainer(model)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Construir DataFrame con el orden correcto de features
    df = pd.DataFrame([data])
    # Rellenar columnas faltantes con 0
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df[features]

    prob = model.predict_proba(df)[:, 1][0]
    # Calcular SHAP para el cliente
    shap_vals = explainer.shap_values(df)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_vals = shap_vals[0]
    top_idx = np.argsort(np.abs(shap_vals))[-3:][::-1]
    factores = {features[i]: shap_vals[i] for i in top_idx}
    return jsonify({'prob_churn': prob, 'factores': factores})

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_version': version})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)