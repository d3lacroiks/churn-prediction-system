import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# 1. Cargar el modelo activo actual
# ------------------------------------------------------------
with open('model_registry/active_version.txt', 'r') as f:
    version_actual = f.read().strip()
modelo_path_actual = f'model_registry/model_{version_actual}.pkl'
model_data_actual = joblib.load(modelo_path_actual)
modelo_actual = model_data_actual['modelo']
features = model_data_actual['features']
print(f"Modelo actual: {version_actual}")

# ------------------------------------------------------------
# 2. Preparar datos de referencia y nuevos
# ------------------------------------------------------------
df = pd.read_csv('telco_churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Feature engineering idéntico al de train_batch.py
cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
if 'customerID' in cat_cols:
    cat_cols.remove('customerID')
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
df['AvgMonthlyCharges'] = df['TotalCharges'] / (df['tenure'] + 1)
df['ChargesToMonthlyRatio'] = df['TotalCharges'] / (df['MonthlyCharges'] + 1)
df['Tenure_x_Monthly'] = df['tenure'] * df['MonthlyCharges']
df['IsNewCustomer'] = (df['tenure'] <= 12).astype(int)
service_cols = [c for c in df.columns if any(p in c for p in ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies'])]
df['NumExtraServices'] = df[service_cols].sum(axis=1)

# Aseguramos que estén todas las features del modelo
for col in features:
    if col not in df.columns:
        df[col] = 0

# Dividimos en referencia (primer 80%) y nuevos (último 20%)
df = df.sort_values('tenure').reset_index(drop=True)
ref_data = df[features].iloc[:int(len(df)*0.8)]
new_data = df[features].iloc[int(len(df)*0.8):]

# ------------------------------------------------------------
# 3. Calcular PSI para detectar drift
# ------------------------------------------------------------
def psi(expected, actual, buckets=10):
    expected = np.array(expected, dtype=np.float64)
    actual = np.array(actual, dtype=np.float64)
    breaks = np.percentile(expected, np.linspace(0, 100, buckets+1))
    expected_perc = np.histogram(expected, breaks)[0] / len(expected)
    actual_perc = np.histogram(actual, breaks)[0] / len(actual)
    mask = (expected_perc > 0) & (actual_perc > 0)
    return np.sum((actual_perc[mask] - expected_perc[mask]) * np.log(actual_perc[mask] / expected_perc[mask]))

# Evaluamos PSI en las 5 variables más importantes según el modelo
importancias = modelo_actual.feature_importances_
top_idx = np.argsort(importancias)[-5:]
psi_values = {}
drift_detectado = False
for idx in top_idx:
    feat = features[idx]
    valor_psi = psi(ref_data[feat], new_data[feat])
    psi_values[feat] = valor_psi
    if valor_psi > 0.25:
        drift_detectado = True
        print(f"⚠️ Drift detectado en {feat}: PSI = {valor_psi:.3f}")

# ------------------------------------------------------------
# 4. Decidir si reentrenar
# ------------------------------------------------------------
if not drift_detectado:
    print("✅ No se detectó drift significativo. No es necesario reentrenar.")
    sys.exit(0)

print("\n🔥 Drift severo detectado. Iniciando reentrenamiento automático...")

# Ejecutar train_batch.py como módulo (suponiendo que tiene todo el código dentro de funciones, pero como es un script vamos a replicar el entrenamiento aquí)
# Volvemos a leer todos los datos (incluyendo ingesta) para entrenar con lo más fresco
df_full = df.copy()
# Agregamos ingesta si existe
ingesta_dir = 'ingesta'
if os.path.exists(ingesta_dir):
    for archivo in os.listdir(ingesta_dir):
        if archivo.endswith('.csv'):
            df_nuevo = pd.read_csv(f'{ingesta_dir}/{archivo}')
            # Aplicar mismo preprocesamiento básico
            df_nuevo['TotalCharges'] = pd.to_numeric(df_nuevo['TotalCharges'], errors='coerce')
            df_nuevo = df_nuevo.dropna(subset=['TotalCharges'])
            df_nuevo['Churn'] = df_nuevo['Churn'].map({'Yes': 1, 'No': 0})
            df_nuevo = pd.get_dummies(df_nuevo, columns=cat_cols, drop_first=True)
            df_nuevo['AvgMonthlyCharges'] = df_nuevo['TotalCharges'] / (df_nuevo['tenure'] + 1)
            df_nuevo['ChargesToMonthlyRatio'] = df_nuevo['TotalCharges'] / (df_nuevo['MonthlyCharges'] + 1)
            df_nuevo['Tenure_x_Monthly'] = df_nuevo['tenure'] * df_nuevo['MonthlyCharges']
            df_nuevo['IsNewCustomer'] = (df_nuevo['tenure'] <= 12).astype(int)
            df_nuevo['NumExtraServices'] = df_nuevo[service_cols].sum(axis=1) if len(service_cols) > 0 else 0
            for col in features:
                if col not in df_nuevo.columns:
                    df_nuevo[col] = 0
            df_full = pd.concat([df_full, df_nuevo], ignore_index=True)

# Entrenar nuevo modelo
X = df_full[features]
y = df_full['Churn']
tscv = TimeSeriesSplit(n_splits=3)
modelos = []
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    scale_pos = (len(y_train) - y_train.sum()) / y_train.sum()
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03,
        scale_pos_weight=scale_pos, random_state=42, verbose=-1
    )
    model.fit(X_train, y_train)
    modelos.append(model)

nuevo_modelo = modelos[-1]

# Evaluar AP del modelo antiguo y del nuevo en los datos de validación (últimos 20%)
X_val = new_data[features]
y_val = df['Churn'].iloc[int(len(df)*0.8):]  # las etiquetas correspondientes
y_pred_old = modelo_actual.predict_proba(X_val)[:, 1]
y_pred_new = nuevo_modelo.predict_proba(X_val)[:, 1]
ap_old = average_precision_score(y_val, y_pred_old)
ap_new = average_precision_score(y_val, y_pred_new)
print(f"AP modelo anterior: {ap_old:.4f}")
print(f"AP modelo nuevo: {ap_new:.4f}")

# ------------------------------------------------------------
# 5. Decidir si promover el nuevo modelo o hacer rollback
# ------------------------------------------------------------
now = datetime.now()
if ap_new >= ap_old:
    version = now.strftime('%Y%m%d_%H%M%S')
    modelo_path = f'model_registry/model_{version}.pkl'
    joblib.dump({'modelo': nuevo_modelo, 'features': features}, modelo_path)
    with open('model_registry/active_version.txt', 'w') as f:
        f.write(version)
    print(f"✅ Modelo nuevo promovido: {version}")
    decision = 'promovido'
else:
    print("⛔ El modelo nuevo empeora el rendimiento. Se mantiene el modelo anterior.")
    decision = 'rollback'

# Guardar log
log_entry = {
    'timestamp': now.isoformat(),
    'psi_max': max(psi_values.values()) if psi_values else 0.0,
    'drift_detectado': drift_detectado,
    'ap_old': ap_old,
    'ap_new': ap_new,
    'decision': decision
}
log_df = pd.DataFrame([log_entry])
log_file = 'pipeline_log.csv'
if not os.path.exists(log_file):
    log_df.to_csv(log_file, index=False)
else:
    log_df.to_csv(log_file, mode='a', header=False, index=False)
print(f"📋 Log actualizado en {log_file}")