import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------
# 1. Carga de datos BASE + ingesta
# -----------------------------------------------
print("Cargando datos...")
df_base = pd.read_csv('telco_churn.csv')
print(f"Registros base: {len(df_base)}")

# Carpeta de ingesta (si existe)
ingesta_dir = 'ingesta'
if os.path.exists(ingesta_dir):
    for archivo in sorted(os.listdir(ingesta_dir)):
        if archivo.endswith('.csv'):
            ruta = os.path.join(ingesta_dir, archivo)
            df_nuevo = pd.read_csv(ruta)
            df_base = pd.concat([df_base, df_nuevo], ignore_index=True)
            print(f"Agregados {len(df_nuevo)} registros de {archivo}")

df = df_base.copy()
print(f"Total registros: {len(df)}")

# -----------------------------------------------
# 2. Limpieza
# -----------------------------------------------
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# -----------------------------------------------
# 3. Feature engineering
# -----------------------------------------------
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

features = [c for c in df.columns if c not in ['customerID', 'Churn']]
print(f"Número de features: {len(features)}")

# -----------------------------------------------
# 4. Validación temporal y entrenamiento
# -----------------------------------------------
df = df.sort_values('tenure').reset_index(drop=True)
X = df[features]
y = df['Churn']

print("Entrenando modelo...")
tscv = TimeSeriesSplit(n_splits=3)
modelos = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    scale_pos = (len(y_train) - y_train.sum()) / y_train.sum()
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03,
        scale_pos_weight=scale_pos, random_state=42, verbose=-1
    )
    model.fit(X_train, y_train)
    modelos.append(model)
    y_pred = model.predict_proba(X_val)[:, 1]
    ap = average_precision_score(y_val, y_pred)
    print(f"  Fold {fold+1} AP: {ap:.4f}")

final_model = modelos[-1]

# -----------------------------------------------
# 5. Guardado del modelo versionado
# -----------------------------------------------
os.makedirs('model_registry', exist_ok=True)
version = datetime.now().strftime('%Y%m%d_%H%M%S')
modelo_path = f'model_registry/model_{version}.pkl'
joblib.dump({'modelo': final_model, 'features': features}, modelo_path)
print(f"✅ Modelo guardado en {modelo_path}")

with open('model_registry/active_version.txt', 'w') as f:
    f.write(version)
print(f"✅ Versión activa: {version}")