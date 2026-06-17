import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Cargar modelo activo
with open('model_registry/active_version.txt', 'r') as f:
    version = f.read().strip()
model_data = joblib.load(f'model_registry/model_{version}.pkl')
model = model_data['modelo']
features = model_data['features']

# Cargar datos y replicar feature engineering
df = pd.read_csv('telco_churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# Selección de columnas categóricas (compatible con pandas 2 y 3)
cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
if 'customerID' in cat_cols:
    cat_cols.remove('customerID')
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Feature engineering
df['AvgMonthlyCharges'] = df['TotalCharges'] / (df['tenure'] + 1)
df['ChargesToMonthlyRatio'] = df['TotalCharges'] / (df['MonthlyCharges'] + 1)
df['Tenure_x_Monthly'] = df['tenure'] * df['MonthlyCharges']
df['IsNewCustomer'] = (df['tenure'] <= 12).astype(int)
service_cols = [c for c in df.columns if any(p in c for p in ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies'])]
df['NumExtraServices'] = df[service_cols].sum(axis=1)

new_data = df[features]
ref_data = new_data.iloc[:int(len(new_data)*0.8)]

def psi(expected, actual, buckets=10):
    expected = np.array(expected, dtype=np.float64)
    actual = np.array(actual, dtype=np.float64)
    breaks = np.percentile(expected, np.linspace(0, 100, buckets+1))
    expected_perc = np.histogram(expected, breaks)[0] / len(expected)
    actual_perc = np.histogram(actual, breaks)[0] / len(actual)
    mask = (expected_perc > 0) & (actual_perc > 0)
    return np.sum((actual_perc[mask] - expected_perc[mask]) * np.log(actual_perc[mask] / expected_perc[mask]))

print(f"Monitor ejecutado {datetime.now()}")
for feat in features[:5]:   # puedes cambiar a features para ver todas
    psi_val = psi(ref_data[feat], new_data[feat])
    if psi_val > 0.25:
        print(f"⚠️ DRIFT en {feat}: PSI = {psi_val:.3f}")
    else:
        print(f"✅ {feat}: PSI = {psi_val:.3f}")