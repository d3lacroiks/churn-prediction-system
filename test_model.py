import joblib
import pandas as pd
import sys

# Cargar modelo activo
with open('model_registry/active_version.txt', 'r') as f:
    version = f.read().strip()
model_path = f'model_registry/model_{version}.pkl'
model_data = joblib.load(model_path)
modelo = model_data['modelo']
features = model_data['features']

print(f"Modelo cargado: {version}")
print(f"Features esperadas: {len(features)}")

# Crear un dato de prueba sencillo (similares al dataset)
df = pd.read_csv('telco_churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Preprocesamiento rápido (igual que en los otros scripts)
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

# Asegurar que están todas las features
for col in features:
    if col not in df.columns:
        df[col] = 0

# Tomar una muestra de prueba
X_test = df[features].iloc[:5]
y_pred = modelo.predict_proba(X_test)[:, 1]
print("Predicciones de prueba:", y_pred)

if len(y_pred) == 5:
    print("✅ Modelo funcionando correctamente.")
    sys.exit(0)
else:
    print("❌ El modelo no produjo las predicciones esperadas.")
    sys.exit(1)