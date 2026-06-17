import pandas as pd
import numpy as np

# Cargar el dataset original
df = pd.read_csv('telco_churn.csv')

# Seleccionar 200 clientes al azar
nuevos = df.sample(200, random_state=42).copy()

# Modificar ligeramente algunas variables para simular datos frescos
nuevos['MonthlyCharges'] = nuevos['MonthlyCharges'] * np.random.uniform(0.9, 1.1, 200)
nuevos['tenure'] = (nuevos['tenure'] + np.random.randint(0, 5, 200)).clip(lower=0)

# Asegurar que TotalCharges sea numérico y rellenar posibles NaN
nuevos['TotalCharges'] = pd.to_numeric(nuevos['TotalCharges'], errors='coerce')
nuevos['TotalCharges'].fillna(nuevos['MonthlyCharges'] * nuevos['tenure'], inplace=True)

# Guardar con header (comportamiento por defecto de to_csv)
nuevos.to_csv('ingesta/nuevos_clientes.csv', index=False)
print("Archivo 'ingesta/nuevos_clientes.csv' creado correctamente.")
print(f"Dimensiones: {nuevos.shape}")
print("Primeras columnas:", nuevos.columns[:5].tolist())