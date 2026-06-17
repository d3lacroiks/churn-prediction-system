import pandas as pd

# Carga base
df_base = pd.read_csv('telco_churn.csv')
print(f"Base: {df_base.shape}")
print(f"TotalCharges no numéricos en base: {pd.to_numeric(df_base['TotalCharges'], errors='coerce').isna().sum()}")

# Carga ingesta
import os
ingesta_dir = 'ingesta'
if os.path.exists(ingesta_dir):
    for archivo in os.listdir(ingesta_dir):
        if archivo.endswith('.csv'):
            df_nuevo = pd.read_csv(f'{ingesta_dir}/{archivo}')
            print(f"\nArchivo {archivo}: {df_nuevo.shape}")
            print(f"Columnas: {df_nuevo.columns.tolist()}")
            print(f"Valores nulos por columna:\n{df_nuevo.isnull().sum()}")
            # Intentar convertir TotalCharges
            temp = pd.to_numeric(df_nuevo['TotalCharges'], errors='coerce')
            print(f"TotalCharges no numéricos en ingesta: {temp.isna().sum()} (de {len(df_nuevo)})")