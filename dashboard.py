import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Cargar el modelo activo
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open('model_registry/active_version.txt', 'r') as f:
        version = f.read().strip()
    model_path = f'model_registry/model_{version}.pkl'
    data = joblib.load(model_path)
    return data['modelo'], data['features'], version

modelo, features, version = load_model()

# ------------------------------------------------------------------
# 2. Interfaz de usuario
# ------------------------------------------------------------------
st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")
st.title("🔮 Predicción de Churn - RetailCorp")
st.markdown(f"**Modelo activo:** `{version}`  |  Nº de características: {len(features)}")

tab1, tab2 = st.tabs(["🔍 Evaluar un cliente", "📊 Clientes en riesgo"])

# ------------------------------------------------------------------
# Tab 1: Formulario de predicción
# ------------------------------------------------------------------
with tab1:
    st.header("Datos del cliente")
    col1, col2, col3 = st.columns(3)

    # Entradas básicas
    with col1:
        tenure = st.number_input("Antigüedad (meses)", 0, 72, 1)
        MonthlyCharges = st.number_input("Cargo mensual ($)", 0.0, 200.0, 70.0)
        TotalCharges = st.number_input("Cargo total acumulado ($)", 0.0, 10000.0, 100.0)
    with col2:
        gender = st.selectbox("Género", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Adulto mayor", [0, 1])
        Partner = st.selectbox("Tiene pareja", ["Yes", "No"])
        Dependents = st.selectbox("Tiene dependientes", ["Yes", "No"])
    with col3:
        InternetService = st.selectbox("Servicio de Internet", ["DSL", "Fiber optic", "No"])
        Contract = st.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Factura electrónica", ["Yes", "No"])
        PaymentMethod = st.selectbox("Método de pago", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    # Servicios adicionales
    st.subheader("Servicios contratados")
    col4, col5, col6 = st.columns(3)
    with col4:
        PhoneService = st.selectbox("Teléfono", ["Yes", "No"])
        MultipleLines = st.selectbox("Múltiples líneas", ["Yes", "No", "No phone service"])
    with col5:
        OnlineSecurity = st.selectbox("Seguridad en línea", ["Yes", "No", "No internet service"])
        OnlineBackup = st.selectbox("Respaldo en línea", ["Yes", "No", "No internet service"])
    with col6:
        DeviceProtection = st.selectbox("Protección de dispositivo", ["Yes", "No", "No internet service"])
        TechSupport = st.selectbox("Soporte técnico", ["Yes", "No", "No internet service"])
        StreamingTV = st.selectbox("TV en streaming", ["Yes", "No", "No internet service"])
        StreamingMovies = st.selectbox("Películas en streaming", ["Yes", "No", "No internet service"])

    # Construir el diccionario de entrada
    input_data = {
        "tenure": tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "gender_Male": 1 if gender == "Male" else 0,
        "SeniorCitizen": int(SeniorCitizen),
        "Partner_Yes": 1 if Partner == "Yes" else 0,
        "Dependents_Yes": 1 if Dependents == "Yes" else 0,
        "PhoneService_Yes": 1 if PhoneService == "Yes" else 0,
        "MultipleLines_Yes": 1 if MultipleLines == "Yes" else 0,
        "MultipleLines_No phone service": 1 if MultipleLines == "No phone service" else 0,
        "InternetService_Fiber optic": 1 if InternetService == "Fiber optic" else 0,
        "InternetService_No": 1 if InternetService == "No" else 0,
        "OnlineSecurity_Yes": 1 if OnlineSecurity == "Yes" else 0,
        "OnlineSecurity_No internet service": 1 if OnlineSecurity == "No internet service" else 0,
        "OnlineBackup_Yes": 1 if OnlineBackup == "Yes" else 0,
        "OnlineBackup_No internet service": 1 if OnlineBackup == "No internet service" else 0,
        "DeviceProtection_Yes": 1 if DeviceProtection == "Yes" else 0,
        "DeviceProtection_No internet service": 1 if DeviceProtection == "No internet service" else 0,
        "TechSupport_Yes": 1 if TechSupport == "Yes" else 0,
        "TechSupport_No internet service": 1 if TechSupport == "No internet service" else 0,
        "StreamingTV_Yes": 1 if StreamingTV == "Yes" else 0,
        "StreamingTV_No internet service": 1 if StreamingTV == "No internet service" else 0,
        "StreamingMovies_Yes": 1 if StreamingMovies == "Yes" else 0,
        "StreamingMovies_No internet service": 1 if StreamingMovies == "No internet service" else 0,
        "Contract_One year": 1 if Contract == "One year" else 0,
        "Contract_Two year": 1 if Contract == "Two year" else 0,
        "PaperlessBilling_Yes": 1 if PaperlessBilling == "Yes" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if PaymentMethod == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if PaymentMethod == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if PaymentMethod == "Mailed check" else 0,
    }

    # Features adicionales (calculadas)
    input_data["AvgMonthlyCharges"] = TotalCharges / (tenure + 1) if tenure > 0 else 0
    input_data["ChargesToMonthlyRatio"] = TotalCharges / (MonthlyCharges + 1) if MonthlyCharges > 0 else 0
    input_data["Tenure_x_Monthly"] = tenure * MonthlyCharges
    input_data["IsNewCustomer"] = 1 if tenure <= 12 else 0
    input_data["NumExtraServices"] = sum([
        input_data.get("OnlineSecurity_Yes", 0),
        input_data.get("OnlineBackup_Yes", 0),
        input_data.get("DeviceProtection_Yes", 0),
        input_data.get("TechSupport_Yes", 0),
        input_data.get("StreamingTV_Yes", 0),
        input_data.get("StreamingMovies_Yes", 0)
    ])

    # Asegurar que todas las features del modelo estén presentes
    df_input = pd.DataFrame([input_data])
    for col in features:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[features]

    # Predecir
    if st.button("Predecir probabilidad de churn"):
        proba = modelo.predict_proba(df_input)[:, 1][0]
        st.metric("Probabilidad de churn", f"{proba:.2%}", delta=None)

        # Explicación SHAP
        explainer = shap.TreeExplainer(modelo)
        shap_vals = explainer.shap_values(df_input)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_vals = shap_vals[0]

        # Mostrar top 5 factores
        idx = np.argsort(np.abs(shap_vals))[-5:][::-1]
        st.subheader("🔎 Factores de riesgo (SHAP)")
        for i in idx:
            col_name = features[i]
            val = shap_vals[i]
            st.write(f"- **{col_name}**: {val:+.3f} {'🔴' if val > 0 else '🟢'}")

        # Gráfico SHAP (waterfall)
        fig, ax = plt.subplots()
        shap.waterfall_plot(shap.Explanation(values=shap_vals, base_values=explainer.expected_value, data=df_input.iloc[0].values, feature_names=features), show=False)
        st.pyplot(fig)
        plt.close()

# ------------------------------------------------------------------
# Tab 2: Clientes con mayor riesgo (batch)
# ------------------------------------------------------------------
with tab2:
    st.header("📈 Clientes con mayor probabilidad de churn")
    # Cargar datos de ejemplo y predecir
    df = pd.read_csv('telco_churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Preprocesamiento rápido (usando el mismo feature engineering)
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if 'customerID' in cat_cols:
        cat_cols.remove('customerID')
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df['AvgMonthlyCharges'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['ChargesToMonthlyRatio'] = df['TotalCharges'] / (df['MonthlyCharges'] + 1)
    df['Tenure_x_Monthly'] = df['tenure'] * df['MonthlyCharges']
    df['IsNewCustomer'] = (df['tenure'] <= 12).astype(int)
    service_cols = [c for c in df.columns if any(p in c for p in ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies'])]
    df['NumExtraServices'] = df[service_cols].sum(axis=1)

    for col in features:
        if col not in df.columns:
            df[col] = 0

    df['prob_churn'] = modelo.predict_proba(df[features])[:, 1]
    top_risk = df.sort_values('prob_churn', ascending=False).head(20)

    # Seleccionar columnas seguras
    cols = ['customerID', 'tenure', 'MonthlyCharges', 'prob_churn']
    if 'Contract_Month-to-month' in top_risk.columns:
        cols.append('Contract_Month-to-month')
    df_show = top_risk[cols].copy()
    if 'Contract_Month-to-month' in df_show.columns:
        df_show = df_show.rename(columns={'Contract_Month-to-month': 'Month-to-month'})
    st.dataframe(df_show)
    st.caption("Los 20 clientes con mayor riesgo según el modelo actual.")