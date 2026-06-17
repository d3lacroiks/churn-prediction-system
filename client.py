import requests
import json

# Datos de ejemplo: un cliente con alta probabilidad de churn
data = {
    "tenure": 1,
    "MonthlyCharges": 100,
    "TotalCharges": 100,
    "gender_Male": 1,
    "SeniorCitizen": 0,
    "Partner_Yes": 0,
    "Dependents_Yes": 0,
    "PhoneService_Yes": 1,
    "MultipleLines_Yes": 1,
    "InternetService_Fiber optic": 1,
    "InternetService_No": 0,
    "OnlineSecurity_Yes": 0,
    "OnlineBackup_Yes": 0,
    "DeviceProtection_Yes": 0,
    "TechSupport_Yes": 0,
    "StreamingTV_Yes": 0,
    "StreamingMovies_Yes": 0,
    "Contract_One year": 0,
    "Contract_Two year": 0,
    "PaperlessBilling_Yes": 1,
    "PaymentMethod_Credit card (automatic)": 0,
    "PaymentMethod_Electronic check": 1,
    "PaymentMethod_Mailed check": 0,
    "AvgMonthlyCharges": 100,
    "ChargesToMonthlyRatio": 1,
    "Tenure_x_Monthly": 100,
    "IsNewCustomer": 1,
    "NumExtraServices": 0
}

resp = requests.post('http://localhost:5000/predict', json=data)
print("Respuesta:", resp.json())