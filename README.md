# 🔮 Sistema de Predicción de Churn

Sistema completo de predicción de abandono de clientes, desarrollado con **Python**, **LightGBM**, **Flask** y **Streamlit**.  
Incluye entrenamiento batch, API REST en tiempo real, monitoreo de drift, reentrenamiento automático y dashboard interactivo.

---

## 📊 Demostración en vivo

| Componente        | URL |
|-------------------|-----|
| **API REST**      | [https://churn-api-d7cg.onrender.com](https://churn-api-d7cg.onrender.com) |
| **Dashboard**     | [https://churn-prediction-system-antghpgwvzwazlbz8vfkoz.streamlit.app](https://churn-prediction-system-antghpgwvzwazlbz8vfkoz.streamlit.app) |

---

## 🧱 Arquitectura

![Arquitectura](arquitectura.png) *(opcional, puedes añadir una imagen más tarde)*

- **train_batch.py** – Entrena un modelo LightGBM con validación temporal y versionado automático.
- **api.py** – API REST en Flask que devuelve probabilidad de churn y factores SHAP.
- **client.py** – Cliente de prueba para consumir la API.
- **dashboard.py** – Dashboard interactivo en Streamlit para explorar clientes en riesgo.
- **monitor.py** – Monitoreo de drift con Population Stability Index (PSI).
- **auto_pipeline.py** – Detección de drift, reentrenamiento automático, validación y promoción/rollback del modelo.

---

## 🚀 Cómo ejecutarlo localmente

1. Clona el repositorio:
   ```bash
   git clone https://github.com/d3lacroiks/churn-prediction-system.git
   cd churn-prediction-system