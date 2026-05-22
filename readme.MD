# Algerian Forest Fires - FWI Prediction Hub

A production-ready Full-Stack Machine Learning Web Application built with the **MERN-equivalent Flask architecture** for Python. This system predicts the **Fire Weather Index (FWI)**—a critical component of the Canadian Forest Fire Weather Index System—utilizing meteorological observations from the Bejaia and Sidi Bel-Abbes regions of Algeria.

The architecture decouples data transformation from model optimization by serializing independent `StandardScaler` and `Ridge` regression artifacts, preventing data leakage and ensuring mathematical consistency during live production inference.

---

## 🏗️ Project Architecture & Layout

```text
FIRST_PROJECT/
│
├── models/
│   ├── ridge.pkl          # Serialized Ridge Linear Regression model coefficients
│   └── scaler.pkl         # Serialized StandardScaler normalization parameters
│
├── templates/
│   ├── index.html         # Portal Landing Page UI
│   └── home.html          # Operational interactive inference dashboard form
│
├── notebooks/             # Research & Data Science sandboxes
│   ├── data_analysis.ipynb
│   ├── feature_sel.ipynb
│   └── model_training.ipynb
│
├── app.py                 # Core production Flask application gateway
├── requirements.txt       # Production execution & wheel library dependencies
└── .gitignore             # Version control exclusion rules (ignores venv/)