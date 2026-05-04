import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load artifacts
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
feature_cols = joblib.load("features.pkl")

st.set_page_config(page_title="Customer Churn Predictor")
st.title("📊 Customer Churn Prediction")
st.markdown("### Enter customer details 👇")

# ---------- INPUTS ----------
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", 0, 72, 12)

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

PaymentMethod = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

MonthlyCharges = st.number_input("Monthly Charges", 18.0, 120.0, 70.0)
TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

# ---------- BUILD DATAFRAME ----------
input_dict = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
}

df_input = pd.DataFrame([input_dict])

# ---------- APPLY SAME ENCODING ----------
for col, le in encoders.items():
    if col in df_input.columns:
        df_input[col] = le.transform(df_input[col])

# Ensure correct column order
df_input = df_input[feature_cols]

# ---------- PREDICTION ----------
if st.button("Predict Churn"):
    try:
        X = scaler.transform(df_input.values)
        pred = model.predict(X)[0]

        if pred == 1:
            st.error("⚠️ High chance of churn")
        else:
            st.success("✅ Customer likely to stay")

    except Exception as e:
        st.error(f"Error: {e}")