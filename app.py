import streamlit as st
import joblib
import numpy as np

# Load model + scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Customer Churn Predictor")
st.title("📊 Customer Churn Prediction")
st.markdown("### Enter customer details 👇")

# INPUTS (REALISTIC RANGE)
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
monthly = st.number_input("Monthly Charges", min_value=18.0, max_value=120.0, value=70.0)
total = st.number_input("Total Charges", min_value=0.0, value=500.0)
support = st.number_input("Customer Support Calls", min_value=0, value=1)
senior = st.selectbox("Senior Citizen", [0, 1])

# Prediction
if st.button("Predict Churn"):

    data = np.array([[senior, tenure, monthly, total, support]])

    try:
        data = scaler.transform(data)
        pred = model.predict(data)[0]

        if pred == 1:
            st.error("⚠️ High chance of churn")
        else:
            st.success("✅ Customer likely to stay")

    except:
        st.error("⚠️ Input mismatch with model. Check features.")