# 📊 Customer Churn Prediction (ML + Streamlit)

## 📌 Overview

This project predicts whether a telecom customer will churn using machine learning. It includes full preprocessing, handling class imbalance, model training, and a Streamlit web app for real-time predictions.

---

## ⚙️ Tech Stack

* Python
* Scikit-learn
* Streamlit

---

## 🚀 Features

* Data cleaning (missing values, duplicates)
* Proper type handling (e.g., TotalCharges conversion)
* Encoding of categorical variables
* Handling class imbalance using SMOTE
* Feature scaling with StandardScaler
* Multiple models trained (Logistic Regression, KNN, Random Forest)
* Evaluation with classification metrics
* Deployment using Streamlit

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix

---

## 🏆 Best Model

Logistic Regression selected based on:

* Highest F1-score
* Highest ROC-AUC
* Higher recall (important for detecting churn)

---

## 🧠 Business Insight

* High monthly charges → higher churn
* Short tenure → higher churn
* Lack of tech support → higher churn

---

## ▶️ Run Locally

pip install -r requirements.txt
streamlit run app.py

---

## 🌐 Deployment

Deployed using Streamlit Cloud

---

## 📁 Project Structure

* app.py → Streamlit web app
* churn_model.pkl → trained model
* scaler.pkl → feature scaling
* encoders.pkl → label encoders
* features.pkl → feature order
* requirements.txt → dependencies

---

## 🎯 Future Improvements

* Try advanced models (XGBoost)
* Add feature importance visualization
* Improve UI/UX
