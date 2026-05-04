# 📊 Customer Churn Prediction (ML + Streamlit)

## 📌 Overview

This project predicts whether a customer will churn using machine learning classification models. It helps businesses identify high-risk customers and take preventive actions.

---

## ⚙️ Tech Stack

* Python
* Scikit-learn
* Streamlit

---

## 🚀 Features

* Data preprocessing (missing values, duplicates)
* Exploratory Data Analysis (EDA)
* Encoding categorical features
* Handling imbalanced data using SMOTE
* Feature scaling using StandardScaler
* Model training (Logistic Regression, KNN, Random Forest)
* Model evaluation using classification metrics
* Interactive web app using Streamlit

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

Logistic Regression was selected because it achieved the highest F1-score and ROC-AUC, and higher recall, which is critical for identifying churn customers.

---

## 🧠 Business Insight

* High monthly charges → Higher churn
* Short tenure → Higher churn
* Lack of support → Higher churn

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
* scaler.pkl → preprocessing
* requirements.txt → dependencies

---

## 🎯 Future Improvements

* Add full feature-based UI
* Improve model with XGBoost
* Add customer segmentation
