# app/app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load('./models/random_forest_best.pkl')

st.title("🔮 Telco Churn Predictor")
st.write("Enter customer details below to predict churn:")

# Define input fields
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner", ["No", "Yes"])
dependents = st.selectbox("Has Dependents", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72, 24)
monthly = st.number_input("Monthly Charges", min_value=0.0)
total = st.number_input("Total Charges", min_value=0.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# Map to model input
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [1 if senior == "Yes" else 0],
    'Partner': [1 if partner == "Yes" else 0],
    'Dependents': [1 if dependents == "Yes" else 0],
    'tenure': [tenure],
    'MonthlyCharges': [monthly],
    'TotalCharges': [total],
    'Contract': [contract],
    'PaperlessBilling': [1 if paperless == "Yes" else 0],
    'PaymentMethod': [payment]
})

# Encode categorical features (must match training preprocessing)
input_data_encoded = pd.get_dummies(input_data)
X_train_cols = pd.read_csv('./data/features.csv').columns
for col in X_train_cols:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0
input_data_encoded = input_data_encoded[X_train_cols]  # align column order

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_data_encoded)[0]
    proba = model.predict_proba(input_data_encoded)[0][1]

    st.subheader("🔍 Prediction Result:")
    st.write(f"**Customer will {'CHURN' if prediction == 1 else 'NOT churn'}**")
    st.write(f"Churn probability: **{proba:.2%}**")
