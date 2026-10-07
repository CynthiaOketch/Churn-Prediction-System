import os
import joblib
import pandas as pd
import streamlit as st

_DIR = os.path.dirname(os.path.abspath(__file__))
pipeline = joblib.load(os.path.join(_DIR, '..', 'models', 'churn_pipeline.pkl'))

_EXPECTED_COLS = {
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod',
}

st.title("Telco Churn Predictor")
st.write("Enter customer details below to predict churn:")

st.subheader("Account & Billing")
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner", ["No", "Yes"])
dependents = st.selectbox("Has Dependents", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72, 24)
monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=65.0)
total = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * 65))
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)",
])

st.subheader("Phone Service")
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
if phone_service == "Yes":
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
else:
    multiple_lines = "No phone service"
    st.info("Multiple Lines: No phone service")

st.subheader("Internet Service")
internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
if internet_service != "No":
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
else:
    online_security = online_backup = device_protection = "No internet service"
    tech_support = streaming_tv = streaming_movies = "No internet service"
    st.info("Add-on services: No internet service")

input_df = pd.DataFrame([{
    'SeniorCitizen':    1 if senior == "Yes" else 0,
    'tenure':           tenure,
    'MonthlyCharges':   monthly,
    'TotalCharges':     total,
    'gender':           gender,
    'Partner':          partner,
    'Dependents':       dependents,
    'PhoneService':     phone_service,
    'MultipleLines':    multiple_lines,
    'InternetService':  internet_service,
    'OnlineSecurity':   online_security,
    'OnlineBackup':     online_backup,
    'DeviceProtection': device_protection,
    'TechSupport':      tech_support,
    'StreamingTV':      streaming_tv,
    'StreamingMovies':  streaming_movies,
    'Contract':         contract,
    'PaperlessBilling': paperless,
    'PaymentMethod':    payment,
}])

assert set(input_df.columns) == _EXPECTED_COLS, (
    f"Column mismatch: {set(input_df.columns) ^ _EXPECTED_COLS}"
)

if st.button("Predict Churn"):
    prediction = pipeline.predict(input_df)[0]
    proba = pipeline.predict_proba(input_df)[0][1]
    st.subheader("Prediction Result:")
    st.write(f"**Customer will {'CHURN' if prediction == 1 else 'NOT churn'}**")
    st.write(f"Churn probability: **{proba:.2%}**")
