import os
import joblib
import pandas as pd
import streamlit as st

_DIR = os.path.dirname(os.path.abspath(__file__))
pipeline = joblib.load(os.path.join(_DIR, '..', 'models', 'churn_pipeline.pkl'))

st.title("Telco Churn Predictor")
st.write("Enter customer details below to predict churn:")

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

# Service features — hardcoded defaults until issue #2 adds the widgets
_phone_service = "Yes"
_multiple_lines = "No"
_internet_service = "Fiber optic"
_online_security = "No"
_online_backup = "No"
_device_protection = "No"
_tech_support = "No"
_streaming_tv = "No"
_streaming_movies = "No"

input_df = pd.DataFrame([{
    'SeniorCitizen':    1 if senior == "Yes" else 0,
    'tenure':           tenure,
    'MonthlyCharges':   monthly,
    'TotalCharges':     total,
    'gender':           gender,
    'Partner':          partner,
    'Dependents':       dependents,
    'PhoneService':     _phone_service,
    'MultipleLines':    _multiple_lines,
    'InternetService':  _internet_service,
    'OnlineSecurity':   _online_security,
    'OnlineBackup':     _online_backup,
    'DeviceProtection': _device_protection,
    'TechSupport':      _tech_support,
    'StreamingTV':      _streaming_tv,
    'StreamingMovies':  _streaming_movies,
    'Contract':         contract,
    'PaperlessBilling': paperless,
    'PaymentMethod':    payment,
}])

if st.button("Predict Churn"):
    prediction = pipeline.predict(input_df)[0]
    proba = pipeline.predict_proba(input_df)[0][1]
    st.subheader("Prediction Result:")
    st.write(f"**Customer will {'CHURN' if prediction == 1 else 'NOT churn'}**")
    st.write(f"Churn probability: **{proba:.2%}**")
