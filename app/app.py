import os

import joblib
import streamlit as st

from app.utils import _EXPECTED_COLS, build_input_frame

_DIR = os.path.dirname(os.path.abspath(__file__))

# Bounds derived from the Telco training dataset
_MAX_TENURE = 72
_MIN_MONTHLY = 18.25
_MAX_MONTHLY = 118.75
_DEFAULT_MONTHLY = 70.35   # median
_MAX_TOTAL = 8684.80
_CONSISTENCY_TOLERANCE = 500.0  # dollars


@st.cache_resource
def load_model():
    model_path = os.path.join(_DIR, '..', 'models', 'churn_pipeline.pkl')
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(
            f"Model artefact not found: `{os.path.normpath(model_path)}`. "
            "Run the training notebook to generate it."
        )
        st.stop()


pipeline = load_model()

st.title("Telco Churn Predictor")
st.write("Enter customer details below to predict churn:")

st.subheader("Account & Billing")
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner", ["No", "Yes"])
dependents = st.selectbox("Has Dependents", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, _MAX_TENURE, 24)
monthly = st.number_input(
    "Monthly Charges ($)",
    min_value=_MIN_MONTHLY, max_value=_MAX_MONTHLY, value=_DEFAULT_MONTHLY,
)
total = st.number_input(
    "Total Charges ($)",
    min_value=0.0, max_value=_MAX_TOTAL,
    value=round(tenure * _DEFAULT_MONTHLY, 2),
)
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

if tenure == 0 and total > 0:
    st.warning(
        f"Total Charges is ${total:,.2f} but tenure is 0 months — "
        "a new customer has not yet been billed. Consider setting Total Charges to 0."
    )
elif tenure > 0:
    expected = round(tenure * monthly, 2)
    if abs(total - expected) > _CONSISTENCY_TOLERANCE:
        st.warning(
            f"Total Charges (${total:,.2f}) differs from "
            f"tenure × Monthly Charges (${expected:,.2f}) "
            f"by more than ${_CONSISTENCY_TOLERANCE:,.0f}. "
            "Verify these values before predicting."
        )

if st.button("Predict Churn"):
    input_df = build_input_frame(
        gender=gender, senior=senior, partner=partner, dependents=dependents,
        tenure=tenure, monthly=monthly, total=total,
        contract=contract, paperless=paperless, payment=payment,
        phone_service=phone_service, multiple_lines=multiple_lines,
        internet_service=internet_service, online_security=online_security,
        online_backup=online_backup, device_protection=device_protection,
        tech_support=tech_support, streaming_tv=streaming_tv,
        streaming_movies=streaming_movies,
    )

    missing = _EXPECTED_COLS - set(input_df.columns)
    if missing:
        st.error(f"Internal error — missing columns: {missing}")
        st.stop()

    prediction = pipeline.predict(input_df)[0]
    proba = pipeline.predict_proba(input_df)[0][1]
    st.subheader("Prediction Result:")
    st.write(f"**Customer will {'CHURN' if prediction == 1 else 'NOT churn'}**")
    st.write(f"Churn probability: **{proba:.2%}**")
