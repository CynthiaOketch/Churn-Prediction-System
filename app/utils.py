import pandas as pd

_EXPECTED_COLS = {
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod',
}


def build_input_frame(
    gender, senior, partner, dependents, tenure, monthly, total,
    contract, paperless, payment, phone_service, multiple_lines,
    internet_service, online_security, online_backup, device_protection,
    tech_support, streaming_tv, streaming_movies,
):
    return pd.DataFrame([{
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


def clean_total_charges(df):
    """Coerce TotalCharges to numeric; set 0 for zero-tenure rows (never billed)."""
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.loc[df['tenure'] == 0, 'TotalCharges'] = 0.0
    assert df['TotalCharges'].notna().all(), "Unexpected NaNs in TotalCharges after imputation"
    return df
