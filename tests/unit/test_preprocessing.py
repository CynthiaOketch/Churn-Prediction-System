import pandas as pd

from app.utils import clean_total_charges


def test_clean_total_charges_zero_tenure_returns_zero():
    df = pd.DataFrame({
        'tenure':       [0, 0, 1, 24],
        'TotalCharges': ['', '  ', '50.00', '1200.00'],
    })
    result = clean_total_charges(df)
    zero_rows = result[result['tenure'] == 0]
    assert (zero_rows['TotalCharges'] == 0.0).all()
    assert result['TotalCharges'].notna().all()
