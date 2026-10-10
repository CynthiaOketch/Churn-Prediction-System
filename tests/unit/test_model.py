import os

import joblib
import pandas as pd
import pytest

_MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


@pytest.fixture(scope='module')
def model():
    return joblib.load(os.path.join(_MODELS_DIR, 'random_forest_best.pkl'))


@pytest.fixture(scope='module')
def features():
    return pd.read_csv(os.path.join(_DATA_DIR, 'features.csv'))


def test_load_pipeline_valid_artifact_returns_estimator(model):
    assert hasattr(model, 'predict_proba')


def test_predict_proba_tenure_changes_probability_changes(model, features):
    low = features.nsmallest(1, 'tenure')
    high = features.nlargest(1, 'tenure')
    prob_low = model.predict_proba(low)[0][1]
    prob_high = model.predict_proba(high)[0][1]
    assert prob_low != prob_high
