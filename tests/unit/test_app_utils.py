from app.utils import _EXPECTED_COLS, build_input_frame


def _kwargs(**overrides):
    base = dict(
        gender="Female", senior="No", partner="No", dependents="No",
        tenure=24, monthly=70.35, total=1688.40,
        contract="Month-to-month", paperless="Yes",
        payment="Electronic check", phone_service="Yes",
        multiple_lines="No", internet_service="Fiber optic",
        online_security="No", online_backup="No", device_protection="No",
        tech_support="No", streaming_tv="No", streaming_movies="No",
    )
    base.update(overrides)
    return base


def test_build_input_frame_all_fields_matches_training_columns():
    df = build_input_frame(**_kwargs())
    assert set(df.columns) == _EXPECTED_COLS


def test_build_input_frame_no_internet_sets_addons_to_no_internet_service():
    df = build_input_frame(**_kwargs(
        internet_service="No",
        online_security="No internet service",
        online_backup="No internet service",
        device_protection="No internet service",
        tech_support="No internet service",
        streaming_tv="No internet service",
        streaming_movies="No internet service",
    ))
    for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']:
        assert df[col].iloc[0] == "No internet service"
