import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import load_model, build_input, run_prediction, FEATURES, DEFAULTS


def test_model_loads():
    model, le = load_model()
    assert model is not None
    assert le is not None


def test_label_encoder_classes():
    _, le = load_model()
    classes = le.classes_.tolist()
    assert 'Dropout' in classes
    assert 'Graduate' in classes
    assert 'Enrolled' in classes


def test_build_input_returns_dataframe():
    df = build_input({})
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, len(FEATURES))


def test_build_input_uses_defaults():
    df = build_input({})
    assert df['Age at enrollment'].iloc[0] == DEFAULTS['Age at enrollment']


def test_build_input_overrides_defaults():
    df = build_input({'Age at enrollment': 25, 'Gender': 0})
    assert df['Age at enrollment'].iloc[0] == 25
    assert df['Gender'].iloc[0] == 0


def test_build_input_ignores_unknown_keys():
    df = build_input({'unknown_field': 99})
    assert df.shape == (1, len(FEATURES))


def test_run_prediction_returns_valid_label():
    model, le = load_model()
    result, error = run_prediction(model, le, {})
    assert error is None
    assert result['prediction'] in ['Dropout', 'Enrolled', 'Graduate']


def test_run_prediction_confidence_sums_to_one():
    model, le = load_model()
    result, error = run_prediction(model, le, {})
    assert error is None
    total = sum(result['confidence'].values())
    assert abs(total - 1.0) < 0.01


def test_run_prediction_risk_level():
    model, le = load_model()
    result, error = run_prediction(model, le, {})
    assert error is None
    assert result['risk_level'] in ['High', 'Medium', 'Low']


def test_run_prediction_high_performing_student():
    model, le = load_model()
    data = {
        'Curricular units 2nd sem (approved)': 6,
        'Curricular units 2nd sem (grade)': 16.0,
        'Curricular units 1st sem (approved)': 6,
        'Curricular units 1st sem (grade)': 15.0,
        'Tuition fees up to date': 1,
        'Debtor': 0,
        'Scholarship holder': 1
    }
    result, error = run_prediction(model, le, data)
    assert error is None
    assert result['prediction'] in ['Graduate', 'Enrolled']


def test_run_prediction_at_risk_student():
    model, le = load_model()
    data = {
        'Curricular units 2nd sem (approved)': 0,
        'Curricular units 2nd sem (grade)': 0.0,
        'Curricular units 1st sem (approved)': 0,
        'Curricular units 1st sem (grade)': 0.0,
        'Tuition fees up to date': 0,
        'Debtor': 1
    }
    result, error = run_prediction(model, le, data)
    assert error is None
    assert result['prediction'] in ['Dropout', 'Enrolled']


def test_top_confidence_is_float():
    model, le = load_model()
    result, error = run_prediction(model, le, {})
    assert error is None
    assert isinstance(result['top_confidence'], float)
    assert 0.0 <= result['top_confidence'] <= 1.0
