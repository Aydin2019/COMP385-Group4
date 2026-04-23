"""
Integration Tests – COMP385 Group 4
AI-Driven Student Performance Prediction System

These tests validate end-to-end flows across the full stack:
  - Multi-step workflows (health → features → predict)
  - Data pipeline integrity (API input → model → structured response)
  - Consistency and repeatability of predictions
  - Edge cases and boundary inputs
  - CORS and content-type handling
  - Feature contract between /features and predict.py
"""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from predict import FEATURES, DEFAULTS


# ─────────────────────────────────────────────
# Fixture
# ─────────────────────────────────────────────

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


# ─────────────────────────────────────────────
# 1. Full Workflow Integration
# ─────────────────────────────────────────────

def test_full_workflow_health_then_features_then_predict(client):
    """
    Integration: simulate a real client session.
    Step 1 – confirm API is alive.
    Step 2 – fetch the feature list.
    Step 3 – build a payload from those features and call /predict.
    All three must succeed and the prediction must be a valid class.
    """
    # Step 1: health check
    r1 = client.get('/health')
    assert r1.status_code == 200
    assert json.loads(r1.data)['status'] == 'ok'

    # Step 2: get feature list
    r2 = client.get('/features')
    assert r2.status_code == 200
    features = json.loads(r2.data)['features']
    assert len(features) == 34

    # Step 3: build payload from first 5 features using defaults and predict
    payload = {f: DEFAULTS[f] for f in features[:5] if f in DEFAULTS}
    r3 = client.post('/predict', content_type='application/json',
                     data=json.dumps(payload))
    assert r3.status_code == 200
    result = json.loads(r3.data)
    assert result['prediction'] in ['Dropout', 'Enrolled', 'Graduate']


def test_features_endpoint_matches_predict_feature_list(client):
    """
    Integration: the /features endpoint must return exactly the same
    34 features that predict.py uses internally. A mismatch would break
    any client that builds payloads from /features.
    """
    r = client.get('/features')
    api_features = json.loads(r.data)['features']
    assert api_features == FEATURES


def test_predict_uses_all_defaults_when_body_is_empty(client):
    """
    Integration: empty POST body must still produce a valid prediction
    by falling back to DEFAULTS. Validates the default-fill pipeline.
    """
    r = client.post('/predict', content_type='application/json',
                    data=json.dumps({}))
    assert r.status_code == 200
    data = json.loads(r.data)
    assert 'prediction' in data
    assert 'confidence' in data
    assert 'risk_level' in data
    assert 'top_confidence' in data


# ─────────────────────────────────────────────
# 2. Data Pipeline Integrity
# ─────────────────────────────────────────────

def test_confidence_scores_are_valid_probability_distribution(client):
    """
    Integration: confidence values returned from the model must be
    non-negative, sum to ~1.0, and cover all three classes.
    """
    r = client.post('/predict', content_type='application/json',
                    data=json.dumps({}))
    data = json.loads(r.data)
    conf = data['confidence']

    assert set(conf.keys()) == {'Dropout', 'Enrolled', 'Graduate'}
    assert all(v >= 0.0 for v in conf.values())
    assert abs(sum(conf.values()) - 1.0) < 0.01


def test_top_confidence_matches_max_of_confidence_dict(client):
    """
    Integration: top_confidence must equal the highest value in the
    confidence dict — validates internal consistency of the response.
    """
    r = client.post('/predict', content_type='application/json',
                    data=json.dumps({}))
    data = json.loads(r.data)
    expected_top = max(data['confidence'].values())
    assert abs(data['top_confidence'] - expected_top) < 0.0001


def test_risk_level_is_consistent_with_prediction(client):
    """
    Integration: risk_level must follow the business rule:
      Dropout → High, Enrolled → Medium, Graduate → Low.
    """
    r = client.post('/predict', content_type='application/json',
                    data=json.dumps({}))
    data = json.loads(r.data)
    expected_map = {'Dropout': 'High', 'Enrolled': 'Medium', 'Graduate': 'Low'}
    assert data['risk_level'] == expected_map[data['prediction']]


def test_response_structure_is_complete(client):
    """
    Integration: every required response key must be present and
    correctly typed — validates the full output contract.
    """
    r = client.post('/predict', content_type='application/json',
                    data=json.dumps({}))
    data = json.loads(r.data)

    assert isinstance(data['prediction'], str)
    assert isinstance(data['risk_level'], str)
    assert isinstance(data['confidence'], dict)
    assert isinstance(data['top_confidence'], float)


# ─────────────────────────────────────────────
# 3. Prediction Consistency
# ─────────────────────────────────────────────

def test_same_input_produces_same_prediction(client):
    """
    Integration: the model must be deterministic — identical payloads
    must return identical predictions across repeated calls.
    """
    payload = {
        'Age at enrollment': 22,
        'Curricular units 1st sem (approved)': 5,
        'Curricular units 1st sem (grade)': 13.0,
        'Curricular units 2nd sem (approved)': 5,
        'Curricular units 2nd sem (grade)': 13.0,
        'Tuition fees up to date': 1,
        'Debtor': 0
    }
    results = []
    for _ in range(3):
        r = client.post('/predict', content_type='application/json',
                        data=json.dumps(payload))
        results.append(json.loads(r.data)['prediction'])

    assert len(set(results)) == 1, "Model returned different results for identical inputs"


def test_high_performing_student_predicts_graduate_or_enrolled(client):
    """
    Integration: a student with strong academic signals should not be
    predicted as Dropout — validates model direction against domain logic.
    """
    payload = {
        'Curricular units 1st sem (approved)': 6,
        'Curricular units 1st sem (grade)': 16.5,
        'Curricular units 2nd sem (approved)': 6,
        'Curricular units 2nd sem (grade)': 17.0,
        'Tuition fees up to date': 1,
        'Debtor': 0,
        'Scholarship holder': 1
    }
    r = client.post('/predict', content_type='application/json',
                    data=json.dumps(payload))
    data = json.loads(r.data)
    assert data['prediction'] in ['Graduate', 'Enrolled']


def test_at_risk_student_predicts_dropout_or_enrolled(client):
    """
    Integration: a student with zero academic progress and financial
    flags should predict Dropout or Enrolled — not Graduate.
    """
    payload = {
        'Curricular units 1st sem (approved)': 0,
        'Curricular units 1st sem (grade)': 0.0,
        'Curricular units 2nd sem (approved)': 0,
        'Curricular units 2nd sem (grade)': 0.0,
        'Tuition fees up to date': 0,
        'Debtor': 1
    }
    r = client.post('/predict', content_type='application/json',
                    data=json.dumps(payload))
    data = json.loads(r.data)
    assert data['prediction'] in ['Dropout', 'Enrolled']


# ─────────────────────────────────────────────
# 4. Edge Cases and Boundary Inputs
# ─────────────────────────────────────────────

def test_predict_with_all_34_features_provided(client):
    """
    Integration: sending a complete payload with all 34 features must
    succeed — validates that no feature causes a pipeline failure.
    """
    payload = {k: v for k, v in DEFAULTS.items()}
    r = client.post('/predict', content_type='application/json',
                    data=json.dumps(payload))
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data['prediction'] in ['Dropout', 'Enrolled', 'Graduate']


def test_predict_with_extra_unknown_fields_ignored(client):
    """
    Integration: unknown keys in the payload must be silently ignored —
    the API should not crash or return an error.
    """
    payload = {
        'Age at enrollment': 21,
        'nonexistent_field': 'garbage',
        'another_fake_key': 99999
    }
    r = client.post('/predict', content_type='application/json',
                    data=json.dumps(payload))
    assert r.status_code == 200
    assert 'prediction' in json.loads(r.data)


def test_predict_with_minimum_age(client):
    """Integration: minimum realistic age at enrollment (17)."""
    r = client.post('/predict', content_type='application/json',
                    data=json.dumps({'Age at enrollment': 17}))
    assert r.status_code == 200
    assert json.loads(r.data)['prediction'] in ['Dropout', 'Enrolled', 'Graduate']


def test_predict_with_maximum_age(client):
    """Integration: older student (60+) — model must handle without error."""
    r = client.post('/predict', content_type='application/json',
                    data=json.dumps({'Age at enrollment': 65}))
    assert r.status_code == 200
    assert json.loads(r.data)['prediction'] in ['Dropout', 'Enrolled', 'Graduate']


# ─────────────────────────────────────────────
# 5. Error Handling and Content-Type
# ─────────────────────────────────────────────

def test_predict_wrong_content_type_returns_400(client):
    """Integration: non-JSON content-type must be rejected with 400."""
    r = client.post('/predict', content_type='text/plain', data='hello')
    assert r.status_code == 400
    data = json.loads(r.data)
    assert 'error' in data


def test_predict_no_content_type_returns_400(client):
    """Integration: missing content-type header must return 400."""
    r = client.post('/predict')
    assert r.status_code == 400


def test_health_endpoint_is_get_only(client):
    """Integration: /health should not accept POST requests."""
    r = client.post('/health')
    assert r.status_code == 405


def test_features_endpoint_is_get_only(client):
    """Integration: /features should not accept POST requests."""
    r = client.post('/features')
    assert r.status_code == 405


# ─────────────────────────────────────────────
# 6. Batch Simulation
# ─────────────────────────────────────────────

def test_multiple_sequential_predictions_all_valid(client):
    """
    Integration: simulate an advisor running predictions on 5 different
    student profiles back-to-back. All must return valid responses.
    """
    profiles = [
        {'Age at enrollment': 18, 'Curricular units 2nd sem (approved)': 0, 'Tuition fees up to date': 0},
        {'Age at enrollment': 22, 'Curricular units 2nd sem (approved)': 5, 'Tuition fees up to date': 1},
        {'Age at enrollment': 30, 'Curricular units 2nd sem (approved)': 6, 'Scholarship holder': 1},
        {'Age at enrollment': 45, 'Debtor': 1, 'Curricular units 1st sem (grade)': 0.0},
        {'Age at enrollment': 20, 'Gender': 0, 'Curricular units 2nd sem (grade)': 14.0},
    ]
    for profile in profiles:
        r = client.post('/predict', content_type='application/json',
                        data=json.dumps(profile))
        assert r.status_code == 200
        data = json.loads(r.data)
        assert data['prediction'] in ['Dropout', 'Enrolled', 'Graduate']
        assert data['risk_level'] in ['High', 'Medium', 'Low']
        assert abs(sum(data['confidence'].values()) - 1.0) < 0.01
