import sys
import os
import pytest
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'ok'


def test_health_returns_model_name(client):
    response = client.get('/health')
    data = json.loads(response.data)
    assert 'model' in data


def test_features_endpoint(client):
    response = client.get('/features')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'features' in data
    assert len(data['features']) == 34


def test_predict_with_empty_body(client):
    response = client.post('/predict', content_type='application/json', data=json.dumps({}))
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data


def test_predict_returns_valid_class(client):
    payload = {
        'Age at enrollment': 20,
        'Gender': 1,
        'Scholarship holder': 0,
        'Debtor': 0,
        'Tuition fees up to date': 1,
        'Curricular units 1st sem (approved)': 5,
        'Curricular units 1st sem (grade)': 12.5,
        'Curricular units 2nd sem (approved)': 5,
        'Curricular units 2nd sem (grade)': 13.0
    }
    response = client.post('/predict', content_type='application/json', data=json.dumps(payload))
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['prediction'] in ['Dropout', 'Enrolled', 'Graduate']


def test_predict_returns_confidence(client):
    response = client.post('/predict', content_type='application/json', data=json.dumps({}))
    data = json.loads(response.data)
    assert 'confidence' in data
    assert 'Dropout' in data['confidence']
    assert 'Graduate' in data['confidence']
    assert 'Enrolled' in data['confidence']


def test_predict_returns_risk_level(client):
    response = client.post('/predict', content_type='application/json', data=json.dumps({}))
    data = json.loads(response.data)
    assert 'risk_level' in data
    assert data['risk_level'] in ['High', 'Medium', 'Low']


def test_predict_no_json(client):
    response = client.post('/predict')
    assert response.status_code == 400


def test_predict_high_risk_student(client):
    payload = {
        'Curricular units 2nd sem (approved)': 0,
        'Curricular units 2nd sem (grade)': 0.0,
        'Curricular units 1st sem (approved)': 0,
        'Tuition fees up to date': 0,
        'Debtor': 1
    }
    response = client.post('/predict', content_type='application/json', data=json.dumps(payload))
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['prediction'] in ['Dropout', 'Enrolled', 'Graduate']
