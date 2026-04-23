import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')

FEATURES = [
    'Marital status', 'Application mode', 'Application order', 'Course',
    'Daytime/evening attendance', 'Previous qualification', 'Nacionality',
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", 'Displaced',
    'Educational special needs', 'Debtor', 'Tuition fees up to date',
    'Gender', 'Scholarship holder', 'Age at enrollment', 'International',
    'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate', 'Inflation rate', 'GDP'
]

DEFAULTS = {
    'Marital status': 1,
    'Application mode': 1,
    'Application order': 1,
    'Course': 9238,
    'Daytime/evening attendance': 1,
    'Previous qualification': 1,
    'Nacionality': 1,
    "Mother's qualification": 1,
    "Father's qualification": 1,
    "Mother's occupation": 5,
    "Father's occupation": 5,
    'Displaced': 0,
    'Educational special needs': 0,
    'Debtor': 0,
    'Tuition fees up to date': 1,
    'Gender': 1,
    'Scholarship holder': 0,
    'Age at enrollment': 20,
    'International': 0,
    'Curricular units 1st sem (credited)': 0,
    'Curricular units 1st sem (enrolled)': 6,
    'Curricular units 1st sem (evaluations)': 6,
    'Curricular units 1st sem (approved)': 5,
    'Curricular units 1st sem (grade)': 12.0,
    'Curricular units 1st sem (without evaluations)': 0,
    'Curricular units 2nd sem (credited)': 0,
    'Curricular units 2nd sem (enrolled)': 6,
    'Curricular units 2nd sem (evaluations)': 6,
    'Curricular units 2nd sem (approved)': 5,
    'Curricular units 2nd sem (grade)': 12.0,
    'Curricular units 2nd sem (without evaluations)': 0,
    'Unemployment rate': 10.8,
    'Inflation rate': 1.4,
    'GDP': 1.74
}


def load_model():
    model_path = os.path.join(ARTIFACTS_DIR, 'best_model.joblib')
    encoder_path = os.path.join(ARTIFACTS_DIR, 'label_encoder.joblib')
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    return model, label_encoder


def build_input(data):
    row = DEFAULTS.copy()
    for key, val in data.items():
        if key in row:
            row[key] = val
    df = pd.DataFrame([row])[FEATURES]
    return df


def run_prediction(model, label_encoder, data):
    try:
        df = build_input(data)
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0]
        classes = label_encoder.classes_.tolist()
        label = label_encoder.inverse_transform([pred])[0]
        confidence = {classes[i]: round(float(proba[i]), 4) for i in range(len(classes))}
        risk_level = 'High' if label == 'Dropout' else 'Medium' if label == 'Enrolled' else 'Low'
        return {
            'prediction': label,
            'risk_level': risk_level,
            'confidence': confidence,
            'top_confidence': round(float(max(proba)), 4)
        }, None
    except Exception as e:
        return None, str(e)
