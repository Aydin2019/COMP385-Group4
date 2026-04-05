from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import load_model, run_prediction

app = Flask(__name__)
CORS(app)

model, label_encoder = load_model()


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'GradientBoosting'})


@app.route('/predict', methods=['POST'])
def predict():
    if not request.content_type or 'application/json' not in request.content_type:
        return jsonify({'error': 'Content-Type must be application/json'}), 400

    data = request.get_json(silent=True)
    if data is None:
        data = {}

    result, error = run_prediction(model, label_encoder, data)
    if error:
        return jsonify({'error': error}), 400

    return jsonify(result)


@app.route('/features', methods=['GET'])
def features():
    feature_list = [
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
    return jsonify({'features': feature_list})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
