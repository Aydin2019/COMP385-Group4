# COMP385 Iteration #1 — Student Academic Success Prediction

This repo trains and evaluates ML models to predict student outcome using the Kaggle dataset:
**Predict students' dropout and academic success**.

## Dataset
- Source: Kaggle (Higher Education predictors of student retention)
- Local file: `data/dataset.csv`
- Target column: `Target` with 3 classes: `Dropout`, `Enrolled`, `Graduate`

## Setup (VS Code)
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Mac: source venv/bin/activate
pip install -r requirements.txt
```

## Run (Iteration #1)
```bash
python src/train.py
python src/evaluate.py
python src/explain.py
python src/bias_check.py
```

## Outputs
- `artifacts/best_model.joblib` (saved preprocessing + model pipeline)
- `artifacts/metadata.json` (best model + params + CV summary)
- `artifacts/metrics.json` (test metrics + confusion matrix)
- `outputs/confusion_matrix.png`
- `outputs/feature_importance.png` (if tree model)
- `outputs/shap_summary.png` (if tree model + SHAP works)
- `outputs/bias_check.txt`

## Notes
Iteration #1 focuses on the **AI capability** (data handling, modeling, validation, explainability, and bias checks).
Full-stack integration is planned for Iteration #2.
