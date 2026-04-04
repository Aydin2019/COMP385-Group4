# COMP385 – AI Capstone Project | Group 4

## AI-Driven Student Performance Prediction System

**Team:** Ajmal Afzalzada (301413451), Hardiksinh Zala (301371146)
**Program:** Software Engineering Technology – Artificial Intelligence
**Course:** COMP385 – Section 401, Centennial College

---

## Project Overview

A supervised machine learning system that predicts student academic outcomes
(Dropout / Enrolled / Graduate) using early-semester academic and demographic data.
Designed to enable proactive academic advising through early risk detection.

---

## Repository Structure

COMP385-Group4/
├── data/
│   └── dataset.csv
├── src/
│   ├── config.py
│   ├── data_prep.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   └── bias_check.py
├── artifacts/
│   ├── metadata.json
│   └── metrics.json
├── outputs/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── bias_check.txt
├── requirements.txt
└── README.md
---

## Iteration #1 – AI Capability Results

| Model | CV Macro F1 | Std Dev |
|---|---|---|
| Logistic Regression | 0.6803 | 0.0112 |
| Random Forest | 0.6968 | 0.0163 |
| **Gradient Boosting** | **0.7040** | **0.0097** |

**Test Set (n = 885):** Accuracy 76.95% | Macro F1 0.7018 | ROC-AUC 0.8915

---

## Setup Instructions
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run
```bash
python src/train.py       # trains all models, selects best, saves artifacts
python src/evaluate.py    # evaluates on test set, saves metrics.json
python src/explain.py     # generates feature importance and SHAP plots
python src/bias_check.py  # runs fairness analysis across demographic groups
```

---

## Dataset

Source: [Higher Education Predictors of Student Retention – Kaggle](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention)
4,424 student records | 34 features | Target: Dropout / Enrolled / Graduate

---

## Iteration #2 – In Progress

- Flask REST API for model serving
- React dashboard for academic advisors
- Unit tests for all AI use cases