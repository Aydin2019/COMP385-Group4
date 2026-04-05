\# COMP385 – AI Capstone Project | Group 4



\## AI-Driven Student Performance Prediction System



\*\*Team:\*\* Ajmal Afzalzada (301413451), Hardiksinh Zala (301371146)

\*\*Program:\*\* Software Engineering Technology – Artificial Intelligence

\*\*Course:\*\* COMP385 – Section 401, Centennial College



\---



\## Project Overview



A supervised machine learning system that predicts student academic outcomes

(Dropout / Enrolled / Graduate) using early-semester academic and demographic data.

Designed to enable proactive academic advising through early risk detection.



\---



\## Repository Structure

COMP385\_Iteration#2\_Group4/

├── artifacts/

│   ├── best\_model.joblib

│   └── label\_encoder.joblib

├── data/

│   └── dataset.csv

├── outputs/

│   ├── confusion\_matrix.png

│   ├── feature\_importance.png

│   └── bias\_check.txt

├── src/                        ← Iteration 1 AI pipeline

│   ├── config.py

│   ├── data\_prep.py

│   ├── train.py

│   ├── evaluate.py

│   ├── explain.py

│   └── bias\_check.py

├── backend/                    ← Iteration 2 Flask API

│   ├── app.py

│   ├── predict.py

│   ├── requirements.txt

│   └── tests/

│       ├── test\_predict.py

│       └── test\_api.py

├── frontend/                   ← Iteration 2 React Dashboard

│   ├── src/

│   │   ├── App.js

│   │   ├── App.css

│   │   └── components/

│   │       ├── Header.js

│   │       ├── PredictionForm.js

│   │       └── ResultCard.js

│   ├── public/

│   └── package.json

└── README.md



\---



\## Iteration 1 – AI Capability Results



| Model | CV Macro F1 | Std Dev |

|---|---|---|

| Logistic Regression | 0.6803 | 0.0112 |

| Random Forest | 0.6968 | 0.0163 |

| \*\*Gradient Boosting\*\* | \*\*0.7040\*\* | \*\*0.0097\*\* |



\*\*Test Set (n = 885):\*\* Accuracy 76.95% | Macro F1 0.7018 | ROC-AUC 0.8915



\---



\## Iteration 2 – Full Stack Setup



\### Backend (Flask API)

```bash

cd backend

python -m venv venv

venv\\Scripts\\activate

pip install flask flask-cors joblib pandas numpy scikit-learn pytest

python app.py

```



API runs on http://localhost:5000



\### API Endpoints



| Method | Endpoint | Description |

|---|---|---|

| GET | /health | Check API status |

| GET | /features | List all 34 input features |

| POST | /predict | Generate student risk prediction |



\### Frontend (React Dashboard)

```bash

cd frontend

npm install

npm start

```



Dashboard runs on http://localhost:3000



\### Run Unit Tests

```bash

cd backend

venv\\Scripts\\activate

python -m pytest tests/ -v

```



\*\*21/21 tests passing\*\*



\---



\## Dataset



Source: \[Higher Education Predictors of Student Retention – Kaggle](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention)

4,424 student records | 34 features | Target: Dropout / Enrolled / Graduate

