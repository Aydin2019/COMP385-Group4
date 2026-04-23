import os, json
from datetime import datetime
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from config import ARTIFACT_DIR, CV_FOLDS, RANDOM_SEED
from data_prep import load_data, make_splits, build_preprocessor

def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    df = load_data()
    X_train, X_test, y_train, y_test, label_encoder = make_splits(df)
    preprocessor = build_preprocessor(X_train)

    candidates = {
        "LogisticRegression": (
            LogisticRegression(max_iter=4000, n_jobs=None),
            {"clf__C": [0.1, 1.0, 10.0]}
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_SEED),
            {"clf__n_estimators": [200, 400],
             "clf__max_depth": [None, 10, 20]}
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=RANDOM_SEED),
            {"clf__n_estimators": [100, 200],
             "clf__learning_rate": [0.05, 0.1]}
        )
    }

    results = {}
    best_name = None
    best_score = -1e9
    best_estimator = None

    for name, (model, grid) in candidates.items():
        pipe = Pipeline([
            ("prep", preprocessor),
            ("clf", model)
        ])

        search = GridSearchCV(
            pipe,
            param_grid=grid,
            cv=CV_FOLDS,
            scoring="f1_macro",
            n_jobs=-1
        )
        search.fit(X_train, y_train)

        cv_scores = cross_val_score(search.best_estimator_, X_train, y_train, cv=CV_FOLDS, scoring="f1_macro", n_jobs=-1)
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        results[name] = {
            "best_params": search.best_params_,
            "grid_best_f1_macro": float(search.best_score_),
            "cv_f1_macro_mean": cv_mean,
            "cv_f1_macro_std": cv_std,
            "cv_f1_macro_scores": cv_scores.tolist()
        }

        if cv_mean > best_score:
            best_score = cv_mean
            best_name = name
            best_estimator = search.best_estimator_

    # Save best pipeline + label encoder
    joblib.dump(best_estimator, os.path.join(ARTIFACT_DIR, "best_model.joblib"))
    joblib.dump(label_encoder, os.path.join(ARTIFACT_DIR, "label_encoder.joblib"))

    metadata = {
        "selected_model": best_name,
        "selection_metric": "cv_f1_macro_mean",
        "cv_f1_macro_mean": best_score,
        "cv_folds": CV_FOLDS,
        "random_seed": RANDOM_SEED,
        "all_model_results": results,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

    with open(os.path.join(ARTIFACT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Training complete.")
    print("Best model:", best_name)
    print("CV macro-F1:", best_score)

if __name__ == "__main__":
    main()
