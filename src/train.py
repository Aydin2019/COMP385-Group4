import json
from datetime import datetime

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import ARTIFACT_DIR, RANDOM_SEED, CV_FOLDS
from data_prep import load_data, make_splits, build_preprocessor


def manual_cv_scores(estimator, X, y, sample_weights, cv):
    scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_fold_train = X.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]
        y_fold_train = y[train_idx]
        y_fold_val = y[val_idx]
        sw_fold_train = sample_weights[train_idx]

        model = clone(estimator)
        model.fit(X_fold_train, y_fold_train, clf__sample_weight=sw_fold_train)
        preds = model.predict(X_fold_val)
        score = f1_score(y_fold_val, preds, average="macro")
        scores.append(score)

    return scores


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X_train, X_test, y_train, y_test, label_encoder, categorical_cols, numeric_cols = make_splits(df)

    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    model_candidates = {
        "LogisticRegression": (
            LogisticRegression(
                max_iter=4000,
                solver="lbfgs",
                multi_class="multinomial",
                random_state=RANDOM_SEED,
            ),
            {
                "clf__C": [0.1, 1.0, 3.0, 10.0]
            },
        ),
        "RandomForest": (
            RandomForestClassifier(
                random_state=RANDOM_SEED,
                n_jobs=-1,
                class_weight="balanced_subsample",
            ),
            {
                "clf__n_estimators": [300, 500],
                "clf__max_depth": [None, 15, 25],
                "clf__min_samples_leaf": [1, 3, 5],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                objective="multi:softprob",
                num_class=len(label_encoder.classes_),
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            {
                "clf__n_estimators": [300, 500],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__max_depth": [4, 6, 8],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.8, 1.0],
            },
        ),
    }

    all_results = {}
    best_name = None
    best_estimator = None
    best_mean = -1

    for model_name, (model, param_grid) in model_candidates.items():
        pipeline = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("clf", model),
            ]
        )

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=1,
        )

        grid.fit(X_train, y_train, clf__sample_weight=sample_weights)

        best_pipeline_for_model = grid.best_estimator_
        cv_scores = manual_cv_scores(
            best_pipeline_for_model,
            X_train,
            y_train,
            sample_weights,
            cv,
        )

        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        all_results[model_name] = {
            "best_params": grid.best_params_,
            "grid_best_f1_macro": float(grid.best_score_),
            "cv_f1_macro_mean": cv_mean,
            "cv_f1_macro_std": cv_std,
            "cv_f1_macro_scores": [float(x) for x in cv_scores],
        }

        if cv_mean > best_mean:
            best_mean = cv_mean
            best_name = model_name
            best_estimator = best_pipeline_for_model

    joblib.dump(best_estimator, ARTIFACT_DIR / "best_model.joblib")
    joblib.dump(label_encoder, ARTIFACT_DIR / "label_encoder.joblib")

    metadata = {
        "selected_model": best_name,
        "selection_metric": "cv_f1_macro_mean",
        "cv_f1_macro_mean": float(best_mean),
        "cv_folds": CV_FOLDS,
        "random_seed": RANDOM_SEED,
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "all_model_results": all_results,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    with open(ARTIFACT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Training complete.")
    print(f"Best model: {best_name}")
    print(f"CV macro-F1: {best_mean}")


if __name__ == "__main__":
    main()