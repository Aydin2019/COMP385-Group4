import json
from datetime import datetime

import joblib
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from config import ARTIFACT_DIR, RANDOM_SEED, CV_FOLDS
from data_prep import load_data, make_splits, build_preprocessor


def manual_cv_scores(estimator, X, y, cv):
    scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_fold_train = X.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]
        y_fold_train = y[train_idx]
        y_fold_val = y[val_idx]

        model = clone(estimator)
        model.fit(X_fold_train, y_fold_train)
        preds = model.predict(X_fold_val)
        score = f1_score(y_fold_val, preds, average="macro")
        scores.append(score)

    return scores


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X_train, X_test, y_train, y_test, label_encoder, categorical_cols, numeric_cols = make_splits(df)

    preprocessor = build_preprocessor(categorical_cols, numeric_cols)

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("over", RandomOverSampler(random_state=RANDOM_SEED)),
            ("clf", XGBClassifier(
                objective="multi:softprob",
                num_class=len(label_encoder.classes_),
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )),
        ]
    )

    param_grid = {
        "clf__n_estimators": [200, 300, 400],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__max_depth": [4, 6],
        "clf__min_child_weight": [1, 3],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
        "clf__reg_lambda": [1.0, 2.0],
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    grid.fit(X_train, y_train)

    best_pipeline = grid.best_estimator_

    cv_scores = manual_cv_scores(
        best_pipeline,
        X_train,
        y_train,
        cv,
    )

    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))

    joblib.dump(best_pipeline, ARTIFACT_DIR / "best_model.joblib")
    joblib.dump(label_encoder, ARTIFACT_DIR / "label_encoder.joblib")

    metadata = {
        "selected_model": "XGBoost_Phase3",
        "selection_metric": "cv_f1_macro_mean",
        "cv_f1_macro_mean": cv_mean,
        "cv_f1_macro_std": cv_std,
        "cv_folds": CV_FOLDS,
        "random_seed": RANDOM_SEED,
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "best_params": grid.best_params_,
        "grid_best_f1_macro": float(grid.best_score_),
        "cv_f1_macro_scores": [float(x) for x in cv_scores],
        "oversampling": "RandomOverSampler",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    with open(ARTIFACT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Training complete.")
    print("Best model: XGBoost_Phase3")
    print(f"Grid best macro-F1: {grid.best_score_}")
    print(f"CV macro-F1: {cv_mean}")
    print(f"CV std: {cv_std}")


if __name__ == "__main__":
    main()