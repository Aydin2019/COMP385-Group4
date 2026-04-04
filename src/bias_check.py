import os
import joblib
import pandas as pd
from sklearn.metrics import f1_score

from config import ARTIFACT_DIR, OUTPUT_DIR, TARGET_COL
from data_prep import load_data, make_splits

SENSITIVE_COLS = ["Gender", "Scholarship holder", "International", "Debtor"]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data()
    X_train, X_test, y_train, y_test, le = make_splits(df)

    model = joblib.load(os.path.join(ARTIFACT_DIR, "best_model.joblib"))
    y_pred = model.predict(X_test)

    lines = []
    lines.append(f"Overall macro-F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
    lines.append("")

    for col in SENSITIVE_COLS:
        if col in X_test.columns:
            lines.append(f"Bias check by {col}:")
            for grp in sorted(X_test[col].unique().tolist()):
                mask = X_test[col] == grp
                score = f1_score(y_test[mask], y_pred[mask], average='macro')
                lines.append(f"  {col}={grp}: macro-F1={score:.4f} (n={int(mask.sum())})")
            lines.append("")

    out_path = os.path.join(OUTPUT_DIR, "bias_check.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print("Saved outputs/bias_check.txt")

if __name__ == "__main__":
    main()
