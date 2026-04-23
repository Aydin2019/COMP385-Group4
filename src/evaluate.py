import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from config import ARTIFACT_DIR, OUTPUT_DIR
from data_prep import load_data, make_splits


def save_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X_train, X_test, y_train, y_test, label_encoder, categorical_cols, numeric_cols = make_splits(df)

    model = joblib.load(ARTIFACT_DIR / "best_model.joblib")

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=list(label_encoder.classes_),
            output_dict=True,
            zero_division=0,
        ),
    }

    cm = confusion_matrix(y_test, y_pred)
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    save_confusion_matrix(cm, list(label_encoder.classes_), cm_path)

    metrics["confusion_matrix"] = cm.tolist()
    metrics["confusion_matrix_path"] = str(cm_path)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        y_test_bin = label_binarize(y_test, classes=np.arange(len(label_encoder.classes_)))
        roc_auc = roc_auc_score(y_test_bin, y_prob, multi_class="ovr")
        metrics["macro_roc_auc_ovr"] = float(roc_auc)

    with open(ARTIFACT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Evaluation complete.")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    if "macro_roc_auc_ovr" in metrics:
        print(f"Macro ROC-AUC (OvR): {metrics['macro_roc_auc_ovr']:.4f}")


if __name__ == "__main__":
    main()