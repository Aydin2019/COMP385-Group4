import os, json
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

from config import ARTIFACT_DIR, OUTPUT_DIR
from data_prep import load_data, make_splits

def save_confusion_matrix(cm, labels, out_path):
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    df = load_data()
    X_train, X_test, y_train, y_test, _ = make_splits(df)

    model = joblib.load(os.path.join(ARTIFACT_DIR, "best_model.joblib"))
    le = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoder.joblib"))
    labels = le.classes_.tolist()

    y_pred = model.predict(X_test)

    # Basic metrics
    acc = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))

    # Classification report
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    save_confusion_matrix(cm, labels, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    # Multiclass ROC-AUC (One-vs-Rest), if predict_proba exists
    roc_auc_ovr = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            y_bin = label_binarize(y_test, classes=list(range(len(labels))))
            roc_auc_ovr = float(roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr"))
        except Exception:
            roc_auc_ovr = None

    metrics = {
        "dataset": {
            "file": "data/dataset.csv",
            "n_rows": int(df.shape[0]),
            "n_features": int(df.shape[1] - 1),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "labels": labels
        },
        "test_results": {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "roc_auc_ovr_macro": roc_auc_ovr,
            "confusion_matrix": cm.tolist()
        },
        "classification_report": report,
        "outputs": {
            "confusion_matrix_png": "outputs/confusion_matrix.png"
        }
    }

    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Evaluation complete.")
    print("Accuracy:", acc)
    print("Macro F1:", f1_macro)
    if roc_auc_ovr is not None:
        print("ROC-AUC (OVR macro):", roc_auc_ovr)
    print("Saved artifacts/metrics.json and outputs/confusion_matrix.png")

if __name__ == "__main__":
    main()
