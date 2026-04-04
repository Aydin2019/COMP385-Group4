import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

from config import ARTIFACT_DIR, OUTPUT_DIR
from data_prep import load_data, make_splits

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()
    X_train, X_test, y_train, y_test, _ = make_splits(df)

    model = joblib.load(os.path.join(ARTIFACT_DIR, "best_model.joblib"))

    # Feature importance for tree models
    clf = model.named_steps.get("clf", None)
    if clf is not None and hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        idx = np.argsort(importances)[-20:]  # top 20
        plt.figure()
        plt.barh(range(len(idx)), importances[idx])
        plt.title("Top 20 Feature Importances")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=200)
        plt.close()
        print("Saved outputs/feature_importance.png")
    else:
        print("Best model does not support feature_importances_. Skipping feature importance plot.")

   
    try:
        import shap
        if clf is not None and hasattr(clf, "feature_importances_"):
            prep = model.named_steps["prep"]
            Xp = prep.transform(X_test)
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(Xp)
            shap.summary_plot(shap_values, Xp, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=200)
            plt.close()
            print("Saved outputs/shap_summary.png")
    except Exception:
      
        pass

if __name__ == "__main__":
    main()
