from __future__ import annotations

from pathlib import Path
import json
import joblib

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from src.data.preprocess import get_splits

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "model.pkl"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    print("EVALUATE SCRIPT STARTED")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found. Train first: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    bundle = get_splits("student-mat.csv")

    y_pred = model.predict(bundle.X_test)
    y_proba = model.predict_proba(bundle.X_test)[:, 1]

    report = classification_report(bundle.y_test, y_pred, output_dict=True)
    cm = confusion_matrix(bundle.y_test, y_pred).tolist()
    auc = float(roc_auc_score(bundle.y_test, y_proba))

    out = {
        "dataset": "student-mat.csv",
        "roc_auc": auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    out_path = RESULTS_DIR / "eval.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Saved evaluation to: {out_path}")
    print(f"ROC-AUC: {auc:.4f}")


if __name__ == "__main__":
    main()