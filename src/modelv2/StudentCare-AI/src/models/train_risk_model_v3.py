from pathlib import Path
import tempfile
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ==========================================
# FIND PROJECT ROOT
# ==========================================
CURRENT_FILE = Path(__file__).resolve()

BASE_DIR = None
for parent in CURRENT_FILE.parents:
    if (parent / "data").exists() and (parent / "src").exists() and (parent / "assets").exists():
        BASE_DIR = parent
        break

if BASE_DIR is None:
    raise FileNotFoundError("Could not find project root directory")


# ==========================================
# DATASET CONFIG
# ==========================================
DATASETS = {
    "math": BASE_DIR / "data" / "processed" / "student-mat-id.csv",
    "por": BASE_DIR / "data" / "processed" / "student-por-id.csv",
}


# ==========================================
# STEP 1 MODEL PATHS (FRIEND'S MODELS)
# ==========================================
STEP1_MODEL_PATHS = {
    "math": BASE_DIR / "src" / "modelv2" / "StudentCare-AI" / "src" / "modelv2" / "Weight_feature_mat_2" / "student_model_v2.pkl",
    "por": BASE_DIR / "src" / "modelv2" / "StudentCare-AI" / "src" / "modelv2" / "Weight_feature_por_2" / "student_model_v2.pkl",
}


# ==========================================
# MLFLOW CONFIG
# ==========================================
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "studentcare-step2-risk-model"


# ==========================================
# STEP 1 FEATURES (USE FRIEND'S MODEL)
# ==========================================
STEP1_FEATURES = [
    "G1",
    "G2",
    "studytime",
    "failures",
    "absences",
    "Medu",
    "Fedu",
    "famrel",
    "freetime",
    "goout",
    "nursery",
    "internet",
    "romantic",
    "Dalc",
    "Walc",
]


# ==========================================
# STEP 2B FEATURES (8 CORE FEATURES)
# ==========================================
STEP2_FEATURES = [
    "G1",
    "G2",
    "failures",
    "absences",
    "studytime",
    "goout",
    "Medu",
    "Walc",
]


# ==========================================
# LOAD DATA
# ==========================================
def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    return pd.read_csv(data_path)


# ==========================================
# LOAD STEP 1 MODEL
# ==========================================
def load_step1_model(subject_name: str):
    model_path = STEP1_MODEL_PATHS.get(subject_name)

    if model_path is None:
        raise ValueError(f"Unsupported subject: {subject_name}")

    if not model_path.exists():
        raise FileNotFoundError(f"Step 1 model not found: {model_path}")

    return joblib.load(model_path)


# ==========================================
# PREPROCESS FOR STEP 1
# ==========================================
def preprocess_step1_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    binary_map = {"yes": 1, "no": 0, 1: 1, 0: 0}
    for col in ["nursery", "internet", "romantic"]:
        if col in df.columns:
            df[col] = df[col].map(binary_map)

    return df


# ==========================================
# STEP 1: ADD PREDICTED G3 FROM FRIEND'S MODEL
# ==========================================
def add_predicted_g3(df: pd.DataFrame, subject_name: str) -> pd.DataFrame:
    df = df.copy()
    df_preprocessed = preprocess_step1_features(df)

    missing_cols = [col for col in STEP1_FEATURES if col not in df_preprocessed.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for Step 1: {missing_cols}")

    step1_model = load_step1_model(subject_name)
    X_step1 = df_preprocessed[STEP1_FEATURES].copy()

    df["predicted_g3"] = step1_model.predict(X_step1)
    return df


# ==========================================
# STEP 2A: LOGIC FOR EXPLANATION
# ==========================================
def create_risk_zone_from_predicted_g3(predicted_g3: float) -> str:
    if predicted_g3 < 10:
        return "High Risk Zone"
    elif predicted_g3 < 12:
        return "Medium Risk Zone"
    return "Low Risk Zone"


# ==========================================
# STEP 2 TARGET LABEL
# ==========================================
def create_risk_label_from_g3(g3: float) -> str:
    if g3 < 10:
        return "High Risk"
    elif g3 < 12:
        return "Medium Risk"
    return "Low Risk"


# ==========================================
# PREPARE DATASET FOR STEP 2B
# ==========================================
def prepare_dataset(df: pd.DataFrame, subject_name: str):
    df = df.copy()

    # Step 1: คำนวณ predicted_g3
    df = add_predicted_g3(df, subject_name=subject_name)

    # Step 2A logic
    df["risk_zone"] = df["predicted_g3"].apply(create_risk_zone_from_predicted_g3)

    # Step 2B target
    df["risk_label"] = df["G3"].apply(create_risk_label_from_g3)

    missing_step2_cols = [col for col in STEP2_FEATURES if col not in df.columns]
    if missing_step2_cols:
        raise ValueError(f"Missing required columns for Step 2B: {missing_step2_cols}")

    # ดึงเฉพาะ 8 ฟีเจอร์ที่ระบุใน STEP2_FEATURES
    X = df[STEP2_FEATURES].copy()
    y = df["risk_label"].copy()

    return X, y, STEP2_FEATURES, df


# ==========================================
# BUILD STEP 2B PIPELINE
# ==========================================
def build_pipeline(features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, features),
        ],
        remainder='drop'  # ทิ้งคอลัมน์อื่นที่ไม่ได้ระบุไว้ ป้องกันฟีเจอร์หลุดรอด
    )

    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline


# ==========================================
# SAVE CONFUSION MATRIX FIGURE
# ==========================================
def save_confusion_matrix_figure(cm, class_names, save_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ==========================================
# TRAIN ONE SUBJECT
# ==========================================
def train_one_subject(subject_name: str, data_path: Path):
    print("\n" + "=" * 70)
    print(f"STEP 2B TRAINING SUBJECT : {subject_name}")
    print("=" * 70)
    print(f"DATA_PATH : {data_path}")

    df = load_data(data_path)
    X, y, features, prepared_df = prepare_dataset(df, subject_name)

    print(f"Dataset shape       : X: {X.shape}, y: {y.shape}")
    print("Features used       :", features)
    print("Risk label counts:")
    print(y.value_counts())
    print("\nRisk zone counts (from predicted_g3):")
    print(prepared_df["risk_zone"].value_counts())

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    pipeline = build_pipeline(features)

    with mlflow.start_run(run_name=f"{subject_name}_step2_risk_model"):
        mlflow.log_param("subject", subject_name)
        mlflow.log_param("dataset", data_path.name)
        mlflow.log_param("dataset_rows", int(X.shape[0]))
        mlflow.log_param("dataset_cols", int(X.shape[1]))
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 8)
        mlflow.log_param("min_samples_split", 5)
        mlflow.log_param("min_samples_leaf", 2)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("num_features", len(features))
        mlflow.log_param("features", ", ".join(features))
        mlflow.log_param("step1_model_path", str(STEP1_MODEL_PATHS[subject_name]))

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")
        cm = confusion_matrix(y_test, y_pred)

        report_text = classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            digits=4,
        )

        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("macro_f1", float(macro_f1))
        mlflow.log_metric("weighted_f1", float(weighted_f1))

        print("\n===== STEP 2B EVALUATION =====")
        print("Accuracy     :", round(acc, 4))
        print("Macro F1     :", round(macro_f1, 4))
        print("Weighted F1 :", round(weighted_f1, 4))
        print("\nClassification Report:")
        print(report_text)
        print("Confusion Matrix:")
        print(cm)

        rf_model = pipeline.named_steps["classifier"]
        importances = rf_model.feature_importances_
        feature_importance_df = pd.DataFrame(
            {"feature": features, "importance": importances}
        ).sort_values(by="importance", ascending=False)

        print("\nTop Feature Importances:")
        print(feature_importance_df)

        for _, row in feature_importance_df.iterrows():
            mlflow.log_metric(f"feat_imp_{row['feature']}", float(row["importance"]))

        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            classes_path = tmpdir / f"{subject_name}_classes.txt"
            with open(classes_path, "w", encoding="utf-8") as f:
                for c in label_encoder.classes_:
                    f.write(f"{c}\n")
            mlflow.log_artifact(str(classes_path))

            report_path = tmpdir / f"{subject_name}_classification_report.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            mlflow.log_artifact(str(report_path))

            cm_path = tmpdir / f"{subject_name}_confusion_matrix.png"
            save_confusion_matrix_figure(cm, label_encoder.classes_, cm_path)
            mlflow.log_artifact(str(cm_path))

            fi_csv_path = tmpdir / f"{subject_name}_feature_importance.csv"
            feature_importance_df.to_csv(fi_csv_path, index=False)
            mlflow.log_artifact(str(fi_csv_path))

            fi_plot_path = tmpdir / f"{subject_name}_feature_importance.png"
            plt.figure(figsize=(8, 5))
            plt.bar(
                feature_importance_df["feature"],
                feature_importance_df["importance"],
            )
            plt.title(f"Feature Importances ({subject_name})")
            plt.xlabel("Feature")
            plt.ylabel("Importance")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(fi_plot_path, dpi=150)
            plt.close()
            mlflow.log_artifact(str(fi_plot_path))

        model_path = BASE_DIR / "src" / "models" / f"{subject_name}_step2_risk_model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)

        print(f"\nSaved model to: {model_path}")
        print(f"Run logged to MLflow: {subject_name}_step2_risk_model")


# ==========================================
# MAIN TRAIN
# ==========================================
def train():
    print("STEP 1 + STEP 2B RISK MODEL TRAINING STARTED")
    print("=" * 70)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    for subject_name, data_path in DATASETS.items():
        train_one_subject(subject_name, data_path)

    print("\n" + "=" * 70)
    print("All Step 1 + Step 2B training finished and logged to MLflow.")


if __name__ == "__main__":
    train()