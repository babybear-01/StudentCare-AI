from __future__ import annotations

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_dataset(name: str) -> pd.DataFrame:
    """
    Load a single dataset from the data directory.
    """
    path = DATA_DIR / name

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    return df


def load_combined_dataset() -> pd.DataFrame:
    """
    Load student-mat.csv and student-por.csv, then combine them
    into a single dataset for one-time training.
    """
    df_mat = load_dataset("student-mat.csv")
    df_por = load_dataset("student-por.csv")

    # Add source/course column
    df_mat = df_mat.copy()
    df_por = df_por.copy()

    df_mat["course"] = "math"
    df_por["course"] = "portuguese"

    # Combine both datasets
    df = pd.concat([df_mat, df_por], ignore_index=True)

    return df


def make_label(df: pd.DataFrame, pass_threshold: int = 10) -> pd.DataFrame:
    """
    Create binary risk label from final grade G3.
    risk = 1 if G3 < pass_threshold, else 0
    """
    if "G3" not in df.columns:
        raise ValueError("Expected column 'G3' in dataset.")

    df = df.copy()
    df["risk"] = (df["G3"] < pass_threshold).astype(int)

    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features (X) and target (y).
    Remove G1, G2, G3, and risk from input features.
    """
    drop_cols = [c for c in ["G1", "G2", "G3", "risk"] if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df["risk"]

    return X, y