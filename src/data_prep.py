import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from config import DATA_PATH, TARGET_COL, RANDOM_SEED, KNOWN_CATEGORICAL_COLUMNS


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def safe_divide(num, den):
        den = den.replace(0, np.nan)
        return num / den

    # Semester 1
    if "Curricular units 1st sem (approved)" in df.columns and "Curricular units 1st sem (enrolled)" in df.columns:
        df["approval_rate_1st"] = safe_divide(
            df["Curricular units 1st sem (approved)"],
            df["Curricular units 1st sem (enrolled)"]
        ).fillna(0)

        df["enrolled_minus_approved_1st"] = (
            df["Curricular units 1st sem (enrolled)"] -
            df["Curricular units 1st sem (approved)"]
        )

    # Semester 2
    if "Curricular units 2nd sem (approved)" in df.columns and "Curricular units 2nd sem (enrolled)" in df.columns:
        df["approval_rate_2nd"] = safe_divide(
            df["Curricular units 2nd sem (approved)"],
            df["Curricular units 2nd sem (enrolled)"]
        ).fillna(0)

        df["enrolled_minus_approved_2nd"] = (
            df["Curricular units 2nd sem (enrolled)"] -
            df["Curricular units 2nd sem (approved)"]
        )

    # Total academic load / approvals
    if (
        "Curricular units 1st sem (approved)" in df.columns and
        "Curricular units 2nd sem (approved)" in df.columns
    ):
        df["approved_total"] = (
            df["Curricular units 1st sem (approved)"] +
            df["Curricular units 2nd sem (approved)"]
        )

    if (
        "Curricular units 1st sem (enrolled)" in df.columns and
        "Curricular units 2nd sem (enrolled)" in df.columns
    ):
        df["enrolled_total"] = (
            df["Curricular units 1st sem (enrolled)"] +
            df["Curricular units 2nd sem (enrolled)"]
        )

    if "approved_total" in df.columns and "enrolled_total" in df.columns:
        df["approval_rate_total"] = safe_divide(
            df["approved_total"],
            df["enrolled_total"]
        ).fillna(0)

    # Grade progression
    if (
        "Curricular units 1st sem (grade)" in df.columns and
        "Curricular units 2nd sem (grade)" in df.columns
    ):
        df["grade_diff_2nd_minus_1st"] = (
            df["Curricular units 2nd sem (grade)"] -
            df["Curricular units 1st sem (grade)"]
        )

        df["grade_avg_1st_2nd"] = (
            df["Curricular units 1st sem (grade)"] +
            df["Curricular units 2nd sem (grade)"]
        ) / 2.0

    # Risk interaction
    if "Debtor" in df.columns and "Tuition fees up to date" in df.columns:
        df["debt_and_not_up_to_date"] = (
            (df["Debtor"] == 1) & (df["Tuition fees up to date"] == 0)
        ).astype(int)

    return df


def detect_column_types(X: pd.DataFrame):
    known_cat = [col for col in KNOWN_CATEGORICAL_COLUMNS if col in X.columns]

    auto_cat = [
        col for col in X.columns
        if str(X[col].dtype) in ("object", "category")
    ]

    categorical_cols = sorted(set(known_cat + auto_cat))
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    return categorical_cols, numeric_cols


def make_splits(df: pd.DataFrame):
    df = add_engineered_features(df)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_encoded,
    )

    categorical_cols, numeric_cols = detect_column_types(X_train)

    return X_train, X_test, y_train, y_test, label_encoder, categorical_cols, numeric_cols


def build_preprocessor(categorical_cols, numeric_cols):
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_cols),
            ("num", numeric_pipeline, numeric_cols),
        ],
        remainder="drop",
    )

    return preprocessor