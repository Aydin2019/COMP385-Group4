import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from config import DATA_PATH, TARGET_COL, RANDOM_SEED, KNOWN_CATEGORICAL_COLUMNS


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def detect_column_types(X: pd.DataFrame):
    """
    Split columns into categorical and numeric groups.
    Uses known dataset columns first, then adds any object/category columns.
    """
    known_cat = [col for col in KNOWN_CATEGORICAL_COLUMNS if col in X.columns]

    auto_cat = [
        col for col in X.columns
        if str(X[col].dtype) in ("object", "category")
    ]

    categorical_cols = sorted(set(known_cat + auto_cat))
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    return categorical_cols, numeric_cols


def make_splits(df: pd.DataFrame):
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