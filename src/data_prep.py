import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from config import DATA_PATH, TARGET_COL, RANDOM_SEED

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def make_splits(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Columns: {list(df.columns)}")

    X = df.drop(columns=[TARGET_COL])
    y_raw = df[TARGET_COL].astype(str)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y
    )
    return X_train, X_test, y_train, y_test, le

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Dataset columns are numeric-coded features (int/float). We treat all as numeric.
    num_cols = X.columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols)
    ])

    return preprocessor
