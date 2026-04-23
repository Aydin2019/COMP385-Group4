from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "dataset.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
OUTPUT_DIR = BASE_DIR / "outputs"

TARGET_COL = "Target"
RANDOM_SEED = 42
CV_FOLDS = 5

# Columns that are coded as categories or indicator flags in your dataset,
# even though many are stored numerically in the CSV.
KNOWN_CATEGORICAL_COLUMNS = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]