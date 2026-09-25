from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset paths: check if 6ab10eb3b23ba_student_resource/dataset exists, else fallback to data/
DATA_STUDENT_DIR = BASE_DIR / "6ab10eb3b23ba_student_resource" / "dataset"
DATA_STANDARD_DIR = BASE_DIR / "data"

if DATA_STUDENT_DIR.exists():
    DATA_DIR = DATA_STUDENT_DIR
else:
    DATA_DIR = DATA_STANDARD_DIR

# Train / Test file paths
TRAIN_S1_PATH = DATA_DIR / "train_source1.csv"
TRAIN_S2_PATH = DATA_DIR / "train_source2.csv"
TRAIN_S3_PATH = DATA_DIR / "train_source3.csv"
TRAIN_GT_PATH = DATA_DIR / "train_ground_truth.csv"

TEST_S1_PATH = DATA_DIR / "test_source1.csv"
TEST_S2_PATH = DATA_DIR / "test_source2.csv"
TEST_S3_PATH = DATA_DIR / "test_source3.csv"

OUTPUT_DIR = BASE_DIR / "submission"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = BASE_DIR / "artifacts"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Standard Column Names
ID_COL = "id"
NAME_COL = "name"
ADDRESS_COL = "address"
CITY_COL = "city"
STATE_COL = "state"
ZIP_COL = "zip"
COUNTRY_COL = "country"
PHONE_COL = "phone"
SOURCE_COL = "source"

COLUMNS = [ID_COL, NAME_COL, ADDRESS_COL, CITY_COL, STATE_COL, ZIP_COL, COUNTRY_COL, PHONE_COL]

# Blocking Hyperparameters
BLOCKING_NGRAM_RANGE = (3, 4)
BLOCKING_TOP_K_CANDIDATES = 35
BLOCKING_MIN_SIMILARITY = 0.25

# Model & Decision Hyperparameters
RANDOM_SEED = 42
N_FOLDS = 5
EARLY_STOPPING_ROUNDS = 50
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "min_child_samples": 30,
    "n_estimators": 500,
    "random_state": RANDOM_SEED,
    "verbose": -1,
    "n_jobs": -1
}

# Dynamic F_0.5 Decision Thresholds Defaults
DEFAULT_SINGLETON_THRESHOLD = 0.35  # If max candidate prob < 0.35 -> Entity is Singleton []
DEFAULT_MATCH_THRESHOLD = 0.50      # Candidates with prob >= 0.50 are selected as matches
