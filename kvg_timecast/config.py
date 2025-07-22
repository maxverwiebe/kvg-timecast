import os
from pathlib import Path

# data paths
DATA_DIR = "/Users/maximilianverwiebe/codingprojects/kvg_ml_analysis/data"
DATA_FILE = "stopoutput123.csv"
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)

# model paths
MODEL_DIR = "models"
MODEL_FILE = "kvg_ensemble_model.pkl"
ENCODERS_FILE = "label_encoders.pkl"
FEATURES_FILE = "feature_names.pkl"

# ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
ENCODERS_PATH = os.path.join(MODEL_DIR, ENCODERS_FILE)
FEATURES_PATH = os.path.join(MODEL_DIR, FEATURES_FILE)

# data processing parameters
MAX_RECORDS = 3_000_000
MAX_DELAY_SECONDS = 3600  # 1 hour max delay

# model parameters
LIGHTGBM_BASE_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "random_state": 42
}

# feature engineering parameters
ROLLING_WINDOWS = [3, 5, 10, 20]
LAG_FEATURES = [1, 2, 3, 5]
ANOMALY_THRESHOLD = 3  # Z-score threshold for anomaly detection

# train-test split
TEST_SIZE = 0.2

# cross-validation
CV_FOLDS = 5

# categorical columns
CATEGORICAL_COLS = ["stopName", "patternText", "direction", "route_stop_key", "route_dir_key"]

# columns to exclude from features
EXCLUDE_COLS = [
    "delay_minutes", "delay_seconds", "actualTime", "plannedTime", "timestamp",
    "busStopID", "tripId", "vehicleId", "status", "routeId"
]

# limit for stacking learner to avoid memory issues (on my machine at least)
STACKING_LEARNER_MAX_SAMPLES = 100_000

# clean duplicated trips at stop
CLEAN_DUPLICATED_TRIPS_AT_STOP = False