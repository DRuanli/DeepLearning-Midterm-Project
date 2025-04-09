from pathlib import Path

# Root directory
ROOT_DIR = Path(__file__).parent.parent.parent.parent

# Config paths
CONFIG_FILE_PATH = ROOT_DIR / "config" / "config.yaml"
PARAMS_FILE_PATH = ROOT_DIR / "params.yaml"

# Data constants
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]
SENTIMENT_MAPPING = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN]

# Column names for data
TEXT_COL = "text"
CONTEXT_COL = "context"
LABEL_COL = "label"

# Model save paths
MODEL_DIR = ROOT_DIR / "artifacts" / "models"
MODEL_CONFIG_DIR = ROOT_DIR / "artifacts" / "model_config"

# Tensorboard logs
TENSORBOARD_LOG_DIR = ROOT_DIR / "artifacts" / "logs"

# Results
RESULTS_DIR = ROOT_DIR / "artifacts" / "results"

# For reproducibility
SEED = 42