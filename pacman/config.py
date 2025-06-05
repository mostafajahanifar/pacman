import os

### ========== PATHS ==========
# Get the repo root (two levels up from config.py)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


for d in [DATA_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

### ========== HYPERPARAMETERS ==========

### ========== DOMAIN CONSTANTS ==========
