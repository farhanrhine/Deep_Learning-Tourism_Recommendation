"""
Central configuration for the Tourism Recommendation System.
All hyperparameters, paths, and settings are defined here.
"""

import os

# ─────────────────────────── Paths ───────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Ensure directories exist
for d in [DATA_DIR, MODEL_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────── Dataset ───────────────────────────
NUM_USERS = 500
NUM_DESTINATIONS = 200
NUM_INTERACTIONS = 10000

# ─────────────────────────── Feature Dimensions ───────────────────────────
USER_FEATURE_DIM = 4     # age(1) + travel_style(1) + budget_level(1) + past_trips(1)
DEST_FEATURE_DIM = 5     # category(1) + avg_cost(1) + popularity(1) + climate(1) + best_season(1)

# ─────────────────────────── Model Hyperparameters ───────────────────────────
EMBEDDING_DIM = 32
GMF_DIM = 32
MLP_HIDDEN_LAYERS = [128, 64, 32]
CONTENT_HIDDEN_LAYERS = [64, 32]
HYBRID_HIDDEN_LAYERS = [64, 32]
DROPOUT_RATE = 0.2

# ─────────────────────────── Training ───────────────────────────
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
EPOCHS = 50
EARLY_STOP_PATIENCE = 7
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ─────────────────────────── Evaluation ───────────────────────────
TOP_K_VALUES = [5, 10, 20]
RATING_THRESHOLD = 3.5   # Ratings above this are considered "relevant"
