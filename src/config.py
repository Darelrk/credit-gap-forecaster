# Configuration constants for Credit-to-GDP Gap Forecasting

import os

# Project root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data paths
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_PATH = DATA_DIR
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_data.csv")

# Model properties
HP_LAMBDA = 400000  # Standar Basel III untuk Credit-to-GDP Gap

# Data Column Names
DATE_COL = 'date'
CREDIT_COL = 'credit'
GDP_COL = 'gdp'
RATIO_COL = 'ratio'
TREND_COL = 'trend'
GAP_COL = 'gap'

# Model Output paths
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
