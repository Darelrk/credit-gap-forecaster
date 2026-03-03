import pandas as pd
import numpy as np
import os
from src.config import PROCESSED_DATA_PATH, GAP_COL, MODELS_DIR
from src.lstm_model import LSTMHyperOptimizer

def main():
    print("="*60)
    print(" LSTM HYPERPARAMETER TUNING - OPTUNA ")
    print("="*60)
    
    # Load data
    df = pd.read_csv(PROCESSED_DATA_PATH)
    data = df[GAP_COL].values
    
    # Initialize Optimizer
    optimizer = LSTMHyperOptimizer(
        n_trials=20, 
        storage=f"sqlite:///{os.path.join(MODELS_DIR, 'lstm_optuna.db')}"
    )
    
    # Run Study
    print(f"Starting {optimizer.n_trials} trials of Bayesian Optimization...")
    study = optimizer.run_study(data)
    
    print("\n" + "-"*40)
    print(" BEST CONFIGURATION FOUND ")
    print("-"*40)
    print(f" Value (RMSE): {study.best_value:.6f}")
    for key, value in study.best_params.items():
        print(f" {key:>15}: {value}")
    print("="*60)

if __name__ == "__main__":
    main()
