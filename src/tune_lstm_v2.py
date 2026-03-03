import pandas as pd
import numpy as np
import os
import optuna
from src.config import PROCESSED_DATA_PATH, GAP_COL, MODELS_DIR
from src.lstm_model import AttentionLSTM, TimeSeriesCrossValidator, LSTMTrainer, LSTMDataProcessor
import torch

def objective(trial, data):
    # Suggest Hyperparameters for AttentionLSTM
    params = {
        'hidden_size': trial.suggest_int("hidden_size", 32, 128),
        'num_layers': trial.suggest_int("num_layers", 1, 3),
        'lr': trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        'dropout': trial.suggest_float("dropout", 0.1, 0.5),
        'input_size': 1,
        'output_size': 1
    }
    
    # Use TimeSeriesCrossValidator (3-fold for speed during tuning, 5-fold for final)
    validator = TimeSeriesCrossValidator(n_splits=3)
    results = validator.validate(AttentionLSTM, data, params, epochs=50)
    
    return results["avg_rmse"]

def main():
    print("="*60)
    print(" DEEP LSTM (ATTENTION) TUNING - OPTUNA + TS-CV ")
    print("="*60)
    
    # Load data
    df = pd.read_csv(PROCESSED_DATA_PATH)
    data = df[GAP_COL].values
    
    # SQLite Storage for persistence
    db_path = os.path.join(MODELS_DIR, 'lstm_v2_optuna.db')
    storage = f"sqlite:///{db_path}"
    
    study = optuna.create_study(
        study_name="lstm_attention_optimization",
        storage=storage,
        direction="minimize",
        load_if_exists=True
    )
    
    n_trials = 15 # 15 trials with CV is quite intensive
    print(f"Starting {n_trials} trials of Bayesian Optimization with 3-Fold CV...")
    study.optimize(lambda t: objective(t, data), n_trials=n_trials)
    
    print("\n" + "="*60)
    print(" BEST CONFIGURATION (ATTENTION LSTM) ")
    print("="*60)
    print(f" Best Avg RMSE: {study.best_value:.6f}")
    for key, value in study.best_params.items():
        print(f" {key:>15}: {value}")
    print("="*60)

if __name__ == "__main__":
    main()
