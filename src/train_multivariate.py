import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from src.config import MODELS_DIR, RESULTS_DIR, GAP_COL
from src.lstm_model import LSTMModel, LSTMTrainer, LSTMDataProcessor
from src.xgb_model import XGBForecaster
from sklearn.metrics import mean_squared_error

def train_multivariate():
    print("=" * 40)
    print("MULTIVARIATE LSTM TRAINING ENGINE")
    print("=" * 40)
    
    # Dataset path
    data_path = "data/processed_data_multivariate.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run sync_macro_data.py first.")
        return

    # 1. Load Data
    data_path = "data/processed_data_multivariate.csv"
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # 2. Select Features (Trivariate: Gap + GDP_Growth + Interest_Rate)
    multi_features = [GAP_COL, 'GDP_Growth', 'Interest_Rate']
    data = df[multi_features].values
    
    print(f"Features: {multi_features}")
    print(f"Data shape: {data.shape}")
    
    # 3. Preprocess
    window_size = 4
    processor = LSTMDataProcessor(window_size=window_size)
    scaled_data = processor.fit_transform(data)
    
    # Split 80/20
    train_size = int(len(scaled_data) * 0.8)
    train_scaled = scaled_data[:train_size]
    test_scaled = scaled_data[train_size - window_size:]
    
    # Create Sequences (Target is Gap at index 0)
    X_train, y_train = processor.create_sequences(train_scaled, target_col_idx=0)
    X_test, y_test = processor.create_sequences(test_scaled, target_col_idx=0)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Model Setup: LSTM
    # Using best params from previous tuning as baseline
    input_size = len(multi_features)
    hidden_size = 64
    num_layers = 2 # Deeper for multivariate
    dropout = 0.2
    
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    trainer = LSTMTrainer(model, lr=0.001)
    
    # Training
    print("\nTraining Multivariate LSTM...")
    epochs = 200
    loss = trainer.train_step(X_train, y_train, epochs=epochs)
    print(f"Final Loss (MSE): {loss:.6f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        preds_scaled = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    
    # Inverse Transform for Evaluation
    # We need to provide a dummy array with 2 columns to the processor
    dummy_pred = np.zeros((len(preds_scaled), len(multi_features)))
    dummy_pred[:, 0] = preds_scaled.flatten()
    inv_preds = processor.inverse_transform(dummy_pred)[:, 0]
    
    dummy_actual = np.zeros((len(y_test), len(multi_features)))
    dummy_actual[:, 0] = y_test.flatten()
    inv_actual = processor.inverse_transform(dummy_actual)[:, 0]
    
    rmse = np.sqrt(mean_squared_error(inv_actual, inv_preds))
    print(f"Test RMSE: {rmse:.4f}")
    
    # XGBoost Multivariate Setup & Train
    print("\nTraining Multivariate XGBoost...")
    xgb_lags = 4
    xgb = XGBForecaster(lags=xgb_lags)
    xgb.train(df[multi_features].iloc[:train_size])
    
    # Save XGBoost Model
    xgb_path = os.path.join(MODELS_DIR, "xgb_model_multivariate.json")
    xgb.save_model(xgb_path)
    print(f"XGBoost Multivariate model saved to {xgb_path}")
    
    return model, processor, xgb

if __name__ == "__main__":
    train_multivariate()
