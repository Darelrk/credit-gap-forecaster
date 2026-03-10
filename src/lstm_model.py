import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

"""
Custom LSTM architecture with Attention mechanism for credit gap forecasting.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

class LSTMDataProcessor:
    """Preprocesses data for LSTM ingestion."""
    
    def __init__(self, window_size: int = 4):
        self.window_size = window_size
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fits the scaler to the training data and transforms it."""
        return self.scaler.fit_transform(data)
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transforms test data using the strictly pre-fitted scaler."""
        return self.scaler.transform(data)
        
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverses the scaling to return values to base magnitude."""
        return self.scaler.inverse_transform(data)
        
    def create_sequences(self, data: np.ndarray, target_col_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms a 2D array into 3D sequences of shape (samples, window_size, features).
        Returns X (all features) and y (target column only).
        """
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i : (i + self.window_size)])
            # Target is the next value of the specific column (usually Credit Gap)
            y.append(data[i + self.window_size, target_col_idx])
        return np.array(X), np.array(y).reshape(-1, 1)


class AttentionLayer(nn.Module):
    """Additive Attention (Bahdanau-style)."""
    def __init__(self, hidden_size: int):
        super(AttentionLayer, self).__init__()
        self.wa = nn.Linear(hidden_size, hidden_size)
        self.va = nn.Linear(hidden_size, 1)

    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # scores = va(tanh(Wa * output))
        scores = self.va(torch.tanh(self.wa(lstm_outputs))) 
        weights = torch.softmax(scores, dim=1) 
        context = torch.sum(weights * lstm_outputs, dim=1) 
        
        return context, weights

class LSTMModel(nn.Module):
    """LSTM sequence model with an integrated attention mechanism."""
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 1, output_size: int = 1, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout)
        
        self.attention = AttentionLayer(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x: torch.Tensor, return_weights: bool = False) -> torch.Tensor:
        lstm_out, _ = self.lstm(x) 
        context, weights = self.attention(lstm_out)
        
        out = self.dropout(context)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        if return_weights:
            return out, weights
        return out

class LSTMTrainer:
    """Handles the training loop for the LSTM model."""
    def __init__(self, model: nn.Module, lr: float = 0.001):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
    def train_step(self, X: np.ndarray, y: np.ndarray, epochs: int = 1) -> float:
        """Executes a single step (epoch) of training or a full training loop and returns final loss."""
        inputs = torch.tensor(X, dtype=torch.float32)
        targets = torch.tensor(y, dtype=torch.float32)
        
        self.model.train()
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward and optimize
            loss.backward()
            self.optimizer.step()
            
        return loss.item()

    def predict_future(self, last_sequence: np.ndarray, steps: int = 4, scenario_input: np.ndarray = None) -> np.ndarray:
        """
        Autoregressive prediction for future steps.
        If scenario_input is provided, it must be shape (steps, num_features - 1) 
        containing the macro variables for each future step.
        """
        self.model.eval()
        predictions = []
        
        # current_seq shape: (1, window_size, num_features)
        current_seq = last_sequence.copy()
        if len(current_seq.shape) == 2:
            current_seq = current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1])
        
        num_features = current_seq.shape[2]
        
        with torch.no_grad():
            for i in range(steps):
                inputs = torch.tensor(current_seq, dtype=torch.float32)
                pred = self.model(inputs) # Expected shape (1, 1)
                
                pred_val = pred.item()
                predictions.append(pred_val)
                
                # Prepare the next feature vector
                # If we have scenario inputs (macro variables), use them.
                # Otherwise, keep macro variables constant at last known value.
                if scenario_input is not None and i < len(scenario_input):
                    # Combine predicted gap with provided macro variables
                    next_features = np.concatenate([[pred_val], scenario_input[i]])
                else:
                    # Naive: keep macro features from the last timestep of current_seq
                    last_macro = current_seq[0, -1, 1:]
                    next_features = np.concatenate([[pred_val], last_macro])
                
                # Roll sequence
                next_features = next_features.reshape(1, 1, num_features)
                current_seq = np.concatenate([current_seq[:, 1:, :], next_features], axis=1)
                
        return np.array(predictions).reshape(-1, 1)
        
    def save_model(self, filepath: str) -> None:
        """Saves PyTorch model state dictionary."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)

class LSTMHyperOptimizer:
    """Kelas untuk mengelola optimasi hyperparameter LSTM menggunakan Optuna."""
    
    def __init__(self, n_trials=20, storage="sqlite:///models/lstm_optuna.db", study_name="lstm_optimization"):
        self.n_trials = n_trials
        self.storage = storage
        self.study_name = study_name

    def objective(self, trial, X_train, y_train, X_val, y_val):
        hidden_size = trial.suggest_int("hidden_size", 32, 128)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        epochs = 50 

        model = LSTMModel(input_size=X_train.shape[2], hidden_size=hidden_size, num_layers=num_layers, output_size=1, dropout=dropout)
        trainer = LSTMTrainer(model=model, lr=lr)
        trainer.train_step(X_train, y_train, epochs=epochs)

        # 4. Evaluate
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_val, dtype=torch.float32)).numpy()
        
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        return rmse

    def run_study(self, data):
        # Preprocessing minimal untuk study
        processor = LSTMDataProcessor(window_size=4)
        scaled_data = processor.fit_transform(data)
        X, y = processor.create_sequences(scaled_data)
        
        # Split 80/20
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction="minimize",
            load_if_exists=True
        )
        
        study.optimize(lambda t: self.objective(t, X_train, y_train, X_val, y_val), n_trials=self.n_trials)
        
        print(f"Best trial: {study.best_trial.value}")
        print(f"Best params: {study.best_params}")
        return study

class TimeSeriesCrossValidator:
    """
    Kelas untuk melakukan validasi silang pada data deret waktu menggunakan pendekatan Rolling Window.
    """
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self.tscv = TimeSeriesSplit(n_splits=n_splits)

    def validate(self, model_class, data: np.ndarray, model_params: dict, epochs: int = 50) -> dict:
        """
        Menjalankan K-Fold Cross Validation dan mengembalikan rata-rata metrik.
        """
        processor = LSTMDataProcessor(window_size=4)
        scaled_data = processor.fit_transform(data)
        X, y = processor.create_sequences(scaled_data)
        
        fold_results = []
        
        print(f"Starting {self.n_splits}-Fold TimeSeries Cross Validation...")
        for i, (train_index, test_index) in enumerate(self.tscv.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Setup Model
            m_params = model_params.copy()
            lr = m_params.pop('lr', 0.001)
            model = model_class(**m_params)
            trainer = LSTMTrainer(model=model, lr=lr)
            
            # Train
            trainer.train_step(X_train, y_train, epochs=epochs)
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                preds_scaled = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
                
            preds = processor.inverse_transform(preds_scaled)
            actuals = processor.inverse_transform(y_test)
            
            rmse = np.sqrt(mean_squared_error(actuals, preds))
            fold_results.append(rmse)
            print(f" Fold {i+1}: RMSE = {rmse:.4f}")
            
        avg_rmse = np.mean(fold_results)
        print(f"Average Cross-Validation RMSE: {avg_rmse:.4f}")
        
        return {
            "avg_rmse": avg_rmse,
            "fold_rmses": fold_results
        }

if __name__ == "__main__":
    # Example usage for smoke testing
    import pandas as pd
    from src.config import PROCESSED_DATA_PATH, GAP_COL
    
    if os.path.exists(PROCESSED_DATA_PATH):
        df = pd.read_csv(PROCESSED_DATA_PATH)
        data = df[GAP_COL].values
        
        processor = LSTMDataProcessor(window_size=4)
        scaled = processor.fit_transform(data.reshape(-1, 1))
        X, y = processor.create_sequences(scaled)
        
        model = LSTMModel(input_size=1)
        trainer = LSTMTrainer(model)
        
        loss = trainer.train_step(X[:10], y[:10], epochs=5)
        print(f"Smoke test complete. Loss: {loss:.4f}")
