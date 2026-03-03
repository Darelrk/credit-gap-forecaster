import pytest
import numpy as np
import torch
import optuna
import os
from src.lstm_model import LSTMHyperOptimizer, LSTMDataProcessor

def test_lstm_hyper_optimizer_init():
    """Memastikan inisialisasi HyperOptimizer berjalan lancar."""
    optimizer = LSTMHyperOptimizer(n_trials=2)
    assert optimizer.n_trials == 2
    assert optimizer.study_name == "lstm_optimization"

def test_optuna_objective_run():
    """Memastikan fungsi objective dapat dijalankan oleh Optuna tanpa crash."""
    # Data dummy
    data = np.random.randn(50, 1)
    processor = LSTMDataProcessor(window_size=4)
    scaled_data = processor.fit_transform(data)
    X, y = processor.create_sequences(scaled_data)
    
    optimizer = LSTMHyperOptimizer(n_trials=1)
    
    # Mocking trial object
    class MockTrial:
        def suggest_int(self, name, low, high): return 32
        def suggest_float(self, name, low, high, log=False): 
            if log: return 0.001
            return 0.2
            
    trial = MockTrial()
    # Panggil objective secara manual
    rmse = optimizer.objective(trial, X, y, X, y)
    assert isinstance(rmse, float)
    assert rmse >= 0

def test_hyper_optimization_integration(tmp_path):
    """Verifikasi integrasi penuh (running minimal trials)."""
    data = np.random.randn(40, 1)
    # Gunakan db temporary untuk test
    db_path = os.path.join(tmp_path, "test_study.db")
    storage_url = f"sqlite:///{db_path}"
    
    optimizer = LSTMHyperOptimizer(n_trials=2, storage=storage_url)
    
    # Run optimization dengan data dummy
    # (Dalam realita ini akan memakan waktu, tapi dengan 2 trials & data kecil harusnya cepat)
    study = optimizer.run_study(data)
    
    assert len(study.trials) == 2
    assert "hidden_size" in study.best_params
    assert "learning_rate" in study.best_params
