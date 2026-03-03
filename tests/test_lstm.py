import pytest
import torch
import numpy as np
from src.lstm_model import AttentionLayer, LSTMModel, TimeSeriesCrossValidator

def test_attention_layer_shapes():
    """Memastikan output layer attention memiliki dimensi yang benar."""
    batch_size = 8
    seq_len = 4
    hidden_size = 64
    
    layer = AttentionLayer(hidden_size)
    dummy_input = torch.randn(batch_size, seq_len, hidden_size)
    
    context, weights = layer(dummy_input)
    
    assert context.shape == (batch_size, hidden_size)
    assert weights.shape == (batch_size, seq_len, 1)
    # Weights harus berjumlah 1 untuk setiap sampel
    assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size, 1))

def test_attention_lstm_forward():
    """Memastikan model AttentionLSTM bisa melakukan inferensi."""
    batch_size = 4
    seq_len = 4
    input_size = 1
    
    model = LSTMModel(input_size=input_size, hidden_size=32)
    dummy_input = torch.randn(batch_size, seq_len, input_size)
    
    output = model(dummy_input)
    assert output.shape == (batch_size, 1)
    
    # Test w/ weights
    output, weights = model(dummy_input, return_weights=True)
    assert output.shape == (batch_size, 1)
    assert weights.shape == (batch_size, seq_len, 1)

def test_timeseries_cv_basic():
    """Verifikasi bahwa loop Cross Validation berjalan tanpa error."""
    data = np.linspace(0, 10, 50)
    validator = TimeSeriesCrossValidator(n_splits=3)
    
    params = {
        'input_size': 1,
        'hidden_size': 16,
        'num_layers': 1,
        'lr': 0.01
    }
    
    # Gunakan epochs kecil untuk test speed
    results = validator.validate(LSTMModel, data, params, epochs=2)
    
    assert "avg_rmse" in results
    assert len(results["fold_rmses"]) == 3
    assert results["avg_rmse"] > 0
