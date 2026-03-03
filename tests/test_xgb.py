import pytest
import pandas as pd
import numpy as np
import os
from src.xgb_model import XGBForecaster

def test_xgb_feature_engineering():
    """Memastikan fitur lag dibuat dengan benar (t-1 s.d t-4)."""
    data = pd.Series([10, 20, 30, 40, 50, 60, 70])
    forecaster = XGBForecaster(lags=4)
    X, y = forecaster.prepare_features(data)
    
    # Dengan 7 data dan lag 4, kita harus punya 7 - 4 = 3 baris
    assert X.shape[0] == 3
    assert X.shape[1] == 4 # 4 lag columns
    # Baris pertama X harus lag dari data ke-4 (indeks 4)
    # y.iloc[0] = 50. X.iloc[0] = [40, 30, 20, 10]
    assert y.iloc[0] == 50
    assert list(X.iloc[0]) == [40.0, 30.0, 20.0, 10.0]

def test_xgb_train_predict():
    """Memastikan model XGBoost bisa dilatih dan melakukan prediksi."""
    # Data dummy sintetik
    data = pd.Series(np.sin(np.linspace(0, 10, 50)) + np.random.normal(0, 0.1, 50))
    forecaster = XGBForecaster(lags=4)
    
    # Train/Test Split
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    forecaster.train(train_data)
    preds = forecaster.predict(test_data)
    
    assert len(preds) == len(test_data) - 4
    assert isinstance(preds, np.ndarray)

def test_xgb_save_load(tmp_path):
    """Verifikasi persistensi model JSON."""
    data = pd.Series(np.random.randn(20))
    forecaster = XGBForecaster(lags=4)
    forecaster.train(data)
    
    model_path = os.path.join(tmp_path, "xgb_test.json")
    forecaster.save_model(model_path)
    assert os.path.exists(model_path)
    
    new_forecaster = XGBForecaster(lags=4)
    new_forecaster.load_model(model_path)
    assert new_forecaster.model is not None
