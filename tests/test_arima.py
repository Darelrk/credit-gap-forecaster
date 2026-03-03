import pytest
import pandas as pd
import numpy as np
from src.arima_model import ARIMAModel

@pytest.fixture
def sample_ts_data() -> pd.DataFrame:
    """Fixture providing sample time series data."""
    dates = pd.date_range(start="2000-01-01", periods=10, freq="QS")
    # Autoregressive-like simple data
    values = np.linspace(10, 20, 10) + np.random.normal(0, 0.5, 10)
    return pd.DataFrame({'date': dates, 'gap': values})

def test_split_data_sequential(sample_ts_data: pd.DataFrame) -> None:
    """Test that data splitting is sequential and not random/shuffled."""
    model = ARIMAModel()
    train_series, test_series = model.split_data(sample_ts_data, 'gap', train_ratio=0.8)
    
    assert len(train_series) == 8
    assert len(test_series) == 2
    # Check sequential integrity: No random shuffling allowed in Time Series
    assert train_series.iloc[-1] == sample_ts_data['gap'].iloc[7]
    assert test_series.iloc[0] == sample_ts_data['gap'].iloc[8]

def test_train_auto_arima_and_metrics(sample_ts_data: pd.DataFrame) -> None:
    """Test auto-arima training process and metric calculation."""
    model = ARIMAModel()
    train_series, test_series = model.split_data(sample_ts_data, 'gap', train_ratio=0.8)
    
    model.train(train_series, test_series)
    
    assert model.model is not None
    assert type(model.order) == tuple
    assert len(model.order) == 3 # (p, d, q)
    assert hasattr(model, 'rmse')
    assert hasattr(model, 'mae')
    assert model.rmse >= 0
    assert model.mae >= 0

def test_forecast_steps() -> None:
    """Test forecasting into the future."""
    model = ARIMAModel()
    # Mock training for standalone testing
    dates = pd.date_range(start="2000-01-01", periods=20, freq="QS")
    values = np.linspace(10, 30, 20)
    train_series = pd.Series(values[:16])
    test_series = pd.Series(values[16:])
    
    model.train(train_series, test_series)
    
    forecast, conf_int = model.forecast(steps=4)
    assert len(forecast) == 4
    assert conf_int.shape == (4, 2)
