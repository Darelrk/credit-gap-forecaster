import os
import pandas as pd
import numpy as np
import pytest
import matplotlib
import tempfile
from src.visualization import Visualizer

def test_visualizer_plot_comparison():
    """Test that Visualizer can create a plot file from dummy data without errors."""
    # Dummy data
    dates = pd.date_range(start='2020-01-01', periods=10, freq='Q')
    actual = pd.Series(np.random.normal(0, 1, 10), index=dates)
    arima = pd.Series(np.random.normal(0, 1, 10), index=dates)
    lstm = pd.Series(np.random.normal(0, 1, 10), index=dates)

    visualizer = Visualizer()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, "test_plot.png")
        
        # Method should save figure to output_path and not raise any GUI exceptions
        visualizer.plot_comparison(
            actual_series=actual,
            arima_series=arima,
            lstm_series=lstm,
            output_path=output_path
        )
        
        # Assert plot file was created
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

def test_visualizer_backend():
    """Ensure matplotlib uses the 'Agg' backend to avoid GUI issues."""
    assert matplotlib.get_backend().lower() == 'agg'
