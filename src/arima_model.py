import pandas as pd
from typing import Tuple, Any

class ARIMAModel:
    """ARIMA Model wrapper for the Credit-to-GDP Gap forecasting."""
    
    def __init__(self):
        self.model = None
        self.order = None
        self.rmse = None
        self.mae = None
    
    def split_data(self, df: pd.DataFrame, target_col: str, train_ratio: float = 0.8) -> Tuple[pd.Series, pd.Series]:
        """Splits time series data sequentially."""
        if target_col not in df.columns:
            raise ValueError(f"Column '{target_col}' not found in DataFrame.")
            
        split_idx = int(len(df) * train_ratio)
        train_series = df[target_col].iloc[:split_idx]
        test_series = df[target_col].iloc[split_idx:]
        
        return train_series, test_series
        
    def train(self, train_series: pd.Series, test_series: pd.Series) -> None:
        """Trains Auto-ARIMA and calculates evaluation metrics on the test set."""
        from pmdarima import auto_arima
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        import numpy as np
        
        # Grid search over p,d,q to find optimal ARIMA parameters
        self.model = auto_arima(
            train_series.dropna(),
            start_p=0, max_p=3,
            start_q=0, max_q=3,
            d=None,          # Let package find optimal differencing
            seasonal=False,  # HP Filter gap is typically non-seasonal
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        self.order = self.model.order
        
        # Forward validation
        predictions = self.model.predict(n_periods=len(test_series))
        
        # Calculate metrics
        self.rmse = float(np.sqrt(mean_squared_error(test_series, predictions)))
        self.mae = float(mean_absolute_error(test_series, predictions))
        
    def forecast(self, steps: int = 4) -> Tuple[pd.Series, Any]:
        """Forecasts future values."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        forecast_values, conf_int = self.model.predict(n_periods=steps, return_conf_int=True)
        return forecast_values, conf_int

    def save_model(self, filepath: str) -> None:
        """Saves the trained model to a file."""
        import joblib
        import os
        if self.model is None:
            raise ValueError("No model trained to save.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        
    def load_model(self, filepath: str) -> None:
        """Loads a trained model from a file."""
        import joblib
        self.model = joblib.load(filepath)
        self.order = self.model.order

if __name__ == "__main__":
    import pandas as pd
    from src.config import PROCESSED_DATA_PATH, GAP_COL

    if os.path.exists(PROCESSED_DATA_PATH):
        df = pd.read_csv(PROCESSED_DATA_PATH)
        arima = ARIMAModel()
        train, test = arima.split_data(df, GAP_COL)
        
        print("Training Auto-ARIMA...")
        arima.train(train, test)
        
        print(f"Optimal Order: {arima.order}")
        print(f"RMSE: {arima.rmse:.4f}")
