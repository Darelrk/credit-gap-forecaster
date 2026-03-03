import xgboost as xgb
import pandas as pd
import numpy as np
import os

class XGBForecaster:
    """Implementasi peramalan Credit Gap menggunakan Gradient Boosting (XGBoost)."""
    
    def __init__(self, lags=4, params=None):
        self.lags = lags
        # Default hyperparameters untuk stabilitas time series
        self.params = params or {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_jobs': 8,
            'random_state': 42
        }
        self.model = None
        
    def prepare_features(self, series: pd.Series) -> (pd.DataFrame, pd.Series):
        """
        Merubah deret waktu 1D menjadi fitur tabular (Lag 1 s.d Lag N).
        """
        df = pd.DataFrame(series)
        df.columns = ['target']
        
        # Buat kolom lag
        for i in range(1, self.lags + 1):
            df[f'lag_{i}'] = df['target'].shift(i)
            
        # Drop baris NaN akibat shifting
        df = df.dropna()
        
        X = df.drop('target', axis=1)
        y = df['target']
        return X, y

    def train(self, series: pd.Series):
        """Melatih model XGBoost pada data historis."""
        X, y = self.prepare_features(series)
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y)
        print("XGBoost training complete.")

    def predict(self, series: pd.Series) -> np.ndarray:
        """Memprediksi nilai gap masa depan (Out-of-sample)."""
        if self.model is None:
            raise ValueError("Model must be trained before calling predict.")
            
        X, _ = self.prepare_features(series)
        return self.model.predict(X)

    def predict_future(self, last_data: pd.Series, steps: int = 4) -> np.ndarray:
        """Prediksi masa depan secara sekuensial (Autoregressive)."""
        if self.model is None:
            raise ValueError("Model must be trained before forecasting.")

        current_data = last_data.tolist()
        future_preds = []
        
        for _ in range(steps):
            # Ambil 'lags' data terakhir untuk fitur
            X_input = np.array(current_data[-self.lags:])[::-1].reshape(1, -1)
            # Sesuai urutan lag_1, lag_2, lag_3, lag_4 di prepare_features
            # Jika prepare_features pakai lag_1=t-1, lag_2=t-2... maka X_input: [t-1, t-2, t-3, t-4]
            # np.array(current_data[-4:])[::-1] -> [last, last-1, last-2, last-3] => [t-1, t-2, t-3, t-4]
            
            # Predict
            pred = self.model.predict(X_input)[0]
            future_preds.append(pred)
            current_data.append(pred)
            
        return np.array(future_preds).reshape(-1, 1)

    def save_model(self, path: str):
        """Simpan model ke format JSON (lebih stabil di lintas platform)."""
        if self.model is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save_model(path)

    def load_model(self, path: str):
        """Muat kembali model dari JSON."""
        self.model = xgb.XGBRegressor()
        self.model.load_model(path)
