import xgboost as xgb
import pandas as pd
import numpy as np
import os

class XGBForecaster:
    """Implementasi peramalan Credit Gap menggunakan Gradient Boosting (XGBoost) berbasis Multivariat."""
    
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
        
    def prepare_features(self, df_or_series) -> (pd.DataFrame, pd.Series):
        """
        Merubah dataframe multivariat menjadi fitur tabular (Lag 1 s.d Lag N) untuk setiap kolom fitur.
        """
        if isinstance(df_or_series, pd.Series):
            df = pd.DataFrame(df_or_series)
            df.columns = ['target']
            target_col = 'target'
        else:
            df = df_or_series.copy()
            if isinstance(df, np.ndarray):
                df = pd.DataFrame(df)
            # Assuming first column is target if not specified
            target_col = df.columns[0]
            
        feat_cols = df.columns
        df_lags = pd.DataFrame(index=df.index)
        df_lags['target'] = df[target_col]
        
        # Buat kolom lag untuk SEMUA fitur (Multivariate)
        for col in feat_cols:
            for i in range(1, self.lags + 1):
                df_lags[f'{col}_lag_{i}'] = df[col].shift(i)
                
        # Drop baris NaN akibat shifting
        df_lags = df_lags.dropna()
        
        X = df_lags.drop('target', axis=1)
        y = df_lags['target']
        return X, y

    def train(self, data):
        """Melatih model XGBoost pada data historis (multivariat/n-dimensional array/dataframe)."""
        X, y = self.prepare_features(data)
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y)
        print("XGBoost Multivariate training complete.")

    def predict(self, data) -> np.ndarray:
        """Memprediksi nilai target masa depan (Out-of-sample) dari data validation."""
        if self.model is None:
            raise ValueError("Model must be trained before calling predict.")
            
        X, _ = self.prepare_features(data)
        return self.model.predict(X).reshape(-1, 1)

    def predict_future(self, current_data, steps: int = 4, scenario_macro: np.ndarray = None) -> np.ndarray:
        """
        Prediksi masa depan secara sekuensial (Autoregressive) untuk Multivariat.
        current_data: DataFrame / Array (window_size >= lags, num_features)
        scenario_macro: Array (steps, num_features - 1) untuk makro asumsi ke depan.
        """
        if self.model is None:
            raise ValueError("Model must be trained before forecasting.")

        if isinstance(current_data, pd.DataFrame) or isinstance(current_data, pd.Series):
            curr_seq = current_data.values.copy()
        else:
            curr_seq = current_data.copy()
            
        if len(curr_seq.shape) == 1:
            curr_seq = curr_seq.reshape(-1, 1)
            
        num_features = curr_seq.shape[1]
        future_preds = []
        
        for i in range(steps):
            # Extract features according to the prepare_features looping order:
            # for col in feat_cols: 
            #    for lag in 1 to self.lags:
            row = []
            for col_idx in range(num_features):
                for lag in range(1, self.lags + 1):
                    # curr_seq[-lag] artinya ambil baris ke (total_baris - lag)
                    row.append(curr_seq[-lag, col_idx])
            
            X_input = np.array(row).reshape(1, -1)
            
            pred = self.model.predict(X_input)[0]
            future_preds.append(pred)
            
            # Prepare next timestep feature vector
            if scenario_macro is not None and i < len(scenario_macro):
                next_macro = scenario_macro[i]
            else:
                next_macro = curr_seq[-1, 1:] # If Univariate this is empty array, which is fine
                
            next_feat = np.concatenate([[pred], next_macro]).reshape(1, -1)
            curr_seq = np.vstack((curr_seq, next_feat))
            
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

