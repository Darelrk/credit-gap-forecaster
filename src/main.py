import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.config import PROCESSED_DATA_PATH, MODELS_DIR, RESULTS_DIR, GAP_COL
from src.arima_model import ARIMAModel
from src.lstm_model import LSTMTrainer, LSTMDataProcessor, LSTMModel
from src.xgb_model import XGBForecaster
from src.visualization import Visualizer

def main():
    print("-" * 30)
    print("CORE FORECASTING PIPELINE")
    print("-" * 30)
    
    # Loading dataset
    print("\nLoading dataset...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    # Gunakan period date string sebagai index agar plotting rapi
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    actual_gap = df[GAP_COL]
    
    # Menyamakan Skema Train/Test untuk Evaluasi (80/20 seq)
    train_size = int(len(df) * 0.8)
    test_actual = actual_gap.iloc[train_size:]
    print(f"Processing {len(df)} quarters. Test starts at {test_actual.index[0].to_period('Q')}")
    
    # Inisialisasi DataFrame untuk Penyatuan Hasil Prediksi (Out-of-sample)
    # LSTM using Sequence Window but we will evaluate on the same Test length
    # LSTM Window size
    window_size = 4
    
    # Initializing models
    print("\nLoading models and generating predictions...")
    
    # --- ARIMA LOGIC ---
    print(" -> Memproses prediksi Auto-ARIMA...")
    arima_path = os.path.join(MODELS_DIR, "arima_model.pkl")
    arima_model = ARIMAModel()
    if os.path.exists(arima_path):
        arima_model.load_model(arima_path)
    else:
        print("ARIMA model not found. Training...")
        train, test = arima_model.split_data(actual_gap)
        arima_model.train(train, test)
        arima_model.save_model(arima_path)
        
    predictions_arima = arima_model.model.predict(n_periods=len(test_actual))
    arima_series = pd.Series(predictions_arima.values, index=test_actual.index)
    
    # --- LSTM LOGIC (Trivariate) ---
    print(" -> Memproses prediksi LSTM PyTorch (Trivariate)...")
    df_multi = pd.read_csv("data/processed_data_multivariate.csv")
    df_multi['date'] = pd.to_datetime(df_multi['date'])
    df_multi.set_index('date', inplace=True)
    multi_features = [GAP_COL, 'GDP_Growth', 'Interest_Rate']
    lstm_data = df_multi[multi_features].values
    
    # Setup data processor
    lstm_processor = LSTMDataProcessor(window_size=window_size)
    # Fit di data Latih
    lstm_processor.fit_transform(lstm_data[:train_size])
    
    # Transform seluruh array raw untuk slicing tensor LSTM Predict sequence 
    all_data_scaled = lstm_processor.transform(lstm_data)
    X_train_scaled, y_train_scaled = lstm_processor.create_sequences(all_data_scaled[:train_size], target_col_idx=0)
    X_test_scaled, y_test_scaled = lstm_processor.create_sequences(all_data_scaled[train_size - window_size:], target_col_idx=0)
    
    lstm_path = os.path.join(MODELS_DIR, "lstm_model_multivariate.pt")
    # Best params dari training bivariate
    lstm_params = {
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.2
    }
    lstm_neural_net = LSTMModel(input_size=len(multi_features), **lstm_params)
    
    if os.path.exists(lstm_path):
        lstm_neural_net.load_state_dict(torch.load(lstm_path))
        lstm_trainer = LSTMTrainer(model=lstm_neural_net)
    else:
        print("LSTM model not found. Training...")
        lstm_trainer = LSTMTrainer(model=lstm_neural_net, lr=0.0014)
        lstm_trainer.train_step(X_train_scaled, y_train_scaled, epochs=150)
        lstm_trainer.save_model(lstm_path)
        
    lstm_neural_net.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        lstm_pred_scaled = lstm_neural_net(test_tensor).numpy()
    
    dummy_pred = np.zeros((len(lstm_pred_scaled), len(multi_features)))
    dummy_pred[:, 0] = lstm_pred_scaled.flatten()
    lstm_pred_inv = lstm_processor.inverse_transform(dummy_pred)[:, 0]
    
    # Menyelaraskan array prediksi ke DateTime indeks Pandas 
    lstm_series = pd.Series(lstm_pred_inv.flatten(), index=test_actual.index)

    # --- XGBOOST LOGIC ---
    print(" -> Memproses prediksi XGBoost ML...")
    xgb_path = os.path.join(MODELS_DIR, "xgb_model_multivariate.json")
    xgb_model = XGBForecaster(lags=window_size)
    
    # Train data series untuk XGB (Multivariat)
    train_multi_features = df_multi[multi_features].iloc[:train_size]
    
    if os.path.exists(xgb_path):
        xgb_model.load_model(xgb_path)
    else:
        print("XGBoost model not found. Training...")
        xgb_model.train(train_multi_features)
        xgb_model.save_model(xgb_path)
        
    # Predict Out-of-sample
    # Predict Out-of-sample
    # XGB predict butuh data input yang punya history untuk lag-nya (Multivariate)
    full_test_input = df_multi[multi_features].iloc[train_size - window_size:]
    predictions_xgb = xgb_model.predict(full_test_input).flatten()
    xgb_series = pd.Series(predictions_xgb, index=test_actual.index)

    # (Blok LSTM v2 lama dihapus karena sudah dikonsolidasi di atas)
    
    # ENSEMBLE CALCULATION (LSTM + XGB) 
    ensemble_series = (lstm_series + xgb_series) / 2

    
    # Evaluation
    print("\nCalculating metrics...")
    def eval_metrics(actual, pred):
        return {
            "RMSE": np.sqrt(mean_squared_error(actual, pred)),
            "MAE": mean_absolute_error(actual, pred)
        }
        
    metrics_arima = eval_metrics(test_actual, arima_series)
    metrics_lstm = eval_metrics(test_actual, lstm_series)
    metrics_xgb = eval_metrics(test_actual, xgb_series)
    metrics_ensemble = eval_metrics(test_actual, ensemble_series)
    
    print("-" * 40)
    print("MODEL PERFORMANCE (TEST SET)")
    print("-" * 40)
    print(f" ARIMA    | RMSE: {metrics_arima['RMSE']:.4f}  | MAE: {metrics_arima['MAE']:.4f}")
    print(f" LSTM     | RMSE: {metrics_lstm['RMSE']:.4f}  | MAE: {metrics_lstm['MAE']:.4f} (Attention)")
    print(f" XGBoost  | RMSE: {metrics_xgb['RMSE']:.4f}  | MAE: {metrics_xgb['MAE']:.4f}")
    print(f" ENSEMBLE | RMSE: {metrics_ensemble['RMSE']:.4f}  | MAE: {metrics_ensemble['MAE']:.4f}")
    
    # Plots - focused on Evaluation (Zoomed into Test Period)
    print("\nGenerating charts (Evaluasi Zoom)...")
    vis = Visualizer()
    eval_chart_path = os.path.join(RESULTS_DIR, "comparative_evaluation_plot.png")
    
    # Calculate a nice window for evaluation view (e.g. Test set starts at iloc[train_size])
    # Let's show 8 quarters before the test set for context
    context_start = max(0, train_size - 8)
    actual_zoom = actual_gap.iloc[context_start:]
    
    vis.plot_comparison(
        actual_series=actual_zoom,
        arima_series=arima_series,
        lstm_series=lstm_series,
        xgb_series=xgb_series,
        ensemble_series=ensemble_series,
        output_path=eval_chart_path
    )
    print(f" -> Tesis Komparatif Berhasil Diekspor >> {eval_chart_path}")
    
    # Future Forecast
    print("\n" + "*" * 40)
    print("FUTURE FORECAST (NEXT 4 QUARTERS)")
    print("*" * 40)
    
    # Ekstrak waktu depan pandas (Period)
    future_dates = pd.date_range(start=actual_gap.index[-1] + pd.DateOffset(months=3), periods=4, freq='QE')
    
    # ARIMA Future
    arima_future = arima_model.model.predict(n_periods=4).values
    
    # LSTM Future (Auto-regressive roll-forward of last known historical seq window)
    last_seq = all_data_scaled[-window_size:] 
    
    # Skenario Makro 2026: 
    # 1. PDB melemah ke 3.5% (Hard Landing)
    # 2. Suku Bunga tinggi ditahan di sekitar ~5.0%
    future_gdp = np.linspace(lstm_data[-1, 1], 3.5, num=4)
    future_ir = np.linspace(lstm_data[-1, 2], 5.0, num=4)
    
    dummy_macro = np.zeros((4, len(multi_features)))
    dummy_macro[:, 1] = future_gdp
    dummy_macro[:, 2] = future_ir
    scaled_macro = lstm_processor.transform(dummy_macro)[:, 1:]
    
    # FIXED: Use lstm_trainer for predict_future
    lstm_future_scaled = lstm_trainer.predict_future(last_seq, steps=4, scenario_input=scaled_macro)
    dummy_future = np.zeros((len(lstm_future_scaled), len(multi_features)))
    dummy_future[:, 0] = lstm_future_scaled.flatten()
    lstm_future = lstm_processor.inverse_transform(dummy_future)[:, 0]
    
    # XGBoost Future
    unscaled_macro_scenario = dummy_macro[:, 1:] # Berisi GDP & IR masa depan yang mentah tak di-scaling
    last_xgb_input = df_multi[multi_features].values[-window_size:]
    xgb_future = xgb_model.predict_future(last_xgb_input, steps=4, scenario_macro=unscaled_macro_scenario).flatten()
    
    # ENSEMBLE PRED
    ensemble_future = (lstm_future + xgb_future) / 2
    
    for i in range(4):
        q_label = future_dates[i].to_period('Q').strftime('%Y-Q%q')
        print(f" [{q_label}] | ARIMA: {arima_future[i]:>5.2f}% | LSTM: {lstm_future[i]:>5.2f}% | XGB: {xgb_future[i]:>5.2f}% | ENSEMBLE: {ensemble_future[i]:>5.2f}%")
        
    print("-" * 40)
    print("Pipeline execution complete.")

if __name__ == "__main__":
    main()
