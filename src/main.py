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
    
    # --- LSTM LOGIC ---
    print(" -> Memproses prediksi LSTM PyTorch...")
    # Setup data processor untuk skala dan sequence seperti saat training
    lstm_processor = LSTMDataProcessor(window_size=window_size)
    # Fit di data Latih
    train_data_array = actual_gap.iloc[:train_size].values.reshape(-1, 1)
    lstm_processor.fit_transform(train_data_array)
    
    # Transform seluruh array raw untuk slicing tensor LSTM Predict sequence 
    all_data_scaled = lstm_processor.transform(actual_gap.values.reshape(-1, 1))
    X_train_scaled, y_train_scaled = lstm_processor.create_sequences(all_data_scaled[:train_size])
    X_test_scaled, y_test_scaled = lstm_processor.create_sequences(all_data_scaled[train_size - window_size:])
    
    lstm_path = os.path.join(MODELS_DIR, "lstm_model.pt")
    # Best params dari tuning
    lstm_params = {
        'hidden_size': 65,
        'num_layers': 1,
        'dropout': 0.3382
    }
    lstm_neural_net = LSTMModel(input_size=1, **lstm_params)
    
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
    
    lstm_pred_inv = lstm_processor.inverse_transform(lstm_pred_scaled)
    # Menyelaraskan array prediksi ke DateTime indeks Pandas 
    lstm_series = pd.Series(lstm_pred_inv.flatten(), index=test_actual.index)

    # --- XGBOOST LOGIC ---
    print(" -> Memproses prediksi XGBoost ML...")
    xgb_path = os.path.join(MODELS_DIR, "xgb_model.json")
    xgb_model = XGBForecaster(lags=window_size)
    
    # Train data series untuk XGB
    train_gap_series = actual_gap.iloc[:train_size]
    
    if os.path.exists(xgb_path):
        xgb_model.load_model(xgb_path)
    else:
        print("XGBoost model not found. Training...")
        xgb_model.train(train_gap_series)
        xgb_model.save_model(xgb_path)
        
    # Predict Out-of-sample
    # XGB predict butuh data input yang punya history untuk lag-nya
    # Kita berikan data mulai dari train_size - window_size agar prediksi pertama tervalidasi
    full_test_input = actual_gap.iloc[train_size - window_size:]
    predictions_xgb = xgb_model.predict(full_test_input)
    xgb_series = pd.Series(predictions_xgb, index=test_actual.index)

    # (Blok LSTM v2 lama dihapus karena sudah dikonsolidasi di atas)

    
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
    
    print("-" * 40)
    print("MODEL PERFORMANCE (TEST SET)")
    print("-" * 40)
    print(f" ARIMA  | RMSE: {metrics_arima['RMSE']:.4f}  | MAE: {metrics_arima['MAE']:.4f}")
    print(f" LSTM   | RMSE: {metrics_lstm['RMSE']:.4f}  | MAE: {metrics_lstm['MAE']:.4f} (Attention)")
    print(f" XGBoost| RMSE: {metrics_xgb['RMSE']:.4f}  | MAE: {metrics_xgb['MAE']:.4f}")
    
    # Plots
    print("\nGenerating charts...")
    vis = Visualizer()
    eval_chart_path = os.path.join(RESULTS_DIR, "comparative_evaluation_plot.png")
    
    vis.plot_comparison(
        actual_series=actual_gap,
        arima_series=arima_series,
        lstm_series=lstm_series,
        xgb_series=xgb_series,
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
    lstm_future_scaled = lstm_trainer.predict_future(last_seq, steps=4)
    lstm_future = lstm_processor.inverse_transform(lstm_future_scaled).flatten()
    
    # XGBoost Future
    xgb_future = xgb_model.predict_future(actual_gap.iloc[-window_size:], steps=4).flatten()
    
    for i in range(4):
        q_label = future_dates[i].to_period('Q').strftime('%Y-Q%q')
        print(f" [{q_label}] | ARIMA: {arima_future[i]:>5.2f}% | LSTM: {lstm_future[i]:>5.2f}% | XGB: {xgb_future[i]:>5.2f}%")
        
    print("-" * 40)
    print("Pipeline execution complete.")

if __name__ == "__main__":
    main()
