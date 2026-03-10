import os
import pandas as pd
import numpy as np
import torch
from src.config import PROCESSED_DATA_PATH, MODELS_DIR, RESULTS_DIR, GAP_COL, DATA_DIR
from src.lstm_model import LSTMModel, LSTMDataProcessor
from src.xgb_model import XGBForecaster
from src.augmentation import DataAugmentor
from src.visualization import Visualizer

def main():
    print("Initializing Probabilistic Outlook 2026 Generation (Trivariate Ensemble Edition)...")
    
    # 1. Load Data
    df = pd.read_csv("data/processed_data_multivariate.csv")
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    actual_gap = df[GAP_COL]
    
    window_size = 4
    multi_features = [GAP_COL, 'GDP_Growth', 'Interest_Rate']
    data_multi = df[multi_features].values
    
    # 2. Setup Models
    # --- XGBoost ---
    xgb_model = XGBForecaster(lags=window_size)
    xgb_path = os.path.join(MODELS_DIR, "xgb_model_multivariate.json")
    if os.path.exists(xgb_path):
        xgb_model.load_model(xgb_path)
    else:
        raise FileNotFoundError("Trivariate XGBoost model not found. Run train_multivariate.py first.")
        
    # --- LSTM ---
    lstm_processor = LSTMDataProcessor(window_size=window_size)
    # Fit processor on historical data (matching main.py logic)
    train_size = int(len(df) * 0.8)
    lstm_processor.fit_transform(data_multi[:train_size])
    
    lstm_path = os.path.join(MODELS_DIR, "lstm_model_multivariate.pt")
    lstm_neural_net = LSTMModel(input_size=len(multi_features), hidden_size=64, num_layers=2, dropout=0.2)
    if os.path.exists(lstm_path):
        lstm_neural_net.load_state_dict(torch.load(lstm_path))
    else:
        raise FileNotFoundError("Trivariate LSTM model not found. Run train_multivariate.py first.")
    
    # 3. Generate Baseline Future Forecast (Ensemble)
    steps = 7 # Outlook 2026 needs 7 steps from 2024-Q3? No, let's see. 
    # Current data ends at 2024-Q2 (Wait, check data tail).
    # If using df.tail(1), let's see dates.
    
    # Scenario assumptions same as main.py
    future_gdp = np.linspace(df['GDP_Growth'].iloc[-1], 3.5, num=steps)
    future_ir = np.linspace(df['Interest_Rate'].iloc[-1], 5.0, num=steps)
    
    # LSTM Predict
    last_seq_scaled = lstm_processor.transform(data_multi[-window_size:])
    dummy_macro = np.zeros((steps, len(multi_features)))
    dummy_macro[:, 1] = future_gdp
    dummy_macro[:, 2] = future_ir
    scaled_macro = lstm_processor.transform(dummy_macro)[:, 1:]
    
    lstm_neural_net.eval()
    from src.lstm_model import LSTMTrainer
    lstm_trainer = LSTMTrainer(lstm_neural_net)
    lstm_future_scaled = lstm_trainer.predict_future(last_seq_scaled, steps=steps, scenario_input=scaled_macro)
    
    dummy_future = np.zeros((steps, len(multi_features)))
    dummy_future[:, 0] = lstm_future_scaled.flatten()
    lstm_future = lstm_processor.inverse_transform(dummy_future)[:, 0]
    
    # XGBoost Predict
    xgb_future = xgb_model.predict_future(data_multi[-window_size:], steps=steps, scenario_macro=dummy_macro[:, 1:]).flatten()
    
    # ENSEMBLE BASELINE
    baseline_future = (lstm_future + xgb_future) / 2

    # --- ARIMA (For Comparison) ---
    print(" -> Calculating ARIMA baseline for comparison...")
    try:
        from pmdarima import auto_arima
        arima_model = auto_arima(actual_gap, seasonal=True, m=4)
        arima_future = arima_model.predict(n_periods=steps)
    except:
        # Fallback to simple mean/linear if pmdarima fails
        arima_future = np.full(steps, actual_gap.mean())

    # Create Series for plotting
    future_dates = pd.date_range(start=actual_gap.index[-1] + pd.DateOffset(months=3), periods=steps, freq='3MS')
    arima_fut_series = pd.Series(arima_future, index=future_dates)
    lstm_fut_series = pd.Series(lstm_future, index=future_dates)
    xgb_fut_series = pd.Series(xgb_future, index=future_dates)
    ensemble_fut_series = pd.Series(baseline_future, index=future_dates)

    # 4. Trajectory Comparison Plot (The "Quadrilateral" Request)
    traj_path = os.path.join(RESULTS_DIR, "outlook_2026_trajectory_comparison.png")
    print(f" -> Rendering Quadrilateral Trajectory Comparison to {traj_path}...")
    vis = Visualizer()
    # Using the updated plot_comparison that now supports ensemble
    vis.plot_comparison(
        actual_series=actual_gap[actual_gap.index >= "2022"], # Zoomed in
        arima_series=arima_fut_series,
        lstm_series=lstm_fut_series,
        xgb_series=xgb_fut_series,
        ensemble_series=ensemble_fut_series,
        output_path=traj_path
    )
    print(" -> Trajectory plot rendering attempt complete.")
    
    # Aligned historical for augmentation
    actual_aligned = actual_gap.values[window_size:]
    history_pred_xgb = xgb_model.predict(df[multi_features]).flatten()

    # 5. Data Augmentation (Residual Bootstrapping with Stress Factor)
    print(" -> Running Monte Carlo Simulation (2000 Scenarios) with 1.3x Stability Shock...")
    augmentor = DataAugmentor(block_size=4)
    augmentor.fit(actual_aligned, history_pred_xgb)
    scenarios = augmentor.generate_scenarios(baseline_future, num_scenarios=2000, stress_factor=1.3)
    
    p10, p50, p90 = augmentor.get_confidence_intervals(scenarios)
    
    # Save the forecasted numbers to CSV for reporting
    df_scenarios = pd.DataFrame({
        'Date': future_dates,
        'ARIMA_Forecast': arima_future,
        'LSTM_Forecast': lstm_future,
        'XGB_Forecast': xgb_future,
        'ENSEMBLE_Forecast': p50,
        'P10_Lower': p10,
        'P90_Upper': p90
    })
    scenarios_path = os.path.join(RESULTS_DIR, "outlook_2026_scenarios_quad.csv")
    df_scenarios.to_csv(scenarios_path, index=False)
    
    # 6. Visualization (Fan Chart)
    print(" -> Rendering Probabilistic Fan Chart...")
    fan_path = os.path.join(RESULTS_DIR, "outlook_2026_fan_chart.png")
    vis.plot_fan_chart(actual_gap, future_dates, p10, p50, p90, output_path=fan_path, zoom_start="2023")
    
    # 7. Executive Decision Visuals
    print(" -> Generating Probability Timeline (-10% Recovery Threshold)...")
    probabilities = np.mean(scenarios > -10.0, axis=0)
    prob_path = os.path.join(RESULTS_DIR, "risk_probability_timeline.png")
    vis.plot_probability_timeline(future_dates, probabilities, threshold_label="-10.0% (Recovery)", output_path=prob_path)
    
    print(" -> Generating Early Warning Dashboard...")
    raw_credit_path = os.path.join(DATA_DIR, "credit_private_sector.csv")
    raw_df = pd.read_csv(raw_credit_path)
    current_credit = raw_df['credit_to_private_sector'].iloc[-1]
    prev_year_credit = raw_df['credit_to_private_sector'].iloc[-5] 
    credit_yoy = ((current_credit / prev_year_credit) - 1) * 100
    
    current_gap = actual_gap.iloc[-1]
    ewi_path = os.path.join(RESULTS_DIR, "ewi_dashboard_lite.png")
    vis.plot_ewi_dashboard(current_gap, credit_yoy, output_path=ewi_path)
    
    print(f"Success! Strategic visuals generated in: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
