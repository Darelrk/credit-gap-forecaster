import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os

class Visualizer:
    """XGBoost based time-series forecaster."""
    
    def __init__(self):
        # Set overall seaborn-like styling for professional data science aesthetics
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def plot_comparison(self, actual_series, arima_series=None, lstm_series=None, xgb_series=None, output_path="comparison.png"):
        """
        Creates a comparative line chart and saves it.
        """
        plt.figure(figsize=(12, 6))
        
        # Plot Actual
        plt.plot(actual_series.index, actual_series.values, label='Actual Credit Gap', color='black', linewidth=2)
        
        if arima_series is not None:
            plt.plot(arima_series.index, arima_series.values, label='ARIMA Forecast', linestyle='--', color='blue')
            
        if lstm_series is not None:
            plt.plot(lstm_series.index, lstm_series.values, label='LSTM Forecast (Attention)', linestyle='--', color='green', linewidth=2)
            
        if xgb_series is not None:
            plt.plot(xgb_series.index, xgb_series.values, label='XGBoost Forecast', linestyle=':', color='orange', linewidth=2)
        
        # Titles and Labels
        plt.title('Performance Comparison: Actual Data vs Models (ARIMA & LSTM)', fontsize=14, pad=15)
        plt.xlabel('Quarters / Timeline', fontsize=12)
        plt.ylabel('Credit-to-GDP Gap (%)', fontsize=12)
        
        # Legenda & Grid
        plt.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
        
        # Simpan ke memori (Headless environment)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
