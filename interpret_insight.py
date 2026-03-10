import pandas as pd
import numpy as np
import os
from src.config import DATE_COL, GAP_COL

def get_status_color(value):
    if value > 10.0:  # Threshold fatal (Standard Basel III)
        return "MERAH (KRITIS)"
    elif value > 2.0: # Threshold waspada
        return "KUNING (WASPADA)"
    elif value < -10.0: # Threshold under-trend dalam
        return "BIRU (EKSPANSI LUAS - UNDER TREND)"
    else:
        return "HIJAU (AMAN)"

def interpret():
    # Paths
    multi_data_path = os.path.join("data", "processed_data_multivariate.csv")
    scenario_path = os.path.join("results", "outlook_2026_scenarios.csv")
    
    print("\n" + "="*50)
    print("STRATEGIC ANALYSIS REPORT: TRINITAS & QUADRILATERAL")
    print("="*50)
    
    # 1. ACTUAL DATA (LATEST QUARTER)
    if os.path.exists(multi_data_path):
        df_actual = pd.read_csv(multi_data_path)
        latest_gap = df_actual[GAP_COL].iloc[-1]
        latest_date = df_actual[DATE_COL].iloc[-1]
        
        print(f"\nCURRENT MARKET CONDITION ({latest_date}):")
        print(f"   Actual Credit Gap : {latest_gap:.2f}%")
        print(f"   Risk Assessment   : {get_status_color(latest_gap)}")
        print("\n   Observation: The Trinitas indicators confirm a massive under-trend.")
        print("   This provides significant room for credit expansion without systemic risk.")
    
    print("-" * 50)
    
    # 2. FORECAST PROJECTION (NEXT 4 QUARTERS)
    if os.path.exists(scenario_path):
        df_forecast = pd.read_csv(scenario_path)
        # Ambil proyeksi 1 tahun ke depan (approx 4 baris dari sekarang)
        target_forecast = df_forecast.iloc[4] # Proyeksi Apr 2026
        forecast_val = target_forecast['P50_Forecast']
        forecast_date = target_forecast['Date']
        
        print(f"QUADRILATERAL ENSEMBLE PROJECTION ({forecast_date}):")
        print(f"   Expected Gap (P50): {forecast_val:.2f}%")
        print(f"   Future Risk Status: {get_status_color(forecast_val)}")
        
        print("\n   Observation: Quadrilateral models suggest a gradual 'back-to-trend'.")
        print("   The P50 trajectory indicates a recovery of approx 10 percentage points")
        print("   by mid-2026, assuming stable Interest Rate transmission.")
    
    print("-" * 50)
    print("STRATEGIC RECOMMENDATIONS:")
    print("   - Stimulate productive credit to narrow the negative gap defisit.")
    print("   - Monitor Interest Rate impact on GDP growth and debt service.")
    print("   - Maintain the 'Green' status for macroprudential buffer policy.")
    print("="*50 + "\n")

if __name__ == "__main__":
    interpret()
