import pandas as pd
import matplotlib.pyplot as plt
import os
from src.config import RESULTS_DIR, PROCESSED_DATA_PATH, GAP_COL

def create_policy_signal_plot():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    latest_gap = df[GAP_COL].iloc[-1]
    
    plt.figure(figsize=(10, 6))
    
    # Define zones
    plt.axhspan(-30, 2, color='green', alpha=0.3, label='Zona AMAN (Green)')
    plt.axhspan(2, 10, color='yellow', alpha=0.3, label='Zona WASPADA (Yellow)')
    plt.axhspan(10, 30, color='red', alpha=0.3, label='Zona KRITIS (Red)')
    
    # Plot current status
    plt.bar(['Status Saat Ini (Credit Gap)'], [latest_gap], color='blue', width=0.5)
    plt.text(0, latest_gap + (1 if latest_gap > 0 else -3), f'{latest_gap:.2f}%', 
             ha='center', va='bottom' if latest_gap > 0 else 'top', fontsize=14, fontweight='bold')
    
    plt.axhline(0, color='black', linewidth=1)
    plt.ylim(-30, 20)
    plt.ylabel('Credit-to-GDP Gap (%)', fontsize=12)
    plt.title('Policy Signal: Ringkasan Risiko Perbankan', fontsize=16, fontweight='bold')
    plt.legend(loc='upper right')
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    save_path = os.path.join(RESULTS_DIR, "policy_signal.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    create_policy_signal_plot()
