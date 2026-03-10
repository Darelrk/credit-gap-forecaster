import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.config import RESULTS_DIR, DATE_COL, CREDIT_COL, GDP_COL

def perform_basic_eda(df):
    """Performs basic statistical analysis on the merged dataframe."""
    print("--- Basic Statistics ---")
    print(df.describe())
    
    # Missing values
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    # Correlation
    print("\n--- Correlation Matrix ---")
    corr = df[[CREDIT_COL, GDP_COL]].corr()
    print(corr)
    return corr


def plot_raw_trends(df):
    """Plots Credit, GDP, and Interest Rate trends."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    plt.figure(figsize=(14, 7), facecolor='#FAFAFA')
    ax = plt.gca()
    ax.set_facecolor('#FAFAFA')
    
    # Use dual axes since Interest Rate is in % scale and others are massive integers
    sns.lineplot(data=df, x=DATE_COL, y=CREDIT_COL, label='Total Credit (Left)', color='#1F77B4')
    sns.lineplot(data=df, x=DATE_COL, y=GDP_COL, label='Real GDP (Left)', color='#2CA02C')
    plt.ylabel('Currency Value (IDR)', fontsize=12)
    
    ax2 = ax.twinx()
    if 'Interest_Rate' in df.columns:
        sns.lineplot(data=df, x=DATE_COL, y='Interest_Rate', label='Interest Rate (Right)', color='#D62728', ax=ax2)
        ax2.set_ylabel('Interest Rate (%)', fontsize=12, color='#D62728')
        ax2.tick_params(axis='y', labelcolor='#D62728')
    
    plt.title('Trivariate Raw Data Trends: Credit, GDP, and Interest Rate Indonesia', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=12)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    save_path = os.path.join(RESULTS_DIR, "raw_trends.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nTrend plot updated and saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    # Load multivariate data to include Interest Rate for the Trinitas view
    data_path = os.path.join("data", "processed_data_multivariate.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    else:
        from src.data_loader import DataLoader
        loader = DataLoader()
        df = loader.merge_datasets()
    
    perform_basic_eda(df)
    plot_raw_trends(df)
