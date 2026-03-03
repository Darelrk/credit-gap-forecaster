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
    """Plots Credit and GDP trends."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=DATE_COL, y=CREDIT_COL, label='Total Credit')
    sns.lineplot(data=df, x=DATE_COL, y=GDP_COL, label='Real GDP')
    plt.title('Raw Data Trends: Credit vs GDP Indonesia')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(RESULTS_DIR, "raw_trends.png")
    plt.savefig(save_path)
    print(f"\nTrend plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    from src.data_loader import DataLoader
    loader = DataLoader()
    df = loader.merge_datasets()
    
    perform_basic_eda(df)
    plot_raw_trends(df)
