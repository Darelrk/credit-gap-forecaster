import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from src.config import RESULTS_DIR, GAP_COL

def run_correlation_analysis():
    print("=" * 40)
    print("MULTIVARIATE CORRELATION ANALYSIS")
    print("=" * 40)
    
    data_path = "data/processed_data_multivariate.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    
    # Select variables for correlation - TRINITAS SYSTEM
    features = ['credit_yoy', 'gdp_yoy', 'ratio', 'gap', 'Interest_Rate']
    corr_df = df[features].copy()
    
    # Calculate Correlation Matrix (Pearson)
    corr_matrix = corr_df.corr(method='pearson')
    print("\nPearson Correlation Matrix:")
    print(corr_matrix.round(3))
    
    # Visualization
    plt.figure(figsize=(8, 6), facecolor='#FAFAFA')
    ax = plt.gca()
    ax.set_facecolor('#FAFAFA')
    
    # Plot heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) if False else None # Optional mask for upper triangle
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, 
                square=True, linewidths=.5, cbar_kws={"shrink": .8},
                annot_kws={"size": 12, "weight": "bold"})
                
    plt.title('Macroeconomic Variables Correlation Matrix', fontsize=16, fontweight='bold', pad=20, color='#333333')
    
    # Save Image
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "multivariate_correlation_matrix.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor='#FAFAFA')
    plt.close()
    
    print(f"\n-> Correlation Matrix visualized and saved to: {out_path}")
    
if __name__ == "__main__":
    import numpy as np # import inside because we used it conditionally or just at the top
    run_correlation_analysis()
