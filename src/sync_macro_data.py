import pandas as pd
import os
from datetime import datetime
from src.config import PROCESSED_DATA_PATH, DATE_COL

# Macro Variables CONFIG (Trivariate Model)
MACRO_SERIES = {
    'GDP_Growth': 'LOCAL_GDP',
    'Interest_Rate': 'IR3TIB01IDQ156N'
}

class MacroSynchronizer:
    """
    Synchronizes external macroeconomic data from FRED/Local dumps 
    into the core Credit Gap dataset.
    """
    
    def __init__(self, processed_path=PROCESSED_DATA_PATH):
        self.processed_path = processed_path
        
    def fetch_from_fred_dumps(self, dump_dir):
        """
        Reads local GDP tracking data and converts it into YoY growth rates.
        """
        macro_dfs = []
        gdp_path = 'data/real_gdp_indonesia.csv'
        if os.path.exists(gdp_path):
            df_gdp = pd.read_csv(gdp_path)
            df_gdp[DATE_COL] = pd.to_datetime(df_gdp['date'])
            # Compute YoY Growth (4 quarters)
            df_gdp['GDP_Growth'] = df_gdp['real_gdp'].pct_change(periods=4) * 100
            # Drop NaN from pct_change
            df_gdp = df_gdp.dropna()
            df_macro = df_gdp[[DATE_COL, 'GDP_Growth']]
            macro_dfs.append(df_macro)
        else:
            print(f"Warning: Data for GDP_Growth not found at {gdp_path}")
            
        # 2. Interest Rate Data
        ir_path = 'data/interest_rate_indonesia.csv'
        if os.path.exists(ir_path):
            df_ir = pd.read_csv(ir_path)
            # Ensure date column name matches
            if 'DATE' in df_ir.columns:
                df_ir.rename(columns={'DATE': DATE_COL}, inplace=True)
            df_ir[DATE_COL] = pd.to_datetime(df_ir[DATE_COL])
            # Rename the value column to 'Interest_Rate'
            if 'IR3TIB01IDQ156N' in df_ir.columns:
                df_ir.rename(columns={'IR3TIB01IDQ156N': 'Interest_Rate'}, inplace=True)
            df_macro_ir = df_ir[[DATE_COL, 'Interest_Rate']]
            macro_dfs.append(df_macro_ir)
        else:
            print(f"Warning: Data for Interest_Rate not found at {ir_path}")
        
        return macro_dfs

    def sync_and_merge(self, macro_list):
        """Merges macro variables into the main processed dataframe."""
        if not os.path.exists(self.processed_path):
            raise FileNotFoundError(f"Base data not found at {self.processed_path}")
            
        df_base = pd.read_csv(self.processed_path)
        df_base[DATE_COL] = pd.to_datetime(df_base[DATE_COL])
        
        for m_df in macro_list:
            df_base = pd.merge(df_base, m_df, on=DATE_COL, how='left')
            
        # Data Cleaning: Handle missing values in macro data
        for col in MACRO_SERIES.keys():
            if col in df_base.columns:
                df_base[col] = df_base[col].ffill().bfill()
        
        # Save output
        output_path = self.processed_path.replace('.csv', '_multivariate.csv')
        df_base.to_csv(output_path, index=False)
        print(f"Multivariate data synced to: {output_path}")
        return output_path

if __name__ == "__main__":
    sync = MacroSynchronizer()
    macro_list = sync.fetch_from_fred_dumps('data/external')
    sync.sync_and_merge(macro_list)
