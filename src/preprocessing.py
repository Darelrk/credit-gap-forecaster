import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter
from src.config import DATE_COL, CREDIT_COL, GDP_COL, RATIO_COL, GAP_COL, TREND_COL

class DataPreprocessor:
    """Handles Feature Engineering and Data Standardization for forecasting."""
    
    @staticmethod
    def harmonize_units(df: pd.DataFrame) -> pd.DataFrame:
        """
        Harmonizes FRED data units into absolute base units.
        Credit: Billions -> Base
        GDP: Millions -> Base
        """
        df_out = df.copy()
        df_out[CREDIT_COL] = df_out[CREDIT_COL] * 1e9
        df_out[GDP_COL] = df_out[GDP_COL] * 1e6
        return df_out

    @staticmethod
    def calculate_yoy_growth(series: pd.Series) -> pd.Series:
        """
        Calculates Year-on-Year growth (percentage change over 4 quarters).
        """
        return series.pct_change(periods=4) * 100

    @staticmethod
    def calculate_ratio(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the Credit-to-GDP Ratio.
        Assumes data is already harmonized to the same base unit.
        """
        df_out = df.copy()
        df_out[RATIO_COL] = (df_out[CREDIT_COL] / df_out[GDP_COL]) * 100
        return df_out
        
    @staticmethod
    def apply_hp_filter(df: pd.DataFrame, lambda_val: float) -> pd.DataFrame:
        """
        Applies Hodrick-Prescott filter to extract the trend and gap.
        """
        df_out = df.copy()
        # Dropna to avoid errors from hpfilter
        clean_ratio = df_out[RATIO_COL].dropna()
        cycle, trend = hpfilter(clean_ratio, lamb=lambda_val)
        
        # Align trend to the main dataframe
        df_out.loc[clean_ratio.index, TREND_COL] = trend
        df_out[GAP_COL] = df_out[RATIO_COL] - df_out[TREND_COL]
        return df_out

    def process_all(self, df: pd.DataFrame, lambda_val: float) -> pd.DataFrame:
        """
        Executes the entire feature engineering pipeline.
        """
        df = self.harmonize_units(df)
        df[f"{CREDIT_COL}_yoy"] = self.calculate_yoy_growth(df[CREDIT_COL])
        df[f"{GDP_COL}_yoy"] = self.calculate_yoy_growth(df[GDP_COL])
        df = self.calculate_ratio(df)
        df = self.apply_hp_filter(df, lambda_val=lambda_val)
        return df

if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.config import PROCESSED_DATA_PATH, HP_LAMBDA
    import os
    
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

    print(f"Loading raw data...")
    loader = DataLoader()
    raw_df = loader.merge_datasets()
    
    print(f"Applying Feature Engineering (HP_LAMBDA={HP_LAMBDA})...")
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.process_all(raw_df, lambda_val=HP_LAMBDA)
    
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data logically saved to: {PROCESSED_DATA_PATH}")
    print("\n--- SAMPLE OUTPUT ---")
    print(processed_df[[DATE_COL, RATIO_COL, TREND_COL, GAP_COL]].head())
    print("\nFeature Engineering Phase 3 Complete.")
