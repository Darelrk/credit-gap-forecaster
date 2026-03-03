import pandas as pd
import os
from src.config import RAW_DATA_PATH, DATE_COL, CREDIT_COL, GDP_COL

class DataLoader:
    """
    Data Ingestion Layer untuk analisis Siklus Kredit.
    Menangani orkestrasi data mentah (Credit & GDP) menjadi dataset terintegrasi
    yang siap diolah untuk kalkulasi Credit Gap sesuai standar Basel III.
    """
    
    def __init__(self, raw_data_path=RAW_DATA_PATH):
        self.raw_data_path = raw_data_path
        
    def load_credit_data(self):
        """Loads credit data from CSV."""
        path = os.path.join(self.raw_data_path, "credit_private_sector.csv")
        df = pd.read_csv(path)
        # Assuming the CSV has columns 'DATE' and 'VALUE'
        df.columns = [DATE_COL, CREDIT_COL]
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        return df
        
    def load_gdp_data(self):
        """Loads GDP data from CSV."""
        path = os.path.join(self.raw_data_path, "real_gdp_indonesia.csv")
        df = pd.read_csv(path)
        # Assuming the CSV has columns 'DATE' and 'VALUE'
        df.columns = [DATE_COL, GDP_COL]
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        return df
        
    def merge_datasets(self):
        """Merges credit and GDP datasets on date."""
        credit_df = self.load_credit_data()
        gdp_df = self.load_gdp_data()
        
        # Merge on date (inner join to get overlapping period 2000-2025)
        merged_df = pd.merge(credit_df, gdp_df, on=DATE_COL, how='inner')
        merged_df = merged_df.sort_values(DATE_COL).reset_index(drop=True)
        
        return merged_df

if __name__ == "__main__":
    loader = DataLoader()
    df = loader.merge_datasets()
    print("Merged Data Snippet:")
    print(df.head())
    print("\nData Info:")
    print(df.info())
