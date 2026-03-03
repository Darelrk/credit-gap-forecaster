import pytest
import pandas as pd
import numpy as np
import os
from src.preprocessing import DataPreprocessor
from src.config import DATE_COL, CREDIT_COL, GDP_COL, RATIO_COL, GAP_COL, TREND_COL

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Fixture providing a sample DataFrame for testing."""
    dates = pd.date_range(start="2000-01-01", periods=8, freq="QS")
    # Simulate Credit in Billions and GDP in Millions
    credit = np.array([1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700], dtype=float)
    gdp = np.array([2000000, 2100000, 2200000, 2300000, 2400000, 2500000, 2600000, 2700000], dtype=float)
    
    return pd.DataFrame({
        DATE_COL: dates,
        CREDIT_COL: credit,
        GDP_COL: gdp
    })

def test_harmonize_units(sample_data: pd.DataFrame) -> None:
    """Test if harmonization scales the columns correctly."""
    processor = DataPreprocessor()
    df_harmonized = processor.harmonize_units(sample_data.copy())
    
    # Credit x 1,000,000,000 (Billions -> Base)
    # GDP x 1,000,000 (Millions -> Base)
    expected_credit = sample_data[CREDIT_COL].iloc[0] * 1e9
    expected_gdp = sample_data[GDP_COL].iloc[0] * 1e6
    
    assert df_harmonized[CREDIT_COL].iloc[0] == expected_credit
    assert df_harmonized[GDP_COL].iloc[0] == expected_gdp

def test_calculate_yoy_growth(sample_data: pd.DataFrame) -> None:
    """Test YoY logic (percentage change across 4 quarters)."""
    processor = DataPreprocessor()
    series = processor.calculate_yoy_growth(sample_data[CREDIT_COL])
    
    # The first 4 elements should be NaN
    assert series.iloc[0:4].isna().all()
    
    # 5th element: (1400 - 1000) / 1000 * 100 = 40.0%
    assert np.isclose(series.iloc[4], 40.0)

def test_calculate_ratio(sample_data: pd.DataFrame) -> None:
    """Test Ratio calculation logic."""
    processor = DataPreprocessor()
    # Assume harmonized data here to prevent skewed ratio
    df_harmonized = processor.harmonize_units(sample_data.copy())
    df_ratio = processor.calculate_ratio(df_harmonized)
    
    expected_ratio = (df_harmonized[CREDIT_COL].iloc[0] / df_harmonized[GDP_COL].iloc[0]) * 100
    assert RATIO_COL in df_ratio.columns
    assert np.isclose(df_ratio[RATIO_COL].iloc[0], expected_ratio)

def test_apply_hp_filter(sample_data: pd.DataFrame) -> None:
    """Test HP Filter application."""
    processor = DataPreprocessor()
    df_harmonized = processor.harmonize_units(sample_data.copy())
    df_ratio = processor.calculate_ratio(df_harmonized)
    df_processed = processor.apply_hp_filter(df_ratio, lambda_val=1600)
    
    assert TREND_COL in df_processed.columns
    assert GAP_COL in df_processed.columns
    # Gap should mathematically be ratio - trend
    expected_gap = df_processed[RATIO_COL].iloc[0] - df_processed[TREND_COL].iloc[0]
    assert np.isclose(df_processed[GAP_COL].iloc[0], expected_gap)
