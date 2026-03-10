import numpy as np
import pandas as pd
from typing import List, Tuple

class DataAugmentor:
    """
    Implements Residual Block Bootstrapping for Time Series Data Augmentation.
    """
    def __init__(self, block_size: int = 4):
        self.block_size = block_size
        self.residuals = None

    def fit(self, actual: np.ndarray, predicted: np.ndarray):
        """
        Calculates residuals from historical data and predictions.
        """
        # Ensure they are the same length
        min_len = min(len(actual), len(predicted))
        self.residuals = actual[-min_len:] - predicted[-min_len:]

    def generate_scenarios(self, baseline_forecast: np.ndarray, num_scenarios: int = 1000, stress_factor: float = 1.0) -> np.ndarray:
        """
        Generates multiple scenarios by adding bootstrapped residuals to the baseline forecast.
        If stress_factor > 1.0, negative residuals (bad shocks) are amplified by the factor.
        Returns: array of shape (num_scenarios, forecast_steps)
        """
        if self.residuals is None:
            raise ValueError("Augmentor must be fitted with residuals first.")

        forecast_steps = len(baseline_forecast)
        scenarios = np.zeros((num_scenarios, forecast_steps))

        for i in range(num_scenarios):
            # Create a synthetic residual series for the forecast period
            synthetic_residual = []
            while len(synthetic_residual) < forecast_steps:
                # Pick a random block from historical residuals
                start_idx = np.random.randint(0, len(self.residuals) - self.block_size + 1)
                block = self.residuals[start_idx : start_idx + self.block_size].copy()
                
                # 🌟 MACRO SHOCK SIMULATION 🌟
                # Amplify negative residuals if stress factor is applied
                if stress_factor > 1.0:
                    block = np.array([r * stress_factor if r < 0 else r for r in block])
                
                synthetic_residual.extend(block)
            
            # Trim and add to baseline
            synthetic_residual = np.array(synthetic_residual[:forecast_steps])
            
            # Add Gaussian noise for additional jittering
            jitter = np.random.normal(0, np.std(self.residuals) * 0.1, forecast_steps)
            
            scenarios[i] = baseline_forecast + synthetic_residual + jitter

        return scenarios

    def get_confidence_intervals(self, scenarios: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates P10, P50 (Median), and P90 from generated scenarios.
        """
        p10 = np.percentile(scenarios, 10, axis=0)
        p50 = np.percentile(scenarios, 50, axis=0)
        p90 = np.percentile(scenarios, 90, axis=0)
        return p10, p50, p90
