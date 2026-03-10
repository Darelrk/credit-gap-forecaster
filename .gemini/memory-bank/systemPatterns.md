# System Patterns: Credit-to-GDP Gap Forecasting

## Arsitektur Sistem & Metodologi
Proyek ini mengadopsi struktur *Pipeline Modular* berbasis keilmuan Data Science:

1. **Data Ingestion** (`sync_macro_data.py`): Menggabungkan dataset FRED dan metadata lokal (Suku Bunga).
2. **Feature Engineering** (`preprocessing.py` & `xgb_model.py` core):
   - **HP Filter**: Basel III ($\lambda = 400.000$).
   - **Trivariate Drivers**: Mengintegrasikan GDP Growth dan Interest Rate sebagai input multivariat.
3. **Quadrilateral Model Architecture**:
   - **ARIMA** (`arima_model.py`): Baseline statistik linear.
   - **Attention-LSTM** (`lstm_model.py`): Neural network sensitif memori jangka panjang.
   - **XGBoost** (`xgb_model.py`): Non-linear pattern capture.
   - **ENSEMBLE** (`main.py`): Hybrid LSTM + XGBoost untuk stabilitas.
4. **Strategic Outlook** (`generate_outlook_2026.py`): Simulasi masa depan (Fan Chart & Monte Carlo).
5. **Visualization** (`visualization.py`): Dashboard headless dengan logika *dynamic x-axis*.
