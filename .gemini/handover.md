# Handover Notes - Credit-to-GDP Gap Forecasting Project

## Project Overview
Project portofolio untuk magang di LPS (Lembaga Penjamin Simpanan). 
Topik: Credit-to-GDP Gap - indikator deteksi krisis sistem keuangan.

## Strategy
Strategi 2 dari Gemini: Memprediksi apakah pertumbuhan kredit terlalu cepat 
dibandingkan pertumbuhan ekonomi (Credit-to-GDP Gap).

## Data Sources
- **FRED API** (Federal Reserve Economic Data)
- Series ID:
  - `CRDQIDAPABIS` - Total Credit to Private Sector (Quarterly, 1976-2025)
  - `NGDPRSAXDCIDQ` - Real GDP Indonesia (Quarterly, 2000-2025)

## Current Status

### ✅ Completed:
1. Downloaded data Credit dari FRED (198 observasi, 1976-2025)
2. Downloaded data GDP dari FRED (104 observasi, 2000-2025)
3. Created project directories: `data/`, `models/`, `notebooks/`, `src/`
4. Created `requirements.txt` dengan dependencies
5. Created venv di `.venv/`
6. Created memory-bank directory di `.opencode/memory-bank/`

### ⏳ Pending:
1. Activate venv dan install dependencies
2. Load & merge dataset (align periode 2000-2025)
3. EDA (Exploratory Data Analysis)
4. Feature Engineering: hitung Credit-to-GDP ratio & gap
5. Build Forecasting Model (ARIMA/Prophet/LSTM)
6. Evaluation & Visualization
7. Dashboard Streamlit (opsional)

## Files Created
```
C:\PYTHON\Forecasting\
├── data/
│   ├── credit_private_sector.csv
│   └── real_gdp_indonesia.csv
├── models/
├── notebooks/
├── src/
├── requirements.txt
├── .venv/                  (virtual environment)
└── .opencode/
    └── memory-bank/        (belum diisi)
```

## Next Steps
1. Activate `.venv\Scripts\activate` (Windows)
2. Run `pip install -r requirements.txt`
3. Load data di Python dan merge berdasarkan date
4. Hitung Credit-to-GDP ratio
5. Forecast menggunakan model time series

## Notes
- Project ini fokus pada analisis makroprudensial untuk LPS
- Target: memprediksi apakah Credit-to-GDP gap menandakan risiko krisis
- Model yang disarankan: ARIMA, Prophet, atau LSTM
