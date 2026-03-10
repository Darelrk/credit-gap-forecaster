# Tech Context: Credit-to-GDP Gap Forecasting

## Stack Teknologi (Terkini)
- **Bahasa**: Python 3.14.2
- **Data Manipulation**: `pandas`, `numpy`
- **Visualisasi**: `matplotlib` (Backend: Agg untuk kompabilitas Windows headless)
- **Ekosistem Pemodelan**:
  - `statsmodels` (Kalkulasi Hodrick-Prescott Filter)
  - `pmdarima` (Auto-ARIMA Pipeline)
  - `torch` (PyTorch) - *Pengganti TensorFlow yang terindikasi cacat kompatibilitas instalasi pada Python v3.14 Windows.*
  - `scikit-learn` (Evaluator Metrik RMSE & MAE + MinMaxScaler)
- **Metodologi Quality Control**: `pytest` (Test-Driven Development)
- **Environment**: Virtual Environment (`.venv`) lokal di direktori kerja.

## Pengaturan Pengembangan
- Data mentah (`.csv`) berada di folder `data/`.
- Kode modular dieksekusi dari `src/` (package).
- Output model statis diekspor sebagai `.pkl` (ARIMA) & `.pt` (PyTorch LSTM) di `models/`.

## Batasan
- Data bersifat kuartalan, membatasi resolusi (*sample size*). Diakali melalui simplifikasi parameter tersembunyi (*hidden node*) di LSTM.
- Memori RAM 16GB pada perangkat ADVAN 1701 cukup memadai untuk *Training Loop* jaringan saraf ringan ini secara CPU-Bound.
