# Project Brief: Credit-to-GDP Gap Forecasting

## Deskripsi Projek
Projek ini bertujuan untuk membangun model peramalan (forecasting) untuk variabel **Credit-to-GDP Gap**. Ini adalah indikator makroprudensial yang digunakan untuk mendeteksi potensi krisis sistem keuangan karena pertumbuhan kredit yang berlebihan dibandingkan dengan pertumbuhan ekonomi.

## Tujuan Utama
- Memprediksi nilai Credit-to-GDP Gap di masa depan.
- Mendeteksi risiko krisis keuangan untuk kepentingan Lembaga Penjamin Simpanan (LPS).

## Dataset
- **Sumber**: FRED API (Federal Reserve Economic Data) & BI/Data Dumps.
- **Series ID / Features**: 
  - `CRDQIDAPABIS`: Total Kredit ke Sektor Swasta (Kuartalan).
  - `NGDPRSAXDCIDQ`: PDB Riil Indonesia (Kuartalan / GDP Growth).
  - `Interest_Rate`: Suku Bunga Acuan (Trivariate Driver).

## Output yang Diharapkan
1. Model peramalan Quadrilateral (ARIMA, LSTM Attention, XGBoost, ENSEMBLE).
2. Visualisasi tren ratio, gap kredit, dan korelasi makro (Interest Rate/GDP).
3. Dashboard Early Warning System (EWI) untuk stabilitas makroprudensial.
