# Brainstorming: Phase 4 (ARIMA Model)

Sesuai dengan praktik `writing-plans` dan `data-scientist`, Phase 4 tidak boleh dieksekusi begitu saja. Model time series klasik seperti ARIMA membutuhkan pengujian statistik yang teliti:

## 1. Fokus Variabel: Gap (Bukan Raw)
Target variabel kita adalah `credit_to_gdp_gap`. Seperti yang dibuktikan oleh ADF test di Phase 2 bahwa raw data tidak stasioner, data `gap` yang telah melalui HP filter secara matematis **cenderung jauh lebih stasioner** karena tren jangka panjangnya sudah tereliminasi.

## 2. Parameter ARIMA (p, d, q)
Berhubung kita berhadapan dengan deteksi dini krisis (makroekonomi skala LPS), kita tidak akan menebak-nebak parameter.
- Kita akan menggunakan **Auto-ARIMA** (melalui library `pmdarima` yang sudah diinstal di Phase 1).
- Pmdarima akan melakukan *grid-search* berdasarkan nilai AIC terendah.
- Mengingat ini adalah deret waktu metrik Ratio Gap (yang bersifat mean-reverting cyclically), kemungkinan nilai order differencing (`d`) adalah 0 atau 1.

## 3. Horizon Prediksi (h)
Untuk tujuan peringatan dini (*Early Warning*):
- Prediksi rentang pendek: **4 Quarter ke depan** (1 Tahun).
- Prediksi rentang menengah: **8 Quarter ke depan** (2 Tahun).
Kedua rentang ini cukup bagi LPS/Regulator untuk mengimplementasikan kebijakan Makroprudensial (*Counter-cyclical Capital Buffer*). Kita akan menggunakan *h=4* sebagai baseline utama.

## 4. Rincian Task per Step (writing-plans style)

### Task 4.1: Persiapan Kelas Base Model & Setup Unit Testing
- **Goal:** Membuat boilerplate kelas dan Test-Driven Development (TDD) harness agar model mudah disubtitusi di masa depan.
- **Implementasi:** Membuat `src/arima_model.py` dan `tests/test_arima.py`.

### Task 4.2: Train-Test Split (Time Series Style)
- **Goal:** Menghindari Data Leakage (Kebocoran Masa Depan ke Masa Lalu).
- **Aturan ML Data Science:** DILARANG menggunakan `train_test_split` acak khas sklearn (Shuffle). Data harus di-split secara sekuensial berdasarkan waktu. 
- **Setup:** 80% data awal sebagai Train, 20% data akhir (kuartal-kuartal paling modern) sebagai Test Set.

### Task 4.3: Eksekusi Auto-ARIMA & Fitting
- **Goal:** Membiarkan mesin mencari parameter `(p, d, q)` terbaik untuk data `gap` kita yang spesifik ini.
- **Library:** `pmdarima.auto_arima()`

### Task 4.4: Forecasting, Evaluasi Metrik, dan Export Model
- **Goal:** Melakukan *In-Sample Forecasting* (untuk test set) dan *Out-of-Sample Forecasting* (Masa depan murni).
- **Evaluation:** Menghitung skor RMSE dan MAE.
- **Artefak:** Menyimpan model (`.pkl` via joblib) ke direktori `models/`.

> File `implementation_plan.md` akan diubah secara radikal untuk mengikuti granularitas *bite-sized task* di atas.
