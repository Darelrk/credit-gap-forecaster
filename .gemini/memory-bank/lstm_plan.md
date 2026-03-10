# Brainstorming: Phase 5 (Deep Learning LSTM)

Fase 5 adalah titik eksploratif di mana kita mengkomparasi *baseline* statis ARIMA dengan kecerdasan buatan berbasis *Deep Learning* (LSTM) yang mampu mengingat pola non-linear jangka panjang.

## 1. Hambatan Kritis: Kompatibilitas Framework
**Masalah**: Pada saat *Setup Environment* (Phase 1), pustaka `tensorflow` **gagal diinstal** karena pip tidak dapat menemukan paket (*wheel*) yang cocok dengan Python versi 3.14.2 di arsitektur Windows yang Anda gunakan (ini wajar, Google/TensorFlow biasanya tertinggal dalam merilis dukungan untuk rilis Python teranyar).

**Solusi Alternatif (Keputusan Dibutuhkan)**:
1.  **Migrasi ke PyTorch**: PyTorch biasanya jauh lebih cepat mengadopsi rilis Python baru (atau versi *nightly* mereka mendukung). Kita bisa menulis modul `lstm_model.py` murni menggunakan `torch.nn`. 
    - *Kelebihan*: Standar industri, sangat modular, *object-oriented* (sangat pas untuk filosofi *Clean Code* kita).
2.  **Fallback ke Machine Learning Ensemble (XGBoost/RandomForest)**: Jika Deep Learning ditolak karena masalah kompatibilitas sistem terlalu rumit, kita ubah regresi time-series ini menggunakan ML konvensional.
3.  **Downgrade Python (Tidak Disarankan)**: Meminta pengguna mengubah environment base ke Python 3.11/3.12 (Membuang-buang waktu setup).

> **Rekomendasi Data Scientist**: Kita akan beralih (*pivot*) menggunakan **PyTorch** untuk LSTM.

## 2. Pre-processing Khusus LSTM (Sangat Krusial)
Arsitektur Neural Network menuntut penanganan data yang sama sekali berbeda dari ARIMA:
1. **MinMaxScaler [-1, 1] atau [0, 1]**: Jaringan LSTM sangat sensitif terhadap skala input. Nilai `gap` kita harus diskalakan. *Leakage Warning*: Scaler hanya boleh di-_fit_ pada **Data Latihlah (Train)**, kemudian digunakan men-_transform_ data Uji (Test).
2. **Sequence Windowing (Sliding Window)**: Data tabuler tunggal harus dikonversi. 
   - *Input (X)*: 4 Kuartal masa lalu (Q1, Q2, Q3, Q4)
   - *Target (y)*: 1 Kuartal di masa depan (Q5).

## 3. Rincian Task TDD (Writing-Plans Style)

Jika kita mengambil jalur PyTorch (atau jika TF ternyata bisa diapakai via nightly/alternatif), berikut rencana TDD kita:

### Task 5.1: Sequence Creator & TDD
- **Goal**: Memecah deret berkala 1D menjadi matriks 3D `[samples, window_size, features]`.
- **Test**: Masukkan array [1,2,3,4,5,6] dengan *window*=3. Validasi *shape* X adalah (3, 3, 1) dan *shape* y (3,).

### Task 5.2: Scaling Data (Anti-Leakage)
- **Goal**: MinMaxScaler yang hanya mempelajari set latih.
- **Implementasi**: Menambah properti `self.scaler` ke dalam class wrapper.

### Task 5.3: Pembangunan Arsitektur & Training Loop
- **Goal**: _Fit_ model ke deret historis.
- **Arsitektur (PyTorch/Keras)**: `LSTM Layer (Hidden: 64)` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Linear/Dense Layer (32)` $\rightarrow$ `Linear/Dense (1)`.
- **Loss Function**: Mean Squared Error (MSE). Optimizer: Adam (Learning Rate: 0.001).
- **Epochs**: ~50-100 dengan _early stopping_ sederhana.

### Task 5.4: Forecasting Out-of-Sample
- **Goal**: LSTM harus melakukan prediksi iteratif untuk menembus rentang masa depan (kuartal $+4$). Nilai prediksi dimasukkan kembali sebagai "masa lalu" untuk memprediksi kuartal berikutnya (*Autoregressive Roll-forward*).

### Task 5.5: Inversi Skala
- **Goal**: Semua angka prediksi dan metrik (RMSE/MAE) direkonstruksi (*inverse transform*) kembali ke unit skala _Gap_ asli (Rasio Persentase).

---
Bila TDD Phase 5 ini berhasil, fase akhir (Phase 6) hanyalah menjalankan `main.py` yang menabrakkan ARIMA vs LSTM di satu layar matriks (*Dashboard*/Plot komparasi pamungkas).
