# Analisis Prediktif Credit-to-GDP Gap: Pendekatan Multi-Model

Proyek ini mengeksplorasi penggunaan kecerdasan buatan dan statistika lanjut untuk memodelkan indikator stabilitas keuangan paling krusial bagi regulator: **Credit-to-GDP Gap**. 

Sistem ini dirancang bukan hanya untuk memprediksi angka, tetapi juga memahami "napas" siklus ekonomi melalui tiga lensa algoritma yang berbeda.

---

## 🧭 Metodologi Dasar: Memisahkan Tren dari Spekulasi

Sebelum peramalan dilakukan, data mentah harus melalui proses **Hodrick-Prescott (HP) Filter**. Dalam dunia filtrasi makroekonomi (Standar Basel III), ini adalah langkah vital untuk:
-   **Tren**: Mengidentifikasi pertumbuhan ekonomi yang fundamental dan sehat.
-   **Gap**: Mengekstrak penyimpangan (anomali) yang seringkali menjadi tanda awal dari penumpukan risiko sistemik.
Dengan $\lambda = 400.000$, sistem ini memastikan rujukan yang stabil untuk data kuartalan Indonesia.

---

## 🧠 Mesin Peramalan: Tiga Pilar Kecerdasan

Sistem ini tidak bergantung pada satu opini tunggal. Kami mengintegrasikan tiga model dengan karakteristik unik:

### 1. LSTM with Attention (The Intelligent Memory)
Ini adalah "otak" utama sistem yang dibangun menggunakan *Deep Learning* (PyTorch).
-   **Mengapa LSTM?**: Data keuangan memiliki ingatan jangka panjang. LSTM mampu mengingat pola krisis masa lalu yang mungkin berulang puluhan kuartal kemudian.
-   **Kekuatan Attention**: Tidak semua data masa lalu sama pentingnya. Mekanisme *Attention* memungkinkan model untuk memberikan bobot lebih pada periode-periode volatil (seperti krisis 2008 atau pandemi) saat memprediksi masa depan, sehingga hasil peramalan lebih adaptif terhadap syok ekonomi.

### 2. ARIMA (The Statistical Foundation)
Model statistik klasik yang bertindak sebagai "jangkar" stabilitas.
-   **Peran**: ARIMA memastikan bahwa prediksi tetap berpijak pada momentum jangka pendek yang logis. Jika model AI terlalu agresif, ARIMA memberikan rujukan linear yang membantu menjaga akurasi tetap realistis.

### 3. XGBoost (The Pattern Matcher)
Algoritma *Gradient Boosting* yang sangat efisien dalam memetakan hubungan non-linier antara fitur-fitur masa kini.
-   **Peran**: XGBoost sangat baik dalam menangkap interaksi rumit antar variabel lag (jeda waktu). Ia berfungsi sebagai pembanding presisi tinggi untuk melihat apakah ada pola tabular tertentu yang terlewatkan oleh model sekuensial.

---

## 📊 Sinergi Pengambilan Keputusan

Hasil akhir dari sistem ini adalah konsensus. Ketika ketiga model menunjukkan arah yang sama (misal: tren mendaki yang tajam), sistem akan memicu sinyal **Peringatan Dini**. 

Pemisahan antara "Kebenaran Statistik" (ARIMA), "Kecerdasan Sekuensial" (LSTM), dan "Efisiensi Pola" (XGBoost) memberikan pandangan 360 derajat bagi regulator untuk menentukan kebijakan makroprudensial yang lebih tepat sasaran.

---

## 🖥️ Ringkasan Teknis (Bagi Pengembang)

Jika Anda ingin mendalami implementasi kodenya:
-   **Engine**: Python 3.14 + PyTorch (Deep Learning).
-   **Workflow**: Jalankan `src/main.py` untuk memulai orkestrasi pipeline lengkap.
-   **Struktur**: Terbagi dalam modul `data_loader` (data), `preprocessing` (HP Filter), dan modul spesifik untuk masing-masing model di dalam folder `src/`.
