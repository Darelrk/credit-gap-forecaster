# Brainstorming Hasil Temuan EDA: Credit-to-GDP Gap

EDA (Exploratory Data Analysis) telah selesai dijalankan pada data `credit_private_sector.csv` dan `real_gdp_indonesia.csv`. Berikut adalah ringkasan temuan dan implikasinya terhadap arsitektur projek kita:

## 1. Distribusi & Kelengkapan Data
- **Periode Analisis**: Setelah proses *merging* berdasarkan tanggal, kita mendapatkan **102 observasi** kuartalan (Q1 2000 - Q1 2025).
- **Missing Values**: Tidak ada data yang kosong (`0 missing values`). Ini mempermudah pipeline karena kita tidak perlu melakukan teknik imputasi data.
- **Skala Data**: 
  - Kredit bergerak di kisaran 400 Ribu hingga 9 Juta.
  - GDP bergerak di kisaran 1 Miliar hingga 3.37 Miliar.
  - *Implikasi*: Perbedaan skala yang masif ini menegaskan pentingnya mengubah data absolut menjadi rasio (Credit/GDP) sebelum dimasukkan ke model prediksi manapun.

## 2. Analisis Korelasi
- **Korelasi Kredit vs GDP**: Sangat kuat, di angka **0.9913**. 
- *Implikasi*: Kredit sektor swasta terbukti tumbuh sejalan dengan PDB Indonesia melintasi waktu. Jika tren GDP naik, kredit pasti naik. Oleh karena itu, *Credit-to-GDP Gap* (selisih antara rasio sebenarnya dengan tren jangka panjangnya) menjadi indikator yang sangat masuk akal untuk mendeteksi anomali "over-heating" (pertumbuhan kredit agresif yang tidak diimbangi kapasitas ekonomi rill).

## 3. Stationarity (ADF Test) - *Temuan Paling Kritis*
- **Credit ADF p-value**: 0.998 (Tidak Stasioner)
- **GDP ADF p-value**: 0.998 (Tidak Stasioner)
- *Implikasi*: 
  1. Data mentah memiliki tren naik yang sangat kuat dan varians yang berubah seiring waktu. 
  2. Kita **TIDAK BOLEH** menggunakan data mentah ini langsung ke model model forecasting seperti ARIMA.
  3. Ini memvalidasi rencana arsitektur kita di `Phase 3`: mengubah data menjadi *Gap* menggunakan **HP Filter** (sebuah bentuk detrending) akan menyelesaikan masalah non-stasioneritas ini.
  4. Nantinya di `Phase 4`, untuk ARIMA, parameter differencing (`d`) kemungkinan besar akan bernilai `0` atau maksimal `1` jika kita melakukan forecasting pada data *Gap* (bukan data raw).

## 4. Visualisasi Tren (raw_trends.png)
Plot menunjukkan garis tren eksponensial lambat pada kedua variabel, tanpa adanya lonjakan (*spike*) ekstrem yang acak. Ini menunjukan ekonomi makro Indonesia pasca-krisis 1998 relatif stabil secara tren jangka panjang.

---
### Rekomendasi Langkah Selanjutnya
Berdasarkan temuan di atas, saya merekomendasikan transisi langsung ke **Phase 3 (Feature Engineering)** untuk mengkalkulasi Ratio dan mengaplikasikan HP Filter untuk ekstraksi Gap, lalu melakukan ADF Test lagi khusus pada indikator Gap tersebut.
