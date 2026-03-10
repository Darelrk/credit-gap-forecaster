# Trinitas Prediksi & Quadrilateral Model: Outlook Makroprudensial

Proyek ini mengeksplorasi penggunaan kecerdasan buatan dan statistika lanjut untuk memantau stabilitas keuangan melalui indikator **Credit-to-GDP Gap** Indonesia dengan pendekatan multi-variabel terintegrasi.

---

## 🧭 Metodologi Dasar: Trinitas Prediksi
Sistem ini memproses tiga pilar data makroekonomi utama untuk sinkronisasi kebijakan:
-   **Total Kredit**: Volume penyaluran perbankan (IDR).
-   **PDB Riil**: Indikator pertumbuhan ekonomi riil.
-   **Interest Rate**: Suku bunga sebagai katalis biaya modal dan transmisi kebijakan.

Data diproses menggunakan **Hodrick-Prescott (HP) Filter** (Standar Basel III) untuk memisahkan tren fundamental dari gap siklikal.

---

## 🧠 Mesin Peramalan: Quadrilateral Prediction System
Sistem mengintegrasikan empat model (Quadrilateral) untuk memberikan pandangan komprehensif:

![Quadrilateral Evaluation](results/comparative_evaluation_plot.png)

1. **LSTM (Attention-based Deep Learning)**: Menangkap memori jangka panjang dan dinamika volatilitas.
2. **XGBoost (Non-Linear Pattern)**: Presisi tinggi dalam mendeteksi hubungan antar-variabel makro.
3. **ARIMA (Statistical Baseline)**: Jangkar statistik linear untuk tren jangka menengah.
4. **Ensemble (Hybrid Engine)**: Menggabungkan kekuatan LSTM dan XGBoost untuk hasil peramalan yang paling reliabel.

---

## 📊 Dashboard Early Warning Indicator (EWI)
Output sistem disajikan dalam dashboard yang mendukung:

![Outlook 2026 Fan Chart](results/outlook_2026_fan_chart.png)

-   **Monitoring Transisi**: Mendeteksi kecepatan pemulihan dari zona *deep under-trend*.
-   **Sinyal Kebijakan**: Indikator lampu lalu lintas untuk mendeteksi penumpukan risiko sistemik atau ruang ekspansi kredit.

---

## 🖥️ Ringkasan Teknis
-   **Engine**: Python 3.14 + PyTorch.
-   **Workflow**: Pipeline orkestrasi melalui `src/main.py`.
-   **Visuals**: Aset resolusi tinggi (300 DPI) di direktori `results/`.

