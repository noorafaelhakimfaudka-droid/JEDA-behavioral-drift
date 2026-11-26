# Ringkasan Eksekutif: Proyek Jeda

**Jeda** adalah sistem peringatan dini berbasis Applied Machine Learning untuk mengamati pergeseran pola perilaku harian dari sinyal pasif ponsel dan actigraphy secara tenang, kausal, dan non-diagnostik.

- **Cakupan Dataset Nyata (CPU-only):** CrossCheck (60 subjek `PROXY`, 5.853 baris), StudentLife (46 subjek `PROXY`, 1.912 baris), NHANES 2011-2014 (14.619 subjek *wrist-actigraphy*), serta OBF-Psychiatric (162 subjek tervalidasi bebas duplikasi dengan Depresjon/Psykose).
- **Arsitektur 9 Fitur Terpadu:** Cermin, Kurva Sehat, Jurang Insight, Deteksi Pergeseran, Waktu, Pemicu, Inkonsistensi Antar-Sinyal, Tombol Semangat, dan Kebijakan Peringatan tiga peran (`Individu`, `Pendamping`, `Klinisi`) dengan gerbang **`ABSTAIN` (`Jeda`)**.
- **Kesiapan Antarmuka:** Seluruh keluaran out-of-sample untuk 12 profil tersamar beserta penjelasan SHAP lokal telah dibungkus ke `artifacts/app_payload.parquet` dan lulus seluruh audit kebersihan istilah, privasi, serta anti-leakage, siap disajikan oleh aplikasi Streamlit `app.py`.
