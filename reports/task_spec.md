# Task Specification: Proyek Jeda (Terkunci)

> **Status:** Terkunci pada `05_data_selection.ipynb`.

## 1. Spesifikasi 9 Fitur Utama

| Fitur | Formulasi ML | Baseline Pembanding | Metrik Utama | Batas Klaim |
|---|---|---|---|---|
| **Cermin** | Skor deviasi perilaku mingguan terhadap baseline pribadi historis + penyesuaian konteks (hari kerja/akhir pekan, masa ujian, kalender Ramadan). | Median atau rata-rata bergerak historis kausal milik individu (tanpa penyesuaian konteks). | Stabilitas skor pada minggu stabil vs peningkatan skor menjelang episode; cek drift & kausalitas. | Tingkat A / B (pola pribadi, bukan diagnosis). |
| **Kurva Sehat** | Regresi kuantil referensi populasi (persentil fitur ritme aktivitas-istirahat berdasarkan usia & jenis kelamin). | Kuantil empiris per kelompok usia-jenis kelamin vs B-spline / LightGBM quantile regression. | Pinball loss (quantile loss), kalibrasi cakupan persentil, pemisahan kelompok klinis vs kontrol (device-shift check). | Tingkat B / C untuk wrist-actigraphy saja; bukan vonis sehat/sakit. |
| **Jurang Insight** | Skor diskordansi = selisih antara self-report (EMA) aktual dan prediksi EMA dari sinyal pasif + interval ketidakpastian. | Prediksi rata-rata EMA historis kausal subjek. | MAE, RMSE, korelasi/peringkat diskordansi terhadap BPRS PROXY, cakupan interval prediksi. | Dilaporkan sebagai diskordansi (selisih EMA vs sinyal pasif), bukan klaim clinical insight. |
| **Deteksi Pergeseran** | Klasifikasi biner terkalibrasi & deteksi anomali peringatan dini kambuh/pergeseran (horizon 14 hari ke depan) + gerbang ABSTAIN. | Aturan ambang deviasi sederhana & LogisticRegression terkalibrasi sebelum RandomForest, LightGBM, XGBoost, CatBoost. | PR-AUC, ROC-AUC, Brier Score, Expected Calibration Error (ECE), F1/Recall pada laju alarm palsu terkontrol + cakupan ABSTAIN. | Tingkat B jika label klinis CrossCheck tersedia (out-of-subject); Tingkat A (PROXY) jika mode fallback. |
| **Waktu** | Model survival sederhana estimasi peluang/arah waktu menuju pergeseran atau kambuh dalam horizon 2-4 minggu ke depan. | Kaplan-Meier non-parametrik sederhana vs CoxPH / survival model ringan. | Concordance index (C-index) out-of-subject & kalibrasi probabilitas 2-4 minggu. | Tingkat A (PROXY) / Tingkat B tergantung ketersediaan label klinis nyata. |
| **Pemicu** | Identifikasi urutan prediktif kanal sinyal yang paling awal bergeser menjelang episode. | Uji perbandingan terhadap acakan urutan waktu (time-shuffle baseline). | Konsistensi peringkat lead-time kanal & selisih lead-time dibanding acakan waktu. | Urutan prediktif observasional semata; DILARANG klaim sebab-akibat (kausalitas). |
| **Inkonsistensi Antar-Sinyal** | Fitur ketidakcocokan arah pergeseran antar-kanal sinyal pada individu yang sama (diverifikasi lewat ablasi & pruning). | Model Deteksi Pergeseran tanpa fitur Inkonsistensi Antar-Sinyal (uji ablasi langsung). | Delta PR-AUC / Brier / stabilitas lintas fold saat fitur inkonsistensi ditambahkan vs di-prune. | Tingkat A / B; jika fitur ini tidak menambah nilai atau memperburuk metrik, di-prune dan dilaporkan jujur. |
| **Tombol Semangat** | Pemilih pesan dukungan deterministik berbasis aturan transparan dari pustaka CSV terkurasi (otonomi, kompetensi, keterhubungan). | Pemeriksa aturan deterministik + filter daftar kata terlarang + gerbang keamanan Perlu Perhatian / ABSTAIN. | 100% lolos audit kata terlarang, 100% deterministik, 100% pengalihan ke pesan netral aman saat Perlu Perhatian / ABSTAIN. | Tingkat A (simulasi aturan transparan); DILARANG klaim efektivitas klinis pesan motivasi. |
| **Kebijakan Peringatan** | Penetapan ambang keputusan berbeda untuk Individu, Pendamping, dan Klinisi + gerbang ABSTAIN di tingkat episode. | Ambang seragam tunggal tanpa ABSTAIN vs kebijakan 3 peran dengan ABSTAIN. | Evaluasi episode (penggabungan alarm beruntun + pencocokan 1-lawan-1): Intent-to-Screen Recall, False Alarm Rate, Median Lead Time, Abstain Rate. | Tingkat A / B (kebijakan ambang operasional). |

## 2. Hipotesis yang Dapat Gagal

| ID | Hipotesis | Aturan Jika Gagal |
|---|---|---|
| `H1 (Model Kompleks vs Baseline: Deteksi Pergeseran)` | Model tree terkalibrasi (LightGBM/XGBoost/CatBoost) mampu melampaui LogisticRegression atau aturan ambang sederhana pada PR-AUC & Brier Score out-of-subject. | Jika selisih negatif atau tidak stabil antar-fold, pilih model sederhana yang lebih stabil dan laporkan hasilnya apa adanya. |
| `H2 (Nilai Tambah Inkonsistensi Antar-Sinyal)` | Fitur inkonsistensi lintas-kanal menambah sinyal prediktif di atas fitur kanal tunggal pada uji ablasi. | Jika ablasi menunjukkan tidak ada perbaikan atau menurunkan stabilitas kalibrasi, lakukan pruning fitur inkonsistensi dari model utama sambil mempertahankan temuan deskriptifnya. |
| `H3 (Device Shift Actigraphy: Kurva Sehat)` | Posisi persentil ritme aktivitas-istirahat pada kurva referensi NHANES membedakan kelompok klinis vs kontrol pada OBF/Psykose/Depresjon. | Jika pergeseran perangkat (device shift) mendominasi perbedaan klinis, laporkan batas transferabilitas antar-perangkat secara eksplisit dan jangan paksakan ke sinyal ponsel. |
| `H4 (Diskordansi EMA vs Sinyal Pasif: Jurang Insight)` | Model sinyal pasif mampu memprediksi EMA di atas baseline rata-rata EMA historis individu sehingga selisihnya bermakna. | Jika sinyal pasif gagal mengalahkan baseline EMA historis, catat keterbatasan prediktif secara jujur dan hindari klaim berlebihan pada Jurang Insight. |
| `H5 (Urutan Pergeseran Awal: Pemicu)` | Urutan kanal yang paling awal bergeser sebelum episode lebih konsisten daripada acakan waktu (time-shuffle). | Jika urutan tidak berbeda dari acakan waktu, laporkan hasil negatif dan hindari menampilkan klaim pola urutan yang menyesatkan. |

## 3. Disiplin Anti-Leakage
- Seluruh evaluasi model prediktif memakai pembagian per subjek (`GroupKFold` / `StratifiedGroupKFold`).
- Seluruh fitur baseline pribadi dan inkonsistensi lintas-kanal bersifat strictly causal (`t < t_current`).
- Pemilihan hyperparameter `Optuna`, kalibrasi probabilitas, dan pemilihan ambang peran hanya dijalankan di dalam *training fold*.
- Data uji buatan (`SYNTHETIC_SMOKE_TEST_ONLY`) dilarang keras masuk evaluasi atau `metrics_registry.csv`.

## 4. Ringkasan Objek Data Terbekukan (`splits.parquet`)

- **CrossCheck (`PROXY`):** 60 subjek, 5,853 baris total, 434 onset episode pergeseran `PROXY`, 4,234 baris siap latih/uji non-buffer (`target_shift_14d_proxy=1` sebanyak 1107, 26.1%), tingkat `ABSTAIN` data kurang 4.2%.
- **StudentLife (`PROXY`):** 49 subjek, 1,912 baris total, 147 onset episode stres `PROXY`, 1,068 baris siap latih/uji non-buffer (`target_stress_shift_proxy=1` sebanyak 263, 24.6%), tingkat `ABSTAIN` data kurang 8.1%.
- **NHANES 2011-2014 (Referensi Kurva Sehat):** 14,619 subjek unik dalam 5 fold.
- **OBF-Psychiatric (Validasi Kurva Sehat):** 160 subjek unik terdeduplikasi dalam 5 fold.
