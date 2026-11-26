# Review Akhir Kelayakan Model: Proyek Jeda

## 1. Hasil Audit Teknis
| Gerbang_Audit | Status | Bukti_Terukur |
|---|---|---|
| Pemisahan Subjek 5-Fold (Anti-Leakage) | LOLOS | CrossCheck=60 subjek, StudentLife=49 subjek, 0 tumpang tindih |
| Keterurutan Kronologis Fitur Kausal | LOLOS | CrossCheck=5853 baris, StudentLife=1912 baris terurut kronologis |
| Integritas & Reproduktibilitas Model .joblib | LOLOS | 4 artefak .joblib utuh; max_diff inferensi ulang = 0.00e+00 |
| Registri Metrik 9 Fitur Utama Lengkap | LOLOS | 9/9 fitur terdaftar di metrics_registry.csv (11 baris evaluasi) |
| Audit Bahasa Aman & Gerbang Pesan (0 Pelanggaran) | LOLOS | 26 pesan pustaka + 556 baris payload terverifikasi aman |
| Audit Privasi & Anonimisasi Payload Antarmuka | LOLOS | 12 profil tersamar, 0 kolom GPS/teks mentah |

## 2. Ringkasan Metrik 9 Fitur Utama (`reports/metrics_registry.csv`)
| fitur | skenario | model | dataset | n_subjek | n_sampel | metrik_utama | nilai_utama | tingkat_klaim |
|---|---|---|---|---|---|---|---|---|
| Cermin | S1: Baseline Kausal | cermin_shift_7d (dengan konteks) | CrossCheck (PROXY) | 60 | 4194 | ROC-AUC | 0.5179 | Tingkat A (PROXY) |
| Deteksi Pergeseran | S2: 5-Fold Out-of-Subject | LogisticRegression (Calibrated) | CrossCheck (PROXY) | 60 | 4234 | PR-AUC | 0.2717 | Tingkat A (PROXY) |
| Deteksi Pergeseran | S2: Uji Lintas-Studi | LogisticRegression (Cross-Study) | StudentLife (PROXY) | 46 | 1068 | PR-AUC | 0.279 | Tingkat C (Lintas Dataset PROXY) |
| Inkonsistensi Antar-Sinyal | S3: Ablasi vs F_plus_cermin | Ablasi LightGBM/LogReg | CrossCheck (PROXY) | 60 | 4234 | Delta PR-AUC | -0.0018 | Tingkat A (PROXY) |
| Kurva Sehat | S4: Regresi Kuantil Referensi | B-Spline Quantile Regression | NHANES 2011-2014 | 14619 | 14619 | Mean Pinball Loss | 2.2919 | Tingkat B (Wrist-Actigraphy) |
| Kurva Sehat | S4: Validasi Kelompok Klinis | Persentil Kontrol Internal (OBF) | OBF-Psychiatric | 162 | 162 | ROC-AUC (schizophrenia mean_activity) | 0.7727 | Tingkat B / C (Wrist-Actigraphy) |
| Jurang Insight | S5: Diskordansi EMA vs Sinyal Pasif | LightGBM + Interval [q10,q90] | CrossCheck | 60 | 5853 | MAE | 5.1706 | Tingkat B (Diskordansi, bukan clinical insight) |
| Waktu | S6: Survival 2-4 Minggu | CoxPHFitter Regularized | CrossCheck (PROXY) | 60 | 256 | C-index OOF | 0.4832 | Tingkat A (PROXY) |
| Pemicu | S7: Urutan Kanal Awal | Median/Mean Lead-Time vs Time-Shuffle | CrossCheck (PROXY) | 47 | 325 | Selisih Lead-Time (hari) | -0.09 | Tingkat A (Observasional) |
| Tombol Semangat | S8: Pustaka CSV & Aturan Aman | Pemilih Deterministik (Tanpa LLM) | libraries/motivation_messages.csv | 0 | 26 | Lolos Audit Kata Terlarang | 1.0 | Tingkat A (Simulasi Aturan) |
| Kebijakan Peringatan | S8: Kebijakan 3 Peran + Gerbang Jeda | Ambang Utility In-Fold + Gerbang ABSTAIN | CrossCheck (PROXY OOF) | 60 | 4234 | Episode Recall (Individu) | 0.295 | Tingkat A / B (Kebijakan Operasional PROXY) |

## 3. Keputusan & Batasan Jujur Sistem
1. **Model Deteksi Pergeseran terpilih:** `LogisticRegression (Calibrated)` dengan `F_plus_cermin` (`PR-AUC=0.2717`, `Brier=0.1937` di CrossCheck `PROXY`; transfer lintas-studi ke StudentLife `PR-AUC=0.279`). Fitur **Inkonsistensi Antar-Sinyal** dipangkas dari klasifier (`keep_inconsistency=False`, `Delta PR-AUC=-0.0018`) namun tetap ditampilkan sebagai indikator deskriptif kanal.
2. **Temuan negatif yang dipertahankan jujur:** **Jurang Insight** (`MAE=5.1706` vs baseline pribadi `2.0352`) dan **Waktu** (`C-index=0.4832`) memiliki daya prediksi terbatas pada label `PROXY` sehingga tidak diklaim sebagai estimator klinis presisi tinggi.
3. **Kebijakan Peringatan & Keamanan:** Gerbang **`ABSTAIN` (`Jeda`)** dan pembatasan pesan deterministik dari `libraries/motivation_messages.csv` (`26` pesan, `0` kata terlarang) memastikan aplikasi web `app.py` beroperasi aman dan non-diagnostik.
