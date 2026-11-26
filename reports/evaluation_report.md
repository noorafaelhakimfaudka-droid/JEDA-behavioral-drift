# Laporan Evaluasi Pemodelan & Kebijakan Peringatan: Proyek Jeda

## 1. Ringkasan Evaluasi Episode (1-Lawan-1)

| kebijakan | episode_total | matched_episodes | alarm_episodes | episode_recall | false_alarm_rate_30d | median_lead_days | abstain_rate | row_sensitivity | row_fpr |
|---|---|---|---|---|---|---|---|---|---|
| Baseline Tunggal (Tanpa Jeda) | 322 | 107 | 298 | 0.3323 | 1.3533 | 3.0 | 0.0 | 0.4255 | 0.4854 |
| Klinisi (+ Gerbang Jeda) | 322 | 133 | 405 | 0.413 | 1.9273 | 3.0 | 0.1838 | 0.5565 | 0.5683 |
| Individu (+ Gerbang Jeda) | 322 | 95 | 275 | 0.295 | 1.2754 | 4.0 | 0.1838 | 0.3098 | 0.3406 |
| Pendamping (+ Gerbang Jeda) | 322 | 76 | 229 | 0.236 | 1.0841 | 5.0 | 0.1838 | 0.2186 | 0.1705 |

## 2. 10 Fitur Teratas Penjelasan SHAP Global (`LogisticRegression (Calibrated)`)

| fitur | mean_abs_shap | mean_signed_shap_pos |
|---|---|---|
| ch_suara_convo_num | 0.2656 | 0.0166 |
| ch_suara_convo_dur_log | 0.2174 | 0.0446 |
| ch_mobilitas_dist_log | 0.1177 | 0.046 |
| ch_suara_amp_mean | 0.0997 | 0.0379 |
| z_causal_ch_suara_convo_dur_log | 0.0958 | -0.0167 |
| ch_ritme_sleep_std_7d | 0.0943 | 0.0122 |
| ch_mobilitas_visits | 0.0841 | 0.0387 |
| cermin_shift_7d | 0.0821 | 0.0053 |
| ctx_ramadan_flag | 0.0674 | 0.0025 |
| ch_ritme_sleep_duration | 0.063 | -0.0022 |

## 3. Analisis Kesalahan & Gerbang Jeda (Peran Individu)

| outcome_individu | jumlah_baris | rerata_prob | rerata_cermin | rerata_missing_kanal | rerata_uncertainty | proporsi_akhir_pekan |
|---|---|---|---|---|---|---|
| FN (Terlewat) | 593 | 0.2529 | 0.9601 | 0.0169 | 0.0081 | 0.0084 |
| FP (Alarm Palsu) | 1065 | 0.2856 | 0.9283 | 0.0056 | 0.0124 | 0.0103 |
| Jeda (ABSTAIN) | 778 | 0.2908 | 0.9493 | 0.9563 | 0.0294 | 0.0167 |
| TN (Tenang Akurat) | 1455 | 0.2475 | 0.938 | 0.0069 | 0.009 | 0.0151 |
| TP (Tepat Dini) | 343 | 0.2927 | 0.9351 | 0.0058 | 0.0135 | 0.0058 |

## 4. Keadilan & Stabilitas Antar-Strata Lama Pengamatan

| strata_lama_pengamatan | n_baris | n_subjek | roc_auc | brier | abstain_rate |
|---|---|---|---|---|---|
| Pendek | 1497 | 34 | 0.5139 | 0.2109 | 0.2098 |
| Menengah | 1494 | 16 | 0.555 | 0.2236 | 0.1064 |
| Panjang | 1243 | 10 | 0.5609 | 0.1371 | 0.2454 |

## 5. Catatan Kejujuran Ilmiah
- Seluruh evaluasi pada CrossCheck menggunakan label **PROXY**, sehingga kekuatan diskriminasi episode bersifat eksploratori.
- **Jurang Insight** menghasilkan `MAE = 5.1706` vs `Baseline_MAE = 2.0352`, sehingga di antarmuka disajikan sebagai *diskordansi kualitatif lembut*, bukan estimasi skor klinis.
- **Kurva Sehat** dikalibrasi pada *wrist-actigraphy* (`NHANES`), terpisah dari sensor ponsel.
