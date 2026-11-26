# Feature Catalog: Proyek Jeda

Seluruh fitur ponsel bersifat kausal (`t <= t_current` untuk agregasi harian dan `t < t_current` untuk *baseline* pribadi), tanpa koordinat GPS mentah maupun isi teks percakapan.

## 1. Keluarga Fitur Sinyal Pasif (CrossCheck & StudentLife)

- **Kanal Gerak (`ch_gerak_*`):** `ch_gerak_active_ratio`, `ch_gerak_still_ratio`, `ch_gerak_night_ratio`
- **Kanal Ritme & Tidur (`ch_ritme_*`):** `ch_ritme_sleep_duration`, `ch_ritme_sleep_std_7d`, `ch_ritme_light_day_night_ratio`, `ch_ritme_unlock_night_ratio`
- **Kanal Mobilitas Abstrak (`ch_mobilitas_*`):** `ch_mobilitas_dist_log`, `ch_mobilitas_visits`
- **Kanal Komunikasi (`ch_komunikasi_*`):** `ch_komunikasi_total_events`, `ch_komunikasi_miss_ratio`
- **Kanal Suara Sekitar (`ch_suara_*`):** `ch_suara_convo_dur_log`, `ch_suara_convo_num`, `ch_suara_amp_mean`
- **Konteks (`ctx_*`):** `ctx_is_weekend`, `ctx_ramadan_flag`, `ctx_exam_period`
- **Cermin (`z_causal_*`, `cermin_*`):** Deviasi z-score kausal per kanal (`t < t_current`) & indeks komposit `cermin_shift_score`, `cermin_shift_7d`
- **Inkonsistensi Antar-Sinyal (`inconsistency_*`):** `inconsistency_CrossChannel_std`, `inconsistency_ActVsSleep`, `inconsistency_SocialVsMobility`, `inconsistency_NightVsDay`

Total fitur terkonstruksi pada CrossCheck: **28 kolom fitur** (5,853 baris).

## 2. Fitur Wrist-Actigraphy (NHANES & OBF-Psychiatric)

- **NHANES (`nhanes_features.parquet`):** `wake_wear_hours`, `sleep_wear_hours`, `non_wear_hours`, `mims_per_wear_min`, `sleep_wake_ratio`, `RIDAGEYR`, `RIAGENDR`, `age_group_band`.
- **OBF-Psychiatric (`obf_actigraphy_features.parquet`):** `mean_activity`, `std_activity`, `zero_activity_ratio`, `nonwear_60m_ratio`, `intradaily_variability`, `interdaily_stability`, `rest_activity_ratio`.
