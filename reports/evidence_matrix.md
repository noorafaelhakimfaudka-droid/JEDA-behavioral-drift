# Matriks Bukti & Keterlacakan Proyek Jeda

| Tahap | Notebook / Artefak | Ringkasan Bukti & Verifikasi | Status |
|---|---|---|---|
| Objektif Bisnis | `notebooks/01_business_objective.ipynb`, `reports/decision_log.md` | Perumusan masalah perubahan bertahap, matriks biaya 3 peran, gerbang `ABSTAIN` | TERVERIFIKASI |
| Tujuan Teknis | `notebooks/02_technical_objectives.ipynb`, `reports/task_spec.md` | Pemetaan 9 fitur ke target ML, baseline, & metrik terukur | TERVERIFIKASI |
| Telaah Data | `notebooks/03_data_discovery.ipynb`, `reports/data_access_status.json`, `reports/data_cards/` | Audit lisensi & profil 6 dataset lokal + `reports/budget.md` | TERVERIFIKASI |
| Validasi Data | `notebooks/04_data_validation.ipynb` | Bukti identitas hash duplikasi `Depresjon`/`Psykose` di `OBF-Psychiatric` | TERVERIFIKASI |
| Seleksi Data | `notebooks/05_data_selection.ipynb`, `data/interim/splits.parquet` | Pembekuan `task_spec.md`, pemisahan 5-fold bebas kebocoran subjek | TERVERIFIKASI |
| Pembersihan Data | `notebooks/06_data_cleaning.ipynb`, `reports/cleaning_log.md` | Pembersihan kausal per individu, penanda `sensor_off` | TERVERIFIKASI |
| Konstruksi Fitur | `notebooks/07_feature_engineering.ipynb`, `reports/feature_catalog.md` | 5 kanal perilaku + konteks + Cermin + Inkonsistensi; lolos uji kausalitas | TERVERIFIKASI |
| Skenario Model | `notebooks/08_modeling_scenarios.ipynb`, `data/processed/modeling_scenarios.json` | 8 skenario terstruktur lintas seluruh 9 fitur utama | TERVERIFIKASI |
| Pemodelan (Bagian 1-3) | `notebooks/09a..09c_*.ipynb`, `artifacts/models/*.joblib` | Cermin, Deteksi Pergeseran, ablasi Inkonsistensi, Kurva Sehat, Jurang Insight, Waktu, Pemicu, Tombol Semangat | TERVERIFIKASI |
| Evaluasi & Kebijakan | `notebooks/10_model_evaluation.ipynb`, `artifacts/app_payload.parquet`, `reports/evaluation_report.md` | Kalibrasi 3 peran + `ABSTAIN`, evaluasi episode 1-lawan-1, SHAP global/lokal | TERVERIFIKASI |
| Review Akhir | `notebooks/11_model_review.ipynb`, `reports/model_review.md`, `reports/executive_summary.md` | Audit anti-leakage, reproduktibilitas deterministik, & audit bahasa aman | TERVERIFIKASI |
