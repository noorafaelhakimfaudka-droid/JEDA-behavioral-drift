# Jeda: Pemantauan Pergeseran Pola Perilaku Harian

Jeda adalah proyek *applied machine learning* untuk mengamati apakah pola perilaku harian seseorang mulai menjauh dari kebiasaannya sendiri. Sinyal yang dipakai adalah sinyal pasif ponsel (tidur, gerak, mobilitas, komunikasi, suara sekitar) dan data *wrist-actigraphy*. Seluruh pipeline dikerjakan di Jupyter Notebook pada CPU dengan RAM 8 GB, lalu disajikan lewat aplikasi Streamlit yang bersifat non-diagnostik.

Proyek ini berstatus **portofolio akademik**. Hasilnya bersifat eksploratori dan tidak boleh dipakai untuk keputusan klinis.

---

## 1. Latar Belakang Masalah

Pada gangguan jiwa berat, perburukan kondisi jarang datang tiba-tiba. Biasanya didahului perubahan kecil pada rutinitas harian yang berlangsung berhari-hari sampai berminggu-minggu:

- durasi dan keteraturan tidur menjadi tidak menentu,
- jangkauan bepergian dan variasi tempat yang dikunjungi menyempit,
- frekuensi komunikasi dan percakapan menurun,
- catatan perasaan harian tidak lagi sejalan dengan aktivitas yang terlihat.

Karena perubahannya bertahap, orang yang mengalaminya sering tidak menyadari, dan orang di sekitarnya baru menyadari setelah gejala jelas. Kontrol berkala ke klinisi biasanya berjarak beberapa minggu, sehingga perubahan di antara dua kunjungan tidak terpantau.

## 2. Rumusan Masalah

1. Dapatkah perubahan perilaku harian diukur terhadap **kebiasaan pribadi** (bukan terhadap angka normal umum) tanpa memakai informasi dari masa depan?
2. Seberapa jauh sinyal pasif ponsel dapat memprediksi pergeseran kondisi dalam horizon 14 hari, diuji pada **subjek yang belum pernah dilihat model**?
3. Bagaimana sistem tetap jujur ketika data tidak lengkap atau prediksi ragu, tanpa memaksakan keluaran?
4. Bagaimana hasil ditampilkan agar bermanfaat bagi pengguna, pendamping, dan klinisi, tanpa angka statistik yang membingungkan dan tanpa vonis diagnosis?

## 3. Gagasan dan Alasan Proyek Ini Dibuat

Jeda berangkat dari dua pengamatan:

- **Jarak antara pemantauan dan kenyataan harian.** Yang tersedia selama ini adalah kunjungan berkala dan catatan subjektif. Sinyal pasif dari ponsel yang sudah ada di saku pengguna dapat mengisi jarak tersebut tanpa membebani pengguna dengan alat tambahan.
- **Sistem peringatan sering gagal karena terlalu banyak alarm.** Alarm palsu membuat pengguna dan pendamping lelah (*alarm fatigue*), sedangkan alarm yang terlambat membuat pergeseran awal terlewat. Karena itu Jeda memakai ambang berbeda untuk tiga peran dan sengaja menahan keluaran (`ABSTAIN`, ditampilkan sebagai status **Jeda**) saat data kurang.

Tidak ada dataset terbuka yang memuat seluruh sinyal sekaligus pada pasien yang sama, sehingga proyek ini menggabungkan beberapa sumber dengan peran yang berbeda dan menyatakan batas klaim tiap sumber secara eksplisit.

## 4. Tujuan

- Membangun pipeline *machine learning* yang bebas kebocoran data (*leakage*) dan dapat dijalankan ulang di laptop biasa.
- Menguji sembilan fitur analitik terhadap pembanding sederhana (*baseline*), dan melaporkan hasil negatif apa adanya.
- Menyediakan aplikasi yang membaca artefak hasil notebook (`.joblib`, `.parquet`, `.csv`) sehingga angka yang tampil dapat ditelusuri ke komputasi aslinya.

## 5. Batasan Mutlak Sistem

- **Bukan alat diagnosis.** Status `Stabil`, `Mulai Bergeser`, `Perlu Perhatian`, dan `Jeda` hanya menggambarkan pergeseran pola terhadap kebiasaan sendiri.
- **Tanpa model bahasa generatif.** Seluruh pesan dukungan berasal dari pustaka 26 kalimat di `libraries/motivation_messages.csv`. Saat status `Perlu Perhatian` atau `Jeda`, pesan otomatis berganti ke pesan netral yang menyarankan menghubungi pendamping tepercaya atau tenaga profesional.
- **Label bersifat proksi (`PROXY`).** Berkas CrossCheck yang tersedia tidak memuat kolom kekambuhan atau skor klinis. Peristiwa pergeseran didefinisikan dari gejala yang dilaporkan sendiri pada *Ecological Momentary Assessment* (EMA), sehingga hasilnya tidak dapat disebut prediksi kekambuhan klinis.
- **Tidak ada klaim sebab-akibat** dan tidak ada klaim kinerja pada populasi Indonesia.

---

## 6. Pendekatan

```text
Data mentah (6 dataset)
   -> validasi & audit duplikasi subjek
   -> seleksi data, label proksi, pembagian 5 lipatan per subjek
   -> pembersihan kausal per individu
   -> fitur 5 kanal perilaku + konteks + baseline pribadi
   -> skenario terdaftar sebelum pelatihan
   -> pelatihan, kalibrasi, ablasi
   -> evaluasi episode & kebijakan tiga peran
   -> audit akhir -> artefak untuk aplikasi
```

Prinsip yang dijaga di seluruh notebook:

- **Baseline kausal.** Kebiasaan pribadi pada hari `t` dihitung hanya dari data sebelum `t`.
- **Pemisahan per subjek.** Validasi silang 5 lipatan memakai `StratifiedGroupKFold`, sehingga satu orang tidak muncul di data latih dan data uji sekaligus.
- **Penyetelan di dalam lipatan latih.** Pencarian hiperparameter (`Optuna`), kalibrasi probabilitas, dan pemilihan ambang tidak menyentuh data uji.
- **Skenario didaftarkan sebelum pelatihan** (notebook 08) agar hasil tidak dipilih setelah melihat angka.

## 7. Data

| Dataset | Peran | Catatan |
|---|---|---|
| CrossCheck | Sumber utama sinyal ponsel dan EMA (60 subjek, 5.853 baris) | Label pergeseran diturunkan sebagai proksi dari EMA |
| StudentLife | Konteks masa ujian dan uji lintas studi (49 subjek, 1.912 baris) | Kohort mahasiswa non-klinis |
| NHANES 2011-2014 | Referensi populasi untuk Kurva Sehat (14.619 subjek) | Khusus *wrist-actigraphy* |
| OBF-Psychiatric | Validasi Kurva Sehat pada kelompok klinis dan kontrol (162 subjek unik) | Mencakup subjek yang sama dengan Psykose dan Depresjon |
| Psykose, Depresjon | Metadata dan pembanding kelompok klinis | Subjeknya identik dengan sebagian OBF-Psychiatric (dibuktikan lewat *hash* seri waktu), sehingga tidak dihitung ganda |

Data mentah tidak disertakan di repositori. Lisensi dan sitasi tiap dataset ada di `reports/data_cards/`. Psykose dan Depresjon berlisensi non-komersial, dan seluruh proyek ini hanya untuk keperluan akademik. Koordinat GPS mentah dan isi percakapan tidak disimpan maupun ditampilkan.

## 8. Sembilan Fitur dan Hasil Evaluasi

Angka di bawah dibaca langsung dari `reports/metrics_registry.csv`.

| Fitur | Pendekatan | Hasil | Batas klaim |
|---|---|---|---|
| **Cermin** | Skor jarak terhadap kebiasaan pribadi 7 hari + konteks | `ROC-AUC = 0.5179`, `PR-AUC = 0.2769` | Tingkat A (`PROXY`) |
| **Deteksi Pergeseran** | `LogisticRegression` terkalibrasi, 14 hari ke depan | `PR-AUC = 0.2717`, `ROC-AUC = 0.5242`, `Brier = 0.1937`; lintas studi (StudentLife) `PR-AUC = 0.279` | Tingkat A / C (`PROXY`) |
| **Inkonsistensi Antar-Sinyal** | Uji ablasi lintas-kanal | `Delta PR-AUC = -0.0018`, fitur dipangkas dari model | Tingkat A (`PROXY`) |
| **Kurva Sehat** | *B-Spline Quantile Regression* pada NHANES | `Pinball Loss = 2.2919`; validasi OBF `ROC-AUC = 0.7727` (skizofrenia) | Tingkat B / C (*wrist-actigraphy*) |
| **Jurang Insight** | `LightGBM` + interval `[q10, q90]` untuk skor EMA | `MAE = 5.1706` vs *baseline* `2.0352` | Tingkat B (diskordansi) |
| **Waktu** | `CoxPHFitter` terregulasi, horizon 2-4 minggu | `C-index OOF = 0.4832` vs Kaplan-Meier `0.4795` | Tingkat A (`PROXY`) |
| **Pemicu** | *Lead-time* kanal dibanding acakan waktu (200 permutasi) | Selisih `-0.09` hari (kanal awal: Ritme dan Tidur) | Tingkat A (observasional) |
| **Tombol Semangat** | Pemilih aturan deterministik dari 26 pesan | Lolos audit kata terlarang dan gerbang aman | Tingkat A (aturan) |
| **Kebijakan Peringatan** | Tiga peran + gerbang `ABSTAIN`, evaluasi episode 1-lawan-1 | Lihat tabel di bawah | Tingkat A / B (`PROXY`) |

Evaluasi episode pada 322 episode (`reports/evaluation_report.md`):

| Kebijakan | Episode recall | Tingkat abstain |
|---|---|---|
| Baseline tunggal (tanpa gerbang jeda) | 0.3323 | 0 |
| Klinisi | 0.413 | 0.1838 |
| Individu | 0.295 | 0.1838 |
| Pendamping | 0.236 | 0.1838 |

### Bacaan jujur atas hasil

- Daya prediksi model **lemah**. Laju positif pada data uji sekitar `0.26`, sedangkan `PR-AUC` Deteksi Pergeseran `0.2717` dan `ROC-AUC` `0.5242`, sehingga hanya sedikit di atas tebakan acak.
- **Jurang Insight** kalah dari *baseline* sederhana (rata-rata EMA historis individu) dan **Waktu** menghasilkan `C-index` di bawah `0.5`. Keduanya tetap ditampilkan di aplikasi sebagai informasi kualitatif, bukan estimasi.
- **Pemicu** hanya selisih `-0.09` hari terhadap acakan waktu, sehingga urutan kanalnya tidak boleh ditafsirkan sebagai pola yang pasti.
- Hasil yang paling menjanjikan adalah validasi **Kurva Sehat** pada OBF (`ROC-AUC = 0.7727`), dan itu pun terbatas pada *wrist-actigraphy*, bukan sensor ponsel.
- Seluruh evaluasi memakai label proksi, sehingga angka ini menggambarkan kemampuan mendeteksi pergeseran gejala yang dilaporkan sendiri, bukan kekambuhan klinis.

Rincian keputusan dan batasan ada di `reports/model_review.md` dan `reports/decision_log.md`.

---

## 9. Aplikasi Streamlit

Aplikasi membaca empat model `.joblib` dan memberi hasil dari catatan yang dimasukkan pengguna.

- **Input:** Check-in Kilat (pilihan kartu), Rekap 7 Hari, impor CSV/JSON (templat di `libraries/jeda_user_template.csv`), dan Contoh Riset (12 profil teranonimkan, 556 baris mingguan *out-of-fold*). Rentang tanggal 2026 sampai 2028.
- **Tiga peran tampilan:** `Individu`, `Pendamping`, `Klinisi`, masing-masing dengan ambang berbeda (`helpers.py`).
- **Mode Data Scientist:** menampilkan nilai `SHAP` lokal dan tabel `metrics_registry.csv`. Mode ini tertutup secara bawaan.
- **Privasi:** catatan diproses oleh proses Python lokal dan disimpan di penyimpanan peramban (`localStorage`). Tidak ada pengiriman data ke layanan luar.
- Dengan kurang dari 3 hari catatan, aplikasi menampilkan status **Jeda** dan tidak menghasilkan pembacaan apa pun.

## 10. Struktur Repositori

```text
JEDA/
├── app.py                          # Aplikasi Streamlit
├── helpers.py                      # Inferensi lokal dari 4 model .joblib dan SHAP
├── config.json                     # Parameter proyek, seed 42, palet warna
├── requirements.txt                # Dependensi Python 3.11 (CPU)
├── .streamlit/config.toml          # Tema Streamlit
├── notebooks/                      # Notebook 01 sampai 11 (alur kerja lengkap)
├── artifacts/
│   ├── ui_component/               # index.html dan data_bundle.js (antarmuka)
│   ├── models/                     # 4 model .joblib
│   ├── figures/                    # Grafik dari notebook
│   └── app_payload.parquet         # Hasil out-of-fold 12 profil teranonimkan
├── libraries/                      # Pustaka pesan dan templat impor
├── reports/                        # Catatan keputusan, spesifikasi, metrik, review akhir, data card
└── data/                           # raw (tidak dilacak), interim, processed (dibuat ulang notebook)
```

## 11. Cara Menjalankan

**Hanya aplikasi** (model dan artefak sudah ada di repositori):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

**Mereproduksi pipeline:** letakkan dataset di `data/raw/` sesuai nama folder yang dibaca notebook (`crosscheck`, `studentlife`, `nhanes`, `obf_psychiatric`, `depresjon`, `psykose`), lalu jalankan notebook `01` sampai `11` berurutan. Notebook membuat ulang `data/interim`, `data/processed`, model, grafik, dan laporan. Seed yang dipakai: `42`.

## 12. Keterbatasan

- Label proksi dari EMA, bukan kekambuhan klinis.
- Sampel kecil (60 subjek CrossCheck) dan populasi non-Indonesia.
- Kinerja prediksi mendekati tebakan acak, sehingga belum layak untuk penggunaan nyata.
- Kurva Sehat berlaku untuk *wrist-actigraphy*, tidak untuk sensor ponsel.
- Aplikasi menerima catatan manual atau impor pengguna; skala fitur diturunkan dari data CrossCheck sehingga input yang berbeda jauh dari data tersebut dapat memberi pembacaan yang kurang andal.
- Pembacaan tidak pernah divalidasi pada pengguna sebenarnya.

## 13. Sitasi dan Lisensi Data

- StudentLife: Wang et al., UbiComp 2014 (CC-BY-4.0).
- Depresjon: Garcia-Ceja et al., MMSys 2018 (CC-BY-NC-4.0).
- Psykose: Jakobsen et al., IEEE CBMS 2020 (CC BY-NC 4.0).
- CrossCheck (Cornell Tech, Precision Behavioral Health Initiative), NHANES (CDC/NCHS), dan OBF-Psychiatric (Simula): lihat `reports/data_cards/`.
