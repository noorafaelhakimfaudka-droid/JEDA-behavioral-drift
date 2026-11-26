# Data dan Memory Budget: Proyek Jeda

- **Target RAM Mesin:** 8 GB RAM (CPU-only)
- **Total Ukuran Terunduh di `data/raw/` Saat Ini:** 193.56 MB

## Rincian Ukuran Disk per Dataset (`data/raw/`)

| Dataset | Ukuran di Disk (MB) | Catatan Anggaran & Efisiensi CPU/RAM |
|---|---:|---|
| CrossCheck (cleaned CSV) | 9.34 | CSV 5.853 baris; ringan di RAM (7.64 MB). Berkas mentah Box dicek terpisah bila perlu label kambuh/BPRS. |
| StudentLife (cleaned CSV) | 1.39 | CSV 1.912 baris; sangat ringan di RAM (1.24 MB). |
| NHANES 2011-2014 (CDC DEMO+PAXDAY + PhysioNet meta) | 21.32 | Memakai ringkasan harian/demografi & metadata (14.693 subjek); data menit/80 Hz mentah tidak dimuat sekaligus agar aman di bawah 8 GB RAM. |
| OBF-Psychiatric (ZIP + CSV terekstrak) | 106.74 | 162 subjek pada 5 kelompok; ringan untuk validasi Kurva Sehat di CPU. |
| Psykose (OSF metadata & fitur) | 0.05 | Metadata & fitur tabular ringan; kontrol beririsan dengan Depresjon & OBF-Psychiatric (diaudit pada tahap validasi data). |
| Depresjon (ZIP + CSV terekstrak) | 54.73 | 55 subjek (23 kondisi + 32 kontrol); ringan untuk validasi Kurva Sehat & audit overlap. |
