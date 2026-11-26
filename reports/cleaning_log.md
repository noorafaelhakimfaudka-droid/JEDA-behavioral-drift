# Cleaning Log: Proyek Jeda

Seluruh langkah pembersihan dijalankan secara kausal di dalam masing-masing individu tanpa imputasi lintas subjek dan tanpa melihat masa depan.

## Ringkasan Pembersihan Sinyal Ponsel

| Dataset | Baris | Subjek | Tidur <0 Sebelum | Tidur <0 Sesudah | Missing Sensor Sebelum (%) | Missing Sensor Sesudah (%) | Baris Sensor Off |
|---|---:|---:|---:|---:|---:|---:|---:|
| CrossCheck | 5,853 | 60 | 0 | 0 | 0.24% | 0.07% | 342 |
| StudentLife | 1,912 | 49 | 18 | 0 | 1.73% | 0.85% | 0 |

## Ringkasan Pembersihan Wrist-Actigraphy

- **NHANES 2011-2014:** Menyaring `130,186` hari mentah menjadi `105,870` hari valid (`PAXQFD == 0` dan `PAXVMD >= 600` menit/hari) pada `14,619` subjek unik dengan usia & jenis kelamin.
- **OBF-Psychiatric:** Memproses `162` subjek unik terdeduplikasi lintas 5 kelompok (`adhd`, `clinical`, `control`, `depression`, `schizophrenia`), termasuk deteksi blok *non-wear* >= 60 menit nol berturut-turut dan metrik ritme sirkadian (IV, IS, *rest-activity ratio*).
