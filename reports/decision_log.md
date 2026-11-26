# Decision Log: Proyek Jeda

| Tahap | Keputusan | Alasan & Dampak |
|---|---|---|
| Setup Lingkungan | Menggunakan `matplotlib==3.9.1.post1` di `requirements.txt` | `matplotlib==3.9.1` tidak menyediakan wheel biner `cp311-win_amd64` di PyPI sehingga memicu kompilasi sumber C++ pada Windows tanpa MSVC. `3.9.1.post1` adalah rilis wheel resmi pengganti `3.9.1`. |
| Objektif Bisnis | Memisahkan **Kebijakan Peringatan** menjadi 3 peran (`Individu`, `Pendamping`, `Klinisi`) + gerbang `ABSTAIN` | Menyeimbangkan risiko *Missed Shift* dan *Alarm Fatigue* secara transparan tanpa diagnosis klinis. |
| Objektif Bisnis | Mengunci batas non-diagnostik dan non-LLM generatif untuk pesan pengguna | Menjaga keamanan pesan dukungan dan kejujuran pelaporan hasil sistem. |
