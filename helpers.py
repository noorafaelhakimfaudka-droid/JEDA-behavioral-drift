import functools
import json
import pathlib
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = pathlib.Path(__file__).resolve().parent

FRIENDLY_FEAT_NAMES = {
    "cermin_shift_7d": "Jarak Ritme Mingguan",
    "cermin_shift_score": "Skor Perubahan Pribadi",
    "ch_gerak_active_ratio": "Proporsi Waktu Aktif",
    "ch_gerak_night_ratio": "Aktivitas Malam Hari",
    "ch_gerak_still_ratio": "Proporsi Waktu Diam",
    "ch_komunikasi_miss_ratio": "Panggilan Tidak Terjawab",
    "ch_komunikasi_total_events": "Total Interaksi Ponsel",
    "ch_mobilitas_dist_log": "Jangkauan Perjalanan",
    "ch_mobilitas_visits": "Variasi Tempat Dikunjungi",
    "ch_ritme_light_day_night_ratio": "Keseimbangan Cahaya Siang-Malam",
    "ch_ritme_sleep_duration": "Durasi Istirahat Harian",
    "ch_ritme_sleep_std_7d": "Keteraturan Jam Tidur",
    "ch_ritme_unlock_night_ratio": "Membuka Layar Tengah Malam",
    "ch_suara_amp_mean": "Suasana Suara Sekitar",
    "ch_suara_convo_dur_log": "Durasi Percakapan Sekitar",
    "ch_suara_convo_num": "Frekuensi Interaksi Suara",
    "ctx_exam_period": "Konteks Masa Padat",
    "ctx_is_weekend": "Konteks Akhir Pekan",
    "ctx_ramadan_flag": "Konteks Ramadan",
    "missing_channel_count": "Kelengkapan Sinyal",
    "sensor_off_flag": "Jeda Sensor",
    "z_causal_ch_gerak_active_ratio": "Perubahan Aktivitas Fisik",
    "z_causal_ch_komunikasi_total_events": "Perubahan Komunikasi",
    "z_causal_ch_mobilitas_dist_log": "Perubahan Mobilitas",
    "z_causal_ch_ritme_sleep_duration": "Perubahan Durasi Istirahat",
    "z_causal_ch_suara_convo_dur_log": "Perubahan Suasana Suara",
}

REF_MEDIANS = {
    "ch_gerak_active_ratio": 0.0115,
    "ch_gerak_night_ratio": 0.0,
    "ch_gerak_still_ratio": 0.9885,
    "ch_komunikasi_miss_ratio": 0.0,
    "ch_komunikasi_total_events": 8.3333,
    "ch_mobilitas_dist_log": 9.0567,
    "ch_mobilitas_visits": 1.6667,
    "ch_ritme_light_day_night_ratio": 4.0193,
    "ch_ritme_sleep_duration": 8.5,
    "ch_ritme_sleep_std_7d": 1.8112,
    "ch_ritme_unlock_night_ratio": 0.0278,
    "ch_suara_amp_mean": 1260.8378,
    "ch_suara_convo_dur_log": 9.2229,
    "ch_suara_convo_num": 15.3333,
}

ROLE_THRESHOLDS = {
    "Klinisi": (0.240, 0.275),
    "Individu": (0.262, 0.295),
    "Pendamping": (0.288, 0.315),
}
TAU_UNC = 0.0292


@functools.lru_cache(maxsize=1)
def load_static_bundle() -> dict[str, Any]:
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    df_pay = pd.read_parquet(ROOT / "artifacts/app_payload.parquet")
    df_msg = pd.read_csv(ROOT / "libraries/motivation_messages.csv")
    df_reg = pd.read_csv(ROOT / "reports/metrics_registry.csv")
    tpl = (ROOT / "libraries/jeda_user_template.csv").read_text(encoding="utf-8")

    rows = []
    for _, r in df_pay.iterrows():
        item = {}
        for k, v in r.items():
            if hasattr(v, "item"):
                v = v.item()
            item[k] = v
        item["cermin_wave_current"] = json.loads(item.pop("cermin_wave_current_json"))
        item["shap_top_features"] = json.loads(item.pop("shap_top_features_json"))
        rows.append(item)

    profiles = sorted(df_pay["profile_id"].unique().tolist())
    return {
        "config": cfg,
        "profiles": profiles,
        "rows": rows,
        "messages": df_msg.to_dict(orient="records"),
        "metrics_registry": df_reg.to_dict(orient="records"),
        "user_csv_template": tpl,
    }


@functools.lru_cache(maxsize=1)
def load_models() -> dict[str, Any]:
    return {
        "shift": joblib.load(ROOT / "artifacts/models/shift_detector.joblib"),
        "kurva": joblib.load(ROOT / "artifacts/models/kurva_sehat_quantiles.joblib"),
        "jurang": joblib.load(ROOT / "artifacts/models/jurang_insight_model.joblib"),
        "waktu": joblib.load(ROOT / "artifacts/models/waktu_survival.joblib"),
    }


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _role_status(prob: float, abstain: int, th_low: float, th_high: float) -> str:
    if abstain == 1:
        return "Jeda"
    if prob >= th_high:
        return "Perlu Perhatian"
    if prob >= th_low:
        return "Mulai Bergeser"
    return "Stabil"


def _channel_state(z_val: float | None, abstain: int) -> str:
    if abstain == 1 or z_val is None or np.isnan(z_val):
        return "Belum ada data"
    az = abs(float(z_val))
    if az >= 1.4:
        return "Bergerak Berbeda"
    if az >= 0.8:
        return "Sedikit Bergeser"
    return "Selaras"


def score_user_timeline(entries: list[dict[str, Any]], profile_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Skor riwayat catatan pengguna secara kausal memakai 4 model .joblib & SHAP."""
    models = load_models()
    shift_b = models["shift"]
    kurva_b = models["kurva"]
    jurang_b = models["jurang"]
    waktu_b = models["waktu"]

    if not entries:
        return {
            "src": "local",
            "n": 0,
            "abstain": 1,
            "sensorOff": 0,
            "missing": 4,
            "ctx": "Hari Biasa",
            "p": None,
            "unc": None,
            "p14": None,
            "p28": None,
            "status": {r: "Jeda" for r in ("Individu", "Pendamping", "Klinisi")},
            "pts": [],
            "labels": [],
            "base": 1.1,
            "cerSum": "Belum ada catatan tersimpan untuk dibandingkan dengan kebiasaan pribadi.",
            "kurva": {
                "pos": None,
                "label": "Belum dapat ditempatkan",
                "sum": "Pita referensi memerlukan catatan istirahat atau aktivitas harian.",
            },
            "selfL": "Belum ada catatan perasaan",
            "passL": "Pola belum cukup terbaca",
            "disc": "Jeda Catatan",
            "jur": "Tambahkan catatan harian agar perbandingan dua sisi dapat dihitung.",
            "waktu": "Data belum cukup untuk membaca kecenderungan 2-4 minggu ke depan.",
            "waktuAbs": "Data belum cukup untuk membaca kecenderungan 2-4 minggu ke depan.",
            "pem": [],
            "chan": [
                ["Gerak", "Belum ada data"],
                ["Ritme & Tidur", "Belum ada data"],
                ["Mobilitas", "Belum ada data"],
                ["Komunikasi & Suara", "Belum ada data"],
            ],
            "incon": "Keselarasan antar-kanal dihitung setelah catatan harian terkumpul.",
            "seed": 0,
            "shap": [],
            "shapNote": "Menunggu minimal 3 hari catatan untuk menjalankan model .joblib.",
            "engine": "joblib_ready",
        }

    df = pd.DataFrame(entries).copy()
    df["tanggal_dt"] = pd.to_datetime(df["tanggal"], errors="coerce")
    df = df.dropna(subset=["tanggal_dt"]).sort_values("tanggal_dt").reset_index(drop=True)
    n = len(df)
    last_raw = df.iloc[-1]

    # Hitung kanal perilaku sesuai skala CrossCheck
    sleep_dur = df.get("durasi_tidur_jam", pd.Series([np.nan] * n)).apply(_safe_float).astype(float)
    unlock_night = df.get("buka_layar_malam", pd.Series([np.nan] * n)).apply(_safe_float).astype(float)
    active_h = df.get("durasi_aktif_jam", pd.Series([np.nan] * n)).apply(_safe_float).astype(float)
    night_act = df.get("aktivitas_malam", pd.Series([np.nan] * n)).apply(_safe_float).astype(float)
    mob_visits = df.get("mobilitas_tempat", pd.Series([np.nan] * n)).apply(_safe_float).astype(float)
    mob_km = df.get("mobilitas_jarak_km", pd.Series([np.nan] * n)).apply(_safe_float).astype(float)
    comm_ev = df.get("interaksi_komunikasi", pd.Series([np.nan] * n)).apply(_safe_float).astype(float)
    miss_call = df.get("panggilan_terlewat", pd.Series([np.nan] * n)).apply(_safe_float).astype(float)
    suara_min = df.get("interaksi_suara_menit", pd.Series([np.nan] * n)).apply(_safe_float).astype(float)
    ema_vals = df.get("refleksi_mandiri", pd.Series([np.nan] * n)).apply(_safe_float).astype(float)

    # Turunkan metrik pelengkap secara kausal saat input ringkas 4-kanal dipakai
    mob_km = mob_km.where(mob_km.notna(), mob_visits * 1.6)
    suara_min = suara_min.where(suara_min.notna(), comm_ev * 2.5)
    unlock_night = unlock_night.where(
        unlock_night.notna(),
        np.where(sleep_dur.notna(), np.clip((8.0 - sleep_dur) * 1.8 + 4.0, 1.0, 20.0), np.nan),
    )
    night_act = night_act.where(
        night_act.notna(),
        np.where(active_h.notna(), np.clip(0.18 - active_h * 0.03, 0.02, 0.25), np.nan),
    )

    feat = pd.DataFrame(index=df.index)
    feat["ch_gerak_active_ratio"] = np.clip(active_h / 120.0, 0.0, 0.25)
    feat["ch_gerak_night_ratio"] = np.clip(night_act * 0.25, 0.0, 0.25)
    feat["ch_gerak_still_ratio"] = 1.0 - feat["ch_gerak_active_ratio"]
    feat["ch_ritme_sleep_duration"] = np.clip(sleep_dur, 0.0, 16.0)
    feat["ch_ritme_sleep_std_7d"] = feat["ch_ritme_sleep_duration"].shift(1).rolling(7, min_periods=2).std().fillna(1.2)
    feat["ch_ritme_unlock_night_ratio"] = np.clip(unlock_night / 35.0, 0.0, 0.5)
    feat["ch_ritme_light_day_night_ratio"] = REF_MEDIANS["ch_ritme_light_day_night_ratio"]
    feat["ch_mobilitas_dist_log"] = np.log1p(np.clip(mob_km * 1000.0, 0.0, 100000.0))
    feat["ch_mobilitas_visits"] = np.clip(mob_visits * 0.65, 0.0, 6.0)
    feat["ch_komunikasi_total_events"] = np.clip(comm_ev * 1.1, 0.0, 100.0)
    feat["ch_komunikasi_miss_ratio"] = np.where(comm_ev.notna(), np.clip(miss_call.fillna(0.0) * 0.0001, 0.0, 0.0015), np.nan)
    feat["ch_suara_convo_dur_log"] = np.log1p(np.clip(suara_min * 150.0, 0.0, 80000.0))
    feat["ch_suara_convo_num"] = np.clip(comm_ev * 1.0, 0.0, 80.0)
    feat["ch_suara_amp_mean"] = REF_MEDIANS["ch_suara_amp_mean"]

    ctx_series = df.get("konteks", pd.Series(["Hari Biasa"] * n)).fillna("Hari Biasa").astype(str)
    feat["ctx_is_weekend"] = ctx_series.str.contains("Pekan", case=False).astype(int)
    feat["ctx_exam_period"] = ctx_series.str.contains("Padat|Ujian", case=False).astype(int)
    feat["ctx_ramadan_flag"] = ctx_series.str.contains("Ramadan", case=False).astype(int)

    # Cek kelengkapan per kanal pada baris terakhir
    ch_missing_flags = {
        "gerak": pd.isna(feat["ch_gerak_active_ratio"].iloc[-1]),
        "ritme": pd.isna(feat["ch_ritme_sleep_duration"].iloc[-1]),
        "mobilitas": pd.isna(feat["ch_mobilitas_dist_log"].iloc[-1]) and pd.isna(feat["ch_mobilitas_visits"].iloc[-1]),
        "komunikasi": pd.isna(feat["ch_komunikasi_total_events"].iloc[-1]),
        "suara": pd.isna(feat["ch_suara_convo_dur_log"].iloc[-1]),
    }
    missing_cnt = int(sum(ch_missing_flags.values()))
    sensor_off_flag = int(missing_cnt >= 3)
    feat["missing_channel_count"] = missing_cnt
    feat["sensor_off_flag"] = sensor_off_flag

    # Hitung z-score kausal murni masa lalu (t < t_current)
    causal_specs = [
        ("ch_gerak_active_ratio", 0.008),
        ("ch_ritme_sleep_duration", 0.8),
        ("ch_mobilitas_dist_log", 1.0),
        ("ch_komunikasi_total_events", 3.5),
        ("ch_suara_convo_dur_log", 0.9),
    ]
    z_cols = []
    for col, min_sd in causal_specs:
        hist_mean = feat[col].shift(1).rolling(14, min_periods=2).mean()
        hist_sd = feat[col].shift(1).rolling(14, min_periods=2).std().clip(lower=min_sd).fillna(min_sd)
        z_name = f"z_causal_{col}"
        feat[z_name] = np.clip((feat[col] - hist_mean) / hist_sd, -4.0, 4.0)
        z_cols.append(z_name)

    feat["cermin_shift_score"] = feat[z_cols].abs().mean(axis=1)
    feat["cermin_shift_7d"] = feat["cermin_shift_score"].rolling(7, min_periods=1).mean()
    inconsistency_std = feat[z_cols].std(axis=1).fillna(0.0)

    # Inferensi dengan shift_detector.joblib
    model_cols = shift_b["features"]
    X_row = feat.tail(1).reindex(columns=model_cols)
    prob_lr = float(shift_b["lr_cal"].predict_proba(X_row)[0, 1])
    prob_lgb = float(shift_b["lgb_cal"].predict_proba(X_row)[0, 1])
    unc_std = float(np.std([prob_lr, prob_lgb]))

    abstain = int((n < 3) or (missing_cnt >= 3) or (sensor_off_flag == 1) or (unc_std > TAU_UNC))
    statuses = {
        role: _role_status(prob_lr, abstain, th_lo, th_hi)
        for role, (th_lo, th_hi) in ROLE_THRESHOLDS.items()
    }

    # SHAP lokal asli dari pipeline calibrated LogisticRegression
    shap_items = []
    cal_list = shift_b["lr_cal"].calibrated_classifiers_
    shap_vec = np.zeros(len(model_cols), dtype=float)
    X_eval = X_row[model_cols]
    for cal_item in cal_list:
        pipe = cal_item.estimator
        x_imp = pipe.named_steps["imp"].transform(X_eval)
        x_sc = pipe.named_steps["sc"].transform(x_imp)
        clf = pipe.named_steps["clf"]
        expl = shap.LinearExplainer(clf, np.zeros((1, len(model_cols))))
        vals = expl.shap_values(x_sc)
        if isinstance(vals, list):
            vals = vals[1]
        arr = np.asarray(vals)
        if arr.ndim == 3:
            arr = arr[:, :, 1]
        shap_vec += arr[0] / len(cal_list)

    top_idx = np.argsort(np.abs(shap_vec))[::-1][:5]
    for idx in top_idx:
        fname = model_cols[int(idx)]
        shap_items.append({
            "fitur": fname,
            "label": FRIENDLY_FEAT_NAMES.get(fname, fname),
            "shap": round(float(shap_vec[int(idx)]), 4),
        })

    # Kurva Sehat menggunakan kurva_sehat_quantiles.joblib (NHANES sleep_wear_hours)
    meta = profile_meta or {}
    age_val = _safe_float(meta.get("usia")) or _safe_float(last_raw.get("usia")) or 30.0
    jk_str = str(meta.get("jk") or meta.get("jenis_kelamin") or last_raw.get("jenis_kelamin") or "")
    gender_bin = 1 if "perempuan" in jk_str.lower() else 0
    X_nh = pd.DataFrame([{"RIDAGEYR": float(age_val), "gender_bin": gender_bin}])
    sleep_models = kurva_b["sleep_wear_hours"]
    qs = sorted(sleep_models.keys())
    q_preds = [float(sleep_models[q].predict(X_nh)[0]) for q in qs]
    recent_sleep = feat["ch_ritme_sleep_duration"].dropna().tail(7)
    if recent_sleep.empty or abstain == 1:
        kurva_obj = {
            "pos": None,
            "label": "Belum dapat ditempatkan",
            "sum": "Pita referensi memerlukan catatan durasi istirahat harian yang cukup.",
        }
    else:
        user_sleep = float(recent_sleep.mean())
        pos = float(np.interp(user_sleep, q_preds, [10.0, 25.0, 50.0, 75.0, 90.0]))
        pos = float(np.clip(pos, 4.0, 96.0))
        if pos < 25.0:
            k_lbl = "Lebih Singkat dari Rentang Umum"
        elif pos > 75.0:
            k_lbl = "Lebih Panjang dari Rentang Umum"
        else:
            k_lbl = "Dalam Rentang Umum Sebaya"
        age_note = "." if (_safe_float(meta.get("usia")) or _safe_float(last_raw.get("usia"))) else " (isi usia pada Rincian Tambahan agar lebih sesuai)."
        kurva_obj = {
            "pos": round(pos, 1),
            "label": k_lbl,
            "sum": f"Estimasi posisi durasi istirahat pada pita kuantil NHANES wrist-actigraphy{age_note}",
        }

    # Jurang Insight menggunakan jurang_insight_model.joblib
    j_cols = jurang_b["features"]
    X_j = feat.tail(1).reindex(columns=j_cols).fillna(pd.Series(REF_MEDIANS))
    ema_pred = float(jurang_b["mean"].predict(X_j)[0])
    ema_recent = ema_vals.dropna().tail(3)
    if ema_recent.empty:
        self_lbl = "Belum ada catatan perasaan"
        disc_status = "Jeda Catatan"
        jur_txt = "Catatan refleksi mandiri belum tersedia sehingga perbandingan dua sisi ditunda sementara."
    else:
        ema_act = float(ema_recent.mean())
        self_lbl = "Terasa cukup berat" if ema_act >= 14 else ("Terasa agak naik turun" if ema_act >= 9 else "Terasa relatif tenang")
        if abstain == 1:
            disc_status = "Jeda Catatan"
            jur_txt = "Sinyal beberapa hari terakhir masih terbatas sehingga perbandingan dua sisi ditunda sementara."
        else:
            diff = abs(ema_act - ema_pred)
            if diff >= 4.5:
                disc_status = "Ada Jarak Halus"
                jur_txt = "Terdapat perbedaan antara catatan harian dan pola aktivitas terukur; diskusi santai dengan pendamping dapat menjadi pertimbangan."
            else:
                disc_status = "Selaras"
                jur_txt = "Catatan harian yang dilaporkan berjalan sejalan dengan pola aktivitas keseharian yang terukur."

    pass_lbl = "Pola belum cukup terbaca" if abstain == 1 else ("Pola aktivitas melambat" if ema_pred >= 11.0 else "Pola aktivitas cukup terjaga")

    # Waktu menggunakan waktu_survival.joblib (CoxPHFitter)
    w_row = pd.DataFrame([{
        "cermin_shift_7d": float(feat["cermin_shift_7d"].iloc[-1]) if pd.notna(feat["cermin_shift_7d"].iloc[-1]) else 0.9,
        "inconsistency_std": float(inconsistency_std.iloc[-1]),
        "active_ratio": float(feat["ch_gerak_active_ratio"].iloc[-1]) if pd.notna(feat["ch_gerak_active_ratio"].iloc[-1]) else REF_MEDIANS["ch_gerak_active_ratio"],
        "sleep_dur": float(feat["ch_ritme_sleep_duration"].iloc[-1]) if pd.notna(feat["ch_ritme_sleep_duration"].iloc[-1]) else REF_MEDIANS["ch_ritme_sleep_duration"],
        "convo_dur_log": float(feat["ch_suara_convo_dur_log"].iloc[-1]) if pd.notna(feat["ch_suara_convo_dur_log"].iloc[-1]) else REF_MEDIANS["ch_suara_convo_dur_log"],
    }])
    sf = waktu_b["coxph"].predict_survival_function(w_row, times=[0, 7, 14, 21, 28])
    surv_days = [0, 7, 14, 21, 28]
    surv_probs = [round(float(np.clip(1.0 - float(sf.iloc[idx, 0]), 0.01, 0.99)), 3) for idx in range(len(surv_days))]
    p14 = surv_probs[2]
    p28 = surv_probs[4]
    waktu_abs = "Data minggu ini masih terbatas; pembacaan arah 2-4 minggu ditunda hingga catatan kembali mencukupi."
    if abstain == 1:
        waktu_txt = waktu_abs
    elif p28 >= 0.52 or statuses["Individu"] == "Perlu Perhatian":
        waktu_txt = "Ada tanda perlambatan dalam 2-4 minggu ke depan yang baiknya diperhatikan lebih awal."
    else:
        waktu_txt = "Kecenderungan ritme 2-4 minggu ke depan terlihat relatif stabil."

    # Cermin & Pemicu & Inkonsistensi (multi-kanal untuk visualisasi interaktif)
    valid_pts = feat["cermin_shift_score"].fillna(0.0).tail(10) if n >= 3 else pd.Series([], dtype=float)
    tail_idx = valid_pts.index
    pts = [round(float(x), 3) for x in valid_pts.tolist()]
    labels = [str(df.loc[idx, "tanggal"])[:10] for idx in tail_idx]
    channel_series = {
        "Tidur & Ritme": [round(float(abs(feat.loc[idx, "z_causal_ch_ritme_sleep_duration"])) if pd.notna(feat.loc[idx, "z_causal_ch_ritme_sleep_duration"]) else 0.0, 3) for idx in tail_idx],
        "Gerak": [round(float(abs(feat.loc[idx, "z_causal_ch_gerak_active_ratio"])) if pd.notna(feat.loc[idx, "z_causal_ch_gerak_active_ratio"]) else 0.0, 3) for idx in tail_idx],
        "Mobilitas": [round(float(abs(feat.loc[idx, "z_causal_ch_mobilitas_dist_log"])) if pd.notna(feat.loc[idx, "z_causal_ch_mobilitas_dist_log"]) else 0.0, 3) for idx in tail_idx],
        "Komunikasi & Suara": [round(float(abs(feat.loc[idx, "z_causal_ch_suara_convo_dur_log"])) if pd.notna(feat.loc[idx, "z_causal_ch_suara_convo_dur_log"]) else 0.0, 3) for idx in tail_idx],
    }
    cer_last = float(feat["cermin_shift_7d"].iloc[-1]) if pd.notna(feat["cermin_shift_7d"].iloc[-1]) else 0.0

    if abstain == 1:
        cer_sum = "Catatan sinyal belum cukup lengkap untuk dibandingkan secara utuh dengan kebiasaan pribadi sebelumnya."
    elif cer_last >= 1.6:
        cer_sum = "Ritme harian terbaru bergerak menjauh dari pola kebiasaan beberapa hari sebelumnya."
    elif cer_last >= 0.95:
        cer_sum = "Terdapat pergeseran bertahap pada kebiasaan harian dibanding hari-hari sebelumnya."
    else:
        cer_sum = "Ritme harian terbaru berjalan selaras dengan kebiasaan pribadi sebelumnya."

    last_z = {
        "Ritme & Istirahat": abs(float(feat["z_causal_ch_ritme_sleep_duration"].iloc[-1])) if pd.notna(feat["z_causal_ch_ritme_sleep_duration"].iloc[-1]) else 0.0,
        "Suasana Suara Sekitar": abs(float(feat["z_causal_ch_suara_convo_dur_log"].iloc[-1])) if pd.notna(feat["z_causal_ch_suara_convo_dur_log"].iloc[-1]) else 0.0,
        "Jangkauan Mobilitas": abs(float(feat["z_causal_ch_mobilitas_dist_log"].iloc[-1])) if pd.notna(feat["z_causal_ch_mobilitas_dist_log"].iloc[-1]) else 0.0,
        "Gerak Harian": abs(float(feat["z_causal_ch_gerak_active_ratio"].iloc[-1])) if pd.notna(feat["z_causal_ch_gerak_active_ratio"].iloc[-1]) else 0.0,
        "Pola Komunikasi": abs(float(feat["z_causal_ch_komunikasi_total_events"].iloc[-1])) if pd.notna(feat["z_causal_ch_komunikasi_total_events"].iloc[-1]) else 0.0,
    }
    pemicu_top3 = [] if abstain == 1 else [k for k, _ in sorted(last_z.items(), key=lambda x: x[1], reverse=True)[:3]]

    chan_statuses = [
        ["Gerak", _channel_state(feat["z_causal_ch_gerak_active_ratio"].iloc[-1], abstain)],
        ["Tidur & Ritme", _channel_state(feat["z_causal_ch_ritme_sleep_duration"].iloc[-1], abstain)],
        ["Mobilitas", _channel_state(feat["z_causal_ch_mobilitas_dist_log"].iloc[-1], abstain)],
        ["Komunikasi & Suara", _channel_state(feat["z_causal_ch_komunikasi_total_events"].iloc[-1], abstain)],
    ]
    channel_scores = {
        "Gerak": round(float(np.clip(1.0 - last_z["Gerak Harian"] / 3.0, 0.2, 1.0)), 2) if abstain == 0 else 0.5,
        "Tidur & Ritme": round(float(np.clip(1.0 - last_z["Ritme & Istirahat"] / 3.0, 0.2, 1.0)), 2) if abstain == 0 else 0.5,
        "Mobilitas": round(float(np.clip(1.0 - last_z["Jangkauan Mobilitas"] / 3.0, 0.2, 1.0)), 2) if abstain == 0 else 0.5,
        "Komunikasi & Suara": round(float(np.clip(1.0 - max(last_z["Pola Komunikasi"], last_z["Suasana Suara Sekitar"]) / 3.0, 0.2, 1.0)), 2) if abstain == 0 else 0.5,
    }
    inc_txt = (
        "Keselarasan antar-kanal dihitung setelah catatan harian cukup."
        if abstain == 1
        else (
            "Keempat kanal kebiasaan utama bergerak cukup selaras satu sama lain."
            if all(st == "Selaras" for _, st in chan_statuses)
            else "Beberapa kebiasaan bergerak tidak sejalan minggu ini, misalnya saat ritme istirahat berubah sementara aktivitas lain tetap padat."
        )
    )

    seed_val = int(last_raw["tanggal_dt"].dayofyear) if pd.notna(last_raw["tanggal_dt"]) else n

    return {
        "src": "local",
        "n": int(n),
        "abstain": abstain,
        "sensorOff": sensor_off_flag,
        "missing": missing_cnt,
        "ctx": str(ctx_series.iloc[-1]),
        "lastDate": str(last_raw["tanggal"])[:10],
        "p": round(prob_lr, 4),
        "unc": round(unc_std, 4),
        "p14": round(p14, 3),
        "p28": round(p28, 3),
        "survivalCurve": {"days": surv_days, "probs": surv_probs},
        "status": statuses,
        "pts": pts,
        "labels": labels,
        "channelSeries": channel_series,
        "base": 1.1,
        "cerSum": cer_sum,
        "kurva": kurva_obj,
        "selfL": self_lbl,
        "passL": pass_lbl,
        "disc": disc_status,
        "jur": jur_txt,
        "waktu": waktu_txt,
        "waktuAbs": waktu_abs,
        "pem": pemicu_top3,
        "chan": chan_statuses,
        "channelScores": channel_scores,
        "incon": inc_txt,
        "seed": seed_val,
        "shap": shap_items,
        "shapNote": "Kontribusi log-odds lokal dihitung langsung dari model LogisticRegression terkalibrasi (.joblib).",
        "engine": "joblib_live",
    }
