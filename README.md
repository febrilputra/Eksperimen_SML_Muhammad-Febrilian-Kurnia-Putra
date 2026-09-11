```text
Eksperimen IBM Telco Customer Churn
|
+-- Dataset      -- salinan IBM dan identitas sumber
+-- Notebook     -- loading, EDA, dan preprocessing manual
+-- Automasi     -- fungsi preprocessing dan GitHub Actions
`-- Hasil        -- fitur numerik, transformer, dan manifest
```

# Eksperimen MSML — Muhammad-Febrilian-Kurnia-Putra

Username Dicoding: **febril_putra**. Kasus proyek adalah klasifikasi pelanggan
berhenti berlangganan menggunakan **IBM Telco Customer Churn**. Repository ini
menangani eksperimen dan preprocessing. Retraining dan publikasi image berada
di repository terpisah [Workflow-CI](https://github.com/febrilputra/Workflow-CI).

| Lokasi | Isi |
| --- | --- |
| `telco_churn_raw/` | CSV IBM asli, lisensi repository sumber, dan keterangan asal data |
| `preprocessing/Eksperimen_Muhammad-Febrilian-Kurnia-Putra.ipynb` | Notebook turunan template resmi; tujuh cell kode sudah dijalankan tanpa error |
| `preprocessing/automate_Muhammad-Febrilian-Kurnia-Putra.py` | Tahapan notebook yang dikemas menjadi fungsi dan CLI |
| `preprocessing/telco_churn_preprocessing/` | Enam CSV train/validation/test, transformer, skema, contoh inferensi, dan manifest |
| `.github/workflows/preprocessing.yml` | Pengujian dan pembuatan ulang preprocessing pada push terkait data/kode atau pemicu manual |
| `tests/test_preprocessing.py` | Pemeriksaan pemisahan pelanggan, median training, kategori baru, dan penolakan numerik invalid |

## Menjalankan ulang

Gunakan Python **3.12**; verifikasi lokal memakai **3.12.10**. Dari root repository:

```powershell
python -m pip install -r requirements.txt
python -m pytest tests -q
python preprocessing/automate_Muhammad-Febrilian-Kurnia-Putra.py
```

Buka notebook dengan kernel Python dari environment yang sama dan jalankan
semua cell secara berurutan. Nama kernel lokal `msml` dapat diganti dengan kernel
Python 3.12 milik pengguna. Notebook memeriksa hash dataset asli sebelum berjalan.
Script CLI menerima `--input`, `--output`, dan `--seed`; input baru harus memenuhi
skema dan validasi pelanggan. Manifest mencatat hash input aktual, sedangkan URL
dan commit IBM tetap menjadi referensi asal dataset.

## Keputusan preprocessing

```text
7043 pelanggan -> split stratified, seed42
                    |
                    +--> train4930 -> fit median, modus, one-hot encoder
                    +--> val1056  -> transform saja
                    `--> test1057 -> transform saja
                                         |
                                    45 fitur numerik
```

`customerID` dan target `Churn` dikeluarkan dari fitur. TotalCharges yang kosong
ketika tenure=0 diisi 0; missing numerik lainnya memakai median training.
Kategori diisi modus training, lalu one-hot encoding dengan penanganan kategori
baru. Tidak ada scaling atau penghapusan outlier otomatis karena model berikutnya
memakai Random Forest. Train, validation, dan test berisi pelanggan yang berbeda.

Hasil notebook manual dibandingkan dengan output script pada direktori terpisah:
semua CSV, skema, manifest, dan checksum artefaknya identik. Workflow mengunggah
dataset baru sebagai artefak `telco-churn-preprocessing-<commit>`; status Actions
harus diperiksa pada tab Actions setelah commit dipublikasikan. Eksekusi lokal
tidak menggantikan bukti keberhasilan Actions online.

Data berupa snapshot contoh IBM. Holdout acak mengukur generalisasi pada distribusi
snapshot ini; belum membuktikan performa prospektif pada pelanggan masa depan.

## Key Concepts — Summary

| Term | Definition |
| --- | --- |
| Churn | Target pelanggan berhenti berlangganan; Yes=1, No=0 |
| Transformer | Median, modus, dan kategori yang dipelajari hanya dari training |
| Manifest | Identitas sumber, anggota split, dan checksum hasil |
| Actions | Runner yang menjalankan ulang preprocessing dan menyimpan hasilnya |

> Dataset yang sama, pemisahan yang tetap, dan transformer yang tersimpan menghubungkan eksperimen dengan model dan inferensi.
