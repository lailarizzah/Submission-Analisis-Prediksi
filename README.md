# Prediksi Penyakit Jantung

Proyek ini merupakan submission akhir dari kelas Machine Learning, yang berfokus pada pembuatan model prediksi apakah seseorang memiliki penyakit jantung atau tidak berdasarkan dataset UCI Heart Disease.

---

## 📁 Struktur Direktori

| Nama File                               | Deskripsi                                                                 |
|----------------------------------------|---------------------------------------------------------------------------|
| `heart_disease_uci.csv`                | Dataset utama yang digunakan dalam proyek                                |
| `Submission_Prediksi_Penyakit_Jantung.ipynb` | Notebook utama berisi seluruh alur proyek      |
| `submission_prediksi_penyakit_jantung.py`    | File Python alternatif untuk eksekusi script                             |
| `Laporan-Submission.md`                | Laporan akhir dalam format markdown                                      |
| `image/`                                | Folder berisi visualisasi yang digunakan dalam laporan dan notebook      |

---

## 📌 Deskripsi Proyek

Tujuan dari proyek ini adalah membangun model klasifikasi untuk memprediksi penyakit jantung berdasarkan berbagai fitur medis. Tiga algoritma digunakan:

- K-Nearest Neighbors (KNN)
- Random Forest
- XGBoost

Model dievaluasi menggunakan metrik **confusion matrix**, **akurasi**, **precision**, **recall**, dan **f1-score**.

---

## 📦 Requirements

File `requirements.txt` berisi dependensi yang digunakan dalam proyek ini. Install dengan:

```bash
pip install -r requirements.txt
