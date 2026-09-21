# Proyek Pengembangan dan Pengoperasian Sistem Machine Learning

|                       |              |
| --------------------- | ------------ |
| **Nama**              | Josa Pratama |
| **Username Dicoding** | josa_pratama |

---

## Deskripsi

Proyek ini membangun pipeline machine learning end-to-end menggunakan **TensorFlow Extended (TFX)** untuk mengklasifikasikan apakah seorang pasien menderita penyakit jantung berdasarkan data klinis. Pipeline dijalankan menggunakan **Apache Beam** (BeamDagRunner) sebagai orchestrator dan mencakup 10 komponen TFX lengkap: ExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Tuner, Trainer, Resolver, Evaluator, dan Pusher. Model yang sudah ditraining di-serve menggunakan **TensorFlow Serving** di cloud platform **Railway**.

---

## Dataset

[Heart Disease UCI Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) — dataset klinis berisi 199 sampel pasien dengan 13 fitur (usia, jenis kelamin, jenis nyeri dada, tekanan darah, kolesterol, dll.) dan label biner (0 = tidak sakit jantung, 1 = sakit jantung).

---

## Masalah

Penyakit jantung merupakan salah satu penyebab kematian utama di seluruh dunia. Deteksi dini sangat penting untuk meningkatkan angka keselamatan pasien, namun proses diagnosis konvensional memerlukan banyak pemeriksaan klinis yang memakan waktu dan biaya.

**Pertanyaan inti:** Bagaimana memprediksi risiko penyakit jantung secara akurat dan cepat berdasarkan data klinis pasien yang mudah diperoleh, sehingga dokter dapat mengambil keputusan lebih awal?

---

## Solusi Machine Learning

Membangun model klasifikasi biner menggunakan **deep neural network** yang diintegrasikan dalam TFX pipeline lengkap dengan komponen:

- **ExampleGen** — ingesti dan split data (80% train / 20% eval)
- **StatisticsGen** — statistik deskriptif dataset
- **SchemaGen** — schema otomatis dari statistik
- **ExampleValidator** — validasi data terhadap schema
- **Transform** — preprocessing dengan `tf.Transform`
- **Tuner** — hyperparameter tuning dengan Keras Tuner (RandomSearch, 5 trials)
- **Trainer** — training model dengan HP terbaik
- **Resolver** — baseline model terbaik yang sudah di-blessed
- **Evaluator** — evaluasi model baru vs baseline dengan TFMA
- **Pusher** — deploy model ke serving directory jika BLESSED

---

## Metode Pengolahan

Seluruh preprocessing diimplementasikan dengan `tf.Transform` agar konsisten antara training dan serving:

- **Fitur numerik** (age, trestbps, chol, thalach, oldpeak, ca): dinormalisasi dengan **Z-Score** (`tft.scale_to_z_score`)
- **Fitur kategorikal** (sex, cp, fbs, restecg, exang, slope, thal): di-encode menggunakan **vocabulary lookup** (`tft.compute_and_apply_vocabulary`)
- **Label** (target): pass-through, di-cast ke `int64`

---

## Arsitektur Model

```
Input (13 fitur, masing-masing shape (1,))
    ↓
Concatenate → [batch, 13]
    ↓
Dense(128, relu) + L2 + Dropout(0.1)
    ↓
Dense(64, relu) + L2 + Dropout(0.4)
    ↓
Dense(32, relu)
    ↓
Dense(1, sigmoid)  ← Output probabilitas
```

- **Optimizer:** Adam (lr=0.001)
- **Loss:** Binary Crossentropy
- **Regularisasi:** L2(0.001) pada dua hidden layer pertama
- **Callbacks:** EarlyStopping (patience=5, monitor val_auc), ReduceLROnPlateau, TensorBoard

---

## Metrik Evaluasi

| Metrik          | Deskripsi                              |
| --------------- | -------------------------------------- |
| Binary Accuracy | Akurasi prediksi biner (threshold 0.5) |
| AUC-ROC         | Area Under the ROC Curve               |
| Precision       | Presisi prediksi positif               |
| Recall          | Sensitivitas terhadap kelas positif    |

Threshold evaluasi TFX: **Binary Accuracy ≥ 0.60** dan tidak boleh turun lebih dari 0.01 dari model baseline.

---

## Performa Model

Hyperparameter terbaik dari Keras Tuner (5 trials, RandomSearch):

| Parameter     | Nilai |
| ------------- | ----- |
| units_1       | 128   |
| units_2       | 64    |
| units_3       | 32    |
| dropout_1     | 0.1   |
| dropout_2     | 0.4   |
| learning_rate | 0.001 |

Hasil training pada validation set:

| Metrik    | Nilai   |
| --------- | ------- |
| Accuracy  | ~84.75% |
| AUC-ROC   | ~0.9003 |
| Precision | ~78.57% |
| Recall    | ~84.62% |

Model dinyatakan **BLESSED** oleh Evaluator TFX. Hasil prediction request ke cloud (10 sampel): **akurasi 90%**.

---

## Opsi Deployment

Model di-serving menggunakan **TensorFlow Serving** yang dideploy ke **Railway** (cloud PaaS).

- **Image:** `tensorflow/serving:latest`
- **REST API Port:** 8501
- **gRPC Port:** 8500
- **Monitoring:** Prometheus metrics via `/monitoring/prometheus/metrics`
- **Model path di container:** `/models/heart_disease_model`

---

## Web App

**URL:** [https://heart-disease-mlops-production.up.railway.app](https://heart-disease-mlops-production.up.railway.app)

| Endpoint                                  | Method | Deskripsi                                |
| ----------------------------------------- | ------ | ---------------------------------------- |
| `/v1/models/heart_disease_model`          | GET    | Status dan versi model                   |
| `/v1/models/heart_disease_model/metadata` | GET    | Metadata model (signature, input/output) |
| `/v1/models/heart_disease_model:predict`  | POST   | Prediksi penyakit jantung                |
| `/monitoring/prometheus/metrics`          | GET    | Prometheus metrics TF Serving            |

---

## Monitoring

Monitoring menggunakan **Prometheus** yang membaca metrik dari TF Serving endpoint `/monitoring/prometheus/metrics`.

Konfigurasi scrape (`monitoring/prometheus.yml`):

- **Scrape interval:** 5 detik
- **Target:** `heart-disease-mlops-production.up.railway.app`
- **Metrics path:** `/monitoring/prometheus/metrics`

Metrik TF Serving yang dipantau:

- `:tensorflow:serving:request_count` — total request ke model
- `:tensorflow:serving:request_latency` — latency per request
- `:tensorflow:serving:runtime_latency` — latency runtime
- `:tensorflow:core:graph_runs` — jumlah eksekusi graph

Grafana dashboard (`monitoring/grafana/dashboards/ml_monitoring.json`) menampilkan real-time visualization dari metrik-metrik tersebut.
