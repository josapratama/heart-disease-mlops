# Proyek Pengembangan dan Pengoperasian Sistem Machine Learning

**Nama:** Josa Pratama  
**Username Dicoding:** josa_pratama

---

| Kategori                    | Deskripsi                                                                                                                                                                                                                                                                                                                          |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dataset**                 | [Heart Disease UCI Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) — 303 sampel, 13 fitur klinis, label biner (ada/tidak penyakit jantung)                                                                                                                                                                         |
| **Masalah**                 | Penyakit jantung adalah penyebab kematian nomor satu di dunia. Deteksi dini berbasis parameter klinis sederhana dapat membantu tenaga medis mengambil keputusan lebih cepat dan akurat                                                                                                                                             |
| **Solusi machine learning** | Klasifikasi biner menggunakan deep neural network yang dibangun di atas pipeline TFX lengkap (ExampleGen → StatisticsGen → SchemaGen → ExampleValidator → Transform → Tuner → Trainer → Resolver → Evaluator → Pusher). Target: akurasi ≥ 80% dan AUC-ROC ≥ 0.85                                                                   |
| **Metode pengolahan**       | Fitur numerik (age, trestbps, chol, thalach, oldpeak, ca) dinormalisasi dengan Z-Score. Fitur kategorikal (sex, cp, fbs, restecg, exang, slope, thal) di-encode menggunakan vocabulary lookup. Seluruh transformasi diimplementasikan dengan `tf.Transform` agar konsisten antara training dan serving                             |
| **Arsitektur model**        | Input (13 fitur) → Dense(128, relu) + BatchNorm + Dropout(0.3) → Dense(64, relu) + BatchNorm + Dropout(0.2) → Dense(32, relu) → Dense(1, sigmoid). Optimizer: Adam, Loss: Binary Crossentropy, Regularisasi: L2 + Early Stopping (monitor val_auc, patience=5). Hyperparameter tuning otomatis menggunakan Keras Tuner             |
| **Metrik evaluasi**         | Binary Accuracy, AUC-ROC, Precision, Recall                                                                                                                                                                                                                                                                                        |
| **Performa model**          | Hasil training dengan hyperparameter terbaik (units=128/128/64, dropout=0.1/0.4, lr=0.001): Accuracy ~84%, AUC-ROC ~0.9177, Precision ~88%, Recall ~64% pada validation set. Model dinyatakan **BLESSED** oleh Evaluator TFX. Hasil prediction request ke cloud (10 sampel): akurasi 90%                                           |
| **Opsi deployment**         | Model di-serving menggunakan **TensorFlow Serving** yang dideploy ke **Railway** (cloud PaaS). TF Serving meng-expose REST API di port 8501 dan gRPC di port 8500. Monitoring menggunakan **Prometheus + Grafana**                                                                                                                 |
| **Web app**                 | [https://heart-disease-mlops-production.up.railway.app](https://heart-disease-mlops-production.up.railway.app) — akses metadata model via `/v1/models/heart_disease_model`, prediksi via `/v1/models/heart_disease_model:predict`                                                                                                  |
| **Monitoring**              | Prometheus mengambil metrik TF Serving dari endpoint `/monitoring/prometheus/metrics` setiap 5 detik. Metrik yang dipantau meliputi `tensorflow:serving:request_count`, `tensorflow:serving:request_latency`, dan `tensorflow:serving:runtime_latency`. Grafana dashboard menampilkan real-time visualization dari metrik tersebut |

---

## Struktur Proyek

```
.
├── josa_pratama-pipeline/
│   ├── data/raw/heart.csv          # Dataset
│   ├── modules/
│   │   ├── transform.py            # Preprocessing logic (tf.Transform)
│   │   └── trainer.py              # Model & training logic (+ Keras Tuner)
│   └── pipeline.py                 # Pipeline definition (BeamDagRunner)
├── serving/
│   ├── Dockerfile                  # TF Serving image untuk deployment cloud
│   ├── docker-compose.yml          # Untuk menjalankan lokal
│   ├── monitoring_config.txt       # TF Serving Prometheus monitoring config
│   ├── railway.json                # Konfigurasi Railway deployment
│   └── model/
│       └── heart_disease_model/    # SavedModel hasil pipeline TFX
├── monitoring/
│   ├── Dockerfile                  # Dockerfile untuk menjalankan Prometheus
│   ├── prometheus.config           # TF Serving monitoring config (Prometheus)
│   ├── prometheus.yml              # Prometheus scrape config
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/        # Prometheus datasource config
│       │   └── dashboards/         # Dashboard provisioning config
│       └── dashboards/
│           └── ml_monitoring.json  # Grafana dashboard definition
├── ss/
│   ├── josa_pratama-deployment.png # Screenshot TF Serving di Railway
│   └── josa_pratama-monitoring.png # Screenshot Prometheus/Grafana metrics
├── heart_disease_pipeline.ipynb    # Notebook pipeline utama (BeamDagRunner)
├── prediction_request.ipynb        # Notebook prediction request ke cloud
└── README.md
```

---

## Cara Menjalankan

### 1. Jalankan Pipeline

```bash
python josa_pratama-pipeline/pipeline.py
```

### 2. Serving Lokal

```bash
cd serving
docker-compose up -d
```

### 3. Test Prediksi ke TF Serving

```bash
curl -X POST http://localhost:8501/v1/models/heart_disease_model:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]}'
```

---

## Referensi

- [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx)
- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
- [Heart Disease UCI Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
