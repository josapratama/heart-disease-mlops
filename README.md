# Proyek Pengembangan dan Pengoperasian Sistem Machine Learning

**Nama:** Josa Pratama  
**Username Dicoding:** josa_pratama

---

| Kategori                    | Deskripsi                                                                                                                                                                                                                                                                                                                                                                      |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Dataset**                 | [Heart Disease UCI Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) — 303 sampel, 13 fitur klinis, label biner (ada/tidak penyakit jantung)                                                                                                                                                                                                                     |
| **Masalah**                 | Penyakit jantung adalah penyebab kematian nomor satu di dunia. Deteksi dini berbasis parameter klinis sederhana dapat membantu tenaga medis mengambil keputusan lebih cepat dan akurat                                                                                                                                                                                         |
| **Solusi machine learning** | Klasifikasi biner menggunakan deep neural network yang dibangun di atas pipeline TFX lengkap (ExampleGen → StatisticsGen → SchemaGen → ExampleValidator → Transform → Tuner → Trainer → Resolver → Evaluator → Pusher). Target: akurasi ≥ 80% dan AUC-ROC ≥ 0.85                                                                                                               |
| **Metode pengolahan**       | Fitur numerik (age, trestbps, chol, thalach, oldpeak, ca) dinormalisasi dengan Z-Score. Fitur kategorikal (sex, cp, fbs, restecg, exang, slope, thal) di-encode menggunakan vocabulary lookup. Seluruh transformasi diimplementasikan dengan `tf.Transform` agar konsisten antara training dan serving                                                                         |
| **Arsitektur model**        | Input (13 fitur) → Dense(128, relu) + Dropout(0.1) → Dense(64, relu) + Dropout(0.4) → Dense(32, relu) → Dense(1, sigmoid). Optimizer: Adam, Loss: Binary Crossentropy, Regularisasi: L2 + Early Stopping (monitor val_auc, patience=5). Hyperparameter tuning otomatis menggunakan Keras Tuner                                                                                 |
| **Metrik evaluasi**         | Binary Accuracy, AUC-ROC, Precision, Recall                                                                                                                                                                                                                                                                                                                                    |
| **Performa model**          | Hasil training dengan hyperparameter terbaik (units_1=128, units_2=64, units_3=32, dropout_1=0.1, dropout_2=0.4, lr=0.001): Accuracy ~84.75%, AUC-ROC ~0.9003 pada validation set (epoch 5). Model dinyatakan **BLESSED** oleh Evaluator TFX. Hasil prediction request ke cloud (10 sampel): akurasi 90%                                                                       |
| **Opsi deployment**         | Model di-serving menggunakan **Flask + TensorFlow** (Python 3.10, Gunicorn) yang dideploy ke **Railway** (cloud PaaS). API meng-expose REST endpoint di port 8080. Monitoring menggunakan **Prometheus** metrics yang di-expose via endpoint `/metrics`                                                                                                                        |
| **Web app**                 | [https://heart-disease-mlops-production.up.railway.app](https://heart-disease-mlops-production.up.railway.app) — health check via `GET /`, prediksi via `POST /predict`, metrics via `GET /metrics`                                                                                                                                                                            |
| **Monitoring**              | Flask app meng-expose Prometheus metrics di endpoint `/metrics`. Metrik yang dipantau: `heart_disease_prediction_requests_total`, `heart_disease_prediction_latency_seconds`, `heart_disease_prediction_confidence`, `heart_disease_positive/negative_predictions_total`, `heart_disease_model_up`. Grafana dashboard menampilkan real-time visualization dari metrik tersebut |

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
│   ├── josa_pratama-monitoring.png # Screenshot Prometheus metrics
│   └── josa_pratama-grafana.png    # Screenshot Grafana dashboard
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

### 3. Test Prediksi ke Flask API

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}'
```

---

## Referensi

- [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx)
- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
- [Heart Disease UCI Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
