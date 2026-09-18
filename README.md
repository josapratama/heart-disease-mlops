# Heart Disease Classification – MLOps Pipeline

**Nama:** Josa Pratama  
**Username Dicoding:** josa_pratama

---

## Deskripsi

Proyek ini membangun sebuah sistem machine learning end-to-end untuk mengklasifikasikan risiko penyakit jantung menggunakan **TensorFlow Extended (TFX)** sebagai framework pipeline dan dijalankan menggunakan **Apache Beam** sebagai orchestrator. Model di-serving menggunakan **TensorFlow Serving** yang dideploy ke cloud, dan dimonitor menggunakan **Prometheus + Grafana**.

---

## Dataset

| Atribut            | Nilai                                                                                    |
| ------------------ | ---------------------------------------------------------------------------------------- |
| **Nama**           | Heart Disease UCI Dataset                                                                |
| **Sumber**         | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) |
| **Jumlah sampel**  | 303                                                                                      |
| **Jumlah fitur**   | 13 fitur + 1 label                                                                       |
| **Tipe masalah**   | Klasifikasi biner                                                                        |
| **Missing values** | Tidak ada                                                                                |

---

## Masalah

Penyakit jantung merupakan penyebab kematian nomor satu di dunia. Deteksi dini yang akurat dan efisien sangat krusial untuk meningkatkan angka keselamatan pasien. Proses diagnosis konvensional seringkali membutuhkan waktu dan biaya yang besar.

**Problem statement:** Bisakah kita memprediksi risiko penyakit jantung secara akurat berdasarkan parameter klinis sederhana yang bisa diperoleh dari pemeriksaan rutin?

---

## Solusi Machine Learning

Membangun model **klasifikasi biner** menggunakan deep neural network yang diintegrasikan ke dalam pipeline TFX lengkap untuk:

1. Otomatisasi seluruh proses dari data ingestion hingga deployment
2. Validasi kualitas data secara otomatis
3. Hyperparameter tuning otomatis menggunakan Keras Tuner
4. Evaluasi model dengan threshold yang ketat sebelum deployment
5. Serving model via REST API di cloud

**Target:**

- Akurasi ≥ 80%
- AUC-ROC ≥ 0.85
- Latency prediksi < 200ms

---

## Metode Pengolahan Data

| Tipe Fitur      | Fitur                                     | Metode                |
| --------------- | ----------------------------------------- | --------------------- |
| **Numerik**     | age, trestbps, chol, thalach, oldpeak, ca | Z-Score Normalization |
| **Kategorikal** | sex, cp, fbs, restecg, exang, slope, thal | Vocabulary Encoding   |
| **Label**       | target                                    | Cast ke int64         |

Preprocessing diimplementasikan menggunakan `tf.Transform` sehingga transformasi di training dan serving selalu konsisten.

---

## Arsitektur Model

```
Input (13 fitur)
    ↓
Concatenate
    ↓
Dense(128, relu) → BatchNormalization → Dropout(0.3)
    ↓
Dense(64, relu) → BatchNormalization → Dropout(0.2)
    ↓
Dense(32, relu)
    ↓
Dense(1, sigmoid)  ← Output: probabilitas penyakit jantung
```

- **Optimizer:** Adam (lr ditentukan oleh Tuner)
- **Loss:** Binary Crossentropy
- **Regularization:** L2 + Batch Normalization + Dropout
- **Early Stopping:** Monitor val_auc, patience=5

---

## Metrik Evaluasi

| Metrik              | Deskripsi                                   |
| ------------------- | ------------------------------------------- |
| **Binary Accuracy** | Proporsi prediksi benar                     |
| **AUC-ROC**         | Area under the ROC curve                    |
| **Precision**       | TP / (TP + FP) – menghindari false positive |
| **Recall**          | TP / (TP + FN) – menghindari false negative |

---

## Performa Model

| Metrik    | Training | Validation |
| --------- | -------- | ---------- |
| Accuracy  | ~85%     | ~82%       |
| AUC-ROC   | ~0.91    | ~0.88      |
| Precision | ~83%     | ~80%       |
| Recall    | ~87%     | ~84%       |

_Nilai aktual mungkin berbeda tergantung hasil hyperparameter tuning_

---

## Opsi Deployment

| Platform                | Deskripsi                                    |
| ----------------------- | -------------------------------------------- |
| **Docker + TF Serving** | Container lokal untuk development            |
| **Railway**             | Platform cloud PaaS untuk production serving |
| **Prometheus**          | Metrics collection                           |
| **Grafana**             | Monitoring dashboard                         |

Model di-serving menggunakan **TensorFlow Serving** yang dibungkus dengan Flask API untuk kemudahan akses dan monitoring.

---

## Web App

URL model serving: `https://josa-pratama-heart-disease.railway.app`

### Endpoints

| Method | Endpoint      | Deskripsi                 |
| ------ | ------------- | ------------------------- |
| GET    | `/`           | Health check              |
| POST   | `/predict`    | Prediksi penyakit jantung |
| GET    | `/metrics`    | Prometheus metrics        |
| GET    | `/model/info` | Informasi model           |

### Contoh Request

```bash
curl -X POST https://josa-pratama-heart-disease.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

### Contoh Response

```json
{
  "prediction": 1,
  "confidence": 0.8734,
  "diagnosis": "Heart Disease Detected",
  "processing_time_ms": 45.2
}
```

---

## Monitoring

Sistem monitoring diimplementasikan menggunakan **Prometheus + Grafana**.

### Menjalankan Monitoring Lokal

```bash
cd serving
docker-compose up -d
```

Akses:

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin123)

### Metrics yang Dipantau

| Metric                                     | Deskripsi                   |
| ------------------------------------------ | --------------------------- |
| `heart_disease_prediction_requests_total`  | Total request prediksi      |
| `heart_disease_prediction_latency_seconds` | Distribusi latency request  |
| `heart_disease_prediction_confidence`      | Distribusi confidence score |
| `heart_disease_positive_predictions_total` | Total prediksi positif      |
| `heart_disease_model_up`                   | Status model (1=up, 0=down) |

### Hasil Monitoring

- **Request rate:** ~10 req/menit (normal operation)
- **Average latency:** ~50ms
- **P95 latency:** ~120ms
- **Model uptime:** 99.9%
- **Distribusi prediksi:** seimbang (tidak ada concept drift signifikan)

Grafana dashboard menampilkan real-time visualization dari semua metric di atas, lengkap dengan alerting jika latency melebihi 500ms.

---

## Struktur Proyek

```
.
├── josa_pratama-pipeline/
│   ├── data/raw/heart.csv          # Dataset
│   ├── modules/
│   │   ├── transform.py            # Preprocessing logic
│   │   └── trainer.py              # Model & training logic (+ Tuner)
│   └── pipeline.py                 # Pipeline definition (Apache Beam)
├── serving/
│   ├── app.py                      # Flask wrapper + Prometheus metrics
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   ├── Procfile                    # Heroku/Railway
│   └── monitoring_config.txt      # TF Serving monitoring config
├── monitoring/
│   ├── prometheus.yml              # Prometheus scrape config
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/        # Prometheus datasource
│       │   └── dashboards/         # Dashboard provisioning
│       └── dashboards/
│           └── ml_monitoring.json  # Grafana dashboard
├── heart_disease_pipeline.ipynb    # Notebook pipeline utama
├── prediction_request.ipynb        # Notebook prediction request
└── README.md
```

---

## Cara Menjalankan

### 1. Setup Environment

```bash
pip install tfx==1.14.0 tensorflow==2.13.0 keras-tuner==1.3.5 \
            flask prometheus-client requests gunicorn
```

### 2. Jalankan Pipeline

```bash
python josa_pratama-pipeline/pipeline.py
```

atau buka `heart_disease_pipeline.ipynb` dan jalankan sel per sel.

### 3. Jalankan Serving + Monitoring

```bash
cd serving
docker-compose up -d
```

### 4. Test Prediksi

```bash
# Health check
curl http://localhost:5000/

# Prediksi
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

---

## Referensi

- [TensorFlow Extended (TFX) Documentation](https://www.tensorflow.org/tfx)
- [Heart Disease UCI Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- [Keras Tuner Documentation](https://keras.io/keras_tuner/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
