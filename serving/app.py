"""
Flask app untuk Heart Disease Model Serving.
Load model TensorFlow langsung (tanpa TF Serving terpisah).

Endpoint:
  GET  /          → health check
  POST /predict   → prediksi penyakit jantung
  GET  /metrics   → Prometheus metrics
"""

import os
import time
from typing import Any, Dict

import numpy as np
import tensorflow as tf
from flask import Flask, Response, jsonify, request
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

app = Flask(__name__)

# ──────────────────────────────────────────────
# Load model saat startup
# ──────────────────────────────────────────────
MODEL_DIR = os.environ.get(
    'MODEL_DIR',
    os.path.join(os.path.dirname(__file__), 'model', 'heart_disease_model', '1')
)

print(f'Loading model dari: {MODEL_DIR}')
try:
    model = tf.saved_model.load(MODEL_DIR)
    infer = model.signatures['serving_default']
    MODEL_LOADED = True
    print('Model berhasil di-load!')

    # Warmup: jalankan dummy inference agar graph sudah ter-compile
    print('Melakukan warmup inference...')
    dummy = tf.train.Example(features=tf.train.Features(feature={
        name: tf.train.Feature(int64_list=tf.train.Int64List(value=[0]))
        for name in ['age','sex','cp','trestbps','chol','fbs','restecg',
                     'thalach','exang','slope','ca','thal']
    }))
    dummy_float = tf.train.Example(features=tf.train.Features(feature={
        'oldpeak': tf.train.Feature(float_list=tf.train.FloatList(value=[0.0]))
    }))
    # Gabungkan semua fitur
    all_features = {}
    all_features.update({k: v for k, v in dummy.features.feature.items()})
    all_features.update({k: v for k, v in dummy_float.features.feature.items()})
    warmup_example = tf.train.Example(
        features=tf.train.Features(feature=all_features)
    )
    warmup_tensor = tf.constant([warmup_example.SerializeToString()])
    _ = infer(examples=warmup_tensor)
    print('Warmup selesai! App siap menerima request.')

except Exception as e:
    MODEL_LOADED = False
    infer = None
    print(f'Gagal load model: {e}')

# Urutan fitur sesuai training
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal'
]

# ──────────────────────────────────────────────
# Prometheus metrics
# ──────────────────────────────────────────────
REQUEST_COUNT = Counter(
    'heart_disease_prediction_requests_total',
    'Total prediction requests',
    ['method', 'endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'heart_disease_prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)
PREDICTION_CONFIDENCE = Histogram(
    'heart_disease_prediction_confidence',
    'Distribution of confidence scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
POSITIVE_PREDICTIONS = Counter(
    'heart_disease_positive_predictions_total',
    'Total positive predictions'
)
NEGATIVE_PREDICTIONS = Counter(
    'heart_disease_negative_predictions_total',
    'Total negative predictions'
)
MODEL_UP = Gauge(
    'heart_disease_model_up',
    'Model availability (1=up, 0=down)'
)

# Set status model saat startup
MODEL_UP.set(1 if MODEL_LOADED else 0)


def build_tf_example(data: Dict[str, Any]) -> bytes:
    """
    Membangun serialized tf.Example dari data pasien.

    Args:
        data: Dictionary berisi feature values.

    Returns:
        Serialized tf.Example bytes.
    """
    feature_dict = {}
    float_features = {'oldpeak'}

    for name in FEATURE_NAMES:
        val = data.get(name, 0)
        if name in float_features:
            feature_dict[name] = tf.train.Feature(
                float_list=tf.train.FloatList(value=[float(val)])
            )
        else:
            feature_dict[name] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[int(val)])
            )

    example = tf.train.Example(
        features=tf.train.Features(feature=feature_dict)
    )
    return example.SerializeToString()


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint."""
    MODEL_UP.set(1 if MODEL_LOADED else 0)
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "model_dir": MODEL_DIR,
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediksi penyakit jantung.

    Request body (JSON):
    {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
        "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
        "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    }
    """
    if not MODEL_LOADED:
        return jsonify({"error": "Model belum ter-load"}), 503

    start_time = time.time()

    try:
        data = request.get_json(force=True)
        if not data:
            REQUEST_COUNT.labels(
                method='POST', endpoint='/predict', status='400'
            ).inc()
            return jsonify({"error": "Request body harus berupa JSON"}), 400

        # Bangun tf.Example dan jalankan inferensi
        serialized = build_tf_example(data)
        input_tensor = tf.constant([serialized])
        output = infer(examples=input_tensor)

        # Ambil hasil prediksi
        output_key = list(output.keys())[0]
        confidence = float(output[output_key].numpy()[0][0])
        prediction = int(confidence >= 0.5)

        # Update Prometheus metrics
        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)
        PREDICTION_CONFIDENCE.observe(confidence)
        REQUEST_COUNT.labels(
            method='POST', endpoint='/predict', status='200'
        ).inc()

        if prediction == 1:
            POSITIVE_PREDICTIONS.inc()
        else:
            NEGATIVE_PREDICTIONS.inc()

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "diagnosis": "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
            "processing_time_ms": round(latency * 1000, 2),
        })

    except Exception as e:
        REQUEST_COUNT.labels(
            method='POST', endpoint='/predict', status='500'
        ).inc()
        return jsonify({"error": str(e)}), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route('/model/info', methods=['GET'])
def model_info():
    """Info model yang sedang berjalan."""
    if not MODEL_LOADED:
        return jsonify({"error": "Model tidak ter-load"}), 503
    return jsonify({
        "model_dir": MODEL_DIR,
        "signatures": list(model.signatures.keys()),
        "status": "ready",
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
