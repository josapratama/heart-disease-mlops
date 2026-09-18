"""
Transform module untuk Heart Disease pipeline.
Berisi fungsi preprocessing dan feature engineering.
"""

import tensorflow as tf
import tensorflow_transform as tft

# Nama fitur numerik
NUMERICAL_FEATURES = [
    'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca'
]

# Nama fitur kategorikal
CATEGORICAL_FEATURES = {
    'sex': 2,
    'cp': 4,
    'fbs': 2,
    'restecg': 3,
    'exang': 2,
    'slope': 3,
    'thal': 4,
}

# Label
LABEL_KEY = 'target'


def transformed_name(key: str) -> str:
    """Mengembalikan nama fitur setelah transformasi."""
    return key + '_xf'


def preprocessing_fn(inputs: dict) -> dict:
    """
    Fungsi preprocessing menggunakan TensorFlow Transform.

    Args:
        inputs: Dictionary berisi raw features dari dataset.

    Returns:
        Dictionary berisi fitur yang sudah ditransformasi.
    """
    outputs = {}

    # Normalisasi fitur numerik menggunakan z-score
    for feature_name in NUMERICAL_FEATURES:
        outputs[transformed_name(feature_name)] = tft.scale_to_z_score(
            tf.cast(inputs[feature_name], tf.float32)
        )

    # One-hot encoding fitur kategorikal
    for feature_name, num_buckets in CATEGORICAL_FEATURES.items():
        outputs[transformed_name(feature_name)] = tft.compute_and_apply_vocabulary(
            tf.strings.as_string(inputs[feature_name]),
            top_k=num_buckets
        )

    # Pass-through label
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
