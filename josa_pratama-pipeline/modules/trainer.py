"""
Trainer module untuk Heart Disease pipeline.
Berisi fungsi pembuatan model dan training.
Kompatibel dengan TFX 1.21.0 + TensorFlow 2.21.0.
"""

import os
from typing import List

import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

# ──────────────────────────────────────────────
# Konstanta fitur
# ──────────────────────────────────────────────
NUMERICAL_FEATURES = [
    'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca'
]

CATEGORICAL_FEATURES = {
    'sex': 2,
    'cp': 4,
    'fbs': 2,
    'restecg': 3,
    'exang': 2,
    'slope': 3,
    'thal': 4,
}

LABEL_KEY = 'target'


def transformed_name(key: str) -> str:
    """Mengembalikan nama fitur setelah transformasi."""
    return key + '_xf'


# ──────────────────────────────────────────────
# Input function
# ──────────────────────────────────────────────
def _gzip_reader_fn(filenames):
    """Membaca TFRecord yang dikompresi GZIP."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _input_fn(file_pattern: List[str],
              tf_transform_output: tft.TFTransformOutput,
              num_epochs: int,
              batch_size: int = 64) -> tf.data.Dataset:
    """
    Membuat dataset untuk training atau evaluasi.

    Args:
        file_pattern: Pola path untuk file TFRecord.
        tf_transform_output: Output dari TFTransform.
        num_epochs: Jumlah epoch.
        batch_size: Ukuran batch.

    Returns:
        tf.data.Dataset.
    """
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    # TF 2.x: make_batched_features_dataset pindah dari experimental ke stable
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )

    return dataset


# ──────────────────────────────────────────────
# Serving function
# ──────────────────────────────────────────────
def _get_serve_tf_examples_fn(model, tf_transform_output):
    """
    Membuat fungsi serving yang menerima serialized tf.Example.

    Args:
        model: Model Keras yang sudah ditraining.
        tf_transform_output: Output dari TFTransform.

    Returns:
        tf.ConcreteFunction untuk serving signature.
    """
    # Simpan transform layer sebagai atribut model agar ikut di-save
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_examples):
        """Menerima raw tf.Example bytes dan mengembalikan prediksi."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY, None)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec
        )
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features, training=False)

    return serve_tf_examples_fn


# ──────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────
def _build_keras_model(hp) -> tf.keras.Model:
    """
    Membangun model Keras Functional untuk klasifikasi penyakit jantung.

    Args:
        hp: Dict hyperparameter atau HyperParameters object dari Keras Tuner.

    Returns:
        Model Keras yang sudah dikompilasi.
    """
    # Helper: ambil nilai HP dari dict atau HyperParameters object
    def get_hp(name, default):
        if isinstance(hp, dict):
            return hp.get(name, default)
        # HyperParameters object — gunakan getattr pada values dict
        try:
            val = hp.get(name)
            return val if val is not None else default
        except Exception:
            return default
    # Input layers — satu per fitur agar kompatibel dengan TFT feature spec
    input_features = []
    for feature in NUMERICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(feature))
        )
    for feature in CATEGORICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(feature))
        )

    # Concatenate semua input
    concatenated = tf.keras.layers.concatenate(input_features)

    # Hidden layer 1
    x = tf.keras.layers.Dense(
        units=get_hp('units_1', 128),
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.L2(0.001),
    )(concatenated)
    x = tf.keras.layers.Dropout(rate=get_hp('dropout_1', 0.3))(x)

    # Hidden layer 2
    x = tf.keras.layers.Dense(
        units=get_hp('units_2', 64),
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.L2(0.001),
    )(x)
    x = tf.keras.layers.Dropout(rate=get_hp('dropout_2', 0.2))(x)

    # Hidden layer 3
    x = tf.keras.layers.Dense(
        units=get_hp('units_3', 32),
        activation='relu',
    )(x)

    # Output layer — binary classification
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input_features, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=get_hp('learning_rate', 0.001)
        ),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ],
    )
    model.summary()
    return model


# ──────────────────────────────────────────────
# Tuner function
# ──────────────────────────────────────────────
def tuner_fn(fn_args: FnArgs):
    """
    Fungsi untuk Tuner component – hyperparameter tuning dengan Keras Tuner.

    Args:
        fn_args: FnArgs object dari TFX Tuner component.

    Returns:
        TunerFnResult dengan tuner dan fit_kwargs.
    """
    import keras_tuner as kt
    from tfx.components.tuner.component import TunerFnResult

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = _input_fn(
        fn_args.train_files,
        tf_transform_output,
        num_epochs=3,
        batch_size=64,
    )
    eval_dataset = _input_fn(
        fn_args.eval_files,
        tf_transform_output,
        num_epochs=1,
        batch_size=64,
    )

    def build_model(hp):
        hyperparams = {
            'units_1':      hp.Int('units_1',   min_value=64,  max_value=256, step=64),
            'units_2':      hp.Int('units_2',   min_value=32,  max_value=128, step=32),
            'units_3':      hp.Int('units_3',   min_value=16,  max_value=64,  step=16),
            'dropout_1':    hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1),
            'dropout_2':    hp.Float('dropout_2', min_value=0.1, max_value=0.4, step=0.1),
            'learning_rate': hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]),
        }
        return _build_keras_model(hyperparams)

    tuner = kt.RandomSearch(
        build_model,
        objective=kt.Objective('val_auc', direction='max'),
        max_trials=5,
        executions_per_trial=1,
        directory=fn_args.working_dir,
        project_name='heart_disease_tuning',
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x':                train_dataset,
            'validation_data':  eval_dataset,
            'steps_per_epoch':  fn_args.train_steps,
            'validation_steps': fn_args.eval_steps,
        },
    )


# ──────────────────────────────────────────────
# Run function (dipanggil oleh Trainer component)
# ──────────────────────────────────────────────
def run_fn(fn_args: FnArgs):
    """
    Fungsi utama training yang dipanggil oleh TFX Trainer component.

    Args:
        fn_args: FnArgs object berisi path, files, dan hyperparameter.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = _input_fn(
        fn_args.train_files,
        tf_transform_output,
        num_epochs=10,
        batch_size=64,
    )
    eval_dataset = _input_fn(
        fn_args.eval_files,
        tf_transform_output,
        num_epochs=1,
        batch_size=64,
    )

    # Gunakan hyperparameter dari Tuner jika tersedia, fallback ke default dict
    if fn_args.hyperparameters:
        hp = fn_args.hyperparameters
    else:
        hp = {
            'units_1':       128,
            'units_2':       64,
            'units_3':       32,
            'dropout_1':     0.3,
            'dropout_2':     0.2,
            'learning_rate': 0.001,
        }

    model = _build_keras_model(hp)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=fn_args.model_run_dir,
            histogram_freq=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=5,
            restore_best_weights=True,
            mode='max',
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=3,
            mode='max',
        ),
    ]

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=callbacks,
        epochs=30,
    )

    # ── Simpan model dalam format SavedModel dengan serving signature ──
    serving_fn = _get_serve_tf_examples_fn(model, tf_transform_output)

    # Tambahkan juga signature transform_features untuk TFMA Evaluator
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def transform_features_fn(serialized_tf_examples):
        """Signature khusus untuk TFMA — mengembalikan transformed features."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY, None)
        parsed = tf.io.parse_example(serialized_tf_examples, feature_spec)
        return model.tft_layer(parsed)

    tf.saved_model.save(
        model,
        fn_args.serving_model_dir,
        signatures={
            'serving_default':    serving_fn,
            'transform_features': transform_features_fn,
        },
    )
