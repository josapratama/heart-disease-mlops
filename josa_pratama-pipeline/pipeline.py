"""
Pipeline utama Heart Disease Classification menggunakan TFX.

Pipeline ini menggunakan Apache Beam sebagai orchestrator dan mencakup
seluruh komponen: ExampleGen, StatisticsGen, SchemaGen, ExampleValidator,
Transform, Tuner, Trainer, Resolver, Evaluator, dan Pusher.
"""

import os
from typing import Optional

import absl.logging
import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    Evaluator,
    ExampleValidator,
    Pusher,
    SchemaGen,
    StatisticsGen,
    Trainer,
    Transform,
    Tuner,
)
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.experimental.latest_blessed_model_resolver import (
    LatestBlessedModelResolver,
)
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import pusher_pb2, trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

# ──────────────────────────────────────────────
# Konstanta path
# ──────────────────────────────────────────────
PIPELINE_NAME = 'josa_pratama-pipeline'

# Root direktori proyek (relatif terhadap lokasi file ini)
_pipeline_root = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.path.join(_pipeline_root, 'data', 'raw')
TRANSFORM_MODULE = os.path.join(_pipeline_root, 'modules', 'transform.py')
TRAINER_MODULE = os.path.join(_pipeline_root, 'modules', 'trainer.py')

PIPELINE_ROOT = os.path.join(_pipeline_root, 'pipeline_output', PIPELINE_NAME)
METADATA_PATH = os.path.join(_pipeline_root, 'pipeline_output', 'metadata',
                              PIPELINE_NAME, 'metadata.db')
SERVING_MODEL_DIR = os.path.join(_pipeline_root, 'pipeline_output',
                                  'serving_model', PIPELINE_NAME)

# ──────────────────────────────────────────────
# Builder fungsi pipeline
# ──────────────────────────────────────────────

def create_pipeline(
    pipeline_name: str = PIPELINE_NAME,
    pipeline_root: str = PIPELINE_ROOT,
    data_root: str = DATA_ROOT,
    transform_module: str = TRANSFORM_MODULE,
    trainer_module: str = TRAINER_MODULE,
    training_steps: int = 1000,
    eval_steps: int = 150,
    serving_model_dir: str = SERVING_MODEL_DIR,
    metadata_path: str = METADATA_PATH,
    beam_pipeline_args: Optional[list] = None,
) -> pipeline.Pipeline:
    """
    Membuat TFX pipeline untuk klasifikasi penyakit jantung.

    Args:
        pipeline_name: Nama pipeline.
        pipeline_root: Root direktori output pipeline.
        data_root: Direktori data mentah.
        transform_module: Path ke modul transform.
        trainer_module: Path ke modul trainer.
        training_steps: Jumlah langkah training.
        eval_steps: Jumlah langkah evaluasi.
        serving_model_dir: Direktori untuk menyimpan model serving.
        metadata_path: Path ke SQLite metadata store.
        beam_pipeline_args: Argumen tambahan untuk Apache Beam.

    Returns:
        TFX Pipeline object.
    """
    if beam_pipeline_args is None:
        beam_pipeline_args = []

    # ── 1. ExampleGen ─────────────────────────────────────────
    # Membaca data CSV dan membagi menjadi train/eval set
    example_gen = CsvExampleGen(input_base=data_root)

    # ── 2. StatisticsGen ──────────────────────────────────────
    # Menghitung statistik deskriptif dari dataset
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs['examples']
    )

    # ── 3. SchemaGen ──────────────────────────────────────────
    # Membuat schema berdasarkan statistik data
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True,
    )

    # ── 4. ExampleValidator ───────────────────────────────────
    # Memvalidasi data menggunakan schema yang sudah dibuat
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'],
    )

    # ── 5. Transform ──────────────────────────────────────────
    # Preprocessing dan feature engineering
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=transform_module,
    )

    # ── 6. Tuner ──────────────────────────────────────────────
    # Hyperparameter tuning otomatis menggunakan Keras Tuner
    tuner = Tuner(
        module_file=trainer_module,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(splits=['train'], num_steps=500),
        eval_args=trainer_pb2.EvalArgs(splits=['eval'], num_steps=100),
    )

    # ── 7. Trainer ────────────────────────────────────────────
    # Training model dengan hyperparameter terbaik dari Tuner
    trainer = Trainer(
        module_file=trainer_module,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        hyperparameters=tuner.outputs['best_hyperparameters'],
        train_args=trainer_pb2.TrainArgs(splits=['train'], num_steps=training_steps),
        eval_args=trainer_pb2.EvalArgs(splits=['eval'], num_steps=eval_steps),
    )

    # ── 8. Resolver ───────────────────────────────────────────
    # Mendapatkan model terbaik yang sudah di-blessed sebelumnya
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id('latest_blessed_model_resolver')

    # ── 9. Evaluator ──────────────────────────────────────────
    # Mengevaluasi model baru dibandingkan dengan model sebelumnya
    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(
                signature_name='serving_default',
                label_key='target_xf',
                preprocessing_function_names=['transform_features'],
            )
        ],
        slicing_specs=[
            tfma.SlicingSpec(),                         # Overall
            tfma.SlicingSpec(feature_keys=['sex']),     # Per gender
        ],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(class_name='BinaryAccuracy'),
                tfma.MetricConfig(class_name='AUC'),
                tfma.MetricConfig(class_name='Precision'),
                tfma.MetricConfig(class_name='Recall'),
                tfma.MetricConfig(
                    class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.6}
                        ),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': -0.01},
                        ),
                    ),
                ),
            ])
        ],
    )

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config,
    )

    # ── 10. Pusher ────────────────────────────────────────────
    # Deploy model yang sudah divalidasi ke serving directory
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )

    # ── Susun komponen pipeline ────────────────────────────────
    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    ]

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        beam_pipeline_args=beam_pipeline_args,
    )


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == '__main__':
    absl.logging.set_verbosity(absl.logging.INFO)

    BeamDagRunner().run(
        create_pipeline()
    )
