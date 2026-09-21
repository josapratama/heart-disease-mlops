FROM tensorflow/serving:2.13.0

COPY serving/model/heart_disease_model /models/heart_disease_model
COPY serving/monitoring_config.txt /model_config/prometheus.config

ENV MODEL_NAME=heart_disease_model
ENV MODEL_BASE_PATH=/models
ENV MONITORING_CONFIG="/model_config/prometheus.config"
ENV PORT=8501
ENV TF_CPP_MIN_LOG_LEVEL=0

RUN echo '#!/bin/bash \n\n\
env \n\
tensorflow_model_server --port=8500 --rest_api_port=${PORT} \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH} \
--monitoring_config_file=${MONITORING_CONFIG} \
--enable_batching=false \
--per_process_gpu_memory_fraction=0 \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh
