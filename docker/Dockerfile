# Unified MLPlatform image for Vertex AI (training + prediction) and Dataproc PySpark.
#
# Build:
#   docker build -f docker/Dockerfile -t mlplatform:latest .
#
# Run modes:
#   Vertex AI Training  :  docker run mlplatform train --dag /app/example_model/pipeline/dags/train.yaml
#   Vertex AI Prediction:  docker run mlplatform serve --dag /app/example_model/pipeline/dags/predict.yaml
#   Spark (direct)      :  docker run mlplatform spark --config run_config.json --input-path data.csv
#   Dataproc            :  (entrypoint overridden by Dataproc — image provides Python environment only)
#
# For Dataproc Serverless, override the base image:
#   docker build --build-arg BASE_IMAGE=us-docker.pkg.dev/cloud-dataproc/container-images/pyspark:<tag> ...

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ARG MODEL_PACKAGE=example_model

# ── system deps (Java for PySpark, procps for health checks) ──────────────
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        procps \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# ── framework: install mlplatform (changes infrequently → cached layer) ───
COPY mlplatform/ /build/mlplatform/
RUN pip install --no-cache-dir \
        /build/mlplatform/[parquet] \
        pyspark==3.5.1 \
        google-cloud-storage>=2.0 \
    && rm -rf /build

# ── model code and its deps (changes per DS iteration) ───────────────────
WORKDIR /app
COPY ${MODEL_PACKAGE}/ /app/${MODEL_PACKAGE}/
RUN if [ -f /app/${MODEL_PACKAGE}/requirements.txt ]; then \
        pip install --no-cache-dir -r /app/${MODEL_PACKAGE}/requirements.txt; \
    fi

# ── entrypoint ────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
