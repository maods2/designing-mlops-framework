#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# MLPlatform unified container entrypoint
#
# Execution modes (set via first arg or MLPLATFORM_MODE env var):
#
#   train  — Run a training workflow          (Vertex AI Custom Training)
#   serve  — Start FastAPI prediction server  (Vertex AI Prediction)
#   spark  — Run spark/main.py directly       (manual Spark invocation)
#   *      — Pass-through to exec "$@"        (Dataproc overrides entrypoint)
#
# Env vars:
#   MLPLATFORM_MODE     Override mode without positional arg
#   MLPLATFORM_PROFILE  Profile name (defaults: cloud-train for train, cloud-online for serve)
# ---------------------------------------------------------------------------

MODE="${MLPLATFORM_MODE:-${1:-}}"

# If first positional arg is a known mode, consume it
case "${1:-}" in
    train|serve|spark) shift ;;
esac

case "$MODE" in
    train)
        PROFILE="${MLPLATFORM_PROFILE:-cloud-train}"
        echo "[mlplatform] Training mode | profile=$PROFILE" >&2
        exec mlplatform run --profile "$PROFILE" "$@"
        ;;

    serve)
        PROFILE="${MLPLATFORM_PROFILE:-cloud-online}"
        echo "[mlplatform] Serving mode | profile=$PROFILE" >&2
        exec mlplatform run --profile "$PROFILE" "$@"
        ;;

    spark)
        echo "[mlplatform] Spark mode" >&2
        exec python -m mlplatform.spark.main "$@"
        ;;

    *)
        # Pass-through: Dataproc (and other orchestrators) override the
        # container command entirely.  When they do, $@ contains their own
        # driver/executor invocation, so we just exec it.
        if [ $# -gt 0 ]; then
            exec "$@"
        fi

        echo "[mlplatform] No mode specified."
        echo ""
        echo "Usage:"
        echo "  docker run <image> train --dag <dag.yaml> [--version V] [--base-path PATH]"
        echo "  docker run <image> serve --dag <dag.yaml> [--base-path PATH]"
        echo "  docker run <image> spark --config <config.json> [--input-path PATH]"
        echo ""
        echo "Environment variables:"
        echo "  MLPLATFORM_MODE     train | serve | spark"
        echo "  MLPLATFORM_PROFILE  Profile name override"
        exit 1
        ;;
esac
