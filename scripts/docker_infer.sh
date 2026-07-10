#!/bin/bash
# scripts/docker_infer.sh
# Docker コンテナ内で推論を実行
# Usage: bash scripts/docker_infer.sh <image_path> [extra args...]
set -eu

cd "$(dirname "$0")/.."

if docker compose version &> /dev/null; then
    COMPOSE="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE="docker-compose"
else
    echo "ERROR: Docker Compose not found."
    exit 1
fi

IMAGE="${1:-data/samples/sample_license.jpg}"
ARGS="${@:2}"

if [ ! -f "$IMAGE" ]; then
    echo "ERROR: Image not found: $IMAGE"
    exit 1
fi

echo "=== Running inference on $IMAGE ==="
$COMPOSE run --rm infer \
    python -m src.cli infer \
    --image "/opt/ml/code/$IMAGE" \
    --document_type driver_license_front \
    --model_path /opt/ml/model \
    $ARGS