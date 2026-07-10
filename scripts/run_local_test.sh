#!/bin/bash
# scripts/run_local_test.sh
# Docker ベースのローカル推論テスト
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

SAMPLE="${1:-data/samples/sample_license.jpg}"
ARGS="${@:2}"

if [ ! -f "$SAMPLE" ]; then
    echo "ERROR: Sample image not found: $SAMPLE"
    echo "Place a sample image (mosaic-applied) at data/samples/sample_license.jpg"
    echo "Or specify a path: bash scripts/run_local_test.sh /path/to/image.jpg"
    exit 1
fi

# 推論コンテナで実行（--model_path でファインチューニング済みモデルを指定可能）
$COMPOSE run --rm infer \
    python -m src.cli infer \
    --image "/opt/ml/code/$SAMPLE" \
    --document_type driver_license_front \
    --model_path /opt/ml/model \
    $ARGS