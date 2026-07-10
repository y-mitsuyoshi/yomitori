#!/bin/bash
# scripts/docker_train.sh
# Docker コンテナ内で TrOCR ファインチューニングを実行
# Usage: bash scripts/docker_train.sh [epochs] [batch_size]
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

EPOCHS="${1:-5}"
BATCH="${2:-8}"

echo "=== TrOCR Fine-tuning (epochs=$EPOCHS, batch=$BATCH) ==="
$COMPOSE run --rm train \
    python -m training.train_trocr \
    --data_dir /opt/ml/code/data/synthetic/driver_license/ \
    --output_dir /opt/ml/model \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH"

echo ""
echo "Training complete. Model saved to Docker volume 'yomitori-models'."
echo "To run inference: bash scripts/run_local_test.sh"