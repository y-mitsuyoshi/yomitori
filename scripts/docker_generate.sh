#!/bin/bash
# scripts/docker_generate.sh
# Docker コンテナ内で合成データを生成
# Usage: bash scripts/docker_generate.sh [count] [output_dir]
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

COUNT="${1:-500}"
OUTPUT="${2:-/workspace/data/synthetic/driver_license/}"

echo "=== Generating $COUNT synthetic images → $OUTPUT ==="
$COMPOSE run --rm --no-deps dev \
    python -m training.generate_synthetic_data \
    --document_type driver_license_front \
    --count "$COUNT" \
    --output "$OUTPUT"
echo "Done."