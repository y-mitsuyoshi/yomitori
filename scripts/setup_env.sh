#!/bin/bash
# scripts/setup_env.sh
# Docker ベースの環境構築
set -eu

cd "$(dirname "$0")/.."

# Docker の確認
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not installed. Install Docker Engine + docker compose plugin."
    exit 1
fi

if docker compose version &> /dev/null; then
    COMPOSE="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE="docker-compose"
else
    echo "ERROR: Docker Compose not found."
    exit 1
fi

echo "Docker detected: $(docker --version)"
echo "Compose: $(${COMPOSE} version)"

# GPU が Docker から見えるか確認
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
    echo "Testing GPU access inside Docker..."
    if docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo "GPU access OK"
    else
        echo "WARNING: NVIDIA Container Toolkit may not be configured."
    fi
else
    echo "WARNING: nvidia-smi not found."
fi

# ベースイメージをビルド（初回は時間がかかります）
echo ""
echo "=== Building base image (初回は10〜15分かかります) ==="
$COMPOSE build base

# 各サービスイメージをビルド（ベースイメージがあれば高速）
echo ""
echo "=== Building service images ==="
$COMPOSE build serve
$COMPOSE build train
$COMPOSE build test

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  推論サーバー起動:   docker compose up serve"
echo "  推論リクエスト送信: bash scripts/infer_client.sh data/samples/sample_license.jpg"
echo "  テスト実行:         docker compose run --rm test"
echo "  合成データ生成:     bash scripts/docker_generate.sh 10"
echo "  ファインチューニング: bash scripts/docker_train.sh 1 4"
echo "  ノートブック:       docker compose up notebook"