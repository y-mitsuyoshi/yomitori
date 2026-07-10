#!/bin/bash
# scripts/setup_env.sh
# Docker ベースの環境構築
# ローカルに Python 環境を作らず、Docker イメージをビルドして使用する
set -eu

cd "$(dirname "$0")/.."

# Docker の確認
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not installed. Install Docker Engine + docker compose plugin."
    exit 1
fi

# Docker Compose の確認 (v2 plugin or v1 standalone)
if docker compose version &> /dev/null; then
    COMPOSE="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE="docker-compose"
else
    echo "ERROR: Docker Compose not found. Install 'docker compose' plugin."
    exit 1
fi

echo "Docker detected: $(docker --version)"
echo "Compose: $(${COMPOSE} version)"

# NVIDIA GPU が Docker から見えるか確認
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
    echo "Testing GPU access inside Docker..."
    if docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo "GPU access OK"
    else
        echo "WARNING: NVIDIA Container Toolkit may not be configured."
        echo "  Install nvidia-container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi
else
    echo "WARNING: nvidia-smi not found. GPU features will be unavailable."
    echo "  CPU-only mode is possible but significantly slower."
fi

# Docker イメージをビルド
echo ""
echo "=== Building Docker images ==="
$COMPOSE build dev
$COMPOSE build test
echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  Dev shell:      $COMPOSE run --rm dev"
echo "  Run tests:      $COMPOSE run --rm test"
echo "  Generate data:  $COMPOSE run --rm generate"
echo "  Train:          $COMPOSE run --rm train"
echo "  Inference:      $COMPOSE run --rm infer"
echo "  Notebook:       $COMPOSE up notebook"
echo "  Build all:      $COMPOSE build"