#!/bin/bash
# scripts/run_sagemaker_local.sh
# SageMaker Local Mode 実行（Docker ベース）
set -eu

cd "$(dirname "$0")/.."

if docker compose version &> /dev/null; then
    COMPOSE="docker compose"
elif command -v docker-compose &> /dev/null; then
    echo "ERROR: Docker Compose not found."
    exit 1
fi

# dev イメージ内で SageMaker Local Mode スクリプトを実行
# SageMaker Local Mode は内部で追加の Docker コンテナを起動するため
# Docker-in-Docker または ソケットマウントが必要
echo "=== SageMaker Local Mode: Training ==="
$COMPOSE run --rm -v /var/run/docker.sock:/var/run/docker.sock dev \
    python -m sagemaker.local_train

echo "=== SageMaker Local Mode: Deploy & Test ==="
$COMPOSE run --rm -v /var/run/docker.sock:/var/run/docker.sock dev \
    python -m sagemaker.local_deploy "$@"