#!/bin/bash
# scripts/docker_build.sh
# 全 Docker イメージをビルド
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

echo "=== Building all Yomitori Docker images ==="
$COMPOSE build
echo ""
echo "All images built successfully:"
docker images | grep yomitori || true