#!/bin/bash
# scripts/docker_test.sh
# Docker コンテナ内でユニットテストを実行
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

echo "=== Running tests in Docker ==="
$COMPOSE run --rm test python -m pytest tests/ -v "$@"