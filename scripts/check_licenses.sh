#!/bin/bash
# scripts/check_licenses.sh
# AGPL/GPL混入をチェックする（Docker コンテナ内またはローカルで実行可能）
set -eu

cd "$(dirname "$0")/.."

# コンテナ内で実行されている場合は直接実行
if [ -f /.dockerenv ] || ! command -v docker &> /dev/null; then
    _run_local
    exit 0
fi

# ホストから実行 → Docker 経由
if docker compose version &> /dev/null; then
    COMPOSE="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE="docker-compose"
else
    echo "ERROR: Docker Compose not found. Run inside container or install Docker."
    exit 1
fi

$COMPOSE run --rm check-licenses
exit 0

_run_local() {
    if ! command -v pip-licenses &> /dev/null; then
        echo "Installing pip-licenses..."
        pip install pip-licenses
    fi

    OUTPUT=$(pip-licenses --with-license-file --format=plain 2>/dev/null || true)

    if echo "$OUTPUT" | grep -qiE '(AGPL|GPL)'; then
        echo "ERROR: AGPL/GPL licensed packages found:"
        echo "$OUTPUT" | grep -iE '(AGPL|GPL)' || true
        exit 1
    fi

    echo "License check passed: no AGPL/GPL found"
}