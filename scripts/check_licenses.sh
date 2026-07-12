#!/bin/bash
# scripts/check_licenses.sh — ライセンスチェック用スクリプト
set -e

echo "=== Running License Checker inside Docker ==="
# docker compose run で base コンテナ上で pip-licenses を実行し、GPLやAGPL等の制限ライセンスがないか検証する
docker compose run --rm base pip-licenses \
  --fail-on GPL,AGPL,LGPL \
  --ignore-packages pip-licenses \
  --summary

echo "=== License Check Passed ==="
