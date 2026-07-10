#!/bin/bash
# scripts/infer_client.sh
# 推論サーバー（serve）に対して画像を送信してOCR結果を取得する
#
# 前提: docker compose up serve でサーバーが起動していること
#
# Usage:
#   bash scripts/infer_client.sh <image_path> [document_type]
#
# Example:
#   bash scripts/infer_client.sh data/samples/sample_license.jpg
#   bash scripts/infer_client.sh data/samples/sample_license.jpg driver_license_front
set -eu

cd "$(dirname "$0")/.."

IMAGE="${1:-data/samples/sample_license.jpg}"
DOC_TYPE="${2:-}"

if [ ! -f "$IMAGE" ]; then
    echo "ERROR: Image not found: $IMAGE"
    echo "Usage: bash scripts/infer_client.sh <image_path> [document_type]"
    exit 1
fi

# サーバーのヘルスチェック
if ! curl -sf http://localhost:8080/ping > /dev/null 2>&1; then
    echo "ERROR: Server is not running. Start it first:"
    echo "  docker compose up serve"
    exit 1
fi

# 画像をbase64エンコード
B64=$(base64 -w 0 "$IMAGE")

if [ -n "$DOC_TYPE" ]; then
    PAYLOAD=$(printf '{"image":"%s","document_type":"%s"}' "$B64" "$DOC_TYPE")
else
    PAYLOAD=$(printf '{"image":"%s"}' "$B64")
fi

# 推論リクエスト送信
curl -s http://localhost:8080/invocations \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" | python3 -m json.tool 2>/dev/null || \
curl -s http://localhost:8080/invocations \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD"