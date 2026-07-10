#!/usr/bin/env bash
set -euo pipefail

# Antigravity 経由でエージェントを実行
# 使い方: .shared-agents/harness/antigravity.sh <agent-name> "<task-description>"

readonly AGENT_NAME="${1:?"使い方: antigravity.sh <agent-name> <task-description>"}"
readonly TASK="${2:?"使い方: antigravity.sh <agent-name> <task-description>"}"

readonly HARNESS_DIR="$(cd "$(dirname "$0")" && pwd)"
readonly PROMPT_FILE="${HARNESS_DIR}/../prompts/${AGENT_NAME}.md"

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "エラー: エージェントプロンプトが見つかりません: $PROMPT_FILE"
  exit 1
fi

if command -v agy &>/dev/null; then
  echo "[harness] agy でエージェント '${AGENT_NAME}' を実行中..."
  PROMPT_CONTENT=$(cat "$PROMPT_FILE")
  exec agy -p "System Prompt:
$PROMPT_CONTENT

Task:
$TASK"
elif command -v antigravity &>/dev/null; then
  echo "[harness] antigravity でエージェント '${AGENT_NAME}' を実行中..."
  exec antigravity run "${AGENT_NAME}" "${TASK}"
else
  echo "エラー: agy または antigravity CLI が見つかりません。"
  exit 1
fi

