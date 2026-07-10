#!/usr/bin/env bash
set -euo pipefail

# OpenCode 経由でエージェントを実行
# 使い方: .shared-agents/harness/opencode.sh <agent-name> "<task-description>"

readonly AGENT_NAME="${1:?"使い方: opencode.sh <agent-name> <task-description>"}"
readonly TASK="${2:?"使い方: opencode.sh <agent-name> <task-description>"}"

if ! command -v opencode &>/dev/null; then
  echo "エラー: opencode CLI が見つかりません。 https://opencode.ai からインストールしてください。"
  exit 1
fi

exec opencode --agent "${AGENT_NAME}" --prompt "${TASK}"
