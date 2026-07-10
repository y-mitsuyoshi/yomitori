#!/usr/bin/env bash
set -euo pipefail

# Claude Code 経由でエージェントを実行
# 使い方: .shared-agents/harness/claude.sh <agent-name> "<task-description>"

readonly AGENT_NAME="${1:?"使い方: claude.sh <agent-name> <task-description>"}"
readonly TASK="${2:?"使い方: claude.sh <agent-name> <task-description>"}"

if ! command -v claude &>/dev/null; then
  echo "エラー: claude CLI が見つかりません。"
  exit 1
fi

exec claude --allowedTools "Bash, Edit, Read, Glob, Grep, Write, WebFetch" \
  --agent "${AGENT_NAME}" "${TASK}"
