# 共有エージェント (Agentic Pod ハーネス)

このディレクトリは全コーディングエージェント（opencode / claude code / antigravity）で共通利用する
エージェントプロンプト・ハーネススクリプト・テンプレートを管理します。

## アーキテクチャ

```
.shared-agents/              # 共通コンポーネント (symlink の実体)
├── prompts/                 # 各エージェントのシステムプロンプト (単一真実源)
├── harness/                 # ランナースクリプト (エージェント実行ラッパー)
└── templates/               # 出力テンプレート

.opencode/agents/            # -> symlink -> ../../.shared-agents/prompts/*.md
.claude/rules/               # -> symlink -> ../../.shared-agents/prompts/*.md
.antigravitycli/agents.json  # -> 共通プロンプトを参照する設定
```

## エージェント一覧

| エージェント | 役割 |
|---|---|
| `architect` | アーキテクチャ・技術設計 |
| `architect-reviewer` | アーキテクチャレビュー |
| `implementer` | 実装 |
| `implementer-reviewer` | 実装レビュー |
| `doc-reviewer` | ドキュメントレビュー |
| `security-reviewer` | セキュリティレビュー |
| `uiux-implementer` | UI/UX 実装 |
| `uiux-reviewer` | UI/UX レビュー |
| `qa-engineer` | QA・テスト自動化 |
| `tech-lead` | テックリード (技術統括) |
| `data-analyst` | データ分析・インサイト抽出 |
| `prd-manager` | PRD (製品要件定義) 作成 |
| `sre` | SRE・インフラ運用 |
| `project-manager` | プロジェクト管理・レポート |

## 使い方

### OpenCode
```bash
opencode --agent architect --prompt "タスク内容"
```

### Claude Code
```bash
claude --allowedTools --agent architect "タスク内容"
```

### Antigravity
```bash
antigravity run architect "タスク内容"
```

### ハーネススクリプト経由 (推奨)
```bash
.shared-agents/harness/run.sh <agent-name> "タスク内容"
```

## GitHub 上での symlink 表示

`.shared-agents/` 内のファイルを実体とし、各ツールの設定ディレクトリから
シンボリックリンクを貼ることで、GitHub 上では:

```
.opencode/agents/architect.md -> ../../.shared-agents/prompts/architect.md
```

のように表示され、クリックで実体に遷移できます。
`.gitattributes` によりクロスプラットフォームでも symlink が保持されます。
