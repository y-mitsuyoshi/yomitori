# Agentic Pod 開発

このプロジェクトは `.shared-agents/` に共有エージェントプロンプトを持つマルチエージェント開発システムを使用しています。

## 利用可能なエージェント

| エージェント | 役割 | コマンド |
|-------------|------|---------|
| architect | アーキテクチャ・技術設計 | `claude --agent architect` |
| architect-reviewer | アーキテクチャレビュー | `claude --agent architect-reviewer` |
| implementer | コード実装 | `claude --agent implementer` |
| implementer-reviewer | 実装レビュー | `claude --agent implementer-reviewer` |
| doc-reviewer | ドキュメントレビュー | `claude --agent doc-reviewer` |
| security-reviewer | セキュリティレビュー | `claude --agent security-reviewer` |
| uiux-implementer | UI/UX実装 | `claude --agent uiux-implementer` |
| uiux-reviewer | UI/UXレビュー | `claude --agent uiux-reviewer` |
| qa-engineer | QA・テスト自動化 | `claude --agent qa-engineer` |
| tech-lead | 技術統括 | `claude --agent tech-lead` |
| data-analyst | データ分析 | `claude --agent data-analyst` |
| prd-manager | 製品要件定義 | `claude --agent prd-manager` |
| sre | インフラ運用 | `claude --agent sre` |
| project-manager | プロジェクト管理 | `claude --agent project-manager` |

## アーキテクチャ

- エージェントプロンプトは `.shared-agents/prompts/` に格納 (単一真実源)
- `.claude/rules/` には共有プロンプトへのシンボリックリンクが含まれている
- 全エージェントは YAGNI、プラグイン型アーキテクチャ、自己修正の原則に従う
- 詳細は `.shared-agents/README.md` を参照

## プロジェクト規約
- **プラグイン型アーキテクチャ**: 新書類対応は `configs/document_types/` への YAML 設定と `src/document_types/` へのクラス追加のみ。コアパイプライン（`src/pipeline/` など）は変更しない。
- **ライセンス制限**: AGPL/GPL ライブラリ（PaddleOCR等）は使用禁止。`scripts/check_licenses.sh` を使用して確認。
- **Python パッケージ構成**: ロジックは `src/` に配置、CLI エントリポイントは `src/cli.py`。
- **テスト**: テストは `pytest` を使用し、すべての新規ロジックやフィールド定義に対して `tests/` 内にテストを作成。
- **自己修正**: 完了宣言前にテストを実行し、エラーがあれば修正する (最大3回試行)。

