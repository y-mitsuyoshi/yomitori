# 段階的（インクリメンタル）追加学習 & 推論テスト手順書

本手順書は、PCのディスク空き容量を節約しながら段階的に学習データを入れ替え、精度を段階的に向上させていく手順と、学習した各モデルバージョンをテストするコマンドをまとめたものです。

---

## 🏛️ 全体フローの概要

1. **データ一時保存による容量節約**: 
   学習用データは1回あたり **20,000行画像** だけ生成し、学習が完了した瞬間に自動的に画像を全削除して空き容量を戻します。
2. **モデル重みの引き継ぎ（継続学習）**:
   前回のステップの学習済みモデル（例: `v2.1`）を読み込んで次の学習を開始するため、短時間で効率よく精度が上昇します。
3. **モデルの命名規則**:
   起点となるメジャーバージョン（例: `v2`）に対して、小数点刻みのマイナーバージョン（`v2.1`, `v2.2`, `v2.3`, ...）で段階的に出力されます。
4. **自動ディスク管理**:
   中間モデルは自動削除され、常に「起点モデル（v2.1）」と「最新モデル」のみが保持されます。

---

## ⚠️ 重要な技術的注意事項

### 継続学習の安全機構

`training/train_trocr.py` には、継続学習時にモデルが崩壊するのを防ぐ安全機構が組み込まれています。

- **判定方法**: ベースモデルのディレクトリに `training_info.json` が存在する場合、継続学習と判定
- **継続学習時の動作**:
  - 保存済みトークナイザーをそのまま使用（語彙リサイズをスキップ）
  - 学習率を自動で `1e-5` に設定（`--learning_rate` を省略した場合）
- **初回学習時の動作**:
  - 新しいトークナイザーに差し替え + 語彙リサイズを実行
  - 学習率を自動で `5e-5` に設定

> **注意**: `--learning_rate` をスクリプトで明示的に指定すると自動設定が上書きされます。
> 通常は省略して自動設定に任せてください。

---

## 🏃‍♂️ 段階的追加学習の実行手順

### 前提条件

- Docker Compose が利用可能であること
- GPU (CUDA) が使えるコンテナ環境が構築済みであること
- 起点モデル `/opt/ml/model/v2.1` が存在すること

### Step 1: スクリプトの設定確認

[scripts/run_incremental_training.sh](file:///home/yuma/projects/yomitori/scripts/run_incremental_training.sh) の上部パラメータを確認します。

```bash
ITERATIONS=5                   # 繰り返す回数
COUNT_REGULAR=1600             # 通常ドキュメント数
COUNT_KANJI=400                # 漢字ブーストドキュメント数
EPOCHS=3                       # 学習エポック数
BATCH_SIZE=4                   # バッチサイズ
START_SEED=2000                # シード値

CURRENT_MODEL="/opt/ml/model/v2.1"  # 起点モデル
VERSION_PREFIX="2"
START_STEP_INDEX=2                   # v2.2 から出力開始
```

### Step 2: 不要な崩壊モデルの削除（初回のみ）

過去の崩壊モデル（v2.2〜v2.7）が残っている場合は削除してディスク容量を回収します。

```bash
docker compose run --rm dev bash -c "rm -rf /opt/ml/model/v2.{2,3,4,5,6,7}"
```

### Step 3: 古い評価データの削除（初回のみ）

以前の巨大な評価セット（10,000枚）が残っている場合は削除します。新しい評価セット（1,000枚）が自動生成されます。

```bash
rm -rf data/synthetic/eval_set
```

### Step 4: 学習の実行

```bash
chmod +x scripts/run_incremental_training.sh  # 初回のみ
./scripts/run_incremental_training.sh
```

**所要時間**: 1イテレーションあたり約2時間（データ生成15分 + 学習1.5時間 + 評価5分）

### 学習が正常かどうかの確認方法

最初のイテレーション開始後、ログに以下が表示されることを確認してください。

#### ✅ 正常（継続学習として検出されている）
```
Continuing from local model — using saved tokenizer (no vocab resize)
Auto learning_rate: 1e-05 (continuation=True)
```

#### ❌ 異常（毎回リサイズが走っている）
```
Auto learning_rate: 5e-05 (continuation=False)
```
この場合は `Ctrl+C` で停止し、設定を確認してください。

#### ✅ Lossの正常な推移
```
epoch 0.1: loss=2.1 → epoch 0.5: loss=1.5 → epoch 1.0: loss=1.2
```
Lossが1エポック内で明確に下がっていれば正常です。

#### ❌ Lossの異常な推移
```
epoch 0.1: loss=9.3 → epoch 0.5: loss=9.2 → epoch 1.0: loss=9.1
```
Lossが9付近から下がらない場合は崩壊しています。`Ctrl+C` で停止してください。

---

## 🔄 2回目以降の実行

スクリプトは前回の最終モデルを起点に自動で引き継ぎます。
パラメータの更新が必要です。

### 例: v2.6 まで完了した後、v2.7〜v2.11 を追加学習する場合

```bash
# scripts/run_incremental_training.sh の設定を更新
ITERATIONS=5
START_SEED=7000                        # ★シードを変更（データの重複防止）
CURRENT_MODEL="/opt/ml/model/v2.6"     # ★前回の最終モデルを指定
START_STEP_INDEX=7                     # ★v2.7 から出力開始
```

```bash
./scripts/run_incremental_training.sh
```

---

## 🌱 シード値（START_SEED）のルールと上限

データの重複を防ぐためのシード値（`START_SEED`）の指定ルールです。

### 推奨ルール
**`START_SEED` ＝ `START_STEP_INDEX` × `1000`**

| ステップ | START_SEED | 出力モデル |
| :--- | :--- | :--- |
| v2.2〜v2.6（ステップ2） | 2000 | v2.2, v2.3, ..., v2.6 |
| v2.7〜v2.11（ステップ7） | 7000 | v2.7, v2.8, ..., v2.11 |
| v2.12〜v2.16（ステップ12） | 12000 | v2.12, v2.13, ..., v2.16 |

---

## 🧪 バージョンごとの推論テスト方法

学習が完了した各モデルバージョンにサンプル画像を通し、文字認識の精度を確認する手順です。

### 手順 1: テスト対象のモデルを指定して推論サーバーを起動

```bash
# v2.2 モデルをテストする場合
docker compose run --rm -p 8080:8080 -e YOMITORI_MODEL_DIR=/opt/ml/model/v2.2 serve
```

### 手順 2: 別ターミナルからテストリクエストを送信

```bash
curl -s -X POST http://localhost:8080/invocations \
    -H "Content-Type: image/jpeg" \
    --data-binary @data/samples/sample_license.jpg | python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))"
```

### 手順 3: サーバーの停止
テストが終了したら、サーバーを実行しているターミナルで **`Ctrl + C`** を押してコンテナを停止します。

---

## 💾 成果物（モデル）のエクスポート

特定のバージョンが高精度になり、バックアップやデプロイを行いたい場合：

```bash
# v2.6 をエクスポートする場合
docker compose run --rm dev bash -c "cd /opt/ml/model/v2.6 && tar czf /opt/ml/code/model_v2.6.tar.gz --exclude='checkpoint-*' ."
```

* 生成されるファイル: `model_v2.6.tar.gz` (サイズ: 約 500〜700 MB)

---

## 🧹 ディスク容量管理

### 自動管理（スクリプトに組み込み済み）
- **一時学習データ**: 各イテレーション終了後に自動削除
- **チェックポイント**: 各イテレーション終了後に自動削除
- **中間モデル**: 新しいモデルの保存後、1つ前の中間モデルを自動削除（起点モデル v2.1 は常に保持）

### 手動クリーンアップ（必要に応じて）
```bash
# Docker ボリューム内の不要モデルを確認
docker compose run --rm dev ls -la /opt/ml/model/

# 特定のモデルを削除
docker compose run --rm dev bash -c "rm -rf /opt/ml/model/v2.X"
```
