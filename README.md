# Yomitori — 日本の身分証明書類向けMLベースOCRエンジン

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Amazon SageMaker へのデプロイを前提とした、日本の身分証明書類向け高精度 OCR エンジン。
**プラグイン型アーキテクチャ**により、新しい書類タイプを追加する際にコアパイプラインのコードを一切変更せずに済む設計。

**すべての実行環境は Docker コンテナとして提供されます。** ホストマシンに Python 環境を構築する必要はありません。

---

## 目次

- [特徴](#特徴)
- [アーキテクチャ](#アーキテクチャ)
- [必要環境](#必要環境)
- [セットアップ（Docker 環境構築）](#セットアップdocker-環境構築)
- [クイックスタート](#クイックスタート)
- [ローカルでの使い方（Docker Compose）](#ローカルでの使い方docker-compose)
  - [開発シェル（dev）](#開発シェルdev)
  - [合成データの生成（generate）](#合成データの生成generate)
  - [TrOCR のファインチューニング（train）](#trocr-のファインチューニングtrain)
  - [推論の実行（infer）](#推論の実行infer)
  - [ユニットテスト（test）](#ユニットテストtest)
  - [Jupyter Notebook（notebook）](#jupyter-notebooknotebook)
- [ファインチューニングの詳細](#ファインチューニングの詳細)
- [実際の手で検証する方法（ステップバイステップ）](#実際の手で検証する方法ステップバイステップ)
- [スクリプトリファレンス](#スクリプトリファレンス)
- [Docker イメージリファレンス](#docker-イメージリファレンス)
- [SageMaker Local Mode での検証](#sagemaker-local-mode-での検証)
- [クラウドへのデプロイ](#クラウドへのデプロイ)
- [ライセンスチェック](#ライセンスチェック)
- [新しい書類タイプの追加方法](#新しい書類タイプの追加方法)
- [出力フォーマット](#出力フォーマット)
- [技術スタック](#技術スタック)
- [トラブルシューティング](#トラブルシューティング)
- [リポジトリ構成](#リポジトリ構成)
- [ライセンス](#ライセンス)

---

## 特徴

- **プラグイン型アーキテクチャ**: 新しい書類は YAML 設定 + DocumentType クラスを追加するだけで対応。コアパイプラインは変更不要。
- **OpenCV 前処理**: ホモグラフィ変換（台形歪み補正）・大津の二値化・CLAHE コントラスト最適化
- **docTR 文字検出** (Apache 2.0): DB-ResNet50 による高精度テキスト行検出
- **TrOCR 文字認識** (MIT): Microsoft TrOCR の日本語ファインチューニング対応
- **ゾーンベース フィールド分類**: 正規化座標（0.0–1.0）でフィールドを特定。2段階判定（包含→最近傍）
- **後処理**: 全角→半角、和暦→ISO 変換、正規表現バリデーション、ホワイトリスト適用
- **完全 Docker 化**: すべての実行環境（開発・学習・推論・テスト）を Docker コンテナで提供。ホスト環境を汚さない
- **SageMaker 対応**: Local Mode → クラウド（ml.g5.xlarge 等）へのシームレス移行
- **ライセンス安全**: AGPL/GPL ライブラリ不使用（`scripts/check_licenses.sh` で自動チェック）

---

## アーキテクチャ

```
入力画像
  ↓
card_detector.py → アスペクト比・色・特徴から書類タイプを自動判定
  ↓
該当するDocumentType（ゾーン定義・バリデーションルール）を解決
  ↓
[共通] OpenCV前処理（ホモグラフィ補正・二値化）
  ↓
[共通] docTR文字領域検出
  ↓
[共通] TrOCR文字認識
  ↓
[書類別] ゾーンベースのフィールド分類（DocumentTypeのゾーン定義を使用）
  ↓
[書類別] バリデーション・正規化（DocumentTypeのルールを使用）
  ↓
構造化JSON出力
```

---

## 必要環境

| 項目 | 要件 |
|---|---|
| OS | Linux（Ubuntu 22.04 推奨 / WSL2 可） |
| Docker | Docker Engine 24+ および Docker Compose v2 plugin |
| GPU | NVIDIA RTX 5070 (VRAM 12GB) 以上を推奨 ※CPU-only でも前処理・テストは可能 |
| NVIDIA Driver | 550+ (CUDA 12.6+ 対応) |
| NVIDIA Container Toolkit | GPU を Docker コンテナから利用するために必要 |

### Docker のインストール

```bash
# Docker Engine + Compose plugin
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin

# NVIDIA Container Toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 動作確認

```bash
# Docker 確認
docker --version
docker compose version

# GPU が Docker から見えるか確認
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

---

## セットアップ（Docker 環境構築）

```bash
git clone <repository-url> yomitori
cd yomitori

# Docker イメージのビルド + 動作確認
bash scripts/setup_env.sh
```

`setup_env.sh` は以下を自動実行します:
1. Docker / Docker Compose / NVIDIA GPU の確認
2. 開発用イメージ (`yomitori:dev`) とテスト用イメージのビルド

完了後、以下のコマンドが利用可能になります:

| コマンド | 説明 |
|---|---|
| `docker compose up serve` | 推論サーバー起動 (http://localhost:8080) |
| `docker compose run --rm dev` | 開発シェルを起動 |
| `docker compose run --rm test` | ユニットテスト実行 |
| `docker compose run --rm generate` | 合成データ生成 |
| `docker compose run --rm train` | TrOCR ファインチューニング |
| `docker compose up notebook` | Jupyter Notebook 起動 (http://localhost:8888) |

---

## クイックスタート

```bash
# 1. セットアップ
bash scripts/setup_env.sh

# 2. テスト実行（GPU 不要）
docker compose run --rm test

# 3. 合成データ生成（10枚で動作確認）
bash scripts/docker_generate.sh 10

# 4. ファインチューニング（1エポックで動作確認）
bash scripts/docker_train.sh 1 4

# 5. サンプル画像で推論
#    data/samples/sample_license.jpg にモザイク済み画像を配置
bash scripts/run_local_test.sh data/samples/sample_license.jpg

# 6. ライセンスチェック
bash scripts/check_licenses.sh
```

---

## サンプル画像で推論をテストする

運転免許証の画像を使ってOCRエンジンの推論を検証する手順です。

### 前提

- `bash scripts/setup_env.sh` が完了していること
- ファインチューニング済みモデルがあること（`bash scripts/docker_train.sh 1 4` で生成済み）

### ステップ1: サンプル画像を配置

モザイク済みの運転免許証画像を `data/samples/` に配置します。

```bash
# 画像を配置（個人情報にモザイク処理を施すこと）
cp /path/to/your/license.jpg data/samples/sample_license.jpg
```

### ステップ2: 推論サーバーを起動

Docker コンテナで推論サーバーを常駐起動します。SageMaker と同じ `/ping`・`/invocations` エンドポイントを提供します。

```bash
docker compose up serve
```

起動後、以下のURLで待ち受けます:

- ヘルスチェック: `http://localhost:8080/ping`
- 推論API: `http://localhost:8080/invocations`

> ターミナルを占有するため、別のターミナルを開いて次のステップに進んでください。

### ステップ3: 推論リクエストを送信

別のターミナルで以下を実行します。

```bash
# スクリプト経由（簡単）
bash scripts/run_local_test.sh data/samples/sample_license.jpg

# または直接クライアントスクリプト
bash scripts/infer_client.sh data/samples/sample_license.jpg

# 書類タイプを明示指定する場合
bash scripts/infer_client.sh data/samples/sample_license.jpg driver_license_front
```

### ステップ4: curl で直接リクエスト（任意）

スクリプトを使わず curl で直接リクエストすることも可能です。

```bash
# ヘルスチェック
curl http://localhost:8080/ping

# 推論リクエスト（base64エンコードした画像を送信）
B64=$(base64 -w 0 data/samples/sample_license.jpg)
curl -s http://localhost:8080/invocations \
    -H "Content-Type: application/json" \
    -d "{\"image\":\"$B64\"}" | python3 -m json.tool
```

### ステップ5: サーバーを停止

推論が終わったらサーバーを停止します。

```bash
# サーバーを起動したターミナルで Ctrl+C
# または別ターミナルから
docker compose down
```

### レスポンスの例

```json
{
  "status": "success",
  "document_type": "driver_license_front",
  "fields": {
    "name": {
      "value": "山田太郎",
      "confidence": 0.95,
      "low_confidence": false
    },
    "birth_date": {
      "raw": "昭和61年5月1日生",
      "iso": "1986-05-01",
      "value": "1986-05-01",
      "confidence": 0.92,
      "low_confidence": false
    },
    "license_number": {
      "value": "010203040506",
      "confidence": 0.99,
      "low_confidence": false,
      "validation_passed": true
    }
  },
  "preprocessing": {
    "homography_applied": true,
    "corrected_image_size": [2400, 1512],
    "fallback_used": false
  },
  "warnings": []
}
```

### status の意味

| 値 | 意味 |
|---|---|
| `success` | 全フィールド正常認識・バリデーション通過 |
| `partial` | 一部フィールドが低信頼度 (confidence < 0.7) またはバリデーション不合格 |
| `failed` | 前処理失敗またはフィールドが1つも抽出できなかった |

### トラブルシューティング

| 現象 | 対処 |
|---|---|
| `Server is not running` | `docker compose up serve` を実行してサーバーを起動 |
| `Model not loaded` | ファインチューニングが未実行。`bash scripts/docker_train.sh 1 4` を実行 |
| `status: "failed"` | 前処理で輪郭検出失敗。ノートブックで前処理を確認: `docker compose up notebook` |
| 文字化け | ファインチューニング不足。学習データを増やして再学習 |
| `low_confidence: true` が多い | 学習エポック数を増やす (`bash scripts/docker_train.sh 10 8`) |

---

## ローカルでの使い方（Docker Compose）

すべての操作は Docker コンテナ内で実行されます。ホストマシンに Python 環境は不要です。

ワークフロー:

```
合成データ生成 → TrOCR ファインチューニング → 推論実行
```

### 開発シェル（dev）

ソースコードをマウントしたインタラクティブな開発環境に入ります。

```bash
docker compose run --rm dev
```

コンテナ内では `/workspace` にプロジェクトがマウントされており、Python / GPU が利用可能です。

```bash
# コンテナ内での操作例
python -m pytest tests/ -v
python -m training.generate_synthetic_data --count 10
python -c "import torch; print(torch.cuda.is_available())"
exit  # コンテナを抜ける
```

### 合成データの生成（generate）

TrOCR のファインチューニングに使用する合成画像を生成します。ダミーの氏名・住所・番号を使用しており、実在する個人情報は含まれません。

```bash
# Docker Compose で直接実行（デフォルト 500枚）
docker compose run --rm generate

# スクリプト経由（枚数を指定）
bash scripts/docker_generate.sh 500

# コンテナ内から実行
docker compose run --rm dev \
    python -m training.generate_synthetic_data \
    --document_type driver_license_front \
    --count 500 \
    --output /workspace/data/synthetic/driver_license/
```

**生成結果の構造:**

```
data/synthetic/driver_license/
├── images/          # 切り出された1行画像 (00000_0.png, 00000_1.png, ...)
└── labels.json      # {"00000_0.png": "氏名 山田太郎", ...}
```

日本語フォント（Noto Sans CJK JP / IPA フォント）は Docker イメージに組み込み済みです。ホスト側でのフォントインストールは不要です。

### TrOCR のファインチューニング（train）

生成した合成データを使用して、TrOCR モデルをファインチューニングします。

```bash
# スクリプト経由（epochs と batch_size を指定）
bash scripts/docker_train.sh 5 8

# Docker Compose で直接実行
docker compose run --rm train

# パラメータをカスタマイズ（VRAM 12GB 向け OLM 対策）
docker compose run --rm --no-deps train \
    python -m training.train_trocr \
    --data_dir /opt/ml/code/data/synthetic/driver_license/ \
    --output_dir /opt/ml/model \
    --epochs 10 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --fp16
```

学習済みモデルは Docker ボリューム `yomitori-models`（`/opt/ml/model`）に永続化されます。

**主要なオプション:**

| オプション | デフォルト | 説明 |
|---|---|---|
| `--data_dir` | (必須) | 学習データディレクトリ（`images/` + `labels.json`） |
| `--output_dir` | `/opt/ml/model` | モデル保存先 |
| `--base_model` | `microsoft/trocr-base-printed` | ベースモデル名 |
| `--batch_size` | `8` | バッチサイズ |
| `--epochs` | `5` | エポック数 |
| `--learning_rate` | `5e-5` | 学習率 |
| `--gradient_accumulation_steps` | `1` | 勾配蓄積ステップ数 |
| `--fp16` | `True` | 半精度学習（VRAM 節約） |
| `--gradient_checkpointing` | `True` | 勾配チェックポイント（VRAM 節約） |

### 推論の実行（infer）

ファインチューニング済みモデル（またはベースモデル）を使用して、画像からテキストを抽出します。

```bash
# スクリプト経由
bash scripts/run_local_test.sh data/samples/sample_license.jpg

# Docker Compose で直接実行
docker compose run --rm infer

# ファインチューニング済みモデルを使わない場合（ベースモデル）
docker compose run --rm infer \
    python -m src.cli infer \
    --image /opt/ml/code/data/samples/sample_license.jpg \
    --document_type driver_license_front

# CPU 強制指定
docker compose run --rm infer \
    python -m src.cli infer \
    --image /opt/ml/code/data/samples/sample_license.jpg \
    --device cpu
```

結果は JSON で標準出力されます。

### ユニットテスト（test）

```bash
# 全テスト実行
docker compose run --rm test

# スクリプト経由
bash scripts/docker_test.sh

# 個別テスト
docker compose run --rm test python -m pytest tests/test_pipeline.py -v
```

テストは GPU / docTR / TrOCR を必要としないユニットテストのみ含まれています（20テスト）。

### Jupyter Notebook（notebook）

```bash
# ノートブックサーバー起動
docker compose up notebook
```

ブラウザで `http://localhost:8888` にアクセス。前処理の可視化やデバッグに使用します。

---

## ファインチューニングの詳細

### 学習の仕組み

1. **合成データジェネレータ** (`training/generate_synthetic_data.py`) が、免許証レイアウトにダミーテキストを配置した画像を生成
2. 生成画像にランダムな変形（傾き ±15°、ガウシアンノイズ、反射シミュレーション、明るさ変動）を適用
3. 画像を行単位で切り出し、`labels.json` に正解テキストを記録
4. **TrOCR 学習スクリプト** (`training/train_trocr.py`) が、HuggingFace Transformers の Trainer API で `microsoft/trocr-base-printed` をファインチューニング
5. 12GB VRAM に収まるよう、fp16 + gradient_checkpointing を有効化

### 評価指標

- **CER (Character Error Rate)**: `jiwer` ライブラリを使用して計算
- `--eval_dir` を指定すると、毎エポックで CER を評価・出力

### VRAM 12GB (RTX 5070) 向けの設定

```bash
# OOM 発生時の対策: batch_size を下げて勾配蓄積で補完
docker compose run --rm train \
    python -m training.train_trocr \
    --data_dir /opt/ml/code/data/synthetic/driver_license/ \
    --output_dir /opt/ml/model \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --epochs 10 \
    --fp16
```

### ハイパーパラメータのチューニング

| パラメータ | 推奨範囲 | 説明 |
|---|---|---|
| `--epochs` | 2–10 | 合成データの場合は5前後が目安。過学習に注意 |
| `--batch_size` | 4–8 | VRAM 12GB では 8 が上限。OOM 時は 4 に下げて `--gradient_accumulation_steps 2` |
| `--learning_rate` | 1e-5 ~ 1e-4 | デフォルト 5e-5。精度が上がらない場合は下げる |
| `--warmup_ratio` | 0.05–0.15 | 学習初期の warmup 割合 |

---

## 実際の手で検証する方法（ステップバイステップ）

以下は、ゼロから環境構築して推論結果を確認するまでの完全な手順です。

### ステップ 1: 環境構築

```bash
cd /home/yuma/projects/yomitori
bash scripts/setup_env.sh
```

確認:
```bash
# テストが通るか確認（GPU 不要）
docker compose run --rm test
# 期待結果: 20 passed
```

### ステップ 2: 合成データの生成（少量で動作確認）

```bash
# 10枚生成して動作確認
bash scripts/docker_generate.sh 10
```

確認:
```bash
ls data/synthetic/driver_license/images/ | head -5
cat data/synthetic/driver_license/labels.json | python3 -m json.tool 2>/dev/null | head -10 || \
  docker compose run --rm dev python -m json.tool /workspace/data/synthetic/driver_license/labels.json | head -10
```

### ステップ 3: TrOCR のファインチューニング（1エポックで動作確認）

```bash
bash scripts/docker_train.sh 1 4
```

完了後、モデルが保存されているか確認:
```bash
# Docker ボリューム内のモデルを確認
docker compose run --rm --no-deps dev ls -la /opt/ml/model/
# config.json, pytorch_model.bin, training_info.json 等が存在すること
```

### ステップ 4: 本格的なファインチューニング

```bash
# データを増やして本格学習
bash scripts/docker_generate.sh 500
bash scripts/docker_train.sh 5 8
```

### ステップ 5: サンプル画像で推論

```bash
# テスト用画像を配置（個人情報にモザイク処理を施すこと）
cp /path/to/your/mosaic_license.jpg data/samples/sample_license.jpg

# 推論実行（ファインチューニング済みモデルを使用）
bash scripts/run_local_test.sh data/samples/sample_license.jpg
```

結果が JSON で標準出力されます。`status` が `success` または `partial` であればパイプラインは正常動作しています。

### ステップ 6: 前処理のデバッグ

推論結果が思わしくない場合、ノートブックで前処理を可視化して原因を特定します。

```bash
docker compose up notebook
# ブラウザで http://localhost:8888 にアクセス
# notebooks/01_explore_preprocessing.ipynb を開く
```

ホモグラフィ補正が正しく適用されているか、二値化結果から輪郭が検出されているかを確認できます。

### ステップ 7: 精度の評価

複数のサンプル画像に対して推論を行い、フィールドごとの正答率を確認します。

```bash
# ノートブックで評価
docker compose up notebook
# notebooks/03_evaluate_pipeline.ipynb を開く
```

---

## スクリプトリファレンス

| スクリプト | 説明 |
|---|---|
| `scripts/setup_env.sh` | Docker 環境のセットアップ（イメージビルド + 動作確認） |
| `scripts/docker_build.sh` | 全 Docker イメージを一括ビルド |
| `scripts/docker_generate.sh [count]` | 合成データ生成 |
| `scripts/docker_train.sh [epochs] [batch_size]` | TrOCR ファインチューニング |
| `scripts/docker_test.sh` | ユニットテスト実行 |
| `scripts/run_local_test.sh [image_path] [document_type]` | 推論サーバー経由でOCR推論を実行 |
| `scripts/infer_client.sh [image_path] [document_type]` | 推論サーバーにリクエスト送信（`run_local_test.sh` と同等） |
| `scripts/run_sagemaker_local.sh` | SageMaker Local Mode 実行 |
| `scripts/check_licenses.sh` | AGPL/GPL ライセンス混入チェック |

---

## Docker イメージリファレンス

| イメージ | Dockerfile | 用途 |
|---|---|---|
| `yomitori:dev` | `docker/Dockerfile.dev` | 開発・テスト・合成データ生成・ノートブック |
| `yomitori:train` | `docker/Dockerfile.train` | TrOCR ファインチューニング（SageMaker Training 互換） |
| `yomitori:infer` | `docker/Dockerfile.infer` | 推論（SageMaker Inference 互換） |

### Docker Compose サービス一覧

| サービス | イメージ | GPU | 説明 |
|---|---|---|---|
| `dev` | yomitori:dev | ✅ | インタラクティブ開発シェル |
| `generate` | yomitori:dev | — | 合成データ生成 |
| `train` | yomitori:train | ✅ | ファインチューニング |
| `infer` | yomitori:infer | ✅ | 推論実行 |
| `test` | yomitori:dev | — | ユニットテスト |
| `check-licenses` | yomitori:dev | — | ライセンスチェック |
| `notebook` | yomitori:dev | ✅ | Jupyter Notebook (port 8888) |

### Docker ボリューム

| ボリューム | マウントポイント | 説明 |
|---|---|---|
| `yomitori-models` | `/opt/ml/model` | ファインチューニング済みモデルの永続化 |
| `yomitori-hf-cache` | `/root/.cache/huggingface` | HuggingFace モデルキャッシュ（ダウンロード時間短縮） |

---

## SageMaker Local Mode での検証

SageMaker Local Mode を使用すると、ローカルの Docker 環境で SageMaker と同等の学習・推論パイプラインを検証できます。

### 前提条件

- Docker と NVIDIA Container Toolkit がインストール済み（セットアップで確認済み）
- 合成データが `data/synthetic/driver_license/` に生成済み

### 実行

```bash
bash scripts/run_sagemaker_local.sh
```

または個別に:

```bash
# 学習
docker compose run --rm -v /var/run/docker.sock:/var/run/docker.sock dev \
    python -m sagemaker.local_train

# デプロイ + テスト推論
docker compose run --rm -v /var/run/docker.sock:/var/run/docker.sock dev \
    python -m sagemaker.local_deploy --sample data/samples/sample_license.jpg
```

> **注意:** SageMaker Local Mode は Docker-in-Docker を使用するため、ホストの Docker ソケット (`/var/run/docker.sock`) をマウントしています。

### Local Mode と docker compose の違い

| | `docker compose run train` | SageMaker Local Mode |
|---|---|---|
| 学習の仕組み | コンテナ内で直接 `train_trocr.py` を実行 | SageMaker SDK が Training コンテナを構築・実行 |
| クラウド互換性 | なし（ローカル専用） | あり（`instance_type` を変更するだけでクラウドへ移行） |
| 用途 | 開発・迅速な検証 | クラウド移行前の最終検証 |

---

## クラウドへのデプロイ

SageMaker クラウド環境へのデプロイは以下の手順で行います。

### 1. モデルの S3 アップロード

```bash
# model.tar.gz を作成
# Docker ボリュームからモデルを取り出す
docker compose run --rm --no-deps dev bash -c "cd /opt/ml/model && tar czf /workspace/model.tar.gz ."

# S3 にアップロード（AWS CLI が必要）
aws s3 cp model.tar.gz s3://your-bucket/yomitori/model.tar.gz
```

### 2. クラウドデプロイ

AWS 認証情報が設定された環境で実行:

```bash
python -m sagemaker.cloud_deploy \
    --model_data s3://your-bucket/yomitori/model.tar.gz \
    --instance_type ml.g5.xlarge
```

### 3. エンドポイントのテスト

```python
import base64, json
from sagemaker.predictor import Predictor

predictor = Predictor(endpoint_name="<出力されたエンドポイント名>")

with open("data/samples/sample_license.jpg", "rb") as f:
    payload = json.dumps({"image": base64.b64encode(f.read()).decode()})

result = predictor.predict(payload, initial_args={"ContentType": "application/json"})
print(json.dumps(json.loads(result), ensure_ascii=False, indent=2))
```

---

## ライセンスチェック

依存ライブラリに AGPL/GPL ライセンスのものが混入していないか確認します。

```bash
# Docker 経由で実行
bash scripts/check_licenses.sh

# または Docker Compose で直接
docker compose run --rm check-licenses
```

- AGPL/GPL が検出された場合は `exit 1` で異常終了
- 問題なければ `License check passed: no AGPL/GPL found` と表示

---

## 新しい書類タイプの追加方法

プラグイン型アーキテクチャにより、コアパイプラインのコードは一切変更せずに新しい書類を追加できます。

### 手順 1: YAML 設定ファイルの作成

`configs/document_types/` に新しい YAML ファイルを作成します。

```yaml
# configs/document_types/mynumber_card_front.yaml
document_type_id: "mynumber_card_front"
card_aspect_ratio: 1.585
normalized_size: [2400, 1512]
card_type: "card"

zones:
  name:
    x0: 0.00
    y0: 0.00
    x1: 0.50
    y1: 0.10
    label: "氏名"
    label_remove: true
  mynumber:
    x0: 0.00
    y0: 0.10
    x1: 0.80
    y1: 0.20
    label: "個人番号"
    label_remove: true
    whitelist: "0123456789"
  # ... 他のフィールド

validation:
  mynumber:
    pattern: "^[0-9]{12}$"
    description: "12桁のマイナンバー"
    check_digit: true
    required: true
```

### 手順 2: DocumentType クラスの作成（必要な場合のみ）

`src/document_types/` に新しいクラスを作成します。

```python
# src/document_types/mynumber_card.py
from src.document_types.base import DocumentType, ValidationRule, Zone

class MyNumberCardFront(DocumentType):
    @property
    def document_type_id(self) -> str:
        return "mynumber_card_front"

    @property
    def card_aspect_ratio(self) -> float:
        return 1.585

    @property
    def normalized_size(self) -> tuple[int, int]:
        return (2400, 1512)

    @property
    def zones(self) -> dict[str, Zone]:
        return { ... }

    @property
    def validation_rules(self) -> dict[str, ValidationRule]:
        return { ... }

    def detect(self, image: np.ndarray) -> float:
        score = self._check_aspect_ratio(image, self.card_aspect_ratio)
        return score
```

### 手順 3: バリデーションルールの追加（必要な場合のみ）

`src/postprocessing/rules/` にカスタムルールクラスを追加します。

### 手順 4: Registry への登録

`src/cli.py` の `build_registry()` 関数に新しい書類タイプを追加します。

```python
def build_registry() -> DocumentTypeRegistry:
    registry = DocumentTypeRegistry()
    registry.register(DriverLicenseFront())
    registry.register(MyNumberCardFront())  # 追加
    return registry
```

**これだけです。`src/pipeline/`, `src/detection/`, `src/recognition/`, `src/preprocessing/` は一切変更不要。**

---

## 出力フォーマット

```json
{
  "status": "success",
  "document_type": "driver_license_front",
  "fields": {
    "name": {
      "value": "山田太郎",
      "confidence": 0.95,
      "low_confidence": false
    },
    "birth_date": {
      "raw": "昭和61年5月1日生",
      "iso": "1986-05-01",
      "value": "1986-05-01",
      "confidence": 0.92,
      "low_confidence": false
    },
    "address": {
      "value": "東京都千代田区霞が関1-2-3",
      "confidence": 0.88,
      "low_confidence": false
    },
    "license_number": {
      "value": "010203040506",
      "confidence": 0.99,
      "low_confidence": false,
      "validation_passed": true
    },
    "expiry_date": {
      "raw": "令和7年12月31日",
      "iso": "2025-12-31",
      "value": "2025-12-31",
      "confidence": 0.91,
      "low_confidence": false,
      "validation_passed": true
    }
  },
  "preprocessing": {
    "homography_applied": true,
    "corrected_image_size": [2400, 1512],
    "fallback_used": false
  },
  "warnings": []
}
```

### `status` の値

| 値 | 意味 |
|---|---|
| `success` | 全フィールドが正常に認識・バリデーション通過 |
| `partial` | 一部フィールドが低信頼度（confidence < 0.7）またはバリデーション不合格 |
| `failed` | 前処理失敗またはフィールドが1つも抽出できなかった |

---

## 技術スタック

| コンポーネント | ライブラリ | ライセンス |
|---|---|---|
| 前処理 | OpenCV | Apache 2.0 |
| 文字検出 | docTR (Mindee) | Apache 2.0 |
| 文字認識 | Microsoft TrOCR | MIT |
| 後処理 | Python 標準ライブラリ | — |
| デプロイ | Amazon SageMaker | Apache 2.0 |
| 学習 | HuggingFace Transformers | Apache 2.0 |
| CER 評価 | jiwer | MIT |
| コンテナ | Docker + PyTorch 2.6 + CUDA 12.6 | Apache 2.0 |

**AGPL/GPL ライブラリおよび PaddleOCR は一切使用していません。**

---

## トラブルシューティング

### Docker コンテナから GPU が見えない

```bash
# NVIDIA Container Toolkit がインストールされているか確認
nvidia-ctk --version

# Docker から GPU が見えるか確認
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# 見えない場合は NVIDIA Container Toolkit を再設定
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 学習時に CUDA Out of Memory (OOM)

```bash
# バッチサイズを下げて勾配蓄積で補完
docker compose run --rm train \
    python -m training.train_trocr \
    --data_dir /opt/ml/code/data/synthetic/driver_license/ \
    --output_dir /opt/ml/model \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --epochs 5
```

### docTR のインストールエラー（Docker ビルド時）

Dockerfile でシステムライブラリ（`libgl1`, `libglib2.0-0`）をインストール済みです。それでもエラーが出る場合は:

```bash
# Docker ビルドキャッシュをクリア
docker compose build --no-cache dev
```

### 推論結果が空（`status: "failed"`）

1. 入力画像が `data/samples/` に正しく配置されているか確認
2. 前処理ノートブックでホモグラフィ補正結果を確認:
   ```bash
   docker compose up notebook
   # notebooks/01_explore_preprocessing.ipynb を開く
   ```
3. `preprocessing.fallback_used` が `true` の場合、輪郭検出に失敗しています。入力画像のコントラストや照明を見直してください。

### ファインチューニング済みモデルが推論時に見つからない

Docker ボリューム `yomitori-models` にモデルが保存されています。確認方法:

```bash
docker compose run --rm --no-deps dev ls -la /opt/ml/model/
```

モデルが存在しない場合は、ファインチューニングが未実行です:

```bash
bash scripts/docker_train.sh 5 8
```

### SageMaker Local Mode が動かない

```bash
# Docker ソケットがマウントされているか確認
# run_sagemaker_local.sh は -v /var/run/docker.sock:/var/run/docker.sock を指定
# 手動実行時も同様にマウントする必要があります

# docker-compose がインストールされているか確認（SageMaker Local Mode が必要）
docker compose run --rm dev pip list | grep docker-compose
# ない場合:
docker compose run --rm dev pip install docker-compose
```

### Docker ディスク容量不足

```bash
# 未使用のイメージ・コンテナを削除
docker system prune -a --volumes
# ※ yomitori 関連のボリュームも削除される場合があるので注意
```

---

## リポジトリ構成

```
yomitori/
├── README.md                       # このファイル
├── LICENSE                         # Apache 2.0
├── requirements-docker.txt          # Docker コンテナ用依存ライブラリ
├── requirements-dev.txt             # 開発用追加ライブラリ（テスト・ノートブック等）
├── docker-compose.yaml              # 全サービス定義（dev/train/infer/test/notebook）
│
├── configs/
│   ├── engine.yaml                  # エンジン全体設定
│   └── document_types/              # 書類タイプ別設定（プラグイン式）
│       ├── driver_license_front.yaml  # 運転免許証（表面）← 実装済み
│       ├── mynumber_card_front.yaml   # マイナンバーカード（表面）← プレースホルダ
│       ├── mynumber_card_back.yaml    # マイナンバーカード（裏面）← プレースホルダ
│       ├── residence_card_front.yaml  # 在留カード（表面）← プレースホルダ
│       └── passport.yaml              # パスポート ← プレースホルダ
│
├── docker/
│   ├── Dockerfile.dev               # 開発・テスト・ノートブック用コンテナ
│   ├── Dockerfile.train              # 学習用コンテナ（SageMaker Training 互換）
│   └── Dockerfile.infer              # 推論用コンテナ（SageMaker Inference 互換）
│
├── src/
│   ├── cli.py                       # CLI エントリポイント
│   ├── preprocessing/               # OpenCV前処理（全書類共通・変更不要）
│   ├── detection/                   # docTR文字検出（全書類共通・変更不要）
│   ├── recognition/                 # TrOCR文字認識（全書類共通・変更不要）
│   ├── pipeline/                    # E2Eパイプライン（全書類共通・変更不要）
│   ├── postprocessing/              # 後処理（バリデーション・正規化）
│   ├── document_types/              # ★ 書類タイププラグイン
│   └── utils/                       # ユーティリティ（画像・ログ・設定）
│
├── training/
│   ├── generate_synthetic_data.py   # 合成データジェネレータ
│   ├── train_trocr.py               # TrOCRファインチューニング
│   └── dataset.py                    # PyTorch Datasetクラス
│
├── sagemaker/
│   ├── local_train.py               # SageMaker Local Mode 学習
│   ├── local_deploy.py              # SageMaker Local Mode デプロイ
│   ├── cloud_deploy.py              # クラウド本番デプロイ
│   └── inference_entry_point.py     # SageMaker推論エントリポイント
│
├── tests/                           # ユニットテスト
├── notebooks/                       # 探索・デバッグ用ノートブック
├── scripts/                         # 環境構築・実行スクリプト（全 Docker ベース）
└── data/                            # サンプル画像・合成データ
```

---

## ライセンス

[Apache License 2.0](LICENSE)