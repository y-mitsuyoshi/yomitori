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
- [セットアップ](#セットアップ)
- [クイックスタート](#クイックスタート)
- [サンプル画像で推論をテストする](#サンプル画像で推論をテストする)
- [ファインチューニング](#ファインチューニング)
- [合成データ生成](#合成データ生成)
- [ユニットテスト](#ユニットテスト)
- [Jupyter Notebook](#jupyter-notebook)
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
- **完全 Docker 化**: すべての実行環境を Docker コンテナで提供。ホスト環境を汚さない
- **SageMaker 対応**: Local Mode → クラウド（ml.g5.xlarge 等）へのシームレス移行
- **ライセンス安全**: AGPL/GPL ライブラリ不使用

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
| GPU | NVIDIA RTX 5070 (VRAM 12GB) 以上を推奨 |
| NVIDIA Driver | 550+ (CUDA 12.8 対応) |
| NVIDIA Container Toolkit | GPU を Docker コンテナから利用するために必要 |

### Docker のインストール

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin

# NVIDIA Container Toolkit
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
docker --version
docker compose version
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

---

## セットアップ

```bash
git clone <repository-url> yomitori
cd yomitori

# ベースイメージをビルド（初回のみ・10〜15分）
docker compose build base

# serve / train / dev イメージをビルド
docker compose build
```

---

## クイックスタート

```bash
# 1. イメージをビルド
docker compose build base && docker compose build

# 2. 合成データを生成（10枚で動作確認）
docker compose run --rm dev python -m training.generate_synthetic_data --count 10

# 3. ファインチューニング（1エポックで動作確認）
docker compose run --rm train python -m training.train_trocr \
    --data_dir /opt/ml/code/data/synthetic/driver_license/ \
    --output_dir /opt/ml/model --epochs 1 --batch_size 4

# 4. 推論サーバーを起動
docker compose up -d serve

# 5. サンプル画像で推論（curl で画像を直接送信）
curl -s http://localhost:8080/ping
curl -s -X POST http://localhost:8080/invocations \
    -H "Content-Type: image/jpeg" \
    --data-binary @data/samples/sample_license.jpg | python3 -m json.tool

# 6. サーバーを停止
docker compose down
```

---

## サンプル画像で推論をテストする

### ステップ1: サンプル画像を配置

モザイク済みの運転免許証画像を `data/samples/` に配置します。

```bash
cp /path/to/your/license.jpg data/samples/sample_license.jpg
```

### ステップ2: 推論サーバーを起動

```bash
docker compose up -d serve
```

起動には約30秒かかります（初回はモデルのダウンロードが必要）。`Application startup complete` がログに出たら準備完了です。

```bash
# 起動状況を確認
docker compose logs -f serve
```

### ステップ3: ヘルスチェック

```bash
curl -s http://localhost:8080/ping
# {"status":"ok","model_loaded":true}
```

### ステップ4: 推論リクエストを送信

curl で画像ファイルを直接送信します。

```bash
curl -s -X POST http://localhost:8080/invocations \
    -H "Content-Type: image/jpeg" \
    --data-binary @data/samples/sample_license.jpg | python3 -m json.tool
```

JSONで書類タイプを明示指定することも可能です。

```bash
curl -s -X POST http://localhost:8080/invocations \
    -H "Content-Type: application/json" \
    -d "{\"image\":\"$(base64 -w 0 data/samples/sample_license.jpg)\",\"document_type\":\"driver_license_front\"}" | python3 -m json.tool
```

### ステップ5: サーバーを停止

```bash
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

---

## ファインチューニング

TrOCRのベースモデル（`microsoft/trocr-base-printed`）は英語専用です。日本語・多言語に対応させるため、デコーダのトークナイザーを多言語対応のものに差し替えてファインチューニングします。

### 多言語対応の仕組み

```
TrOCRベースモデル（画像エンコーダはそのまま）
  ↓
デコーダのトークナイザーを多言語対応に差し替え
  ↓
合成データ（日本語テキスト）でファインチューニング
  ↓
多言語テキストを認識できるモデルが完成
```

### 使用するトークナイザー

| トークナイザー | ライセンス | 対応言語 | 指定方法 |
|---|---|---|---|
| `xlm-roberta-base`（デフォルト） | MIT | 100言語（日本語・英語・中国語等） | `--decoder_tokenizer xlm-roberta-base` |
| `cl-tohoku/bert-base-japanese-v3` | Apache 2.0 | 日本語のみ | `--decoder_tokenizer cl-tohoku/bert-base-japanese-v3` |
| `bert-base-multilingual-cased` | Apache 2.0 | 多言語 | `--decoder_tokenizer bert-base-multilingual-cased` |

### 手順

```bash
# 1. 合成データを生成
docker compose run --rm dev python -m training.generate_synthetic_data \
    --document_type driver_license_front \
    --count 500 \
    --output /opt/ml/code/data/synthetic/driver_license/

# 2. ファインチューニングを実行（日本語・デフォルト）
docker compose run --rm train python -m training.train_trocr \
    --data_dir /opt/ml/code/data/synthetic/driver_license/ \
    --output_dir /opt/ml/model \
    --epochs 5 \
    --batch_size 8

# 3. 多言語対応でファインチューニングする場合
docker compose run --rm train python -m training.train_trocr \
    --data_dir /opt/ml/code/data/synthetic/driver_license/ \
    --output_dir /opt/ml/model \
    --decoder_tokenizer xlm-roberta-base \
    --epochs 5 \
    --batch_size 8

# 4. 学習済みモデルを確認
docker compose run --rm dev ls -la /opt/ml/model/
```

学習済みモデルは Docker ボリューム `yomitori-models`（`/opt/ml/model`）に永続化されます。

### 主なオプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--data_dir` | (必須) | 学習データディレクトリ（`images/` + `labels.json`） |
| `--output_dir` | `/opt/ml/model` | モデル保存先 |
| `--base_model` | `microsoft/trocr-base-printed` | TrOCRベースモデル（画像エンコーダ用） |
| `--decoder_tokenizer` | `xlm-roberta-base` | デコーダのトークナイザー（多言語対応・MIT ライセンス） |
| `--batch_size` | `8` | バッチサイズ |
| `--epochs` | `5` | エポック数 |
| `--learning_rate` | `5e-5` | 学習率 |
| `--gradient_accumulation_steps` | `1` | 勾配蓄積ステップ数 |
| `--fp16` | `True` | 半精度学習（VRAM 節約） |

### VRAM 12GB (RTX 5070) 向けの設定

OOM 発生時はバッチサイズを下げて勾配蓄積で補完します。

```bash
docker compose run --rm train python -m training.train_trocr \
    --data_dir /opt/ml/code/data/synthetic/driver_license/ \
    --output_dir /opt/ml/model \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --epochs 10 \
    --fp16
```

### モデルのバージョン管理と追加学習（継続学習）

モデルのバージョン管理や、既存の学習済みモデル（例: `v1`）をベースにした追加のファインチューニング（`v2`, `v3` ...）は、`--output_dir` と `--base_model` を組み合わせることで簡単に行えます。

#### 1. 初回の学習 (v1)
ベースモデルから開始し、出力をバージョンごとのディレクトリ（例: `/opt/ml/model/v1`）に保存します。
```bash
docker compose run --rm train python -m training.train_trocr \
    --data_dir /opt/ml/code/data/synthetic/driver_license/ \
    --output_dir /opt/ml/model/v1 \
    --base_model microsoft/trocr-base-printed \
    --epochs 5
```

#### 2. 追加学習 (v1 から v2 へ)
前回作成した `v1` モデルを `--base_model` に指定し、新しいデータを追加学習させて `v2` に保存します。
```bash
docker compose run --rm train python -m training.train_trocr \
    --data_dir /opt/ml/code/data/synthetic/driver_license/ \
    --output_dir /opt/ml/model/v2 \
    --base_model /opt/ml/model/v1 \
    --epochs 5
```

#### 3. 推論時に特定バージョンをロードする
特定のバージョン（例: `v2`）で推論サーバーを起動する場合は、環境変数 `YOMITORI_MODEL_DIR` で対象のパスを指定します。

`docker-compose.yaml` の `serve` サービスの `YOMITORI_MODEL_DIR` を変更するか、以下のようにコマンド起動時に一時的に上書きします。
```bash
docker compose run --rm -p 8080:8080 -e YOMITORI_MODEL_DIR=/opt/ml/model/v2 serve
```

#### 4. 日本語特化モデルと多言語対応モデルの二重起動と自動ルーティング

商用利用において、「日本の身分証（免許証・マイナンバーカード）」は高速かつ高精度な日本語特化モデル（例：東北大学BERTトークナイザー版）で読み取り、「在留カードやパスポート」は多言語対応モデル（例：Facebook XLM-RoBERTa版）で読み取る、といった**自動使い分け（ルーティング）**が可能です。

##### ① ディレクトリ構成
以下のように `YOMITORI_MODEL_DIR`（デフォルト: `/opt/ml/model`）配下に `japanese` と `multilingual` という名前でそれぞれのモデルを配置します。

```
/opt/ml/model/
├── japanese/         # 日本語特化モデル (cl-tohoku で学習)
│   ├── config.json
│   └── pytorch_model.bin
└── multilingual/     # 多言語対応モデル (xlm-roberta-base で学習)
    ├── config.json
    └── pytorch_model.bin
```

##### ② 各モデルの学習方法
各々のモデルは、辞書（`--decoder_tokenizer`）を指定して別々の出力先に保存します。

* **日本語特化モデルの学習**:
  ```bash
  docker compose run --rm train python -m training.train_trocr \
      --base_model microsoft/trocr-base-printed \
      --decoder_tokenizer cl-tohoku/bert-base-japanese-v3 \
      --data_dir /opt/ml/code/data/synthetic/driver_license/ \
      --output_dir /opt/ml/model/japanese \
      --epochs 5
  ```

* **多言語対応モデルの学習**:
  ```bash
  docker compose run --rm train python -m training.train_trocr \
      --base_model microsoft/trocr-base-printed \
      --decoder_tokenizer xlm-roberta-base \
      --data_dir /opt/ml/code/data/synthetic/driver_license/ \
      --output_dir /opt/ml/model/multilingual \
      --epochs 5
  ```

##### ③ サーバーの起動とルーティング挙動
上記のようにモデルを配置して推論サーバーを起動すると、自動的に両方のモデルがロードされます。

```bash
docker compose up -d serve
```

リクエストの画像からカードの種類が判定されたのち、内部で自動的に以下の振り分けが行われます。
* `passport` (パスポート)、`residence_card_front` (在留カード) ➡ **多言語対応モデル (`multilingual`)** で推論
* `driver_license_front` (運転免許証)、`mynumber_card_front` (マイナンバーカード) ➡ **日本語特化モデル (`japanese`)** で推論

*(※サブフォルダが存在しない場合は、従来通り `/opt/ml/model` 直下のモデルのみを読み込み、両方のルーティングで共通して使用する後方互換モードとして動作します)*

---

### 学習後に推論する

```bash
# サーバーを再起動すると学習済みモデルが自動ロードされる
docker compose down
docker compose up -d serve

# 起動完了を待つ（ログで "Application startup complete" を確認）
docker compose logs -f serve

# 推論
curl -s -X POST http://localhost:8080/invocations \
    -H "Content-Type: image/jpeg" \
    --data-binary @data/samples/sample_license.jpg | python3 -m json.tool
```

学習済みモデルには多言語トークナイザーが保存されているため、推論時に自動的に多言語テキストが認識されます。

---

## 合成データ生成

TrOCR のファインチューニングに使用する合成画像を生成します。ダミーの氏名・住所・番号を使用しており、実在する個人情報は含まれません。

```bash
docker compose run --rm dev python -m training.generate_synthetic_data \
    --document_type driver_license_front \
    --count 500 \
    --output /opt/ml/code/data/synthetic/driver_license/
```

生成結果の構造:

```
data/synthetic/driver_license/
├── images/          # 切り出された1行画像 (00000_0.png, 00000_1.png, ...)
└── labels.json      # {"00000_0.png": "氏名 山田太郎", ...}
```

日本語フォント（Noto Sans CJK JP / IPA フォント）は Docker イメージに組み込み済みです。

### 文字種の拡張（`--kanji_boost`）

デフォルトのダミーデータ（40名字 × 30名前 = 1,200通り）は、本番で遭遇する多様な漢字を網羅できません。`--kanji_boost` オプションを付けると、CJK統合漢字プール（約5,000字）からランダムに漢字を生成した氏名・住所を使用します。これにより、未知の漢字に対する認識精度が向上します。

```bash
# 通常のダミーデータ（固定リストから生成）
docker compose run --rm dev python -m training.generate_synthetic_data \
    --count 500 --output /opt/ml/code/data/synthetic/driver_license/

# 漢字ブーストモード（ランダム漢字で文字種を拡張）
docker compose run --rm dev python -m training.generate_synthetic_data \
    --count 500 --kanji_boost \
    --output /opt/ml/code/data/synthetic/driver_license_kanji/
```

> **商用リリースに向けた推奨戦略**: 固定リスト（`--kanji_boost` なし）で基本構造を学習させた後、`--kanji_boost` ありのデータで追加学習（継続学習）することで、両方の精度を確保できます。

---

## ユニットテスト

```bash
# 全テスト実行
docker compose run --rm dev python -m pytest tests/ -v
```

テストは GPU / docTR / TrOCR を必要としないユニットテスト20個です。

---

## Jupyter Notebook

```bash
docker compose up notebook
```

ブラウザで `http://localhost:8888` にアクセス。前処理の可視化やデバッグに使用します。

---

## SageMaker Local Mode での検証

SageMaker Local Mode を使用すると、ローカルの Docker 環境で SageMaker と同等の学習・推論パイプラインを検証できます。クラウドへ移行する前に、Local Mode で最終検証することを推奨します。

### `docker compose up serve` との違い

| | `docker compose up serve` | SageMaker Local Mode |
|---|---|---|
| 用途 | 開発・迅速な検証 | クラウド移行前の最終検証 |
| 仕組み | FastAPI サーバーが直接推論 | SageMaker SDK がコンテナを構築・実行 |
| クラウド互換性 | なし | あり（`instance_type` を変更するだけでクラウドへ移行） |
| エンドポイント | `/ping`, `/invocations` | SageMaker Predictor API |

### 前提条件

- Docker ソケットがマウント可能であること
- 合成データが `data/synthetic/driver_license/` に生成済みであること

### ステップ1: 学習（Local Mode）

SageMaker SDK 経由で学習コンテナを構築し、学習を実行します。

```bash
docker compose run --rm -v /var/run/docker.sock:/var/run/docker.sock dev \
    python -m sagemaker.local_train
```

学習完了後、モデルアーティファクトが作成されます。

### ステップ2: model.tar.gz を作成

SageMaker はモデルを `model.tar.gz` 形式で扱います。Docker ボリュームから取り出して圧縮します。

```bash
docker compose run --rm dev bash -c "cd /opt/ml/model && tar czf /opt/ml/code/model.tar.gz ."
```

### ステップ3: エンドポイントをデプロイ + サンプル画像で推論

SageMaker SDK 経由でローカルエンドポイントを構築し、サンプル画像の推論を実行します。

```bash
docker compose run --rm -v /var/run/docker.sock:/var/run/docker.sock dev \
    python -m sagemaker.local_deploy --sample data/samples/sample_license.jpg
```

推論結果がJSONで標準出力されます。

### ステップ4: 別の画像で推論（任意）

`--sample` オプションで任意の画像を指定できます。

```bash
docker compose run --rm -v /var/run/docker.sock:/var/run/docker.sock dev \
    python -m sagemaker.local_deploy --sample data/samples/sample_license2.jpg
```

> **注意:** SageMaker Local Mode は Docker-in-Docker を使用するため、ホストの Docker ソケット (`/var/run/docker.sock`) をマウントしています。

---

## クラウドへのデプロイ

SageMaker クラウドにデプロイする手順です。カスタム Docker イメージ（`yomitori:infer`）を ECR にプッシュし、`model.tar.gz` を S3 にアップロードしてエンドポイントを構築します。

### 1. model.tar.gz を作成

```bash
docker compose run --rm dev bash -c "cd /opt/ml/model && tar czf /opt/ml/code/model.tar.gz ."
```

### 2. S3 にモデルをアップロード

```bash
aws s3 cp model.tar.gz s3://your-bucket/yomitori/model.tar.gz
```

### 3. ECR に Docker イメージをプッシュ

```bash
# ECR リポジトリを作成
aws ecr create-repository --repository-name yomitori

# ログイン
aws ecr get-login-password | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com

# イメージにタグを付けてプッシュ
docker tag yomitori:infer <account-id>.dkr.ecr.<region>.amazonaws.com/yomitori:infer
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/yomitori:infer
```

### 4. エンドポイントをデプロイ

```bash
python -m sagemaker.cloud_deploy \
    --model_data s3://your-bucket/yomitori/model.tar.gz \
    --ecr_image <account-id>.dkr.ecr.<region>.amazonaws.com/yomitori:infer \
    --role arn:aws:iam::<account-id>:role/service-role/AmazonSageMaker-ExecutionRole \
    --instance_type ml.g5.xlarge
```

### 5. エンドポイントのテスト

```python
from sagemaker.predictor import Predictor
from sagemaker.serializers import DataSerializer
import json

predictor = Predictor(endpoint_name="<エンドポイント名>")
predictor.serializer = DataSerializer(content_type="image/jpeg")

with open("data/samples/sample_license.jpg", "rb") as f:
    result = predictor.predict(f.read())

print(json.dumps(json.loads(result), ensure_ascii=False, indent=2))
```

> **注意:** SageMaker の PyTorch プリビルドイメージには docTR・TrOCR・CUDA 12.8 が含まれていないため、カスタム Docker イメージ（`yomitori:infer`）を ECR にプッシュして使用します。

---

## ライセンスチェック

```bash
docker compose run --rm dev bash -c "pip install pip-licenses && pip-licenses | grep -iE '(AGPL|GPL)' && exit 1 || echo 'License check passed'"
```

---

## 新しい書類タイプの追加方法

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

validation:
  mynumber:
    pattern: "^[0-9]{12}$"
    description: "12桁のマイナンバー"
    required: true
```

### 手順 2: DocumentType クラスの作成（必要な場合のみ）

`src/document_types/` に新しいクラスを作成します。

### 手順 3: Registry への登録

`src/cli.py` の `build_registry()` 関数に追加します。

```python
def build_registry() -> DocumentTypeRegistry:
    registry = DocumentTypeRegistry()
    registry.register(DriverLicenseFront())
    registry.register(MyNumberCardFront())  # 追加
    return registry
```

**コアパイプライン（`src/pipeline/`, `src/detection/`, `src/recognition/`, `src/preprocessing/`）は変更不要。**

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

---

## 技術スタック

| コンポーネント | ライブラリ | ライセンス |
|---|---|---|
| 前処理 | OpenCV | Apache 2.0 |
| 文字検出 | docTR (Mindee) | Apache 2.0 |
| 文字認識 | Microsoft TrOCR | MIT |
| トークナイザー（多言語） | Facebook xlm-roberta-base | MIT |
| トークナイザー（日本語特化） | 東北大学 cl-tohoku/bert-base-japanese-v3 | Apache 2.0 |
| デプロイ | Amazon SageMaker | Apache 2.0 |
| 学習 | HuggingFace Transformers | Apache 2.0 |
| コンテナ | Docker + PyTorch 2.6 + CUDA 12.8 | Apache 2.0 |

---

## トラブルシューティング

### Docker コンテナから GPU が見えない

```bash
nvidia-ctk --version
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
# 見えない場合:
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 学習時に CUDA Out of Memory (OOM)

```bash
docker compose run --rm train python -m training.train_trocr \
    --data_dir /opt/ml/code/data/synthetic/driver_license/ \
    --output_dir /opt/ml/model \
    --batch_size 2 --gradient_accumulation_steps 4 --epochs 5
```

### 推論結果が空（`status: "failed"`）

1. 入力画像が `data/samples/` に正しく配置されているか確認
2. 前処理ノートブックでホモグラフィ補正結果を確認: `docker compose up notebook`
3. `preprocessing.fallback_used` が `true` の場合、輪郭検出に失敗しています

### 推論サーバーが起動しない

```bash
# ログを確認
docker compose logs serve

# モデルが存在するか確認
docker compose run --rm dev ls -la /opt/ml/model/

# モデルがない場合はファインチューニングを実行
docker compose run --rm train python -m training.train_trocr \
    --data_dir /opt/ml/code/data/synthetic/driver_license/ \
    --output_dir /opt/ml/model --epochs 1 --batch_size 4
```

---

## リポジトリ構成

```
yomitori/
├── README.md
├── LICENSE
├── requirements-docker.txt          # Docker コンテナ用依存ライブラリ
├── requirements-dev.txt             # 開発用追加ライブラリ
├── docker-compose.yaml              # サービス定義（serve/train/dev/notebook）
│
├── configs/
│   ├── engine.yaml
│   └── document_types/              # 書類タイプ別設定（プラグイン式）
│
├── docker/
│   ├── Dockerfile.base               # 共通ベースイメージ（CUDA 12.8 + PyTorch）
│   ├── Dockerfile.dev               # 開発・合成データ生成・ノートブック用
│   ├── Dockerfile.train              # 学習用（SageMaker Training 互換）
│   └── Dockerfile.infer              # 推論サーバー用（SageMaker Inference 互換）
│
├── src/
│   ├── cli.py                       # CLI エントリポイント
│   ├── preprocessing/               # OpenCV前処理（共通・変更不要）
│   ├── detection/                   # docTR文字検出（共通・変更不要）
│   ├── recognition/                 # TrOCR文字認識（共通・変更不要）
│   ├── pipeline/                    # E2Eパイプライン（共通・変更不要）
│   ├── postprocessing/              # 後処理（バリデーション・正規化）
│   ├── document_types/              # ★ 書類タイププラグイン
│   │   ├── base.py                  # DocumentType ABC・Zone・ValidationRule
│   │   ├── registry.py              # 自動判定レジストリ
│   │   ├── driver_license.py        # 運転免許証（表面）
│   │   └── mynumber_card.py         # マイナンバーカード（表面）
│   └── utils/                       # ユーティリティ
│
├── training/
│   ├── generate_synthetic_data.py   # 合成データジェネレータ
│   ├── train_trocr.py               # TrOCRファインチューニング
│   └── dataset.py                    # PyTorch Dataset
│
├── sagemaker/
│   ├── serve.py                     # FastAPI 推論サーバー
│   ├── local_train.py               # SageMaker Local Mode 学習
│   ├── local_deploy.py              # SageMaker Local Mode デプロイ
│   ├── cloud_deploy.py              # クラウドデプロイ
│   └── inference_entry_point.py     # SageMaker推論エントリポイント
│
├── tests/                           # ユニットテスト
├── notebooks/                       # 探索・デバッグ用ノートブック
└── data/                            # サンプル画像・合成データ
```

---

## ライセンス

[Apache License 2.0](LICENSE)