# SageMaker クラウドデプロイ手順書

学習済みモデルをAmazon SageMakerクラウドにデプロイし、本番運用する手順を説明します。

## 目次

- [概要](#概要)
- [前提条件](#前提条件)
- [手順1: model.tar.gzの作成](#手順1-modeltargzの作成)
- [手順2: S3にモデルをアップロード](#手順2-s3にモデルをアップロード)
- [手順3: ECRにDockerイメージをプッシュ](#手順3-ecrにdockerイメージをプッシュ)
- [手順4: SageMakerエンドポイントをデプロイ](#手順4-sagemakerエンドポイントをデプロイ)
- [手順5: エンドポイントのテスト](#手順5-エンドポイントのテスト)
- [手順6: エンドポイントの運用](#手順6-エンドポイントの運用)
- [コスト目安](#コスト目安)
- [トラブルシューティング](#トラブルシューティング)

---

## 概要

SageMakerクラウドにデプロイするために必要な2つの要素:

```
SageMakerクラウド
├── ① model.tar.gz（学習済みモデル）  → Amazon S3 にアップロード
└── ② Dockerイメージ（推論コンテナ）   → Amazon ECR にプッシュ
```

| 要素 | 中身 | アップロード先 |
|---|---|---|
| ① model.tar.gz | config.json, model.safetensors, tokenizer等 | S3バケット |
| ② Dockerイメージ | CUDA 12.8 + PyTorch + docTR + TrOCR | ECRリポジトリ |

> **なぜ自作Dockerイメージが必要か**: SageMaker標準のPyTorchイメージにはdocTR・TrOCR・CUDA 12.8が含まれていないため、`yomitori:infer` イメージをECRにプッシュして使用します。

---

## 前提条件

| 項目 | 要件 |
|---|---|
| AWSアカウント | 有効なAWSアカウント |
| AWS CLI | インストール済み・認証設定済み（`aws configure`） |
| IAMロール | SageMaker実行ロール（`AmazonSageMaker-ExecutionRole`） |
| S3バケット | モデル保存用のS3バケット |
| Docker | ローカルで `yomitori:infer` イメージがビルド済み |
| 学習済みモデル | ファインチューニング済みモデルが `/opt/ml/model/` に保存済み |

### AWS CLIの初期設定（未設定の場合）

```bash
# AWS認証情報の設定
aws configure
# → Access Key ID: [あなたのアクセスキー]
# → Secret Access Key: [あなたのシークレットキー]
# → Default region name: ap-northeast-1（東京リージョン推奨）
# → Default output format: json
```

### 必要なIAM権限

デプロイに使用するAWSユーザー/IAMロールには以下の権限が必要です:

```
- sagemaker:CreateEndpoint
- sagemaker:CreateEndpointConfig
- sagemaker:CreateModel
- sagemaker:InvokeEndpoint
- sagemaker:DeleteEndpoint
- ecr:CreateRepository
- ecr:BatchCheckLayerAvailability
- ecr:CompleteLayerUpload
- ecr:InitiateLayerUpload
- ecr:PutImage
- ecr:UploadLayerPart
- s3:PutObject
- s3:GetObject
- iam:PassRole
```

---

## 手順1: model.tar.gzの作成

学習済みモデルを `model.tar.gz` に圧縮します。

### 特定バージョンのモデルを圧縮

```bash
# v1（japanese/）のモデルを圧縮
docker compose run --rm dev bash -c "cd /opt/ml/model/japanese && tar czf /opt/ml/code/model.tar.gz --exclude='checkpoint-*' ."

# v2のモデルを圧縮
docker compose run --rm dev bash -c "cd /opt/ml/model/v2 && tar czf /opt/ml/code/model.tar.gz --exclude='checkpoint-*' ."
```

> **`--exclude='checkpoint-*'` の理由**: チェックポイントディレクトリは推論に不要で、ファイルサイズが大きいため除外します。

### model.tar.gzの中身を確認

```bash
# ファイルサイズの確認
ls -lh model.tar.gz
# → 約500MB〜1GB（モデルサイズによる）

# 中身の確認
tar tzf model.tar.gz
```

**期待される内容:**
```
./config.json
./model.safetensors
./preprocessor_config.json
./tokenizer.json
./sentencepiece.bpe.model
./generation_config.json
./training_info.json
```

---

## 手順2: S3にモデルをアップロード

### S3バケットの作成（初回のみ）

```bash
# バケット名は全世界で一意である必要があります
aws s3 mb s3://yomitori-models-<your-name>
# 例: s3://yomitori-models-yamada
```

### model.tar.gzをS3にアップロード

```bash
aws s3 cp model.tar.gz s3://yomitori-models-<your-name>/driver_license/v1/model.tar.gz
```

### アップロードの確認

```bash
aws s3 ls s3://yomitori-models-<your-name>/driver_license/v1/
# → model.tar.gz
```

### バージョン管理

複数バージョンのモデルをS3で管理できます:

```bash
# v1
aws s3 cp model.tar.gz s3://yomitori-models-<your-name>/driver_license/v1/model.tar.gz

# v2（別バージョン）
aws s3 cp model.tar.gz s3://yomitori-models-<your-name>/driver_license/v2/model.tar.gz

# v3
aws s3 cp model.tar.gz s3://yomitori-models-<your-name>/driver_license/v3/model.tar.gz
```

---

## 手順3: ECRにDockerイメージをプッシュ

### ECRリポジトリの作成（初回のみ）

```bash
aws ecr create-repository --repository-name yomitori
```

**出力例:**
```json
{
    "repository": {
        "repositoryUri": "123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/yomitori",
        "repositoryArn": "arn:aws:ecr:ap-northeast-1:123456789012:repository/yomitori"
    }
}
```

> `repositoryUri` の `123456789012` があなたのAWSアカウントIDです。以降の手順で使用します。

### ECRにログイン

```bash
# アカウントIDとリージョンを変数に設定
AWS_ACCOUNT_ID=123456789012
AWS_REGION=ap-northeast-1

aws ecr get-login-password | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
# → Login Succeeded
```

### Dockerイメージにタグを付けてプッシュ

```bash
# イメージにタグを付ける
docker tag yomitori:infer ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/yomitori:infer

# ECRにプッシュ（約5〜10分）
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/yomitori:infer
```

### プッシュの確認

```bash
aws ecr describe-images --repository-name yomitori
```

---

## 手順4: SageMakerエンドポイントをデプロイ

### SageMaker実行ロールの確認

```bash
# SageMaker実行ロールのARNを確認
aws iam list-roles --path-prefix /service-role/ | grep SageMaker
# → "Arn": "arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole"
```

> ロールが存在しない場合は、AWSコンソールで作成してください:
> IAM → ロール → ロールの作成 → SageMaker → AmazonSageMakerFullAccess

### エンドポイントをデプロイ

```bash
docker compose run --rm dev python -m scripts.cloud_deploy \
    --model_data s3://yomitori-models-<your-name>/driver_license/v1/model.tar.gz \
    --ecr_image ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/yomitori:infer \
    --role arn:aws:iam::${AWS_ACCOUNT_ID}:role/service-role/AmazonSageMaker-ExecutionRole \
    --instance_type ml.g5.xlarge
```

**インスタンスタイプの選択:**

| インスタンスタイプ | GPU | VRAM | 料金（東京） | 用途 |
|---|---|---|---|---|
| `ml.g5.xlarge` | 1× A10G | 24GB | $1.36/時間 | 最小構成・テスト用 |
| `ml.g5.2xlarge` | 1× A10G | 24GB | $1.85/時間 | 本番小規模 |
| `ml.g4dn.xlarge` | 1× T4 | 16GB | $0.80/時間 | コスト重視 |
| `ml.g4dn.2xlarge` | 1× T4 | 16GB | $0.94/時間 | 本番小規模 |

> **推奨**: テストは `ml.g5.xlarge`、本番は `ml.g5.2xlarge` 以上を推奨。

**デプロイ完了まで約5〜10分かかります。**

成功時の出力例:
```
Endpoint name: yomitori-2026-07-12-xxxx

Test with:
  predictor.predict(open('image.jpg', 'rb').read(),
      initial_args={'ContentType': 'image/jpeg'})
```

---

## 手順5: エンドポイントのテスト

### Python（SageMaker SDK）でテスト

```python
# test_endpoint.py
from sagemaker.predictor import Predictor
from sagemaker.serializers import DataSerializer
import json

# エンドポイント名を指定（手順4の出力で確認）
ENDPOINT_NAME = "yomitori-2026-07-12-xxxx"  # ← 実際の名前に変更
AWS_REGION = "ap-northeast-1"

predictor = Predictor(
    endpoint_name=ENDPOINT_NAME,
    sagemaker_session=None,  # デフォルトセッションを使用
)
predictor.serializer = DataSerializer(content_type="image/jpeg")

# 画像ファイルを読んで推論
with open("data/samples/sample_license.jpg", "rb") as f:
    result = predictor.predict(f.read())

print(json.dumps(json.loads(result), ensure_ascii=False, indent=2))
```

実行:
```bash
python test_endpoint.py
```

### curlでテスト

```bash
# エンドポイントのURLを取得
ENDPOINT_URL="https://runtime.sagemaker.${AWS_REGION}.amazonaws.com/endpoints/${ENDPOINT_NAME}/invocations"

# 画像を送信して推論
curl -s -X POST ${ENDPOINT_URL} \
    -H "Content-Type: image/jpeg" \
    --data-binary @data/samples/sample_license.jpg | python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))"
```

> **注意**: curlで直接アクセスする場合はAWS SigV4署名が必要です。 SageMaker SDK（Python）の使用を推奨します。

### 推論結果の確認

```json
{
  "status": "partial",
  "document_type": "driver_license_front",
  "fields": {
    "name": {
      "value": "山田太郎",
      "confidence": 0.92,
      "low_confidence": false
    },
    "license_number": {
      "value": "010203040506",
      "confidence": 0.95,
      "low_confidence": false,
      "validation_passed": true
    }
  },
  "overall_confidence": 0.89,
  "preprocessing": {
    "homography_applied": true,
    "corrected_image_size": [2400, 1512],
    "fallback_used": false
  },
  "warnings": []
}
```

---

## 手順6: エンドポイントの運用

### エンドポイントの状態確認

```bash
aws sagemaker describe-endpoint --endpoint-name ${ENDPOINT_NAME}
# → "EndpointStatus": "InService" / "Creating" / "Failed"
```

### エンドポイントの削除（不要になったら）

```bash
aws sagemaker delete-endpoint --endpoint-name ${ENDPOINT_NAME}
aws sagemaker delete-endpoint-config --endpoint-config-name ${ENDPOINT_NAME}-config
aws sagemaker delete-model --model-name ${ENDPOINT_NAME}-model
```

> **重要**: エンドポイントは稼働中ずっと課金されます。不要になったら必ず削除してください。

### モデルのバージョンアップ

新しいモデル（v2）でエンドポイントを更新する手順:

```bash
# 1. v2のmodel.tar.gzをS3にアップロード
aws s3 cp model.tar.gz s3://yomitori-models-<your-name>/driver_license/v2/model.tar.gz

# 2. 新しいエンドポイントをデプロイ（Blue-Green Deployment）
docker compose run --rm dev python -m scripts.cloud_deploy \
    --model_data s3://yomitori-models-<your-name>/driver_license/v2/model.tar.gz \
    --ecr_image ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/yomitori:infer \
    --role arn:aws:iam::${AWS_ACCOUNT_ID}:role/service-role/AmazonSageMaker-ExecutionRole \
    --instance_type ml.g5.xlarge

# 3. 新エンドポイントの動作確認後、旧エンドポイントを削除
aws sagemaker delete-endpoint --endpoint-name <旧エンドポイント名>
```

### Auto Scalingの設定（本番運用）

```bash
# Auto Scalingターゲットを登録
aws application-autoscaling register-scalable-target \
    --service-namespace sagemaker \
    --resource-id endpoint/${ENDPOINT_NAME}/variant/AllTraffic \
    --scalable-dimension sagemaker:variant:DesiredInstanceCount \
    --min-capacity 1 \
    --max-capacity 4

# スケーリングポリシーを設定
aws application-autoscaling put-scaling-policy \
    --policy-name yomitori-scaling-policy \
    --service-namespace sagemaker \
    --resource-id endpoint/${ENDPOINT_NAME}/variant/AllTraffic \
    --scalable-dimension sagemaker:variant:DesiredInstanceCount \
    --policy-type TargetTrackingScaling \
    --target-tracking-scaling-policy-configuration \
        '{"TargetValue": 50.0, "PredefinedMetricSpecification": {"PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"}}'
```

---

## コスト目安

### デプロイ時の費用（1回）

| 項目 | 費用 |
|---|---|
| S3アップロード（model.tar.gz 約500MB） | $0.01/月 |
| ECRプッシュ（イメージ約5GB） | $0.25/月 |
| エンドポイント作成（初回起動） | 約10分分のGPU料金 |

### 稼働時の費用（月額）

| 構成 | インスタンス | 時間単価 | 月額（24時間稼働） |
|---|---|---|---|
| テスト | ml.g5.xlarge | $1.36/h | 約$980 |
| 本番小規模 | ml.g5.2xlarge | $1.85/h | 約$1,332 |
| コスト重視 | ml.g4dn.xlarge | $0.80/h | 約$576 |

> **コスト削減のコツ**:
> - テスト時は使い終わったらすぐエンドポイントを削除
> - 本番ではAuto Scalingを設定し、夜間はインスタンス数を減らす
> - `ml.g4dn.xlarge`（T4 GPU）でも推論速度は十分な場合が多い

---

## トラブルシューティング

### エンドポイントのステータスが "Failed" になる

```bash
# エンドポイントの詳細を確認
aws sagemaker describe-endpoint --endpoint-name ${ENDPOINT_NAME}

# エンドポイントのログを確認
aws logs describe-log-streams --log-group-name /aws/sagemaker/Endpoints/${ENDPOINT_NAME}
aws logs get-log-events \
    --log-group-name /aws/sagemaker/Endpoints/${ENDPOINT_NAME} \
    --log-stream-name <log-stream-name>
```

**よくある原因:**
- IAMロールにS3アクセス権限がない → `AmazonS3ReadOnlyAccess` を追加
- ECRイメージにアクセスできない → ロールにECRアクセス権限を追加
- model.tar.gz が破損 → 再作成して再アップロード

### ECRプッシュで "no basic auth credentials" エラー

```bash
# 再ログイン
aws ecr get-login-password | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
```

### 推論がタイムアウトする

```bash
# インスタンスタイプを上げる
docker compose run --rm dev python -m scripts.cloud_deploy \
    --model_data s3://... \
    --ecr_image ... \
    --role ... \
    --instance_type ml.g5.2xlarge  # より大きなインスタンス
```

### 古いエンドポイントの一括削除

```bash
# 全エンドポイントの一覧
aws sagemaker list-endpoints --status-equal InService

# 特定プレフィックスのエンドポイントを一括削除
for ep in $(aws sagemaker list-endpoints --query 'Endpoints[?EndpointName.contains(@, `yomitori`)].EndpointName' --output text); do
    echo "Deleting: $ep"
    aws sagemaker delete-endpoint --endpoint-name $ep
done
```