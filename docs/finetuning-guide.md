# ファインチューニング手順書

日本語運転免許証OCRモデルのファインチューニング手順を段階的に説明します。

## 目次

- [前提条件](#前提条件)
- [v1: 動作確認（約1時間）](#v1-動作確認約35分)
- [v2: 商用基礎精度（約7時間）](#v2-商用基礎精度約4時間)
- [v3: 実用精度（約20時間）](#v3-実用精度約11時間)
- [v4: 最高精度（約33時間）](#v4-最高精度約19時間)
- [継続学習（v1→v2→v3）](#継続学習v1v2v3)
- [推論サーバーでの検証](#推論サーバーでの検証)
- [トラブルシューティング](#トラブルシューティング)
- [データ構成の内訳](#データ構成の内訳)

---

## 前提条件

| 項目 | 要件 |
|---|---|
| OS | Linux（Ubuntu 22.04 / WSL2） |
| GPU | NVIDIA RTX 5070 (VRAM 12GB) 以上 |
| NVIDIA Driver | 550+ (CUDA 12.8対応) |
| Docker | Docker Engine 24+ / Docker Compose v2 |

### 初回セットアップ（初回のみ）

```bash
# ベースイメージをビルド（KEN_ALL.CSV + フォント含む・10〜15分）
docker compose build base

# 各サービスイメージをビルド
docker compose build
```

---

## v1: 動作確認（約40分）

**目的**: パイプライン全体が正しく動作することを確認する。

### 手順 1: 合成データを生成（1,000枚）

```bash
docker compose run --rm dev python -m training.generate_synthetic_data \
    --count 1000 \
    --output /opt/ml/code/data/synthetic/driver_license/ \
    --seed 42
```

**生成される内容:**
- 1,000枚の免許証風画像 × 10行 = 10,000行のテキスト画像
- 約270種の名字 × 129種の名前 = 約35,000通りの氏名パターン
- 全国約12万件の住所（KEN_ALL.CSV自動ダウンロード）からランダム選択
- 44都道府県の本籍データ
- 47都道府県の公安委員会名
- 複数フォント（Noto Sans/Serif CJK、IPA、IPAex、Takao）
- 画像Augmentation（回転・ノイズ・ぼかし・JPEG劣化・透視変形・照明ムラ）

**免許証の全項目（10項目）:**
氏名・生年月日・住所・本籍・交付年月日・有効期限・免許の条件・免許証番号・免許の種類・公安委員会名

**確認:**
```bash
# 生成されたファイル数を確認
ls data/synthetic/driver_license/images/ | wc -l
# → 10000

# ラベルを確認
head -5 data/synthetic/driver_license/labels.json
# → {"00000_0.png": "氏名 山田太郎", "00000_1.png": "生年月日 昭和40年3月15日生", ...}
```

### 手順 2: ファインチューニングを実行（3エポック・約30分）

```bash
docker compose run --rm train python -m training.train_trocr \
    --data_dir /opt/ml/code/data/synthetic/driver_license/ \
    --output_dir /opt/ml/model/japanese \
    --decoder_tokenizer cl-tohoku/bert-base-japanese-v3 \
    --epochs 3 \
    --batch_size 4 \
    --fp16
```

**実行中のログの見方:**
```
# 学習の進行状況
0%|          | 0/6000 [00:00<?, ?it/s]     ← 開始
50%|█████     | 3000/6000 [15:00<15:00]   ← 進行中
100%|██████████| 6000/6000 [30:00<00:00]  ← 完了

# 評価結果（毎エポック終了時）
{'eval_loss': 7.27, 'eval_cer': 0.947, 'epoch': 1.0}  ← CER（文字誤り率）
```

> **CERとは**: 文字誤り率。0.0 = 完全正解、1.0 = 完全不正解。
> v1ではCER 0.8〜0.95程度が想定範囲（少データ・少エポックのため）。

### 手順 3: 学習済みモデルを確認

```bash
# モデルファイルの確認
docker compose run --rm dev ls -la /opt/ml/model/japanese/

# training_info.jsonの確認
docker compose run --rm dev cat /opt/ml/model/japanese/training_info.json
```

**出力例:**
```
config.json
model.safetensors
preprocessor_config.json
tokenizer.json
sentencepiece.bpe.model
generation_config.json
training_args.bin
training_info.json
checkpoint-2000/
checkpoint-4000/
checkpoint-6000/
```

### 手順 4: SageMaker Local Mode でテスト

v1の学習済みモデルを使って、SageMaker Local Mode で推論テストを行います。
これは SageMaker クラウド環境と同じ仕組み（Dockerコンテナ経由で推論）をローカルで再現するテストです。

#### 4-1: model.tar.gz を作成

SageMaker はモデルを `model.tar.gz` 形式で扱います。学習済みモデルを圧縮します。

```bash
# japanese/ ディレクトリの中身を model.tar.gz に圧縮
docker compose run --rm dev bash -c "cd /opt/ml/model/japanese && tar czf /opt/ml/code/model.tar.gz --exclude='checkpoint-*' ."
```

#### 4-2: SageMaker Local Mode で推論テスト

SageMaker Local Mode は Docker-in-Docker を使用します。
`dev` サービスにはDockerソケットがマウント済みなので、そのまま実行できます:

```bash
# SageMaker Local Mode で推論テスト（デフォルト方式）
docker compose run --rm dev python -m scripts.local_deploy --sample data/samples/sample_license.jpg
```

**実行の流れ:**
1. `model.tar.gz` を読み込み
2. `yomitori:infer` イメージでローカルエンドポイントを構築（SageMaker SDK経由）
3. サンプル画像をエンドポイントに送信
4. 推論結果をJSONで標準出力に表示

> **model.tar.gz が無い場合**: スクリプトが自動的に `/opt/ml/model/japanese` から作成します。

**出力例:**
```json
{
  "status": "partial",
  "document_type": "driver_license_front",
  "fields": {
    "name": {
      "value": "山田太郎",
      "confidence": 0.253,
      "low_confidence": true
    },
    ...
  },
  "overall_confidence": 0.2,
  "preprocessing": {
    "homography_applied": true,
    "corrected_image_size": [2400, 1512],
    "fallback_used": false
  },
  "warnings": ["birth_date: validation failed - 和暦生年月日"]
}
```

> **v1の注意点**: データ量が少ないため、認識結果はほとんど意味のない文字列になります。
> これは正常な動作です。v1の目的は「パイプライン全体が動くこと」の確認です。

#### 4-3: 別のサンプル画像でテスト

```bash
# 別のサンプル画像で推論
docker compose run --rm dev python -m scripts.local_deploy --sample data/samples/sample_license2.jpg
```

#### 4-4: serve方式でのテスト（代替）

SageMaker Local Modeの代わりに `docker compose up serve` + HTTPリクエスト方式も使えます:

```bash
docker compose run --rm dev python -m scripts.local_deploy --sample data/samples/sample_license.jpg --method serve
```

#### 4-5: クリーンアップ

SageMaker Local Mode はバックグラウンドでDockerコンテナを作成します。
テスト終了後にクリーンアップしてください:

```bash
# ローカルエンドポイントのコンテナを削除
docker ps -a --filter "name=sagemaker" --format "{{.ID}}" | xargs -r docker rm -f

# model.tar.gz を削除（不要な場合）
rm -f model.tar.gz
```

#### 4-6: （参考）2つの方式の違い

| | SageMaker Local Mode (`--method sagemaker`) | serve方式 (`--method serve`) |
|---|---|---|
| 用途 | クラウド移行前の最終検証 | 開発・迅速な検証 |
| 仕組み | SageMaker SDK がコンテナを構築・実行 | FastAPI サーバーが直接推論 |
| クラウド互換性 | あり（`instance_type` を変更するだけで移行） | なし |
| エンドポイント | SageMaker Predictor API | `/ping`, `/invocations` |
| 実行コマンド | `docker compose run --rm dev python -m scripts.local_deploy` | `docker compose run --rm dev python -m scripts.local_deploy --method serve` |
| デフォルト | ✅ こちらがデフォルト | |

迅速にテストしたい場合は `docker compose up serve` も使用できます:

```bash
docker compose up -d serve
curl -s http://localhost:8080/ping
curl -s -X POST http://localhost:8080/invocations \
    -H "Content-Type: image/jpeg" \
    --data-binary @data/samples/sample_license.jpg | python3 -m json.tool
docker compose down
```

---

## v2: 商用基礎精度（約7〜8時間）

**目的**: 商用レベルの基礎精度を達成する。

### 手順 1: 大量の合成データを生成（10,000枚）

```bash
docker compose run --rm dev python -m training.generate_synthetic_data \
    --count 10000 \
    --output /opt/ml/code/data/synthetic/driver_license_v2/ \
    --seed 100
```

> **注意**: v1とは別のディレクトリ（`driver_license_v2/`）に出力し、v1のデータを上書きしないようにします。

### 手順 2: ファインチューニングを実行（5エポック・約6〜7時間）

```bash
docker compose run --rm train python -m training.train_trocr \
    --data_dir /opt/ml/code/data/synthetic/driver_license_v2/ \
    --output_dir /opt/ml/model/v2 \
    --decoder_tokenizer cl-tohoku/bert-base-japanese-v3 \
    --epochs 5 \
    --batch_size 4 \
    --fp16
```

### 手順 3: 推論サーバーでテスト

```bash
# v2モデルを指定してサーバー起動
docker compose down
docker compose run --rm -p 8080:8080 -e YOMITORI_MODEL_DIR=/opt/ml/model/v2 serve

# 推論テスト
curl -s -X POST http://localhost:8080/invocations \
    -H "Content-Type: image/jpeg" \
    --data-binary @data/samples/sample_license.jpg | python3 -m json.tool

# サーバー停止（Ctrl+C）
```

---

## v3: 実用精度（約20時間）

**目的**: 商用デプロイ可能な精度を達成する。

### 手順 1: さらに大量のデータを生成（30,000枚）

```bash
# 通常データ 20,000枚
docker compose run --rm dev python -m training.generate_synthetic_data \
    --count 20000 \
    --output /opt/ml/code/data/synthetic/driver_license_v3/ \
    --seed 200

# 漢字ブーストデータ 10,000枚（未知の漢字対策）
docker compose run --rm dev python -m training.generate_synthetic_data \
    --count 10000 \
    --output /opt/ml/code/data/synthetic/driver_license_v3_kanji/ \
    --kanji_boost \
    --seed 300
```

### 手順 2: データを結合

```bash
# labels.json と images/ を結合
docker compose run --rm dev python -c "
import json, shutil
from pathlib import Path

base = Path('/opt/ml/code/data/synthetic/driver_license_v3')
kanji = Path('/opt/ml/code/data/synthetic/driver_license_v3_kanji')

# kanjiデータを結合
for img in (kanji / 'images').glob('*.png'):
    shutil.copy(img, base / 'images' / f'kanji_{img.name}')

# labels.jsonを結合
with open(base / 'labels.json', 'r') as f:
    main_labels = json.load(f)
with open(kanji / 'labels.json', 'r') as f:
    kanji_labels = json.load(f)

for k, v in kanji_labels.items():
    main_labels[f'kanji_{k}'] = v

with open(base / 'labels.json', 'w') as f:
    json.dump(main_labels, f, ensure_ascii=False, indent=2)

print(f'Combined: {len(main_labels)} line crops')
"
```

### 手順 3: ファインチューニング（5エポック・約18時間）

```bash
docker compose run --rm train python -m training.train_trocr \
    --data_dir /opt/ml/code/data/synthetic/driver_license_v3/ \
    --output_dir /opt/ml/model/v3 \
    --decoder_tokenizer cl-tohoku/bert-base-japanese-v3 \
    --epochs 5 \
    --batch_size 4 \
    --fp16
```

> **時間短縮のコツ**: 学習をバックグラウンドで実行し、`docker compose logs -f train` で進行状況を監視します。

---

## v4: 最高精度（約33時間）

**目的**: 最高精度のモデルを構築する。

### 手順 1: 最大規模のデータを生成（50,000枚）

```bash
# 通常データ 30,000枚
docker compose run --rm dev python -m training.generate_synthetic_data \
    --count 30000 \
    --output /opt/ml/code/data/synthetic/driver_license_v4/ \
    --seed 400

# 漢字ブーストデータ 20,000枚
docker compose run --rm dev python -m training.generate_synthetic_data \
    --count 20000 \
    --output /opt/ml/code/data/synthetic/driver_license_v4_kanji/ \
    --kanji_boost \
    --seed 500
```

### 手順 2: データを結合

v3と同様の手順でlabels.jsonとimages/を結合します。

### 手順 3: ファインチューニング（5エポック・約30時間）

```bash
docker compose run --rm train python -m training.train_trocr \
    --data_dir /opt/ml/code/data/synthetic/driver_license_v4/ \
    --output_dir /opt/ml/model/v4 \
    --decoder_tokenizer cl-tohoku/bert-base-japanese-v3 \
    --epochs 5 \
    --batch_size 4 \
    --fp16
```

> **注意**: 約33時間かかります。学習中にPCをスリープしないよう設定してください。また、途中で停止した場合はチェックポイントから再開できます。

---

## 継続学習（v1→v2→v3）

前回の学習済みモデルをベースに追加学習することで、学習時間を短縮できます。

```bash
# v1モデルをベースにv2を学習（v2をゼロから学習するより高速）
docker compose run --rm train python -m training.train_trocr \
    --data_dir /opt/ml/code/data/synthetic/driver_license_v2/ \
    --output_dir /opt/ml/model/v2 \
    --base_model /opt/ml/model/japanese \
    --decoder_tokenizer cl-tohoku/bert-base-japanese-v3 \
    --epochs 3 \
    --batch_size 4 \
    --fp16

# v2モデルをベースにv3を学習
docker compose run --rm train python -m training.train_trocr \
    --data_dir /opt/ml/code/data/synthetic/driver_license_v3/ \
    --output_dir /opt/ml/model/v3 \
    --base_model /opt/ml/model/v2 \
    --decoder_tokenizer cl-tohoku/bert-base-japanese-v3 \
    --epochs 2 \
    --batch_size 4 \
    --fp16
```

> **継続学習のメリット**: エポック数を減らせるため、学習時間を大幅に短縮できます。

---

## 推論サーバーでの検証

### 特定バージョンで推論サーバーを起動

```bash
# v1（japanese）で起動
docker compose up -d serve

# v2で起動
docker compose down
docker compose run --rm -p 8080:8080 -e YOMITORI_MODEL_DIR=/opt/ml/model/v2 serve

# v3で起動
docker compose down
docker compose run --rm -p 8080:8080 -e YOMITORI_MODEL_DIR=/opt/ml/model/v3 serve
```

### 推論テスト

```bash
# ヘルスチェック
curl -s http://localhost:8080/ping

# サンプル画像で推論
curl -s -X POST http://localhost:8080/invocations \
    -H "Content-Type: image/jpeg" \
    --data-binary @data/samples/sample_license.jpg | python3 -m json.tool

# 別のサンプル画像で推論
curl -s -X POST http://localhost:8080/invocations \
    -H "Content-Type: image/jpeg" \
    --data-binary @data/samples/sample_license2.jpg | python3 -m json.tool
```

### 推論結果の確認ポイント

```json
{
  "status": "partial",          // success=全フィールド正常 / partial=一部低信頼度 / failed=失敗
  "overall_confidence": 0.85,   // 全フィールドの平均信頼度
  "fields": {
    "name": {
      "value": "山田太郎",       // 認識されたテキスト
      "confidence": 0.92,        // このフィールドの信頼度
      "low_confidence": false    // true の場合は信頼度が低い（< 0.7）
    },
    "license_number": {
      "value": "010203040506",
      "confidence": 0.95,
      "validation_passed": true  // バリデーション結果
    }
  },
  "warnings": []                 // バリデーション失敗等の警告
}
```

---

## トラブルシューティング

### CUDA Out of Memory (OOM)

VRAM 12GBでOOMが発生した場合:

```bash
# バッチサイズを2に下げる
docker compose run --rm train python -m training.train_trocr \
    --data_dir /opt/ml/code/data/synthetic/driver_license/ \
    --output_dir /opt/ml/model/japanese \
    --decoder_tokenizer cl-tohoku/bert-base-japanese-v3 \
    --batch_size 2 \
    --gradient_accumulation_steps 2 \
    --epochs 3 --fp16
```

### 学習が途中で止まった場合

チェックポイントから再開できます:

```bash
# チェックポイントの確認
docker compose run --rm dev ls /opt/ml/model/japanese/checkpoint-*/

# チェックポイントをベースに再開
docker compose run --rm train python -m training.train_trocr \
    --data_dir /opt/ml/code/data/synthetic/driver_license/ \
    --output_dir /opt/ml/model/japanese \
    --base_model /opt/ml/model/japanese/checkpoint-6000 \
    --decoder_tokenizer cl-tohoku/bert-base-japanese-v3 \
    --epochs 2 --fp16
```

### 推論結果が空（status: "failed"）

1. 入力画像が `data/samples/` に正しく配置されているか確認
2. モデルが正しくロードされているかログを確認:
   ```bash
   docker compose logs serve | grep "Model loaded"
   ```
3. `preprocessing.fallback_used` が `true` の場合、ホモグラフィ補正に失敗しています

### Docker コンテナからGPUが見えない

```bash
# GPU認識の確認
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# 見えない場合
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## データ構成の内訳

### 合成データに含まれる多様性

| 項目 | v1（1,000枚） | v2（10,000枚） | v3（30,000枚） | v4（50,000枚） |
|---|---|---|---|---|
| 行数 | 10,000行 | 100,000行 | 300,000行 | 500,000行 |
| 名字 | 約270種 | 約270種 | 約270種 | 約270種 |
| 名前 | 129種 | 129種 | 129種 | 129種 |
| 氏名パターン | 約35,000通り | 約35,000通り | 約35,000通り | 約35,000通り |
| 住所 | 約12万件 | 約12万件 | 約12万件 | 約12万件 |
| 本籍 | 44都道府県 | 44都道府県 | 44都道府県 | 44都道府県 |
| 公安委員会名 | 47都道府県 | 47都道府県 | 47都道府県 | 47都道府県 |
| 免許種類 | 17種類 | 17種類 | 17種類 | 17種類 |
| 免許の条件 | 17種類 | 17種類 | 17種類 | 17種類 |
| フォント | 最大10種類 | 最大10種類 | 最大10種類 | 最大10種類 |
| 漢字ブースト | なし | なし | 10,000枚追加 | 20,000枚追加 |
| Augmentation | 回転・ノイズ・ぼかし・JPEG劣化・透視変形・照明ムラ | 同左 | 同左 | 同左 |
| 項目数 | 10項目 | 10項目 | 10項目 | 10項目 |

### 学習設定の比較

| 項目 | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| データ数 | 1,000枚 | 10,000枚 | 30,000枚 | 50,000枚 |
| エポック | 3 | 5 | 5 | 5 |
| バッチサイズ | 4 | 4 | 4 | 4 |
| FP16 | ✓ | ✓ | ✓ | ✓ |
| Early Stopping | patience=3 | patience=3 | patience=3 | patience=3 |
| Label Smoothing | 0.1 | 0.1 | 0.1 | 0.1 |
| Weight Decay | 0.01 | 0.01 | 0.01 | 0.01 |
| 学習率 | 5e-5 | 5e-5 | 5e-5 | 5e-5 |
| **想定時間** | **約1時間** | **約7時間** | **約20時間** | **約33時間** |


### 各バージョンの目標CER

| バージョン | 想定CER | 精度の目安 |
|---|---|---|
| v1 | 0.70〜0.95 | パイプライン動作確認レベル |
| v2 | 0.30〜0.60 | 商用基礎レベル（半分程度の文字が正解） |
| v3 | 0.10〜0.30 | 実用レベル（大部分の文字が正解） |
| v4 | 0.05〜0.15 | 高精度レベル（商用デプロイ可能） |

> **CERの目安**: 
> - 0.1 = 90%の文字が正解
> - 0.05 = 95%の文字が正解
> - 実用上は0.1以下が目標