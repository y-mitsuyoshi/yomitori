# 段階的（インクリメンタル）追加学習 & 推論テスト手順書

本手順書は、PCのディスク空き容量を節約しながら段階的に学習データを入れ替え、精度99%のモデルへと段階的に進化させていく手順と、学習した各モデルバージョンをテストするコマンドをまとめたものです。

---

## 🏛️ 全体フローの概要

1. **データ一時保存による容量節約**: 
   学習用データは1回あたり **3万枚（約66GB）** だけ生成し、学習が完了した瞬間に自動的に画像を全削除して空き容量を戻します。
2. **モデル重みの引き継ぎ（継続学習）**:
   前回のステップの学習済みモデル（例: `v2.1`）を読み込んで次の学習を開始するため、短時間で効率よく精度が上昇します。
3. **モデルの命名規則**:
   起点となるメジャーバージョン（例: `v2`）に対して、小数点刻みのマイナーバージョン（`v2.1`, `v2.2`, `v2.3`, `v2.4`...）で段階的に出力されます。

---

## 🏃‍♂️ 段階的追加学習の実行手順

### 1回目：v2 モデルからスタートし、v2.4 へ上げる (合計12万枚分)

1. **スクリプトの実行権限を確認** (初回のみ):
   ```bash
   chmod +x scripts/run_incremental_training.sh
   ```
2. **スクリプトの設定を確認**:
   [scripts/run_incremental_training.sh](file:///home/yuma/projects/yomitori/scripts/run_incremental_training.sh) の上部パラメータが以下になっていることを確認します。
   ```bash
   START_SEED=1000                    # 初期のシード値
   CURRENT_MODEL="/opt/ml/model/v2"   # 起点にするv2モデルのパス
   VERSION_PREFIX="2"                 # 接頭辞 (v2.x)
   START_STEP_INDEX=1                 # 保存はv2.1からスタート
   ```
3. **学習の実行**:
   ```bash
   ./scripts/run_incremental_training.sh
   ```
   * **所要時間**: 約 6 〜 8 時間
   * **最終出力**: `/opt/ml/model/v2.4` （Dockerボリューム内に保存）

---

### 2回目：v2.4 からスタートし、v2.8 へ上げる (さらに12万枚分)

1. **スクリプトのパラメータ書き換え**:
   [scripts/run_incremental_training.sh](file:///home/yuma/projects/yomitori/scripts/run_incremental_training.sh) の上部設定を次のように変更します。
   ```bash
   START_SEED=2000                      # ★シードを変更（重複防止）
   CURRENT_MODEL="/opt/ml/model/v2.4"   # ★起点に前回の成果物v2.4を指定
   VERSION_PREFIX="2"                   # 接頭辞はそのまま (v2.x)
   START_STEP_INDEX=5                   # ★保存はv2.5からスタート
   ```
2. **学習の実行**:
   ```bash
   ./scripts/run_incremental_training.sh
   ```
   * **所要時間**: 約 6 〜 8 時間
   * **最終出力**: `/opt/ml/model/v2.8`

---

## 🧪 バージョンごとの推論テスト方法

学習が完了した各モデルバージョン（`v2.1` や `v2.4` など）にサンプル画像を通し、文字認識の精度を確認する手順です。

### 手順 1: テスト対象のモデルを指定して推論サーバーを起動

環境変数 `YOMITORI_MODEL_DIR` でコンテナ内のモデルディレクトリを切り替えて、FastAPIサーバーを起動します。

> [!NOTE]
> 起動するコマンドに渡すモデルパス（`v2.1`, `v2.2` など）を適宜変更してください。

* **v2.1 モデルをテストする場合**:
  ```bash
  docker compose run --rm -p 8080:8080 -e YOMITORI_MODEL_DIR=/opt/ml/model/v2.1 serve
  ```
* **v2.4 モデルをテストする場合**:
  ```bash
  docker compose run --rm -p 8080:8080 -e YOMITORI_MODEL_DIR=/opt/ml/model/v2.4 serve
  ```

※このコマンドを実行すると、ターミナルがサーバー起動状態でロック（待機）されます。

---

### 手順 2: 別ターミナルからテストリクエストを送信

ホスト側で**別のターミナル**を開き、以下の `curl` コマンドでサンプル画像を送信します。

* **サンプル画像1でテスト（表面全体）**:
  ```bash
  curl -s -X POST http://localhost:8080/invocations \
      -H "Content-Type: image/jpeg" \
      --data-binary @data/samples/sample_license.jpg | python3 -m json.tool
  ```
* **サンプル画像2でテスト（表面別パターン）**:
  ```bash
  curl -s -X POST http://localhost:8080/invocations \
      -H "Content-Type: image/jpeg" \
      --data-binary @data/samples/sample_license2.jpg | python3 -m json.tool
  ```

### 手順 3: サーバーの停止
テストが終了したら、手順1のサーバーを実行しているターミナルで **`Ctrl + C`** を押してコンテナを停止します。

---

## 💾 成果物（モデル）のエクスポート

特定のバージョン（例: `v2.4`）が十分に高精度になり、バックアップやデプロイを行いたい場合は、以下のコマンドでホスト側のプロジェクトルートに圧縮ファイルを書き出します。

```bash
# v2.4 をエクスポートする場合
docker compose run --rm dev bash -c "cd /opt/ml/model/v2.4 && tar czf /opt/ml/code/model_v2.4.tar.gz --exclude='checkpoint-*' ."
```

* 生成されるファイル: `model_v2.4.tar.gz` (サイズ: 約 500〜700 MB)
* この圧縮ファイルは Googleドライブ などに安全にアップロードして保管することができます。
