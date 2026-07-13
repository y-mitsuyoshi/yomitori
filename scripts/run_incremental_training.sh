#!/usr/bin/env bash
#
# scripts/run_incremental_training.sh
# 
# 段階的（インクリメンタル）な追加学習を自動化するスクリプト。
# 各イテレーションごとに合成データを生成し、学習が完了したら自動的に
# ディスクから合成データを削除して空き容量を維持します。
#
# 使い方:
#   ./scripts/run_incremental_training.sh
#
# 初回実行前に:
#   chmod +x scripts/run_incremental_training.sh
#

set -euo pipefail

# --- 設定パラメータ ---
ITERATIONS=5                   # 繰り返す回数（1回実行につき1つのマイナーバージョンを上げる）
COUNT_REGULAR=1600             # 通常ドキュメント数（1600枚生成 * 10行 = 16,000行画像）
COUNT_KANJI=400                # 漢字ブーストドキュメント数（400枚生成 * 10行 = 4,000行画像）
EPOCHS=3                       # 学習エポック数
BATCH_SIZE=4                   # バッチサイズ
START_SEED=2000                # 初期シード値（次回は3000、その次は4000のように変えます）

# 【バージョン設定】
# 起点とする初期モデル（例: /opt/ml/model/v2.1）
CURRENT_MODEL="/opt/ml/model/v2.1"

# 新しく出力するモデルのバージョン接頭辞 (例: "2" と指定すると、v2.x と命名されます)
VERSION_PREFIX="2"
# 出力のインデックス (v2.2 を出力する場合は 2, v2.3 を出力する場合は 3)
START_STEP_INDEX=2

# フォルダパス定義（ホスト側相対パス）
TEMP_DIR="data/synthetic/temp_train"
TEMP_KANJI_DIR="data/synthetic/temp_train_kanji"
EVAL_DIR="data/synthetic/eval_set"


# ----------------------

echo "=== 段階的追加学習自動化スクリプト ==="
echo "総イテレーション数: ${ITERATIONS}"
echo "各ステップ通常カード数: ${COUNT_REGULAR}枚 (約 $((COUNT_REGULAR * 10))行画像), 漢字ブースト: ${COUNT_KANJI}枚 (約 $((COUNT_KANJI * 10))行画像)"
echo "エポック数: ${EPOCHS}, 学習率: 自動（継続学習: 1e-5）"
echo "起点のベースモデル: ${CURRENT_MODEL}"
echo "======================================"

# 1. 固定の評価用データセットがなければ作成 (100枚のカード = 1,000行画像)
if [ ! -d "${EVAL_DIR}" ] || [ ! -f "${EVAL_DIR}/labels.json" ]; then
    echo "[INFO] 評価用データセット（${EVAL_DIR}）が見つからないため、新規生成します..."
    mkdir -p "${EVAL_DIR}"
    docker compose run --rm dev python -m training.generate_synthetic_data \
        --count 100 \
        --output "/opt/ml/code/${EVAL_DIR}" \
        --seed 9999
    echo "[INFO] 評価用データセットの生成完了。"
else
    echo "[INFO] 既存の評価用データセット（${EVAL_DIR}）を使用します。"
fi

# 2. ループ実行
for ((i=1; i<=ITERATIONS; i++))
do
    SEED=$((START_SEED + i))
    STEP_INDEX=$((START_STEP_INDEX + i - 1))
    NEXT_MODEL="/opt/ml/model/v${VERSION_PREFIX}.${STEP_INDEX}"
    echo ""
    echo "--------------------------------------------------"
    echo "  イテレーション ${i} / ${ITERATIONS} (SEED: ${SEED})"
    echo "  ベースモデル: ${CURRENT_MODEL}"
    echo "  出力先モデル: ${NEXT_MODEL}"
    echo "--------------------------------------------------"

    # 一時フォルダのクリーンアップ
    rm -rf "${TEMP_DIR}" "${TEMP_KANJI_DIR}"
    mkdir -p "${TEMP_DIR}/images" "${TEMP_KANJI_DIR}/images"

    # A. 通常データの生成
    echo "[Step 1/4] 通常合成データを生成中 (${COUNT_REGULAR}枚)..."
    docker compose run --rm dev python -m training.generate_synthetic_data \
        --count "${COUNT_REGULAR}" \
        --output "/opt/ml/code/${TEMP_DIR}" \
        --seed "${SEED}"

    # B. 漢字ブーストデータの生成
    echo "[Step 2/4] 漢字ブーストデータを生成中 (${COUNT_KANJI}枚)..."
    docker compose run --rm dev python -m training.generate_synthetic_data \
        --count "${COUNT_KANJI}" \
        --output "/opt/ml/code/${TEMP_KANJI_DIR}" \
        --kanji_boost \
        --seed "$((SEED + 10000))"

    # C. データの結合
    echo "[Step 3/4] データをマージしています..."
    docker compose run --rm dev python -c "
import json, shutil
from pathlib import Path

base = Path('/opt/ml/code/${TEMP_DIR}')
kanji = Path('/opt/ml/code/${TEMP_KANJI_DIR}')

# 画像のコピー
for img in (kanji / 'images').glob('*.png'):
    shutil.copy(img, base / 'images' / f'kanji_{img.name}')

# labels.jsonの結合
with open(base / 'labels.json', 'r') as f:
    main_labels = json.load(f)
with open(kanji / 'labels.json', 'r') as f:
    kanji_labels = json.load(f)

for k, v in kanji_labels.items():
    main_labels[f'kanji_{k}'] = v

with open(base / 'labels.json', 'w') as f:
    json.dump(main_labels, f, ensure_ascii=False, indent=2)

print(f'Merge complete. Total lines: {len(main_labels)}')
"

    # D. トレーニングの実行
    #    --learning_rate を省略すると、train_trocr.py が自動で設定する
    #    （継続学習: 1e-5, 初回学習: 5e-5）
    echo "[Step 4/4] トレーニングを実行中..."
    docker compose run --rm train python -m training.train_trocr \
        --data_dir "/opt/ml/code/${TEMP_DIR}" \
        --eval_dir "/opt/ml/code/${EVAL_DIR}" \
        --output_dir "${NEXT_MODEL}" \
        --base_model "${CURRENT_MODEL}" \
        --decoder_tokenizer cl-tohoku/bert-base-japanese-v3 \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --fp16

    # E. 学習データの削除（空き容量確保）
    echo "[INFO] イテレーション ${i} の学習が完了しました。ディスク容量解放のため合成データを削除します..."
    rm -rf "${TEMP_DIR}" "${TEMP_KANJI_DIR}"
    # 新しいモデルフォルダ内の不要なチェックポイント（一時保存用）も削除してDocker容量を節約
    docker compose run --rm dev bash -c "rm -rf ${NEXT_MODEL}/checkpoint-*"
    echo "[INFO] ディスク容量解放完了。"

    # F. 1つ前のモデルを削除（最新＋起点のみ保持してDocker容量を節約）
    #    ※ 起点モデル（CURRENT_MODEL の初期値）は削除しない
    PREV_STEP_INDEX=$((STEP_INDEX - 1))
    PREV_MODEL="/opt/ml/model/v${VERSION_PREFIX}.${PREV_STEP_INDEX}"
    INITIAL_MODEL="/opt/ml/model/v2.1"  # 起点モデル（これは常に保持）
    if [ "${PREV_MODEL}" != "${INITIAL_MODEL}" ] && [ "${PREV_MODEL}" != "/opt/ml/model/v2" ]; then
        echo "[INFO] 古いモデル ${PREV_MODEL} を削除してディスク容量を回収します..."
        docker compose run --rm dev bash -c "rm -rf ${PREV_MODEL}"
    fi

    # 次のループのベースモデルを更新
    CURRENT_MODEL="${NEXT_MODEL}"
done

echo ""
echo "=============================================="
echo " 全ての段階的追加学習が正常に完了しました！"
echo " 最終モデル: ${CURRENT_MODEL}"
echo "=============================================="
