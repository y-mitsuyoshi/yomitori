"""TrOCR fine-tuning script (SageMaker-compatible).

Uses HuggingFace Transformers Trainer API to fine-tune
TrOCR on synthetic document line crops.
Designed to run on RTX 5070 (12 GB VRAM) with fp16 + gradient checkpointing.

多言語対応: デコーダのトークナイザーを多言語対応のものに差し替えることで、
日本語・英語・多言語のテキスト認識が可能になる。
"""

import argparse
import json
from pathlib import Path

import torch

from training.dataset import TrOCRLineDataset, collate_fn
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_multilingual_processor(base_model: str, decoder_tokenizer: str):
    """TrOCRProcessorを構築する。

    ベースモデルの画像プロセッサー + 指定した多言語トークナイザーを組み合わせる。

    Args:
        base_model: TrOCRベースモデル名（画像エンコーダ用）。
        decoder_tokenizer: デコーダトークナイザー名。

    Returns:
        TrOCRProcessor インスタンス。
    """
    from transformers import AutoTokenizer, TrOCRProcessor

    processor = TrOCRProcessor.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(decoder_tokenizer)
    
    # BERT tokenizer lacks bos/eos tokens, map them to cls/sep tokens
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.bos_token_id = tokenizer.cls_token_id
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.sep_token
        tokenizer.eos_token_id = tokenizer.sep_token_id

    processor.tokenizer = tokenizer
    return processor


def build_multilingual_model(base_model: str, decoder_tokenizer: str):
    """VisionEncoderDecoderModelを構築する。

    ベースモデルがローカルパス（継続学習）の場合は、保存済みのトークナイザーを
    そのまま使用し、語彙リサイズをスキップする。
    HuggingFaceモデル名の場合は、デコーダの語彙を指定したトークナイザーに合わせてリサイズする。

    Args:
        base_model: TrOCRベースモデル名またはローカルパス。
        decoder_tokenizer: デコーダトークナイザー名。

    Returns:
        (model, processor) のタプル。
    """
    from pathlib import Path
    from transformers import (
        AutoTokenizer,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
    )

    model = VisionEncoderDecoderModel.from_pretrained(base_model)
    processor = TrOCRProcessor.from_pretrained(base_model)

    # 継続学習かどうかを判定（ベースモデルがローカルディレクトリの場合）
    is_continuation = Path(base_model).exists() and (Path(base_model) / "tokenizer.json").exists()

    if is_continuation:
        # 継続学習: 保存済みのトークナイザーをそのまま使用（語彙リサイズ不要）
        logger.info("  Continuing from local model — using saved tokenizer (no vocab resize)")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        processor.tokenizer = tokenizer
    else:
        # 初回学習: 新しいトークナイザーに差し替え + 語彙リサイズ
        tokenizer = AutoTokenizer.from_pretrained(decoder_tokenizer)

        # BERT tokenizer lacks bos/eos tokens, map them to cls/sep tokens
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.cls_token
            tokenizer.bos_token_id = tokenizer.cls_token_id
        if tokenizer.eos_token is None:
            tokenizer.eos_token = tokenizer.sep_token
            tokenizer.eos_token_id = tokenizer.sep_token_id

        processor.tokenizer = tokenizer

        # デコーダの語彙をトークナイザーに合わせてリサイズ
        model.config.decoder.vocab_size = len(tokenizer)
        model.decoder.resize_token_embeddings(len(tokenizer))

        # 生成用の設定を更新
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.vocab_size = len(tokenizer)

        # generation_config も明示的に更新する
        if model.generation_config is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            model.generation_config.decoder_start_token_id = tokenizer.bos_token_id
            model.generation_config.eos_token_id = tokenizer.eos_token_id

    return model, processor


def compute_cer(eval_pred, processor) -> dict:
    """Compute Character Error Rate (CER) for evaluation.

    Args:
        eval_pred: HuggingFace EvalPrediction.
        processor: TrOCRProcessor for decoding.

    Returns:
        Dict with ``cer`` metric.
    """
    import jiwer

    import numpy as np

    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    pred_texts = processor.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id or 0)
    ref_texts = processor.batch_decode(labels, skip_special_tokens=True)

    cer = jiwer.cer(ref_texts, pred_texts) if len(ref_texts) == len(pred_texts) else float("nan")
    return {"cer": cer}


def main() -> int:
    """CLI entry point for TrOCR fine-tuning.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR (multilingual)")
    parser.add_argument("--data_dir", type=str, required=True, help="Training data dir")
    parser.add_argument("--eval_dir", type=str, default=None, help="Eval data dir")
    parser.add_argument("--output_dir", type=str, default="/opt/ml/model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/trocr-base-printed",
        help="TrOCR base model (encoder + image processor)",
    )
    parser.add_argument(
        "--decoder_tokenizer",
        type=str,
        default="xlm-roberta-base",
        help="Decoder tokenizer for multilingual support "
             "(default: xlm-roberta-base, 100 languages, MIT license). "
             "Alternative: 'cl-tohoku/bert-base-japanese-v3' (Japanese only, Apache 2.0).",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--learning_rate", type=float, default=None,
        help="Learning rate. Auto: 5e-5 for initial training, 1e-5 for continuation.",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing (0.0=off, 0.1=recommended)")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience (0=off)")
    parser.add_argument("--eval_split", type=float, default=0.1, help="Fraction of training data to use for evaluation (0=off)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 学習率の自動設定（継続学習の場合は低くする）
    from pathlib import Path
    is_continuation = Path(args.base_model).exists() and (Path(args.base_model) / "tokenizer.json").exists()
    if args.learning_rate is None:
        args.learning_rate = 1e-5 if is_continuation else 5e-5
        logger.info("  Auto learning_rate: %s (continuation=%s)", args.learning_rate, is_continuation)

    torch.manual_seed(args.seed)

    logger.info("Building model:")
    logger.info("  Base model: %s", args.base_model)
    logger.info("  Decoder tokenizer: %s", args.decoder_tokenizer)

    model, processor = build_multilingual_model(args.base_model, args.decoder_tokenizer)

    if args.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()

    train_ds = TrOCRLineDataset(args.data_dir, processor)
    eval_ds = None
    if args.eval_dir:
        eval_ds = TrOCRLineDataset(args.eval_dir, processor)
    elif args.eval_split > 0:
        from torch.utils.data import random_split
        total = len(train_ds)
        eval_size = max(1, int(total * args.eval_split))
        train_size = total - eval_size
        train_ds, eval_ds = random_split(
            train_ds, [train_size, eval_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
        logger.info("Split: train=%d, eval=%d", train_size, eval_size)

    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    callbacks = []
    if eval_ds and args.early_stopping_patience > 0:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=0.001,
        ))
        logger.info("Early stopping enabled (patience=%d)", args.early_stopping_patience)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        label_smoothing_factor=args.label_smoothing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds else "no",
        logging_steps=50,
        save_total_limit=3,
        predict_with_generate=True,
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="cer" if eval_ds else None,
        greater_is_better=False if eval_ds else None,
        report_to=[],
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=lambda batch: collate_fn(batch, processor),
        processing_class=processor.tokenizer,
        compute_metrics=lambda pred: compute_cer(pred, processor),
        callbacks=callbacks,
    )

    logger.info("Starting training: %d epochs, batch=%d", args.epochs, args.batch_size)
    trainer.train()

    logger.info("Saving model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    with (Path(args.output_dir) / "training_info.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Training complete. Model saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())