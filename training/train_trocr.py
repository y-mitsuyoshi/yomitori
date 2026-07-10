"""TrOCR fine-tuning script (SageMaker-compatible).

Uses HuggingFace Transformers Trainer API to fine-tune
``microsoft/trocr-base-printed`` on synthetic Japanese document line crops.
Designed to run on RTX 5070 (12 GB VRAM) with fp16 + gradient checkpointing.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.dataset import TrOCRLinedataset, collate_fn
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_cer(eval_pred, processor) -> dict:
    """Compute Character Error Rate (CER) for evaluation.

    Args:
        eval_pred: HuggingFace EvalPrediction.
        processor: TrOCRProcessor for decoding.

    Returns:
        Dict with ``cer`` metric.
    """
    import jiwer
    from transformers import EvalPrediction

    predictions, labels = eval_pred
    pred_texts = processor.batch_decode(predictions, skip_special_tokens=True)
    label_ids = labels[label_ids != -100] if hasattr(labels, "__getitem__") else labels  # type: ignore
    ref_texts = processor.batch_decode([label_ids], skip_special_tokens=True)  # type: ignore

    cer = jiwer.cer(ref_texts, pred_texts) if len(ref_texts) == len(pred_texts) else float("nan")
    return {"cer": cer}


def main() -> int:
    """CLI entry point for TrOCR fine-tuning.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR")
    parser.add_argument("--data_dir", type=str, required=True, help="Training data dir")
    parser.add_argument("--eval_dir", type=str, default=None, help="Eval data dir")
    parser.add_argument("--output_dir", type=str, default="/opt/ml/model")
    parser.add_argument("--base_model", type=str, default="microsoft/trocr-base-printed")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
    )

    processor = TrOCRProcessor.from_pretrained(args.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(args.base_model)

    # Configure for generation
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    if args.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()

    train_ds = TrOCRLinedataset(args.data_dir, processor)
    eval_ds = None
    if args.eval_dir:
        eval_ds = TrOCRLinedataset(args.eval_dir, processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds else "no",
        logging_steps=50,
        save_total_limit=3,
        predict_with_generate=True,
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
    )

    logger.info("Starting training: %d epochs, batch=%d", args.epochs, args.batch_size)
    trainer.train()

    logger.info("Saving model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # Save training metadata
    with (Path(args.output_dir) / "training_info.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Training complete. Model saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())