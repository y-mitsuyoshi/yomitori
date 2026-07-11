"""PyTorch Dataset for TrOCR fine-tuning on synthetic line images."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class TrOCRLineDataset(Dataset):
    """Dataset of cropped text-line images paired with ground-truth text.

    Expects a directory with:
      - ``images/`` containing .png files
      - ``labels.json``: ``{"image_name": "text", ...}``

    Args:
        data_dir: Path to the dataset directory.
        processor: HuggingFace TrOCRProcessor instance.
        max_target_length: Maximum target sequence length.
    """

    def __init__(
        self,
        data_dir: str | Path,
        processor,
        max_target_length: int = 128,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_target_length = max_target_length

        labels_path = self.data_dir / "labels.json"
        with labels_path.open("r", encoding="utf-8") as f:
            self.labels: dict[str, str] = json.load(f)

        self.image_files = sorted(self.labels.keys())
        if not self.image_files:
            raise ValueError(f"No labeled images found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        """Get a single training sample.

        Args:
            idx: Index.

        Returns:
            Dict with pixel_values and labels tensors.
        """
        from PIL import Image

        img_name = self.image_files[idx]
        img_path = self.data_dir / "images" / img_name
        text = self.labels[img_name]

        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


def collate_fn(batch: list[dict], processor) -> dict:
    """Collate function for DataLoader.

    Args:
        batch: List of samples.
        processor: TrOCRProcessor for batching.

    Returns:
        Batched dict with pixel_values, labels, and decoder_input_ids.
    """
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])

    tokenizer = processor.tokenizer
    pad_id = tokenizer.pad_token_id or 0
    bos_id = getattr(tokenizer, "bos_token_id", None) or getattr(tokenizer, "cls_token_id", None) or 0

    decoder_input_ids = labels.clone()
    decoder_input_ids[decoder_input_ids == -100] = pad_id
    shifted = decoder_input_ids.new_zeros(decoder_input_ids.shape)
    shifted[:, 1:] = decoder_input_ids[:, :-1].clone()
    shifted[:, 0] = bos_id
    shifted[shifted == pad_id] = pad_id

    return {"pixel_values": pixel_values, "labels": labels, "decoder_input_ids": shifted}