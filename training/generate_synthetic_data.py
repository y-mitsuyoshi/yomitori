"""Synthetic document image generator for TrOCR training data.

Generates fake driver's-license-like images with random text, distortions,
and noise. Uses Japanese system fonts (e.g. Noto Sans CJK JP).
"""

import argparse
import json
import random
import string
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Dummy data (does not represent real people)
_DUMMY_NAMES = [
    "山田太郎", "佐藤花子", "鈴木一郎", "田中美咲", "渡辺健太",
    "伊藤百合子", "高橋直樹", "中村陽子", "小林大輔", "加藤翔",
]
_DUMMY_ADDRESSES = [
    "東京都千代田区霞が関1-1-1",
    "大阪府大阪市中央区大手前1-2-3",
    "神奈川県横浜市中区桜丘1-5-10",
    "愛知県名古屋市中区三の丸3-2-1",
    "京都府京都市上京区烏丸通一条1-1",
]
_ERAS = ["昭和", "平成", "令和"]
_LICENSE_TYPES = ["普通", "中型", "大型", "自動二輪", "原付"]
_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/ipa/ipagp.ttf",
    "/usr/share/fonts/ipa/ipam.ttf",
]


def _find_font() -> str:
    """Find an available Japanese font.

    Returns:
        Path to the first available font.

    Raises:
        FileNotFoundError: If no font is found.
    """
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return p
    raise FileNotFoundError(
        "No Japanese font found. Install Noto Sans CJK JP or IPA fonts."
    )


def _random_license_number() -> str:
    """Generate a 12-digit license number.

    Returns:
        Random 12-digit string.
    """
    return "".join(random.choices(string.digits, k=12))


def _random_era_date(era: str) -> str:
    """Generate a random date in Japanese-era format.

    Args:
        era: Era name.

    Returns:
        Date string like ``"昭和61年5月1日生"``.
    """
    year = random.randint(1, 30)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{era}{year}年{month}月{day}日"


def _apply_distortions(image: np.ndarray) -> np.ndarray:
    """Apply random tilt, noise, brightness, and reflection overlays.

    Args:
        image: Input image.

    Returns:
        Augmented image.
    """
    h, w = image.shape[:2]

    # Rotation ±15°
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Brightness/contrast variation
    alpha = random.uniform(0.8, 1.2)
    beta = random.uniform(-30, 30)
    image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Gaussian noise
    if random.random() < 0.5:
        noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)

    # Partial reflection overlay
    if random.random() < 0.3:
        overlay = image.copy()
        cv2.ellipse(
            overlay,
            (random.randint(0, w), random.randint(0, h)),
            (random.randint(50, 200), random.randint(20, 80)),
            0, 0, 360,
            (200, 200, 200),
            -1,
        )
        cv2.addWeighted(overlay, 0.15, image, 0.85, 0, image)

    return image


def _draw_field(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    label: str,
    value: str,
    pos: tuple[int, int],
    label_color: tuple[int, int, int] = (0, 0, 0),
    value_color: tuple[int, int, int] = (20, 20, 20),
) -> tuple[int, str]:
    """Draw a label + value field on the image.

    Args:
        draw: PIL ImageDraw object.
        font: Font to use.
        label: Label string.
        value: Value string.
        pos: (x, y) top-left position.
        label_color: Label text color.
        value_color: Value text color.

    Returns:
        Tuple of (y_offset, full_text) for label pairing.
    """
    x, y = pos
    draw.text((x, y), label, fill=label_color, font=font)
    bbox = draw.textbbox((x, y), label, font=font)
    value_x = bbox[2] + 8
    draw.text((value_x, y), value, fill=value_color, font=font)
    return y, f"{label} {value}"


def generate_one(
    width: int = 2400,
    height: int = 1512,
    font_path: str | None = None,
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Generate a single synthetic driver's-license-like image.

    Args:
        width: Image width.
        height: Image height.
        font_path: Path to font file.

    Returns:
        Tuple of (image_bgr, list_of_(line_text, field_name)).
    """
    if font_path is None:
        font_path = _find_font()
    font = ImageFont.truetype(font_path, 48)
    font_small = ImageFont.truetype(font_path, 32)

    pil = Image.new("RGB", (width, height), (240, 240, 245))
    draw = ImageDraw.Draw(pil)

    # Photo placeholder (right side)
    draw.rectangle(
        [width - 700, 80, width - 80, 600],
        fill=(200, 200, 210),
        outline=(100, 100, 100),
        width=3,
    )
    draw.text((width - 480, 320), "写真", fill=(120, 120, 120), font=font)

    # Fields
    name = random.choice(_DUMMY_NAMES)
    birth_era = random.choice(_ERAS)
    birth_date = _random_era_date(birth_era) + "生"
    address = random.choice(_DUMMY_ADDRESSES)
    issue_era = random.choice(_ERAS)
    issue_date = _random_era_date(issue_era)
    expiry_era = random.choice(_ERAS)
    expiry_date = _random_era_date(expiry_era)
    license_number = _random_license_number()
    license_type = random.choice(_LICENSE_TYPES)

    lines: list[tuple[str, str]] = []
    y = 60
    for label, value, fname in [
        ("氏名", name, "name"),
        ("生年月日", birth_date, "birth_date"),
        ("住所", address, "address"),
        ("交付", issue_date, "issue_date"),
        ("有効期限", expiry_date, "expiry_date"),
        ("条件等", "眼鏡等", "conditions"),
        ("免許証番号", license_number, "license_number"),
        ("免許種類", license_type, "license_type"),
    ]:
        draw.text((60, y), label, fill=(0, 0, 0), font=font)
        bbox = draw.textbbox((60, y), label, font=font)
        draw.text((bbox[2] + 16, y), value, fill=(20, 20, 20), font=font)
        lines.append((f"{label} {value}", fname))
        y += 100

    # Header
    draw.text((60, height - 80), "運転免許証", fill=(0, 0, 100), font=font_small)

    image = np.array(pil)
    image = _apply_distortions(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, lines


def crop_lines_from_image(
    image: np.ndarray,
    line_texts: list[tuple[str, str]],
) -> list[tuple[np.ndarray, str, str]]:
    """Crop text lines from a synthetic image (simple fixed layout).

    Args:
        image: Full document image (BGR).
        line_texts: List of (full_text, field_name).

    Returns:
        List of (line_image, text, field_name).
    """
    h, w = image.shape[:2]
    line_h = h // len(line_texts) if line_texts else h
    crops: list[tuple[np.ndarray, str, str]] = []
    for i, (text, fname) in enumerate(line_texts):
        y0 = i * line_h
        y1 = (i + 1) * line_h
        crop = image[y0:y1, :]
        crops.append((crop, text, fname))
    return crops


def main() -> int:
    """CLI entry point for synthetic data generation.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic document images for training"
    )
    parser.add_argument(
        "--document_type",
        default="driver_license_front",
        help="Document type to generate",
    )
    parser.add_argument("--count", type=int, default=500, help="Number of images")
    parser.add_argument(
        "--output",
        default="data/synthetic/driver_license/",
        help="Output directory",
    )
    parser.add_argument("--font", default=None, help="Font file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.output)
    (out / "images").mkdir(parents=True, exist_ok=True)

    labels: dict[str, str] = {}
    for i in range(args.count):
        image, lines = generate_one(font_path=args.font)
        crops = crop_lines_from_image(image, lines)
        for j, (crop, text, _fname) in enumerate(crops):
            name = f"{i:05d}_{j}.png"
            cv2.imwrite(str(out / "images" / name), crop)
            labels[name] = text

    with (out / "labels.json").open("w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    logger.info("Generated %d documents (%d line crops) in %s", args.count, len(labels), out)
    print(f"Generated {len(labels)} line crops → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())