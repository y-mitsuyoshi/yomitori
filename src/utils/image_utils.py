"""Image decoding, encoding, and visualization utilities."""

import base64
import io
from typing import Any

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def decode_image(data: bytes) -> np.ndarray:
    """Decode raw image bytes (JPEG/PNG/etc.) into a BGR ndarray.

    Args:
        data: Raw image bytes.

    Returns:
        Image as a BGR ndarray (HxWxC).

    Raises:
        ValueError: If the bytes cannot be decoded.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    return img


def decode_base64_image(b64_str: str) -> np.ndarray:
    """Decode a base64-encoded image string into a BGR ndarray.

    Args:
        b64_str: Base64-encoded image data (with or without data URI prefix).

    Returns:
        Image as a BGR ndarray.
    """
    if "," in b64_str and b64_str.startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    return decode_image(raw)


def encode_image(image: np.ndarray, ext: str = ".png") -> bytes:
    """Encode an image ndarray to bytes.

    Args:
        image: BGR ndarray.
        ext: File extension / format (e.g. ``.png``, ``.jpg``).

    Returns:
        Encoded image bytes.
    """
    success, buf = cv2.imencode(ext, image)
    if not success:
        raise ValueError("Failed to encode image")
    return buf.tobytes()


def resize_image(
    image: np.ndarray,
    target_size: tuple[int, int],
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Resize an image to the given (width, height).

    Args:
        image: Input image.
        target_size: (width, height).
        interpolation: OpenCV interpolation flag.

    Returns:
        Resized image.
    """
    w, h = target_size
    return cv2.resize(image, (w, h), interpolation=interpolation)


def draw_bboxes(
    image: np.ndarray,
    bboxes: list[list[tuple[float, float]]],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on an image copy.

    Args:
        image: Input BGR image.
        bboxes: List of bboxes, each a list of (x, y) points (pixel coords).
        color: Box color (BGR).
        thickness: Line thickness.

    Returns:
        Image with drawn boxes.
    """
    out = image.copy()
    for box in bboxes:
        pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
    return out


def visualize_preprocessing(
    image: np.ndarray,
    gray: np.ndarray | None = None,
    binary: np.ndarray | None = None,
    contours_img: np.ndarray | None = None,
    corrected: np.ndarray | None = None,
) -> None:
    """Visualize preprocessing steps using matplotlib.

    This is intended for use in Jupyter notebooks or debug scripts.

    Args:
        image: Original image (BGR).
        gray: Grayscale image.
        binary: Binary image.
        contours_img: Image with contours drawn.
        corrected: Corrected (homography) image.
    """
    import matplotlib.pyplot as plt

    steps: list[tuple[str, np.ndarray]] = [("Original (BGR→RGB)", image)]
    if gray is not None:
        steps.append(("Grayscale", gray))
    if binary is not None:
        steps.append(("Binary (Otsu)", binary))
    if contours_img is not None:
        steps.append(("Contours", contours_img))
    if corrected is not None:
        steps.append(("Corrected", corrected))

    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, steps):
        if len(img.shape) == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def save_image(image: np.ndarray, path: str, ext: str = ".png") -> None:
    """Save an image to disk.

    Args:
        image: BGR image.
        path: Output file path.
        ext: File extension/format.
    """
    success = cv2.imwrite(path, image)
    if not success:
        raise IOError(f"Failed to write image to {path}")