"""Text normalization utilities."""

import re
import unicodedata

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Japanese era → Western year conversion table
_ERA_TABLE: dict[str, int] = {
    "明治": 1868,
    "大正": 1912,
    "昭和": 1926,
    "平成": 1989,
    "令和": 2019,
}

_FULLWIDTH_DIGIT_MAP = {0xFF10 + i: 0x30 + i for i in range(10)}  # ０-９ → 0-9


def fullwidth_to_halfwidth(text: str) -> str:
    """Convert full-width ASCII characters to half-width.

    Args:
        text: Input text possibly containing full-width chars.

    Returns:
        Text with full-width ASCII converted to half-width.
    """
    return unicodedata.normalize("NFKC", text)


def remove_label(text: str, label: str) -> str:
    """Remove a leading label string from text.

    Args:
        text: Recognized text (e.g. ``"氏名 山田太郎"``).
        label: Label to remove (e.g. ``"氏名"``).

    Returns:
        Text with the label prefix removed.
    """
    pattern = r"^" + re.escape(label) + r"[\s　:：]*"
    return re.sub(pattern, "", text)


def apply_whitelist(text: str, whitelist: str) -> str:
    """Keep only characters in the whitelist.

    Args:
        text: Input text.
        whitelist: String of allowed characters.

    Returns:
        Filtered text.
    """
    if not whitelist:
        return text
    allowed = set(whitelist)
    return "".join(c for c in text if c in allowed)


def japanese_era_to_iso(text: str) -> tuple[str, str | None]:
    """Convert a Japanese-era date string to ISO-8601 (YYYY-MM-DD).

    Handles forms like:
      - 昭和61年5月1日生
      - 令和7年12月31日

    Args:
        text: Date string in Japanese era format.

    Returns:
        Tuple of (raw_text, iso_string_or_None).
    """
    raw = text.strip()
    m = re.search(
        r"(明治|大正|昭和|平成|令和)(\d{1,2}|元)年(\d{1,2})月(\d{1,2})日",
        raw,
    )
    if not m:
        logger.debug("japanese_era_to_iso: no match in %r", raw)
        return raw, None

    era = m.group(1)
    year_str = m.group(2)
    month = int(m.group(3))
    day = int(m.group(4))

    if year_str == "元":
        year = 1
    else:
        year = int(year_str)

    base = _ERA_TABLE.get(era)
    if base is None:
        return raw, None

    western = base + year - 1
    try:
        iso = f"{western:04d}-{month:02d}-{day:02d}"
    except (ValueError, TypeError):
        return raw, None

    return raw, iso