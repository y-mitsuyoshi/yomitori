"""Field-level character whitelist application."""

import re


def apply_whitelist(text: str, whitelist: str) -> str:
    """Remove characters not in the whitelist.

    Args:
        text: Input text.
        whitelist: String of allowed characters.

    Returns:
        Filtered text containing only characters in the whitelist.
    """
    if not whitelist:
        return text
    allowed = set(whitelist)
    return "".join(c for c in text if c in allowed)


def apply_blacklist(text: str, blacklist: str) -> str:
    """Remove characters in the blacklist.

    Args:
        text: Input text.
        blacklist: String of characters to remove.

    Returns:
        Filtered text.
    """
    if not blacklist:
        return text
    banned = set(blacklist)
    return "".join(c for c in text if c not in banned)