"""Group membership helpers for fairness reporting (bias_detector, metrics)."""
from __future__ import annotations

import re

# Whole-word acronyms plus spelled-out names. Plain substring matching ("iit" in name)
# counted KIIT University as an IIT and "Community"/"Infinity"/"United" as NITs.
_TOP_INSTITUTION_RE = re.compile(
    r"\b(?:iit|iiit|nit|bits)\b"
    r"|indian institute of (?:information )?technology"
    r"|national institute of technology"
    r"|birla institute of technology",
    re.IGNORECASE,
)


def is_top_institution(name: str | None) -> bool:
    """IIT / IIIT / NIT / BITS."""
    return bool(name) and _TOP_INSTITUTION_RE.search(name or "") is not None


def city_key(location: str | None) -> str:
    """Normalise a free-text location ("Bengaluru, Karnataka") to a city key ("bengaluru")."""
    if not location:
        return ""
    return location.split(",")[0].strip().lower()
