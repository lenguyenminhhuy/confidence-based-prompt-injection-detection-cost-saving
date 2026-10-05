"""Text normalisation shared by the near-duplicate screens.

Used by scripts/data/cross_eval_dedup.py before the character 5-gram Jaccard
comparison: zero-width spaces removed, lower-cased, punctuation replaced by a
space, whitespace collapsed.
"""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")
ZWSP = "​"


def _normalize(text: str) -> str:
    text = text.replace(ZWSP, "").lower()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()
