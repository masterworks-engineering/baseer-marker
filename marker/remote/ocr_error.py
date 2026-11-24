"""Torch-free stand-in for surya's DistilBert OCR-error classifier.

Flags text-layer text that is garbled (missing glyph mappings, replacement
chars, mostly symbols) so those pages/blocks get re-read by the VLM.
"""

import re
from typing import List

from marker.remote.schema import OCRErrorDetectionResult

_REPLACEMENT = "�"
_PUA = re.compile(r"[-]")  # private-use glyphs = unmapped fonts


class HeuristicOCRErrorPredictor:
    disable_tqdm = False

    def __init__(
        self,
        bad_char_fraction: float = 0.08,
        min_alnum_fraction: float = 0.45,
        min_chars: int = 20,
    ):
        self.bad_char_fraction = bad_char_fraction
        self.min_alnum_fraction = min_alnum_fraction
        self.min_chars = min_chars

    def _score(self, text: str) -> float:
        """Rough P(bad) in [0, 1]."""
        stripped = "".join(text.split())
        if len(stripped) < self.min_chars:
            return 0.0
        bad = stripped.count(_REPLACEMENT) + len(_PUA.findall(stripped))
        if bad / len(stripped) >= self.bad_char_fraction:
            return 1.0
        # Non-Latin scripts count as letters via isalpha(), so Arabic/CJK pass.
        alnum = sum(c.isalnum() for c in stripped)
        if alnum / len(stripped) < self.min_alnum_fraction:
            return 0.9
        return 0.0

    def __call__(self, texts: List[str], batch_size: int | None = None):
        scores = [self._score(t) for t in texts]
        labels = ["bad" if s >= 0.5 else "good" for s in scores]
        return OCRErrorDetectionResult(texts=texts, labels=labels, scores=scores)
