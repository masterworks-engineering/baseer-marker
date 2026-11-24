"""Single import point for the model-facing types the builders/processors use.

With ``MODEL_BACKEND=surya`` these are surya's own classes (upstream behaviour,
untouched); with ``chandra`` they are the torch-free equivalents in this package.
Keeping the switch here means the builders differ from upstream by one import line.
"""

from typing import Any

from marker.settings import settings

if settings.MODEL_BACKEND == "surya":
    from surya.recognition.schema import BlockOCRResult, PageOCRResult  # noqa: F401
    from surya.layout.schema import LayoutBox, LayoutResult  # noqa: F401
    from surya.recognition import _detect_repeat_loop  # noqa: F401
else:
    from marker.remote.schema import (
        BlockOCRResult,
        PageOCRResult,
        LayoutBox,
        LayoutResult,
    )  # noqa: F401
    from marker.remote.util import detect_repeat_loop as _detect_repeat_loop  # noqa: F401

# Predictor classes are only used as annotations; the objects come from models.py
LayoutPredictor = RecognitionPredictor = FastLayoutPredictor = OCRErrorPredictor = Any

__all__ = [
    "BlockOCRResult",
    "PageOCRResult",
    "LayoutBox",
    "LayoutResult",
    "LayoutPredictor",
    "RecognitionPredictor",
    "FastLayoutPredictor",
    "OCRErrorPredictor",
    "_detect_repeat_loop",
]
