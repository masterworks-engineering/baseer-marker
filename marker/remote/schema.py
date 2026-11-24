"""Result types shaped like surya's, so marker's builders/processors are unchanged."""

from typing import List, Optional

from pydantic import BaseModel

from marker.schema.polygon import PolygonBox


class LayoutBox(PolygonBox):
    label: str  # canonical (surya-style) label, e.g. "SectionHeader"
    raw_label: str  # label as the model emitted it
    position: int  # reading-order index
    count: int = 0  # estimated OCR token budget for the block
    confidence: Optional[float] = None


class LayoutResult(BaseModel):
    bboxes: List[LayoutBox]
    image_bbox: List[float]
    raw: Optional[str] = None
    error: bool = False


class BlockOCRResult(PolygonBox):
    label: str
    raw_label: str = ""
    reading_order: int = 0
    html: str = ""
    skipped: bool = False
    error: bool = False


class PageOCRResult(BaseModel):
    blocks: List[BlockOCRResult]
    image_bbox: List[float]


class OCRErrorDetectionResult(BaseModel):
    texts: List[str]
    labels: List[str]
    scores: List[float] = []
