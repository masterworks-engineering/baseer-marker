"""Layout + OCR predictors backed by a remote Chandra OCR 2 server (vLLM / any
OpenAI-compatible endpoint). Drop-in replacements for surya's ``LayoutPredictor``
and ``RecognitionPredictor`` - same call signatures and result shapes, but no
torch/transformers: the model runs on the server.

Chandra answers a page with HTML blocks carrying ``data-bbox`` (0-1000 grid) and
``data-label``. One such call gives layout and text together.
"""

from __future__ import annotations

import base64
import io
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional

from marker.logger import get_logger
from marker.remote.schema import (
    BlockOCRResult,
    LayoutBox,
    LayoutResult,
    PageOCRResult,
)
from marker.remote.util import detect_repeat_loop
from marker.settings import settings
from PIL import Image

logger = get_logger()

# Chandra label -> marker's canonical (surya-style) layout label
CHANDRA_LABEL_MAP = {
    "Caption": "Caption",
    "Footnote": "Footnote",
    "Equation-Block": "Equation",
    "List-Group": "ListGroup",
    "Page-Header": "PageHeader",
    "Page-Footer": "PageFooter",
    "Image": "Picture",
    "Section-Header": "SectionHeader",
    "Table": "Table",
    "Text": "Text",
    "Complex-Block": "Figure",
    "Code-Block": "Code",
    "Form": "Form",
    "Table-Of-Contents": "TableOfContents",
    "Figure": "Figure",
    "Chemical-Block": "ChemicalBlock",
    "Diagram": "Diagram",
    "Bibliography": "Bibliography",
    "Blank-Page": "BlankPage",
    "block": "Text",
}

# Marker extracts these as images; their text is a description, not content.
IMAGE_LABELS = {"Picture", "Figure", "Diagram"}

_FENCE = re.compile(r"^```(?:html)?\s*|\s*```$", re.IGNORECASE)


def _image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _unwrap_block_html(raw: str) -> str:
    """Block-mode answers still come wrapped in ``<div data-bbox data-label>``;
    keep only the inner content so downstream (table/equation) code sees it."""
    from bs4 import BeautifulSoup

    raw = _FENCE.sub("", raw.strip())
    soup = BeautifulSoup(raw, "html.parser")
    top = [t for t in soup.children if getattr(t, "name", None)]
    if top and all(t.name == "div" and t.get("data-label") for t in top):
        return "".join(t.decode_contents() for t in top).strip()
    return raw


def _flatten_table_cells(html: str) -> str:
    """Cells arrive as ``<td><p>a</p><p>b</p></td>``; marker's markdown table
    writer would print those tags literally, so join paragraphs with ``<br>``."""
    if "<table" not in html:
        return html
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for cell in soup.find_all(["td", "th"]):
        paras = cell.find_all("p")
        for i, p in enumerate(paras):
            if i < len(paras) - 1:
                p.insert_after(soup.new_tag("br"))
            p.unwrap()
    return str(soup)


def _box_polygon(x0: float, y0: float, x1: float, y1: float) -> List[List[float]]:
    x1 = max(x1, x0 + 1)
    y1 = max(y1, y0 + 1)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


class ChandraClient:
    """Thin sync client: one image + prompt in, raw model text out."""

    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI(
            base_url=settings.CHANDRA_BASE_URL,
            api_key=settings.CHANDRA_API_KEY,
            timeout=settings.CHANDRA_TIMEOUT,
            max_retries=0,  # we retry ourselves (repeat-loop aware)
        )
        self.model = settings.CHANDRA_MODEL_NAME

    def generate(self, image: Image.Image, prompt_type: str) -> Optional[str]:
        """Returns raw model text, or None on failure / unrecoverable repeat loop."""
        from chandra.model.util import scale_to_fit
        from chandra.prompts import PROMPT_MAPPING
        from openai import BadRequestError

        content = [
            {
                "type": "image_url",
                "image_url": {"url": _image_to_data_url(scale_to_fit(image))},
            },
            {"type": "text", "text": PROMPT_MAPPING[prompt_type]},
        ]
        temperature, top_p = 0.0, 0.1
        for attempt in range(settings.CHANDRA_MAX_RETRIES + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=settings.CHANDRA_MAX_TOKENS,
                    temperature=temperature,
                    top_p=top_p,
                )
                raw = completion.choices[0].message.content or ""
                if not detect_repeat_loop(raw):
                    return raw
                logger.warning(
                    "Chandra output looped (attempt %d); retrying", attempt + 1
                )
            except (
                BadRequestError
            ) as exc:  # deterministic (e.g. max_tokens > model len): no retry
                logger.error("Chandra server rejected the request: %s", exc)
                return None
            except Exception as exc:  # network / server error
                logger.warning(
                    "Chandra request failed (attempt %d): %s", attempt + 1, exc
                )
                time.sleep(min(2 * (attempt + 1), 10))
            # Retry hotter so greedy decoding does not repeat the same loop
            temperature = min(0.2 * (attempt + 1), 0.8)
            top_p = 0.95
        return None


class _Base:
    disable_tqdm = False

    def __init__(self, client: ChandraClient | None = None):
        self.client = client or ChandraClient()

    def _map(self, fn: Callable, items: list) -> list:
        if len(items) <= 1:
            return [fn(i) for i in items]
        with ThreadPoolExecutor(max_workers=settings.CHANDRA_MAX_WORKERS) as pool:
            return list(pool.map(fn, items))

    @staticmethod
    def _parse_blocks(raw: str, image: Image.Image):
        from chandra.output import parse_layout

        return parse_layout(raw, image)


class ChandraLayoutPredictor(_Base):
    """Layout from a full-page Chandra call (blocks + labels + reading order)."""

    def __call__(self, images: List[Image.Image], **_) -> List[LayoutResult]:
        return self._map(self._layout_one, images)

    def _layout_one(self, image: Image.Image) -> LayoutResult:
        w, h = image.size
        image_bbox = [0.0, 0.0, float(w), float(h)]
        raw = self.client.generate(image, "ocr_layout")
        if raw is None:
            return LayoutResult(bboxes=[], image_bbox=image_bbox, error=True)
        try:
            blocks = self._parse_blocks(raw, image)
        except Exception as exc:
            logger.warning("Could not parse Chandra layout output: %s", exc)
            return LayoutResult(bboxes=[], image_bbox=image_bbox, raw=raw, error=True)

        boxes = []
        for block in blocks:
            label = CHANDRA_LABEL_MAP.get(block.label, "Text")
            x0, y0, x1, y1 = block.bbox
            tokens = int(len(block.content) / 3)
            boxes.append(
                LayoutBox(
                    polygon=_box_polygon(x0, y0, x1, y1),
                    label=label,
                    raw_label=block.label,
                    position=len(boxes),
                    count=max(50, round(tokens / 50) * 50),
                )
            )
        if not boxes and raw.strip():
            # Text came back but nothing parsed: treat as one full-page text block
            boxes.append(
                LayoutBox(
                    polygon=_box_polygon(0, 0, w, h),
                    label="Text",
                    raw_label="block",
                    position=0,
                )
            )
        return LayoutResult(bboxes=boxes, image_bbox=image_bbox, raw=raw)


class ChandraRecognitionPredictor(_Base):
    """OCR: full-page (``full_page=True``) or per layout box (``full_page=False``)."""

    def __call__(
        self,
        images: List[Image.Image],
        layout_results: Optional[List[LayoutResult]] = None,
        full_page: bool = False,
        **_,
    ) -> List[PageOCRResult]:
        layouts = layout_results or [None] * len(images)
        jobs = list(zip(images, layouts))
        fn = self._full_page if full_page else self._blocks
        return self._map(lambda job: fn(*job), jobs)

    # -- full page --------------------------------------------------------
    def _full_page(
        self, image: Image.Image, layout: Optional[LayoutResult]
    ) -> PageOCRResult:
        w, h = image.size
        image_bbox = [0.0, 0.0, float(w), float(h)]
        raw = self.client.generate(image, "ocr_layout")
        blocks: List[BlockOCRResult] = []
        parsed = []
        if raw is not None:
            try:
                parsed = self._parse_blocks(raw, image)
            except Exception as exc:
                logger.warning("Could not parse Chandra page output: %s", exc)
        if not parsed:
            # Whole-page call failed: fall back to per-block OCR if we have a layout
            if layout is not None and layout.bboxes:
                return self._blocks(image, layout)
            return PageOCRResult(blocks=[], image_bbox=image_bbox)

        for idx, block in enumerate(parsed):
            label = CHANDRA_LABEL_MAP.get(block.label, "Text")
            x0, y0, x1, y1 = block.bbox
            skipped = label in IMAGE_LABELS and not settings.CHANDRA_KEEP_FIGURE_HTML
            blocks.append(
                BlockOCRResult(
                    polygon=_box_polygon(x0, y0, x1, y1),
                    label=label,
                    raw_label=block.label,
                    reading_order=idx,
                    html="" if skipped else _flatten_table_cells(block.content),
                    skipped=skipped,
                )
            )
        return PageOCRResult(blocks=blocks, image_bbox=image_bbox)

    # -- per block --------------------------------------------------------
    def _blocks(self, image: Image.Image, layout: LayoutResult) -> PageOCRResult:
        w, h = image.size
        results: List[BlockOCRResult | None] = [None] * len(layout.bboxes)

        def run(idx: int) -> None:
            box = layout.bboxes[idx]
            base = dict(
                polygon=box.polygon,
                label=box.label,
                raw_label=box.raw_label,
                reading_order=box.position,
            )
            if box.label in IMAGE_LABELS:
                results[idx] = BlockOCRResult(**base, skipped=True)
                return
            x0, y0, x1, y1 = box.bbox
            crop = image.crop(
                (
                    max(0, int(x0)),
                    max(0, int(y0)),
                    min(w, int(x1) + 1),
                    min(h, int(y1) + 1),
                )
            )
            raw = self.client.generate(crop, "ocr")
            if raw is None:
                results[idx] = BlockOCRResult(**base, error=True)
                return
            results[idx] = BlockOCRResult(
                **base, html=_flatten_table_cells(_unwrap_block_html(raw))
            )

        self._map(run, list(range(len(layout.bboxes))))
        return PageOCRResult(blocks=results, image_bbox=[0.0, 0.0, float(w), float(h)])
