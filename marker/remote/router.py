"""Per-page OCR-or-text-layer routing, ported from baseer-document-parser's
``marker_chandra`` router (heuristics + PDFium checks).

Decides, per page, whether pdftext's text layer can be trusted or the page must
be re-read by the VLM. It catches what the text-quality checks alone miss:
Latin-glyph junk, narrow label columns / ruled tables (structure loss), reversed
RTL bullet lines, unmapped fonts and meaningful embedded images.

Used by ``LineBuilder`` (via the ``page_router`` dependency) to set
``ocr_errors_detected`` - flagged pages go to full-page Chandra OCR.
"""

from __future__ import annotations

import ctypes
import functools
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from pdftext.postprocessing import merge_text

from marker.logger import get_logger

logger = get_logger()

PDFIUM_LOCK = threading.RLock()  # PDFium is not thread-safe


class _ContextError(Exception):
    def __init__(self, message: str = "", **context):
        super().__init__(message)
        self.context = context


class RoutingError(_ContextError):
    pass


class PdfExtractionError(_ContextError):
    pass


def _serialized(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with PDFIUM_LOCK:
            return func(*args, **kwargs)

    return wrapper


# ---- garble check (language independent) ----

_LATIN_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_VOWEL_RE = re.compile(r"[aeiouyAEIOUY]")
_LOWER_RE = re.compile(r"[a-z]")
_CASE_FLIP_RE = re.compile(r"[a-z][A-Z]")
_GARBLED_MIN_CHARS = 40
_GARBLED_MIN_SHARE = 0.12


def _word_looks_wrong(word: str) -> bool:
    """True when a Latin token is unlikely to be a real word.

    Either signal is sufficient: (1) a lowercase→uppercase transition
    mid-word ("AiAll", "crE"), or (2) no vowel — but the no-vowel check is
    applied ONLY to tokens that contain a lowercase letter (a real-word
    shape). All-uppercase tokens are treated as acronyms (PDF, CEO, SQL,
    HTTP) and never flagged, so acronym-dense prose doesn't masquerade as
    glyph junk. Broken-font junk still trips one of the two signals (it
    carries lowercase runs and/or case flips); normal prose trips neither.
    """
    if _CASE_FLIP_RE.search(word):
        return True
    if _LOWER_RE.search(word) and not _VOWEL_RE.search(word):
        return True
    return False


def text_is_gibberish(
    text: str,
    *,
    min_words: int = 5,
    ratio_threshold: float = 0.4,
) -> bool:
    """Language-independent check: does this text look like extracted-glyph junk?

    Scores only Latin-letter tokens, so genuine Arabic/Devanagari/CJK text
    (which yields no such tokens) is never flagged. Returns True when at
    least ``min_words`` Latin tokens are present and the fraction that look
    wrong (``_word_looks_wrong``) meets ``ratio_threshold``. Evaluated
    per text block by ``selective_chandra_recover`` so a clean English
    column cannot mask a garbled Arabic column on the same (two-column)
    page.
    """
    if not text:
        return False
    words = _LATIN_WORD_RE.findall(text)
    if len(words) < min_words:
        return False
    wrong = sum(1 for word in words if _word_looks_wrong(word))
    return wrong / len(words) >= ratio_threshold


def blocks_look_garbled(block_texts: list[str]) -> bool:
    """Language-independent judgment of one page's text, given block by block.

    Every parser that decides whether to trust a page's extracted text asks this
    one question. Glyph-substitution junk has no U+FFFD characters and reads as
    ordinary Latin letters, so character-class checks do not see it.
    """
    garbled_chars = sum(len(block) for block in block_texts if text_is_gibberish(block))
    return bool(garbled_chars) and (
        garbled_chars >= _GARBLED_MIN_CHARS
        or garbled_chars / max(len("\n".join(block_texts)), 1) >= _GARBLED_MIN_SHARE
    )


# --- thresholds -------------------------------------------------------

MIN_TEXT_CHARS = 20  # near-empty text layer -> escalate

# marker's PdfProvider defaults (providers/pdf.py)
OCR_SPACE_THRESHOLD = 0.7
OCR_NEWLINE_THRESHOLD = 0.6
OCR_ALPHANUM_THRESHOLD = 0.3
OCR_INVALID_CHAR_MIN = 6
OCR_INVALID_CHAR_FRACTION = 0.03

# our own line-overlap thresholds (not marker's raw values - see module docstring)
LINE_OVERLAP_IOU_THRESHOLD = 0.3  # two lines count as "overlapping" above this
LINE_OVERLAP_NEIGHBOR_THRESHOLD = (
    2  # a line overlapping more than this many others is suspect
)
LINE_OVERLAP_PAGE_FRACTION_THRESHOLD = (
    0.5  # escalate if more than this fraction of lines are suspect
)

# Table-structure proxy. On a clean born-digital PDF, table CELL TEXT reads
# fine via pdftext (no overlap, no bad ratios) - the thing that actually
# breaks is losing row/column STRUCTURE when we flatten to plain paragraphs.
# A real quirk found by inspecting evaluation documents' own pdftext output: a
# table row-number label (e.g. "4-1") that sits in its own narrow column
# often comes back from pdftext as its own small, disjoint block, separate
# from that row's content block - while a simple inline table (label and
# text sharing one block, e.g. "2-1\nRequirement text...") does not produce
# this pattern and is already handled fine directly. So: several small blocks
# on one page is a reasonable structural signal for "this page has real table
# structure that plain-paragraph rendering would destroy" - not a
# text-quality problem, so the other two checks miss it entirely. Page-margin
# header/footer text (running footers, page numbers) is excluded first since
# it is also narrow but is not table structure.
HEADER_FOOTER_MARGIN_FRACTION = 0.08  # top/bottom 8% of page height ignored
NARROW_BLOCK_WIDTH_FRACTION = 0.08  # block narrower than this fraction of page width
NARROW_BLOCK_COUNT_THRESHOLD = 1  # escalate if more than this many narrow blocks remain

# Table-grid signal. pdftext does not expose tables: it folds a whole ruled
# table (and the paragraph above it) into one ordinary block, so block geometry
# says nothing. What it does preserve is each cell as its own span, so a table
# row is a LINE whose spans are separated by horizontal gaps far wider than any
# word space (observed on real tables: 35-240pt; justified prose stays under
# ~10pt, a bullet marker plus its text ~10pt). A page is a table page when
# several such rows exist and their cells start at repeating x positions
# (columns) - a single wide gap (a right-aligned page label, a signature line)
# repeated nowhere else is not enough.
TABLE_CELL_GAP_PT = (
    18.0  # min horizontal gap between two spans to call them separate cells
)
TABLE_MIN_ROWS = 3  # row-like lines needed before the page can count as a table
TABLE_MIN_COLUMNS = 2  # x positions that must each recur on >= TABLE_MIN_ROWS rows
TABLE_COLUMN_X_TOLERANCE_PT = 3.0  # cell start x's within this are the same column

# RTL line-order signal. For a line that mixes a left-anchored LTR marker (a
# bullet or "1.") with Arabic/Hebrew text, pdfium can emit the line's spans in
# right-to-left order, i.e. the marker and any English label come out AFTER the
# RTL text ("... عام 1986م :content RTL / Bilingual•"), and the RTL chunks
# themselves arrive out of order. The text layer is simply wrong for that line
# - no rendering of it can recover the reading order - so the page needs OCR.
# A genuinely RTL bullet sits at the RIGHT edge, so its leftmost span is RTL
# text, not a marker, and does not trigger this.
_RTL_CHAR_RE = re.compile("[\u0590-\u08ff\ufb1d-\ufdff\ufe70-\ufeff]")
_LEADING_MARKER_RE = re.compile(r"^(?:[-*•◦▪‣●]|\d{1,3}[.)])(?:\s|$)")
RTL_ORDER_MIN_SPANS = 3

# TOC dot-leaders / form underscores are legitimate low-alphanumeric-density
# typography, not garbled OCR - collapse them before the ratio checks run
# (same fix marker applies for the same reason).
_LEADER_RUN_RE = re.compile(r"(?:[.·•…_-]\s*){3,}")


@dataclass
class RouteDecision:
    route: str  # "pdftext" | "chandra"
    reasons: List[str] = field(default_factory=list)


def _alphanum_ratio(text: str) -> float:
    text = text.replace(" ", "").replace("\n", "")
    if len(text) == 0:
        return 1.0
    return sum(1 for c in text if c.isalnum()) / len(text)


def _bad_text_ratio_reason(text: str) -> str | None:
    """Mirrors PdfProvider.detect_bad_ocr. Returns a reason string, or None if OK."""
    cleaned = _LEADER_RUN_RE.sub(" ", text)

    spaces = len(re.findall(r"\s+", cleaned))
    alpha_chars = len(re.sub(r"\s+", "", cleaned))
    if spaces / (alpha_chars + spaces + 1e-9) > OCR_SPACE_THRESHOLD:
        return "high_space_ratio"

    newlines = len(re.findall(r"\n+", cleaned))
    non_newlines = len(re.sub(r"\n+", "", cleaned))
    if newlines / (newlines + non_newlines + 1e-9) > OCR_NEWLINE_THRESHOLD:
        return "high_newline_ratio"

    if _alphanum_ratio(cleaned) < OCR_ALPHANUM_THRESHOLD:
        return "low_alphanum_ratio"

    invalid_chars = cleaned.count("�")
    if invalid_chars > max(
        OCR_INVALID_CHAR_MIN, len(cleaned) * OCR_INVALID_CHAR_FRACTION
    ):
        return "invalid_chars"

    return None


# Some producers drop the lam of "ال" (and of lam-alef ligatures) from the text
# layer: "الأمن" -> "األمن", "الاختبار" -> "االختبار". Correct Arabic almost
# never has an alef directly followed by another alef form, so a few of these
# on a page mean the layer is damaged even though every character is valid.
_ARABIC_LIGATURE_LOSS_RE = re.compile("\u0627[\u0627\u0623\u0625\u0622]")
ARABIC_LIGATURE_LOSS_MIN = 3


def _bbox_area(bbox: List[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _intersection_area(a: List[float], b: List[float]) -> float:
    min_x, min_y = max(a[0], b[0]), max(a[1], b[1])
    max_x, max_y = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, max_x - min_x) * max(0.0, max_y - min_y)


def _dense_overlap_fraction(line_bboxes: List[List[float]]) -> float:
    """Fraction of lines that overlap more than LINE_OVERLAP_NEIGHBOR_THRESHOLD
    other lines by more than LINE_OVERLAP_IOU_THRESHOLD of the smaller line's
    area. High on a dense table grid; low on normal flowing paragraph text.
    """
    n = len(line_bboxes)
    if n == 0:
        return 0.0

    areas = [_bbox_area(b) for b in line_bboxes]
    suspect = 0
    for i in range(n):
        neighbors = 0
        for j in range(n):
            if i == j:
                continue
            inter = _intersection_area(line_bboxes[i], line_bboxes[j])
            smaller = min(areas[i], areas[j]) or 1e-9
            if inter / smaller > LINE_OVERLAP_IOU_THRESHOLD:
                neighbors += 1
        if neighbors > LINE_OVERLAP_NEIGHBOR_THRESHOLD:
            suspect += 1
    return suspect / n


def _narrow_block_count(page: dict) -> int:
    page_width = page["width"]
    page_height = page["height"]
    top = HEADER_FOOTER_MARGIN_FRACTION * page_height
    bottom = (1 - HEADER_FOOTER_MARGIN_FRACTION) * page_height

    count = 0
    for block in page["blocks"]:
        x0, y0, x1, y1 = block["bbox"]
        if y1 <= top or y0 >= bottom:
            continue  # header/footer margin - not content structure
        if (x1 - x0) < NARROW_BLOCK_WIDTH_FRACTION * page_width:
            count += 1
    return count


def in_margin_band(line: dict, page: dict) -> bool:
    """True if the line sits entirely inside the page's header/footer band.
    Lines without a bbox are assumed to be content."""
    bbox = line.get("bbox")
    height = page.get("height")
    if not bbox or not height:
        return False
    return (
        bbox[3] <= HEADER_FOOTER_MARGIN_FRACTION * height
        or bbox[1] >= (1 - HEADER_FOOTER_MARGIN_FRACTION) * height
    )


def _row_cell_starts(line: dict) -> list[float] | None:
    """x0 of every cell on a line, or None if the line is not table-row-like.

    A line is row-like when at least two non-empty spans are separated by a
    gap of TABLE_CELL_GAP_PT or more. Spans without a bbox (hand-built test
    dicts, or a pdftext version that omits it) simply carry no signal.
    """
    spans = [s for s in line["spans"] if s.get("text", "").strip() and s.get("bbox")]
    if len(spans) < 2:
        return None
    spans.sort(key=lambda s: s["bbox"][0])
    if not any(
        nxt["bbox"][0] - cur["bbox"][2] >= TABLE_CELL_GAP_PT
        for cur, nxt in zip(spans, spans[1:])
    ):
        return None
    return [s["bbox"][0] for s in spans]


def _table_grid(page: dict) -> tuple[int, int] | None:
    """(row_count, column_count) if the page has a table-like grid, else None.

    Header/footer margins are excluded first, as in _narrow_block_count:
    running headers like "Title ........ Page 3" are wide-gapped but are not
    table structure.
    """
    page_height = page["height"]
    top = HEADER_FOOTER_MARGIN_FRACTION * page_height
    bottom = (1 - HEADER_FOOTER_MARGIN_FRACTION) * page_height

    rows: list[list[float]] = []
    for block in page["blocks"]:
        for line in block["lines"]:
            bbox = line.get("bbox")
            if bbox and (bbox[3] <= top or bbox[1] >= bottom):
                continue
            starts = _row_cell_starts(line)
            if starts:
                rows.append(starts)
    if len(rows) < TABLE_MIN_ROWS:
        return None

    # Cluster every cell-start x into columns (1-D, tolerance-based), then
    # count how many distinct ROWS touch each column.
    xs = sorted(x for starts in rows for x in starts)
    clusters: list[list[float]] = [[xs[0]]]
    for x in xs[1:]:
        if x - clusters[-1][-1] <= TABLE_COLUMN_X_TOLERANCE_PT:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    columns = 0
    for cluster in clusters:
        lo, hi = cluster[0], cluster[-1]
        rows_in_column = sum(1 for starts in rows if any(lo <= x <= hi for x in starts))
        if rows_in_column >= TABLE_MIN_ROWS:
            columns += 1
    if columns < TABLE_MIN_COLUMNS:
        return None
    return len(rows), columns


def _has_reversed_rtl_marker_line(page: dict) -> bool:
    """True if any content line has a leading LTR marker (bullet / ordinal) as
    its leftmost span, RTL text elsewhere, and its spans in right-to-left
    stream order - the signature of pdfium having reversed a mixed line."""
    for block in page["blocks"]:
        for line in block["lines"]:
            if in_margin_band(line, page):
                continue
            spans = [
                s for s in line["spans"] if s.get("text", "").strip() and s.get("bbox")
            ]
            if len(spans) < RTL_ORDER_MIN_SPANS:
                continue
            if not any(_RTL_CHAR_RE.search(s["text"]) for s in spans):
                continue
            x0s = [s["bbox"][0] for s in spans]
            if not all(a > b for a, b in zip(x0s, x0s[1:])):
                continue  # not strictly right-to-left in stream order
            leftmost = spans[-1]
            if _LEADING_MARKER_RE.match(leftmost["text"].strip() + " "):
                return True
    return False


def _block_texts(page: dict) -> list[str]:
    """The page's text one pdftext block at a time, so a clean column cannot
    average away a garbled one."""
    return [
        merge_text({"blocks": [block]}, sort=False, hyphens=False).strip()
        for block in page["blocks"]
    ]


def route_page(page: dict) -> RouteDecision:
    """Decide whether a pdftext page dict (from provider.extract_pages) can be
    trusted as-is, or needs full-page escalation to Chandra.

    Raises ``RoutingError`` (with the page number, if present on the dict) if
    the page dict is missing fields this function requires — that always
    means an upstream extraction bug, not a routing decision to make.
    """
    page_num = page.get("page") if isinstance(page, dict) else None
    try:
        text = merge_text(page, sort=False, hyphens=False)
        stripped = text.strip()

        if len(stripped) < MIN_TEXT_CHARS:
            return RouteDecision("chandra", ["near_empty_text_layer"])

        reasons = []

        ratio_reason = _bad_text_ratio_reason(text)
        if ratio_reason:
            reasons.append(ratio_reason)

        if blocks_look_garbled(_block_texts(page)):
            reasons.append("garbled_text")

        if len(_ARABIC_LIGATURE_LOSS_RE.findall(text)) >= ARABIC_LIGATURE_LOSS_MIN:
            reasons.append("arabic_ligature_loss")

        line_bboxes = [
            line["bbox"] for block in page["blocks"] for line in block["lines"]
        ]
        if line_bboxes:
            overlap_frac = _dense_overlap_fraction(line_bboxes)
            if overlap_frac > LINE_OVERLAP_PAGE_FRACTION_THRESHOLD:
                reasons.append(f"dense_line_overlap({overlap_frac:.2f})")

        narrow_blocks = _narrow_block_count(page)
        if narrow_blocks > NARROW_BLOCK_COUNT_THRESHOLD:
            reasons.append(f"narrow_label_blocks({narrow_blocks})")

        if _has_reversed_rtl_marker_line(page):
            reasons.append("rtl_marker_line_order")

        grid = _table_grid(page)
        if grid:
            reasons.append(f"table_grid_layout(rows={grid[0]},cols={grid[1]})")
    except (KeyError, TypeError, IndexError) as exc:
        raise RoutingError(f"malformed page dict: {exc}", page_num=page_num) from exc

    if reasons:
        return RouteDecision("chandra", reasons)
    return RouteDecision("pdftext", [])


# ---- PDFium page checks ----
def _object_bounds(obj) -> tuple[float, float, float, float]:
    # The Python wrapper renamed get_pos to get_bounds; PDFium's ABI is stable.
    coordinates = [ctypes.c_float() for _ in range(4)]
    if not pdfium_c.FPDFPageObj_GetBounds(
        obj.raw, *(ctypes.byref(value) for value in coordinates)
    ):
        raise RuntimeError("PDFium could not read page-object bounds")
    return tuple(value.value for value in coordinates)


# Below this fraction of the page area, an embedded image is treated as
# decorative (a logo, bullet icon, background flourish) rather than content
# worth describing. Tuned against one real evaluation example (a bar chart at
# ~0.145) with generous headroom - not validated against a range of image
# sizes.
MEANINGFUL_IMAGE_AREA_FRACTION = 0.05


@_serialized
def has_meaningful_image(pdf_path: str | Path, page_num: int) -> bool:
    """True if the page has an embedded raster image (chart, photo, diagram,
    scanned figure) large enough to plausibly carry content pdftext's text-only
    extraction cannot see at all - as opposed to a small logo/icon.

    pdftext has no concept of images in its own output (it is a pure text
    extractor), so this is a separate raw-pdfium check, the same primitive
    marker's own PdfProvider.check_page() uses (there, to detect a page
    that's *dominated* by one image and should be skipped/OCR'd wholesale;
    here, to catch the opposite case - a normal text page with one
    significant image sitting in it that would otherwise go undescribed).
    """
    pdf_path = str(pdf_path)
    try:
        doc = pdfium.PdfDocument(pdf_path)
    except Exception as exc:
        raise PdfExtractionError(
            f"pypdfium2 failed to open document: {exc}",
            file_path=pdf_path,
            page_num=page_num,
            stage="has_meaningful_image",
        ) from exc
    try:
        page = doc[page_num]
        page_area = page.get_width() * page.get_height()
        if page_area <= 0:
            return False
        for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE]):
            left, bottom, right, top = _object_bounds(obj)
            area = max(0.0, right - left) * max(0.0, top - bottom)
            if area / page_area >= MEANINGFUL_IMAGE_AREA_FRACTION:
                return True
        return False
    except Exception as exc:
        # Best-effort signal, not a hard requirement for the page's own
        # extraction to succeed — log and treat as "no meaningful image"
        # rather than failing the whole page over a detection heuristic.
        logger.warning(
            "router: has_meaningful_image failed for %s page %d: %s",
            pdf_path,
            page_num,
            exc,
        )
        return False
    finally:
        doc.close()


# Table-ruling-line signal. Tables are drawn as long thin horizontal/vertical
# strokes (or hairline filled rectangles); pdftext sees none of that, and the
# text-position signals in heuristics.py miss tables whose cells wrap or sit
# close together. A page with a ruled grid is a table page.
#
# Criterion: at least RULE_MIN_VERTICAL distinct vertical rules AND at least
# RULE_MIN_HORIZONTAL horizontal rule. Distinct = collinear segments merged (a
# border drawn as two segments is one rule). Vertical rules are what separate
# columns, and they are the discriminating half: on the cybersecurity sample
# every table page has 3 and no other page has any, while decorative letterhead
# art only produces horizontal lines. Horizontal needs only 1 so a table
# continuing from the previous page (no top border, just a bottom border) still
# counts. Known gap: tables ruled with horizontal lines only (no column
# dividers) are not detected here - the text-alignment signal has to catch them.
RULE_MIN_LENGTH_PT = 40.0  # shorter strokes are underlines, bullets, ornaments
RULE_MAX_THICKNESS_PT = 3.0  # thicker objects are shapes/blocks, not rules
RULE_MIN_VERTICAL = 2
RULE_MIN_HORIZONTAL = 1
RULE_COLLINEAR_TOLERANCE_PT = 2.0
# A rule spanning (nearly) the whole page is a page border/frame, not a table.
RULE_PAGE_SPAN_FRACTION = 0.9
# Header/footer art (letterhead lines, corner ornaments, page-number box) lives
# in the outer bands; same fraction heuristics.py uses for text furniture.
RULE_MARGIN_BAND_FRACTION = 0.08


@_serialized
def has_unmapped_text(pdf_path: str | Path, page_num: int) -> bool:
    """True when PDFium itself cannot map the page's glyphs to Unicode.

    A composite font without a ToUnicode map stores glyph numbers, so extraction
    returns arbitrary characters of any script. Only the font can say so: the
    text reveals nothing a language-independent check could read. A few such
    characters are ordinary (a dingbat bullet), hence the same allowance as
    invalid characters in the routing thresholds.
    """
    try:
        doc = pdfium.PdfDocument(pdf_path)
        try:
            textpage = doc[page_num].get_textpage()
            try:
                total = pdfium_c.FPDFText_CountChars(textpage.raw)
                unmapped = sum(
                    pdfium_c.FPDFText_HasUnicodeMapError(textpage.raw, index) > 0
                    for index in range(total)
                )
            finally:
                textpage.close()
        finally:
            doc.close()
    except Exception as exc:
        raise PdfExtractionError(
            f"failed to read Unicode mapping: {exc}",
            file_path=str(pdf_path),
            page_num=page_num,
            stage="unicode_map_check",
        ) from exc
    return unmapped > max(OCR_INVALID_CHAR_MIN, total * OCR_INVALID_CHAR_FRACTION)


def _distinct_positions(values: list[float]) -> int:
    ordered = sorted(values)
    count = 0
    previous: float | None = None
    for value in ordered:
        if previous is None or value - previous > RULE_COLLINEAR_TOLERANCE_PT:
            count += 1
        previous = value
    return count


@_serialized
def table_ruling_lines(pdf_path: str | Path, page_num: int) -> tuple[int, int] | None:
    """``(horizontal_rules, vertical_rules)`` if the page has a ruled grid
    (see the criterion above), else ``None``. Counts are of DISTINCT rules.

    Counts path objects that are long and thin: horizontal if at least
    ``RULE_MIN_LENGTH_PT`` wide and at most ``RULE_MAX_THICKNESS_PT`` tall,
    vertical the other way round. Objects lying entirely inside the top or
    bottom margin band (decorative letterhead/footer art) and rules spanning
    the whole page (borders) are ignored. The bands exclude only objects
    *entirely* inside them, so a table's side borders that start at the page
    top still count.

    Like ``has_meaningful_image`` this is a best-effort detection signal: on
    any pdfium error it logs and reports no grid instead of failing the page.
    """
    pdf_path = str(pdf_path)
    try:
        doc = pdfium.PdfDocument(pdf_path)
    except Exception as exc:
        raise PdfExtractionError(
            f"pypdfium2 failed to open document: {exc}",
            file_path=pdf_path,
            page_num=page_num,
            stage="table_ruling_lines",
        ) from exc
    try:
        page = doc[page_num]
        width, height = page.get_width(), page.get_height()
        if width <= 0 or height <= 0:
            return None
        horizontal_y: list[float] = []
        vertical_x: list[float] = []
        for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_PATH]):
            left, bottom, right, top = _object_bounds(obj)
            w, h = right - left, top - bottom
            if (
                top <= RULE_MARGIN_BAND_FRACTION * height
                or bottom >= (1 - RULE_MARGIN_BAND_FRACTION) * height
            ):
                continue
            if h <= RULE_MAX_THICKNESS_PT and w >= RULE_MIN_LENGTH_PT:
                if w < RULE_PAGE_SPAN_FRACTION * width:
                    horizontal_y.append(bottom)
            elif w <= RULE_MAX_THICKNESS_PT and h >= RULE_MIN_LENGTH_PT:
                if h < RULE_PAGE_SPAN_FRACTION * height:
                    vertical_x.append(left)
        horizontal = _distinct_positions(horizontal_y)
        vertical = _distinct_positions(vertical_x)
        if horizontal >= RULE_MIN_HORIZONTAL and vertical >= RULE_MIN_VERTICAL:
            return horizontal, vertical
        return None
    except Exception as exc:
        logger.warning(
            "router: table_ruling_lines failed for %s page %d: %s",
            pdf_path,
            page_num,
            exc,
        )
        return None
    finally:
        doc.close()


class PageRouter:
    """Per-page OCR routing for marker's LineBuilder.

    ``decide`` runs the same sequence as baseer's parser: text-layer heuristics,
    then unmapped-font text, meaningful images and ruled table grids. Any failure
    to inspect a page escalates it to OCR (the safe side).
    """

    def decide(
        self, pdf_path: str, page_idx: int, pdftext_page: dict | None
    ) -> RouteDecision:
        if not pdftext_page:
            return RouteDecision("chandra", ["no_text_layer"])
        try:
            decision = route_page(pdftext_page)
            if decision.route == "pdftext" and has_unmapped_text(pdf_path, page_idx):
                decision.route = "chandra"
                decision.reasons.append("unmapped_font_text")
            if decision.route == "pdftext" and has_meaningful_image(pdf_path, page_idx):
                decision.route = "chandra"
                decision.reasons.append("meaningful_image_present")
            if decision.route == "pdftext":
                grid = table_ruling_lines(pdf_path, page_idx)
                if grid:
                    decision.route = "chandra"
                    decision.reasons.append(
                        f"table_ruling_lines(h={grid[0]},v={grid[1]})"
                    )
            return decision
        except (RoutingError, PdfExtractionError) as exc:
            logger.warning(
                "Router could not inspect page %d (%s); sending to OCR", page_idx, exc
            )
            return RouteDecision("chandra", ["router_error"])
