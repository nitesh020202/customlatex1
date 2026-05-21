import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types as genai_types
import fitz  # PyMuPDF — replaces pdf2image + Poppler
import json
import os
import io
import base64
import hashlib
import tempfile
import time
import re
# ---------------- CONFIGURATION ----------------


def _safe_response_text(response) -> str | None:
    """
    Safely extract the model's actual reply text.
    - Skips 'thought' parts (internal thinking tokens).
    - Falls back to scanning parts if response.text is None.
    - Prints finish_reason when response is truly empty.
    """
    # First try the fast path
    try:
        txt = response.text
        if txt is not None and txt.strip():
            return txt
    except Exception:
        pass

    # Scan parts, skipping thought (thinking) parts
    try:
        for cand in (response.candidates or []):
            for part in (cand.content.parts or []):
                if getattr(part, "thought", False):
                    continue          # skip internal thinking text
                t = getattr(part, "text", None)
                if t and t.strip():
                    return t
    except Exception:
        pass

    # Debug: print why response is empty
    try:
        reason = response.candidates[0].finish_reason
        print(f"  [empty response] finish_reason={reason}")
    except Exception:
        pass
    return None


def _call_gemini(client, model_name: str, contents: list, cfg) -> str | None:
    """Make one Gemini call and return clean text, or None on failure."""
    response = client.models.generate_content(
        model=model_name, contents=contents, config=cfg,
    )
    return _safe_response_text(response)


def _make_page_configs():
    """
    Build configs one-by-one so a single failure never kills the whole list.
    Order: plain → json-mime → thinking (progressive budgets).
    """
    cfgs = []

    # 1. Plain — works on every model
    try:
        cfgs.append(genai_types.GenerateContentConfig(
            temperature=0, max_output_tokens=65536))
    except Exception as _e:
        print(f"[cfg] plain failed: {_e}")

    # 2. JSON mime — clean output, no markdown
    try:
        cfgs.append(genai_types.GenerateContentConfig(
            temperature=0, max_output_tokens=65536,
            response_mime_type="application/json"))
    except Exception as _e:
        print(f"[cfg] json-mime failed: {_e}")

    # 3-5. Thinking (skip if ThinkingConfig not supported)
    for budget in [1024, 8192, 24576]:
        try:
            cfgs.append(genai_types.GenerateContentConfig(
                temperature=0, max_output_tokens=65536,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=budget)))
        except Exception as _e:
            print(f"[cfg] thinking budget={budget} failed: {_e}")

    if not cfgs:
        # Ultimate fallback — bare minimum
        cfgs = [genai_types.GenerateContentConfig(
            temperature=0, max_output_tokens=65536)]

    return cfgs

st.set_page_config(
    page_title="Single Question LaTeX Extractor",
    page_icon="\U0001f4d0",
    layout="wide"
)

# ── Canonical field order ──
FIELD_ORDER = [
    "questionid", "question", "option1", "option2", "option3", "option4",
    "Answer", "Explanation", "course", "subjectname", "chapter", "practice",
    "subtopic", "medium", "difficulty", "question_type", "previous_year",
    "marks", "class", "book", "question_bucket",
]

# ================================================================
# PyMuPDF PDF HELPERS (no Poppler needed)
# ================================================================

EXTRACTION_DPI = 350  # High DPI for clear math, fine lines, small text

def pdf_page_to_png_bytes(pdf_path: str, page_num: int, dpi: int = EXTRACTION_DPI) -> bytes:
    """Render PDF page to PNG with OpenCV enhancement for best OCR quality."""
    doc  = fitz.open(pdf_path)
    page = doc[page_num]
    zoom = dpi / 72
    mat  = fitz.Matrix(zoom, zoom)
    pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    doc.close()
    try:
        import cv2, numpy as np
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        # Mild denoise — removes scan noise while keeping thin lines
        img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
        # Sharpen — makes LaTeX symbols and diagram edges crisper
        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        img = cv2.filter2D(img, -1, kernel)
        # CLAHE contrast normalisation — helps faint diagrams and light print
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        _, buf = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return buf.tobytes()
    except Exception:
        return pix.tobytes("png")


def compute_template_image_hashes(pdf_path: str) -> set:
    """Return MD5 hashes of images that appear on 3+ pages (logos, headers, template graphics)."""
    try:
        doc = fitz.open(pdf_path)
        hash_page_count: dict[str, int] = {}
        for page_num in range(len(doc)):
            page = doc[page_num]
            seen_this_page: set[str] = set()
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                try:
                    raw = doc.extract_image(xref)["image"]
                    h = hashlib.md5(raw).hexdigest()
                    if h not in seen_this_page:
                        seen_this_page.add(h)
                        hash_page_count[h] = hash_page_count.get(h, 0) + 1
                except Exception:
                    pass
        doc.close()
        return {h for h, cnt in hash_page_count.items() if cnt >= 3}
    except Exception:
        return set()


# ================================================================
# IMAGE EXTRACTION — RASTER + VECTOR (HANDLES ALL PDF IMAGE TYPES)
# ================================================================

_IMG_FIELDS = ["question", "Explanation", "option1", "option2", "option3", "option4"]

_DIAGRAM_KWS = [
    'figure', 'fig.', 'fig ', 'diagram', 'image', 'shown', 'given', 'above', 'below',
    'graph', 'circuit', 'spinner', 'marble', 'bag', 'chart', 'map', 'plot',
    'shape', 'triangle', 'rectangle', 'circle', 'polygon', 'structure', 'illustration',
    'चित्र', 'आकृति', 'दिया गया', 'नीचे', 'ऊपर', 'आरेख', 'ग्राफ', 'परिपथ',
]


def extract_page_embedded_images(pdf_path: str, page_num: int,
                                  skip_hashes: set | None = None) -> list[dict]:
    """Extract ALL raster images from a PDF page.

    Strategy:
    • JPEG / PNG → use raw compressed bytes directly (keeps file small).
    • CMYK JPEG / JPEG2000 / JBIG2 / CCITT → convert via Pixmap to RGB PNG.
      (These formats cause black-screen in browsers if sent as raw bytes.)
    • Uses get_images(full=True) + get_image_info(xrefs=True) to catch images
      inside Form XObjects as well.
    • Size threshold 50×40 px.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    seen_xrefs: set[int] = set()

    for img_info in page.get_images(full=True):
        xref = img_info[0]
        if xref > 0:
            seen_xrefs.add(xref)

    try:
        for info in page.get_image_info(xrefs=True):
            xref = info.get("xref", 0)
            if xref and xref > 0:
                seen_xrefs.add(xref)
    except Exception:
        pass

    # Build xref → y_pos map for top-to-bottom sorting
    xref_ypos: dict[int, float] = {}
    try:
        for info in page.get_image_info(hashes=False):
            xref = info.get("xref", 0)
            if xref and xref > 0:
                bbox = info.get("bbox", [0, 0, 0, 0])
                xref_ypos[xref] = float(bbox[1]) if len(bbox) >= 4 else 0.0
    except Exception:
        pass

    images = []
    for xref in seen_xrefs:
        try:
            img_data = doc.extract_image(xref)
            raw      = img_data["image"]
            ext      = img_data.get("ext", "png").lower()
            w        = img_data.get("width",  0)
            h        = img_data.get("height", 0)
            cs_n     = img_data.get("colorspace", 3)  # number of color components

            if skip_hashes and hashlib.md5(raw).hexdigest() in skip_hashes:
                continue
            if w < 40 or h < 30:
                continue
            if _is_useless_image(raw):
                continue

            # Standard RGB JPEG or PNG → send compressed bytes as-is (small payload)
            if ext in ("jpg", "jpeg") and cs_n == 3:
                mime = "image/jpeg"
                b64  = base64.b64encode(raw).decode()
            elif ext == "png" and cs_n in (1, 3):
                mime = "image/png"
                b64  = base64.b64encode(raw).decode()
            else:
                # CMYK JPEG, JPEG2000, JBIG2, CCITT, indexed, etc.
                # → decode via Pixmap and re-encode as RGB PNG
                pix = fitz.Pixmap(doc, xref)
                if pix.colorspace and pix.colorspace.n != 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                if pix.alpha:
                    pix = fitz.Pixmap(pix, 0)
                raw  = pix.tobytes("png")
                mime = "image/png"
                b64  = base64.b64encode(raw).decode()

            images.append({
                "mime_type": mime,
                "image_base64": b64,
                "data_uri": f"data:{mime};base64,{b64}",
                "y_pos": xref_ypos.get(xref, 9999.0),
            })
        except Exception:
            pass

    doc.close()
    return images


def _is_useless_image(raw_bytes: bytes) -> bool:
    """
    Return True only for truly blank/solid images — NOT for line drawings.
    A triangle (black lines on white) has ~95% white pixels but IS meaningful.
    Only filter if < 1% of pixels are dark (near-blank) OR solid single colour.
    """
    try:
        pix = fitz.Pixmap(raw_bytes)
        if pix.width < 10 or pix.height < 10:
            return True
        samples = pix.samples
        n = pix.n
        total = len(samples) // n
        step = max(1, total // 400)
        dark_count = 0
        sampled = 0
        for i in range(0, total, step):
            px = samples[i*n : i*n + min(3, n)]
            if len(px) >= 3:
                brightness = (px[0] + px[1] + px[2]) / 3
                if brightness < 180:   # counts as a meaningful dark pixel
                    dark_count += 1
                sampled += 1
        if sampled == 0:
            return False
        dark_ratio = dark_count / sampled
        # Less than 0.5% dark pixels → blank/empty image → skip
        if dark_ratio < 0.005:
            return True
        # Solid black fill (>98% very dark) → also useless
        if dark_ratio > 0.98:
            return True
        return False
    except Exception:
        return False


def _inflate_rect(r: fitz.Rect, d: float) -> fitz.Rect:
    """Expand a Rect by d on all sides — works on all PyMuPDF versions."""
    return fitz.Rect(r.x0 - d, r.y0 - d, r.x1 + d, r.y1 + d)


def _inflate_rect_xy(r: fitz.Rect, dx: float, dy: float) -> fitz.Rect:
    """Expand a Rect by dx horizontally and dy vertically."""
    return fitz.Rect(r.x0 - dx, r.y0 - dy, r.x1 + dx, r.y1 + dy)


def _smart_clip_rect(page: "fitz.Page", cluster_rect: fitz.Rect) -> fitz.Rect:
    """
    Build a precise clip rect by expanding the cluster to include ONLY nearby
    diagram labels (short text ≤ 10 chars within 40pt, or any text within 12pt).
    Long question sentences are excluded — this prevents the "extra content" problem.
    Falls back to 45pt fixed padding if text extraction fails.
    """
    try:
        data = page.get_text("dict", flags=0)
    except Exception:
        return _inflate_rect(cluster_rect, 45)

    clip = fitz.Rect(cluster_rect)

    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sr = fitz.Rect(span.get("bbox", (0, 0, 0, 0)))
                text = span.get("text", "").strip()
                if not text or sr.is_empty:
                    continue
                # Distance from the cluster bounding box to this text span
                dx = max(0.0, max(cluster_rect.x0 - sr.x1, sr.x0 - cluster_rect.x1))
                dy = max(0.0, max(cluster_rect.y0 - sr.y1, sr.y0 - cluster_rect.y1))
                dist = (dx * dx + dy * dy) ** 0.5
                if dist <= 12:
                    clip = clip | sr          # touching/overlapping — always a label
                elif dist <= 40 and len(text) <= 10:
                    clip = clip | sr          # close short text — vertex / angle / length label

    # Ensure minimum useful size (fallback if no text found)
    min_clip = _inflate_rect(cluster_rect, 20)
    if clip.width < min_clip.width or clip.height < min_clip.height:
        clip = min_clip

    # Small final margin so stroke widths / descenders are not clipped
    return fitz.Rect(
        max(0,               clip.x0 - 8),
        max(0,               clip.y0 - 8),
        min(page.rect.width, clip.x1 + 8),
        min(page.rect.height, clip.y1 + 8),
    )


def _cluster_rects(rects: list, gap: int = 25) -> list:
    """Merge rectangles that are within `gap` points of each other."""
    clusters: list[fitz.Rect] = []
    for rect in rects:
        expanded = _inflate_rect(rect, gap)
        merged = False
        for i, cluster in enumerate(clusters):
            if expanded.intersects(cluster):
                clusters[i] = clusters[i] | rect
                merged = True
                break
        if not merged:
            clusters.append(fitz.Rect(rect))
    return clusters


def _crop_diagrams_from_image(image_bytes: bytes, regions: list[dict]) -> list[dict]:
    """
    Crop diagram sub-regions from a raster image using normalized (0–1) coordinates.
    Tries PIL first, falls back to fitz.Pixmap cropping.

    Each region dict must have: x0_norm, y0_norm, x1_norm, y1_norm (all 0.0–1.0).
    Returns list of {mime_type, image_base64, data_uri, y_pos} dicts.
    """
    if not regions or not image_bytes:
        return []

    PAD = 0.07   # 7% padding on all sides — captures angle arcs, vertex labels, side annotations

    def _do_crop_pil():
        from PIL import Image
        import io as _io
        img = Image.open(_io.BytesIO(image_bytes))
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        w, h = img.size
        out = []
        for r in sorted(regions, key=lambda x: x.get('idx', x.get('y0_norm', 0))):
            x0 = max(0, int((r.get('x0_norm', 0) - PAD) * w))
            y0 = max(0, int((r.get('y0_norm', 0) - PAD) * h))
            x1 = min(w, int((r.get('x1_norm', 1) + PAD) * w))
            y1 = min(h, int((r.get('y1_norm', 1) + PAD) * h))
            if x1 - x0 < 20 or y1 - y0 < 20:
                continue
            crop = img.crop((x0, y0, x1, y1))
            buf = _io.BytesIO()
            crop.save(buf, format='PNG')
            b64 = base64.b64encode(buf.getvalue()).decode()
            out.append({
                "mime_type": "image/png",
                "image_base64": b64,
                "data_uri": f"data:image/png;base64,{b64}",
                "y_pos": float(y0),
            })
        return out

    def _do_crop_fitz():
        pix = fitz.Pixmap(image_bytes)
        w, h = pix.width, pix.height
        out = []
        for r in sorted(regions, key=lambda x: x.get('idx', x.get('y0_norm', 0))):
            x0 = max(0, int((r.get('x0_norm', 0) - PAD) * w))
            y0 = max(0, int((r.get('y0_norm', 0) - PAD) * h))
            x1 = min(w, int((r.get('x1_norm', 1) + PAD) * w))
            y1 = min(h, int((r.get('y1_norm', 1) + PAD) * h))
            if x1 - x0 < 20 or y1 - y0 < 20:
                continue
            # fitz.Pixmap(src_pixmap, clip_irect) crops the pixmap
            clip_pix = fitz.Pixmap(pix, fitz.IRect(x0, y0, x1, y1))
            raw = clip_pix.tobytes('png')
            b64 = base64.b64encode(raw).decode()
            out.append({
                "mime_type": "image/png",
                "image_base64": b64,
                "data_uri": f"data:image/png;base64,{b64}",
                "y_pos": float(y0),
            })
        return out

    try:
        result = _do_crop_pil()
        if result:
            return result
    except Exception:
        pass
    try:
        return _do_crop_fitz()
    except Exception as e:
        print(f"[crop_diagrams] failed: {e}")
        return []


def extract_vector_diagram_regions(pdf_path: str, page_num: int) -> list[dict]:
    """
    Extract only REAL diagrams — triangles, graphs, circuits, physics/chemistry figures.
    Conservative thresholds to avoid capturing text lines, underlines, borders.

    Rules:
    - Minimum 60×60 pt AND area > 6000 sq pt (real diagrams are always substantial)
    - Must have COMPLEX structure: at least 4 distinct path elements in the cluster
    - Rendered region must pass _is_useless_image check (no solid fills)
    - GAP method removed — it captures too many false positives (text separators)
    """
    try:
        doc  = fitz.open(pdf_path)
        page = doc[page_num]
        crop_mat = fitz.Matrix(3.0, 3.0)
        images: list[dict] = []
        captured: list[fitz.Rect] = []

        def _already_covered(r: fitz.Rect) -> bool:
            return any(
                cap.intersect(r).get_area() / max(r.get_area(), 1) > 0.5
                for cap in captured
            )

        def _render(clip: fitz.Rect, y_pos: float):
            clip = clip & page.rect
            if clip.width < 10 or clip.height < 10:
                return
            pix = page.get_pixmap(matrix=crop_mat, clip=clip, colorspace=fitz.csRGB)
            raw = pix.tobytes("png")
            if _is_useless_image(raw):     # skip solid-colour fills
                return
            b64 = base64.b64encode(raw).decode()
            images.append({
                "mime_type": "image/png",
                "image_base64": b64,
                "data_uri": f"data:image/png;base64,{b64}",
                "y_pos": y_pos,
            })
            captured.append(clip)

        # ── DRAWINGS: cluster vector paths, capture all real diagrams ──
        # KEY FIX: single-path shapes (triangle, circle, polygon drawn as one closed path)
        # each form a cluster with cnt=1. Remove cnt<2 filter — use SIZE to distinguish
        # real diagrams from flat fraction bars / underlines (area≈0).
        drawings = page.get_drawings()
        if drawings:
            raw_rects = [fitz.Rect(d["rect"]) for d in drawings
                         if d.get("rect") and max(fitz.Rect(d["rect"]).width,
                                                   fitz.Rect(d["rect"]).height) >= 4]

            clusters: list[fitz.Rect] = []
            cluster_counts: list[int] = []
            cluster_max_elem_h: list[float] = []  # tallest individual element in cluster
            for rect in raw_rects:
                # 50pt horizontal gap: merges same-row triangles (Vijay/Tower/Ajay side-by-side).
                # 20pt vertical gap: keeps sub-part diagrams that are stacked vertically SEPARATE.
                # Different rows are typically 30-60pt apart — smaller Y gap keeps them distinct.
                exp = _inflate_rect_xy(rect, 50, 20)
                merged = False
                for i, c in enumerate(clusters):
                    if exp.intersects(c):
                        clusters[i] = clusters[i] | rect
                        cluster_counts[i] += 1
                        cluster_max_elem_h[i] = max(cluster_max_elem_h[i], rect.height)
                        merged = True
                        break
                if not merged:
                    clusters.append(fitz.Rect(rect))
                    cluster_counts.append(1)
                    cluster_max_elem_h.append(rect.height)

            page_area = page.rect.width * page.rect.height
            for cr, cnt, max_elem_h in zip(clusters, cluster_counts, cluster_max_elem_h):
                # Must have at least one element with real height — rules out flat fraction
                # bars (all h≈0) which cluster together but are NOT diagrams.
                if max_elem_h < 10:
                    continue
                # Must be a physically real region (not a tiny mark)
                if cr.width < 20 or cr.height < 20:
                    continue
                # Minimum area filter
                if cr.get_area() < 800:
                    continue
                # Skip page-spanning borders / background fills
                if cr.get_area() > 0.25 * page_area:
                    continue
                # Skip very wide thin strips (formula equation lines)
                aspect = cr.width / max(cr.height, 1)
                if aspect > 5.0 and cr.height < 30:
                    continue
                if _already_covered(cr):
                    continue
                # Smart clip: expands to include actual diagram labels (short nearby text)
                # but stops at question text (long sentences) — no fixed oversized padding.
                clip = _smart_clip_rect(page, cr)
                _render(clip, cr.y0)

        # ── ANNOTATIONS: ink drawings, hand-drawn shapes ──────────────────
        try:
            for annot in page.annots():
                ar = fitz.Rect(annot.rect)
                if ar.width >= 15 and ar.height >= 15 and not _already_covered(ar):
                    _render(_inflate_rect(ar, 8), ar.y0)
        except Exception:
            pass

        doc.close()
        images.sort(key=lambda x: x.get("y_pos", 0))
        return images
    except Exception:
        return []


def extract_all_page_images(pdf_path: str, page_num: int,
                             skip_hashes: set | None = None) -> list[dict]:
    """Return ALL images sorted TOP-TO-BOTTOM by y_pos (matches PDF reading order)."""
    raster = extract_page_embedded_images(pdf_path, page_num, skip_hashes)
    vector = extract_vector_diagram_regions(pdf_path, page_num)
    # Deduplicate: skip vector regions that overlap with a raster image position
    deduped_vector = [v for v in vector
                      if not any(abs(v.get("y_pos", 0) - r.get("y_pos", 9999)) < 30
                                 for r in raster)]
    combined = raster + deduped_vector
    # Sort top-to-bottom — [DIAGRAM_0] = topmost, matches Gemini's numbering in prompt
    combined.sort(key=lambda x: x.get("y_pos", 9999))
    return combined


# ================================================================
# DIAGRAM INJECTION HELPERS
# ================================================================

def _normalize_diagram_refs(text: str) -> str:
    """Normalize every Gemini diagram-placeholder variant → [DIAGRAM_X]."""
    if not text:
        return text
    text = re.sub(r'\bdiagram\[(\d+)\]',               r'[DIAGRAM_\1]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[diagram_(\d+)\]',                 r'[DIAGRAM_\1]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[DIAGRAM(\d+)\]',                  r'[DIAGRAM_\1]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[image[_ ]?(\d+)\]',               r'[DIAGRAM_\1]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[IMG[_ ]?(\d+)\]',                 r'[DIAGRAM_\1]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[Figure[_ ]?(\d+)\]',              r'[DIAGRAM_\1]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[Fig\.?\s*(\d+)\]',                r'[DIAGRAM_\1]', text, flags=re.IGNORECASE)
    text = re.sub(r'<image[_ ]?(\d+)>',                 r'[DIAGRAM_\1]', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<!\[)\bdiagram\s+(\d+)\b(?!\])', r'[DIAGRAM_\1]', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<!\[)\bfigure\s+(\d+)\b(?!\])',  r'[DIAGRAM_\1]', text, flags=re.IGNORECASE)
    return text


def _img_tag(data_uri: str, width: int = 480) -> str:
    return (f'<img src="{data_uri}" '
            f'style="max-width:{width}px;width:100%;height:auto;'
            f'display:block;margin:10px auto;border-radius:2px"/>')


def _placeholder_tag() -> str:
    """Yellow-highlighted placeholder shown wherever an image/diagram exists."""
    return (
        '<span style="background-color:#fff176;color:#000000;font-weight:bold;'
        'padding:2px 6px;font-size:14px;font-family:Arial,sans-serif">'
        'Replace the image'
        '</span>'
    )


def _cleanup_placeholders(q: dict) -> dict:
    """Remove every leftover [DIAGRAM_X] / [DIAGRAM] string from all text fields."""
    for field in _IMG_FIELDS:
        if field in q and q[field]:
            q[field] = re.sub(r'\[DIAGRAM_?\d*\]', '', q[field])
            q[field] = re.sub(r'\[DIAGRAM\]',       '', q[field])
    return q


def inject_diagrams_into_question(q: dict, page_images: list[dict]) -> dict:
    """Replace every [DIAGRAM_X] placeholder with a green 'Replace the image' placeholder.

    Works for question, option1-4, and Explanation.
    Placement is controlled entirely by [DIAGRAM_X] tags Gemini puts in the text.
    Actual image bytes are discarded — only the position is used.
    """
    # ── Step 1: normalise all variant notations in every field ──
    for field in _IMG_FIELDS:
        if field in q and q[field]:
            q[field] = _normalize_diagram_refs(q[field])

    # ── Step 2: no images → clean every placeholder and exit ──
    if not page_images:
        q.pop("diagram_placements", None)
        q.pop("diagram_image_indices", None)
        return _cleanup_placeholders(q)

    ph = _placeholder_tag()   # single green placeholder for every diagram slot

    # ── Step 3: replace [DIAGRAM_X] with placeholder; track which indices were used ──
    injected_indices: set[int] = set()
    for field in _IMG_FIELDS:
        if field not in q or not q[field]:
            continue
        for match in re.findall(r'\[DIAGRAM_(\d+)\]', q[field]):
            idx = int(match)
            if 0 <= idx < len(page_images):
                q[field] = q[field].replace(f'[DIAGRAM_{idx}]', ph)
                injected_indices.add(idx)
            else:
                q[field] = q[field].replace(f'[DIAGRAM_{idx}]', '')

    # ── Step 4: fallback — diagram_placements (position / field hints from Gemini) ──
    placements       = q.pop("diagram_placements", []) or []
    indices_fallback = q.pop("diagram_image_indices", []) or []

    for p in placements:
        idx      = p.get("image_index", 0)
        field    = p.get("field", "Explanation")
        position = p.get("position", "end")
        if 0 <= idx < len(page_images) and field in q and idx not in injected_indices:
            if position == "start":
                q[field] = ph + "\n" + (q.get(field) or "")
            elif position == "replace" and "[DIAGRAM]" in q.get(field, ""):
                q[field] = q[field].replace("[DIAGRAM]", ph)
            else:
                q[field] = (q.get(field) or "") + "\n" + ph
            injected_indices.add(idx)

    # ── Step 5: fallback — diagram_image_indices with keyword matching ──
    q_text   = (q.get("question", "")    or "").lower()
    exp_text = (q.get("Explanation", "") or "").lower()
    for idx in indices_fallback:
        if 0 <= idx < len(page_images) and idx not in injected_indices:
            if any(kw in q_text for kw in _DIAGRAM_KWS):
                q["question"]    = (q.get("question")    or "") + f"\n{ph}"
                injected_indices.add(idx)
            elif any(kw in exp_text for kw in _DIAGRAM_KWS):
                q["Explanation"] = (q.get("Explanation") or "") + f"\n{ph}"
                injected_indices.add(idx)

    # ── Step 5b: LAST RESORT — Gemini placed ZERO tags despite images being sent ──
    if not injected_indices:
        remaining = [i for i in range(len(page_images)) if i not in injected_indices]
        if remaining:
            prepend = "\n".join(ph for _ in remaining)
            q["question"] = prepend + "\n" + (q.get("question") or "")
            injected_indices.update(remaining)
    else:
        remaining = [i for i in range(len(page_images)) if i not in injected_indices]
        for idx in remaining:
            q["Explanation"] = (q.get("Explanation") or "") + f"\n{ph}"
            injected_indices.add(idx)

    # ── Step 6: final cleanup — remove any leftover placeholders ──
    return _cleanup_placeholders(q)

# ================================================================
# LIVE PREVIEW HELPERS
# ================================================================

def _fmt_latex(text: str) -> str:
    if not text or not isinstance(text, str):
        return ''
    text = re.sub(r'<img[^>]*>', '[📷]', text, flags=re.IGNORECASE)
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\\\\\((.+?)\\\\\)', r'$\1$', text, flags=re.DOTALL)
    text = re.sub(r'\\\\\[(.+?)\\\\\]', r'$$\1$$', text, flags=re.DOTALL)
    return text


# ── LCM/HCF ladder line patterns ──
_LADDER_DIVISOR_RE = re.compile(r'^\s*\d+\s*\|')      # "2 | 15  20  25"
_LADDER_SEP_RE     = re.compile(r'^\s*\|[_\s]+$')      # "  |___________"
_LADDER_EMPTY_DIV  = re.compile(r'^\s*\|')             # "  | 1  1  1" (empty divisor row)
_LADDER_FINAL_RE   = re.compile(r'^\s+\d[\d\s]+$')     # "    1   1   1" (no pipe, final row)
_LADDER_ROW_PARSE  = re.compile(r'^(\s*\d*)\s*\|\s*(.+)$')  # parse "div | nums"


def _lcm_ladder_to_html(ladder_lines: list) -> str:
    """Convert plain-text LCM/HCF ladder → <table class="lcm-ladder-t"> with
    border-collapse:collapse so the vertical line is perfectly continuous (zero gap).
    Each divisor row gets border-right (vertical) + border-bottom (horizontal).
    Final row (1,1,1) gets no borders — indented, matches PDF exactly.
    """
    rows       = []   # list of (divisor_str, numbers_str)
    final_nums = None

    for line in ladder_lines:
        stripped = line.strip()
        if _LADDER_SEP_RE.match(stripped):
            continue                          # drop "  |___" — replaced by CSS borders
        m = _LADDER_ROW_PARSE.match(stripped)
        if m:
            rows.append((m.group(1).strip(), m.group(2)))
        elif _LADDER_FINAL_RE.match(line):
            final_nums = line.strip()         # "1   1   1"

    if not rows:
        return ''

    VL = '2px solid #111'    # vertical line (right border of divisor cell)
    HL = '1.5px solid #111'  # horizontal line (bottom border of each row)

    # Inline border:none resets the global "table td {border:1px solid #444}" rule
    S_D = (f'border:none;border-right:{VL};border-bottom:{HL};'
           f'text-align:right;padding:3px 6px 3px 4px;vertical-align:middle;'
           f'font-weight:bold;min-width:20px')
    S_N = (f'border:none;border-bottom:{HL};'
           f'padding:3px 10px 3px 8px;white-space:pre;vertical-align:middle')

    tbl = ('<table class="lcm-ladder-t" style="border-collapse:collapse;'
           'font-family:\'Courier New\',Courier,monospace;font-size:14px;'
           'line-height:1.9;margin:8px 0">')

    for div, nums in rows:
        de = div.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
        ne = nums.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
        tbl += f'<tr><td style="{S_D}">{de}</td><td style="{S_N}">{ne}</td></tr>'

    if final_nums is not None:
        fe = final_nums.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
        # No borders on final row — just indent to align with numbers above
        SF_D = 'border:none;padding:3px 6px 3px 4px;min-width:20px'
        SF_N = 'border:none;padding:3px 10px 3px 8px;white-space:pre;vertical-align:middle'
        tbl += f'<tr><td style="{SF_D}"></td><td style="{SF_N}">{fe}</td></tr>'

    tbl += '</table>'
    return tbl


def _md_table_to_html(text: str) -> str:
    # Normalise <br> → \n so patterns work even after newline_to_br ran
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    lines = text.split('\n')
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ── Standard markdown table (| col | col |) ──
        if '|' in line and stripped.startswith('|') and not _LADDER_SEP_RE.match(stripped):
            table_lines = []
            while i < len(lines) and '|' in lines[i] and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1
            rows = [r for r in table_lines if not re.match(r'^\s*\|[\s\-|:]+\|\s*$', r)]
            if rows:
                html = '<table class="qtable" style="border-collapse:collapse;margin:8px 0">'
                for ri, row in enumerate(rows):
                    cells = [c.strip() for c in row.strip().strip('|').split('|')]
                    if ri == 0:
                        html += ('<tr>' + ''.join(
                            f'<th style="border:1px solid #333;padding:5px 10px;'
                            f'background:#dbeafe;font-weight:bold;text-align:center">{c}</th>'
                            for c in cells) + '</tr>')
                    else:
                        html += ('<tr>' + ''.join(
                            f'<td style="border:1px solid #333;padding:5px 10px;'
                            f'text-align:left;vertical-align:middle">{c}</td>'
                            for c in cells) + '</tr>')
                html += '</table>'
                out.append(html)
            continue

        # ── LCM/HCF plain-text ladder: "2 | 15  20  25" (does NOT start with |) ──
        if _LADDER_DIVISOR_RE.match(stripped):
            ladder_lines = []
            while i < len(lines):
                s = lines[i]
                ss = s.strip()
                if (_LADDER_DIVISOR_RE.match(ss) or
                        _LADDER_SEP_RE.match(ss) or
                        _LADDER_EMPTY_DIV.match(ss) or
                        (ladder_lines and _LADDER_FINAL_RE.match(s))):
                    ladder_lines.append(s)
                    i += 1
                elif ss == '' and ladder_lines:
                    break
                else:
                    break
            if len(ladder_lines) >= 2:
                out.append(_lcm_ladder_to_html(ladder_lines))
            else:
                out.extend(ladder_lines)
            continue

        out.append(line)
        i += 1
    return '\n'.join(out)


def _normalize_html_table(table_html: str) -> str:
    """Add CSS class to Gemini-generated tables that lack one.
    Skip lcm-ladder-t tables — they carry precise inline styles already.
    """
    head = table_html[:120].lower()
    if 'lcm-ladder-t' in head:
        return table_html   # already fully styled inline
    if 'class=' not in head:
        return table_html.replace('<table', '<table class="ltable"', 1)
    if 'qtable' not in head and 'ltable' not in head and 'lcm-table' not in head:
        return re.sub(r'class="', 'class="ltable ', table_html, count=1, flags=re.IGNORECASE)
    return table_html


_KEEP_RE = re.compile(
    r'(<img\b[^>]*?>'
    r'|<table\b[^>]*?>.*?</table\s*>'
    r'|<pre\b[^>]*?>.*?</pre\s*>)',
    re.IGNORECASE | re.DOTALL
)


def _clean_for_html(text: str) -> str:
    if not text or not isinstance(text, str):
        return ''
    # _md_table_to_html normalises <br>→\n and detects ladder/markdown tables
    text = _md_table_to_html(text)
    segments = _KEEP_RE.split(text)
    result = []
    for seg in segments:
        if _KEEP_RE.match(seg):
            # Normalise Gemini HTML tables that lack a CSS class
            if seg.lower().startswith('<table'):
                seg = _normalize_html_table(seg)
            result.append(seg)
        else:
            s = seg
            s = re.sub(r'<br\s*/?>', '\n', s, flags=re.IGNORECASE)
            s = re.sub(r'<[^>]+>', '', s)
            s = re.sub(r'\\\\\((.+?)\\\\\)', r'\\(\1\\)', s, flags=re.DOTALL)
            s = re.sub(r'\\\\\[(.+?)\\\\\]', r'\\[\1\\]', s, flags=re.DOTALL)
            s = s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            s = s.replace('\n', '<br>')
            result.append(s)
    return ''.join(result)


def _build_preview_html(questions: list) -> str:
    if not questions:
        return '<p style="color:#888;padding:16px;font-family:sans-serif">Waiting for questions…</p>'
    body_parts = []
    for q in questions:
        qid  = q.get('questionid', '?')
        qtxt = _clean_for_html(str(q.get('question', '')))
        opts_html = ''
        for i in range(1, 5):
            opt = q.get(f'option{i}', '')
            if opt:
                lbl = chr(64 + i)
                opts_html += (
                    f'<div class="opt">'
                    f'<span class="lbl">({lbl})</span>&nbsp;{_clean_for_html(str(opt))}'
                    f'</div>'
                )
        ans = q.get('Answer', '')
        exp = q.get('Explanation', '')
        ans_html = f'<div class="ans">&#x2705; <b>Answer:</b> {ans}</div>' if ans else ''
        exp_html = (
            f'<div class="exp">&#x1F4A1; <b>Explanation:</b> {_clean_for_html(str(exp))}</div>'
            if exp else ''
        )
        body_parts.append(
            f'<div class="qblock">'
            f'<div class="qtext"><b>Q{qid}.</b>&nbsp;{qtxt}</div>'
            f'<div class="opts">{opts_html}</div>'
            f'{ans_html}{exp_html}'
            f'</div>'
        )
    body = '\n'.join(body_parts)
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">\n'
        '<script>\n'
        'window.MathJax = {\n'
        '  tex: {\n'
        '    inlineMath: [["\\\\(","\\\\)"]],\n'
        '    displayMath: [["\\\\[","\\\\]"]],\n'
        '    processEscapes: true\n'
        '  },\n'
        '  options: { skipHtmlTags: ["script","noscript","style","textarea"] }\n'
        '};\n'
        '</script>\n'
        '<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>\n'
        '<style>\n'
        'body{font-family:"Times New Roman",serif;font-size:14px;line-height:1.8;color:#111;background:#fff;padding:12px;margin:0}\n'
        '.qblock{margin-bottom:20px;padding:10px 14px;border-left:3px solid #2563eb;background:#f8fafc;border-radius:3px}\n'
        '.qtext{margin-bottom:8px}.opts{margin-left:18px}.opt{margin:4px 0}\n'
        '.lbl{font-weight:bold;color:#374151}\n'
        '.ans{margin-top:8px;color:#15803d;font-size:13px}\n'
        '.exp{margin-top:6px;color:#374151;font-size:13px}\n'
        'table{border-collapse:collapse;margin:8px 0;max-width:100%;font-size:13px;font-family:"Times New Roman",serif}\n'
        'table td,table th{border:1px solid #444;padding:5px 10px;vertical-align:middle;text-align:center}\n'
        'table th{background:#dbeafe;font-weight:bold;color:#1e3a8a}\n'
        'table tr:nth-child(even) td{background:#f8fafc}\n'
        'table.lcm-table td:first-child,.lcm-div{border-right:2.5px solid #000!important;background:#eef2ff;font-weight:bold;min-width:30px;text-align:center}\n'
        'table.lcm-table td{min-width:38px;text-align:center;padding:4px 8px}\n'
        'table.lcm-table tr:last-child td{border-top:2px solid #111}\n'
        'table.qtable td,table.qtable th,table.ltable td,table.ltable th{text-align:left}\n'
        'table.match-col td:first-child,table.match-col th:first-child{border-right:2px solid #555}\n'
        'img{max-width:100%;height:auto;display:block;margin:6px auto;border-radius:2px}\n'
        '.img-row{display:flex;flex-wrap:wrap;gap:8px;align-items:flex-start;margin:6px 0}\n'
        '.img-row img{max-width:calc(50% - 4px);flex:0 1 auto;margin:0}\n'
        '.compare-wrap{display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap;margin:8px 0}\n'
        '.compare-wrap>*{flex:1 1 auto;min-width:180px}\n'
        'mjx-container[display="true"]{display:block;margin:6px 0;overflow-x:auto}\n'
        '.MathJax{font-size:1em!important}\n'
        'table.lcm-ladder-t{border-collapse:collapse}'
        '\n'
        '</style></head><body>\n'
        + body +
        '\n</body></html>'
    )


# ================================================================
# SECTION CONFIG HELPERS
# ================================================================
def build_section_page_map(section_configs, total_pages):
    page_marks = [""] * total_pages
    for sec in section_configs:
        start = sec.get("start_page", 1)
        end   = sec.get("end_page", total_pages)
        marks = str(sec.get("marks", "")).strip()
        for p in range(start - 1, min(end, total_pages)):
            page_marks[p] = marks
    return page_marks

# ================================================================
# CHECKPOINT HELPERS
# ================================================================
def get_checkpoint_path(filename):
    base = os.path.splitext(filename)[0]
    return os.path.join(CHECKPOINT_DIR, f"{base}_checkpoint.json")


def save_checkpoint(filename, page_results, total_pages):
    checkpoint = {
        "total_pages": total_pages,
        "completed_pages": {
            # Only save pages that have actual questions — never save empty []
            str(i): data for i, data in enumerate(page_results)
            if data is not None and len(data) > 0
        }
    }
    with open(get_checkpoint_path(filename), "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False)


def load_checkpoint(filename, total_pages):
    path = get_checkpoint_path(filename)
    if not os.path.exists(path):
        return [None] * total_pages, 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        if checkpoint.get("total_pages", 0) != total_pages:
            return [None] * total_pages, 0
        page_results = [None] * total_pages
        for idx_str, data in checkpoint.get("completed_pages", {}).items():
            page_results[int(idx_str)] = data
        done = sum(1 for r in page_results if r is not None)
        return page_results, done
    except Exception:
        return [None] * total_pages, 0


def delete_checkpoint(filename):
    path = get_checkpoint_path(filename)
    if os.path.exists(path):
        os.remove(path)

# ================================================================
# ENHANCED POST-PROCESSING HELPERS
# ================================================================
def remove_cite_tags(text):
    if not text or not isinstance(text, str):
        return text
    return re.sub(r'\s*\[cite\s*:\s*\d+\]', '', text).strip()


def convert_dollar_to_latex(text):
    if not text or not isinstance(text, str):
        return text
    text = re.sub(r'\$\$(.+?)\$\$',
                  lambda m: '\\\\[' + m.group(1) + '\\\\]',
                  text, flags=re.DOTALL)
    text = re.sub(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)',
                  lambda m: '\\\\(' + m.group(1) + '\\\\)',
                  text, flags=re.DOTALL)
    return text


def remove_base64_images(text):
    # Do NOT strip base64 <img> tags — they are the actual diagram images.
    # Only remove bare data-URI strings that appear outside <img> tags
    # (e.g. raw Gemini hallucinations that were never wrapped in an img tag).
    if not text or not isinstance(text, str):
        return text
    # Strip raw data-URI blobs that are NOT already inside an src="..." attribute
    text = re.sub(
        r'(?<!src=["\'])(?<!src=)data:image/[^;]+;base64,[A-Za-z0-9+/=]{100,}',
        '', text
    )
    return text


def newline_to_br(text):
    """Replace all \\n newlines with <br> tags for HTML rendering."""
    if not text or not isinstance(text, str):
        return text
    # Don't convert newlines inside HTML table tags
    if '<table' in text.lower():
        return text
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = text.replace('\n', '<br>')
    return text


def clean_explanation_prefix(text):
    if not text or not isinstance(text, str):
        return text
    text = text.strip()
    prefix_pattern = re.compile(
        r'^(?:Ans(?:wer)?\.?\s*:?\s*(?:\([A-Da-d]\))?\s*|Sol(?:ution)?\.?\s*:?\s*)',
        re.IGNORECASE
    )
    hindi_prefix_pattern = re.compile(
        r'^(?:उत्तर\s*:?\s*|हल\s*:?\s*|समाधान\s*:?\s*|व्याख्या\s*:?\s*)',
        re.UNICODE
    )
    for _ in range(3):
        cleaned = prefix_pattern.sub('', text).strip()
        cleaned = hindi_prefix_pattern.sub('', cleaned).strip()
        if cleaned == text:
            break
        if not cleaned:
            break
        text = cleaned
    return text


# ================================================================
# FIX: ENHANCED previous_year EXTRACTION FROM QUESTION TEXT
# ================================================================

_PY_INLINE_PATTERNS = [
    re.compile(
        r'[-–—]?\s*[\(\[]\s*'
        r'((?:Exercise|Ex|Miscellaneous\s+Exercise|Misc\.?\s*Exercise|Miscellaneous|'
        r'Example|Ex\.?|NCERT|PYQ|Previous\s+Year|Exemplar)\s*[-–]?\s*[\d\.]+(?:[-–]\d+)?'
        r'(?:\s*[-–]\s*\d+)*)\s*[\)\]]',
        re.IGNORECASE
    ),
    re.compile(
        r'[-–—]?\s*[\(\[]?\s*\b((19|20)\d{2})\b\s*[\)\]]?',
        re.IGNORECASE
    ),
    re.compile(
        r'[-–—]?\s*[\(\[]?\s*'
        r'((?:JEE\s*(?:Main|Advanced|Mains)?|NEET|AIIMS|CBSE|ICSE|BITSAT|MHT[- ]?CET|'
        r'WBJEE|KCET|UPSEE|VITEEE|COMEDK|NDA|CDS|UPSC|Board)\s*[-–]?\s*(?:(19|20)\d{2})?)\s*'
        r'[\)\]]?',
        re.IGNORECASE
    ),
    re.compile(
        r'[-–—]\s*'
        r'((?:Miscellaneous\s+Exercise|Misc\.?\s*Exercise|NCERT\s+Exemplar|'
        r'Exercise|Example|Ex\.?)\s*[-–]?\s*[\d\.]+(?:[-–]\d+)*)',
        re.IGNORECASE
    ),
]

_PY_END_TAG_RE = re.compile(
    r'\s*[-–—]?\s*[\(\[]\s*'
    r'((?:Exercise|Ex\.?|Misc(?:ellaneous)?\s*Exercise|Example|NCERT|PYQ|'
    r'JEE\s*(?:Main|Advanced|Mains)?|NEET|AIIMS|CBSE|ICSE|Board|'
    r'(?:19|20)\d{2})'
    r'[\s\d\.,-]*)\s*[\)\]]\s*$',
    re.IGNORECASE
)


def extract_previous_year_from_text(text: str) -> tuple[str, str]:
    if not text or not isinstance(text, str):
        return "", text

    m = _PY_END_TAG_RE.search(text)
    if m:
        ref = m.group(1).strip()
        cleaned = text[:m.start()].strip()
        cleaned = re.sub(r'[\s\-–—]+$', '', cleaned)
        return ref, cleaned

    for pattern in _PY_INLINE_PATTERNS:
        m = pattern.search(text)
        if m:
            ref = m.group(1).strip() if m.lastindex and m.group(1) else m.group(0).strip()
            if m.start() >= max(0, len(text) - 120):
                cleaned = text[:m.start()].strip()
                cleaned = re.sub(r'[\s\-–—]+$', '', cleaned)
                if ref:
                    return ref, cleaned

    return "", text


def fix_previous_year_field(py_val: str) -> str:
    if not py_val or not isinstance(py_val, str):
        return ""
    val = py_val.strip()
    garbage = {
        'null', 'none', 'n/a', 'not applicable', 'not found',
        'not available', 'unknown', 'na', '-', '', 'no'
    }
    if val.lower() in garbage:
        return ""

    if re.search(r'\b(19|20)\d{2}\b', val):
        return val
    if re.search(
        r'(ncert|exercise|example|ex\.?\s*\d|miscellaneous|misc|exemplar|'
        r'jee|neet|aiims|cbse|icse|pyq|previous\s*year)',
        val, re.IGNORECASE
    ):
        return val
    return ""


# ================================================================
# SUBTOPIC AUTO-FILL — Hindi + English keywords
# ================================================================

# ── SUBTOPIC MAPS: English → Hindi ──
_SUBTOPIC_HINDI_MAP = {
    # Physics
    "Laws of Motion": "गति के नियम",
    "Work Energy Power": "कार्य ऊर्जा और शक्ति",
    "Gravitation": "गुरुत्वाकर्षण",
    "Current Electricity": "विद्युत धारा",
    "Electrostatics": "स्थिरवैद्युतिकी",
    "Magnetism": "चुंबकत्व",
    "Waves": "तरंगें",
    "Optics": "प्रकाशिकी",
    "Thermodynamics": "ऊष्मागतिकी",
    "Semiconductors": "अर्धचालक",
    "Atoms and Nuclei": "परमाणु और नाभिक",
    "Friction": "घर्षण",
    "Rotational Motion": "घूर्णन गति",
    "Oscillations": "दोलन",
    "Fluid Mechanics": "तरल यांत्रिकी",
    "Kinematics": "गतिकी",
    "Units and Measurements": "मात्रक और मापन",
    "Magnetic Effect of Current": "विद्युत धारा का चुंबकीय प्रभाव",
    "Electromagnetic Induction": "विद्युत चुम्बकीय प्रेरण",
    "AC Circuits": "प्रत्यावर्ती धारा परिपथ",
    "Dual Nature of Matter": "पदार्थ की द्वैत प्रकृति",
    "Communication Systems": "संचार व्यवस्था",
    "Ray Optics": "किरण प्रकाशिकी",
    "Wave Optics": "तरंग प्रकाशिकी",
    # Chemistry
    "Mole Concept": "मोल संकल्पना",
    "Chemical Equilibrium": "रासायनिक साम्य",
    "Acids Bases and Salts": "अम्ल क्षार और लवण",
    "Electrochemistry": "विद्युत रसायन",
    "Organic Chemistry": "कार्बनिक रसायन",
    "Periodic Table": "आवर्त सारणी",
    "Chemical Bonding": "रासायनिक बंधन",
    "Chemical Kinetics": "रासायनिक बलगतिकी",
    "Order of Reaction": "अभिक्रिया की कोटि",
    "Thermochemistry": "ऊष्मा रसायन",
    "Solutions": "विलयन",
    "Solid State": "ठोस अवस्था",
    "Surface Chemistry": "पृष्ठ रसायन",
    "Coordination Compounds": "उपसहसंयोजक यौगिक",
    "Polymers": "बहुलक",
    "Biomolecules": "जैव अणु",
    # Mathematics
    "Integration": "समाकलन",
    "Differentiation": "अवकलन",
    "Limits and Continuity": "सीमा और सातत्य",
    "Matrices and Determinants": "आव्यूह और सारणिक",
    "Vectors": "सदिश",
    "Probability": "प्रायिकता",
    "Conic Sections": "शंकु परिच्छेद",
    "Sequences and Series": "अनुक्रम और श्रेणी",
    "Trigonometry": "त्रिकोणमिति",
    "Complex Numbers": "सम्मिश्र संख्याएँ",
    "Sets Relations Functions": "समुच्चय संबंध और फलन",
    "Straight Lines": "सरल रेखाएँ",
    "Binomial Theorem": "द्विपद प्रमेय",
    "Statistics": "सांख्यिकी",
    "Mathematical Reasoning": "गणितीय विवेचन",
    "3D Geometry": "त्रिविमीय ज्यामिति",
    "Differential Equations": "अवकल समीकरण",
    "Relations and Functions": "संबंध और फलन",
    "Inverse Trigonometry": "प्रतिलोम त्रिकोणमिति",
    "Linear Programming": "रैखिक प्रोग्रामन",
    # Biology
    "Cell Biology": "कोशिका जीवविज्ञान",
    "Photosynthesis": "प्रकाश संश्लेषण",
    "Respiration": "श्वसन",
    "Genetics": "आनुवंशिकी",
    "Evolution": "विकास",
    "Ecology": "पारिस्थितिकी",
    "Endocrine System": "अंतःस्रावी तंत्र",
    "Digestive System": "पाचन तंत्र",
    "Nervous System": "तंत्रिका तंत्र",
    "Reproduction": "जनन",
    "Plant Physiology": "पादप शरीर क्रिया विज्ञान",
    "Human Health and Disease": "मानव स्वास्थ्य और रोग",
    "Biotechnology": "जैव प्रौद्योगिकी",
    "Biodiversity": "जैव विविधता",
    "Microbes": "सूक्ष्मजीव",
}


def _get_subtopic_in_language(subtopic_en: str, is_hindi: bool) -> str:
    """Convert English subtopic to Hindi if is_hindi=True."""
    if not is_hindi:
        return subtopic_en
    return _SUBTOPIC_HINDI_MAP.get(subtopic_en, subtopic_en)


def infer_subtopic_from_question(q: dict, user_subject: str, user_chapter: str,
                                  is_hindi: bool = False) -> str:
    existing = _s(q.get("subtopic"))
    if existing:
        return existing

    question_text = (_s(q.get("question")) + " " + _s(q.get("Explanation"))).lower()
    chapter_lower = user_chapter.lower() if user_chapter else ""
    subject_lower = user_subject.lower() if user_subject else ""

    # Physics
    if "physics" in subject_lower or "phy" in subject_lower or "भौतिक" in subject_lower:
        kw_map = [
            (["friction", "rough", "slipping", "sliding", "coefficient of friction",
              "घर्षण", "खुरदरा", "फिसलन"], "Friction"),
            (["newton", "force", "motion", "inertia", "momentum",
              "न्यूटन", "बल", "गति", "जड़त्व", "संवेग"], "Laws of Motion"),
            (["work", "energy", "power", "kinetic", "potential", "conservative",
              "कार्य", "ऊर्जा", "शक्ति", "गतिज", "स्थितिज"], "Work Energy Power"),
            (["gravitation", "gravity", "orbital", "escape velocity", "satellite",
              "गुरुत्वाकर्षण", "गुरुत्व", "उपग्रह"], "Gravitation"),
            (["current", "resistance", "ohm", "circuit", "kirchhoff", "battery", "cell",
              "धारा", "प्रतिरोध", "परिपथ", "बैटरी"], "Current Electricity"),
            (["electric field", "charge", "coulomb", "potential", "capacitor", "gauss",
              "विद्युत क्षेत्र", "आवेश", "संधारित्र"], "Electrostatics"),
            (["magnetic", "lorentz", "ampere", "solenoid", "biot", "flux",
              "चुम्बकीय", "चुंबक"], "Magnetism"),
            (["wave", "frequency", "amplitude", "interference", "diffraction", "sound",
              "तरंग", "आवृत्ति", "आयाम", "ध्वनि"], "Waves"),
            (["optics", "lens", "mirror", "refraction", "reflection", "prism", "snell",
              "प्रकाशिकी", "लेंस", "दर्पण", "अपवर्तन", "परावर्तन"], "Optics"),
            (["thermodynamics", "heat", "temperature", "entropy", "carnot",
              "ऊष्मागतिकी", "ऊष्मा", "तापमान"], "Thermodynamics"),
            (["semiconductor", "diode", "transistor", "logic gate",
              "अर्धचालक", "डायोड", "ट्रांजिस्टर"], "Semiconductors"),
            (["atom", "bohr", "nuclear", "radioactive", "decay", "fission", "fusion",
              "परमाणु", "नाभिक", "रेडियोसक्रिय"], "Atoms and Nuclei"),
        ]
        for keywords, subtopic in kw_map:
            if any(kw in question_text for kw in keywords):
                return _get_subtopic_in_language(subtopic, is_hindi)

    # Chemistry
    elif "chemistry" in subject_lower or "chem" in subject_lower or "रसायन" in subject_lower:
        kw_map = [
            (["order of reaction", "rate law", "half life", "first order", "zero order",
              "अभिक्रिया की कोटि", "दर नियम", "अर्ध आयु"], "Order of Reaction"),
            (["mole", "avogadro", "stoichiometry", "मोल", "अवोगाद्रो"], "Mole Concept"),
            (["equilibrium", "le chatelier", "साम्य", "ले-शातेलिए"], "Chemical Equilibrium"),
            (["acid", "base", "ph", "buffer", "अम्ल", "क्षार", "बफर"], "Acids Bases and Salts"),
            (["electrochemistry", "galvanic", "electrolysis", "faraday",
              "विद्युत रसायन", "विद्युत अपघटन"], "Electrochemistry"),
            (["organic", "functional group", "iupac", "isomer", "alkane",
              "कार्बनिक", "क्रियात्मक समूह", "आइसोमर"], "Organic Chemistry"),
            (["periodic", "ionization energy", "आवर्त", "आयनन ऊर्जा"], "Periodic Table"),
            (["chemical bonding", "covalent", "ionic", "hybridization",
              "रासायनिक बंध", "सहसंयोजक", "आयनिक"], "Chemical Bonding"),
            (["kinetics", "rate of reaction", "activation energy",
              "अभिक्रिया की दर", "सक्रियण ऊर्जा"], "Chemical Kinetics"),
            (["thermochemistry", "enthalpy", "entropy", "gibbs", "hess",
              "ऊष्मारसायन", "एन्थैल्पी", "एन्ट्रॉपी"], "Thermochemistry"),
            (["solution", "molarity", "molality", "raoult", "osmosis",
              "विलयन", "मोलरता", "परासरण"], "Solutions"),
        ]
        for keywords, subtopic in kw_map:
            if any(kw in question_text for kw in keywords):
                return _get_subtopic_in_language(subtopic, is_hindi)

    # Mathematics
    elif "math" in subject_lower or "maths" in subject_lower or "गणित" in subject_lower:
        kw_map = [
            (["integrate", "integral", "∫", "समाकल", "एकीकरण"], "Integration"),
            (["differentiate", "derivative", "dy/dx", "अवकल", "व्युत्पन्न"], "Differentiation"),
            (["limit", "continuity", "सीमा", "सातत्य"], "Limits and Continuity"),
            (["matrix", "determinant", "आव्यूह", "सारणिक"], "Matrices and Determinants"),
            (["vector", "dot product", "cross product", "सदिश", "अदिश गुणनफल"], "Vectors"),
            (["probability", "bayes", "conditional", "संभावना", "प्रायिकता"], "Probability"),
            (["conic", "parabola", "ellipse", "hyperbola", "circle",
              "शंकु", "परवलय", "दीर्घवृत्त", "अतिपरवलय", "वृत्त"], "Conic Sections"),
            (["sequence", "series", "ap", "gp", "श्रेणी", "अनुक्रम", "समांतर श्रेढ़ी"], "Sequences and Series"),
            (["trigonometry", "sin", "cos", "tan", "त्रिकोणमिति"], "Trigonometry"),
            (["complex number", "argand", "सम्मिश्र संख्या", "आर्गण्ड"], "Complex Numbers"),
            (["set", "relation", "function", "समुच्चय", "संबंध", "फलन"], "Sets Relations Functions"),
            (["straight line", "slope", "सरल रेखा", "ढाल"], "Straight Lines"),
            (["binomial theorem", "द्विपद प्रमेय"], "Binomial Theorem"),
        ]
        for keywords, subtopic in kw_map:
            if any(kw in question_text for kw in keywords):
                return _get_subtopic_in_language(subtopic, is_hindi)

    # Biology
    elif "biology" in subject_lower or "bio" in subject_lower or "जीव" in subject_lower:
        kw_map = [
            (["cell", "mitochondria", "nucleus", "कोशिका", "माइटोकॉन्ड्रिया", "केन्द्रक"], "Cell Biology"),
            (["photosynthesis", "chlorophyll", "प्रकाश संश्लेषण", "क्लोरोफिल"], "Photosynthesis"),
            (["respiration", "atp", "glycolysis", "श्वसन", "ग्लाइकोलाइसिस"], "Respiration"),
            (["genetics", "mendel", "allele", "gene", "आनुवंशिकी", "जीन", "मेंडल"], "Genetics"),
            (["evolution", "natural selection", "विकास", "प्राकृतिक चयन"], "Evolution"),
            (["ecology", "ecosystem", "food chain", "पारिस्थितिकी", "पारितंत्र", "खाद्य श्रृंखला"], "Ecology"),
            (["hormone", "endocrine", "हार्मोन", "अंतःस्रावी"], "Endocrine System"),
            (["digestion", "enzyme", "पाचन", "एंजाइम"], "Digestive System"),
            (["nervous", "neuron", "तंत्रिका", "न्यूरॉन"], "Nervous System"),
            (["reproduction", "meiosis", "mitosis", "जनन", "अर्धसूत्री विभाजन"], "Reproduction"),
            (["plant", "root", "stem", "leaf", "पौधा", "जड़", "तना", "पत्ती"], "Plant Physiology"),
        ]
        for keywords, subtopic in kw_map:
            if any(kw in question_text for kw in keywords):
                return _get_subtopic_in_language(subtopic, is_hindi)

    if chapter_lower and len(chapter_lower) > 4:
        return user_chapter.strip()

    return ""


# ================================================================
# OTHER POST-PROCESSING HELPERS
# ================================================================

def scrub_references_from_question(text):
    if not text or not isinstance(text, str):
        return text
    pattern = re.compile(
        r'\s*[-–—]?\s*[\(\[]\s*'
        r'(?:Exercise|Ex\.?|Misc(?:ellaneous)?\s*Exercise|Example|NCERT|PYQ|'
        r'JEE\s*(?:Main|Advanced)?|NEET|AIIMS|CBSE|ICSE|Board)'
        r'[\s\d\.,-]*[\)\]]\s*$',
        re.IGNORECASE
    )
    text = pattern.sub('', text)
    text = re.compile(r'\s*[\(\[]\s*(?:19|20)\d{2}\s*[\)\]]\s*$').sub('', text)
    text = re.sub(r'(\\+\))\)+$', r'\1', text)
    text = re.sub(r'(\.\s*)\)+$', r'\1', text)
    return text.strip()


def is_garbage_field(value, field_name):
    if not value or not isinstance(value, str):
        return False
    garbage_patterns = [
        r'(?i)^(STD\s*\d+\s*[-–]\s*TEST)',
        r'(?i)^TEST\s+(Chapter|Topic|Subtopic|Subject)',
        r'(?i)^(As per image|From image|See image|N/A|null|none)$',
        r'(?i)^(identify from|extract from|identify specific)',
        r'(?i)^(unknown|not specified|not found|not available)$',
    ]
    for pattern in garbage_patterns:
        if re.search(pattern, value.strip()):
            return True
    return False


def text_continues_naturally(prev_text, next_text):
    if not prev_text or not next_text:
        return False
    prev_end = prev_text.strip()[-50:] if len(prev_text) > 50 else prev_text.strip()
    next_start = next_text.strip()[:50] if len(next_text) > 50 else next_text.strip()
    continuation_words = [
        'as', 'the', 'and', 'or', 'but', 'so', 'therefore', 'hence', 'thus',
        'then', 'also', 'if', 'when', 'where', 'which', 'that', 'this', 'these',
        'those', 'its', 'their', 'from', 'with', 'without', 'by', 'for', 'of',
        'to', 'in', 'on', 'at', 'since', 'because', 'while', 'although',
        'और', 'या', 'लेकिन', 'इसलिए', 'अतः', 'यदि', 'जब', 'जहाँ', 'जो',
        'यह', 'वह', 'इस', 'उस', 'से', 'के', 'की', 'को', 'में', 'पर'
    ]
    first_word = next_start.split()[0].lower() if next_start.split() else ""
    ends_mid = prev_end[-1] not in '.!?।'
    is_continuation = (
        ends_mid or
        first_word in continuation_words or
        next_start[0].islower() or
        re.match(r'^[\(\[]', next_start) or
        re.match(r'^\d+[\.\)]', next_start) is None
    )
    return is_continuation


# ================================================================
# SPLIT COMBINED QUESTIONS — FIXED: don't split sub-parts
# ================================================================

_NEW_Q_AFTER_ANS_RE = re.compile(
    r'(?:Ans(?:wer)?|Sol(?:ution)?|उत्तर|हल)[.\s:→]*'
    r'(?:\(?[A-D1-4]\)?)?\s*'
    r'(?:.*?\n)+?'
    r'\s*(\d+[\.\)]\s+[A-Z\(]|Q\.?\s*\d+[\.\)])',
    re.IGNORECASE | re.DOTALL
)

_ANS_BLOCK_RE = re.compile(
    r'\n?\s*(?:Ans(?:wer)?|Sol(?:ution)?|उत्तर|हल)[.\s:→]*(\(?[A-D1-4]\)?)?\s*',
    re.IGNORECASE
)

# ── Only match top-level numbered questions, NOT sub-parts ──
_Q_NUMBER_RE = re.compile(
    r'(?:^|\n)\s*(\d+[\.\)]\s+)',
    re.MULTILINE
)

# ── FIX: Pattern to detect sub-parts — a., b., c., (i), (ii), (a), (b), 1., 2., 3. ──
_SUBPART_RE = re.compile(
    r'(?:^|\n)\s*(?:'
    r'[a-d][\.\)]\s+'       # a. b. c. d. or a) b) c) d)
    r'|\([a-d]\)\s+'        # (a) (b) (c) (d)
    r'|\([ivxIVX]+\)\s+'    # (i) (ii) (iii) (iv)
    r'|[ivx]+[\.\)]\s+'     # i. ii. iii. iv.
    r'|\(\d+\)\s+'          # (1) (2) (3) (4)
    r')',
    re.MULTILINE
)


def split_combined_questions(q):
    question_text = _s(q.get("question"))
    if not question_text:
        return [q]
    if not _NEW_Q_AFTER_ANS_RE.search(question_text):
        return [q]

    # ── FIX: If question has sub-parts (a, b, c, i, ii, 1, 2 etc.), do NOT split ──
    if _SUBPART_RE.search(question_text):
        return [q]

    q_starts = [m.start() for m in _Q_NUMBER_RE.finditer(question_text)]
    if len(q_starts) <= 1:
        return [q]
    segments = []
    for i, start in enumerate(q_starts):
        end = q_starts[i + 1] if i + 1 < len(q_starts) else len(question_text)
        segments.append(question_text[start:end].strip())
    if len(segments) <= 1:
        return [q]
    result = []
    for seg in segments:
        new_q = {k: v for k, v in q.items()}
        new_q["questionid"] = ""
        new_q["question"]    = seg
        new_q["Answer"]      = ""
        new_q["Explanation"] = ""
        ans_match = _ANS_BLOCK_RE.search(seg)
        if ans_match:
            new_q["question"] = seg[:ans_match.start()].strip()
            ans_body = seg[ans_match.end():].strip()
            letter_match = re.match(r'^(\(?[A-D1-4]\)?)', ans_body)
            if letter_match:
                raw = re.sub(r'[^A-D1-4]', '', letter_match.group(1)).upper()
                letter_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4'}
                new_q["Answer"] = letter_map.get(raw, raw)
                ans_body = ans_body[letter_match.end():].strip()
            ans_body = re.sub(
                r'^(?:Sol(?:ution)?|Explanation|व्याख्या|हल)[.\s:→]*',
                '', ans_body, flags=re.IGNORECASE
            ).strip()
            new_q["Explanation"] = ans_body
        if _s(new_q["question"]):
            result.append(new_q)
    return result if len(result) > 1 else [q]


def _s(val):
    """Safe strip — handles None/non-string values."""
    return (val or "").strip() if isinstance(val, (str, type(None))) else str(val).strip()


_ANS_PREFIX_RE = re.compile(
    r'^(?:ans(?:wer)?|sol(?:ution)?|therefore|hence|thus|explanation|'
    r'correct\s+(?:option|answer)|the\s+answer|'
    r'उत्तर|हल|अतः|इसलिए|व्याख्या|सही\s+(?:विकल्प|उत्तर))[.\s:→]*',
    re.IGNORECASE | re.UNICODE
)


def is_continuation_fragment(q):
    qid   = _s(q.get("questionid"))
    qtext = _s(q.get("question"))
    if qid == "" and qtext == "":
        return True
    if qid == "" and qtext and _ANS_PREFIX_RE.match(qtext):
        return True
    return False


def _redistribute_fragment_fields(frag):
    frag = dict(frag)
    qtext = _s(frag.get("question"))
    if not qtext:
        return frag
    ans_match = re.match(
        r'^(?:ans(?:wer)?[.\s:→]*|उत्तर[.\s:→]*)\(?([A-D1-4])\)?\s*[,.]?\s*',
        qtext, re.IGNORECASE | re.UNICODE
    )
    if ans_match:
        frag["Answer"] = frag.get("Answer") or ans_match.group(1)
        rest = qtext[ans_match.end():].strip()
        rest = re.sub(r'^(?:sol(?:ution)?|explanation|हल|व्याख्या)[.\s:→]*', '', rest,
                      flags=re.IGNORECASE).strip()
        if rest:
            frag["Explanation"] = ((_s(frag.get("Explanation")) + " " + rest).strip()
                                   if _s(frag.get("Explanation")) else rest)
        frag["question"] = ""
        return frag
    if _ANS_PREFIX_RE.match(qtext):
        text = _ANS_PREFIX_RE.sub("", qtext).strip()
        if text:
            frag["Explanation"] = ((_s(frag.get("Explanation")) + " " + text).strip()
                                   if _s(frag.get("Explanation")) else text)
        frag["question"] = ""
    return frag


def _question_needs_answer(q):
    return _s(q.get("Answer")) == "" and _s(q.get("Explanation")) == ""


def merge_question_parts(base_q, continuation_q):
    continuation_q = _redistribute_fragment_fields(continuation_q)
    merged = dict(base_q)
    base_question = _s(merged.get("question"))
    cont_question = _s(continuation_q.get("question"))
    if cont_question:
        merged["question"] = (base_question + " " + cont_question).strip() if base_question else cont_question
    base_exp = _s(merged.get("Explanation"))
    cont_exp = _s(continuation_q.get("Explanation"))
    if cont_exp:
        merged["Explanation"] = (base_exp + " " + cont_exp).strip() if base_exp else cont_exp
    for i in range(1, 5):
        key = f"option{i}"
        if not _s(merged.get(key)) and _s(continuation_q.get(key)):
            merged[key] = continuation_q[key]
    if not _s(merged.get("Answer")) and _s(continuation_q.get("Answer")):
        merged["Answer"] = continuation_q["Answer"]
    if not _s(str(merged.get("marks", ""))) and _s(str(continuation_q.get("marks", ""))):
        merged["marks"] = continuation_q["marks"]
    if not _s(merged.get("previous_year")) and _s(continuation_q.get("previous_year")):
        merged["previous_year"] = continuation_q["previous_year"]
    if not _s(merged.get("subtopic")) and _s(continuation_q.get("subtopic")):
        merged["subtopic"] = continuation_q["subtopic"]
    if not _s(merged.get("question_bucket")) and _s(continuation_q.get("question_bucket")):
        merged["question_bucket"] = continuation_q["question_bucket"]
    return merged


# ================================================================
# MARKS / ANSWER / FIELD FIXES
# ================================================================
_SECTION_EACH_RE = re.compile(
    r'(\d+)\s*[Mm]arks?\s+[Ee]ach|'
    r'[Ee]ach\s+(?:of\s+)?(\d+)\s*[Mm]arks?|'
    r'questions?\s+of\s+(\d+)\s*[Mm]arks?|'
    r'(\d+)\s*अंक\s+(?:प्रत्येक|each)|'
    r'प्रत्येक\s+(?:प्रश्न\s+)?(\d+)\s*अंक',
)


def extract_section_marks_from_text(text):
    if not text:
        return ""
    m = _SECTION_EACH_RE.search(str(text))
    if m:
        for g in m.groups():
            if g:
                return g
    return ""


def fix_marks_field(marks_val):
    if marks_val is None:
        return ""
    val = str(marks_val).strip()
    if re.match(r'^\d+$', val):
        return val
    patterns = [
        r'\[(\d+)\s*[Mm]arks?\]', r'\((\d+)\s*[Mm]arks?\)',
        r'[Mm]arks?\s*[:\-]\s*(\d+)', r'(\d+)\s*[Mm]arks?',
        r'\[(\d+)\s*अंक\]', r'\((\d+)\s*अंक\)',
        r'(\d+)\s*अंक',
        r'\[(\d+)\]', r'\((\d+)\)', r'(\d+)',
    ]
    for pattern in patterns:
        m = re.search(pattern, val)
        if m:
            return m.group(1)
    return ""


def infer_marks_from_sections(all_questions, page_marks_map=None, question_page_map=None):
    if page_marks_map and question_page_map:
        for i, q in enumerate(all_questions):
            page_idx = question_page_map.get(i, -1)
            if page_idx >= 0 and page_idx < len(page_marks_map):
                section_mark = page_marks_map[page_idx]
                if section_mark:
                    existing = fix_marks_field(q.get("marks", ""))
                    if not existing:
                        q["marks"] = section_mark

    current_section_marks = ""
    for q in all_questions:
        candidate_texts = [q.get("marks", ""), q.get("Explanation", ""), q.get("question", "")]
        found_marks = ""
        for text in candidate_texts:
            found_marks = extract_section_marks_from_text(text)
            if found_marks:
                break
        if found_marks:
            current_section_marks = found_marks
        marks_now = str(q.get("marks", "")).strip()
        cleaned_marks = fix_marks_field(marks_now)
        if cleaned_marks:
            q["marks"] = cleaned_marks
        elif current_section_marks:
            q["marks"] = current_section_marks

    for i in range(len(all_questions) - 1, -1, -1):
        if not str(all_questions[i].get("marks", "")).strip():
            for j in range(i + 1, min(i + 10, len(all_questions))):
                fwd = str(all_questions[j].get("marks", "")).strip()
                if fwd:
                    all_questions[i]["marks"] = fwd
                    break

    return all_questions


def fix_answer_field(answer_val, options, question_type):
    q_type = str(question_type).lower()
    if "true" in q_type or "false" in q_type or "सत्य" in q_type or "असत्य" in q_type:
        if not answer_val or not isinstance(answer_val, str):
            return ""
        val = answer_val.strip().lower()
        if val in ("true", "1", "t", "yes", "सत्य", "हाँ"):
            return "1"
        if val in ("false", "0", "f", "no", "असत्य", "नहीं"):
            return "0"
        return answer_val
    if "numeric" in q_type or "integer" in q_type or "संख्यात्मक" in q_type:
        if not answer_val and answer_val != 0:
            return ""
        val = str(answer_val).strip()
        try:
            return str(int(float(val)))
        except Exception:
            m = re.search(r'\d+', val)
            return m.group(0) if m else val
    if "mcq" in q_type or "multiple" in q_type or "बहुविकल्पीय" in q_type:
        if not answer_val or not isinstance(answer_val, str):
            return ""
        val = answer_val.strip()
        if val in ('1', '2', '3', '4'):
            return val
        letter_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4',
                      'अ': '1', 'ब': '2', 'स': '3', 'द': '4'}
        inner = re.sub(r'[()]', '', val).strip().upper()
        if inner in letter_map:
            return letter_map[inner]
        if inner in ('1', '2', '3', '4'):
            return inner
        return answer_val
    return str(answer_val).strip() if answer_val else ""


# ================================================================
# QUESTION BUCKET NORMALISER
# ================================================================
VALID_BUCKETS = {"Beginner", "Target", "Advance Climb", "Must Do"}


def normalize_question_bucket(bucket_raw):
    if not bucket_raw or not isinstance(bucket_raw, str):
        return "Beginner"
    val = bucket_raw.strip()
    if val in VALID_BUCKETS:
        return val
    lower = val.lower()
    for valid in VALID_BUCKETS:
        if valid.lower() == lower:
            return valid
    if "advance" in lower or "climb" in lower:
        return "Advance Climb"
    if "must" in lower or "critical" in lower or "important" in lower:
        return "Must Do"
    if "target" in lower or "moderate" in lower or "medium" in lower:
        return "Target"
    if "begin" in lower or "easy" in lower or "basic" in lower or "simple" in lower:
        return "Beginner"
    return "Beginner"


# ================================================================
# QUESTION TYPE NORMALISER
# ================================================================
def normalize_question_type(q_type_raw):
    q_type = str(q_type_raw).strip()
    lower  = q_type.lower()
    if "true" in lower or "false" in lower or "सत्य" in lower or "असत्य" in lower:
        return "True/False"
    elif "assertion" in lower or "कथन" in lower:
        return "Assertion and Reasoning Questions ( A& R )"
    elif "match" in lower or "मिलान" in lower:
        return "Match the Column Question"
    elif "case" in lower or "प्रकरण" in lower:
        return "Case Based Questions (CBQ)"
    elif "blank" in lower or "filling" in lower or "fill" in lower or "रिक्त" in lower:
        return "Filling Blank"
    elif "numeric" in lower or "integer" in lower or "संख्यात्मक" in lower:
        return "Numeric"
    elif "subjective" in lower or "दीर्घ" in lower or "लघु" in lower:
        return "Subjective"
    else:
        return "MCQs"


def enforce_question_type_rules(q):
    q_type = q.get("question_type", "MCQs")

    if q_type == "MCQs":
        ans = str(q.get("Answer", "")).strip()
        if ans not in ("1", "2", "3", "4"):
            letter_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4'}
            inner = re.sub(r'[^A-Da-d1-4]', '', ans).upper()
            if inner and inner[0] in letter_map:
                q["Answer"] = letter_map[inner[0]]
            elif inner and inner[0] in ('1', '2', '3', '4'):
                q["Answer"] = inner[0]
            else:
                q["Answer"] = ""

    elif q_type == "True/False":
        q["option1"] = q["option2"] = q["option3"] = q["option4"] = ""
        ans = str(q.get("Answer", "")).strip().lower()
        if ans in ("true", "1", "t", "yes", "सत्य") or (ans and "true" in ans and "false" not in ans):
            q["Answer"] = "1"
        elif ans in ("false", "0", "f", "no", "असत्य") or (ans and "false" in ans):
            q["Answer"] = "0"
        else:
            q["Answer"] = ""

    elif q_type == "Numeric":
        q["option1"] = q["option2"] = q["option3"] = q["option4"] = ""
        ans = str(q.get("Answer", "")).strip()
        try:
            q["Answer"] = str(int(float(ans))) if ans else ""
        except Exception:
            m = re.search(r'\d+', ans)
            q["Answer"] = m.group(0) if m else ""

    elif q_type == "Filling Blank":
        q["option1"] = q["option2"] = q["option3"] = q["option4"] = ""

    elif q_type in (
        "Subjective",
        "Assertion and Reasoning Questions ( A& R )",
        "Match the Column Question",
        "Case Based Questions (CBQ)",
    ):
        q["option1"] = q["option2"] = q["option3"] = q["option4"] = ""
        q["Answer"] = ""

    return q


def apply_field_order(q):
    ordered = {}
    for k in FIELD_ORDER:
        ordered[k] = q.get(k, "")
    for k, v in q.items():
        if k not in ordered and k != "category":
            ordered[k] = v
    return ordered


_ANY_TABLE_RE = re.compile(
    r'(<table\b[^>]*>.*?</table\s*>)',
    re.IGNORECASE | re.DOTALL
)

# Matches solution text that sneaks into the question field:
# "Sol.", "Ans.", "LCM =", "HCF =", "∴", "Therefore", "Hence", "उत्तर", "हल"
_SOLUTION_SNEAK_RE = re.compile(
    r'(?:<br\s*/?>|\n|\s{2,})\s*'
    r'(?:'
    r'sol(?:ution)?\.?\s*[:\-→]?|'
    r'ans(?:wer)?\.?\s*[:\-→]?|'
    r'explanation\s*[:\-→]?|'
    r'lcm\s*[=:]|hcf\s*[=:]|'
    r'∴\s*lcm|∴\s*hcf|'
    r'therefore[,\s]|hence[,\s]|thus[,\s]|'
    r'उत्तर\s*[:\-]?|हल\s*[:\-]?|व्याख्या\s*[:\-]?'
    r')',
    re.IGNORECASE | re.DOTALL
)


def fix_misplaced_tables(q):
    """Move HTML tables AND embedded solution text from 'question' field → 'Explanation'.
    Runs after every merge/stitch operation as a safety net.
    Skip: Match the Column (tables in question are intentional).
    """
    q_type = q.get("question_type", "")
    if "Match" in q_type:
        return q

    question_text = q.get("question", "")
    if not question_text or not isinstance(question_text, str):
        return q

    extra_parts = []
    clean_q = question_text

    # ── Step 1: extract and remove <table> blocks from question ──
    if '<table' in clean_q.lower():
        tables_found = _ANY_TABLE_RE.findall(clean_q)
        if tables_found:
            clean_q = _ANY_TABLE_RE.sub('', clean_q)
            extra_parts.extend(tables_found)

    # ── Step 1b: extract plain-text LCM/HCF ladder from question ──
    # Detect "2 | 15  20  25" lines inside the question field
    _ladder_in_q = re.search(r'(?:<br>|\n|\s{2,})\s*\d+\s*\|', clean_q)
    if _ladder_in_q and _ladder_in_q.start() > 10:
        ladder_tail = clean_q[_ladder_in_q.start():]
        clean_q = clean_q[:_ladder_in_q.start()].strip()
        if ladder_tail.strip():
            extra_parts.append(ladder_tail.strip())

    # ── Step 2: split off solution text that sneaks into question ──
    sol_match = _SOLUTION_SNEAK_RE.search(clean_q)
    if sol_match and sol_match.start() > 15:
        solution_tail = clean_q[sol_match.start():].strip()
        clean_q = clean_q[:sol_match.start()].strip()
        if solution_tail:
            extra_parts.insert(0, solution_tail)

    if not extra_parts:
        return q

    # ── Step 3: clean up question text ──
    clean_q = re.sub(r'(<br\s*/?>\s*){2,}', '<br>', clean_q).strip()
    clean_q = re.sub(r'\s{2,}', ' ', clean_q).strip()

    # ── Step 4: prepend rescued content to Explanation ──
    existing_exp = (q.get("Explanation", "") or "").strip()
    rescued = '\n'.join(p.strip() for p in extra_parts if p.strip())
    q["Explanation"] = (rescued + '\n' + existing_exp).strip() if existing_exp else rescued
    q["question"] = clean_q
    return q


# ================================================================
# UNIFIED CLEAN_QUESTION
# ================================================================
def clean_question(q, page_images=None,
                   user_subject="", user_course="", user_class="",
                   user_chapter="", user_practice="", user_book="",
                   is_hindi=False):
    """Full post-processing pipeline for a single extracted question."""
    math_fields     = ["question", "option1", "option2", "option3", "option4", "Explanation"]
    metadata_fields = ["subjectname", "chapter", "practice", "subtopic", "medium",
                       "difficulty", "question_type", "course"]

    for k in list(q.keys()):
        if q[k] is None:
            q[k] = ""

    q.pop("category", None)
    q["question_type"] = normalize_question_type(q.get("question_type") or "MCQs")
    q["question_bucket"] = normalize_question_bucket(q.get("question_bucket") or "")

    # ── FIX 1: medium must ALWAYS come from is_hindi toggle, not hardcoded ──
    q["medium"] = "Hindi" if is_hindi else "English"

    # ── FIX: Extract previous_year from question text ──
    existing_py = fix_previous_year_field(_s(q.get("previous_year")))
    if not existing_py:
        q_text = _s(q.get("question"))
        ref, cleaned_q = extract_previous_year_from_text(q_text)
        if ref:
            existing_py = fix_previous_year_field(ref)
            if existing_py:
                q["question"] = cleaned_q
        if not existing_py:
            exp_text = _s(q.get("Explanation"))
            ref, cleaned_exp = extract_previous_year_from_text(exp_text)
            if ref:
                existing_py = fix_previous_year_field(ref)
                if existing_py:
                    q["Explanation"] = cleaned_exp

    q["previous_year"] = existing_py

    if "question" in q:
        q["question"] = scrub_references_from_question(q["question"])

    for field in list(q.keys()):
        if isinstance(q[field], str):
            q[field] = remove_cite_tags(q[field])

    if "Explanation" in q:
        q["Explanation"] = clean_explanation_prefix(q["Explanation"])
        # Strip stray inline HTML tags (e.g. <b>, <i>, <u>) but keep <img> and <table>
        if q["Explanation"] and isinstance(q["Explanation"], str):
            q["Explanation"] = re.sub(
                r'<(?!img\b|/img\b|table\b|/table\b|tr\b|/tr\b|td\b|/td\b|th\b|/th\b|br\b|div\b|/div\b)[^>]+>',
                '', q["Explanation"]
            )

    for field in math_fields:
        if field in q:
            q[field] = convert_dollar_to_latex(q[field])

    if "Answer" in q:
        options = {
            "A": q.get("option1") or "", "B": q.get("option2") or "",
            "C": q.get("option3") or "", "D": q.get("option4") or "",
        }
        q["Answer"] = fix_answer_field(q["Answer"], options, q["question_type"])

    q["marks"] = fix_marks_field(q.get("marks") or "")

    for field in metadata_fields:
        if field in q and is_garbage_field(q[field], field):
            q[field] = ""

    q = enforce_question_type_rules(q)

    # CRITICAL: inject diagrams FIRST — before any further text processing
    # that could accidentally strip the injected <img> tags
    q = inject_diagrams_into_question(q, page_images or [])

    if user_subject:  q["subjectname"] = user_subject
    if user_course:   q["course"]      = user_course
    if user_class:    q["class"]       = user_class
    if user_chapter:  q["chapter"]     = user_chapter
    q["practice"] = user_practice if user_practice else q.get("practice", "")
    if user_book:     q["book"]        = user_book

    # ── FIX 1 (again, final override): medium is always from toggle ──
    q["medium"] = "Hindi" if is_hindi else "English"

    # ── FIX: Pass is_hindi to subtopic inference ──
    if not _s(q.get("subtopic")):
        inferred = infer_subtopic_from_question(q, user_subject, user_chapter, is_hindi=is_hindi)
        if inferred:
            q["subtopic"] = inferred

    # ── Move any misplaced LCM/data tables out of question into Explanation ──
    q = fix_misplaced_tables(q)

    # ── FIX 3: newline_to_br skips table content ──
    br_fields = ["question", "option1", "option2", "option3", "option4", "Explanation"]
    for field in br_fields:
        if field in q:
            q[field] = newline_to_br(q[field])

    return apply_field_order(q)


# ================================================================
# JSON CLEANER
# ================================================================
def _syntax_repair(text):
    text = re.sub(r',\s*([}\]])', r'\1', text)
    text = re.sub(r'\bNone\b', 'null', text)
    text = re.sub(r'\bTrue\b', 'true', text)
    text = re.sub(r'\bFalse\b', 'false', text)
    text = re.sub(r'//[^\n]*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text


def _close_open_structures(text):
    text = text.rstrip().rstrip(',')
    depth_brace = 0
    depth_bracket = 0
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth_brace += 1
        elif ch == '}':
            depth_brace -= 1
        elif ch == '[':
            depth_bracket += 1
        elif ch == ']':
            depth_bracket -= 1
    text += '}' * max(0, depth_brace)
    text += ']' * max(0, depth_bracket)
    return text


def clean_json_response(text):
    if not text:
        return None
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    def _try_parse(s):
        s = _syntax_repair(s)
        try:
            return json.loads(s)
        except Exception:
            pass
        s2 = _close_open_structures(s)
        try:
            result = json.loads(s2)
            if result:
                print(f"⚠️ JSON repaired: {len(result) if isinstance(result, list) else 1} item(s)")
            return result
        except Exception:
            return None

    def _only_dicts(lst):
        """Keep only dict items — drops stray strings/numbers in parsed list."""
        if not isinstance(lst, list):
            return None
        dicts = [x for x in lst if isinstance(x, dict)]
        return dicts if dicts else None

    start_idx = text.find('[')
    end_idx   = text.rfind(']')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        result = _try_parse(text[start_idx:end_idx + 1])
        if result and isinstance(result, list):
            clean = _only_dicts(result)
            if clean:
                return clean
        result = _try_parse(text[start_idx:])
        if result and isinstance(result, list):
            clean = _only_dicts(result)
            if clean:
                return clean

    obj_start = text.find('{')
    obj_end   = text.rfind('}')
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        result = _try_parse(text[obj_start:obj_end + 1])
        if result and isinstance(result, dict):
            return [result]

    objects = []
    depth = 0
    in_string = False
    escape = False
    obj_start_pos = None
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            if depth == 0:
                obj_start_pos = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and obj_start_pos is not None:
                fragment = text[obj_start_pos:i + 1]
                try:
                    obj = json.loads(_syntax_repair(fragment))
                    if isinstance(obj, dict) and obj:
                        objects.append(obj)
                except Exception:
                    pass
                obj_start_pos = None
    if objects:
        print(f"⚠️ JSON recovered {len(objects)} object(s)")
        return objects

    print("⚠️ JSON Parse failed")
    return None




# ================================================================
# SINGLE QUESTION EXTRACTOR — CORE FUNCTION
# ================================================================

def extract_single_question(
    pages_bytes: list,
    page_images_list: list,
    api_key: str,
    model_name: str,
    source_image_bytes: bytes | None = None,   # raster image (JPG/PNG) being uploaded
    is_hindi: bool = False,
    user_subject: str = "",
    user_course: str = "",
    user_class: str = "",
    user_chapter: str = "",
    user_practice: str = "",
    user_book: str = "",
    progress_cb=None,
):
    """
    Extract ONE complete question (possibly spanning 1–4 pages) and return a clean JSON dict.
    Retries with every available config (including higher thinking budgets) until valid JSON
    is returned. Unlimited retries — as many API calls as needed.

    source_image_bytes: raw bytes of the uploaded raster image (JPG/PNG). When set and no
    pre-extracted diagrams exist, Gemini is asked to return diagram bounding boxes
    (_diagrams field) so we can crop and embed them.
    """
    client     = genai.Client(api_key=api_key)
    medium_val = "Hindi" if is_hindi else "English"

    # Flatten all per-page diagram images into a single list (top-to-bottom order)
    all_page_images: list = []
    for pi_list in page_images_list:
        all_page_images.extend(pi_list)

    num_pages    = len(pages_bytes)
    num_diagrams = len(all_page_images)

    # True when input is a raster image with no pre-extracted vector/raster diagrams
    is_image_input = (source_image_bytes is not None) and (num_diagrams == 0)

    # ── Diagram block ──
    if num_diagrams > 0:
        _lines = "\n".join(
            f"  [DIAGRAM_{i}] = image {i + 1} (top-to-bottom across all pages)"
            for i in range(num_diagrams)
        )
        diagram_note = (
            f"DIAGRAM RULE — {num_diagrams} IMAGE(S) DETECTED\n"
            f"I am sending {num_diagrams} extracted image(s) AFTER the page image(s):\n"
            f"{_lines}\n\n"
            "MANDATORY: Place each [DIAGRAM_X] tag EXACTLY at the position in the text where\n"
            "that diagram physically appears (before the sub-part it illustrates, or right after\n"
            "the sentence/label that refers to it).\n"
            f"You MUST use ALL indices 0–{num_diagrams - 1}. NEVER omit a [DIAGRAM_X] tag.\n"
            "If a diagram appears beside a sub-part like (i), place its tag on its own line before (i)."
        )
    elif is_image_input:
        # ── Image upload: ask Gemini to detect diagrams and return bounding boxes ──
        diagram_note = (
            "DIAGRAM DETECTION — RASTER IMAGE INPUT\n"
            "This page is a raster image (JPG/PNG). No pre-extracted diagrams available.\n"
            "YOU must detect ALL non-text visual elements — EVERY single one, miss NONE:\n"
            "  • Geometry: triangles, circles, polygons, rectangles, number lines, coordinate axes\n"
            "  • Physics: circuits, force diagrams, ray diagrams, pulley/spring systems, wave diagrams\n"
            "  • Chemistry: structural formulae, benzene rings, apparatus, electron-dot diagrams, periodic graphs\n"
            "  • Biology: cell diagrams, organ/anatomy, plant/animal drawings, food chains, classification trees\n"
            "  • Math: graphs, Venn diagrams, probability trees, flowcharts, bar/pie/line charts, canvas drawings\n"
            "  • Commerce/Accounts: ledger tables with rulings, T-accounts, cash-flow diagrams, balance sheets\n"
            "  • Vector/drawn: any shape drawn with lines/curves, hand-drawn sketches, canvas illustrations\n"
            "  • General: any drawing, sketch, photo, table-as-image, or figure that is NOT pure running text\n\n"
            "NUMBERING RULE: assign idx=0 to the TOPMOST diagram, idx=1 to the next, etc. (sequential, no gaps).\n\n"
            "For EACH diagram found:\n"
            "  ① Place [DIAGRAM_X] tag in the question/Explanation EXACTLY at the position where the diagram sits.\n"
            "  ② Add an entry to _diagrams: {\"idx\": X, \"x0_norm\": 0–1, \"y0_norm\": 0–1, \"x1_norm\": 0–1, \"y1_norm\": 0–1}\n"
            "     Coordinates are NORMALIZED (0.0 = top-left, 1.0 = bottom-right of the full image).\n"
            "     INCLUDE all text labels that are PART OF the diagram (A, B, 20m, θ, ∠, etc.) inside the box.\n"
            "     Add ~2% padding on all sides so labels are not clipped.\n"
            "If NO diagrams exist: \"_diagrams\": []"
        )
    else:
        diagram_note = (
            "No pre-extracted diagrams on this page.\n"
            "If you see any figure, graph, shape, or circuit → place [DIAGRAM_0] at that position."
        )

    # ── Language block ──
    if is_hindi:
        lang_block = (
            "\n═══ CRITICAL LANGUAGE RULE — HINDI MEDIUM ═══\n"
            "This question is in HINDI (Devanagari script).\n"
            "Extract ALL text EXACTLY in Hindi. DO NOT translate to English.\n"
            "Math formulas stay as LaTeX. Subtopic MUST be in Hindi.\n"
        )
        subtopic_rule = (
            '"subtopic" MUST be in HINDI — e.g. "\\u0918\\u0930\\u094d\\u0937\\u0923", '
            '"\\u0938\\u092e\\u093e\\u0915\\u0932\\u0928", "\\u0917\\u0924\\u093f \\u0915\\u0947 \\u0928\\u093f\\u092f\\u092e". NEVER leave blank.'
        )
    else:
        lang_block = ""
        subtopic_rule = '"subtopic" in English — e.g. "Integration", "Laws of Motion", "Friction". NEVER leave blank.'

    # For raster image input, add _diagrams field to schema so Gemini returns bounding boxes
    diagrams_schema_field = (
        ',\n  "_diagrams": [{"idx": 0, "x0_norm": 0.0, "y0_norm": 0.0, "x1_norm": 1.0, "y1_norm": 1.0}]'
        if is_image_input else ""
    )

    prompt = f"""You are extracting ONE complete educational question that spans {num_pages} page(s).
ALL the page images I provide form a SINGLE question — possibly with sub-parts (i)(ii)(iii)(iv) or (a)(b)(c)(d).
{lang_block}
══════════════════════════════════════════
CRITICAL SUB-PARTS RULE
══════════════════════════════════════════
If the question has sub-parts with roman numerals (i)(ii)(iii)(iv) or letters (a)(b)(c)(d)
or numbers (1)(2)(3)(4), ALL sub-parts belong TOGETHER in the SINGLE "question" field.
NEVER split sub-parts into separate JSON objects.
The question may span 2–4 pages — read ALL pages together as ONE question.

══════════════════════════════════════════
DIAGRAM RULES
══════════════════════════════════════════
{diagram_note}

══════════════════════════════════════════
EXTRACTION RULES
══════════════════════════════════════════
1. Copy ALL text EXACTLY as printed — verbatim, word-for-word, character-by-character.
2. Convert ALL mathematical expressions to LaTeX:
   Inline math  → \\\\( ... \\\\)
   Display math → \\\\[ ... \\\\]
3. Extract the COMPLETE solution/explanation (every step verbatim).
4. previous_year: look for tags like (Example-1), (Exercise-7.1-8), JEE 2024, NEET 2023,
   CBSE, NCERT, PYQ. Extract exactly as printed and REMOVE from the "question" field.
5. NEVER put solution steps, "Sol.", "Ans." or working in the "question" field.

══════════════════════════════════════════
TABLE RULES
══════════════════════════════════════════
LCM/HCF division ladder → PLAIN TEXT:
  2 | 15  20  25
    |___________
  (Use "divisor | numbers" rows + "  |___" separators)

All other tables (data/match/frequency) → HTML:
  <table class="ltable" style="border-collapse:collapse">
    <tr><th style="border:1px solid #333;padding:5px 10px">Header</th></tr>
    <tr><td style="border:1px solid #333;padding:5px 10px">Data</td></tr>
  </table>

══════════════════════════════════════════
QUESTION TYPE — pick exactly one
══════════════════════════════════════════
MCQs | True/False | Numeric | Filling Blank | Subjective |
Assertion and Reasoning Questions ( A& R ) | Match the Column Question | Case Based Questions (CBQ)

For A&R, options MUST be EXACTLY:
  option1: "Both Assertion (A) and Reason (R) are true and Reason (R) is the correct explanation of Assertion (A)."
  option2: "Both Assertion (A) and Reason (R) are true but Reason (R) is NOT the correct explanation of Assertion (A)."
  option3: "Assertion (A) is true but Reason (R) is false."
  option4: "Assertion (A) is false but Reason (R) is true."

Answer encoding:
  MCQ        → 1 / 2 / 3 / 4  (A=1 B=2 C=3 D=4)
  True/False → 1 (True)  or  0 (False)
  Numeric    → the numeric value
  Subjective / Match / A&R / CBQ → ""

══════════════════════════════════════════
OUTPUT — SINGLE JSON OBJECT (not an array)
══════════════════════════════════════════
{{
  "questionid": "1",
  "question": "<complete verbatim question with ALL sub-parts, LaTeX math, and [DIAGRAM_X] tags>",
  "option1": "<option A text or empty>",
  "option2": "<option B text or empty>",
  "option3": "<option C text or empty>",
  "option4": "<option D text or empty>",
  "Answer": "<per encoding above>",
  "Explanation": "<complete solution — every step verbatim — tables as HTML>",
  "course": "{user_course}",
  "subjectname": "{user_subject}",
  "chapter": "{user_chapter}",
  "practice": "{user_practice}",
  "subtopic": "<{subtopic_rule}>",
  "medium": "{medium_val}",
  "difficulty": "<Easy | Medium | Hard>",
  "question_type": "<type from list above>",
  "previous_year": "<extracted reference or empty>",
  "marks": "<digit or empty>",
  "class": "{user_class}",
  "book": "{user_book}",
  "question_bucket": "<Beginner | Target | Advance Climb | Must Do>"{diagrams_schema_field}
}}

Return ONLY the JSON object. No markdown fences. No preamble. No array brackets.
Unicode/Devanagari must be included as-is (UTF-8). Use \\n for newlines inside strings.
medium = "{medium_val}" always. NEVER put solution steps in the "question" field.
"""

    # ── Build API content: prompt + page images + diagram images ──
    content_parts: list = [prompt]
    for pb in pages_bytes:
        content_parts.append(genai_types.Part.from_bytes(data=pb, mime_type="image/png"))
    for pi in all_page_images:
        content_parts.append(genai_types.Part.from_bytes(
            data=base64.b64decode(pi["image_base64"]), mime_type=pi["mime_type"]
        ))

    # ── Config list: standard configs + extended thinking budgets ──
    all_configs = _make_page_configs()
    for budget in [32768, 65536]:
        try:
            all_configs.append(genai_types.GenerateContentConfig(
                temperature=0, max_output_tokens=65536,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=budget),
            ))
        except Exception:
            pass

    last_raw: str | None = None

    for attempt_num, cfg in enumerate(all_configs):
        label = f"attempt {attempt_num + 1}/{len(all_configs)}"
        if progress_cb:
            progress_cb(f"API {label} …")
        try:
            raw = _call_gemini(client, model_name, content_parts, cfg)
            if not raw:
                if progress_cb:
                    progress_cb(f"{label}: empty response, retrying…")
                continue
            last_raw = raw

            parsed = clean_json_response(raw)
            if parsed:
                q = parsed[0] if isinstance(parsed, list) else parsed
                if isinstance(q, dict) and q.get("question"):
                    # ── IMAGE INPUT: count detected diagram regions; use placeholders ──
                    effective_images = list(all_page_images)
                    if is_image_input and source_image_bytes:
                        _raw_regions = q.pop("_diagrams", None) or []
                        _valid_regions = [
                            r for r in (_raw_regions if isinstance(_raw_regions, list) else [])
                            if isinstance(r, dict) and 'x0_norm' in r and 'y0_norm' in r
                        ]
                        if _valid_regions:
                            # We only need the COUNT — one placeholder entry per detected diagram
                            _max_idx = max(r.get('idx', i) for i, r in enumerate(_valid_regions))
                            effective_images = [{"data_uri": "", "mime_type": "image/png"}
                                                for _ in range(_max_idx + 1)]
                            if progress_cb:
                                progress_cb(f"Detected {len(effective_images)} diagram(s) — placeholders inserted")
                        # Fallback: if [DIAGRAM_X] tags exist but no _diagrams returned
                        if not effective_images:
                            _all_q_text = " ".join(
                                str(v) for v in q.values() if isinstance(v, str)
                            )
                            _n_tags = len(re.findall(r'\[DIAGRAM_\d+\]', _all_q_text))
                            if _n_tags:
                                effective_images = [{"data_uri": "", "mime_type": "image/png"}
                                                    for _ in range(_n_tags)]
                    q = clean_question(
                        q,
                        page_images=effective_images,
                        user_subject=user_subject,
                        user_course=user_course,
                        user_class=user_class,
                        user_chapter=user_chapter,
                        user_practice=user_practice,
                        user_book=user_book,
                        is_hindi=is_hindi,
                    )
                    q["questionid"] = 1
                    q["medium"]     = medium_val
                    if progress_cb:
                        progress_cb(f"Got valid JSON on {label}")
                    return apply_field_order(q)

            if progress_cb:
                progress_cb(f"{label}: JSON parse failed, retrying…")
            time.sleep(1)

        except Exception as e:
            err = str(e)
            if progress_cb:
                progress_cb(f"{label} error: {err[:120]}")
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = min(60 * (attempt_num + 1), 300)
                if progress_cb:
                    progress_cb(f"Rate limited — waiting {wait}s…")
                time.sleep(wait)
            elif "500" in err or "503" in err:
                time.sleep(5)

    # ── Last resort: ask model to fix its own broken JSON ──
    if last_raw:
        if progress_cb:
            progress_cb("JSON repair follow-up call…")
        try:
            fix_prompt = (
                "The text below should be a single JSON object for one educational question. "
                "Fix ALL syntax errors and return ONLY valid JSON. "
                "DO NOT change any content, field names, or values — only fix syntax. "
                "Preserve all Unicode/Devanagari characters exactly.\n\n"
                f"Broken JSON:\n{last_raw[:10000]}\n\n"
                "Return ONLY the fixed JSON object. No markdown. No explanation."
            )
            fix_r  = client.models.generate_content(
                model=model_name, contents=[fix_prompt],
                config=genai_types.GenerateContentConfig(temperature=0, max_output_tokens=65536),
            )
            parsed = clean_json_response(_safe_response_text(fix_r) or "")
            if parsed:
                q = parsed[0] if isinstance(parsed, list) else parsed
                if isinstance(q, dict):
                    q = clean_question(
                        q,
                        page_images=all_page_images,
                        user_subject=user_subject,
                        user_course=user_course,
                        user_class=user_class,
                        user_chapter=user_chapter,
                        user_practice=user_practice,
                        user_book=user_book,
                        is_hindi=is_hindi,
                    )
                    q["questionid"] = 1
                    q["medium"]     = medium_val
                    return apply_field_order(q)
        except Exception:
            pass

    return None


# ================================================================
# STREAMLIT UI — SINGLE QUESTION LATEX EXTRACTOR
# ================================================================

# ── Session state ──
for _k in ["sq_result", "sq_json_text", "sq_rendered_pages", "sq_file_name"]:
    if _k not in st.session_state:
        st.session_state[_k] = None

st.markdown("""<style>
.block-container{padding-top:0.5rem!important;max-width:100%!important}
header{visibility:hidden}
textarea{font-family:"Courier New",monospace!important;font-size:12px!important}
</style>""", unsafe_allow_html=True)

st.title("\U0001f4d0 Single Question LaTeX Extractor")
st.caption(
    "Upload a **PDF (1–4 pages) or image** containing ONE question "
    "(sub-parts i, ii, iii, iv are fine). "
    "Extracts 100% accurate LaTeX JSON with embedded diagrams."
)
st.markdown("---")

# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini API Key", type="password", placeholder="AIza…")
    model_choice = st.selectbox("Model", [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite-preview-06-17",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.0-pro-exp",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ])
    is_hindi = st.toggle(
        "\U0001f1ee\U0001f1f3 Hindi Medium",
        value=False,
        help="Enable for Hindi PDFs — all text stays in Devanagari",
    )
    st.markdown("---")
    st.subheader("\U0001f4cb Metadata (optional)")
    _SUBJ_OPTS = [
        "", "Math", "Physics", "Chemistry", "Biology",
        "Hindi", "English", "Social Science",
        "गणित", "भौतिक विज्ञान",
        "रसायन विज्ञान",
        "जीव विज्ञान", "हिंदी",
    ]
    user_subject  = st.selectbox("Subject",  _SUBJ_OPTS)
    user_course   = st.text_input("Course",   placeholder="JEE Main, NEET, CBSE…")
    user_class    = st.text_input("Class",    placeholder="10, 11, 12…")
    user_chapter  = st.text_input("Chapter",  placeholder="Laws of Motion…")
    user_practice = st.text_input("Practice", placeholder="Exercise 2.1…")
    user_book     = st.text_input("Book",     placeholder="NCERT, RD Sharma…")
    st.markdown("---")
    if st.button("\U0001f50c Test API Connection"):
        if not api_key:
            st.error("Enter API key first!")
        else:
            with st.spinner("Testing…"):
                try:
                    _c = genai.Client(api_key=api_key)
                    _r = _c.models.generate_content(
                        model=model_choice,
                        contents=["Reply with: OK"],
                        config=genai_types.GenerateContentConfig(
                            temperature=0, max_output_tokens=10
                        ),
                    )
                    _txt = _safe_response_text(_r)
                    if _txt:
                        st.success(f"✅ API OK — {_txt.strip()[:60]}")
                    else:
                        st.warning("Connected but empty response")
                except Exception as _te:
                    st.error(f"❌ {_te}")

# ── File uploader ──
uploaded_file = st.file_uploader(
    "\U0001f4c2 Upload PDF or Image (JPG / PNG) — single question, up to 4 pages",
    type=["pdf", "jpg", "jpeg", "png"],
)

# Reset when a different file is uploaded
if uploaded_file and uploaded_file.name != st.session_state.sq_file_name:
    st.session_state.sq_result         = None
    st.session_state.sq_json_text      = None
    st.session_state.sq_rendered_pages = None
    st.session_state.sq_file_name      = uploaded_file.name

if uploaded_file:
    is_pdf     = uploaded_file.name.lower().endswith(".pdf")
    file_bytes = uploaded_file.getvalue()

    _sfx = (".pdf" if is_pdf
            else ".jpg" if uploaded_file.name.lower().endswith((".jpg", ".jpeg"))
            else ".png")
    _tmp = tempfile.NamedTemporaryFile(delete=False, suffix=_sfx)
    _tmp.write(file_bytes)
    _tmp.close()
    tmp_path = _tmp.name

    try:
        # ── Load pages ──
        if is_pdf:
            _doc = fitz.open(tmp_path)
            _n   = min(len(_doc), 4)
            _doc.close()
            rendered_pages   = {}
            pages_bytes_hi   = []
            for _pi in range(_n):
                rendered_pages[_pi] = pdf_page_to_png_bytes(tmp_path, _pi, dpi=100)
                pages_bytes_hi.append(pdf_page_to_png_bytes(tmp_path, _pi, dpi=350))
            skip_hashes      = compute_template_image_hashes(tmp_path)
            page_images_list = [
                extract_all_page_images(tmp_path, _pi, skip_hashes=skip_hashes)
                for _pi in range(_n)
            ]
        else:
            # Convert image to PNG for consistent handling
            try:
                _pix = fitz.Pixmap(file_bytes)
                if _pix.colorspace and _pix.colorspace.n != 3:
                    _pix = fitz.Pixmap(fitz.csRGB, _pix)
                if _pix.alpha:
                    _pix = fitz.Pixmap(_pix, 0)
                _png_bytes = _pix.tobytes("png")
            except Exception:
                _png_bytes = file_bytes
            rendered_pages    = {0: _png_bytes}
            pages_bytes_hi    = [_png_bytes]
            page_images_list  = [[]]
            skip_hashes       = set()
            _source_img_bytes = _png_bytes

        # Cache rendered pages once per file
        if st.session_state.sq_rendered_pages is None:
            st.session_state.sq_rendered_pages = rendered_pages

        n_pages = len(pages_bytes_hi)
        st.success(f"✅ Loaded **{n_pages} page(s)** from `{uploaded_file.name}`")

        # ── Extract button ──
        _xbtn_col, _ = st.columns([1, 3])
        with _xbtn_col:
            do_extract = st.button(
                "\U0001f680 Extract Question", type="primary", use_container_width=True
            )

        _prog_box = st.empty()

        if do_extract:
            if not api_key:
                st.error("Enter Gemini API key in the sidebar first!")
            else:
                def _update_prog(msg: str):
                    _prog_box.info(f"⏳ {msg}")

                with st.spinner("Extracting with Gemini AI — may take 10–60 s …"):
                    _result = extract_single_question(
                        pages_bytes        = pages_bytes_hi,
                        page_images_list   = page_images_list,
                        api_key            = api_key,
                        model_name         = model_choice,
                        source_image_bytes = (
                            locals().get("_source_img_bytes")
                            if not is_pdf else None
                        ),
                        is_hindi           = is_hindi,
                        user_subject       = user_subject,
                        user_course        = user_course,
                        user_class         = user_class,
                        user_chapter       = user_chapter,
                        user_practice      = user_practice,
                        user_book          = user_book,
                        progress_cb        = _update_prog,
                    )

                _prog_box.empty()

                if _result:
                    st.session_state.sq_result    = _result
                    st.session_state.sq_json_text = json.dumps(
                        _result, indent=2, ensure_ascii=False
                    )
                    st.success("✅ Extraction complete!")
                    st.rerun()
                else:
                    st.error(
                        "❌ Extraction failed after all retries. "
                        "Check API key / model, or try a clearer scan."
                    )

        # ──────────────────────────────────────────────────────
        # 3-COLUMN LAYOUT: Uploaded file | JSON editor | Preview
        # ──────────────────────────────────────────────────────
        st.markdown("---")
        col_pdf, col_editor, col_preview = st.columns([1, 1.3, 1])

        # ── Left: file / PDF preview ──
        with col_pdf:
            st.markdown("### \U0001f4c4 Uploaded File")
            _rp = st.session_state.sq_rendered_pages or rendered_pages
            with st.container(height=720):
                for _pi in sorted(_rp.keys()):
                    st.image(_rp[_pi], caption=f"Page {_pi + 1}", use_container_width=True)

        # ── Middle: JSON editor ──
        with col_editor:
            st.markdown("### ✏️ JSON Editor")
            _cur_json = st.session_state.sq_json_text

            if _cur_json:
                _edited = st.text_area(
                    "Edit extracted JSON:",
                    value=_cur_json,
                    height=660,
                    key="sq_json_editor",
                )
                # Live validation + sync back to session state
                if _edited != _cur_json:
                    try:
                        _parsed_edit = json.loads(_edited)
                        st.session_state.sq_result    = _parsed_edit
                        st.session_state.sq_json_text = _edited
                        st.caption("✅ Valid JSON — preview updated")
                    except json.JSONDecodeError as _je:
                        st.warning(f"⚠️ JSON syntax error: {_je}")

                _fname = os.path.splitext(uploaded_file.name)[0] + "_latex.json"
                st.download_button(
                    "⬇️ Download JSON",
                    data=st.session_state.sq_json_text.encode("utf-8"),
                    file_name=_fname,
                    mime="application/json; charset=utf-8",
                    use_container_width=True,
                )
            else:
                st.info("JSON will appear here after extraction.")
                st.text_area(
                    "JSON Editor (empty)",
                    value='{\n  "questionid": "",\n  "question": "",\n  "..": "..."\n}',
                    height=660,
                    disabled=True,
                    key="sq_json_editor_empty",
                )

        # ── Right: rendered preview ──
        with col_preview:
            st.markdown("### \U0001f52c Rendered Preview")
            _res = st.session_state.sq_result
            if _res:
                try:
                    _pq = (
                        _res
                        if isinstance(_res, dict)
                        else json.loads(st.session_state.sq_json_text)
                    )
                    with st.container(height=720):
                        components.html(
                            _build_preview_html([_pq]), height=700, scrolling=True
                        )
                except Exception as _pe:
                    st.error(f"Preview error: {_pe}")
            else:
                st.info("Preview will appear here after extraction.")

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

else:
    st.info("\U0001f446 Upload a PDF or image file to get started.")
    st.markdown("""
**How to use:**
1. Enter your Gemini API key in the sidebar
2. (Optional) Fill in Subject, Chapter, etc.
3. Upload a PDF or image — the file should contain **ONE question**
4. Click **\U0001f680 Extract Question**
5. Review / edit the LaTeX JSON in the middle panel
6. Download the JSON

**The question can:**
- Span 1–4 pages
- Have roman numeral sub-parts: (i), (ii), (iii), (iv)
- Have lettered sub-parts: (a), (b), (c), (d)
- Contain diagrams, tables, and LaTeX math
- Be in Hindi or English medium

**Supported types:** MCQs, True/False, Numeric, Subjective, Case Based (CBQ),
Assertion & Reasoning, Match the Column, Fill in the Blank
""")

# ════════════════════════════════════════════════════════════════════
# CUT IMAGE → LATEX  (standalone section, always visible)
# ════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("## ✂️ Cut Image → LaTeX")
st.caption("Upload any cropped diagram, equation, or table image — get its LaTeX / text instantly.")

_ci_col_img, _ci_col_latex = st.columns([1, 1], gap="large")

with _ci_col_img:
    st.markdown("#### 🖼 Upload Cut Image")
    _ci_file = st.file_uploader(
        "Drop a JPG / PNG here",
        type=["jpg", "jpeg", "png"],
        key="cut_image_uploader",
    )
    if _ci_file:
        _ci_bytes = _ci_file.getvalue()
        st.image(_ci_bytes, use_container_width=True)

with _ci_col_latex:
    st.markdown("#### 📋 Extracted LaTeX")

    # Persist result across reruns
    if "ci_result" not in st.session_state:
        st.session_state.ci_result = ""
    if "ci_file_name" not in st.session_state:
        st.session_state.ci_file_name = ""

    # Reset when a new file is uploaded
    if _ci_file and _ci_file.name != st.session_state.ci_file_name:
        st.session_state.ci_result    = ""
        st.session_state.ci_file_name = _ci_file.name

    if not api_key:
        st.warning("⚠️ Enter your Gemini API key in the sidebar first.")

    _ci_extract_btn = st.button(
        "⚡ Extract LaTeX", type="primary",
        disabled=(_ci_file is None or not api_key),
        key="cut_image_extract_btn",
    )

    if _ci_extract_btn and _ci_file and api_key:
        _ci_prog = st.empty()
        with st.spinner("Extracting with Gemini AI…"):
            try:
                _ci_client = genai.Client(api_key=api_key)

                # Convert to PNG once
                try:
                    _ci_pix = fitz.Pixmap(_ci_bytes)
                    if _ci_pix.colorspace and _ci_pix.colorspace.n != 3:
                        _ci_pix = fitz.Pixmap(fitz.csRGB, _ci_pix)
                    if _ci_pix.alpha:
                        _ci_pix = fitz.Pixmap(_ci_pix, 0)
                    _ci_png = _ci_pix.tobytes("png")
                except Exception:
                    _ci_png = _ci_bytes

                _ci_prompt = (
                    "You are a precise LaTeX extractor for educational content.\n"
                    "The image is a CROPPED portion of a textbook or exam paper.\n\n"
                    "RULES — follow every one exactly:\n"
                    "1. Copy ALL text verbatim, character by character (Hindi/English as-is).\n"
                    "2. Convert EVERY mathematical expression to LaTeX:\n"
                    "   • Inline  → \\\\( ... \\\\)\n"
                    "   • Display → \\\\[ ... \\\\]\n"
                    "3. Geometric diagram → write one short description line, then list every\n"
                    "   labelled vertex, side, angle, and measurement in LaTeX on separate lines.\n"
                    "   Example:  Right triangle ABC with ∠B = 90°\n"
                    "             \\\\( AB = 20\\\\text{ m},\\quad BC = h,\\quad \\\\angle A = 30° \\\\)\n"
                    "4. Table → output as HTML <table> with borders.\n"
                    "5. Do NOT add any JSON, markdown fences, headings, or commentary.\n"
                    "6. Do NOT skip or summarise anything — 100% complete extraction.\n\n"
                    "Return ONLY the extracted content."
                )

                _ci_contents = [
                    _ci_prompt,
                    genai_types.Part.from_bytes(data=_ci_png, mime_type="image/png"),
                ]

                # Try every config in order (same as main extractor) until we get a result
                _ci_configs = _make_page_configs()
                # Add high-budget thinking for complex diagrams
                for _bud in [8192, 24576]:
                    try:
                        _ci_configs.append(genai_types.GenerateContentConfig(
                            temperature=0, max_output_tokens=8192,
                            thinking_config=genai_types.ThinkingConfig(thinking_budget=_bud),
                        ))
                    except Exception:
                        pass

                _ci_result_text = ""
                for _ci_attempt, _ci_cfg in enumerate(_ci_configs, 1):
                    _ci_prog.info(f"⏳ Attempt {_ci_attempt}/{len(_ci_configs)}…")
                    try:
                        _ci_resp = _ci_client.models.generate_content(
                            model=model_choice,
                            contents=_ci_contents,
                            config=_ci_cfg,
                        )
                        _ci_txt = _safe_response_text(_ci_resp)
                        if _ci_txt and _ci_txt.strip():
                            _ci_result_text = _ci_txt.strip()
                            break
                    except Exception as _ci_ae:
                        _err = str(_ci_ae)
                        if "429" in _err or "RESOURCE_EXHAUSTED" in _err:
                            time.sleep(30)
                        elif "500" in _err or "503" in _err:
                            time.sleep(5)

                _ci_prog.empty()
                if _ci_result_text:
                    st.session_state.ci_result = _ci_result_text
                else:
                    st.error("All attempts returned empty. Try a different model or image.")
            except Exception as _ci_e:
                _ci_prog.empty()
                st.error(f"Error: {_ci_e}")

    if st.session_state.ci_result:
        # st.code has a built-in copy button in the top-right corner
        st.code(st.session_state.ci_result, language="latex")
    elif _ci_file and api_key:
        st.info("Click **⚡ Extract LaTeX** to extract.")
