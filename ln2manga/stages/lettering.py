"""Stage 8 — lettering: draw speech/thought/shout/narration bubbles + dialogue with Pillow.

For each panel rect we fetch the PanelSpec's dialogue, fit the text (binary-ish font search +
word wrap), size a bubble around it, place bubbles top-down without overlapping, and point a
tail toward the speaker side. Pure Pillow.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .. import bubbles
from ..artifacts import PageLayout, PanelSpec, Rect
from ..config import Settings

PAD = 18
GAP = 14

# Sentence terminators (kept with the sentence) and clause separators (cut after).
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:—–-])\s+|\s+[—–]\s+")

# Shot-aware vertical budget for a single bubble, as a fraction of panel height.
# Tighter shots (close-ups) get a smaller bubble so it can tuck into an empty band
# instead of sprawling across the (centered, dominant) face.
_SHOT_MAX_H_FRAC = {
    "establishing": 0.45,
    "wide": 0.45,
    "medium": 0.40,
    "close_up": 0.30,
    "extreme_close_up": 0.24,
    "insert": 0.40,
}
# Dark-ink fraction above which a band is considered "full of face/detail".
_DENSE_BAND = 0.18
# SMALL-FACE avoidance: a footprint that buries a SMALL, concentrated dark cluster (a likely
# chibi / secondary / background face) is penalized on top of its mean-ink score, so it is
# avoided even when the surrounding emptiness averages the small face away. The penalty is
# added to the candidate's mean-ink score, scaled by the fraction of small-cluster cells the
# footprint covers; this value is large relative to the mean-ink tie tolerance so covering a
# small face reliably outweighs a tiny mean-ink advantage, while never overriding a genuinely
# clear footprint (which covers no cluster and so gets zero penalty).
_SMALL_FACE_PENALTY = 0.30
# A "small face" cluster is detected as dark ink concentrated in a window roughly this fraction
# of the panel's MIN dimension, where the same dark is NOT also present across a much wider
# surrounding window — i.e. an isolated small blob, not part of a large filled region (a
# dominant close-up face fills the panel and is handled by the full-face fallback instead).
# Tuned so chibi/secondary/background faces (~40-120px on a ~750px panel) are flagged while a
# large dominant face's interior (~200px+) mostly is not (its wide surround is just as dark).
_SMALL_FACE_WIN_FRAC = 0.10
# Local dark-ink fraction (within the small window) above which the window is "dark enough" to
# be a face/detail cluster, and the ratio by which the small window must be denser than its
# wider surround for the dark to count as CONCENTRATED (isolated small blob) rather than a
# corner of a large dense region.
_SMALL_FACE_LOCAL_INK = 0.20
_SMALL_FACE_CONCENTRATION = 1.6
# A close-up is one of these shot types (centered face dominates the panel).
_CLOSE_SHOTS = ("close_up", "extreme_close_up")
# Readable floor for the dense-panel crowd-trim: a crowded line is shortened to fit, never
# gutted below roughly this many characters (a short clause).
_ABRIDGE_FLOOR_CHARS = 24
# Organic bubble SHAPES have a smaller usable interior than their bounding box (thick ink
# border + non-rectangular silhouette), so the target is sized a bit larger than the plain
# text box; the text is then abridged to the ACTUAL detected interior, guaranteeing it fits
# inside the bubble without a clip — and, being roomier, with less truncation.
# Kept modest (was 1.3): a big inflate made ≤3 bubbles too tall to stack in shorter panels,
# which forced the geometric stacking-skip to drop later bubbles. The bottom-margin reserve
# below (not a fatter bubble) is what now guarantees the text clears the outline, so the
# inflate only needs to cover the interior-vs-bounding-box gap, not the clip safety.
_ORGANIC_INFLATE = 1.12
# Fraction of an organic bubble's detected interior HEIGHT reserved at the BOTTOM as a safety
# margin before fitting/centering text. place_bubble reports the silhouette's interior height,
# but the TRUE usable height is shorter near the rounded bottom + tail notch, so the last
# wrapped line's baseline/descenders can land on or below the outline. We shrink the usable
# interior by this fraction (off the bottom) and center the text in the REDUCED region so the
# drawn block always ends strictly above (interior.y1 - margin). Speech has a bottom tail and
# needs more reservation than narration/box (flat-bottomed rectangle, no tail).
_ORGANIC_BOTTOM_MARGIN_FRAC = 0.10
_ORGANIC_BOTTOM_MARGIN_FRAC_SPEECH = 0.18


# ── surfacing dropped dialogue ───────────────────────────────────────────────
def _record_dropped(dropped: "list | None", dlines) -> None:
    """Append the raw text of each non-empty DialogueLine in `dlines` to `dropped`
    (a no-op when `dropped` is None). Used to surface spoken lines that did not fit
    their panel out-of-band, without drawing any on-page marker."""
    if dropped is None:
        return
    for d in dlines:
        t = (getattr(d, "text", "") or "").strip()
        if t:
            dropped.append(t)


# Drop priority: when a panel genuinely cannot hold every line, drop the LEAST important
# first. A panel's narration/caption (scene/context) and a `shout` (a beat the reader must
# not miss) outrank ordinary `speech`; lower number = keep first.
_DROP_PRIORITY = {"narration": 0, "caption": 0, "shout": 1}


def _drop_priority(style: str | None) -> int:
    """Keep-priority for a dialogue style (lower = more important, kept first)."""
    return _DROP_PRIORITY.get(style or "", 2)


# ── abridging over-long dialogue ─────────────────────────────────────────────
def _abridge(text: str, max_chars: int) -> str:
    """Shorten `text` to <= `max_chars` characters at the most coherent boundary.

    Tries, in order of decreasing coherence:
      1. SENTENCE boundaries — keep the first one or two whole sentences.
      2. CLAUSE boundaries — keep whole clauses (split on , ; : — –).
      3. WHOLE WORDS — drop trailing words; never split mid-word.
    Whitespace is collapsed. A single "…" is appended ONLY when the text had to
    be cut mid-thought (i.e. the kept text is shorter than the collapsed input and
    does not already end on a sentence terminator). Pure: no I/O, no fonts.
    """
    text = " ".join((text or "").split())  # collapse all whitespace runs
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text

    def _ends_sentence(s: str) -> bool:
        return bool(s) and s[-1] in ".!?…"

    # 1) sentence boundaries: greedily keep the first whole sentences that fit,
    #    preferring one or two sentences over a single long run.
    sentences = [s for s in _SENT_SPLIT.split(text) if s]
    if len(sentences) > 1:
        kept = ""
        for sent in sentences:
            trial = (kept + " " + sent).strip()
            if len(trial) <= max_chars:
                kept = trial
            else:
                break
        if kept:
            # whole sentence(s) kept -> already coherent, no ellipsis needed.
            return kept

    # 2) clause boundaries: keep whole clauses up to the budget.
    clauses = [c for c in _CLAUSE_SPLIT.split(text) if c and c.strip()]
    if len(clauses) > 1:
        kept = ""
        for clause in clauses:
            clause = clause.strip()
            trial = (kept + " " + clause).strip() if kept else clause
            if len(trial) <= max_chars - 1:  # leave room for a trailing "…"
                kept = trial
            else:
                break
        if kept and len(kept) < len(text):
            kept = kept.rstrip(" ,;:—–-")
            return kept if _ends_sentence(kept) else kept + "…"

    # 3) whole words: drop trailing words until it fits, never splitting a word.
    words = text.split()
    kept = ""
    for w in words:
        trial = (kept + " " + w).strip() if kept else w
        if len(trial) <= max_chars - 1:  # leave room for a trailing "…"
            kept = trial
        else:
            break
    if not kept:  # even the first word exceeds the budget; hand back that lone word
        kept = words[0] if words else ""
        return kept[:max_chars] if len(kept) > max_chars else kept
    kept = kept.rstrip(" ,;:—–-")
    return kept if _ends_sentence(kept) else kept + "…"


# ── fonts ──────────────────────────────────────────────────────────────────
def _font_path(settings: Settings, kind: str) -> str | None:
    names = [settings.lettering.get(kind, ""), settings.lettering.get("fallback_font", "")]
    for n in names:
        if n and (settings.fonts_dir / n).exists():
            return str(settings.fonts_dir / n)
    ttfs = sorted(settings.fonts_dir.glob("*.ttf"))
    return str(ttfs[0]) if ttfs else None


def _font(path: str | None, size: int) -> ImageFont.FreeTypeFont:
    if path:
        return ImageFont.truetype(path, size)
    # Pillow 10+ honors a size argument here; without it load_default() returns a
    # fixed ~13px bitmap that collapses every font size to one tiny, unreadable size
    # (the _fit_text shrink loop and _char_budget all become no-ops).
    return ImageFont.load_default(size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: float) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        # hard-break a token that cannot fit on a line by itself, so a single
        # long word (URL, katakana run, hyphen-less compound) never overflows.
        if draw.textlength(word, font=font) > max_w:
            if cur:
                lines.append(cur)
                cur = ""
            chunk = ""
            for ch in word:
                if draw.textlength(chunk + ch, font=font) <= max_w or not chunk:
                    chunk += ch
                else:
                    lines.append(chunk)
                    chunk = ch
            cur = chunk
            continue
        trial = (cur + " " + word).strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _char_budget(draw, font_path, max_w, max_h, min_font) -> int:
    """Estimate how many characters readably fit a bubble at the MINIMUM font.

    Used to decide when a line is genuinely too long: chars-per-line (max_w at the
    smallest allowed font) × the number of lines that fit max_h. Uses the min font
    so we only abridge when text would overflow even at the smallest readable size.
    Returns a generous-but-finite budget (>= 1).
    """
    font = _font(font_path, int(min_font))
    line_h = int(min_font) + 5
    max_lines = max(1, int(max_h // line_h))
    # average glyph advance for a representative sample (avoids per-char measuring).
    sample = "abcdefghijklmnopqrstuvwxyz etaoinshrdlu"
    avg_w = draw.textlength(sample, font=font) / max(1, len(sample))
    if avg_w <= 0:
        return max(1, int(max_w))  # degenerate font metrics; fall back to width-ish
    chars_per_line = max(1, int(max_w / avg_w))
    return max(1, chars_per_line * max_lines)


def _fit_text(draw, text, font_path, max_w, max_h, max_font, min_font):
    for size in range(int(max_font), int(min_font) - 1, -2):
        font = _font(font_path, size)
        lines = _wrap(draw, text, font, max_w)
        line_h = size + 5
        th = line_h * len(lines)
        tw = max((draw.textlength(l, font=font) for l in lines), default=0)
        if th <= max_h and tw <= max_w:
            return lines, font, tw, th, line_h
    font = _font(font_path, int(min_font))
    lines = _wrap(draw, text, font, max_w)
    line_h = int(min_font) + 5
    tw = max((draw.textlength(l, font=font) for l in lines), default=0)
    return lines, font, tw, line_h * len(lines), line_h


# ── bubble shapes ──────────────────────────────────────────────────────────
def _draw_text_block(draw, lines, font, cx, top, line_h):
    y = top
    for ln in lines:
        w = draw.textlength(ln, font=font)
        draw.text((cx - w / 2, y), ln, fill=0, font=font)
        y += line_h


def _ellipse(draw, cx, cy, rx, ry):
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255, outline=0, width=3)


def _tail(draw, cx, cy, rx, ry, target):
    """Triangle tail from the bubble edge toward `target` (x, y)."""
    tx, ty = target
    ang = math.atan2(ty - cy, tx - cx)
    base = 16
    bx, by = cx + rx * 0.7 * math.cos(ang), cy + ry * 0.7 * math.sin(ang)
    perp = ang + math.pi / 2
    p1 = (bx + base * math.cos(perp), by + base * math.sin(perp))
    p2 = (bx - base * math.cos(perp), by - base * math.sin(perp))
    draw.polygon([p1, p2, target], fill=255, outline=0)
    # erase the seam (base chord p1->p2) where the tail meets the bubble, so the
    # mouth stays open instead of a black chord cutting across the bubble interior.
    draw.line([p1, p2], fill=255, width=3)


def _bubble_size(tw, th, style):
    """Bubble semi-axes. Ellipse/star bubbles must CONTAIN the rectangular text block: an ellipse
    through the rect's corners needs ~sqrt(2)-expanded axes, else the text overflows the bubble.
    Rectangular narration boxes only need a small pad."""
    if style in ("narration", "speech"):   # rounded rectangle: holds the text rect with small pad
        return tw / 2 + PAD, th / 2 + PAD
    return tw / 2 * 1.42 + 8, th / 2 * 1.42 + 8   # shout/thought ellipse-ish need sqrt2 to contain text


def _draw_bubble(draw, cx, cy, tw, th, style, tail_target):
    rx, ry = _bubble_size(tw, th, style)
    if style == "narration":
        draw.rectangle([cx - rx, cy - ry, cx + rx, cy + ry], fill=255, outline=0, width=3)
        return
    if style == "shout":
        pts = []
        n = 16
        for i in range(n):
            a = 2 * math.pi * i / n
            r = (rx, ry) if i % 2 == 0 else (rx * 0.78, ry * 0.78)
            pts.append((cx + r[0] * math.cos(a), cy + r[1] * math.sin(a)))
        draw.polygon(pts, fill=255, outline=0)
        if tail_target:
            _tail(draw, cx, cy, rx, ry, tail_target)
        return
    # speech -> rounded rectangle (more text space than an oval); thought -> cloud-ish ellipse
    if style == "speech":
        draw.rounded_rectangle([cx - rx, cy - ry, cx + rx, cy + ry],
                               radius=max(8, min(rx, ry) * 0.5), fill=255, outline=0, width=3)
        if tail_target:
            _tail(draw, cx, cy, rx, ry, tail_target)
        return
    _ellipse(draw, cx, cy, rx, ry)
    if style == "thought" and tail_target:
        tx, ty = tail_target
        for i, frac in enumerate((0.45, 0.7, 0.9)):
            bx = cx + (tx - cx) * frac
            by = cy + (ty - cy) * frac
            r = 10 - i * 3
            draw.ellipse([bx - r, by - r, bx + r, by + r], fill=255, outline=0, width=2)
    elif tail_target:
        _tail(draw, cx, cy, rx, ry, tail_target)


# ── content-aware placement ──────────────────────────────────────────────────
def _shot_max_h_frac(shot_type: str | None) -> float:
    return _SHOT_MAX_H_FRAC.get(shot_type or "", 0.40)


def _emptiest_band(page_gray: "Image.Image", rect: Rect, bubble_h: float,
                   floor_frac: float = 0.0):
    """Return (top_y, density) of the emptiest horizontal band of height ~bubble_h
    inside `rect`, scanning the panel pixels of `page_gray` (an "L" image).

    Faces and detail are dense dark ink, so the emptiest band avoids them. The band's
    top is searched between rect.y0+PAD and rect.y1-bubble_h-PAD. `floor_frac` lets the
    caller restrict the search to the lower part of the panel (e.g. full-face fallback).
    Returns (None, 1.0) if the panel/band cannot be measured.
    """
    bubble_h = max(1, int(bubble_h))
    # crop the panel region, clamped to the image bounds.
    px0 = max(0, int(rect.x0))
    py0 = max(0, int(rect.y0))
    px1 = min(page_gray.width, int(rect.x1))
    py1 = min(page_gray.height, int(rect.y1))
    if px1 - px0 < 4 or py1 - py0 < 4:
        return None, 1.0
    crop = page_gray.crop((px0, py0, px1, py1))
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is a hard dep of the pipeline
        return None, 1.0
    arr = np.asarray(crop, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return None, 1.0
    small = arr[::4, ::4]
    if small.size == 0:
        return None, 1.0
    # dark-ink mask: ink is dark on a light page; treat < 128 as ink.
    ink = small < 128
    sh = small.shape[0]
    # row density profile (fraction of dark pixels per downsampled row).
    row_ink = ink.mean(axis=1)  # length sh
    scale_y = (py1 - py0) / sh  # page-px per downsampled row

    band_rows = max(1, int(round(bubble_h / scale_y)))
    band_rows = min(band_rows, sh)
    # cumulative sum for O(1) band-density queries.
    cum = np.concatenate([[0.0], np.cumsum(row_ink)])

    # restrict the search start to honor floor_frac (lower band of the panel).
    lo_row = int(round(floor_frac * (sh - band_rows))) if sh > band_rows else 0
    lo_row = max(0, min(lo_row, max(0, sh - band_rows)))
    hi_row = sh - band_rows  # inclusive

    best_row, best_density = lo_row, None
    for r in range(lo_row, hi_row + 1):
        density = (cum[r + band_rows] - cum[r]) / band_rows
        if best_density is None or density < best_density:
            best_density, best_row = density, r
    if best_density is None:
        return None, 1.0
    # convert downsampled row back to a page-space top-y, then offset into the band a touch.
    top_y = py0 + int(round(best_row * scale_y))
    # keep within the panel's usable vertical range.
    top_y = max(int(rect.y0) + PAD, min(top_y, int(rect.y1) - bubble_h - PAD))
    return top_y, best_density


def _best_footprint(page_gray: "Image.Image", rect: Rect, bubble_w: float,
                    bubble_h: float, prefer_side: str | None = None,
                    floor_frac: float = 0.0):
    """Return (cx, top, ink_frac) for the lowest-ink placement of a bubble whose
    bounding box is bubble_w x bubble_h, searched over a small grid of (x, y) slots
    inside `rect`. Pure Pillow/numpy on the un-lettered panel pixels.

    Where `_emptiest_band` averages dark ink across the WHOLE panel width per row —
    so a SMALL off-center face (chibi / secondary / background character) sits in an
    otherwise-empty row band and gets averaged away — this scores the ink under each
    candidate's ACTUAL FOOTPRINT. A footprint can therefore dodge a small concentrated
    dark cluster sideways, not just vertically, so the bubble avoids small faces even
    when their row-band looks empty on average.

    Scoring favours the least-ink footprint; among near-ties (within a small ink
    tolerance) it biases toward a panel CORNER/EDGE (where a face is least likely
    centered) and, secondarily, toward `prefer_side` ("left"/"right", the speaker
    side's OPPOSITE corner already chosen by the caller). Returns (None, None, 1.0)
    if the panel cannot be measured.
    """
    bubble_w = max(1, int(bubble_w))
    bubble_h = max(1, int(bubble_h))
    px0 = max(0, int(rect.x0))
    py0 = max(0, int(rect.y0))
    px1 = min(page_gray.width, int(rect.x1))
    py1 = min(page_gray.height, int(rect.y1))
    if px1 - px0 < 4 or py1 - py0 < 4:
        return None, None, 1.0
    crop = page_gray.crop((px0, py0, px1, py1))
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is a hard dep of the pipeline
        return None, None, 1.0
    arr = np.asarray(crop, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return None, None, 1.0
    small = arr[::4, ::4]
    if small.size == 0:
        return None, None, 1.0
    ink = (small < 128).astype(np.float32)
    sh, sw = small.shape
    scale_y = (py1 - py0) / sh
    scale_x = (px1 - px0) / sw

    # footprint size in downsampled cells (>=1, clamped to the map).
    fw = max(1, min(sw, int(round(bubble_w / scale_x))))
    fh = max(1, min(sh, int(round(bubble_h / scale_y))))

    # 2-D summed-area table for O(1) rectangle-sum queries over any footprint.
    sat = np.zeros((sh + 1, sw + 1), dtype=np.float32)
    sat[1:, 1:] = ink.cumsum(axis=0).cumsum(axis=1)

    def ink_at(r, c):  # mean ink over the fw x fh footprint anchored at (r, c)
        total = (sat[r + fh, c + fw] - sat[r, c + fw]
                 - sat[r + fh, c] + sat[r, c])
        return float(total) / (fw * fh)

    # ── SMALL-FACE cluster mask ───────────────────────────────────────────────
    # Mean-ink minimization alone dilutes a SMALL concentrated dark blob (chibi /
    # secondary / background face): a big bubble footprint covering it averages the
    # blob away against the surrounding emptiness, so two footprints — one burying a
    # small face, one over blank art — can tie on mean ink. Detect such blobs and
    # build a 0/1 cluster mask so a footprint covering one is penalized below.
    #
    # A cluster cell is dark ink that is CONCENTRATED: its local density over a small
    # (~small-face-sized) window is high, AND that local density is markedly greater
    # than the density over a much wider surrounding window. The wider-window test is
    # what isolates a SMALL blob from a corner of a LARGE filled region (a dominant
    # close-up face that fills the panel — already handled by the full-face fallback),
    # so this only steers the bubble off genuinely small/secondary faces.
    win = max(1, int(round(_SMALL_FACE_WIN_FRAC * min(sh, sw))))
    win = min(win, sh, sw)
    surround = min(min(sh, sw), max(win + 1, int(round(win * 2.5))))

    def _window_mean(side):
        # mean ink over a `side`x`side` window centered on each cell, via the ink SAT,
        # returned as an (sh, sw) array (edge windows are clamped to the map bounds).
        half = side // 2
        rows_i = np.arange(sh)
        cols_i = np.arange(sw)
        r0 = np.clip(rows_i - half, 0, sh)
        r1 = np.clip(rows_i - half + side, 0, sh)
        c0 = np.clip(cols_i - half, 0, sw)
        c1 = np.clip(cols_i - half + side, 0, sw)
        # broadcast row/col bounds into 2-D corner-index grids for the SAT lookup.
        R0, C0 = np.meshgrid(r0, c0, indexing="ij")
        R1, C1 = np.meshgrid(r1, c1, indexing="ij")
        area = np.maximum(1, (R1 - R0) * (C1 - C0)).astype(np.float32)
        total = sat[R1, C1] - sat[R0, C1] - sat[R1, C0] + sat[R0, C0]
        return total / area

    local = _window_mean(win)
    wide = _window_mean(surround)
    # CONCENTRATED dark: locally dense and noticeably denser than the wider surround.
    cluster = ((local >= _SMALL_FACE_LOCAL_INK)
               & (local >= _SMALL_FACE_CONCENTRATION * wide)
               & (ink > 0)).astype(np.float32)
    csat = np.zeros((sh + 1, sw + 1), dtype=np.float32)
    csat[1:, 1:] = cluster.cumsum(axis=0).cumsum(axis=1)

    def cluster_at(r, c):  # fraction of the footprint that overlaps small-cluster cells
        total = (csat[r + fh, c + fw] - csat[r, c + fw]
                 - csat[r + fh, c] + csat[r, c])
        return float(total) / (fw * fh)

    # usable top-left anchor ranges (the footprint must stay inside the panel).
    r_hi = sh - fh
    c_hi = sw - fw
    if r_hi < 0 or c_hi < 0:
        return None, None, 1.0
    # restrict the vertical search to honor floor_frac (lower band of the panel), used
    # by the full-face bottom-anchor fallback to keep the bubble over the chin/body.
    r_lo = int(round(floor_frac * r_hi)) if r_hi > 0 else 0
    r_lo = max(0, min(r_lo, r_hi))

    # scan a modest grid of slots (cheap: ~ a few dozen footprint queries). A step of
    # ~1/3 of the footprint gives overlapping windows so a small cluster can't slip
    # between slots, while keeping the candidate count tiny.
    r_step = max(1, fh // 3)
    c_step = max(1, fw // 3)
    rows = list(range(r_lo, r_hi + 1, r_step))
    cols = list(range(0, c_hi + 1, c_step))
    if rows[-1] != r_hi:
        rows.append(r_hi)
    if cols[-1] != c_hi:
        cols.append(c_hi)

    # each candidate carries its raw mean ink, its small-face penalty (cluster coverage
    # scaled), and the combined SCORE that placement minimizes. Scoring by the combined
    # score — not raw mean ink — means a footprint burying a small concentrated face is
    # ranked worse than an equally-(or slightly-more-)inky footprint that misses it, so
    # the bubble dodges small/secondary faces the plain mean-ink scan averaged away.
    cands = []  # (score, raw_ink, r, c)
    for r in rows:
        for c in cols:
            raw = ink_at(r, c)
            score = raw + _SMALL_FACE_PENALTY * cluster_at(r, c)
            cands.append((score, raw, r, c))
    if not cands:
        return None, None, 1.0

    min_score = min(s for s, _ink, _r, _c in cands)
    # tie tolerance: footprints within this much SCORE of the best are "as good"; among
    # them we choose by corner/edge bias so a tie doesn't default to the dead center.
    tol = 0.02

    def corner_cost(r, c):
        # cost favouring an UPPER edge/corner of the panel (smaller = preferred).
        cr = (r + fh / 2) / max(1, sh)   # normalized center row (0=top, 1=bottom)
        cc = (c + fw / 2) / max(1, sw)   # normalized center col (0=left, 1=right)
        # Vertical: prefer the TOP — keeps the first bubble high so later bubbles can
        # stack BELOW it (no-overlap), matching the original top-corner heuristic.
        # (A near-empty footprint anywhere already dodges the face; this only breaks
        # ties among equally-empty slots, so it won't pull the bubble onto a face.)
        dr = cr
        # Horizontal: prefer either LEFT or RIGHT edge over the center (faces tend to
        # be horizontally centered), so pick the nearer side.
        dc = min(cc, 1.0 - cc)
        cost = dr + dc
        # nudge toward the preferred horizontal side when given (the caller passes the
        # corner OPPOSITE the speaker), without overriding the corner pull.
        if prefer_side == "left":
            cost += 0.15 * cc
        elif prefer_side == "right":
            cost += 0.15 * (1.0 - cc)
        return cost

    near = [(corner_cost(r, c), score, raw, r, c)
            for score, raw, r, c in cands if score <= min_score + tol]
    # corner bias first, then the combined score (small-face penalty included) as the
    # tiebreak, so among corner-equal slots the one over the least ink/face still wins.
    near.sort(key=lambda t: (t[0], t[1]))
    _cc, _best_score, best_ink, best_r, best_c = near[0]

    top_y = py0 + int(round(best_r * scale_y))
    cx = px0 + (best_c + fw / 2) * scale_x
    # clamp to the panel's usable range (mirror the caller's own clamps).
    top_y = max(int(rect.y0) + PAD, min(top_y, int(rect.y1) - bubble_h - PAD))
    cx = min(max(cx, rect.x0 + bubble_w / 2 + 4), rect.x1 - bubble_w / 2 - 4)
    return cx, top_y, best_ink


# ── per-panel + page ───────────────────────────────────────────────────────
def letter_panel(draw, rect: Rect, spec: PanelSpec, settings: Settings,
                 page_img: "Image.Image | None" = None, page_live: "Image.Image | None" = None,
                 shape_lib: "dict | None" = None, dropped: "list | None" = None) -> None:
    """Letter one panel. If `dropped` is given, every non-empty dialogue line that
    is NOT drawn (beyond the per-panel cap, or abandoned because it overflows the
    panel bottom) is appended to it as its raw text, so the caller can surface the
    loss out-of-band (no on-page marker is drawn)."""
    if not spec or not spec.dialogue:
        return
    # per-panel cap on how many bubbles a single panel may carry (configurable so a
    # dense panel can be tuned without code changes). Lines beyond the cap are dropped.
    cap = max(1, int(settings.lettering.get("max_bubbles_per_panel", 4)))
    capped = spec.dialogue[:cap]
    _record_dropped(dropped, spec.dialogue[cap:])  # lines beyond the per-panel cap
    max_w = rect.w * float(settings.lettering.get("line_width_frac", 0.62))
    # also cap the wrap width so the *expanded* ellipse bubble (text*1.42 + pad) still
    # fits the panel with a small gutter — on a narrow panel the 0.62 frac alone lets a
    # full line produce a bubble wider than the panel (bleeds into the gutter).
    _ellipse_w_cap = ((rect.w - 2 * (PAD + 4)) - 16) / 1.42
    if _ellipse_w_cap > 8:
        max_w = min(max_w, _ellipse_w_cap)
    shot = getattr(spec, "shot_type", None)
    # split the panel's vertical bubble budget across however many SPOKEN bubbles it has, so
    # multiple bubbles in one panel each get a fair slice and don't crowd/overlap. Narration
    # captions are excluded from this count: a short caption must not halve the speech bubble's
    # budget nor disable the single-speech-bubble face-avoidance band-anchoring below.
    n_lines = max(1, sum(1 for d in capped
                         if (d.text or "").strip() and d.style != "narration"))
    max_h = rect.h * _shot_max_h_frac(shot)
    if n_lines > 1:
        max_h = max(rect.h * 0.16, max_h / n_lines)
    is_close = shot in _CLOSE_SHOTS
    # prepare a grayscale page for content analysis (only if a page image was given).
    page_gray = None
    if page_img is not None:
        page_gray = page_img if page_img.mode == "L" else page_img.convert("L")
    side = spec.speaker_side

    # Indices (into `capped`) of the non-empty lines, in reading order.
    line_idxs = [i for i, d in enumerate(capped) if (d.text or "").strip()]

    def _place(order: "list[int]", draw_enabled: bool) -> list[int]:
        """Run the top-down stacking placement over `order` (indices into `capped`,
        in the on-page reading order they should appear). Returns the indices that
        were actually placed (drew, or — in a measure pass — would have drawn).
        A later bubble that cannot fit below its predecessor stops the run (the
        no-overlap invariant): every index from that point on is left unplaced.
        Drawing side effects (compositing/Pillow) only happen when draw_enabled."""
        y_cursor = rect.y0 + PAD
        first_placed = False  # whether the empty-band anchor has been applied yet
        placed: list[int] = []
        for di in order:
            dline = capped[di]
            text = (dline.text or "").strip()
            if not text:
                continue
            kind = "shout_font" if dline.style == "shout" else "dialogue_font"
            fpath = _font_path(settings, kind)
            # ABRIDGE first: if the line is too long to fit the bubble even at the
            # minimum font, shorten it to a coherent sentence/clause/word boundary so
            # the bubble shows clean, readable text instead of a mid-word hard clip.
            budget = _char_budget(draw, fpath, max_w, max_h, settings.lettering["min_font"])
            # DENSE panels abridge HARDER, but GENTLY: only genuinely crowded panels (>=4 spoken
            # bubbles) shrink each line's char budget by dense_abridge_frac, and never below a
            # readable floor (~24 chars) — so a crowded line is shortened to fit, not gutted.
            # Panels with <=3 spoken lines keep their full budget (the common case after the parse
            # caps ~3 lines/panel), so most dialogue is untouched.
            if n_lines >= 4:
                frac = float(settings.lettering.get("dense_abridge_frac", 1.0))
                budget = max(int(budget * frac), min(budget, _ABRIDGE_FLOOR_CHARS))
            text = _abridge(text, budget)
            lines, font, tw, th, line_h = _fit_text(
                draw, text, fpath, max_w, max_h,
                settings.lettering["max_font"], settings.lettering["min_font"])

            # Safety net: if wrapping still produced more lines than fit the panel's
            # vertical budget (e.g. an unbreakable token, or a residual long word),
            # truncate so an over-long speech can never spill past the bubble.
            max_lines = max(1, int(max_h // line_h))
            if len(lines) > max_lines:
                lines = lines[:max_lines]
                lines[-1] = (lines[-1][:-1] if len(lines[-1]) > 1 else lines[-1]) + "…"
                th = line_h * len(lines)
                tw = max((draw.textlength(l, font=font) for l in lines), default=tw)
            # keep the (sqrt2-expanded) bubble within the panel; size the bubble to CONTAIN the text.
            tw = min(tw, rect.w * 0.66)
            rx, ry = _bubble_size(tw, th, dline.style)
            bubble_w = 2 * rx
            bubble_h = 2 * ry
            # When an organic SHAPE will be used for this style, inflate the target so the shape's
            # smaller interior still holds the text (done BEFORE placement so the no-overlap stacking
            # accounts for the real size). Capped to the panel so it can't bleed into the gutter.
            if page_live is not None and (shape_lib or {}).get(dline.style):
                bubble_w = min(int(bubble_w * _ORGANIC_INFLATE), int(rect.w - 8))
                bubble_h = min(int(bubble_h * _ORGANIC_INFLATE), int(rect.h - 8))

            # ── content-aware vertical placement ──────────────────────────────
            # Only the FIRST bubble is anchored to an empty band; later bubbles stack
            # downward from the running cursor so they never overlap the first.
            top = y_cursor
            bottom_anchored = False
            # footprint_cx, when set by the FIRST-bubble footprint scorer, overrides the
            # speaker-side horizontal default below (the scorer already dodged small faces
            # sideways). Left None for later bubbles / no-pixels so the default applies.
            footprint_cx = None
            # The FIRST bubble is anchored to the emptiest band (face-avoidance), even when the
            # panel also carries a narration caption (which would otherwise stack the speech bubble
            # from the top straight onto a centered close-up face). Later bubbles stack strictly
            # top-down from y_cursor (below the anchored first) so bubbles can NEVER overlap.
            if page_gray is not None and not first_placed:
                # FOOTPRINT-scored placement: score candidate slots by the dark ink under the
                # bubble's ACTUAL footprint and pick the emptiest, so a SMALL off-center face
                # (chibi / secondary / background) is avoided sideways too — not just the single
                # densest full-width row band, which averages a small face away. Bias toward the
                # corner OPPOSITE the speaker (where a face is least likely centered).
                opp_side = ("left" if side == "right"
                            else "right" if side == "left" else None)
                fp_cx, fp_top, fp_ink = _best_footprint(page_gray, rect, bubble_w, bubble_h,
                                                        prefer_side=opp_side)
                # Full-width band density is kept as the fallback trigger / comparison baseline.
                band_top, band_density = _emptiest_band(page_gray, rect, bubble_h)
                if fp_cx is not None:
                    footprint_cx, top = fp_cx, fp_top
                elif band_top is not None:
                    top = band_top
                # Effective density for the full-face fallback: the LEAST ink we could find under
                # any footprint (fp_ink) — if even the best footprint is dense, no clear spot
                # exists (a face fills the panel) and we bottom-anchor; otherwise the footprint
                # already found a clear, small-face-avoiding spot and no fallback is needed.
                density = fp_ink if fp_ink is not None else band_density
                # FULL-FACE fallback: even the emptiest footprint is dense (e.g. an extreme
                # close-up filling the panel). Shrink the font and bottom-anchor the
                # bubble over the chin/body rather than the eyes.
                if density is not None and density >= _DENSE_BAND:
                    shrink_min = max(int(settings.lettering["min_font"]),
                                     int(settings.lettering["max_font"] * 0.6))
                    # the fallback bubble is shorter (max_h*0.7); re-abridge to its
                    # tighter budget so the shrunken bubble still shows coherent text.
                    fb_budget = _char_budget(draw, fpath, max_w, max_h * 0.7,
                                             settings.lettering["min_font"])
                    fb_text = _abridge(text, fb_budget)
                    lines, font, tw, th, line_h = _fit_text(
                        draw, fb_text, fpath, max_w, max_h * 0.7,
                        shrink_min, settings.lettering["min_font"])
                    max_lines = max(1, int((max_h * 0.7) // line_h))
                    if len(lines) > max_lines:
                        lines = lines[:max_lines]
                        lines[-1] = (lines[-1][:-1] if len(lines[-1]) > 1 else lines[-1]) + "…"
                        th = line_h * len(lines)
                    tw = min(max((draw.textlength(l, font=font) for l in lines), default=tw),
                             rect.w * 0.66)
                    rx, ry = _bubble_size(tw, th, dline.style)
                    bubble_w = 2 * rx
                    bubble_h = 2 * ry
                    # bottom-anchor: prefer the emptiest FOOTPRINT in the LOWER half of the
                    # panel (resized bubble), so the chin/body spot also dodges any small
                    # off-center detail sideways; fall back to the lower-half row band.
                    lo_cx, lo_top, _ = _best_footprint(page_gray, rect, bubble_w, bubble_h,
                                                       prefer_side=opp_side, floor_frac=0.5)
                    if lo_cx is not None:
                        footprint_cx, top = lo_cx, lo_top
                    else:
                        lo_top2, _ = _emptiest_band(page_gray, rect, bubble_h, floor_frac=0.5)
                        top = lo_top2 if lo_top2 is not None else int(rect.y1 - bubble_h - PAD)
                    bottom_anchored = True

            # horizontal placement biases toward the speaker side
            if side == "right":
                cx = rect.x1 - bubble_w / 2 - PAD
            elif side == "left":
                cx = rect.x0 + bubble_w / 2 + PAD
            else:
                cx = (rect.x0 + rect.x1) / 2
            # CLOSE-UP corner bias: for the first bubble, shove it toward the TOP corner
            # OPPOSITE the speaker side so it sits off the centered face (tail still points
            # back toward the speaker). Only when not already bottom-anchored by the fallback.
            if is_close and not first_placed and not bottom_anchored:
                if side == "right":      # speaker on the right -> bubble to the LEFT corner
                    cx = rect.x0 + bubble_w / 2 + PAD
                elif side == "left":     # speaker on the left -> bubble to the RIGHT corner
                    cx = rect.x1 - bubble_w / 2 - PAD
                # bias the first close-up bubble toward the TOP only when we have no pixels to
                # analyze; with a page image the empty band already avoids the (centered) face.
                if page_gray is None:
                    top = min(top, rect.y0 + PAD)
            # FOOTPRINT override (FIRST bubble, with pixels): the footprint scorer already chose
            # the emptiest x-slot (dodging small off-center faces sideways) with a corner/speaker
            # bias, so it wins over the blind speaker-side / close-up corner default above.
            if footprint_cx is not None:
                cx = footprint_cx
            cx = min(max(cx, rect.x0 + bubble_w / 2 + 4), rect.x1 - bubble_w / 2 - 4)

            cy = top + bubble_h / 2
            if cy + bubble_h / 2 > rect.y1 - 4:
                # Overflows the panel bottom. Only the FIRST bubble (nothing above it)
                # may be nudged UP to fit — pulling a LATER (stacked) bubble up would
                # drag it over the previous bubble's bottom, causing overlap. A later
                # bubble that can't fit below its predecessor is simply skipped: stop the
                # run so the no-overlap invariant holds. Unplaced indices are surfaced as
                # dropped by the caller (which also protects high-priority lines).
                if first_placed:
                    return placed  # no room below the previous bubble; skip rather than overlap
                cy = rect.y1 - 4 - bubble_h / 2
                if cy - bubble_h / 2 < rect.y0 + 2:
                    return placed  # out of vertical room in this panel

            first_placed = True
            # ── recompute the tail target AFTER vertical placement is chosen ───
            tail_target = None
            # only a real left/right speaker direction gets a directional tail;
            # 'center'/'none' get a tail-less bubble (no misleading straight-down tail).
            if dline.style in ("speech", "shout", "thought") and side in ("left", "right"):
                tail_len = min(int(rect.h * 0.22), 95)        # short manga tail
                dirx = 1 if side == "right" else -1
                tx = cx + dirx * tail_len * 0.55
                ty = cy + bubble_h / 2 + tail_len
                tail_target = (min(max(tx, rect.x0 + 6), rect.x1 - 6),
                               min(ty, rect.y1 - 6))

            placed.append(di)
            shapes = (shape_lib or {}).get(dline.style)
            if shapes and page_live is not None and draw_enabled:
                # ORGANIC mode: composite a cached API bubble SHAPE onto the untouched panel, then
                # render the (already abridged) text crisply with Pillow inside the shape's interior.
                shape_path = shapes[(spec.panel_number + di) % len(shapes)]
                interior = bubbles.place_bubble(page_live, shape_path, int(cx), int(cy),
                                                int(bubble_w), int(bubble_h))
                # DEFECT 1 fix — reserve a BOTTOM safety margin inside the detected interior.
                # place_bubble reports the silhouette's interior height, but the TRUE usable
                # height is shorter near the rounded bottom + tail notch, so the last wrapped
                # line's descenders could land on/below the outline. Shrink the usable height
                # off the BOTTOM (speech has a tail -> larger reserve than narration/box) and
                # center the text in the REDUCED region so the block ends strictly above
                # (interior.y1 - margin), clearing the outline.
                m_frac = (_ORGANIC_BOTTOM_MARGIN_FRAC_SPEECH if dline.style == "speech"
                          else _ORGANIC_BOTTOM_MARGIN_FRAC)
                bottom_margin = int(round(interior.h * m_frac))
                usable_bottom = interior.y1 - bottom_margin     # text must end strictly above this
                iw = max(8, interior.w - 8)
                ih = max(8, (usable_bottom - interior.y0) - 8)  # reduced usable interior height
                # GUARANTEE the text fits the REDUCED interior: abridge to its char budget, fit
                # the font, then hard-cap lines to the reduced height (no overflow). Abridging to
                # the margined budget also keeps the block compact (fewer lines), which helps the
                # next bubble stack below without a geometric skip.
                # Re-abridge the coherent SOURCE string (`text`, the space-correct abridged line),
                # NOT " ".join(lines): _wrap hard-breaks an over-long token into adjacent chunks
                # carrying no separating space, so re-joining them with " " would inject spurious
                # mid-word spaces that the interior re-wrap would then draw as visible word breaks.
                itext = _abridge(text,
                                 _char_budget(draw, fpath, iw, ih, settings.lettering["min_font"]))
                ilines, ifont, _itw, ith, ilh = _fit_text(
                    draw, itext, fpath, iw, ih,
                    settings.lettering["max_font"], settings.lettering["min_font"])
                # CLAMP to the reduced interior height so the centered draw can never exceed it.
                # When the shape's detected interior is short/irregular, ilh may be taller
                # than ih (a single line won't fit); in that case keep exactly one line
                # (possibly clipped by the shape) rather than forcing more lines below.
                if ilh > ih:
                    ilines = ilines[:1]
                    ith = ilh * len(ilines)
                else:
                    imax = max(1, int(ih // ilh))
                    if len(ilines) > imax:
                        ilines = ilines[:imax]
                        ilines[-1] = (ilines[-1][:-1] if len(ilines[-1]) > 1 else ilines[-1]) + "…"
                        ith = ilh * len(ilines)
                # Center within the REDUCED region [interior.y0, usable_bottom]; its bottom edge
                # (block_top + ith) then sits at/above usable_bottom, never crossing the outline.
                block_top = (interior.y0 + usable_bottom) / 2 - ith / 2
                block_top = min(block_top, usable_bottom - ith)  # belt-and-suspenders
                _draw_text_block(draw, ilines, ifont, (interior.x0 + interior.x1) / 2,
                                 block_top, ilh)
            elif draw_enabled:
                _draw_bubble(draw, cx, cy, tw, th, dline.style, tail_target)
                _draw_text_block(draw, lines, font, cx, cy - th / 2, line_h)
            # advance past the actual drawn extent (tail tip reaches below the body)
            # so the next bubble never overlaps the previous bubble's tail.
            body_bottom = cy + bubble_h / 2
            tail_bottom = tail_target[1] if tail_target else body_bottom
            y_cursor = max(body_bottom, tail_bottom) + GAP
        return placed

    # MEASURE pass (no drawing / no compositing) in reading order: how many bubbles fit?
    fit_in_order = _place(line_idxs, draw_enabled=False)
    n_fit = len(fit_in_order)

    if n_fit >= len(line_idxs):
        # everything fits — draw straight through in reading order.
        _place(line_idxs, draw_enabled=True)
        return

    # DROP path: the panel can hold only n_fit of its line_idxs lines. Choose WHICH to keep
    # by importance (never sacrifice the only narration/caption or a shout to keep excess
    # speech), then RENDER the kept lines in their original on-page reading order (placement
    # order is never reshuffled). Ties break on reading order, so among equal-priority lines
    # the earlier one is kept.
    by_priority = sorted(line_idxs, key=lambda i: (_drop_priority(capped[i].style), i))
    keep = sorted(by_priority[:n_fit])
    actually_placed = set(_place(keep, draw_enabled=True))
    # surface every non-empty line NOT actually drawn as dropped, preserving reading order.
    # (Use what really drew, not just `keep`: a differently-sized keep-set could in rare
    # cases fit one fewer than the measure pass; those are reported too, not silently lost.)
    drop_idxs = [i for i in line_idxs if i not in actually_placed]
    _record_dropped(dropped, [capped[i] for i in drop_idxs])


def run(settings: Settings, pages: list[PageLayout], specs: list[PanelSpec],
        chapter_number: int, client=None, tracker=None, cache=None) -> list[str]:
    by_num = {s.panel_number: s for s in specs}
    # ORGANIC bubble mode (opt-in): build a cached library of API-generated transparent bubble
    # SHAPES once, then composite them onto untouched panels (Pillow still renders the text).
    # Falls back to drawn Pillow bubbles if the library can't be built.
    organic = (str(settings.lettering.get("bubble_style", "drawn")).lower() == "organic"
               and client is not None and cache is not None)
    shape_lib: dict = {}
    if organic:
        try:
            shape_lib = bubbles.ensure_shape_library(client, settings, tracker, cache)
        except Exception as e:
            print(f"[letter] WARNING: organic bubble library unavailable "
                  f"({type(e).__name__}: {str(e)[:100]}); using drawn bubbles for the "
                  f"whole chapter", file=sys.stderr)
            shape_lib = {}
        # PARTIAL library: ensure_shape_library silently skips any style it could not
        # build, which would route that style to drawn bubbles with no signal. Surface it.
        missing = [s for s in bubbles.STYLES if s not in shape_lib]
        if missing and shape_lib:  # total-failure already warned above
            print(f"[letter] WARNING: organic bubbles unavailable for styles "
                  f"{missing}; used drawn bubbles for those", file=sys.stderr)

    # Self-clean this chapter's stale LETTERED pages first so a re-run producing FEWER pages
    # never leaves orphan _lettered.png files behind (the layout stage does the same for the
    # base pages). The PDF/CBZ are rebuilt from the manifest, so this only tidies loose PNGs.
    settings.out_dir.mkdir(parents=True, exist_ok=True)
    for stale in settings.out_dir.glob(f"chapter-{chapter_number}_page_*_lettered.png"):
        try:
            stale.unlink()
        except OSError:
            pass

    out_paths: list[str] = []
    dropped: list[tuple[int, int, str]] = []  # (page_number, panel_number, text)
    for page in pages:
        img = Image.open(page.image_path).convert("L")
        # snapshot the un-lettered artwork for content-aware placement, so analysis
        # reflects the original panel pixels (faces/detail), not earlier bubbles.
        page_src = img.copy()
        draw = ImageDraw.Draw(img)
        for pn, rect in zip(page.panel_numbers, page.rects):
            panel_dropped: list[str] = []
            letter_panel(draw, rect, by_num.get(pn), settings, page_img=page_src,
                         page_live=img, shape_lib=shape_lib, dropped=panel_dropped)
            dropped.extend((page.page_number, pn, t) for t in panel_dropped)
        out_path = settings.out_dir / f"chapter-{chapter_number}_page_{page.page_number:02d}_lettered.png"
        img.save(out_path)
        out_paths.append(str(out_path))
    if dropped:
        detail = "; ".join(f"p{pg}/panel{pn}: {t!r}" for pg, pn, t in dropped)
        print(f"[letter] WARNING: {len(dropped)} dialogue line(s) dropped because they "
              f"did not fit their panels: {detail}", file=sys.stderr)
    mpath = settings.artifacts_dir / f"chapter-{chapter_number}.lettered.json"
    mpath.write_text(json.dumps(out_paths, indent=2), encoding="utf-8")
    return out_paths
