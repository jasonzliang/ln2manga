"""Validated "organic" speech-bubble feature.

Instead of drawing bubbles with bare Pillow primitives (smooth ellipses / regular stars), this
module builds a small *library* of AI-generated empty bubble silhouettes — one set of tiles per
style — and lets the lettering stage composite a tile onto the page and draw the text inside the
tile's interior. The result reads as a hand-inked manga bubble rather than a vector shape.

VERIFIED FACTS baked in:
  - gpt-image-2 REJECTS background="transparent" (HTTP 400). The transparent-background option is
    only available on the gpt-image-1 family, so the bubbles model DEFAULTS to "gpt-image-1-mini".
  - The generated transparent tile's INTERIOR is NOT opaque white — it comes back transparent /
    near-black. So after decoding we FLOOD-FILL the interior to opaque white, otherwise dark page
    art shows through the bubble and the lettered text is unreadable.

The library is CONTENT-ADDRESSED cached on the exact generation inputs, so re-running the pipeline
never re-pays for tiles it already has. ensure_shape_library never raises: any failure (no client,
API error, decode error) simply skips that style so the caller can fall back to the Pillow bubbles.
"""
from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .artifacts import Rect
from .cache import Cache, _atomic_write_bytes, cache_key
from .config import Settings
from .cost import BudgetExceeded, CostTracker
from .retry import with_retry

STYLES = ("speech", "shout", "thought", "narration")

# One empty bubble silhouette per style. Each prompt describes a SINGLE empty bubble of the right
# manga shape — white fill, bold black outline, no text inside — centered on a fully transparent
# background so the tile can be composited straight onto a page.
STYLE_PROMPTS: dict[str, str] = {
    "speech": (
        "A single empty manga speech balloon shaped like a ROUNDED RECTANGLE — a wide rectangle "
        "with softly rounded corners, NOT an oval — with a clean bold solid black outline and a "
        "flat pure white fill, and a short pointed tail at the bottom edge. The rounded-rectangle "
        "shape holds more text than an oval. Completely EMPTY inside — absolutely no text, letters, "
        "words or symbols. The bubble is centered and fills most of the frame. Everything outside "
        "the outline is a fully transparent background. Flat black-and-white manga ink style, no "
        "shading, no gradient."
    ),
    "shout": (
        "A single empty manga shout / scream balloon. One spiky jagged explosion-shaped burst "
        "bubble with sharp pointed star-like edges, a bold solid black outline and a flat pure "
        "white fill. Completely EMPTY inside — absolutely no text, letters, words or symbols. The "
        "bubble is centered and fills most of the frame. Everything outside the jagged outline is "
        "a fully transparent background. Flat black-and-white manga ink style, no shading."
    ),
    "thought": (
        "A single empty manga thought bubble. One fluffy cloud-shaped bubble made of soft bumpy "
        "rounded lobes, with a bold solid black outline and a flat pure white fill, and a trail of "
        "two or three small circles below it. Completely EMPTY inside — absolutely no text, letters, "
        "words or symbols. The cloud is centered and fills most of the frame. Everything outside the "
        "outline is a fully transparent background. Flat black-and-white manga ink style, no shading."
    ),
    "narration": (
        "A single empty manga narration caption box. One rounded rectangle box with a bold solid "
        "black outline and a flat pure white fill. Completely EMPTY inside — absolutely no text, "
        "letters, words or symbols. The box is centered and fills most of the frame. Everything "
        "outside the box outline is a fully transparent background. Flat black-and-white manga ink "
        "style, no shading, no gradient."
    ),
}

# Default generation params (overridable via settings.bubbles).
_DEFAULT_MODEL = "gpt-image-1-mini"
_DEFAULT_SIZE = "1024x1024"
_DEFAULT_QUALITY = "low"
_DEFAULT_COUNT = 2

# Image generation is non-deterministic, so a single tile can come back with an unfillable silhouette
# (post-processing ValueError) even for a style that usually succeeds. Because that raise happens
# AFTER the paid API call, discarding it would re-pay the same index every run (the cache key has no
# attempt nonce). Re-roll a fresh image a bounded number of times so the first acceptable output is
# recorded + cached, turning unbounded per-run re-pay into a bounded one-time cost.
_MAX_TILE_ATTEMPTS = 3

# Bump whenever _fill_interior_opaque's post-processing changes: cached tiles store the PROCESSED
# PNG keyed on the generation inputs only, so without this token a re-run would serve tiles baked by
# the OLD processing and the fix would never appear. v2 = silhouette-sized erosion + degenerate-tile
# guards (no all-black / all-white-box bubbles). v3 = morphological close of the outline mask before
# the border flood + a post-fill white-interior-fraction guard (rejects outline-only / double-line
# frames whose enclosed interior is empty/transparent, e.g. the broken narration_1 tile).
_PROC_VERSION = 3

# Minimum opaque-white interior fraction = (white core pixels) / (silhouette bbox area). An
# outline-only / double-line frame whose interior leaked transparent (the flood-fill reached the
# would-be interior through gaps in the border) erodes to a thin core that fills only a sliver of its
# bounding box; reject anything below this so a hollow tile cannot ship with no white text backing.
_MIN_INTERIOR_FRAC = 0.20

# Interior detection for the text Rect: opaque + clearly white.
_WHITE_MIN = 240
_ALPHA_MIN = 200
_INTERIOR_MARGIN = 6
# Minimum fraction of the interior Rect that must be white. A non-convex shape (spiky shout star,
# lobed thought cloud) has a global white bbox that spans the ink/transparent arms, so text centered
# in that bbox lands on the outline; if the bbox is too sparse we shrink it toward the white centroid.
_MIN_WHITE_FRAC = 0.6


# ── stable-name materialization ─────────────────────────────────────────────────
def _link_or_copy(src: Path, dest: Path, data: bytes) -> None:
    """Materialize the stable, human-named `dest` as a HARDLINK to the content-hash cache file
    `src` (same inode, zero extra bytes). The cache file is a write-once content-addressed entry,
    so the link never goes stale; the unlink-then-link re-points `dest` if the target changed.

    On a filesystem that can't hardlink (cross-device / unsupported / Windows) fall back to the
    atomic temp + os.replace write of `data` (same hardening as the content cache), so an interrupt
    mid-loop cannot leave a truncated {style}_{i}.png that place_bubble later loads un-revalidated.
    """
    dest = Path(dest)
    try:
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        os.link(src, dest)
    except OSError:
        _atomic_write_bytes(dest, data)


# ── settings access ──────────────────────────────────────────────────────────
def _bubbles_cfg(settings: Settings) -> dict[str, Any]:
    cfg = getattr(settings, "bubbles", {}) or {}
    return cfg if isinstance(cfg, dict) else {}


# ── flood fill ────────────────────────────────────────────────────────────────
def _fill_interior_opaque(img: Image.Image) -> Image.Image:
    """Make the bubble interior + outline OPAQUE WHITE while keeping the true outside transparent.

    The generated tile comes back with a transparent (or near-black) interior, so pasting it onto a
    dark page would leave the bubble see-through and the text unreadable. We robustly find the
    "outside" as the set of transparent pixels CONNECTED to the image border (flood-fill the alpha
    mask inward from every corner). Every pixel NOT in that outside region is part of the bubble
    (interior or outline) and is forced opaque: dark outline pixels stay black, everything else
    becomes pure white. The outside stays transparent.
    """
    from PIL import ImageFilter
    img = img.convert("RGBA")
    arr = np.asarray(img)  # (H, W, 4), read-only view
    alpha = arr[:, :, 3]

    # Transparent-ish pixels are candidates for "outside" (the background); opaque pixels are the
    # ink/fill of the bubble and can never be flooded as outside.
    transparent = alpha < 128

    # Morphologically CLOSE the OPAQUE (outline) mask before flooding: a double-line / dashed border
    # has small transparent gaps the exterior flood can leak through into the would-be interior,
    # which would then be (wrongly) treated as outside. Dilating then eroding the opaque mask seals
    # those gaps so the flood stays out, WITHOUT permanently thickening the outline (we only use the
    # sealed mask to define `transparent` for the flood; the kept pixels are still classed below).
    opaque_img = Image.fromarray(((~transparent) * 255).astype(np.uint8), "L")
    ck = max(3, (min(transparent.shape) // 90) | 1)        # close kernel (odd); ~ outline thickness
    closed = np.asarray(opaque_img.filter(ImageFilter.MaxFilter(ck))
                                  .filter(ImageFilter.MinFilter(ck))) > 127
    transparent = transparent & ~closed

    # BFS flood from all border transparent pixels: anything reachable through transparent pixels
    # from the image edge is the true outside. Enclosed transparent pixels (the interior) are NOT
    # reached, so they get filled white below.
    h, w = transparent.shape
    outside = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    def _seed(r: int, c: int) -> None:
        if transparent[r, c] and not outside[r, c]:
            outside[r, c] = True
            stack.append((r, c))

    for c in range(w):
        _seed(0, c)
        _seed(h - 1, c)
    for r in range(h):
        _seed(r, 0)
        _seed(r, w - 1)

    while stack:
        r, c = stack.pop()
        if r > 0:
            _seed(r - 1, c)
        if r + 1 < h:
            _seed(r + 1, c)
        if c > 0:
            _seed(r, c - 1)
        if c + 1 < w:
            _seed(r, c + 1)

    # A FULLY-OPAQUE tile has no border-connected transparent background at all, so the flood finds
    # nothing and `outside` is empty. Filling such a tile would turn the WHOLE frame into a solid
    # white box with no outline, so reject it: the raise propagates out of _generate_tile into
    # ensure_shape_library's try/except, which skips this style and falls back to drawn bubbles.
    if outside.sum() == 0:
        raise ValueError("bubble tile has no transparent background")

    # "inside" = the filled bubble SILHOUETTE (interior + outline + whatever fill the model used).
    inside = ~outside

    # A FULLY-TRANSPARENT tile floods everything as `outside` (so the guard above passes), leaving
    # `inside` all-False. The `ys.max()`/`xs.max()` below would then raise a cryptic numpy "zero-size
    # array" error before the descriptive guards; reject it here with a clear message instead.
    if not inside.any():
        raise ValueError("bubble tile is fully transparent (no silhouette)")

    # The model frequently returns a BLACK-FILLED interior, so we must NOT keep dark interior pixels
    # black (that hides the text). Instead force a CLEAN bubble from the silhouette: erode it to get
    # an interior CORE -> opaque white, and the remaining outer RING -> the black outline. This gives
    # a readable white bubble with a uniform black outline regardless of the model's fill.
    h2, w2 = inside.shape
    k = max(7, (min(h2, w2) // 90) | 1)               # erosion kernel (odd); ~ a few px of outline
    # Size the kernel from the SILHOUETTE's own bounding box, not the full tile: a small/thin
    # silhouette eroded by a tile-sized kernel disappears entirely, painting an all-black blob.
    ys, xs = np.nonzero(inside)
    span = min(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
    k = max(3, min(k, ((span // 3) | 1)))             # keep odd, never exceed ~1/3 of the silhouette
    inside_img = Image.fromarray((inside * 255).astype(np.uint8), "L")
    core = np.asarray(inside_img.filter(ImageFilter.MinFilter(k))) > 127

    # If erosion wiped out the whole core, the silhouette is too thin to carry a white interior;
    # refuse rather than emit an all-black tile (also triggers the drawn-bubble fallback).
    if core.sum() == 0 and inside.sum() > 0:
        raise ValueError("bubble silhouette too thin")

    # Real INTERIOR measure: the two guards above pass for an outline-only / DOUBLE-LINE frame whose
    # enclosed interior is empty/transparent — there IS a transparent background (so `outside` is
    # non-empty) and eroding the thick stroke still leaves a non-empty core (pixels within the stroke
    # thickness), yet almost none of the silhouette's bounding box is white. Compute the opaque-white
    # core fraction over the silhouette bbox and reject a hollow tile so it never ships with no white
    # text backing (the raise reaches ensure_shape_library's try/except -> drawn-bubble fallback).
    bbox_area = float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
    if bbox_area > 0 and core.sum() / bbox_area < _MIN_INTERIOR_FRAC:
        raise ValueError("bubble interior not filled")

    out = arr.copy()
    out[outside] = (0, 0, 0, 0)                        # true outside stays transparent
    out[inside & ~core] = (0, 0, 0, 255)              # outer ring -> black outline
    out[core] = (255, 255, 255, 255)                  # interior core -> white (text goes here)
    return Image.fromarray(out, "RGBA")


# ── one tile ──────────────────────────────────────────────────────────────────
def _usage_dict(resp: Any) -> dict:
    """Extract real API token usage from a response (mirrors imagegen._usage_dict)."""
    u = getattr(resp, "usage", None)
    if u is None:
        return {}
    return {k: getattr(u, k, None)
            for k in ("input_tokens", "output_tokens", "total_tokens")}


def _generate_tile(client, model: str, prompt: str, size: str,
                   quality: str, timeout: float = 240.0) -> tuple[bytes, dict]:
    """Call the image API for one tile and return (filled-interior PNG bytes, usage dict). May raise.

    Wrapped in with_retry (like imagegen.generate_image) so a transient RateLimit/Timeout/
    Connection/InternalServerError is retried instead of silently dropping the whole style to
    Pillow, and a per-call timeout stops a hung connection from stalling the lettering stage.
    """
    resp = with_retry(client.images.generate)(
        model=model,
        prompt=prompt,
        background="transparent",
        output_format="png",
        size=size,
        quality=quality,
        n=1,
        timeout=timeout,
    )
    raw = base64.b64decode(resp.data[0].b64_json)
    img = Image.open(io.BytesIO(raw)).convert("RGBA")
    filled = _fill_interior_opaque(img)
    buf = io.BytesIO()
    filled.save(buf, "PNG")
    return buf.getvalue(), _usage_dict(resp)


def _generate_tile_retrying(client, model: str, prompt: str, size: str, quality: str,
                            timeout: float, tracker: CostTracker, style: str, i: int,
                            est: float) -> tuple[bytes, dict]:
    """Generate one tile, re-rolling a fresh image on a post-processing ValueError.

    A non-deterministic generation can produce an unfillable silhouette (e.g. a transparent / hollow
    tile) that _fill_interior_opaque rejects with a ValueError. That raise is AFTER the paid API
    call, so a bare give-up would re-pay the same index every run. Re-roll up to _MAX_TILE_ATTEMPTS
    times so the first acceptable output is the one recorded + cached. Each attempt costs an API call,
    so every attempt is budget-checked and the last attempt's ValueError propagates (skipping the
    style); transient API errors are already retried inside _generate_tile and propagate unchanged.
    """
    for attempt in range(_MAX_TILE_ATTEMPTS):
        tracker.check(est, is_image=True)
        try:
            return _generate_tile(client, model, prompt, size, quality, timeout)
        except ValueError as e:
            if attempt + 1 >= _MAX_TILE_ATTEMPTS:
                raise
            print(f"[bubbles] style {style!r} tile {i} unusable "
                  f"({e}); re-rolling ({attempt + 2}/{_MAX_TILE_ATTEMPTS})", file=sys.stderr)
    raise AssertionError("unreachable")  # loop either returns or raises


# ── public: build the library ──────────────────────────────────────────────────
def ensure_shape_library(client, settings: Settings, tracker: CostTracker, cache: Cache,
                         count: int | None = None) -> dict[str, list[Path]]:
    """Ensure `count` bubble tiles exist per style and return {style: [Path, ...]}.

    Content-addressed: tiles already in the cache are reused with no API call. On a missing client
    or ANY failure for a given style, that style is skipped (omitted / partial) so the caller can
    fall back to Pillow-drawn bubbles. Never raises out of this function.
    """
    cfg = _bubbles_cfg(settings)
    if count is None:
        count = int(cfg.get("count_per_style", _DEFAULT_COUNT))
    model = str(cfg.get("model", _DEFAULT_MODEL))
    size = str(cfg.get("size", _DEFAULT_SIZE))
    quality = str(cfg.get("quality", _DEFAULT_QUALITY))
    timeout = float(cfg.get("timeout_s", settings.image.get("timeout_s", 240)))

    out_dir = settings.cache_dir("bubbles")
    library: dict[str, list[Path]] = {}

    if client is None:
        return library

    # gpt-image-2 rejects background="transparent" (HTTP 400), so every tile would fail and the
    # whole feature would silently fall back to drawn bubbles. Warn loudly and bail early rather
    # than burn a retry storm against a guaranteed-to-fail model.
    if model.startswith("gpt-image-2"):
        print("[warn] bubbles.model must be a gpt-image-1-family model "
              "(gpt-image-2 rejects background=transparent); organic bubbles disabled",
              file=sys.stderr)
        return library

    skipped: list[str] = []
    for style in STYLES:
        prompt = STYLE_PROMPTS[style]
        # Buffer (path, bytes) and only materialize the stable copies once the whole style
        # succeeds, so a later-tile failure leaves no orphaned {style}_0.png on disk. Clear any
        # pre-existing stable copies first so a reduced count_per_style leaves no stale high-index
        # files behind.
        for old in out_dir.glob(f"{style}_*.png"):
            old.unlink(missing_ok=True)
        pending: list[tuple[Path, Path, bytes]] = []   # (stable name, content-hash cache file, bytes)
        ok = True
        for i in range(count):
            key = cache_key({"op": "bubble", "model": model, "prompt": prompt,
                             "size": size, "quality": quality, "i": i,
                             "proc": _PROC_VERSION})
            try:
                cached = cache.get("bubbles", key)
                if cached is not None:
                    png = cached
                else:
                    est = tracker.estimate_image(model, quality)
                    png, usage = _generate_tile_retrying(
                        client, model, prompt, size, quality, timeout, tracker, style, i, est)
                    actual = tracker.estimate_image_from_usage(model, usage)
                    tracker.record("image", model, max(est, actual),
                                   {"op": "bubble", "style": style, "i": i, "usage": usage})
                    cache.put("bubbles", key, png,
                              meta={"style": style, "i": i, "model": model})
                # Both branches leave the PROCESSED tile at the content-hash path; the stable name is
                # hardlinked to it below (no second copy of the bytes).
                pending.append((out_dir / f"{style}_{i}.png",
                                cache.path("bubbles", key), png))
            except BudgetExceeded as e:
                # Budget caps are not generic API failures: report once and loudly so the run
                # makes clear WHY organic bubbles were abandoned.
                print(f"[bubbles] budget reached building {style!r} tiles: {e}; "
                      f"using drawn bubbles", file=sys.stderr)
                ok = False
                break
            except Exception as e:
                # Skip this style entirely so the caller falls back to Pillow — but say so, instead
                # of degrading silently (matches panels.py's [warn] precedent).
                print(f"[bubbles] style {style!r} failed "
                      f"({type(e).__name__}: {str(e)[:120]}); using drawn bubbles",
                      file=sys.stderr)
                ok = False
                break
        if ok and pending:
            # Materialize the stable, human-named files now that the whole style succeeded. Each is
            # a HARDLINK to its content-hash cache file (same inode, zero extra bytes), with an
            # atomic temp + os.replace copy as the fallback where hardlinks are unsupported.
            for stable, cache_file, png in pending:
                _link_or_copy(cache_file, stable, png)
            library[style] = [p for p, _, _ in pending]
        else:
            skipped.append(style)

    # A partial library mixes organic bubbles for the succeeded styles with drawn bubbles for the
    # skipped ones on the same page; surface that so the visual inconsistency is at least audible.
    if library and skipped:
        print(f"[bubbles] partial library: styles {sorted(skipped)} fell back to drawn bubbles "
              f"while {sorted(library)} use organic tiles", file=sys.stderr)
    return library


# ── public: place a tile on a page ──────────────────────────────────────────────
def place_bubble(page: Image.Image, shape_path: str | Path, cx: int, cy: int,
                 target_w: int, target_h: int) -> Rect:
    """Composite a bubble tile centered at (cx, cy) onto a greyscale page and return the text Rect.

    `page` is an "L" (greyscale) PIL Image, mutated in place. The tile is resized to
    (target_w, target_h) with LANCZOS and alpha-composited so pixels OUTSIDE the tile's alpha stay
    exactly as they were. Returns the interior Rect (page coordinates) where text should be drawn:
    the bbox of the scaled tile's opaque-white interior, shrunk by ~6px, offset to the page. For
    non-convex tiles (shout/thought) the bbox is shrunk toward the white centroid until it is mostly
    white so text does not land on the ink/gaps. If no clear interior is found, returns a centered
    Rect at ~60% of the tile. The returned Rect is always clamped on-page.
    """
    target_w = max(1, int(target_w))
    target_h = max(1, int(target_h))
    shape = Image.open(shape_path).convert("RGBA").resize((target_w, target_h), Image.LANCZOS)

    left = int(cx) - target_w // 2
    top = int(cy) - target_h // 2

    # Composite onto the greyscale page: paste the tile's greyscale using its alpha as the mask, so
    # only pixels covered by the bubble change.
    gray = shape.convert("L")
    alpha = shape.getchannel("A")
    page.paste(gray, (left, top), alpha)

    # Interior = opaque + clearly white pixels of the scaled tile.
    arr = np.asarray(shape)
    rgb_white = (arr[:, :, 0] > _WHITE_MIN) & (arr[:, :, 1] > _WHITE_MIN) & (arr[:, :, 2] > _WHITE_MIN)
    interior = rgb_white & (arr[:, :, 3] > _ALPHA_MIN)

    ys, xs = np.nonzero(interior)
    if xs.size and ys.size:
        x0 = int(xs.min()) + _INTERIOR_MARGIN
        y0 = int(ys.min()) + _INTERIOR_MARGIN
        x1 = int(xs.max()) - _INTERIOR_MARGIN
        y1 = int(ys.max()) - _INTERIOR_MARGIN
        if x1 <= x0 or y1 <= y0:
            x0, y0, x1, y1 = _centered_box(target_w, target_h)
        else:
            x0, y0, x1, y1 = _fit_white_rect(interior, x0, y0, x1, y1, target_w, target_h)
    else:
        x0, y0, x1, y1 = _centered_box(target_w, target_h)

    # Offset tile-local coords into page coords and clamp to the page so a tile near a page edge can
    # never yield a negative / off-page Rect (the docstring promises page coordinates).
    x0 = max(0, x0 + left)
    y0 = max(0, y0 + top)
    x1 = min(page.width, x1 + left)
    y1 = min(page.height, y1 + top)
    if x1 <= x0 or y1 <= y0:
        cx0, cy0, cx1, cy1 = _centered_box(target_w, target_h)
        x0 = max(0, min(page.width - 1, cx0 + left))
        y0 = max(0, min(page.height - 1, cy0 + top))
        x1 = max(x0 + 1, min(page.width, cx1 + left))
        y1 = max(y0 + 1, min(page.height, cy1 + top))
    return Rect(x0=x0, y0=y0, x1=x1, y1=y1)


def _fit_white_rect(interior: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                    w: int, h: int) -> tuple[int, int, int, int]:
    """Shrink the [x0,x1]x[y0,y1] box toward the white centroid until it is mostly white.

    The raw interior bbox of a non-convex shape spans the ink/transparent arms (a cross or star is
    only ~36% white inside its bbox), so text centered there lands on the outline. We contract the
    box symmetrically about the white centroid until the enclosed white fraction clears the
    threshold; if it collapses first we fall back to a centered box.
    """
    sub = interior[y0:y1 + 1, x0:x1 + 1]
    if sub.size == 0 or sub.mean() >= _MIN_WHITE_FRAC:
        return x0, y0, x1, y1
    ys, xs = np.nonzero(interior)
    cx = float(xs.mean())
    cy = float(ys.mean())
    for _ in range(64):
        bw, bh = x1 - x0, y1 - y0
        if bw <= 2 or bh <= 2:
            return _centered_box(w, h)
        # contract ~8% per side toward the centroid
        x0 = int(round(x0 + (cx - x0) * 0.08))
        x1 = int(round(x1 - (x1 - cx) * 0.08))
        y0 = int(round(y0 + (cy - y0) * 0.08))
        y1 = int(round(y1 - (y1 - cy) * 0.08))
        if x1 <= x0 or y1 <= y0:
            return _centered_box(w, h)
        sub = interior[y0:y1 + 1, x0:x1 + 1]
        if sub.size and sub.mean() >= _MIN_WHITE_FRAC:
            return x0, y0, x1, y1
    return x0, y0, x1, y1


def _centered_box(w: int, h: int) -> tuple[int, int, int, int]:
    """A centered box covering ~60% of the tile (tile-local coords)."""
    bw, bh = int(w * 0.6), int(h * 0.6)
    x0 = (w - bw) // 2
    y0 = (h - bh) // 2
    return x0, y0, x0 + bw, y0 + bh
