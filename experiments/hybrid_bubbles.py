#!/usr/bin/env python3
"""STANDALONE prototype: HYBRID speech bubbles.

Pipeline under test:
  1. images.generate(background="transparent", output_format="png") -> an EMPTY manga
     speech-bubble SHAPE (NO text) on a transparent background. No source art in the
     request => zero art to degrade.
  2. Alpha-composite that tile onto an UNTOUCHED B&W panel at a content-aware empty
     region.
  3. Find the bubble's interior (largest opaque/white region of the tile) and render the
     dialogue text INSIDE it with Pillow (crisp, deterministic, free, exact text).

This file does NOT import or modify anything under ln2manga/stages/ or config.

Run:  OPENAI_API_KEY=... python3 experiments/hybrid_bubbles.py
Budget: ~3 generate calls at gpt-image-2 LOW quality (~$0.04 each) -> ~$0.12.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

MANGA_DIR = Path("/home/jason/Desktop/ln2manga/data/cache/manga")
OUT_DIR = Path("/home/jason/Desktop/ln2manga/data/experiments")
TILE_DIR = OUT_DIR / "bubbles"
FONT_DIR = Path("/home/jason/Desktop/ln2manga/assets/fonts")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TILE_DIR.mkdir(parents=True, exist_ok=True)

# NOTE: gpt-image-2 REJECTS background="transparent" (400: "Transparent background is not
# supported for this model."). Transparency is only supported by gpt-image-1 / -1-mini.
# gpt-image-1-mini is also the cheapest (low=$0.01) and renders a clean outline bubble.
MODEL = "gpt-image-1-mini"
TILE_SIZE = "1024x1024"   # square tile for a bubble shape; cheapest official size
QUALITY = "low"           # an empty shape needs no detail -> low quality keeps cost down
FMT = "png"

# Prompt for an EMPTY bubble (NO text inside, transparent bg).
SHAPE_PROMPTS = {
    "speech": (
        "a single empty black-and-white manga speech bubble: white fill, bold black "
        "outline, small tail, nothing inside, transparent background"
    ),
    "shout": (
        "a single empty black-and-white manga shout/scream balloon: spiky jagged "
        "explosion outline, white fill, bold black outline, small tail, nothing inside, "
        "transparent background"
    ),
    "speech2": (
        "a single empty rounded manga speech balloon, oval shape: white fill, thin clean "
        "black outline, short pointed tail at the bottom, completely empty inside, "
        "transparent background"
    ),
}

# Two panels with genuine empty regions (verified via vision):
#  - panel 19: Subaru carrying Petra; big empty WHITE space top-right.  "Subaru!" (shout)
#  - panel 4 : 5 packed faces; empty band TOP-CENTER.                   speech
JOBS = [
    {
        "n": 1,
        "panel": 19,
        "shape": "shout",
        "text": "SUBARU!",
        # normalized target box (x0,y0,x1,y1) for the bubble CENTER region, in [0,1].
        # top-right empty quadrant of panel 19.
        "target": (0.50, 0.02, 0.99, 0.34),
    },
    {
        "n": 2,
        "panel": 4,
        "shape": "speech",
        "text": "You're outrageous.",
        # empty band top-center of panel 4 (between Emilia & Anastasia heads).
        "target": (0.30, 0.02, 0.72, 0.22),
    },
]


# ---------------------------------------------------------------------------
# 1) generate empty bubble tiles
# ---------------------------------------------------------------------------
def gen_tile(client: OpenAI, shape: str) -> Path:
    out = TILE_DIR / f"tile_{shape}.png"
    if out.exists():
        print(f"  [cache] tile_{shape}.png already present, reusing")
        return out
    # reuse the earlier transparent-background probe for the plain speech bubble (free)
    probe = TILE_DIR / "_probe_gpt-image-1-mini.png"
    if shape == "speech" and probe.exists():
        out.write_bytes(probe.read_bytes())
        print(f"  [reuse] seeded tile_speech.png from transparent-bg probe")
        return out
    print(f"  [generate] shape={shape!r} ...")
    resp = client.images.generate(
        model=MODEL,
        prompt=SHAPE_PROMPTS[shape],
        size=TILE_SIZE,
        quality=QUALITY,
        background="transparent",
        output_format=FMT,
        moderation="low",
        n=1,
    )
    png = base64.b64decode(resp.data[0].b64_json)
    out.write_bytes(png)
    usage = getattr(resp, "usage", None)
    u = {k: getattr(usage, k, None) for k in ("input_tokens", "output_tokens", "total_tokens")} if usage else {}
    print(f"      -> {out} ({len(png)} bytes) usage={u}")
    return out


# ---------------------------------------------------------------------------
# 2) alpha cleanliness report for a tile
# ---------------------------------------------------------------------------
def alpha_report(tile_path: Path) -> dict:
    im = Image.open(tile_path).convert("RGBA")
    a = np.asarray(im)[:, :, 3]
    total = a.size
    fully_transp = int((a == 0).sum())
    fully_opaque = int((a == 255).sum())
    partial = total - fully_transp - fully_opaque
    return {
        "tile": tile_path.name,
        "size": im.size,
        "mode": Image.open(tile_path).mode,
        "pct_transparent": round(100 * fully_transp / total, 2),
        "pct_opaque": round(100 * fully_opaque / total, 2),
        "pct_partial_alpha": round(100 * partial / total, 2),
    }


# ---------------------------------------------------------------------------
# 3) crop tile to the bubble's tight bbox (drop transparent margins)
# ---------------------------------------------------------------------------
def crop_to_bubble(tile_path: Path) -> Image.Image:
    im = Image.open(tile_path).convert("RGBA")
    a = np.asarray(im)[:, :, 3]
    ys, xs = np.where(a > 16)
    if len(xs) == 0:
        return im
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    return im.crop((int(x0), int(y0), int(x1) + 1, int(y1) + 1))


# ---------------------------------------------------------------------------
# 3b) the API bubble is OUTLINE-ONLY (transparent interior). Detect the region
#     enclosed by the outline, FILL it white, and return that interior mask.
# ---------------------------------------------------------------------------
def fill_interior_white(tile_rgba: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """The generated bubble is a black outline on a transparent field (no fill). To make a
    usable opaque bubble we:
      1. treat any non-transparent pixel as the OUTLINE (the drawn ink),
      2. flood-fill 'background' inward from all 4 borders through transparent pixels,
      3. interior = transparent pixels NOT reached from the border (i.e. enclosed),
      4. paint interior + outline-adjacent gaps solid white, keep the black outline on top.
    Returns (opaque_tile_rgba, interior_mask)."""
    from scipy import ndimage

    arr = np.asarray(tile_rgba).copy()
    a = arr[:, :, 3]
    H, W = a.shape
    ink = a > 40                       # the drawn outline (anything meaningfully opaque)
    transp = ~ink                      # candidate background/interior

    # Label transparent components (4-connectivity). The bubble INTERIOR is the transparent
    # component that BEST fills the centre of the tile. Selecting by "largest" fails on a
    # shout balloon: its outward spikes make the surrounding between-spike ring LARGER than
    # the central body. Instead, among reasonably-sized non-border components, pick the one
    # whose centroid is nearest the tile centre (the body interior sits dead-centre; the
    # exterior / between-spike pockets wrap the edges). Robust for round AND spiky shapes.
    lbl, n = ndimage.label(transp)
    if n == 0:
        return tile_rgba, np.zeros_like(ink)
    H2, W2 = ink.shape
    border_labels = set(lbl[0, :]) | set(lbl[-1, :]) | set(lbl[:, 0]) | set(lbl[:, -1])
    border_labels.discard(0)
    sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
    objs = ndimage.find_objects(lbl)
    total = transp.sum()
    margin = max(2, int(0.01 * min(H2, W2)))
    # The bubble interior is an ENCLOSED blob whose bounding box stays clear of the tile
    # edges (the body sits inside the outline). The exterior / between-spike region, by
    # contrast, has a bbox that reaches the tile edges. So: among non-border components
    # large enough to matter, keep those whose bbox is inset from all 4 edges, and pick the
    # largest of those. (On a shout balloon this correctly rejects the bigger outer ring.)
    inset, anyc = [], []
    for i in range(n):
        lab = i + 1
        if lab in border_labels or sizes[i] < 0.02 * total:
            continue
        sl = objs[i]
        ys, xs = sl[0], sl[1]
        anyc.append((sizes[i], lab))
        if (ys.start >= margin and xs.start >= margin
                and ys.stop <= H2 - margin and xs.stop <= W2 - margin):
            inset.append((sizes[i], lab))
    pool = inset or anyc
    if not pool:                       # fallback: largest component overall
        pool = [(sizes[i], i + 1) for i in range(n)]
    best_label = max(pool)[1]
    interior = lbl == best_label       # enclosed transparent area = bubble inside
    fill = interior | ink              # white inside + ink footprint -> solid opaque bubble

    out = arr.copy()
    # paint the whole bubble footprint white & opaque first ...
    out[fill, 0] = 255
    out[fill, 1] = 255
    out[fill, 2] = 255
    out[fill, 3] = 255
    # ... then lay the original black outline back on top (where ink was dark)
    rgb = arr[:, :, :3].astype(np.int16).sum(axis=2)
    dark_ink = ink & (rgb < 360)       # dark outline pixels (sum of 3 channels < ~120 each)
    out[dark_ink, 0] = 0
    out[dark_ink, 1] = 0
    out[dark_ink, 2] = 0
    out[dark_ink, 3] = 255
    return Image.fromarray(out, "RGBA"), interior


# ---------------------------------------------------------------------------
# 4) locate the bubble INTERIOR (largest inscribed box) from an interior mask
# ---------------------------------------------------------------------------
def find_interior_bbox(interior_mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the largest axis-aligned rectangle fully inside `interior_mask` (the bubble
    inside, excluding outline)."""
    white = interior_mask
    if white.sum() == 0:
        return None
    # largest-rectangle-of-1s in a binary matrix (histogram method) -> robust interior box
    H, W = white.shape
    heights = np.zeros(W, dtype=np.int32)
    best = (0, 0, 0, 0, 0)  # area, x0, y0, x1, y1
    for row in range(H):
        heights = np.where(white[row], heights + 1, 0)
        # largest rectangle in this histogram row
        stack: list[int] = []
        i = 0
        hh = list(heights) + [0]
        while i < len(hh):
            if not stack or hh[i] >= hh[stack[-1]]:
                stack.append(i)
                i += 1
            else:
                top = stack.pop()
                h = hh[top]
                w = i if not stack else i - stack[-1] - 1
                area = h * w
                if area > best[0]:
                    left = (stack[-1] + 1) if stack else 0
                    best = (area, left, row - h + 1, i - 1, row)
        # end histogram
    if best[0] == 0:
        return None
    _, x0, y0, x1, y1 = best
    return (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# 5) fit + draw text inside an interior box with Pillow
# ---------------------------------------------------------------------------
def load_font(style: str, size: int) -> ImageFont.FreeTypeFont:
    name = "Bangers-Regular.ttf" if style == "shout" else "ComicNeue-Bold.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


def wrap_text(draw, text, font, max_w):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_text_in_box(canvas: Image.Image, box, text, style) -> dict:
    """Auto-size the largest font whose wrapped text fits within `box` (with padding),
    then draw centered. Returns fit diagnostics."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    pad = int(min(bw, bh) * 0.12)
    inner_w, inner_h = bw - 2 * pad, bh - 2 * pad
    draw = ImageDraw.Draw(canvas)
    chosen = None
    for fs in range(max(8, int(bh * 0.6)), 7, -2):
        font = load_font(style, fs)
        lines = wrap_text(draw, text, font, inner_w)
        line_h = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
        gap = int(line_h * 0.18)
        total_h = len(lines) * line_h + (len(lines) - 1) * gap
        widest = max((draw.textlength(ln, font=font) for ln in lines), default=0)
        if total_h <= inner_h and widest <= inner_w:
            chosen = (font, fs, lines, line_h, gap, total_h)
            break
    if chosen is None:
        font = load_font(style, 8)
        lines = wrap_text(draw, text, font, inner_w)
        line_h = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
        gap = int(line_h * 0.18)
        total_h = len(lines) * line_h + (len(lines) - 1) * gap
        chosen = (font, 8, lines, line_h, gap, total_h)
        fit_ok = False
    else:
        font, fs, lines, line_h, gap, total_h = chosen
        fit_ok = True
    # draw centered vertically + horizontally
    cy = y0 + pad + (inner_h - total_h) // 2
    asc = font.getbbox("Ag")[1]
    for ln in lines:
        w = draw.textlength(ln, font=font)
        cx = x0 + pad + (inner_w - w) // 2
        draw.text((cx, cy - asc), ln, font=font, fill=(0, 0, 0, 255))
        cy += line_h + gap
    return {"font_px": chosen[1], "lines": lines, "fit_ok": fit_ok}


# ---------------------------------------------------------------------------
# 6) art-preservation check: pixels outside the bubble bbox must be identical
# ---------------------------------------------------------------------------
def art_preservation(orig_path: Path, result: Image.Image, paste_box) -> dict:
    orig = Image.open(orig_path).convert("RGB")
    res = result.convert("RGB")
    o = np.asarray(orig).astype(np.int16)
    r = np.asarray(res).astype(np.int16)
    diff = np.abs(o - r).max(axis=2)  # per-pixel max channel diff
    px0, py0, px1, py1 = paste_box
    mask = np.ones(diff.shape, dtype=bool)
    mask[py0:py1, px0:px1] = False  # exclude the bubble region
    outside = diff[mask]
    changed = int((outside > 0).sum())
    changed_big = int((outside > 8).sum())
    return {
        "outside_pixels": int(outside.size),
        "outside_changed_any": changed,
        "outside_changed_gt8": changed_big,
        "pct_outside_changed": round(100 * changed / max(1, outside.size), 6),
        "pixel_identical_outside": changed == 0,
    }


def main() -> int:
    client = OpenAI()
    print(f"model={MODEL} tile_size={TILE_SIZE} quality={QUALITY} bg=transparent\n")

    # generate the distinct shapes we need (+ one extra for variety inspection)
    shapes_needed = sorted({j["shape"] for j in JOBS} | {"speech2"})
    print("== generating empty bubble tiles ==")
    tiles = {s: gen_tile(client, s) for s in shapes_needed}

    print("\n== alpha cleanliness ==")
    alpha = {s: alpha_report(p) for s, p in tiles.items()}
    for s, rep in alpha.items():
        print(f"  {s}: {rep}")

    print("\n== compositing + lettering ==")
    summary = []
    for job in JOBS:
        src = MANGA_DIR / f"panel_{job['panel']:04d}.png"
        if not src.exists():
            print(f"  !! missing {src}", file=sys.stderr)
            return 2
        panel = Image.open(src).convert("RGBA")
        PW, PH = panel.size

        # crop tile to tight bubble bbox, then FILL its (transparent) interior white
        tile = crop_to_bubble(tiles[job["shape"]])
        tile, _ = fill_interior_white(tile)

        # scale tile to fit the normalized target box
        tx0, ty0, tx1, ty1 = job["target"]
        box_w = int((tx1 - tx0) * PW)
        box_h = int((ty1 - ty0) * PH)
        # preserve aspect, fit inside target box
        scale = min(box_w / tile.width, box_h / tile.height)
        new_w = max(1, int(tile.width * scale))
        new_h = max(1, int(tile.height * scale))
        tile_r = tile.resize((new_w, new_h), Image.LANCZOS)

        # center tile within target box
        paste_x = int(tx0 * PW) + (box_w - new_w) // 2
        paste_y = int(ty0 * PH) + (box_h - new_h) // 2

        # composite (untouched panel + opaque white bubble overlay)
        composed = panel.copy()
        composed.alpha_composite(tile_r, (paste_x, paste_y))

        # recompute interior mask on the RESIZED, filled tile -> largest inscribed box
        arr_r = np.asarray(tile_r)
        a_r = arr_r[:, :, 3]
        rgb_r = arr_r[:, :, :3].astype(np.int16).sum(axis=2)
        interior_mask_r = (a_r > 200) & (rgb_r > 690)   # opaque + near-white interior
        interior = find_interior_bbox(interior_mask_r)
        if interior is None:
            print(f"  job {job['n']}: !! no interior found")
            interior_panel = None
            text_diag = {"fit_ok": False, "reason": "no interior"}
        else:
            ix0, iy0, ix1, iy1 = interior
            interior_panel = (paste_x + ix0, paste_y + iy0, paste_x + ix1, paste_y + iy1)
            text_diag = draw_text_in_box(composed, interior_panel, job["text"], job["shape"])

        paste_box = (paste_x, paste_y, paste_x + new_w, paste_y + new_h)
        art = art_preservation(src, composed, paste_box)

        out = OUT_DIR / f"hybrid_{job['n']}.png"
        composed.convert("RGB").save(out)
        rec = {
            "n": job["n"], "panel": job["panel"], "shape": job["shape"],
            "text": job["text"], "out": str(out),
            "tile_bbox_on_panel": paste_box,
            "interior_bbox_on_panel": interior_panel,
            "interior_found": interior is not None,
            "interior_frac_of_tile": (
                round(((interior[2]-interior[0])*(interior[3]-interior[1])) /
                      (new_w*new_h), 3) if interior else None),
            "text_fit": text_diag,
            "art_preservation": art,
        }
        summary.append(rec)
        print(f"  job {job['n']} panel {job['panel']} -> {out}")
        print(f"     interior_found={rec['interior_found']} "
              f"interior_frac={rec['interior_frac_of_tile']} "
              f"text_fit_ok={text_diag.get('fit_ok')} font_px={text_diag.get('font_px')}")
        print(f"     art: identical_outside={art['pixel_identical_outside']} "
              f"changed={art['outside_changed_any']}/{art['outside_pixels']}")

    (OUT_DIR / "hybrid_summary.json").write_text(
        json.dumps({"alpha": alpha, "jobs": summary}, indent=2, default=int))
    print(f"\nwrote {OUT_DIR/'hybrid_summary.json'}")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
