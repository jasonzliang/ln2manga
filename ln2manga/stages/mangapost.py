"""Stage 6 — mangapost: deterministically enforce a black & white manga look.

gpt-image tends to return soft/grayscale or faintly tinted art; this pass makes it read as
inked manga: grayscale -> autocontrast -> posterize to a few tones -> ordered-dither
screentone on midtones -> multiply an edge map back in for inked outlines. Pure Pillow/numpy.
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageOps

from ..config import Settings

_BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
                   dtype=np.float32)


def mangaize(img: Image.Image, *, tones: int = 0, halftone: bool = False,
             ink_lines: bool = False, halftone_cell: int = 4, denoise: bool = False,
             ink_threshold: int = 105) -> Image.Image:
    """Default: CLEAN manga-style GREYSCALE.

    The panels already come out as black-and-white manga art from the image model, so the default
    just converts to greyscale + light autocontrast — NO posterize, halftone screentone, or edge
    ink. Those add a print-style dot/grain look that is wrong for on-screen viewing, so they are
    OPT-IN: set tones>0 to posterize, halftone:true for screentone, ink_lines:true for edge ink.
    """
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g, cutoff=1)
    if denoise:
        g = g.filter(ImageFilter.MedianFilter(3))
    if not (int(tones) or halftone or ink_lines):
        return g                                   # clean greyscale (default) — no grain
    arr = np.asarray(g).astype(np.float32)
    arr = 255.0 * np.power(np.clip(arr / 255.0, 0, 1), 0.7)   # gamma<1 lifts midtones toward white
    h, w = arr.shape

    levels = max(2, int(tones))
    poster = np.round(arr / 255.0 * (levels - 1)) / (levels - 1) * 255.0

    if halftone:
        cell = max(1, int(halftone_cell))
        # Enlarge each Bayer dot to a `cell`x`cell` block so screentone survives on-page downscale.
        thresh = (np.kron(_BAYER4, np.ones((cell, cell), dtype=np.float32)) + 0.5) / 16.0 * 255.0
        bh, bw = thresh.shape
        tiled = np.tile(thresh, (h // bh + 1, w // bw + 1))[:h, :w]
        dither = np.where(arr > tiled, 255.0, 0.0)
        midtone = (arr > 95) & (arr < 175)        # screentone ONLY true mid-greys
        poster = np.where(midtone, dither, poster)

    poster = np.where(arr >= 205, 255.0, poster)  # keep highlights as clean white paper
    out = Image.fromarray(np.clip(poster, 0, 255).astype(np.uint8), "L")

    if ink_lines:
        # Median-filter the edge map (drops lone-pixel speckle), then zero the outer 2px (FIND_EDGES
        # leaves a bright border ring -> black frame). Thin ink: high threshold, NO dilation, so only
        # strong contours are inked (was thick MaxFilter+lowthr).
        edges = g.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MedianFilter(3))
        ea = np.asarray(edges).copy()
        ea[:2, :] = 0
        ea[-2:, :] = 0
        ea[:, :2] = 0
        ea[:, -2:] = 0
        lines = ImageOps.invert(
            Image.fromarray(ea, "L").point(lambda v: 255 if v > ink_threshold else 0))
        out = ImageChops.multiply(out, lines)

    return out.convert("L")


def run(settings: Settings, panel_manifest: list[dict], chapter_number: int) -> list[dict]:
    cfg = settings.mangapost
    out_manifest: list[dict] = []
    for item in panel_manifest:
        img = Image.open(item["path"]).convert("RGB")
        bw = mangaize(img, tones=int(cfg.get("tones", 5)),
                      halftone=bool(cfg.get("halftone", True)),
                      ink_lines=bool(cfg.get("ink_lines", True)),
                      halftone_cell=int(cfg.get("halftone_cell", 4)),
                      denoise=bool(cfg.get("denoise", True)),
                      ink_threshold=int(cfg.get("ink_threshold", 105)))
        out_path = settings.cache_dir("manga") / f"panel_{item['panel_number']:04d}.png"
        bw.save(out_path)
        out_manifest.append({"panel_number": item["panel_number"], "path": str(out_path)})
    mpath = settings.artifacts_dir / f"chapter-{chapter_number}.mangaimgs.json"
    mpath.write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")
    return out_manifest
