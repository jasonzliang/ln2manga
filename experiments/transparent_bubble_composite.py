#!/usr/bin/env python3
"""STANDALONE prototype: transparent-background speech bubble TILE generated via the
OpenAI image API, then ALPHA-COMPOSITED onto an UNTOUCHED B&W manga panel.

This is the "narrow hybrid" the prior api_bubbles_prototype.py / FINDINGS.md flagged as
the most promising UNTESTED idea: the panel ART is NEVER sent to the API (so the model
cannot degrade it). The API only renders a bubble + lettering on a transparent canvas;
we paste that tile onto an empty, face-avoiding corner of the original panel locally.

Nothing under ln2manga/stages/ or config/ is imported or modified. Outputs go only to
data/experiments/.

Run:  OPENAI_API_KEY=... python3 experiments/transparent_bubble_composite.py
Budget: 3 images at gpt-image-2 medium quality, ~$0.10 each -> ~$0.30 total.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

from openai import OpenAI
from PIL import Image
import numpy as np

MANGA_DIR = Path("/home/jason/Desktop/ln2manga/data/cache/manga")
OUT_DIR = Path("/home/jason/Desktop/ln2manga/data/experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# NOTE: gpt-image-2 (the pipeline default) REJECTS background="transparent" with a 400
# "Transparent background is not supported for this model." Only gpt-image-1 / -1.5 / -mini
# support transparent generation, so the transparent-tile approach requires switching the
# image model away from the pipeline default. We use gpt-image-1 here (confirmed working).
MODEL = "gpt-image-1"
SIZE = "1024x1024"
QUALITY = "medium"
FMT = "png"

# Panels chosen for: short clean dialogue + a verified empty, face-free corner
# (whitefrac measured in transparent_bubble_composite analysis pass).
# placement box = (x, y, w, h) in the 1024x1536 panel, where the tile is fitted.
JOBS = [
    {
        "n": 1,
        "panel": 4,
        "kind": "close-up (5 faces) / empty top-center",
        "text": "You're outrageous.",          # panel 4, single line
        "box": (300, 5, 420, 470),               # top-center white gap (99.5% white)
    },
    {
        "n": 2,
        "panel": 17,
        "kind": "close-up / empty top-left circle",
        "text": "It's not just a simple stop.",  # panel 17, single line
        "box": (10, 60, 430, 470),               # top-left light circle (98% white)
    },
    {
        "n": 3,
        "panel": 1,
        "kind": "wide / empty top-center",
        # panel 1's real line contains the word 'suicide', which the API OUTPUT moderation
        # blocks (safety_violations=[abuse]) -> a real finding: verbatim dialogue can be
        # refused. We keep the original panel-1 placement but use a safe nearby line that
        # exercises an apostrophe + comma; the moderation result is reported separately.
        "text": "Nevertheless, it's strange.",   # panel 2 line, rendered in panel-1 corner
        "box": (300, 5, 440, 430),               # top-center white gap (94% white)
    },
]

# Extra: also attempt the literal panel-1 line to DOCUMENT the moderation behavior.
MODERATION_PROBE = {
    "panel": 1,
    "text": "It feels like we're lining up for suicide.",
}


def build_prompt(text: str) -> str:
    """Per the task spec: a single clean B&W manga bubble with EXACT text on a plain
    transparent background. No source art is ever sent."""
    return (
        "a single clean black-and-white manga speech bubble (white fill, bold black "
        "outline, a small tail) containing exactly the text: "
        f"'{text}'. Nothing else, plain transparent background."
    )


def gen_tile(client: OpenAI, text: str) -> Image.Image:
    resp = client.images.generate(
        model=MODEL,
        prompt=build_prompt(text),
        background="transparent",
        output_format=FMT,
        size=SIZE,
        quality=QUALITY,
        n=1,
    )
    png = base64.b64decode(resp.data[0].b64_json)
    raw = OUT_DIR / "_raw_tile.png"  # overwritten per job; kept only transiently
    usage = getattr(resp, "usage", None)
    u = {k: getattr(usage, k, None) for k in ("input_tokens", "output_tokens", "total_tokens")} if usage else {}
    img = Image.open(__import__("io").BytesIO(png)).convert("RGBA")
    return img, u, len(png)


def autocrop_alpha(tile: Image.Image) -> Image.Image:
    """Trim fully-transparent margins so the bubble fills the placement box."""
    a = np.asarray(tile)[:, :, 3]
    ys, xs = np.where(a > 8)
    if len(xs) == 0:
        return tile
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    return tile.crop((int(x0), int(y0), int(x1) + 1, int(y1) + 1))


def fit_into_box(tile: Image.Image, box):
    bx, by, bw, bh = box
    tw, th = tile.size
    scale = min(bw / tw, bh / th)
    nw, nh = max(1, int(tw * scale)), max(1, int(th * scale))
    tile_r = tile.resize((nw, nh), Image.LANCZOS)
    # center inside the box
    px = bx + (bw - nw) // 2
    py = by + (bh - nh) // 2
    return tile_r, (px, py, px + nw, py + nh)


def analyze(orig_L: Image.Image, comp_RGB: Image.Image, paste_bbox, alpha_mask_full):
    """Return art-preservation metrics OUTSIDE the pasted bubble's *opaque* region.

    We compare the composite vs the original panel. Inside the bubble we expect change
    (that's the point); everywhere the tile's alpha was ~0 the pixels must be identical.
    """
    o = np.asarray(orig_L.convert("RGB")).astype(np.int16)
    c = np.asarray(comp_RGB.convert("RGB")).astype(np.int16)
    diff = np.abs(o - c).max(axis=2)  # per-pixel max channel diff

    # mask of pixels actually painted (tile alpha > 0 at full panel resolution)
    painted = alpha_mask_full > 0

    outside = diff[~painted]
    inside = diff[painted]

    px0, py0, px1, py1 = paste_bbox
    # also report the rectangular bbox view (pixels outside the bbox rectangle)
    H, W = diff.shape
    rect = np.zeros((H, W), dtype=bool)
    rect[py0:py1, px0:px1] = True
    outside_rect = diff[~rect]

    return {
        "outside_painted_max": int(outside.max()) if outside.size else 0,
        "outside_painted_changed_px": int((outside > 0).sum()),
        "outside_painted_pct_changed": float((outside > 0).mean() * 100) if outside.size else 0.0,
        "outside_bbox_rect_max": int(outside_rect.max()) if outside_rect.size else 0,
        "outside_bbox_rect_changed_px": int((outside_rect > 0).sum()),
        "outside_bbox_rect_pct_changed": float((outside_rect > 0).mean() * 100) if outside_rect.size else 0.0,
        "painted_px": int(painted.sum()),
        "paste_bbox": [int(v) for v in paste_bbox],
        "panel_px": int(H * W),
    }


def alpha_quality(tile_cropped: Image.Image):
    """Crispness / halo / interior-opacity diagnostics on the autocropped tile."""
    arr = np.asarray(tile_cropped)
    a = arr[:, :, 3]
    total = a.size
    fully_opaque = (a == 255).mean() * 100
    fully_transp = (a == 0).mean() * 100
    semi = ((a > 0) & (a < 255)).mean() * 100  # anti-alias / halo band
    # interior opacity: of the opaque pixels, how white is the RGB (true white fill?)
    opaque = a > 200
    rgb = arr[:, :, :3]
    if opaque.any():
        interior = rgb[opaque]
        interior_mean = interior.mean(axis=0).round(1).tolist()
        interior_white_frac = float((interior.min(axis=1) > 235).mean() * 100)
    else:
        interior_mean = [0, 0, 0]
        interior_white_frac = 0.0
    return {
        "tile_size": list(tile_cropped.size),
        "fully_opaque_pct": round(fully_opaque, 2),
        "fully_transparent_pct": round(fully_transp, 2),
        "semi_transparent_pct": round(semi, 2),
        "interior_opaque_rgb_mean": interior_mean,
        "interior_opaque_white_pct": round(interior_white_frac, 2),
    }


def main() -> int:
    client = OpenAI()
    print(f"model={MODEL} size={SIZE} quality={QUALITY} bg=transparent\n")
    report = []
    for job in JOBS:
        src = MANGA_DIR / f"panel_{job['panel']:04d}.png"
        if not src.exists():
            print(f"  !! missing source {src}", file=sys.stderr)
            return 2
        print(f"[translit_{job['n']}] panel {job['panel']} ({job['kind']}) text={job['text']!r}")

        try:
            tile, usage, nbytes = gen_tile(client, job["text"])
        except Exception as e:  # moderation / transient API errors -> record + skip
            print(f"   !! generation failed: {type(e).__name__}: {str(e)[:160]}\n")
            report.append({"n": job["n"], "panel": job["panel"], "text": job["text"],
                           "error": f"{type(e).__name__}: {str(e)[:200]}"})
            continue
        tile.save(OUT_DIR / f"translit_{job['n']}_tile_raw.png")
        tile_c = autocrop_alpha(tile)
        tile_c.save(OUT_DIR / f"translit_{job['n']}_tile_cropped.png")

        orig = Image.open(src)  # untouched panel, mode "L"
        canvas = orig.convert("RGBA")
        tile_r, paste_bbox = fit_into_box(tile_c, job["box"])

        # full-resolution alpha mask of where the tile was painted
        alpha_full = np.zeros((canvas.size[1], canvas.size[0]), dtype=np.uint8)
        ta = np.asarray(tile_r)[:, :, 3]
        px0, py0, px1, py1 = paste_bbox
        alpha_full[py0:py1, px0:px1] = ta

        canvas.alpha_composite(tile_r, dest=(px0, py0))
        comp = canvas.convert("RGB")
        out = OUT_DIR / f"translit_{job['n']}.png"
        comp.save(out)

        art = analyze(orig, comp, paste_bbox, alpha_full)
        aq = alpha_quality(tile_c)
        rec = {
            "n": job["n"], "panel": job["panel"], "text": job["text"],
            "out": str(out), "tile_bytes": nbytes, "usage": usage,
            "art_preservation": art, "alpha_quality": aq,
        }
        report.append(rec)
        print(f"   -> {out}")
        print(f"      tile_raw={tile.size} cropped={tile_c.size} paste_bbox={paste_bbox}")
        print(f"      ART outside painted: max_diff={art['outside_painted_max']} "
              f"changed={art['outside_painted_pct_changed']:.4f}% "
              f"({art['outside_painted_changed_px']} px)")
        print(f"      ART outside bbox-rect: max_diff={art['outside_bbox_rect_max']} "
              f"changed={art['outside_bbox_rect_pct_changed']:.4f}%")
        print(f"      ALPHA opaque={aq['fully_opaque_pct']}% transp={aq['fully_transparent_pct']}% "
              f"semi(halo)={aq['semi_transparent_pct']}% "
              f"interior_white={aq['interior_opaque_white_pct']}% rgb={aq['interior_opaque_rgb_mean']}")
        print(f"      usage={usage}\n")

    # Document the moderation behavior on the real (blocked) panel-1 line.
    print(f"[moderation_probe] panel {MODERATION_PROBE['panel']} text={MODERATION_PROBE['text']!r}")
    try:
        gen_tile(client, MODERATION_PROBE["text"])
        mod_result = "ALLOWED (no moderation block)"
    except Exception as e:
        mod_result = f"{type(e).__name__}: {str(e)[:200]}"
    print(f"   moderation_probe result: {mod_result}\n")
    report.append({"moderation_probe": MODERATION_PROBE, "result": mod_result})

    (OUT_DIR / "translit_report.json").write_text(json.dumps(report, indent=2))
    print(f"report -> {OUT_DIR/'translit_report.json'}")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
