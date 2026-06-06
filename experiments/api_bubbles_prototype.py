#!/usr/bin/env python3
"""STANDALONE prototype: OpenAI image-API speech bubbles (per-panel) vs current Pillow bubbles.

This file does NOT import or modify anything under ln2manga/stages/ or config. It is a
self-contained probe to evaluate whether generating manga speech bubbles via
client.images.edit on a *single finished B&W panel* (~1024x1536) is viable as an
alternative to the pipeline's Pillow letterer.

Why per-panel (not full page): editing the full composited 1488x2126 page through
gpt-image resamples it through ~1024-class sizes at the wrong aspect ratio and degrades
the whole page. A single 1024x1536 panel matches the model's native portrait ratio, so
only that one panel is round-tripped.

Run:  OPENAI_API_KEY=... python3 experiments/api_bubbles_prototype.py
Budget: 3 images at gpt-image-2 medium quality (~$0.10-0.12 each -> ~$0.30 total).
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

from openai import OpenAI

MANGA_DIR = Path("/home/jason/Desktop/ln2manga/data/cache/manga")
OUT_DIR = Path("/home/jason/Desktop/ln2manga/data/experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-image-2"
SIZE = "1024x1536"
QUALITY = "medium"
FMT = "png"

# Three panels chosen for variety from chapter-1 (real dialogue from panels.json).
# - close-up / face-dense   : panel 4  (5 faces packed, empty band top-center)
# - wide / sparse           : panel 19 (Subaru carrying Petra, big empty top-right)
# - multi-character / crowded: panel 20 (6 characters in a row, empty top-left/center)
JOBS = [
    {
        "n": 1,
        "panel": 4,
        "kind": "close-up (face-dense)",
        # panel 4 dialogue, speaker_side center, style speech
        "text": "You're outrageous.",
    },
    {
        "n": 2,
        "panel": 19,
        "kind": "wide (sparse)",
        # panel 19 first line, Petra, style shout
        "text": "Subaru!",
    },
    {
        "n": 3,
        "panel": 20,
        "kind": "multi-character (crowded)",
        # panel 20 first lines, Petra, style speech
        "text": "Welcome, guests. I am Petra Leyte, a maid of this residence.",
    },
]


def build_prompt(text: str) -> str:
    """A bubble-only edit prompt. We instruct the model to ADD one clean white manga
    speech bubble with the EXACT text, in EMPTY space, never over a face, and to leave
    the rest of the artwork untouched (no restyle, no redraw)."""
    return (
        "Add ONE clean white manga speech bubble to this black-and-white manga panel. "
        f'The bubble must contain exactly this text, spelled correctly: "{text}". '
        "Requirements:\n"
        "- Place the bubble in an EMPTY area of the panel (blank/white background space). "
        "Do NOT cover any character's face, eyes, or head.\n"
        "- Bubble: smooth white oval/rounded outline with a thin solid black border and a "
        "small pointed tail aimed toward the speaking character.\n"
        "- Lettering: crisp black uppercase comic/manga style, centered, fully legible, "
        "correctly spelled, no extra or garbled letters, no other text anywhere.\n"
        "- Do NOT change, restyle, redraw, recolor, or degrade any existing artwork, "
        "linework, screentone, or characters. Keep the original black-and-white panel "
        "exactly as-is except for the single added bubble.\n"
        "- Output the full panel at the same size and framing."
    )


def main() -> int:
    client = OpenAI()
    print(f"model={MODEL} size={SIZE} quality={QUALITY} fmt={FMT}\n")
    for job in JOBS:
        src = MANGA_DIR / f"panel_{job['panel']:04d}.png"
        if not src.exists():
            print(f"  !! missing source {src}", file=sys.stderr)
            return 2
        prompt = build_prompt(job["text"])
        print(f"[api_{job['n']}] panel {job['panel']} ({job['kind']}) text={job['text']!r}")
        with src.open("rb") as fh:
            # gpt-image-2 auto-applies high input fidelity and REJECTS input_fidelity,
            # so we do not pass it (mirrors the pipeline's imagegen.edit_image gate).
            resp = client.images.edit(
                model=MODEL,
                image=(src.name, fh.read(), "image/png"),
                prompt=prompt,
                size=SIZE,
                quality=QUALITY,
                output_format=FMT,
                n=1,
            )
        png = base64.b64decode(resp.data[0].b64_json)
        out = OUT_DIR / f"api_{job['n']}.png"
        out.write_bytes(png)
        usage = getattr(resp, "usage", None)
        u = {k: getattr(usage, k, None) for k in ("input_tokens", "output_tokens", "total_tokens")} if usage else {}
        print(f"           -> {out}  ({len(png)} bytes)  usage={u}")
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
