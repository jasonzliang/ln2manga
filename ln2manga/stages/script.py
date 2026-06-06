"""Stage 3 — script: turn each PanelSpec into a natural-language image prompt.

Pure Python (no API). gpt-image models prefer descriptive natural language over tag soup.
Per-character descriptors are injected from the chapter cast (roster characters use their frozen
bible descriptor; discovered characters use the parser-recorded descriptor), and every prompt
ends with the global style block that forbids in-image text.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..artifacts import PanelPrompt, PanelSpec
from ..bible import active_style, descriptor_for
from ..config import Settings

_SHOT = {
    "establishing": "wide establishing shot of the location",
    "wide": "wide shot",
    "medium": "medium shot from the waist up",
    "close_up": "close-up on the face",
    "extreme_close_up": "extreme close-up",
    "insert": "insert detail shot",
}


def build_prompt(spec: PanelSpec, cast: dict[str, str] | None = None,
                 settings: Settings | None = None) -> PanelPrompt:
    present: list[str] = []
    for n in spec.characters_present:
        if n and n not in present:
            present.append(n)

    parts: list[str] = [_SHOT.get(spec.shot_type, "medium shot") + "."]
    if present:
        who = "; ".join(descriptor_for(n, cast, settings) for n in present)
        parts.append(f"Characters in frame: {who}.")
    if spec.action:
        parts.append(f"Action: {spec.action}.")
    if spec.setting:
        parts.append(f"Setting: {spec.setting}.")
    if spec.emotion and spec.emotion != "neutral":
        parts.append(f"Mood: {spec.emotion}.")
    if spec.angle and spec.angle != "eye-level":
        parts.append(f"Camera angle: {spec.angle}.")
    parts.append("Leave clear empty space (sky, wall, or negative space) for speech bubbles.")
    parts.append(active_style(settings))
    parts.append("NO letters, numbers, logos, brand marks or insignia on clothing, "
                 "accessories or any object.")

    return PanelPrompt(panel_number=spec.panel_number, prompt=" ".join(parts), ref_sheets=present)


def _load_cast(settings: Settings, chapter_number: int) -> dict[str, str]:
    p = settings.artifacts_dir / f"chapter-{chapter_number}.cast.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        return {}


def run(settings: Settings, specs: list[PanelSpec], chapter_number: int) -> list[PanelPrompt]:
    cast = _load_cast(settings, chapter_number)
    prompts = [build_prompt(s, cast, settings) for s in specs]
    out = settings.artifacts_dir / f"chapter-{chapter_number}.prompts.json"
    out.write_text(json.dumps([p.model_dump() for p in prompts], indent=2,
                              ensure_ascii=False), encoding="utf-8")
    return prompts
