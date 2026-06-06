"""Client factory: the real OpenAI client, or a zero-spend MockClient for --dry-run.

The MockClient mimics the exact surface we use:
  - images.generate(...) / images.edit(...) -> obj with .data[0].b64_json and .usage
  - responses.parse(...)                    -> obj with .output_parsed and .usage
so the entire pipeline (including layout/lettering/export) runs end-to-end for $0.
"""
from __future__ import annotations

import base64
import io
import re
import textwrap
from types import SimpleNamespace
from typing import Any

from PIL import Image, ImageDraw


def build_client(dry_run: bool):
    if dry_run:
        return MockClient()
    from openai import OpenAI
    return OpenAI()


# ── helpers ────────────────────────────────────────────────────────────────
def _parse_size(size: str) -> tuple[int, int]:
    try:
        w, h = size.lower().split("x")
        return int(w), int(h)
    except Exception:
        return 1024, 1536


def _placeholder_png(size: str, label: str, prompt: str) -> bytes:
    w, h = _parse_size(size)
    img = Image.new("L", (w, h), 235)
    d = ImageDraw.Draw(img)
    d.rectangle([6, 6, w - 7, h - 7], outline=40, width=4)
    d.text((24, 20), f"[MOCK] {label}", fill=20)
    wrapped = textwrap.fill(prompt, width=max(20, w // 16))[:1400]
    d.multiline_text((24, 70), wrapped, fill=70, spacing=6)
    # a couple of grey blocks so mangapost/halftone have midtones to work with
    d.rectangle([w * 0.15, h * 0.45, w * 0.85, h * 0.6], fill=150)
    d.ellipse([w * 0.3, h * 0.68, w * 0.7, h * 0.9], fill=110)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _img_response(png: bytes) -> SimpleNamespace:
    return SimpleNamespace(
        data=[SimpleNamespace(b64_json=base64.b64encode(png).decode(), url=None)],
        usage=SimpleNamespace(input_tokens=0, output_tokens=0, total_tokens=0),
    )


# ── mock ───────────────────────────────────────────────────────────────────
class _MockImages:
    def generate(self, *, model: str, prompt: str, size: str = "1024x1536",
                 quality: str = "medium", **kw: Any) -> SimpleNamespace:
        return _img_response(_placeholder_png(size, "sheet", prompt))

    def edit(self, *, model: str, image: Any, prompt: str, size: str = "1024x1536",
             quality: str = "medium", **kw: Any) -> SimpleNamespace:
        nrefs = len(image) if isinstance(image, (list, tuple)) else 1
        return _img_response(_placeholder_png(size, f"panel (refs={nrefs})", prompt))


class _MockResponses:
    def parse(self, *, model: str, input: Any, text_format: Any,
              **kw: Any) -> SimpleNamespace:
        from .artifacts import CharacterRef, ChunkParse, DialogueLine, PanelSpec
        from .bible import get_character, roster_names

        text = _collect_text(input)
        names = roster_names()
        # this LN marks speech as `Speaker: [line]` (or `???: [line]`); one panel per line
        line_re = re.compile(r"^(\?\?\?|[A-Za-z][\w'’ .]{0,24}?)\s*:\s*\[(.+?)\]\s*$")
        lines_in = [l.strip() for l in text.split("\n") if l.strip()]
        panels: list[PanelSpec] = []
        for i, line in enumerate(lines_in[:8]):
            m = line_re.match(line)
            if m:
                c = get_character(m.group(1))
                present = [c.name] if c else []
                dlg = [DialogueLine(speaker=(c.name if c else None),
                                    text=m.group(2)[:160], style="speech")]
                action = f"{(c.name if c else 'a character')} speaking"
                shot = "close_up"
            else:
                present = []
                for n in names:
                    if re.search(rf"\b{re.escape(n)}\b", line, re.I):
                        cc = get_character(n)
                        if cc and cc.name not in present:
                            present.append(cc.name)
                dlg = []
                action = line[:120]
                shot = ["establishing", "medium", "wide"][i % 3]
            panels.append(PanelSpec(
                characters_present=present[:3],
                shot_type=shot,
                setting="(mock setting)",
                action=action,
                emotion="neutral",
                dialogue=dlg,
                speaker_side="right" if i % 2 else "left",
            ))
        if not panels:
            panels = [PanelSpec(action="(empty chunk)", setting="(mock)")]
        # FULL coverage: one CharacterRef per distinct name across all panels, so dry-run
        # exercises charsheet/script handling of non-roster characters too.
        cast: list[CharacterRef] = []
        seen: set[str] = set()
        for p in panels:
            for n in p.characters_present:
                if n in seen:
                    continue
                seen.add(n)
                c = get_character(n)
                descriptor = c.descriptor if c else f"a character named {n}"
                cast.append(CharacterRef(name=n, descriptor=descriptor))
        parsed = text_format(panels=panels, cast=cast) if text_format is ChunkParse else text_format()
        return SimpleNamespace(
            output_parsed=parsed,
            usage=SimpleNamespace(input_tokens=0, output_tokens=0, total_tokens=0),
        )

    def create(self, **kw: Any) -> SimpleNamespace:
        # Agentic reference search uses responses.create; return a clean empty result
        # under dry-run instead of AttributeError (which would fall back anyway).
        return SimpleNamespace(
            output=[],
            output_text="",
            usage=SimpleNamespace(input_tokens=0, output_tokens=0, total_tokens=0),
            id="mock",
        )


def _collect_text(input_messages: Any) -> str:
    if isinstance(input_messages, str):
        return input_messages
    user_parts: list[str] = []
    all_parts: list[str] = []
    for m in input_messages or []:
        role = m.get("role") if isinstance(m, dict) else None
        c = m.get("content") if isinstance(m, dict) else None
        texts: list[str] = []
        if isinstance(c, str):
            texts = [c]
        elif isinstance(c, list):
            texts = [p["text"] for p in c if isinstance(p, dict) and "text" in p]
        all_parts += texts
        if role == "user":
            user_parts += texts
    return "\n".join(user_parts or all_parts)  # prefer the user prose only


class MockClient:
    def __init__(self) -> None:
        self.images = _MockImages()
        self.responses = _MockResponses()
