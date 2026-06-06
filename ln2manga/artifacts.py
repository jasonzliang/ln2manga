"""Typed data contracts that flow between pipeline stages, plus JSON load/save helpers.

Every stage reads and writes these models as JSON under data/artifacts/, so any stage
can be re-run independently from disk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar

from pydantic import BaseModel, Field

ShotType = Literal[
    "establishing", "wide", "medium", "close_up", "extreme_close_up", "insert"
]
Emotion = Literal[
    "neutral", "happy", "sad", "angry", "shocked", "afraid",
    "determined", "embarrassed", "pained", "smug", "serious",
]
BubbleStyle = Literal["speech", "thought", "shout", "narration"]
SpeakerSide = Literal["left", "right", "center", "none"]


# ── Stage 1: scrape ────────────────────────────────────────────────────────
class Chapter(BaseModel):
    arc: str
    number: int
    title: str
    url: str
    paragraphs: list[str]
    scene_breaks: list[int] = Field(default_factory=list)  # paragraph indices of ※ rows


# ── Stage 2: parse (the LLM Structured-Output target) ──────────────────────
class DialogueLine(BaseModel):
    speaker: Optional[str] = None       # roster name, or null for narration
    text: str
    style: BubbleStyle = "speech"


class PanelSpec(BaseModel):
    panel_number: int = 0               # global, assigned after parsing
    characters_present: list[str] = Field(default_factory=list)  # order = ref priority
    shot_type: ShotType = "medium"
    angle: str = "eye-level"
    composition: str = "centered"
    setting: str = ""                   # background / location
    action: str = ""                    # what is happening
    emotion: Emotion = "neutral"
    dialogue: list[DialogueLine] = Field(default_factory=list)
    speaker_side: SpeakerSide = "center"  # where the speaker is -> bubble tail direction


class CharacterRef(BaseModel):
    """A character that visibly appears, with a concise visual descriptor (so even non-roster
    characters can be drawn consistently)."""
    name: str
    descriptor: str = ""


class ChunkParse(BaseModel):
    """What the LLM returns for one prose chunk."""
    panels: list[PanelSpec]
    cast: list[CharacterRef] = Field(default_factory=list)   # every character visible in the chunk


# ── Stage 3: script ────────────────────────────────────────────────────────
class PanelPrompt(BaseModel):
    panel_number: int
    prompt: str
    ref_sheets: list[str] = Field(default_factory=list)  # paths, identity-critical first


# ── Stage 7: layout ────────────────────────────────────────────────────────
class Rect(BaseModel):
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0


class PageLayout(BaseModel):
    page_number: int
    template: str
    panel_numbers: list[int]            # global panel ids in RTL reading order
    rects: list[Rect]                   # same order as panel_numbers
    image_path: str = ""                # rendered (un-lettered) page png


# ── JSON helpers ───────────────────────────────────────────────────────────
_T = TypeVar("_T", bound=BaseModel)


def save_model(obj: BaseModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(obj.model_dump_json(indent=2), encoding="utf-8")


def load_model(model: Type[_T], path: str | Path) -> _T:
    return model.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_models(objs: list[BaseModel], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [o.model_dump() for o in objs]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_models(model: Type[_T], path: str | Path) -> list[_T]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [model.model_validate(d) for d in data]
