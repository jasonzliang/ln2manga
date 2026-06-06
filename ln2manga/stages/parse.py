"""Stage 2 — parse (LLM): prose -> ordered List[PanelSpec] via Structured Outputs.

Uses client.responses.parse(text_format=ChunkParse) on gpt-5.4-mini (verified working).
Per-chunk results are cached on disk (keyed by model+text) so re-runs never re-pay.
"""
from __future__ import annotations

import json

from ..artifacts import ChunkParse, PanelSpec
from ..bible import active_roster
from ..cache import Cache, cache_key
from ..config import Settings
from ..cost import CostTracker
from ..retry import with_retry

SYSTEM = """You are a manga storyboard director. Convert the given light-novel passage into an \
ordered sequence of manga panels. Each panel is ONE clear visual moment.

For every panel provide:
- characters_present: names of ALL characters VISIBLE in the panel (most important first). Use the \
roster's canonical name when a character matches the roster; for any OTHER visible character (a side \
character, a maid, a named creature like a dragon, etc.) use their proper name if identifiable, else \
a short label. Empty for scenery/no-character panels. Only include characters actually SHOWN in the \
panel, never ones merely mentioned in dialogue.
- shot_type, angle, composition: cinematographic framing.
- setting: the location/background, concisely.
- action: a concise VISUAL description of what is happening (what an artist would draw).
- emotion: the dominant emotion.
- dialogue: spoken lines. The source marks speech as `Speaker: [line]`. Put the spoken text \
(without brackets) in `text`, map `Speaker` to the character's name, or null if it is `???` or \
pure narration. style = narration/shout/thought/speech. Keep lines short; split long speeches. \
Keep each panel to AT MOST about 3 short dialogue lines.
- speaker_side: where the speaker is in the frame (for the bubble tail).

ALSO fill `cast`: EVERY character that visibly appears anywhere in this passage, each with a concise \
VISUAL descriptor (hair, eyes, outfit, distinctive features). This is REQUIRED — especially for \
characters NOT in the roster — so they can be drawn consistently across panels.

Aim for roughly one panel per important beat. If a conversation has more back-and-forth than about \
3 lines, split it across a few consecutive panels with varied shots — alternate the speaker and cut \
to reaction close-ups — so no panel is overcrowded with text. Be economical: only split when the \
dialogue would otherwise overflow, and never create near-empty, redundant, or filler panels. Do not \
invent events. Prefer close-ups for emotional dialogue and wide/establishing shots for new \
locations."""


def _roster_block(settings: Settings | None = None) -> str:
    lines = []
    for c in active_roster(settings).values():
        alias = f" (aka {', '.join(c.aliases)})" if c.aliases else ""
        lines.append(f"- {c.name}{alias}")
    return ("Known characters with established designs (use these canonical names when a character "
            "matches; other visible characters are fine too):\n" + "\n".join(lines))


def _split_oversize(p: str, chunk_chars: int) -> list[str]:
    """Slice a single paragraph longer than chunk_chars into <= chunk_chars pieces,
    cutting on the nearest preceding whitespace within each window to avoid mid-word
    cuts; fall back to a hard slice when no whitespace is found."""
    pieces: list[str] = []
    while len(p) > chunk_chars:
        window = p[:chunk_chars]
        cut = window.rfind(" ")
        if cut <= 0:                                  # no usable break -> hard slice
            cut = chunk_chars
        pieces.append(p[:cut])
        p = p[cut:].lstrip(" ")
    if p:
        pieces.append(p)
    return pieces


def chunk_paragraphs(paragraphs: list[str], scene_breaks: list[int],
                     chunk_chars: int) -> list[str]:
    breaks = set(scene_breaks)
    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for i, p in enumerate(paragraphs):
        if cur and (i in breaks or size + len(p) > chunk_chars):
            chunks.append("\n".join(cur))
            cur, size = [], 0
        # bound a runaway single paragraph so no chunk ever exceeds chunk_chars
        if len(p) > chunk_chars:
            if cur:                                   # flush any pending chunk first
                chunks.append("\n".join(cur))
                cur, size = [], 0
            pieces = _split_oversize(p, chunk_chars)
            chunks.extend(pieces[:-1])                # whole pieces stand alone
            p = pieces[-1]                            # tail re-enters normal accumulation
        cur.append(p)
        size += len(p) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _parse_chunk(client, settings: Settings, tracker: CostTracker,
                 cache: Cache, text: str) -> ChunkParse:
    model = settings.models["text"]
    # Fold in EVERY output-determining input (not just model+text): the SYSTEM prompt,
    # the roster block, and the reasoning/token settings all shape the parse, so a change
    # to any of them must invalidate the persistent disk cache (mirrors imagegen/bubbles).
    key = cache_key({
        "op": "parse",
        "model": model,
        "text": text,
        "system": SYSTEM,
        "roster": _roster_block(settings),
        "effort": settings.parse.get("reasoning_effort", "low"),
        "max_out": int(settings.parse["max_output_tokens"]),
    })
    hit = cache.get("parse", key, ext="json")
    if hit is not None:
        return ChunkParse.model_validate_json(hit.decode())

    # text cost is normally recorded from real usage; the pre-check uses a small estimate,
    # which we also keep as a fallback charge if the response carries no usage object (#13).
    est = tracker.estimate_text(model, 4000, settings.parse["max_output_tokens"])
    tracker.check(est)

    call = with_retry(client.responses.parse)
    resp = call(
        model=model,
        reasoning={"effort": settings.parse.get("reasoning_effort", "low")},
        max_output_tokens=int(settings.parse["max_output_tokens"]),
        input=[
            {"role": "system", "content": SYSTEM + "\n\n" + _roster_block(settings)},
            {"role": "user", "content": text},
        ],
        text_format=ChunkParse,
    )
    parsed: ChunkParse | None = getattr(resp, "output_parsed", None)
    if parsed is None:
        raise RuntimeError(
            f"parse produced no output for chunk starting {text[:60]!r}... "
            f"(model refusal or output truncated at "
            f"max_output_tokens={settings.parse['max_output_tokens']}). "
            "Raise parse.max_output_tokens or lower parse.chunk_chars; "
            "if this is a moderation refusal, edit the passage.")
    # Always charge for a billed call before caching, so spend can't slip below real cost
    # and breach the cap. Prefer real usage; fall back to the pre-call estimate otherwise.
    usage = getattr(resp, "usage", None)
    if usage is not None:
        tracker.record_text_from_usage(model, usage)
    else:
        tracker.record("text", model, est, {"note": "usage_missing_fallback_estimate"})
    cache.put("parse", key, parsed.model_dump_json(indent=2).encode(), ext="json")
    return parsed


def run(client, settings: Settings, tracker: CostTracker, cache: Cache,
        chapter) -> list[PanelSpec]:
    from ..bible import get_character

    chunks = chunk_paragraphs(chapter.paragraphs, chapter.scene_breaks,
                              int(settings.parse["chunk_chars"]))
    panels: list[PanelSpec] = []
    cast: dict[str, str] = {}                       # character name -> visual descriptor
    for text in chunks:
        result = _parse_chunk(client, settings, tracker, cache, text)
        for cr in result.cast:
            c = get_character(cr.name, settings)
            key = (c.name if c else (cr.name or "").strip())
            if not key:
                continue
            if c:                                   # roster character -> frozen bible descriptor
                cast[key] = c.descriptor
            elif len(cr.descriptor or "") > len(cast.get(key, "")):
                cast[key] = cr.descriptor or key
        for spec in result.panels:
            norm: list[str] = []
            for nm in spec.characters_present:       # canonicalize roster aliases
                c = get_character(nm, settings)
                name = c.name if c else (nm or "").strip()
                if name and name not in norm:
                    norm.append(name)
            spec.characters_present = norm
            spec.panel_number = len(panels) + 1
            panels.append(spec)

    # guarantee every VISIBLE character has a descriptor (so it can be sheeted)
    for spec in panels:
        for nm in spec.characters_present:
            if nm not in cast:
                c = get_character(nm, settings)
                cast[nm] = c.descriptor if c else nm

    a = settings.artifacts_dir
    # Cache stability: a discovered (non-roster) character's descriptor comes from the
    # per-parse LLM `cast`, which drifts slightly on every re-parse. That descriptor keys the
    # charsheet prompt, so any drift regenerates the sheet AND cascades into regenerating every
    # panel the character appears in (real, costly waste). Make re-parses idempotent by reusing
    # the previously-persisted descriptor whenever a discovered character reappears; only a
    # genuinely new (never-seen) character takes the fresh LLM descriptor. Roster characters are
    # unaffected (they already carry the frozen bible descriptor). On a first run there is no
    # prior file, so the new descriptors are used as-is. Order is preserved (in-place updates).
    prev_path = a / f"chapter-{chapter.number}.cast.json"
    if prev_path.exists():
        try:
            prev_cast = json.loads(prev_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            prev_cast = {}                          # corrupt/unreadable prior -> ignore, don't crash
        if isinstance(prev_cast, dict):
            for nm in cast:
                if get_character(nm, settings):     # roster: keep frozen bible descriptor
                    continue
                prev = prev_cast.get(nm)
                if isinstance(prev, str) and prev:  # seen before -> reuse stable descriptor
                    cast[nm] = prev

    (a / f"chapter-{chapter.number}.panels.json").write_text(
        json.dumps([p.model_dump() for p in panels], indent=2, ensure_ascii=False), encoding="utf-8")
    (a / f"chapter-{chapter.number}.cast.json").write_text(
        json.dumps(cast, indent=2, ensure_ascii=False), encoding="utf-8")
    return panels
