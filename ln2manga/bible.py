"""Character bible — work-agnostic.

The DEFAULT roster + art style below describe Re:Zero Arc 6 (Pleiades Watchtower), but a
different creative work can be targeted entirely from config: set `settings.bible.roster`
(or `settings.bible.roster_file`) and `settings.bible.global_style` and the same pipeline
draws that work instead. When no `bible` config is present the built-in Re:Zero default is
used, so default behavior is unchanged.

Descriptors are FROZEN strings injected verbatim into every prompt — never paraphrase, or
faces/proportions drift between panels. Each character gets ONE cached reference sheet that
is passed into every panel generation.
"""
from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
from pydantic import BaseModel

from .config import PROJECT_ROOT

# Appended to every image prompt. Keeps art monochrome and TEXT-FREE (we letter in code).
# This is the DEFAULT style; `settings.bible.global_style` overrides it everywhere.
GLOBAL_STYLE = (
    "Black-and-white Japanese manga art, clean confident ink linework, screentone shading, "
    "high contrast, expressive faces, dynamic composition, single comic panel. "
    "Absolutely NO text, NO speech bubbles, NO captions, NO sound-effect lettering, "
    "NO signature, NO watermark, NO borders."
)


class Character(BaseModel):
    name: str
    aliases: list[str] = []
    descriptor: str
    sheet_path: str | None = None
    # Optional REAL reference image for this character, set in code (a local file path or an
    # http(s) URL). Opt-in: only used when references.enabled is true in config. Config-level
    # sources (settings.references["sources"] / the external_file map) take priority over this.
    ref_image: str | None = None


# Built-in DEFAULT roster (Re:Zero). Keyed by lowercased name. Used when `settings.bible` is
# absent/empty (and by the no-`settings` callers, e.g. the dry-run MockClient).
DEFAULT_ROSTER: dict[str, Character] = {
    c.name.lower(): c for c in [
        Character(name="Subaru", aliases=["Natsuki Subaru", "Subaru Natsuki"],
                  descriptor=("Subaru Natsuki, a 17-year-old Japanese boy with short messy black hair, "
                              "sharp narrow eyes with faint eye-bags, lean athletic build; wears a "
                              "plain grey and orange tracksuit jacket bearing NO letters, numbers or logos")),
        Character(name="Emilia", aliases=["the half-elf"],
                  descriptor=("Emilia, a beautiful half-elf girl with very long flowing silver-white hair, "
                              "large round eyes, slightly pointed ears, a single flower hair clip; wears an "
                              "ornate white dress with a high collar")),
        Character(name="Beatrice", aliases=["Beako"],
                  descriptor=("Beatrice, a small doll-like girl who appears about twelve, with long pale "
                              "ringlet drill-curls tied with butterfly bows, wide eyes; wears an elaborate "
                              "frilly rococo pink dress")),
        Character(name="Otto", aliases=["Otto Suwen"],
                  descriptor=("Otto Suwen, a young merchant man with short ginger-brown hair and an anxious "
                              "friendly face; wears a layered travelling coat and scarf")),
        Character(name="Garfiel", aliases=["Garfiel Tinsel", "Gar"],
                  descriptor=("Garfiel Tinsel, a muscular teenage boy with short cropped blond hair, a "
                              "prominent fang tooth, fierce eyes and a scar across the nose; wears fingerless "
                              "gloves and a rugged torn jacket")),
        Character(name="Ram",
                  descriptor=("Ram, a composed maid with short pink hair swept over the right eye, calm sharp "
                              "eyes; wears a black and white frilled maid dress")),
        Character(name="Julius", aliases=["Julius Juukulius"],
                  descriptor=("Julius Juukulius, a refined knight with neatly combed purple hair and one loose "
                              "fringe strand, narrow elegant eyes; wears an ornate royal knight's uniform with a rapier")),
        Character(name="Anastasia", aliases=["Anastasia Hoshin", "Echidna"],
                  descriptor=("Anastasia Hoshin, a young woman with long wavy lavender hair worn under a distinctive "
                              "white fluffy fur cossack hat, large blue-green eyes and a shrewd smile; wears a white "
                              "fur-trimmed cape over a long pale cream-and-pink dress")),
        Character(name="Meili", aliases=["Meili Portroute"],
                  descriptor=("Meili Portroute, a young girl with long dark-blue hair in two braids and sleepy "
                              "half-lidded eyes; wears a simple dark one-piece dress")),
        Character(name="Shaula",
                  descriptor=("Shaula, a tall lithe woman with an extremely long black ponytail and star motifs, "
                              "cheerful eyes; wears a modest dark fitted bodysuit")),
        Character(name="Patrasche", aliases=["Patorasche"],
                  descriptor=("Patrasche, a large lizard-like quadruped GROUND dragon (earth dragon) with sleek "
                              "dark scales, a long low body, sturdy reptilian legs and a long tail — like a giant "
                              "monitor lizard, NOT a winged Western dragon and NOT bipedal; clean black-and-white "
                              "manga linework, no screentone")),
    ]
}

# Back-compat module alias: the default roster, importable as `bible.ROSTER` as before.
ROSTER: dict[str, Character] = DEFAULT_ROSTER


# ── work-agnostic resolution from settings ───────────────────────────────────────────────────
#
# `settings.bible` (a plain dict; Settings has extra="allow") may carry:
#   roster:        an inline list of {name, aliases, descriptor, ref_image} character mappings
#   roster_file:   path to a YAML file holding that same list (resolved relative to PROJECT_ROOT)
#   global_style:  a string that overrides GLOBAL_STYLE everywhere it is appended
# When `settings` is None, or `settings.bible` is absent/empty, the built-in Re:Zero DEFAULT_ROSTER
# + GLOBAL_STYLE are used, so default behavior is byte-identical.

def _roster_from_items(items: Any) -> dict[str, Character]:
    if not isinstance(items, (list, tuple)):
        raise ValueError("bible roster must be a list of character mappings")
    out: dict[str, Character] = {}
    for it in items:
        c = it if isinstance(it, Character) else Character(**it)
        out[c.name.lower()] = c
    return out


def _load_roster_file(rel: str) -> dict[str, Character]:
    p = Path(rel)
    if not p.is_absolute():
        p = PROJECT_ROOT / rel
    if not p.exists():
        raise FileNotFoundError(f"bible.roster_file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("roster"), (list, tuple)):
        data = data["roster"]            # allow either a bare list or a {roster: [...]} mapping
    return _roster_from_items(data)


def _bible_cfg(settings: Any) -> dict[str, Any]:
    raw = getattr(settings, "bible", None) if settings is not None else None
    return raw if isinstance(raw, dict) else {}


@lru_cache(maxsize=None)
def _resolve_cached(roster_key: str, roster_file: str | None,
                    global_style: str | None) -> tuple[dict[str, Character], str]:
    """Pure, deterministic resolution keyed only by the bible config's content (so identical
    configs share one parse, and a changed config re-resolves). Avoids mutable module/settings
    state. `roster_key` is the JSON of the inline roster (empty when none given)."""
    import json
    if roster_key:
        roster = _roster_from_items(json.loads(roster_key))
    elif roster_file:
        roster = _load_roster_file(roster_file)
    else:
        roster = DEFAULT_ROSTER
    return roster, (global_style if global_style is not None else GLOBAL_STYLE)


def resolve_bible(settings: Any = None) -> tuple[dict[str, Character], str]:
    """Return the (roster, global_style) active for `settings`. Defaults to Re:Zero + GLOBAL_STYLE."""
    cfg = _bible_cfg(settings)
    if not cfg:
        return DEFAULT_ROSTER, GLOBAL_STYLE
    import json
    inline = cfg.get("roster")
    roster_key = json.dumps(inline, sort_keys=True, default=str) if inline else ""
    style = cfg.get("global_style")
    return _resolve_cached(roster_key, cfg.get("roster_file"),
                           style if isinstance(style, str) else None)


def active_roster(settings: Any = None) -> dict[str, Character]:
    return resolve_bible(settings)[0]


def active_style(settings: Any = None) -> str:
    return resolve_bible(settings)[1]


def roster_names(settings: Any = None) -> list[str]:
    out: list[str] = []
    for c in active_roster(settings).values():
        out.append(c.name)
        out.extend(c.aliases)
    # longer names first so "Subaru Natsuki" matches before "Subaru"
    return sorted(set(out), key=len, reverse=True)


def get_character(name: str, settings: Any = None) -> Character | None:
    if not name:
        return None
    roster = active_roster(settings)
    key = name.strip().lower()
    if key in roster:
        return roster[key]
    # EXACT name/alias match only. A substring fallback (audit H2) wrongly merged distinct
    # characters into roster ones ("Beatrice clone" -> Beatrice, "Subaru's mother" -> Subaru),
    # silently denying them their own sheet and rendering them as the wrong character. The parser
    # is instructed to emit canonical roster names, so exact matching suffices; an off-name simply
    # becomes its own discovered character (the safe failure: an extra sheet, never a wrong one).
    for c in roster.values():
        if key == c.name.lower() or any(key == a.lower() for a in c.aliases):
            return c
    return None


def ref_sheet_prompt_for(name: str, descriptor: str, settings: Any = None) -> str:
    """Reference-sheet prompt for ANY character (roster or discovered), from a descriptor."""
    return (
        f"Character reference model sheet of {descriptor or name}. "
        "Show the SAME character three times on a plain white background: a full-body front view, "
        "a full-body side view, and a head-and-shoulders close-up. Neutral standing pose, neutral "
        "expression, consistent design across all three. " + active_style(settings)
    )


def ref_sheet_prompt(char: Character, settings: Any = None) -> str:
    return ref_sheet_prompt_for(char.name, char.descriptor, settings)


def descriptor_for(name: str, cast: dict[str, str] | None = None, settings: Any = None) -> str:
    """Resolve a character's visual descriptor: bible roster first, then the chapter cast, else
    just the name."""
    c = get_character(name, settings)
    if c:
        return c.descriptor
    if cast and cast.get(name):
        return cast[name]
    return name


@lru_cache(maxsize=64)
def _composite_sheets_cached(paths: tuple[str, ...], height: int) -> bytes:
    """Pure, deterministic tiling (open -> resize -> tile -> PNG-encode), memoized on the
    content-addressed sheet paths + height. The same cast recurs across consecutive panels in a
    scene, so identical composites are rebuilt repeatedly otherwise. Returns the SAME bytes object
    on a cache hit, so the encoded PNG is byte-identical (the panel cache key hashes these bytes)."""
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        w = int(im.width * height / im.height)
        imgs.append(im.resize((w, height)))
    if not imgs:
        raise ValueError("no sheets to composite")
    total_w = sum(im.width for im in imgs)
    canvas = Image.new("RGB", (total_w, height), (255, 255, 255))
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.width
    buf = io.BytesIO()
    canvas.save(buf, "PNG")
    return buf.getvalue()


def composite_sheets(paths: list[str], height: int = 768) -> bytes:
    """Fallback: tile several reference sheets into one image (for >N-character panels).

    Memoized via `_composite_sheets_cached` keyed on the (ordered) paths + height. Callers must
    treat the returned bytes as immutable — they are hashed into the panel content-cache key and
    the same object is shared across cache hits."""
    return _composite_sheets_cached(tuple(paths), height)
