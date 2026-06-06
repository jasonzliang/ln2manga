"""Tests for the work-agnostic character bible.

The roster + global art style are the DEFAULT (Re:Zero) unless `settings.bible` overrides them.
These tests pin the four behaviors that make ln2manga retargetable from config:
  (a) an inline `settings.bible.roster` (and a `roster_file`) override the default roster;
  (b) absent `bible` config -> the built-in Re:Zero default roster is used, unchanged;
  (c) `settings.bible.global_style` overrides the style in the prompts the stages build;
  (d) a character not in the (custom) roster still falls back to the per-chapter cast descriptor.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

# make the package importable without an editable install
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ln2manga import bible  # noqa: E402
from ln2manga.artifacts import PanelSpec  # noqa: E402
from ln2manga.stages import charsheet, parse, script  # noqa: E402


def _set_bible(settings, value):
    # settings.bible is an extra="allow" field; pydantic guards normal setattr, so set it raw.
    object.__setattr__(settings, "bible", value)


# ── (b) absent config -> built-in Re:Zero default, byte-identical to no-settings ─────────────
def test_absent_config_uses_default_rezero_roster(settings):
    assert not getattr(settings, "bible", None)        # default.yaml ships no `bible` section
    roster, style = bible.resolve_bible(settings)
    assert roster is bible.DEFAULT_ROSTER              # the built-in object, not a copy
    assert style == bible.GLOBAL_STYLE
    # the public helpers behave identically with-or-without settings on the default path
    assert bible.get_character("Subaru", settings) is bible.get_character("Subaru")
    assert bible.get_character("Subaru", settings).name == "Subaru"
    assert bible.descriptor_for("Subaru", None, settings) == \
        bible.DEFAULT_ROSTER["subaru"].descriptor
    assert bible.roster_names(settings) == bible.roster_names()
    assert parse._roster_block(settings) == parse._roster_block()


def test_default_roster_alias_and_exact_match_rules_preserved(settings):
    # longer alias matches first (canonicalizes to the same character)
    assert bible.get_character("Subaru Natsuki", settings).name == "Subaru"
    assert bible.get_character("Beako", settings).name == "Beatrice"
    # EXACT match only: an off-name is NOT merged into a roster character
    assert bible.get_character("Beatrice clone", settings) is None
    assert bible.get_character("", settings) is None


# ── (a) inline roster overrides the default ──────────────────────────────────────────────────
_CUSTOM = [
    {"name": "Aria", "aliases": ["the Songbird"],
     "descriptor": "Aria, a girl with teal twin-braids and a star pendant"},
    {"name": "Brand", "descriptor": "Brand, a grey-haired knight in dark plate armor"},
]


def test_inline_roster_overrides_default(settings):
    _set_bible(settings, {"roster": _CUSTOM})
    roster = bible.active_roster(settings)
    assert set(roster) == {"aria", "brand"}
    # the default Re:Zero cast is gone entirely
    assert bible.get_character("Subaru", settings) is None
    # the new cast resolves (incl. alias matching on the custom roster)
    assert bible.get_character("the Songbird", settings).name == "Aria"
    assert bible.descriptor_for("Aria", None, settings).startswith("Aria, a girl with teal")
    # the parser's roster block reflects the new cast
    block = parse._roster_block(settings)
    assert "Aria" in block and "Brand" in block and "Subaru" not in block


def test_roster_file_overrides_default(settings, tmp_path):
    f = tmp_path / "roster.yaml"
    f.write_text(textwrap.dedent("""
        - name: Kira
          aliases: [K]
          descriptor: Kira, a pilot with goggles and a flight jacket
        - name: Mara
          descriptor: Mara, an engineer with a wrench and oil-stained overalls
    """), encoding="utf-8")
    # roster_file is resolved relative to project root, so pass an absolute path here
    _set_bible(settings, {"roster_file": str(f)})
    roster = bible.active_roster(settings)
    assert set(roster) == {"kira", "mara"}
    assert bible.get_character("K", settings).name == "Kira"        # alias match
    assert bible.get_character("Subaru", settings) is None          # default gone


# ── (c) global_style override threads into every built prompt ────────────────────────────────
_STYLE = "Full-color watercolor storybook art, soft pastel washes."


def test_global_style_override_in_script_prompt(settings):
    _set_bible(settings, {"roster": _CUSTOM, "global_style": _STYLE})
    pp = script.build_prompt(
        PanelSpec(panel_number=1, characters_present=["Aria"], action="singing on a hill"),
        cast={"Aria": "ignored — roster descriptor wins"}, settings=settings)
    assert _STYLE in pp.prompt
    assert bible.GLOBAL_STYLE not in pp.prompt
    assert "Aria, a girl with teal twin-braids" in pp.prompt        # roster descriptor used


def test_global_style_override_in_charsheet_prompts(settings):
    _set_bible(settings, {"roster": _CUSTOM, "global_style": _STYLE})
    # AI-sheet prompt
    sheet = bible.ref_sheet_prompt_for("Aria", bible.descriptor_for("Aria", None, settings),
                                       settings)
    assert _STYLE in sheet and bible.GLOBAL_STYLE not in sheet
    # synthesized-from-reference prompt
    synth = charsheet._synth_prompt("Aria", "Aria, a girl with teal twin-braids", 2, settings)
    assert _STYLE in synth and bible.GLOBAL_STYLE not in synth


def test_global_style_without_roster_keeps_default_roster(settings):
    # style-only override must not disturb the default roster
    _set_bible(settings, {"global_style": _STYLE})
    assert bible.active_style(settings) == _STYLE
    assert bible.get_character("Subaru", settings).name == "Subaru"


# ── (d) unknown-work fallback: not-in-roster char resolves via the chapter cast ──────────────
def test_non_roster_character_falls_back_to_cast_descriptor(settings):
    _set_bible(settings, {"roster": _CUSTOM})
    cast = {"Goblin": "a small green goblin with a rusty dagger"}
    # not in the custom roster, but present in the cast -> cast descriptor wins
    assert bible.get_character("Goblin", settings) is None
    assert bible.descriptor_for("Goblin", cast, settings) == cast["Goblin"]
    # and the script prompt uses it too
    pp = script.build_prompt(
        PanelSpec(panel_number=1, characters_present=["Goblin"], action="lurking"),
        cast=cast, settings=settings)
    assert "a small green goblin with a rusty dagger" in pp.prompt
    # neither in roster nor cast -> bare name (the final fallback)
    assert bible.descriptor_for("Nobody", cast, settings) == "Nobody"


# ── composite_sheets: byte-identical output, memoized on repeat calls ─────────────────────────
def _write_sheets(tmp_path):
    """Two visually-distinct PNGs whose order matters to the tiled composite."""
    import io

    from PIL import Image
    paths = []
    for i, (size, color) in enumerate([((40, 60), 30), ((50, 70), 220)]):
        buf = io.BytesIO()
        Image.new("L", size, color).save(buf, "PNG")
        p = tmp_path / f"sheet_{i}.png"
        p.write_bytes(buf.getvalue())
        paths.append(str(p))
    return paths


def test_composite_sheets_byte_identical_and_cached(tmp_path):
    """The composite must be byte-identical across calls AND not be recomputed on a repeat call
    with the same paths — the panel content-cache key hashes these bytes, so churn would invalidate
    every panel that tiles its refs."""
    bible._composite_sheets_cached.cache_clear()
    paths = _write_sheets(tmp_path)

    # Count the actual pixel work: Image.open runs once per path on an UNCACHED compute, never on a
    # cache hit. Spying on it proves the heavy work is skipped on the second call.
    real_open = bible.Image.open
    calls = {"n": 0}

    def _counting_open(*a, **k):
        calls["n"] += 1
        return real_open(*a, **k)

    bible.Image.open = _counting_open
    try:
        first = bible.composite_sheets(paths)
        assert calls["n"] == len(paths)                 # opened each sheet once (real compute)
        second = bible.composite_sheets(paths)
        assert calls["n"] == len(paths)                 # NO further opens -> served from cache
    finally:
        bible.Image.open = real_open

    # Cache hit returns the SAME object -> trivially byte-identical, and equal across calls.
    assert second is first
    assert first == second

    # Byte-identical to a fresh, uncached recompute (the guardrail: cache must not alter bytes).
    bible._composite_sheets_cached.cache_clear()
    fresh = bible.composite_sheets(paths)
    assert fresh == first

    # The public list signature and a tuple key resolve to the same cached bytes; order/height
    # are part of the key (different order -> a distinct composite).
    assert bible.composite_sheets(paths) is bible._composite_sheets_cached(tuple(paths), 768)
    assert bible.composite_sheets(list(reversed(paths))) != first
