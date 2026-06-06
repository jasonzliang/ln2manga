from ln2manga.artifacts import Chapter
from ln2manga.cache import Cache
from ln2manga.clients import MockClient
from ln2manga.cost import CostTracker
from ln2manga.stages import parse


def test_system_prompt_caps_dialogue_and_warns_against_padding():
    """The SYSTEM prompt must cap dialogue per panel and tell the LLM to split dense
    exchanges across a few panels, while still warning against padding/filler — this is
    what stops the lettering stage from dropping overflow bubbles without exploding the
    panel count."""
    sys = parse.SYSTEM
    low = sys.lower()
    # a concrete per-panel dialogue cap (~3 lines)
    assert "at most about 3 short dialogue lines" in low
    # split dense back-and-forth across a few panels
    assert "split it across a few consecutive panels" in low
    # anti-padding caveat that keeps the split measured
    assert "filler panels" in low
    assert "only split when the dialogue would otherwise overflow" in low


def test_chunking_respects_scene_breaks():
    paras = ["a" * 100 for _ in range(10)]
    chunks = parse.chunk_paragraphs(paras, scene_breaks=[5], chunk_chars=100_000)
    assert len(chunks) == 2          # forced split at the scene break


def test_chunking_respects_size():
    paras = ["x" * 3000 for _ in range(4)]
    chunks = parse.chunk_paragraphs(paras, scene_breaks=[], chunk_chars=5000)
    assert len(chunks) >= 2


def test_chunking_splits_oversize_single_paragraph():
    # a single paragraph far larger than chunk_chars must still be bounded
    chunks = parse.chunk_paragraphs(["a" * 20000], scene_breaks=[], chunk_chars=6000)
    assert len(chunks) > 1
    assert all(len(c) <= 6000 for c in chunks)
    assert "".join(chunks) == "a" * 20000      # no characters lost


def test_chunking_oversize_split_avoids_midword_cuts():
    word = "word "
    para = word * 4000                           # 20000 chars, all on whitespace bounds
    chunks = parse.chunk_paragraphs([para.strip()], scene_breaks=[], chunk_chars=6000)
    assert all(len(c) <= 6000 for c in chunks)
    for c in chunks[:-1]:                         # interior pieces shouldn't end mid-word
        assert c.endswith("word")


def test_parse_no_output_error_is_actionable(settings):
    """A refusal/truncation must raise an error naming the chunk and a remedy (not the
    old opaque message)."""
    import pytest

    class _NoneParsedClient:
        class _Responses:
            def parse(self, *, model, input, text_format, **kw):
                from types import SimpleNamespace
                return SimpleNamespace(output_parsed=None, usage=None)

        def __init__(self):
            self.responses = self._Responses()

    ch = Chapter(arc="arc-6", number=1, title="t", url="u",
                 paragraphs=["The hero drew his sword and charged forward."],
                 scene_breaks=[])
    tracker = CostTracker(10.0, settings.ledger_path, settings.prices_usd)
    cache = Cache(settings.cache_dir)
    with pytest.raises(RuntimeError) as exc:
        parse.run(_NoneParsedClient(), settings, tracker, cache, ch)
    msg = str(exc.value)
    assert "The hero drew" in msg                # names the offending chunk
    assert "max_output_tokens" in msg            # points at a remedy


def test_parse_with_mock_client(settings):
    ch = Chapter(arc="arc-6", number=1, title="t",
                 url="u",
                 paragraphs=['Subaru clenched his fists. "I won\'t give up," he said.',
                             'Emilia watched him with worry.'],
                 scene_breaks=[])
    client = MockClient()
    tracker = CostTracker(10.0, settings.ledger_path, settings.prices_usd)
    cache = Cache(settings.cache_dir)
    specs = parse.run(client, settings, tracker, cache, ch)
    assert specs, "expected panels from mock parse"
    assert [s.panel_number for s in specs] == list(range(1, len(specs) + 1))
    assert tracker.spent == 0.0      # mock = free


class _NoUsageClient:
    """Returns a valid parse with usage=None to exercise the fallback charge (#13)."""

    class _Responses:
        def parse(self, *, model, input, text_format, **kw):
            from types import SimpleNamespace

            from ln2manga.artifacts import PanelSpec
            parsed = text_format(panels=[PanelSpec(action="x", setting="y")])
            return SimpleNamespace(output_parsed=parsed, usage=None)

    def __init__(self):
        self.responses = self._Responses()


def test_parse_charges_fallback_estimate_when_usage_missing(settings):
    """#13: a billed text call whose response carries no usage must still increment spend,
    so the cap can't be silently breached."""
    ch = Chapter(arc="arc-6", number=1, title="t", url="u",
                 paragraphs=["Some prose paragraph for a single chunk."],
                 scene_breaks=[])
    tracker = CostTracker(10.0, settings.ledger_path, settings.prices_usd)
    cache = Cache(settings.cache_dir)
    specs = parse.run(_NoUsageClient(), settings, tracker, cache, ch)
    assert specs
    assert tracker.spent > 0.0       # fallback estimate charged, not dropped
    assert tracker.events[-1]["meta"].get("note") == "usage_missing_fallback_estimate"


def test_parse_cache_key_folds_in_all_output_determining_inputs(settings, monkeypatch):
    """The parse cache key must change when ANY output-shaping input changes (SYSTEM
    prompt, roster block, reasoning effort, max_output_tokens) — otherwise editing the
    prompt or roster, or changing the reasoning/token settings, would silently serve
    stale parses from the persistent disk cache."""
    seen = {}

    class _Cache(Cache):
        # capture the key, then force a miss so _parse_chunk doesn't short-circuit
        def get(self, stage, key, ext="png"):
            seen["key"] = key
            return None
        def put(self, stage, key, data, ext="png", meta=None):
            return None

    def _key() -> str:
        tracker = CostTracker(10.0, settings.ledger_path, settings.prices_usd)
        parse._parse_chunk(_NoUsageClient(), settings, tracker, _Cache(settings.cache_dir),
                           "a stable chunk of prose")
        return seen["key"]

    base = _key()
    assert len(base) == 32
    assert _key() == base                        # deterministic for identical inputs

    monkeypatch.setattr(parse, "SYSTEM", parse.SYSTEM + " EXTRA")
    assert _key() != base
    monkeypatch.undo()

    monkeypatch.setattr(parse, "_roster_block", lambda settings=None: "different roster block")
    assert _key() != base
    monkeypatch.undo()

    settings.parse["reasoning_effort"] = "high"
    assert _key() != base
    settings.parse["reasoning_effort"] = "low"

    orig = int(settings.parse["max_output_tokens"])
    settings.parse["max_output_tokens"] = orig + 1
    assert _key() != base
    settings.parse["max_output_tokens"] = orig
    assert _key() == base                        # fully restored -> same key


class _DescriptorClient:
    """Returns one panel containing the given discovered (non-roster) characters, each with
    a caller-supplied descriptor. Lets a test drive what the 'LLM' emits for the cast so we
    can assert descriptor stability across re-parses."""

    def __init__(self, chars: dict[str, str]):
        self._chars = chars
        outer = self

        class _Responses:
            def parse(self, *, model, input, text_format, **kw):
                from types import SimpleNamespace

                from ln2manga.artifacts import CharacterRef, PanelSpec
                names = list(outer._chars)
                panel = PanelSpec(action="x", setting="y", characters_present=names)
                cast = [CharacterRef(name=n, descriptor=d) for n, d in outer._chars.items()]
                parsed = text_format(panels=[panel], cast=cast)
                return SimpleNamespace(output_parsed=parsed, usage=None)

        self.responses = _Responses()


class _NoCache(Cache):
    """Never persists, so each parse.run re-invokes the client (mirrors a fresh LLM call)
    instead of replaying the on-disk chunk cache from the prior run."""

    def get(self, stage, key, ext="png"):
        return None

    def put(self, stage, key, data, ext="png", meta=None):
        return None


def test_discovered_descriptor_is_stable_across_reparses(settings):
    """Cache stability: re-parsing a chapter must NOT change a previously-seen discovered
    (non-roster) character's descriptor, even when the LLM emits a different one — otherwise
    the character's sheet (and every panel using it) needlessly regenerates. A brand-new
    character appearing only on the second run must still get its fresh descriptor."""
    import json
    from pathlib import Path

    ch = Chapter(arc="arc-6", number=1, title="t", url="u",
                 paragraphs=["A scene with a stranger."], scene_breaks=[])
    tracker = CostTracker(10.0, settings.ledger_path, settings.prices_usd)
    cache = _NoCache(settings.cache_dir)

    # first parse: a discovered character "Rem" with descriptor v1
    parse.run(_DescriptorClient({"Rem": "a blue-haired maid, version one"}),
              settings, tracker, cache, ch)
    cast_path = Path(settings.artifacts_dir) / "chapter-1.cast.json"
    first = json.loads(cast_path.read_text(encoding="utf-8"))
    assert first["Rem"] == "a blue-haired maid, version one"

    # second parse: same character with a DRIFTED descriptor + a brand-new character
    parse.run(_DescriptorClient({"Rem": "a blue-haired maid, version TWO (drifted)",
                                 "Felt": "a scrappy blonde thief"}),
              settings, tracker, cache, ch)
    second = json.loads(cast_path.read_text(encoding="utf-8"))
    # already-seen discovered character -> descriptor reused (stable cache key)
    assert second["Rem"] == "a blue-haired maid, version one"
    # genuinely new discovered character -> its fresh descriptor is taken
    assert second["Felt"] == "a scrappy blonde thief"


def test_panel_prompt_forbids_logos_on_clothing_and_objects():
    """A QA audit found generated panels render brand marks / a stray "N" on clothing.
    The built panel prompt must explicitly forbid letters/numbers/logos on clothing and
    objects, on top of the existing global no-text clause."""
    from ln2manga.artifacts import PanelSpec
    from ln2manga.stages.script import build_prompt

    pp = build_prompt(PanelSpec(panel_number=1, characters_present=["Subaru"],
                                action="jogging in a tracksuit"))
    assert ("NO letters, numbers, logos, brand marks or insignia on clothing, "
            "accessories or any object.") in pp.prompt
