"""Tests for the opt-in real-reference-image feature (Part A)."""
from __future__ import annotations

import io
import json
import os
from types import SimpleNamespace

import pytest
from PIL import Image

from ln2manga import references
from ln2manga.cache import Cache
from ln2manga.cost import CostTracker
from ln2manga.stages import charsheet


def _png_bytes(size=(40, 60), color=128) -> bytes:
    buf = io.BytesIO()
    Image.new("L", size, color).save(buf, "PNG")
    return buf.getvalue()


def _enable_refs(settings, *, mode="stylize", sources=None):
    """Attach a references config to the settings instance (model has no such field)."""
    cfg = {"enabled": True, "mode": mode, "external_file": "config/__none__.yaml",
           "sources": sources or {}}
    # attach directly (object.__setattr__ is what load_references_config uses internally too)
    object.__setattr__(settings, "references", cfg)
    return cfg


# ── resolve_reference ────────────────────────────────────────────────────────
def test_resolve_local_path_returns_png(settings, tmp_path):
    src = tmp_path / "subaru_src.png"
    src.write_bytes(_png_bytes())
    _enable_refs(settings, sources={"Subaru": str(src)})

    out = references.resolve_reference("Subaru", settings)

    assert out is not None
    assert out.exists()
    assert out.suffix == ".png"
    # it is a readable image
    Image.open(out).load()


def test_resolve_alias_is_case_insensitive(settings, tmp_path):
    src = tmp_path / "beako.png"
    src.write_bytes(_png_bytes())
    # "Beako" is an alias of Beatrice; look it up via lowercase canonical name
    _enable_refs(settings, sources={"beako": str(src)})

    out = references.resolve_reference("Beatrice", settings)
    assert out is not None and out.exists()


def test_resolve_no_source_returns_none(settings):
    _enable_refs(settings, sources={})
    assert references.resolve_reference("Subaru", settings) is None
    assert references.reference_source("Subaru", settings) is None


def test_resolve_url_downloads_and_caches(settings, monkeypatch):
    fake = _png_bytes(color=200)
    calls = {"n": 0}

    def fake_get(url, headers=None, timeout=None):
        calls["n"] += 1
        assert url.startswith("https://")
        assert "User-Agent" in (headers or {})
        return SimpleNamespace(content=fake, raise_for_status=lambda: None)

    monkeypatch.setattr(references.requests, "get", fake_get)
    _enable_refs(settings, sources={"Emilia": "https://example.com/emilia.png"})

    out = references.resolve_reference("Emilia", settings)
    assert out is not None and out.exists()
    assert calls["n"] == 1
    Image.open(out).load()

    # second call is served from disk cache -> no extra download
    out2 = references.resolve_reference("Emilia", settings)
    assert out2 == out
    assert calls["n"] == 1


def _search_client(url, *, usage=None):
    """Fake OpenAI client whose responses.create returns `url` in output_text."""
    class _Resp:
        output_text = url
        output = []

    r = _Resp()
    r.usage = usage

    class _Responses:
        def create(self, **kw):
            return r

    class _Client:
        responses = _Responses()

    return _Client()


def test_online_search_finds_and_downloads(settings, monkeypatch):
    object.__setattr__(settings, "references",
                       {"enabled": True, "mode": "raw", "sources": {},
                        "external_file": "config/__none__.yaml",
                        "search": {"enabled": True, "series": "Re:Zero", "tool": "web_search",
                                   "max_urls": 3, "min_bytes": 1, "verify": False,
                                   "agentic": False, "cost_per_call": 0.0}})
    url = "https://example.com/official_subaru.png"
    img = _png_bytes(size=(256, 384))   # big enough to pass the searched-image dimension gate

    def fake_get(u, headers=None, timeout=None):
        assert u == url
        return SimpleNamespace(content=img, raise_for_status=lambda: None)

    monkeypatch.setattr(references.requests, "get", fake_get)
    out = references.resolve_reference("Subaru", settings, client=_search_client(url))
    assert out is not None and out.exists()
    Image.open(out).load()


def test_online_search_skips_too_small_then_uses_next(settings, monkeypatch):
    object.__setattr__(settings, "references",
                       {"enabled": True, "mode": "raw", "sources": {},
                        "external_file": "config/__none__.yaml",
                        "search": {"enabled": True, "tool": "web_search", "max_urls": 5,
                                   "min_bytes": 200, "verify": False, "agentic": False, "cost_per_call": 0.0}})
    small = "https://example.com/placeholder.png"   # 404-ish tiny image -> rejected by min_bytes
    good = "https://example.com/real.png"
    big = _png_bytes(size=(256, 384))                # ~800B compressed -> passes min_bytes=200
    tiny = _png_bytes(size=(8, 8))                   # ~73B -> rejected

    def fake_get(u, headers=None, timeout=None):
        return SimpleNamespace(content=(tiny if u == small else big),
                               raise_for_status=lambda: None)

    monkeypatch.setattr(references.requests, "get", fake_get)
    client = _search_client(f"{small}\n{good}")     # two candidates, first too small
    out = references.resolve_reference("Subaru", settings, client=client)
    assert out is not None and out.exists()


def test_search_disabled_or_no_client_returns_none(settings):
    object.__setattr__(settings, "references",
                       {"enabled": True, "sources": {}, "external_file": "config/__none__.yaml",
                        "search": {"enabled": False}})
    assert references.resolve_reference("Subaru", settings, client=object()) is None
    # even with search enabled, no client -> no search
    object.__setattr__(settings, "references",
                       {"enabled": True, "sources": {}, "external_file": "config/__none__.yaml",
                        "search": {"enabled": True}})
    assert references.resolve_reference("Subaru", settings, client=None) is None


def _io_aware_client(search_text, verify_answer):
    """responses.create returns `search_text` for a string input (search) and `verify_answer`
    for a list input (vision verify)."""
    class _R:
        def __init__(self, t):
            self.output_text = t
            self.output = []
            self.usage = None

    class _Responses:
        def create(self, **kw):
            return _R(search_text) if isinstance(kw.get("input"), str) else _R(verify_answer)

    class _Client:
        responses = _Responses()

    return _Client()


def test_search_set_keeps_multiple_verified(settings, monkeypatch):
    object.__setattr__(settings, "references",
                       {"enabled": True, "mode": "stylize", "sources": {},
                        "external_file": "config/__none__.yaml",
                        "search": {"enabled": True, "tool": "web_search", "max_urls": 8,
                                   "max_keep": 2, "min_bytes": 1, "verify": True,
                                   "agentic": False, "cost_per_call": 0.0}})
    urls = "\n".join(f"https://example.com/{c}.png" for c in "abc")
    big = _png_bytes(size=(256, 384))
    monkeypatch.setattr(references.requests, "get",
                        lambda u, headers=None, timeout=None:
                        SimpleNamespace(content=big, raise_for_status=lambda: None))
    out = references.resolve_reference_set("Subaru", settings,
                                           client=_io_aware_client(urls, "yes"))
    assert len(out) == 2                     # capped at max_keep
    assert all(p.exists() for p in out)


def test_search_verify_rejects_wrong_character(settings, monkeypatch):
    object.__setattr__(settings, "references",
                       {"enabled": True, "sources": {}, "external_file": "config/__none__.yaml",
                        "search": {"enabled": True, "tool": "web_search", "max_urls": 3,
                                   "max_keep": 3, "min_bytes": 1, "verify": True,
                                   "agentic": False, "cost_per_call": 0.0}})
    big = _png_bytes(size=(256, 384))
    monkeypatch.setattr(references.requests, "get",
                        lambda u, headers=None, timeout=None:
                        SimpleNamespace(content=big, raise_for_status=lambda: None))
    # vision says "no" -> every candidate rejected -> empty set (would fall back to AI sheet)
    rejected = references.resolve_reference_set(
        "Subaru", settings, client=_io_aware_client("https://example.com/wrong.png", "no"))
    assert rejected == []
    # a different character whose vision check passes -> kept
    kept = references.resolve_reference_set(
        "Emilia", settings, client=_io_aware_client("https://example.com/right.png", "yes"))
    assert len(kept) >= 1


def test_explicit_sources_list_returns_multiple(settings, tmp_path):
    a = tmp_path / "a.png"
    a.write_bytes(_png_bytes())
    b = tmp_path / "b.png"
    b.write_bytes(_png_bytes())
    object.__setattr__(settings, "references",
                       {"enabled": True, "mode": "stylize", "external_file": "config/__none__.yaml",
                        "sources": {"Subaru": [str(a), str(b)]}, "search": {"enabled": False}})
    assert len(references.reference_sources("Subaru", settings)) == 2
    out = references.resolve_reference_set("Subaru", settings)
    assert len(out) == 2 and all(p.exists() for p in out)


def test_agentic_search_loop(settings, monkeypatch):
    """The pipeline's agentic search: model emits a verify_image function-call, we download+verify
    and feed the result back; loop ends when enough verified. Mirrors the real Responses tool-loop."""
    object.__setattr__(settings, "references",
                       {"enabled": True, "mode": "stylize", "external_file": "config/__none__.yaml",
                        "sources": {}, "search": {"enabled": True, "agentic": True,
                                                  "agentic_rounds": 4, "max_keep": 1, "min_bytes": 1,
                                                  "verify": True, "tool": "web_search",
                                                  "cost_per_call": 0.0}})
    big = _png_bytes(size=(256, 384))
    monkeypatch.setattr(references.requests, "get",
                        lambda u, headers=None, timeout=None:
                        SimpleNamespace(content=big, raise_for_status=lambda: None))

    fc = SimpleNamespace(type="function_call", name="verify_image", call_id="c1",
                         arguments=json.dumps({"url": "https://example.com/official.png"}))

    def make(output, text=""):
        return SimpleNamespace(output=output, output_text=text, id="r1", usage=None)

    def create(**kw):
        if "tools" in kw:                                  # agentic round
            if isinstance(kw.get("input"), str):
                return make([fc])                          # round 1: ask to verify a URL
            return make([SimpleNamespace(type="message")], "done")   # after tool output: stop
        return make([], "yes")                             # vision verify -> yes

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    out = references.resolve_reference_set("Subaru", settings, client=client)
    assert len(out) == 1 and out[0].exists()


def _searched_marker(settings, name):
    """Path to the '<slug>.searched' completed-search marker for a character."""
    char = references.bible.get_character(name)
    canon = char.name if char else name
    return settings.cache_dir("refs") / f"{references._slug(canon)}.searched"


def _agentic_cfg(settings, **search_over):
    search = {"enabled": True, "agentic": True, "agentic_rounds": 4, "max_keep": 1,
              "min_bytes": 1, "verify": True, "tool": "web_search", "cost_per_call": 0.03}
    search.update(search_over)
    object.__setattr__(settings, "references",
                       {"enabled": True, "mode": "stylize", "external_file": "config/__none__.yaml",
                        "sources": {}, "search": search})


class _CountingTracker:
    """Minimal CostTracker stand-in: records check()/record() calls and can raise on check()."""

    def __init__(self, *, raise_on_check_n=None):
        self.checks = 0
        self.records = []
        self._raise_on = raise_on_check_n   # 1 => raise on the FIRST check(), etc.

    def check(self, estimate, *, is_image=False):
        self.checks += 1
        if self._raise_on is not None and self.checks == self._raise_on:
            from ln2manga.cost import BudgetExceeded
            raise BudgetExceeded("test cap")

    def record(self, *a, **k):
        self.records.append((a, k))

    def estimate_text(self, *a, **k):
        return 0.0


def test_agentic_first_call_error_marks_searched(settings):
    """Regression: when the FIRST agentic API call fails (transient/rate-limit), the search is
    cached as completed so a re-run NEVER re-pays the full agentic search+verify loop."""
    _agentic_cfg(settings)

    class _Boom:
        class responses:
            @staticmethod
            def create(**kw):
                raise RuntimeError("transient 503")

    out = references.resolve_reference_set("Subaru", settings, client=_Boom())
    assert out == []
    assert _searched_marker(settings, "Subaru").exists()   # errored attempt is cached -> no re-search


def test_agentic_budget_exceeded_does_not_mark_searched(settings):
    """Regression: a BudgetExceeded from the pre-call guard must NOT be cached as a completed
    search, so the character re-runs once the user raises the budget."""
    _agentic_cfg(settings)
    tracker = _CountingTracker(raise_on_check_n=1)   # raise on the very first pre-call check

    class _Client:
        class responses:
            @staticmethod
            def create(**kw):
                raise AssertionError("API must not be called after the budget guard trips")

    from ln2manga.cost import BudgetExceeded
    with pytest.raises(BudgetExceeded):
        references.resolve_reference_set("Subaru", settings, client=_Client(), tracker=tracker)
    assert not _searched_marker(settings, "Subaru").exists()   # NOT cached -> re-runs after raise


def test_verify_match_checks_budget_before_call():
    """Regression: _verify_match guards the budget BEFORE its billable vision call."""
    sc = {"verify_model": "gpt-5.4-mini", "cost_per_call": 0.03}
    tracker = _CountingTracker()

    class _Client:
        class responses:
            @staticmethod
            def create(**kw):
                assert tracker.checks == 1   # check() happened before the create()
                return SimpleNamespace(output_text="yes", output=[], usage=None)

    assert references._verify_match(_png_bytes(), "Subaru", "desc", sc, _Client(), tracker) is True
    assert tracker.checks == 1


def test_verify_match_fails_closed_when_verify_call_errors():
    """Fail CLOSED: when the vision verify call itself raises, _verify_match must REJECT
    (return False) so an unverifiable, possibly wrong-character image is never anchored."""
    sc = {"verify_model": "gpt-5.4-mini", "cost_per_call": 0.0}

    class _Client:
        class responses:
            @staticmethod
            def create(**kw):
                raise RuntimeError("transient verify outage")

    assert references._verify_match(_png_bytes(), "Petra", "desc", sc, _Client()) is False


def test_search_verify_call_error_keeps_nothing(settings, monkeypatch):
    """End-to-end fail-closed: if the vision verify call errors for every candidate, nothing is
    kept and the resolve returns [] -> the character gracefully falls back to the AI sheet."""
    object.__setattr__(settings, "references",
                       {"enabled": True, "sources": {}, "external_file": "config/__none__.yaml",
                        "search": {"enabled": True, "tool": "web_search", "max_urls": 3,
                                   "max_keep": 3, "min_bytes": 1, "verify": True,
                                   "agentic": False, "cost_per_call": 0.0}})
    big = _png_bytes(size=(256, 384))
    monkeypatch.setattr(references.requests, "get",
                        lambda u, headers=None, timeout=None:
                        SimpleNamespace(content=big, raise_for_status=lambda: None))

    class _Responses:
        def create(self, **kw):
            if isinstance(kw.get("input"), str):   # search call -> return a candidate URL
                return SimpleNamespace(output_text="https://example.com/petra.png",
                                       output=[], usage=None)
            raise RuntimeError("transient verify outage")   # vision verify -> errors

    client = SimpleNamespace(responses=_Responses())
    out = references.resolve_reference_set("Petra", settings, client=client)
    assert out == []   # rejected -> nothing kept -> graceful fallback (no crash)


def test_verify_budget_exceeded_propagates_and_skips_marker(settings, monkeypatch):
    """Regression: a verify-time BudgetExceeded propagates out of the agentic loop and leaves the
    search UNMARKED (overshoot is capped at the documented hard limit, not silently cached)."""
    _agentic_cfg(settings, cost_per_call=0.0)   # round guards never trip; only the verify guard does
    # First check() is the round-1 pre-call guard (cost 0.0); the second is the verify guard.
    tracker = _CountingTracker(raise_on_check_n=2)

    fc = SimpleNamespace(type="function_call", name="verify_image", call_id="c1",
                         arguments=json.dumps({"url": "https://example.com/official.png"}))

    def create(**kw):
        if "tools" in kw:                               # agentic round: ask to verify a URL
            return SimpleNamespace(output=[fc], output_text="", id="r1", usage=None)
        return SimpleNamespace(output_text="yes", output=[], usage=None)   # vision verify

    monkeypatch.setattr(references.requests, "get",
                        lambda u, headers=None, timeout=None:
                        SimpleNamespace(content=_png_bytes(size=(256, 384)),
                                        raise_for_status=lambda: None))
    from ln2manga.cost import BudgetExceeded
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    with pytest.raises(BudgetExceeded):
        references.resolve_reference_set("Subaru", settings, client=client, tracker=tracker)
    assert not _searched_marker(settings, "Subaru").exists()


def test_agentic_per_round_budget_exceeded_propagates_and_cleans_partial_set(settings, monkeypatch):
    """Regression: a BudgetExceeded from the PER-ROUND pre-call guard (after round 1 verified a
    member but before reaching max_keep) must PROPAGATE (not be swallowed-then-marked) AND must
    unlink the partial `{slug}.set*.png` members. Otherwise the search is either marked complete
    or short-circuits on the leftover partial set on the next run, and never resumes (audit)."""
    _agentic_cfg(settings, max_keep=2, cost_per_call=0.03)
    # check #1 = round-1 guard; #2 = verify guard (verifies one member -> set0.png); #3 = per-round
    # guard before the continuation -> raise there, with one partial member already on disk.
    tracker = _CountingTracker(raise_on_check_n=3)

    fc = SimpleNamespace(type="function_call", name="verify_image", call_id="c1",
                         arguments=json.dumps({"url": "https://example.com/official.png"}))

    def create(**kw):
        if "tools" in kw:                               # agentic round: ask to verify a URL
            return SimpleNamespace(output=[fc], output_text="", id="r1", usage=None)
        return SimpleNamespace(output_text="yes", output=[], usage=None)   # vision verify -> yes

    monkeypatch.setattr(references.requests, "get",
                        lambda u, headers=None, timeout=None:
                        SimpleNamespace(content=_png_bytes(size=(256, 384)),
                                        raise_for_status=lambda: None))
    from ln2manga.cost import BudgetExceeded
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    with pytest.raises(BudgetExceeded):
        references.resolve_reference_set("Subaru", settings, client=client, tracker=tracker)
    refs_dir, slug = _refs_dir(settings, "Subaru")
    assert not _searched_marker(settings, "Subaru").exists()         # UNMARKED -> resumes later
    assert list(refs_dir.glob(f"{slug}.set*.png")) == []             # partial set cleaned up


def test_verify_budget_exceeded_does_not_leave_reusable_partial_set(settings, monkeypatch):
    """Regression (complement of the per-round fix): a verify-time BudgetExceeded that fires AFTER
    one member was already materialized must also unlink the partial `{slug}.set*.png`, so the
    next run does not short-circuit on the leftover set and skip the resume."""
    _agentic_cfg(settings, max_keep=2, cost_per_call=0.0)   # round guards (cost 0) never trip
    # check #1 = round-1 guard (cost 0, no raise); #2 = first verify (succeeds -> set0.png on disk);
    # #3 = second verify guard -> raises mid-loop with a partial set already materialized.
    tracker = _CountingTracker(raise_on_check_n=3)

    fc1 = SimpleNamespace(type="function_call", name="verify_image", call_id="c1",
                          arguments=json.dumps({"url": "https://example.com/a.png"}))
    fc2 = SimpleNamespace(type="function_call", name="verify_image", call_id="c2",
                          arguments=json.dumps({"url": "https://example.com/b.png"}))

    def create(**kw):
        if "tools" in kw:                               # one round asks to verify TWO urls
            return SimpleNamespace(output=[fc1, fc2], output_text="", id="r1", usage=None)
        return SimpleNamespace(output_text="yes", output=[], usage=None)   # vision verify -> yes

    monkeypatch.setattr(references.requests, "get",
                        lambda u, headers=None, timeout=None:
                        SimpleNamespace(content=_png_bytes(size=(256, 384)),
                                        raise_for_status=lambda: None))
    from ln2manga.cost import BudgetExceeded
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    with pytest.raises(BudgetExceeded):
        references.resolve_reference_set("Subaru", settings, client=client, tracker=tracker)
    refs_dir, slug = _refs_dir(settings, "Subaru")
    assert not _searched_marker(settings, "Subaru").exists()
    assert list(refs_dir.glob(f"{slug}.set*.png")) == []   # no reusable partial set left behind


def test_single_shot_budget_exceeded_cleans_cand_staging(settings, monkeypatch):
    """Regression: when the single-shot verify guard raises BudgetExceeded mid-loop, the just-
    fetched `{slug}.cand*.png` staging files are swept (no orphaned disk litter), and the search
    is left UNMARKED with no set*.png so it resumes once budget is raised."""
    object.__setattr__(settings, "references",
                       {"enabled": True, "mode": "stylize", "sources": {},
                        "external_file": "config/__none__.yaml",
                        "search": {"enabled": True, "tool": "web_search", "max_urls": 3,
                                   "max_keep": 2, "min_bytes": 1, "verify": True,
                                   "agentic": False, "cost_per_call": 0.0}})
    # check #1 = search_reference_urls guard (cost 0.0, no raise); #2 = first verify guard -> raise,
    # by which point cand0.png has already been fetched to disk.
    tracker = _CountingTracker(raise_on_check_n=2)
    big = _png_bytes(size=(256, 384))
    monkeypatch.setattr(references.requests, "get",
                        lambda u, headers=None, timeout=None:
                        SimpleNamespace(content=big, raise_for_status=lambda: None))
    from ln2manga.cost import BudgetExceeded
    client = _io_aware_client("https://example.com/a.png\nhttps://example.com/b.png", "yes")
    with pytest.raises(BudgetExceeded):
        references.resolve_reference_set("Subaru", settings, client=client, tracker=tracker)
    refs_dir, slug = _refs_dir(settings, "Subaru")
    assert list(refs_dir.glob(f"{slug}.cand*.png")) == []   # staging litter swept
    assert list(refs_dir.glob(f"{slug}.set*.png")) == []
    assert not _searched_marker(settings, "Subaru").exists()


def test_resolve_url_network_error_returns_none(settings, monkeypatch):
    def boom(*a, **k):
        raise references.requests.RequestException("network down")

    monkeypatch.setattr(references.requests, "get", boom)
    _enable_refs(settings, sources={"Otto": "https://example.com/otto.png"})

    assert references.resolve_reference("Otto", settings) is None


# ── BUG 1: resolve_reference COPIES (does not move) the cached set so it survives reuse ─────────
def _refs_dir(settings, name):
    char = references.bible.get_character(name)
    canon = char.name if char else name
    return settings.cache_dir("refs"), references._slug(canon)


def test_resolve_reference_copies_set_and_subsequent_set_is_anchored(settings, monkeypatch):
    """BUG 1: after a search yields {slug}.set0.png, resolve_reference must leave BOTH set0.png
    AND the canonical {slug}.png on disk (copy, not move), so a later resolve_reference_set reuses
    the cached set and the character stays ANCHORED (not [])."""
    object.__setattr__(settings, "references",
                       {"enabled": True, "mode": "stylize", "sources": {},
                        "external_file": "config/__none__.yaml",
                        "search": {"enabled": True, "tool": "web_search", "max_urls": 3,
                                   "max_keep": 1, "min_bytes": 1, "verify": False,
                                   "agentic": False, "cost_per_call": 0.0}})
    big = _png_bytes(size=(256, 384))
    monkeypatch.setattr(references.requests, "get",
                        lambda u, headers=None, timeout=None:
                        SimpleNamespace(content=big, raise_for_status=lambda: None))

    refs_dir, slug = _refs_dir(settings, "Roswaal")
    out = references.resolve_reference(
        "Roswaal", settings, client=_search_client("https://example.com/roswaal.png"))
    assert out is not None and out.exists()
    set0 = refs_dir / f"{slug}.set0.png"
    base = refs_dir / f"{slug}.png"
    assert set0.exists(), "set0.png must SURVIVE (copy, not move) for later reuse"
    assert base.exists() and out == base

    # a subsequent resolve_reference_set reuses the cached set -> anchored, NOT []
    again = references.resolve_reference_set("Roswaal", settings, client=_search_client(""))
    assert again == [set0]


def _resolved_with_search(settings, monkeypatch, name="Roswaal", url="https://example.com/x.png"):
    """Run resolve_reference via a one-shot search and return (canonical dest, set0 source)."""
    object.__setattr__(settings, "references",
                       {"enabled": True, "mode": "stylize", "sources": {},
                        "external_file": "config/__none__.yaml",
                        "search": {"enabled": True, "tool": "web_search", "max_urls": 3,
                                   "max_keep": 1, "min_bytes": 1, "verify": False,
                                   "agentic": False, "cost_per_call": 0.0}})
    big = _png_bytes(size=(256, 384))
    monkeypatch.setattr(references.requests, "get",
                        lambda u, headers=None, timeout=None:
                        SimpleNamespace(content=big, raise_for_status=lambda: None))
    refs_dir, slug = _refs_dir(settings, name)
    dest = references.resolve_reference(name, settings, client=_search_client(url))
    return dest, refs_dir / f"{slug}.set0.png"


def test_resolve_reference_dest_is_byte_identical_to_set0(settings, monkeypatch):
    """The canonical {slug}.png is byte-identical to its set0.png source (behavior unchanged)."""
    dest, set0 = _resolved_with_search(settings, monkeypatch)
    assert dest is not None and dest.exists() and set0.exists()
    assert dest.read_bytes() == set0.read_bytes()


def test_resolve_reference_dest_shares_inode_with_set0(settings, monkeypatch):
    """When hardlinks are supported, the canonical {slug}.png shares set0.png's inode (zero bytes)."""
    dest, set0 = _resolved_with_search(settings, monkeypatch)
    assert os.stat(dest).st_ino == os.stat(set0).st_ino


def test_resolve_reference_copy_fallback_when_hardlink_unsupported(settings, monkeypatch):
    """When os.link raises OSError, the canonical {slug}.png is still produced as a byte-identical
    real copy (a distinct inode), and set0.png still survives for reuse."""
    monkeypatch.setattr(references.os, "link",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no hardlinks")))
    dest, set0 = _resolved_with_search(settings, monkeypatch)
    assert dest is not None and dest.exists() and set0.exists()
    assert dest.read_bytes() == set0.read_bytes()
    assert os.stat(dest).st_ino != os.stat(set0).st_ino   # real copy, distinct inode


def test_reuse_check_falls_back_to_base_png(settings):
    """BUG 1(b): when the .searched marker exists but no {slug}.set*.png remains (a base-name-only
    cache from an earlier resolve_reference move), reuse the base {slug}.png instead of returning
    [] (which would re-treat the character as 'searched, found nothing')."""
    object.__setattr__(settings, "references",
                       {"enabled": True, "mode": "stylize", "sources": {},
                        "external_file": "config/__none__.yaml",
                        "search": {"enabled": True, "tool": "web_search", "max_keep": 3,
                                   "min_bytes": 1, "verify": False, "agentic": False,
                                   "cost_per_call": 0.0}})
    refs_dir, slug = _refs_dir(settings, "Roswaal")
    refs_dir.mkdir(parents=True, exist_ok=True)
    base = refs_dir / f"{slug}.png"
    base.write_bytes(_png_bytes(size=(256, 384)))
    (refs_dir / f"{slug}.searched").write_text("", encoding="utf-8")   # marked, but no set*.png

    out = references.resolve_reference_set("Roswaal", settings, client=_search_client(""))
    assert out == [base]

    # and with NEITHER set*.png NOR base present, .searched still means "found nothing" -> []
    base.unlink()
    assert references.resolve_reference_set("Roswaal", settings, client=_search_client("")) == []


# ── BUG 2: searched-image quality (200px floor + prefer-largest) ───────────────────────────────
def test_searched_image_sub_200px_rejected_over_200px_passes(settings):
    """BUG 2(a): for SEARCHED images (min_bytes>0) a sub-200px image is rejected while a >=200px
    one passes. Explicit user refs (min_bytes=0) are unaffected (covered elsewhere)."""
    refs_dir = settings.cache_dir("refs")
    refs_dir.mkdir(parents=True, exist_ok=True)
    small = _png_bytes(size=(180, 260), color=90)    # min dim 180 < 200 -> rejected
    big = _png_bytes(size=(220, 300), color=90)       # min dim 220 >= 200 -> accepted
    assert references._save_png(small, refs_dir / "small.png", min_bytes=1) is None
    assert references._save_png(big, refs_dir / "big.png", min_bytes=1) is not None
    # trusted (min_bytes=0): even a tiny image is kept (unchanged behavior)
    tiny = _png_bytes(size=(40, 60))
    assert references._save_png(tiny, refs_dir / "trusted.png", min_bytes=0) is not None


def test_single_shot_prefers_largest_verified(settings, monkeypatch):
    """BUG 2(b): when more candidates verify than `keep`, the LARGEST by pixel area is kept (a
    full-size official design beats a smaller-but-still-valid thumbnail)."""
    object.__setattr__(settings, "references",
                       {"enabled": True, "mode": "stylize", "sources": {},
                        "external_file": "config/__none__.yaml",
                        "search": {"enabled": True, "tool": "web_search", "max_urls": 8,
                                   "max_keep": 1, "min_bytes": 1, "verify": True,
                                   "agentic": False, "cost_per_call": 0.0}})
    small_url = "https://example.com/small.png"
    big_url = "https://example.com/big.png"
    small = _png_bytes(size=(220, 300))    # both pass the 200px floor and verify
    big = _png_bytes(size=(512, 768))

    def fake_get(u, headers=None, timeout=None):
        return SimpleNamespace(content=(big if u == big_url else small),
                               raise_for_status=lambda: None)

    monkeypatch.setattr(references.requests, "get", fake_get)
    # small URL listed FIRST, so a non-size-aware loop would have kept it; prefer-largest keeps big
    out = references.resolve_reference_set(
        "Subaru", settings, client=_io_aware_client(f"{small_url}\n{big_url}", "yes"))
    assert len(out) == 1 and out[0].exists()
    from PIL import Image as _Image
    with _Image.open(out[0]) as im:
        assert im.size == (512, 768)        # the LARGER verified design won


# ── charsheet integration ────────────────────────────────────────────────────
class _ExplodingImages:
    """images.generate/edit must NOT be called in raw mode."""

    def generate(self, **kw):
        raise AssertionError("images.generate must not be called when a raw ref is provided")

    def edit(self, **kw):
        raise AssertionError("images.edit must not be called when a raw ref is provided")


class _ExplodingClient:
    def __init__(self):
        self.images = _ExplodingImages()


def _tracker(settings):
    return CostTracker(settings.budget["max_usd"], settings.ledger_path,
                       settings.prices_usd, 10_000, dry_run=False)


def test_charsheet_raw_ref_skips_image_api(settings, tmp_path):
    src = tmp_path / "subaru.png"
    src.write_bytes(_png_bytes())
    _enable_refs(settings, mode="raw", sources={"Subaru": str(src)})

    specs = [SimpleNamespace(characters_present=["Subaru"])]
    client = _ExplodingClient()
    cache = Cache(settings.cache_dir)

    sheets = charsheet.run(client, settings, _tracker(settings), cache, specs, chapter_number=99)

    assert "Subaru" in sheets
    # dedup (#15): raw mode points the sheet at the resolved reference itself (no char_<name>.png copy)
    Image.open(sheets["Subaru"]).load()   # the sheet path is a valid image


def test_charsheet_disabled_does_not_use_refs(settings, tmp_path, monkeypatch):
    # references disabled -> resolve_reference must not even be consulted
    src = tmp_path / "subaru.png"
    src.write_bytes(_png_bytes())
    object.__setattr__(settings, "references",
                       {"enabled": False, "mode": "raw", "sources": {"Subaru": str(src)}})

    called = {"n": 0}
    real_resolve = references.resolve_reference

    def spy(name, s):
        called["n"] += 1
        return real_resolve(name, s)

    monkeypatch.setattr(charsheet.references, "resolve_reference", spy)

    # a recording client whose generate returns a valid png (default AI path)
    import base64

    def img_response(png):
        return SimpleNamespace(
            data=[SimpleNamespace(b64_json=base64.b64encode(png).decode(), url=None)],
            usage=SimpleNamespace(input_tokens=0, output_tokens=0, total_tokens=0))

    class GenClient:
        class images:
            @staticmethod
            def generate(**kw):
                return img_response(_png_bytes())

            @staticmethod
            def edit(**kw):
                raise AssertionError("edit not expected")

    specs = [SimpleNamespace(characters_present=["Subaru"])]
    cache = Cache(settings.cache_dir)
    sheets = charsheet.run(GenClient(), settings, _tracker(settings), cache, specs, 99)

    assert called["n"] == 0          # refs disabled -> resolve never called
    assert "Subaru" in sheets        # still produced via the AI path
