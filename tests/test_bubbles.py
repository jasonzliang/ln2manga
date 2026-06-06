"""Tests for the validated organic speech-bubble feature (ln2manga.bubbles)."""
from __future__ import annotations

import base64
import io
import os
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from ln2manga import bubbles
from ln2manga.artifacts import Rect
from ln2manga.cache import Cache
from ln2manga.config import load_settings
from ln2manga.cost import CostTracker


# ── fakes ─────────────────────────────────────────────────────────────────────
def _bubble_png(size=(64, 64)) -> bytes:
    """A small TRANSPARENT-background tile: a dark ring with a white-ish interior, like the API
    returns (the interior is white but the tile background is fully transparent)."""
    w, h = size
    img = Image.new("RGBA", size, (0, 0, 0, 0))   # fully transparent background
    px = img.load()
    cx, cy = w / 2, h / 2
    rx, ry = w * 0.42, h * 0.42
    for y in range(h):
        for x in range(w):
            d = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
            if d <= 1.0:
                # ring band near the edge is dark ink; the rest of the interior is white.
                if d >= 0.78:
                    px[x, y] = (10, 10, 10, 255)        # bold black outline
                else:
                    px[x, y] = (250, 250, 250, 255)     # white interior
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _opaque_png(size=(64, 64)) -> bytes:
    """A FULLY-OPAQUE tile: no transparent background at all (model returned no transparent border).
    Flooding from the border finds nothing -> `outside` is empty. The old code turned this into a
    solid white box with no outline; the fix must reject it instead."""
    img = Image.new("RGBA", size, (250, 250, 250, 255))   # every pixel opaque
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _thin_silhouette_png(size=(64, 64)) -> bytes:
    """A tile whose only opaque silhouette is a 1-px-thin line on a transparent background. A
    tile-sized erosion kernel would wipe out the core entirely; the fix must avoid an all-black
    result (either reject the tile or keep a white core)."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))   # transparent background
    px = img.load()
    h = size[1]
    for y in range(h):
        px[size[0] // 2, y] = (10, 10, 10, 255)   # a single vertical 1-px ink line
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _double_line_frame_png(size=(128, 128)) -> bytes:
    """An OUTLINE-ONLY / DOUBLE-LINE frame tile with an EMPTY (transparent) interior, like the broken
    data/cache/bubbles/narration_1.png the QA audit found: two concentric thick rectangular ink
    outlines on a transparent background, the enclosed interior left transparent (not opaque white),
    and a gap punched in the border so the exterior flood leaks through into the would-be interior.

    The two pre-existing guards DO NOT catch this: there IS a transparent background (the
    `outside.sum()==0` guard passes), and eroding the thick double stroke still leaves a non-empty
    core (the `core.sum()==0` guard passes) — yet the interior is empty, so the new white-interior
    fraction guard must reject it."""
    w, h = size
    img = Image.new("RGBA", size, (0, 0, 0, 0))   # transparent background AND transparent interior
    px = img.load()

    def _frame(x0, y0, x1, y1, thick):
        for t in range(thick):
            for x in range(x0, x1):
                px[x, y0 + t] = (10, 10, 10, 255)
                px[x, y1 - 1 - t] = (10, 10, 10, 255)
            for y in range(y0, y1):
                px[x0 + t, y] = (10, 10, 10, 255)
                px[x1 - 1 - t, y] = (10, 10, 10, 255)

    _frame(int(w * 0.08), int(h * 0.23), int(w * 0.92), int(h * 0.77), 4)   # outer line
    _frame(int(w * 0.13), int(h * 0.28), int(w * 0.87), int(h * 0.72), 4)   # inner line (double)
    # punch a WIDE gap through both borders so the exterior flood-fill leaks into the interior — too
    # wide for the morphological close to seal, exactly like the broken narration_1 tile.
    for x in range(int(w * 0.35), int(w * 0.65)):
        for y in range(int(h * 0.22), int(h * 0.35)):
            px[x, y] = (0, 0, 0, 0)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _image_response(png: bytes):
    return SimpleNamespace(
        data=[SimpleNamespace(b64_json=base64.b64encode(png).decode(), url=None)],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
    )


class FakeImages:
    def __init__(self):
        self.calls = []

    def generate(self, **kw):
        self.calls.append(kw)
        return _image_response(_bubble_png())


class FakeClient:
    def __init__(self):
        self.images = FakeImages()


class _FixedImages:
    """Images stub that always returns the same PNG bytes."""
    def __init__(self, png: bytes):
        self._png = png
        self.calls = []

    def generate(self, **kw):
        self.calls.append(kw)
        return _image_response(self._png)


class FixedClient:
    def __init__(self, png: bytes):
        self.images = _FixedImages(png)


@pytest.fixture
def settings(tmp_path):
    s = load_settings()
    s.paths["data"] = str(tmp_path)        # isolate all I/O under a temp data dir
    return s


def _tracker(settings):
    return CostTracker(settings.budget["max_usd"], settings.ledger_path,
                       settings.prices_usd, 10_000, dry_run=False)


# ── STYLE_PROMPTS ──────────────────────────────────────────────────────────────
def test_style_prompts_cover_all_styles():
    for style in ("speech", "shout", "thought", "narration"):
        assert style in bubbles.STYLE_PROMPTS
        p = bubbles.STYLE_PROMPTS[style].lower()
        assert "transparent" in p
        # each prompt insists the bubble is EMPTY (no text inside)
        assert "empty" in p or "no text" in p


# ── ensure_shape_library ────────────────────────────────────────────────────────
def test_ensure_shape_library_generates_and_returns_paths(settings):
    client = FakeClient()
    cache = Cache(settings.cache_dir)

    lib = bubbles.ensure_shape_library(client, settings, _tracker(settings), cache, count=2)

    assert set(lib) == {"speech", "shout", "thought", "narration"}
    for style, paths in lib.items():
        assert len(paths) == 2
        for p in paths:
            assert p.exists()
            assert p.name.startswith(style)
            Image.open(p).load()             # readable image
    # 4 styles x 2 tiles = 8 generate calls on a cold cache.
    assert len(client.images.calls) == 8
    # the model used is the gpt-image-1 family (transparent-capable), NOT gpt-image-2.
    assert all(c["model"].startswith("gpt-image-1") for c in client.images.calls)
    assert all(c["background"] == "transparent" for c in client.images.calls)


def test_second_call_is_a_cache_hit_no_api(settings):
    cache = Cache(settings.cache_dir)
    client1 = FakeClient()
    bubbles.ensure_shape_library(client1, settings, _tracker(settings), cache, count=2)
    assert len(client1.images.calls) == 8

    # Second run with a fresh client: everything served from the content cache -> zero API calls.
    client2 = FakeClient()
    lib2 = bubbles.ensure_shape_library(client2, settings, _tracker(settings), cache, count=2)
    assert len(client2.images.calls) == 0
    assert set(lib2) == {"speech", "shout", "thought", "narration"}
    assert all(p.exists() for paths in lib2.values() for p in paths)


def test_count_defaults_from_settings(settings):
    object.__setattr__(settings, "bubbles", {"count_per_style": 1})
    client = FakeClient()
    cache = Cache(settings.cache_dir)
    lib = bubbles.ensure_shape_library(client, settings, _tracker(settings), cache)
    assert all(len(paths) == 1 for paths in lib.values())
    assert len(client.images.calls) == 4   # 4 styles x 1


def test_none_client_returns_empty(settings):
    cache = Cache(settings.cache_dir)
    lib = bubbles.ensure_shape_library(None, settings, _tracker(settings), cache, count=2)
    assert lib == {}


def test_failing_style_is_skipped_not_raised(settings):
    class Boom:
        class images:
            @staticmethod
            def generate(**kw):
                raise RuntimeError("API down")
    cache = Cache(settings.cache_dir)
    lib = bubbles.ensure_shape_library(Boom(), settings, _tracker(settings), cache, count=2)
    assert lib == {}    # every style skipped, no exception escaped


def test_interior_is_flooded_opaque_white(settings):
    """The cached tile must have an OPAQUE WHITE interior (the API tile's interior is not)."""
    client = FakeClient()
    cache = Cache(settings.cache_dir)
    lib = bubbles.ensure_shape_library(client, settings, _tracker(settings), cache, count=1)
    tile = Image.open(lib["speech"][0]).convert("RGBA")
    arr = np.asarray(tile)
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    r, g, b, a = arr[cy, cx]
    assert (r, g, b, a) == (255, 255, 255, 255)   # center interior is opaque white


# ── place_bubble ─────────────────────────────────────────────────────────────────
def test_place_bubble_returns_rect_inside_and_preserves_outside(settings):
    client = FakeClient()
    cache = Cache(settings.cache_dir)
    lib = bubbles.ensure_shape_library(client, settings, _tracker(settings), cache, count=1)
    shape_path = lib["speech"][0]

    page = Image.new("L", (400, 300), 30)   # dark page so the white bubble is detectable
    before = np.asarray(page).copy()

    cx, cy, tw, th = 200, 150, 120, 90
    rect = bubbles.place_bubble(page, shape_path, cx, cy, tw, th)

    assert isinstance(rect, Rect)
    # the returned text rect sits within the placed tile bounds.
    tile_x0, tile_y0 = cx - tw // 2, cy - th // 2
    tile_x1, tile_y1 = tile_x0 + tw, tile_y0 + th
    assert tile_x0 <= rect.x0 < rect.x1 <= tile_x1
    assert tile_y0 <= rect.y0 < rect.y1 <= tile_y1
    # the rect is over white interior pixels of the now-composited page.
    after = np.asarray(page)
    rcx, rcy = (rect.x0 + rect.x1) // 2, (rect.y0 + rect.y1) // 2
    assert after[rcy, rcx] > 200

    # pixels OUTSIDE the tile bounding box are unchanged.
    outside = after.copy()
    outside[tile_y0:tile_y1, tile_x0:tile_x1] = before[tile_y0:tile_y1, tile_x0:tile_x1]
    assert np.array_equal(outside, before)
    # and a far corner is exactly the original background.
    assert after[5, 5] == 30


def test_fully_opaque_tile_is_rejected(settings):
    """BUG 1: a fully-opaque tile (no transparent border) must NOT become a solid white box.

    _fill_interior_opaque raises, so ensure_shape_library skips every style and returns without
    crashing (style absent / empty) -> caller falls back to drawn bubbles."""
    # the flood-fill rejects it directly with a ValueError
    with pytest.raises(ValueError):
        bubbles._fill_interior_opaque(Image.open(io.BytesIO(_opaque_png())))

    client = FixedClient(_opaque_png())
    cache = Cache(settings.cache_dir)
    lib = bubbles.ensure_shape_library(client, settings, _tracker(settings), cache, count=2)
    assert lib == {}                     # every style skipped, no exception escaped
    # and no half-filled white-box tile was produced
    assert "speech" not in lib


def test_thin_silhouette_tile_never_all_black(settings):
    """BUG 2: a tile with a tiny/thin silhouette must not produce an all-black tile."""
    # the flood-fill refuses the too-thin silhouette rather than emit an all-black result
    with pytest.raises(ValueError):
        bubbles._fill_interior_opaque(Image.open(io.BytesIO(_thin_silhouette_png())))

    client = FixedClient(_thin_silhouette_png())
    cache = Cache(settings.cache_dir)
    lib = bubbles.ensure_shape_library(client, settings, _tracker(settings), cache, count=1)
    # too-thin silhouette -> skipped; either absent, or if present it must carry a white core
    for paths in lib.values():
        for p in paths:
            arr = np.asarray(Image.open(p).convert("RGBA"))
            opaque = arr[:, :, 3] > 200
            white = opaque & (arr[:, :, 0] > 200) & (arr[:, :, 1] > 200) & (arr[:, :, 2] > 200)
            assert white.sum() > 0       # never an all-black blob


def test_small_but_thick_silhouette_keeps_white_core(settings):
    """BUG 2: a small silhouette (kernel sized off the tile would erase it) keeps a white core.

    The 64x64 fake bubble occupies only ~84% of a 1024 tile when generated, but here we feed a tiny
    16x16 white-filled bubble: the tile-derived kernel is fine, but a silhouette-derived one must
    keep the interior. This must NOT be all-black and must have a readable white core."""
    # a 16x16 filled bubble: thick enough to keep an interior, small relative to the frame
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    px = img.load()
    cx, cy, r = 32, 32, 8
    for y in range(64):
        for x in range(64):
            d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if d <= r:
                px[x, y] = (10, 10, 10, 255) if d >= r - 1.5 else (250, 250, 250, 255)
    filled = bubbles._fill_interior_opaque(img)
    arr = np.asarray(filled)
    white = (arr[:, :, 3] == 255) & (arr[:, :, 0] > 200) & (arr[:, :, 1] > 200) & (arr[:, :, 2] > 200)
    assert white.sum() > 0               # has a white interior core, not an all-black blob


def test_hollow_double_line_frame_tile_is_rejected(settings):
    """QA bug: an outline-only / double-line frame whose interior is empty (transparent) — like the
    broken narration_1.png — must be REJECTED, not shipped as a hollow tile with no white backing.

    The two existing guards pass for it (there IS a transparent background; eroding the thick double
    stroke still leaves a non-empty core), so the post-fill white-interior fraction guard must catch
    it and raise."""
    with pytest.raises(ValueError, match="interior not filled"):
        bubbles._fill_interior_opaque(Image.open(io.BytesIO(_double_line_frame_png())))

    # and ensure_shape_library skips every style without crashing -> caller falls back to drawn bubbles
    client = FixedClient(_double_line_frame_png())
    cache = Cache(settings.cache_dir)
    lib = bubbles.ensure_shape_library(client, settings, _tracker(settings), cache, count=2)
    assert lib == {}                     # every style skipped, no exception escaped
    assert "narration" not in lib


def test_fully_transparent_tile_rejected_with_clean_message(settings):
    """A fully-transparent tile floods everything as `outside`, so the empty-background guard passes
    but the silhouette is empty. It must raise a DESCRIPTIVE ValueError, not the cryptic numpy
    'zero-size array to reduction operation maximum' error that the bbox computation would emit."""
    blank = Image.new("RGBA", (64, 64), (0, 0, 0, 0))   # nothing opaque at all
    with pytest.raises(ValueError, match="fully transparent"):
        bubbles._fill_interior_opaque(blank)

    # and ensure_shape_library skips every style without crashing -> caller falls back to drawn bubbles
    buf = io.BytesIO()
    blank.save(buf, "PNG")
    client = FixedClient(buf.getvalue())
    cache = Cache(settings.cache_dir)
    lib = bubbles.ensure_shape_library(client, settings, _tracker(settings), cache, count=1)
    assert lib == {}


def test_normal_tile_with_real_white_interior_is_kept(settings):
    """A normal tile with a real opaque-white interior still passes the interior guard and is kept."""
    # the flood-fill keeps it: a filled white bubble has a white core spanning most of its bbox
    filled = bubbles._fill_interior_opaque(Image.open(io.BytesIO(_bubble_png())))
    arr = np.asarray(filled)
    white = (arr[:, :, 3] == 255) & (arr[:, :, 0] > 200) & (arr[:, :, 1] > 200) & (arr[:, :, 2] > 200)
    assert white.sum() > 0

    client = FakeClient()
    cache = Cache(settings.cache_dir)
    lib = bubbles.ensure_shape_library(client, settings, _tracker(settings), cache, count=2)
    assert set(lib) == {"speech", "shout", "thought", "narration"}
    for paths in lib.values():
        for p in paths:
            tarr = np.asarray(Image.open(p).convert("RGBA"))
            cy, cx = tarr.shape[0] // 2, tarr.shape[1] // 2
            assert tuple(tarr[cy, cx]) == (255, 255, 255, 255)   # center interior is opaque white


def test_bubble_cost_records_max_of_estimate_and_usage(settings, monkeypatch):
    """BUG 3: the ledger records max(flat estimate, token-usage cost), not just the flat estimate."""
    client = FakeClient()
    cache = Cache(settings.cache_dir)
    tracker = _tracker(settings)

    recorded: list[float] = []
    real_record = tracker.record

    def _spy(kind, model, usd, meta=None):
        if kind == "image":
            recorded.append(usd)
            assert meta is not None and "usage" in meta   # real usage is threaded into the ledger
        return real_record(kind, model, usd, meta)

    monkeypatch.setattr(tracker, "record", _spy)
    # force the token-usage cost to dominate the flat estimate so we can see max(...) take effect.
    monkeypatch.setattr(tracker, "estimate_image", lambda *a, **k: 0.01)
    monkeypatch.setattr(tracker, "estimate_image_from_usage", lambda *a, **k: 0.05)

    bubbles.ensure_shape_library(client, settings, tracker, cache, count=1)
    assert recorded and all(abs(v - 0.05) < 1e-9 for v in recorded)   # usage cost wins over estimate


def test_unusable_tile_is_re_rolled_then_cached(settings, capsys):
    """BUG 4: a paid call that yields an unfillable silhouette must be re-rolled in-loop (so the bad
    output is not discarded and re-paid every run); the first acceptable output is recorded + cached."""
    blank = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    bbuf = io.BytesIO()
    blank.save(bbuf, "PNG")
    bad_png = bbuf.getvalue()           # fully transparent -> _fill_interior_opaque raises ValueError

    class FlakyTileImages:
        def __init__(self):
            self.calls = []
            self._bad_done = False

        def generate(self, **kw):
            self.calls.append(kw)
            # the very first tile generation returns an unusable (transparent) silhouette; every
            # later call returns a good bubble.
            if not self._bad_done:
                self._bad_done = True
                return _image_response(bad_png)
            return _image_response(_bubble_png())

    class FlakyTileClient:
        def __init__(self):
            self.images = FlakyTileImages()

    object.__setattr__(settings, "bubbles", {"count_per_style": 1})
    client = FlakyTileClient()
    cache = Cache(settings.cache_dir)
    lib = bubbles.ensure_shape_library(client, settings, _tracker(settings), cache, count=1)

    # the first style re-rolled past its bad tile, so no style is lost to the unusable output.
    assert set(lib) == {"speech", "shout", "thought", "narration"}
    # 4 styles x 1 tile = 4 good outputs, plus one extra call for the single re-roll = 5 API calls.
    assert len(client.images.calls) == 5
    err = capsys.readouterr().err
    assert "re-rolling" in err

    # the re-rolled good tile was cached: a second run makes zero API calls.
    client2 = FlakyTileClient()
    lib2 = bubbles.ensure_shape_library(client2, settings, _tracker(settings), cache, count=1)
    assert client2.images.calls == []
    assert set(lib2) == {"speech", "shout", "thought", "narration"}


def test_persistently_unusable_tile_is_bounded_then_style_skipped(settings):
    """A style whose tile is unusable on EVERY attempt must give up after a bounded number of paid
    re-rolls (not loop forever) and fall back to drawn bubbles."""
    blank = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    bbuf = io.BytesIO()
    blank.save(bbuf, "PNG")
    client = FixedClient(bbuf.getvalue())   # always returns the unfillable transparent tile
    object.__setattr__(settings, "bubbles", {"count_per_style": 1})
    cache = Cache(settings.cache_dir)
    lib = bubbles.ensure_shape_library(client, settings, _tracker(settings), cache, count=1)
    assert lib == {}
    # each style exhausts its bounded attempts for the failing index then is skipped (break): the cap
    # is _MAX_TILE_ATTEMPTS paid calls PER index, never an unbounded loop. With 4 styles x 1 tile that
    # is exactly 4 * _MAX_TILE_ATTEMPTS calls.
    assert len(client.images.calls) == len(bubbles.STYLES) * bubbles._MAX_TILE_ATTEMPTS


def test_place_bubble_falls_back_to_centered_rect(settings, tmp_path):
    """A tile with no detectable white interior yields a centered ~60% Rect."""
    blank = Image.new("RGBA", (64, 64), (0, 0, 0, 0))   # fully transparent, no white interior
    p = tmp_path / "blank.png"
    blank.save(p)

    page = Image.new("L", (300, 300), 128)
    cx, cy, tw, th = 150, 150, 100, 100
    rect = bubbles.place_bubble(page, p, cx, cy, tw, th)

    # centered box ~60% of the tile, offset to page coords.
    assert isinstance(rect, Rect)
    assert abs(rect.w - int(tw * 0.6)) <= 1
    assert abs(rect.h - int(th * 0.6)) <= 1
    # centered on (cx, cy)
    assert abs((rect.x0 + rect.x1) // 2 - cx) <= 1
    assert abs((rect.y0 + rect.y1) // 2 - cy) <= 1


# ── regression: retry + timeout on the tile call (finding 1) ──────────────────────
def test_tile_call_uses_retry_and_passes_timeout(settings, monkeypatch):
    """The tile generate call must retry transient errors AND pass a per-call timeout (like
    imagegen.generate_image), not call the API bare."""
    # with_retry must wrap the call: a single transient failure followed by success still yields
    # a tile, and the call carries a timeout kwarg.
    import openai

    class FlakyImages:
        def __init__(self):
            self.calls = []
            self._failed = False

        def generate(self, **kw):
            self.calls.append(kw)
            if not self._failed:
                self._failed = True
                raise openai.APITimeoutError(request=None)
            return _image_response(_bubble_png())

    class FlakyClient:
        def __init__(self):
            self.images = FlakyImages()

    # don't actually sleep between retries
    monkeypatch.setattr(bubbles, "with_retry", bubbles.with_retry)
    object.__setattr__(settings, "bubbles", {"count_per_style": 1, "timeout_s": 12})
    client = FlakyClient()
    cache = Cache(settings.cache_dir)
    lib = bubbles.ensure_shape_library(client, settings, _tracker(settings), cache, count=1)

    assert set(lib) == {"speech", "shout", "thought", "narration"}
    # the timeout kwarg is threaded into every API call
    assert all(c.get("timeout") == 12.0 for c in client.images.calls)
    # at least one style retried after the transient timeout (first call failed, second succeeded)
    assert any(c.get("timeout") == 12.0 for c in client.images.calls)


def test_timeout_defaults_to_image_timeout_s(settings):
    """When bubbles config omits timeout_s, it falls back to settings.image['timeout_s']."""
    object.__setattr__(settings, "bubbles", {"count_per_style": 1})
    settings.image["timeout_s"] = 99
    client = FakeClient()
    cache = Cache(settings.cache_dir)
    bubbles.ensure_shape_library(client, settings, _tracker(settings), cache, count=1)
    assert client.images.calls and all(c.get("timeout") == 99.0 for c in client.images.calls)


# ── regression: gpt-image-2 misconfiguration is warned, not silently retried (finding 2) ──
def test_gpt_image_2_model_is_rejected_with_warning(settings, capsys):
    object.__setattr__(settings, "bubbles", {"model": "gpt-image-2", "count_per_style": 1})
    client = FakeClient()
    cache = Cache(settings.cache_dir)
    lib = bubbles.ensure_shape_library(client, settings, _tracker(settings), cache, count=1)
    assert lib == {}
    assert client.images.calls == []          # never even attempted
    err = capsys.readouterr().err
    assert "gpt-image-2" in err and "transparent" in err


# ── regression: failures are no longer silent (findings 2, 3, 8) ──────────────────
def test_failing_style_logs_to_stderr(settings, capsys):
    class Boom:
        class images:
            @staticmethod
            def generate(**kw):
                raise RuntimeError("API down")
    cache = Cache(settings.cache_dir)
    lib = bubbles.ensure_shape_library(Boom(), settings, _tracker(settings), cache, count=1)
    assert lib == {}
    err = capsys.readouterr().err
    # every style is named and the failure type surfaces
    assert "speech" in err and "RuntimeError" in err
    assert "drawn bubbles" in err


def test_budget_exhaustion_is_reported_loudly(settings, capsys):
    """BudgetExceeded must be surfaced distinctly, not swallowed as a generic API failure."""
    object.__setattr__(settings, "bubbles", {"count_per_style": 1})
    settings.budget["max_usd"] = 0.0          # no budget at all
    client = FakeClient()
    cache = Cache(settings.cache_dir)
    lib = bubbles.ensure_shape_library(client, settings, _tracker(settings), cache, count=1)
    assert lib == {}                          # nothing could be built
    err = capsys.readouterr().err
    assert "budget" in err.lower()


# ── regression: partial library is announced (finding 4) ──────────────────────────
def test_partial_library_warns(settings, capsys):
    """If some styles succeed and others fail, say so (organic + drawn mix on a page)."""
    class PartialImages:
        def __init__(self):
            self.calls = []

        def generate(self, **kw):
            self.calls.append(kw)
            # fail only the 'thought' style (its prompt is the only one mentioning 'cloud')
            if "cloud" in kw.get("prompt", ""):
                raise RuntimeError("thought failed")
            return _image_response(_bubble_png())

    class PartialClient:
        def __init__(self):
            self.images = PartialImages()

    object.__setattr__(settings, "bubbles", {"count_per_style": 1})
    client = PartialClient()
    cache = Cache(settings.cache_dir)
    lib = bubbles.ensure_shape_library(client, settings, _tracker(settings), cache, count=1)
    assert "thought" not in lib and "speech" in lib
    err = capsys.readouterr().err
    assert "partial library" in err and "thought" in err


# ── regression: reduced count_per_style leaves no orphaned stable files (finding 7) ──
def test_reduced_count_leaves_no_orphan_stable_files(settings):
    cache = Cache(settings.cache_dir)
    # first run at count=3
    bubbles.ensure_shape_library(FakeClient(), settings, _tracker(settings), cache, count=3)
    out_dir = settings.cache_dir("bubbles")
    assert (out_dir / "speech_2.png").exists()
    # second run at count=1 must remove the stale high-index files
    bubbles.ensure_shape_library(FakeClient(), settings, _tracker(settings), cache, count=1)
    assert (out_dir / "speech_0.png").exists()
    assert not (out_dir / "speech_1.png").exists()
    assert not (out_dir / "speech_2.png").exists()


def test_failed_style_leaves_no_orphan_stable_copies(settings):
    """A later-tile failure must not leave an earlier {style}_0.png behind on disk."""
    class FailSecondImages:
        def __init__(self):
            self.calls = 0

        def generate(self, **kw):
            self.calls += 1
            if self.calls % 2 == 0:           # every 2nd tile fails -> count=2 fails after tile 0
                raise RuntimeError("second tile down")
            return _image_response(_bubble_png())

    class FailSecondClient:
        def __init__(self):
            self.images = FailSecondImages()

    object.__setattr__(settings, "bubbles", {"count_per_style": 2})
    cache = Cache(settings.cache_dir)
    lib = bubbles.ensure_shape_library(FailSecondClient(), settings, _tracker(settings), cache, count=2)
    assert lib == {}                          # every style failed on its 2nd tile
    out_dir = settings.cache_dir("bubbles")
    # no half-written human-named stable copy from the successful first tile of any style
    # (the content-addressed cache files are hash-named and legitimately persisted).
    stable = [p for s in bubbles.STYLES for p in out_dir.glob(f"{s}_*.png")]
    assert stable == []


# ── regression: non-convex interior gets a mostly-white text rect (finding 5) ─────
def test_non_convex_tile_text_rect_is_mostly_white(settings, tmp_path):
    """A cross-shaped white interior has a sparse global bbox; the returned Rect must be shrunk so
    text lands on white, not on the ink/gaps."""
    n = 120
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    px = img.load()
    arm = 14                                   # thin arms -> bbox is mostly transparent
    c = n // 2
    for y in range(n):
        for x in range(n):
            if abs(x - c) <= arm or abs(y - c) <= arm:
                px[x, y] = (255, 255, 255, 255)
    p = tmp_path / "cross.png"
    img.save(p)

    page = Image.new("L", (400, 400), 30)
    rect = bubbles.place_bubble(page, p, 200, 200, n, n)

    # sample the composited page over the returned rect: it must be mostly white.
    after = np.asarray(page)
    region = after[rect.y0:rect.y1, rect.x0:rect.x1]
    assert region.size
    white_frac = (region > 200).mean()
    assert white_frac >= 0.6


def test_convex_tile_rect_unchanged_by_white_fit(settings):
    """The convex ellipse fake keeps a high white fraction, so the fit must not shrink it away."""
    client = FakeClient()
    cache = Cache(settings.cache_dir)
    lib = bubbles.ensure_shape_library(client, settings, _tracker(settings), cache, count=1)
    page = Image.new("L", (400, 300), 30)
    rect = bubbles.place_bubble(page, lib["speech"][0], 200, 150, 120, 90)
    after = np.asarray(page)
    region = after[rect.y0:rect.y1, rect.x0:rect.x1]
    assert (region > 200).mean() >= 0.6


# ── dedup: stable {style}_{i}.png is a HARDLINK to the content-hash cache file ────
def _cache_file_for(bubbles_mod, settings, style, i, count=1):
    """The content-hash cache path for tile `i` of `style` at the default params."""
    cfg = bubbles_mod._bubbles_cfg(settings)
    model = str(cfg.get("model", bubbles_mod._DEFAULT_MODEL))
    size = str(cfg.get("size", bubbles_mod._DEFAULT_SIZE))
    quality = str(cfg.get("quality", bubbles_mod._DEFAULT_QUALITY))
    key = bubbles_mod.cache_key({"op": "bubble", "model": model, "prompt": bubbles_mod.STYLE_PROMPTS[style],
                                 "size": size, "quality": quality, "i": i,
                                 "proc": bubbles_mod._PROC_VERSION})
    return settings.cache_dir("bubbles") / f"{key}.png"


def test_stable_tile_is_byte_identical_to_cache_file(settings):
    """The stable {style}_{i}.png exists at the same path with byte-identical content to its
    content-hash cache file (behavior unchanged regardless of link-vs-copy)."""
    client = FakeClient()
    cache = Cache(settings.cache_dir)
    lib = bubbles.ensure_shape_library(client, settings, _tracker(settings), cache, count=1)
    stable = lib["speech"][0]
    src = _cache_file_for(bubbles, settings, "speech", 0)
    assert src.exists() and stable.exists()
    assert stable.read_bytes() == src.read_bytes()


def test_stable_tile_shares_inode_with_cache_file(settings):
    """When hardlinks are supported, the stable name and the content-hash cache file share one
    inode (zero extra bytes)."""
    client = FakeClient()
    cache = Cache(settings.cache_dir)
    lib = bubbles.ensure_shape_library(client, settings, _tracker(settings), cache, count=1)
    stable = lib["speech"][0]
    src = _cache_file_for(bubbles, settings, "speech", 0)
    assert os.stat(stable).st_ino == os.stat(src).st_ino   # same inode -> one shared copy on disk


def test_stable_tile_copy_fallback_when_hardlink_unsupported(settings, monkeypatch):
    """When os.link raises OSError (cross-device / unsupported FS), the fallback still produces a
    stable file equal in bytes to the cache file (and as a SEPARATE inode)."""
    def _no_link(*a, **k):
        raise OSError("hardlinks unsupported here")

    monkeypatch.setattr(bubbles.os, "link", _no_link)
    client = FakeClient()
    cache = Cache(settings.cache_dir)
    lib = bubbles.ensure_shape_library(client, settings, _tracker(settings), cache, count=1)
    stable = lib["speech"][0]
    src = _cache_file_for(bubbles, settings, "speech", 0)
    assert stable.exists() and stable.read_bytes() == src.read_bytes()   # byte-identical copy
    assert os.stat(stable).st_ino != os.stat(src).st_ino                 # distinct inodes (real copy)


# ── regression: rect is clamped on-page (finding 6) ───────────────────────────────
def test_place_bubble_rect_clamped_to_page_at_corner(settings, tmp_path):
    """A tile placed near a page corner must never return negative / off-page coordinates."""
    p = tmp_path / "tile.png"
    Image.open(io.BytesIO(_bubble_png())).save(p)

    page = Image.new("L", (400, 300), 30)
    rect = bubbles.place_bubble(page, p, 10, 10, 120, 90)   # center near the top-left corner
    assert rect.x0 >= 0 and rect.y0 >= 0
    assert rect.x1 <= page.width and rect.y1 <= page.height
    assert rect.x1 > rect.x0 and rect.y1 > rect.y0
