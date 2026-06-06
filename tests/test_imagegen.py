"""Regression tests for imagegen._decode validation (refusal / corrupt-cache guard)."""
import base64
from types import SimpleNamespace

import pytest

from ln2manga.cache import Cache
from ln2manga.cost import CostTracker
from ln2manga.imagegen import _decode, generate_image


def _ctx(settings, model="gpt-image-2"):
    settings.models["image"] = model
    return (CostTracker(10.0, settings.ledger_path, settings.prices_usd),
            Cache(settings.cache_dir))


def _resp(b64):
    return SimpleNamespace(data=[SimpleNamespace(b64_json=b64, url=None)])


def test_decode_returns_bytes_for_valid_png(png_bytes):
    resp = _resp(base64.b64encode(png_bytes).decode())
    assert _decode(resp) == png_bytes


def test_decode_rejects_none_b64():
    with pytest.raises(RuntimeError, match="no image data"):
        _decode(_resp(None))


def test_decode_rejects_empty_b64():
    # base64.b64decode("") returns b"" WITHOUT raising -> must be caught as a refusal.
    with pytest.raises(RuntimeError, match="no image data"):
        _decode(_resp(""))


def test_decode_rejects_empty_data_list():
    with pytest.raises(RuntimeError, match="no image data"):
        _decode(SimpleNamespace(data=[]))


def test_decode_rejects_missing_data_attr():
    with pytest.raises(RuntimeError, match="no image data"):
        _decode(SimpleNamespace())


def test_decode_rejects_none_data_element():
    # A null payload entry must report the same refusal, not a raw AttributeError.
    with pytest.raises(RuntimeError, match="no image data"):
        _decode(SimpleNamespace(data=[None]))


def test_decode_rejects_data_element_missing_b64_attr():
    with pytest.raises(RuntimeError, match="no image data"):
        _decode(SimpleNamespace(data=[SimpleNamespace()]))


def test_decode_rejects_valid_base64_non_image():
    # Valid base64 that is not an image must not reach the content cache.
    garbage = base64.b64encode(b"not a png, just bytes").decode()
    with pytest.raises(RuntimeError, match="undecodable image"):
        _decode(_resp(garbage))


def test_corrupt_response_never_poisons_cache(settings):
    """A refusal/garbage response must raise instead of writing a corrupt cache file."""
    class BadImages:
        def __init__(self):
            self.calls = []

        def generate(self, **kw):
            self.calls.append(kw)
            return _resp("")  # empty -> refusal

    class BadClient:
        def __init__(self):
            self.images = BadImages()

    tracker, cache = _ctx(settings)
    client = BadClient()
    with pytest.raises(RuntimeError, match="no image data"):
        generate_image(client, settings, tracker, cache,
                       stage="sheets", prompt="p", quality="high")
    # Nothing was written to the content cache, so a retry would re-call the API.
    stage_dir = settings.cache_dir("sheets")
    assert not (stage_dir.exists() and any(stage_dir.rglob("*.png")))
