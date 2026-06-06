import pytest

from ln2manga.cache import Cache, cache_key


def test_cache_key_deterministic_and_sensitive():
    a = cache_key({"model": "x", "prompt": "hi"})
    b = cache_key({"prompt": "hi", "model": "x"})    # order-independent
    c = cache_key({"model": "x", "prompt": "bye"})
    assert a == b
    assert a != c


def test_cache_key_includes_ref_bytes():
    base = cache_key({"m": 1})
    with_ref = cache_key({"m": 1}, ref_bytes=[b"abc"])
    other_ref = cache_key({"m": 1}, ref_bytes=[b"xyz"])
    assert base != with_ref != other_ref
    assert with_ref != base


def test_cache_roundtrip(settings):
    cache = Cache(settings.cache_dir)
    assert cache.get("panels", "k") is None
    cache.put("panels", "k", b"\x89PNG-data", meta={"q": "low"})
    assert cache.get("panels", "k") == b"\x89PNG-data"


def test_cache_put_writes_sidecar_meta(settings):
    cache = Cache(settings.cache_dir)
    p = cache.put("panels", "k", b"\x89PNG-data", meta={"q": "low"})
    sidecar = p.with_suffix(p.suffix + ".json")
    assert sidecar.exists()
    import json
    assert json.loads(sidecar.read_text(encoding="utf-8")) == {"q": "low"}


def test_cache_put_leaves_no_tmp_files(settings):
    # Atomic write uses a temp file + os.replace; on success no .tmp must remain
    # so a torn write can never be served as a permanent silent cache hit.
    cache = Cache(settings.cache_dir)
    p = cache.put("panels", "k", b"\x89PNG-data", meta={"q": "low"})
    leftover = list(p.parent.glob("*.tmp")) + list(p.parent.glob(".*.tmp"))
    assert leftover == []


def test_cache_put_is_atomic_replace_not_truncate(monkeypatch, settings):
    # If the payload write fails after a prior good value exists, the old value must
    # survive intact (atomic replace) instead of being truncated to a partial hit.
    cache = Cache(settings.cache_dir)
    cache.put("panels", "k", b"\x89PNG-GOOD-DATA")
    assert cache.get("panels", "k") == b"\x89PNG-GOOD-DATA"

    import ln2manga.cache as cache_mod
    orig_replace = cache_mod.os.replace

    def boom(src, dst):
        raise OSError("simulated crash mid-write")

    monkeypatch.setattr(cache_mod.os, "replace", boom)
    with pytest.raises(OSError):
        cache.put("panels", "k", b"PARTIAL")
    monkeypatch.setattr(cache_mod.os, "replace", orig_replace)

    # Old value intact; no torn bytes served; no temp turds left behind.
    assert cache.get("panels", "k") == b"\x89PNG-GOOD-DATA"
    p = cache.path("panels", "k")
    leftover = list(p.parent.glob("*.tmp")) + list(p.parent.glob(".*.tmp"))
    assert leftover == []
