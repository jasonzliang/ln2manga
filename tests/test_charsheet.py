"""Tests for the charsheet stage's named-sheet materialization (ln2manga.stages.charsheet).

Focus: the browsable data/out/reference-sheets/<slug>.png view is a HARDLINK to the sheet's
content-hash cache file (zero extra bytes), with a real-copy fallback where hardlinks are
unsupported. Behavior (the named file exists at the same path with byte-identical content) is
unchanged.
"""
from __future__ import annotations

import base64
import io
import os
import re
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from ln2manga.cache import Cache
from ln2manga.cost import CostTracker
from ln2manga.stages import charsheet


def _png_bytes(size=(40, 60), color=128) -> bytes:
    buf = io.BytesIO()
    Image.new("L", size, color).save(buf, "PNG")
    return buf.getvalue()


def _img_response(png: bytes):
    return SimpleNamespace(
        data=[SimpleNamespace(b64_json=base64.b64encode(png).decode(), url=None)],
        usage=SimpleNamespace(input_tokens=0, output_tokens=0, total_tokens=0))


class _GenClient:
    """AI-path client whose images.generate returns a fixed PNG (refs disabled)."""
    def __init__(self, png: bytes):
        self._png = png

        class images:
            @staticmethod
            def generate(**kw):
                return _img_response(png)

            @staticmethod
            def edit(**kw):
                raise AssertionError("edit not expected")

        self.images = images


def _tracker(settings):
    return CostTracker(settings.budget["max_usd"], settings.ledger_path,
                       settings.prices_usd, 10_000, dry_run=False)


def _disable_refs(settings):
    object.__setattr__(settings, "references", {"enabled": False})


def _named_path(settings, name):
    slug = re.sub(r"[^\w-]+", "_", name).strip("_").lower() or "ref"
    return settings.out_dir / "reference-sheets" / f"{slug}.png"


def _run_one(settings, png=None):
    """Run charsheet for a single character via the AI path; return (sheet cache path, named path)."""
    _disable_refs(settings)
    png = png or _png_bytes()
    specs = [SimpleNamespace(characters_present=["Subaru"])]
    cache = Cache(settings.cache_dir)
    sheets = charsheet.run(_GenClient(png), settings, _tracker(settings), cache, specs, 99)
    assert "Subaru" in sheets
    return Path(sheets["Subaru"]), _named_path(settings, "Subaru")


# ── _link_or_copy helper ────────────────────────────────────────────────────────
def test_link_or_copy_hardlinks_same_inode(tmp_path):
    src = tmp_path / "src.png"
    src.write_bytes(_png_bytes())
    dest = tmp_path / "dest.png"
    charsheet._link_or_copy(src, dest)
    assert dest.exists() and dest.read_bytes() == src.read_bytes()
    assert os.stat(dest).st_ino == os.stat(src).st_ino


def test_link_or_copy_replaces_existing_dest(tmp_path):
    """An existing dest (possibly pointing at an old hash) is unlinked then re-linked to src."""
    src = tmp_path / "src.png"
    src.write_bytes(_png_bytes(color=10))
    dest = tmp_path / "dest.png"
    dest.write_bytes(_png_bytes(color=250))   # stale prior content
    charsheet._link_or_copy(src, dest)
    assert dest.read_bytes() == src.read_bytes()
    assert os.stat(dest).st_ino == os.stat(src).st_ino


def test_link_or_copy_falls_back_to_real_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(charsheet.os, "link",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no hardlinks")))
    src = tmp_path / "src.png"
    src.write_bytes(_png_bytes())
    dest = tmp_path / "dest.png"
    charsheet._link_or_copy(src, dest)
    assert dest.exists() and dest.read_bytes() == src.read_bytes()
    assert os.stat(dest).st_ino != os.stat(src).st_ino   # distinct inode -> real copy


# ── named-sheet view is a hardlink to the content-hash cache file ─────────────────
def test_named_sheet_is_byte_identical_to_cache_file(settings):
    src, named = _run_one(settings)
    assert src.exists() and named.exists()
    assert named.read_bytes() == src.read_bytes()


def test_named_sheet_shares_inode_with_cache_file(settings):
    src, named = _run_one(settings)
    assert os.stat(named).st_ino == os.stat(src).st_ino   # zero-extra-bytes shared inode


def test_named_sheet_copy_fallback_when_hardlink_unsupported(settings, monkeypatch):
    monkeypatch.setattr(charsheet.os, "link",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no hardlinks")))
    src, named = _run_one(settings)
    assert named.exists() and named.read_bytes() == src.read_bytes()
    assert os.stat(named).st_ino != os.stat(src).st_ino   # real copy, distinct inode
