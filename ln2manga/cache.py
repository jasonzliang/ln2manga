"""Content-addressed disk cache.

Image generation is non-deterministic and has no seed, so we cache on the *inputs*
(model + prompt + params + reference-image bytes). Identical inputs => skip the paid call.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def _atomic_write_bytes(p: Path, data: bytes) -> None:
    # Write atomically (temp file + os.replace in the same dir) so an interrupted write
    # (Ctrl-C / OOM / disk-full) can never leave a truncated payload that p.exists() then
    # serves as a permanent, silent cache hit (mirrors cost.py:_save's #11 hardening).
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix="." + p.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def cache_key(payload: dict[str, Any], ref_bytes: Iterable[bytes] = ()) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    for b in ref_bytes:
        h.update(b":")
        h.update(hashlib.sha256(b).digest())
    return h.hexdigest()[:32]


class Cache:
    def __init__(self, root_for_stage):
        # root_for_stage: callable(stage)->Path  (Settings.cache_dir)
        self._root = root_for_stage

    def path(self, stage: str, key: str, ext: str = "png") -> Path:
        return self._root(stage) / f"{key}.{ext}"

    def get(self, stage: str, key: str, ext: str = "png") -> bytes | None:
        p = self.path(stage, key, ext)
        return p.read_bytes() if p.exists() else None

    def put(self, stage: str, key: str, data: bytes, ext: str = "png",
            meta: dict | None = None) -> Path:
        p = self.path(stage, key, ext)
        p.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(p, data)
        if meta is not None:
            _atomic_write_bytes(
                p.with_suffix(p.suffix + ".json"),
                json.dumps(meta, indent=2, ensure_ascii=False).encode("utf-8"))
        return p
