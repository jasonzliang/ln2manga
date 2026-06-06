"""Shared OpenAI image helpers: caching, budget guard, retry, model-aware params.

Verified facts baked in:
  - panels MUST use images.edit (only it accepts `image=` references + `input_fidelity`).
  - gpt-image-2* rejects `input_fidelity` (it auto-applies high fidelity) -> gate it off.
  - images.edit has NO `moderation` param (images.generate does) -> only send it to generate.
  - gpt-image responses return base64 (`b64_json`), never a URL.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

from PIL import Image

from .cache import Cache, cache_key
from .config import Settings
from .cost import CostTracker
from .retry import with_retry

MAX_REFS = 5  # gpt-image preserves ~first 5 input images at high fidelity


def _supports_input_fidelity(model: str) -> bool:
    return not model.startswith("gpt-image-2")


def _decode(resp: Any) -> bytes:
    data = getattr(resp, "data", None)
    first = data[0] if data else None
    b64 = getattr(first, "b64_json", None)
    if not b64:
        raise RuntimeError("image API returned no image data (content-filter refusal?)")
    png = base64.b64decode(b64)
    try:
        Image.open(io.BytesIO(png)).verify()  # reject empty/garbage before it is cached
    except Exception as e:
        raise RuntimeError(f"image API returned an undecodable image ({len(png)} bytes)") from e
    return png


def _usage_dict(resp: Any) -> dict:
    u = getattr(resp, "usage", None)
    if u is None:
        return {}
    return {k: getattr(u, k, None)
            for k in ("input_tokens", "output_tokens", "total_tokens")}


def generate_image(client, settings: Settings, tracker: CostTracker, cache: Cache, *,
                   stage: str, prompt: str, quality: str) -> Path:
    """Generate an image and return the PATH of the single content-addressed cache file (callers
    reference this path directly — no duplicate stable copies)."""
    model = settings.models["image"]
    size = settings.image["size"]
    fmt = settings.image["output_format"]
    # Resolve moderation once so the cache key and the API call cannot drift (#14):
    # moderation affects the produced image, so it is part of the cache identity.
    moderation = settings.image.get("moderation", "auto")
    key = cache_key({"op": "gen", "model": model, "prompt": prompt,
                     "size": size, "quality": quality, "fmt": fmt,
                     "moderation": moderation})
    path = cache.path(stage, key)
    if path.exists():
        return path

    est = tracker.estimate_image(model, quality)
    tracker.check(est, is_image=True)
    resp = with_retry(client.images.generate)(
        model=model, prompt=prompt, size=size, quality=quality,
        output_format=fmt, moderation=moderation, n=1,
        timeout=float(settings.image.get("timeout_s", 240)),
    )
    png = _decode(resp)
    actual = tracker.estimate_image_from_usage(model, _usage_dict(resp))
    tracker.record("image", model, max(est, actual), {"op": "gen", "stage": stage,
                                                      "usage": _usage_dict(resp)})
    return cache.put(stage, key, png, meta={"prompt": prompt[:600], "quality": quality})


def edit_image(client, settings: Settings, tracker: CostTracker, cache: Cache, *,
               stage: str, prompt: str, ref_bytes: list[bytes], quality: str) -> Path:
    """Edit (reference-anchored generate) and return the PATH of the content-cache file."""
    model = settings.models["image"]
    size = settings.image["size"]
    fmt = settings.image["output_format"]
    refs = ref_bytes[:MAX_REFS]
    key = cache_key({"op": "edit", "model": model, "prompt": prompt,
                     "size": size, "quality": quality, "fmt": fmt}, ref_bytes=refs)
    path = cache.path(stage, key)
    if path.exists():
        return path

    est = tracker.estimate_image(model, quality)
    tracker.check(est, is_image=True)
    images = [(f"ref{i}.png", b, "image/png") for i, b in enumerate(refs)]
    kwargs: dict[str, Any] = dict(
        model=model, image=(images if len(images) > 1 else images[0]),
        prompt=prompt, size=size, quality=quality, output_format=fmt, n=1,
        timeout=float(settings.image.get("timeout_s", 240)),
    )
    if _supports_input_fidelity(model):
        kwargs["input_fidelity"] = "high"
    resp = with_retry(client.images.edit)(**kwargs)
    png = _decode(resp)
    actual = tracker.estimate_image_from_usage(model, _usage_dict(resp))
    tracker.record("image", model, max(est, actual),
                   {"op": "edit", "stage": stage, "refs": len(refs),
                    "usage": _usage_dict(resp)})
    return cache.put(stage, key, png, meta={"prompt": prompt[:600], "quality": quality,
                                            "refs": len(refs)})
