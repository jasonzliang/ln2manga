"""Stage 4 — charsheet: generate ONE B&W reference sheet for EVERY character that appears.

Coverage: every distinct character in any panel's `characters_present` gets a sheet — roster
characters use their frozen bible descriptor; other discovered characters (e.g. Petra, a dragon)
use the descriptor the parser recorded in the chapter `cast`. Each sheet is either AI-generated,
or (when references.enabled) built from real official images (explicit URLs or online search).

Performance: sheets are generated CONCURRENTLY (concurrency.image workers).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import references
from ..bible import active_style, descriptor_for, ref_sheet_prompt_for
from ..cache import Cache
from ..config import Settings
from ..cost import BudgetExceeded, CostTracker
from ..imagegen import MAX_REFS, edit_image, generate_image


def _link_or_copy(src, dest) -> None:
    """Materialize `dest` as a HARDLINK to `src` (same inode, zero extra bytes), falling back to a
    real byte copy on a filesystem that can't hardlink (cross-device / unsupported / Windows).

    `src` is a WRITE-ONCE content-addressed cache file (its path is the hash of its content), so a
    hardlink never goes stale; the unlink-then-link re-points `dest` at the current hash file if the
    target changed (e.g. after re-anchoring a character produced a new hash).
    """
    dest = Path(dest)
    try:
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        os.link(src, dest)
    except OSError:
        shutil.copyfile(src, dest)


def _synth_prompt(name: str, descriptor: str, n_refs: int, settings: Settings | None = None) -> str:
    who = descriptor or name
    src = ("the provided official reference image" if n_refs <= 1
           else f"the {n_refs} provided official reference images (all the SAME character)")
    return (
        f"Using {src} of {who}, synthesize ONE clean black-and-white manga character reference "
        "sheet: front view + side view + head close-up, plain white background. Keep the face, "
        "hairstyle, outfit and proportions IDENTICAL to the references (render colors as manga "
        "tones); reconcile any differences between the references into one consistent design. "
        + active_style(settings)
    )


def characters_in(specs) -> list[str]:
    """Every distinct character visible across the chapter (roster AND discovered), in first-seen
    order."""
    names: list[str] = []
    for s in specs:
        for n in s.characters_present:
            if n and n not in names:
                names.append(n)
    return names


def _load_cast(settings: Settings, chapter_number: int) -> dict[str, str]:
    p = settings.artifacts_dir / f"chapter-{chapter_number}.cast.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        return {}


def _provided_sheet(client, settings: Settings, tracker: CostTracker, cache: Cache,
                    name: str, descriptor: str):
    """Build a sheet from real reference images if any resolve (explicit sources OR online search
    that verifies the official design). Returns the sheet PATH (no duplicate copy), or None to
    fall back to AI."""
    ref_paths = references.resolve_reference_set(name, settings, client, tracker)
    if not ref_paths:
        return None
    ref_bytes = [p.read_bytes() for p in ref_paths][:MAX_REFS]
    mode = (references.references_config(settings).get("mode") or "stylize").lower()
    if mode == "raw":
        print(f"[refs] {name}: using provided reference image as-is (raw, $0)", file=sys.stderr)
        return ref_paths[0]            # the resolved ref IS the sheet (no extra copy)
    # stylize: synthesize ONE B&W manga sheet from ALL verified official references
    path = edit_image(client, settings, tracker, cache, stage="sheets",
                      prompt=_synth_prompt(name, descriptor, len(ref_bytes), settings),
                      ref_bytes=ref_bytes, quality=settings.image["sheet_quality"])
    print(f"[refs] {name}: synthesized B&W sheet from {len(ref_bytes)} official reference(s)",
          file=sys.stderr)
    return path


def run(client, settings: Settings, tracker: CostTracker, cache: Cache,
        specs, chapter_number: int) -> dict[str, str]:
    cast = _load_cast(settings, chapter_number)
    use_refs = bool(references.references_config(settings).get("enabled"))
    names = characters_in(specs)
    workers = max(1, int(settings.concurrency.get("image", 3)))

    def _make_sheet(name: str) -> tuple[str, str | None]:
        descriptor = descriptor_for(name, cast, settings)
        try:
            if use_refs:
                p = _provided_sheet(client, settings, tracker, cache, name, descriptor)
                if p is not None:
                    return (name, str(p))
            # default / fallback: AI-generated sheet (returns its content-cache path)
            path = generate_image(client, settings, tracker, cache, stage="sheets",
                                  prompt=ref_sheet_prompt_for(name, descriptor, settings),
                                  quality=settings.image["sheet_quality"])
            return (name, str(path))
        except BudgetExceeded as e:
            print(f"[warn] charsheet budget reached at {name}: {e}", file=sys.stderr)
            return (name, None)
        except Exception as e:
            print(f"[warn] sheet for {name} failed ({type(e).__name__}): {str(e)[:120]} "
                  f"-> character will use text-only prompt", file=sys.stderr)
            return (name, None)

    sheets: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for name, path in ex.map(_make_sheet, names):
            if path:
                sheets[name] = path

    out = settings.artifacts_dir / f"chapter-{chapter_number}.sheets.json"
    out.write_text(json.dumps(sheets, indent=2, ensure_ascii=False), encoding="utf-8")

    # Human-readable copies: the cache files are content-addressed hashes (shared across chapters
    # for cache reuse), so the filenames aren't names. Mirror each sheet to a NAMED file under
    # data/out/reference-sheets/<name>.png so the sheets are browsable by character. The hash file
    # stays the cache key; this is just a friendly view (the .sheets.json manifest is the mapping).
    # HARDLINK the named file to the hash file (same inode, zero extra bytes) — the hash file is a
    # write-once content-addressed cache entry, so the link never goes stale.
    named_dir = settings.out_dir / "reference-sheets"
    named_dir.mkdir(parents=True, exist_ok=True)
    for name, path in sheets.items():
        slug = re.sub(r"[^\w-]+", "_", name).strip("_").lower() or "ref"
        try:
            _link_or_copy(path, named_dir / f"{slug}.png")
        except OSError:
            pass
    return sheets
