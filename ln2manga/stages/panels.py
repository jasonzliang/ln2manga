"""Stage 5 — panels: generate each panel's art, anchored on character reference sheets.

Every panel with known characters goes through images.edit(image=[sheets...]) so the cast
stays consistent. An explicit "preserve identity / change only the pose" instruction plus a
"no text" instruction is appended. Panels with no roster character fall back to generate.

SCENE-BACKGROUND CONSISTENCY (2-pass): panels are grouped into scenes — maximal runs of
consecutive panels sharing the same `setting` (a "same…" setting continues the previous
scene). The first panel of each scene is the BACKGROUND ANCHOR; PASS 1 generates anchors as
usual (character sheets only) and PASS 2 generates the rest with the anchor image added as an
extra reference (placed first) so the background/location stays consistent within the scene.
"""
from __future__ import annotations

import io
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw

from ..bible import composite_sheets
from ..cache import Cache
from ..config import Settings
from ..cost import BudgetExceeded, CostTracker
from ..imagegen import MAX_REFS, edit_image, generate_image

PRESERVE = (
    " Keep each character's face, hairstyle, hair length, eye design, body proportions, "
    "costume and line style EXACTLY as in the reference sheet(s); do not redesign them. "
    "Change only the pose, expression and scene."
)

BACKGROUND = (
    " Keep the SAME background, location, environment and time of day as the first reference "
    "image (the establishing shot of this scene); change only the characters' poses, "
    "expressions and framing."
)

SOFTEN = (" The depiction is tasteful, wholesome, non-suggestive, and fully clothed.")

# The "Action: …" clause is the usual moderation trigger. Strip it from "Action:" up to the next
# labeled segment or trailing guard, or the end of the prompt. DOTALL + the lazy `.*?` tolerates
# internal commas/periods in the action text without swallowing later segments. The stop-lookahead
# must include the trailing-suffix leaders ("Leave clear empty space"/GLOBAL_STYLE's "Black-and-white"
# /the "NO letters" guard) so a *trailing* Action: clause (emitted when setting=='', emotion=='neutral'
# and angle=='eye-level' — no Setting:/Mood:/Camera follows) stops before the style/preserve guards
# instead of running to end-of-string and stripping GLOBAL_STYLE + PRESERVE too.
_ACTION_RE = re.compile(
    r"\s*Action:\s.*?(?=\s(?:Setting:|Mood:|Camera|Leave clear empty space|Black-and-white|NO letters)|\s*$)",
    re.IGNORECASE | re.DOTALL,
)


@lru_cache(maxsize=64)
def _read_anchor_bytes(anchor_path: str) -> bytes:
    """Read a scene-anchor PNG once and reuse across every dependent panel of that scene.

    In PASS 2 each dependent panel anchors on the SAME scene-anchor file; reading it fresh per
    panel re-does identical disk I/O. The anchor is a stable content-cache file written in PASS 1
    (never mutated afterwards), so memoizing on its path is sound and the bytes are byte-identical
    to a fresh read -> the panel cache key is unchanged."""
    return Path(anchor_path).read_bytes()


def _is_safety_reject(e: Exception) -> bool:
    """True ONLY for a clear OpenAI safety/moderation 400 (e.g. BadRequestError "rejected by the
    safety system"). Conservative by design: generic errors (timeouts, 500s, rate limits, plain
    RuntimeErrors) return False so they still placeholder as before."""
    name = type(e).__name__.lower()
    if "badrequest" not in name:                # 400-class only; not timeouts/5xx/rate-limits
        return False
    msg = str(e).lower()
    return any(s in msg for s in (
        "safety system", "safety_system", "content policy", "moderation",
        "rejected by the safety",
    ))


def _sanitize_prompt(prompt: str) -> str:
    """Soften a safety-rejected panel prompt: drop the 'Action:' clause (the usual trigger) and
    append an explicit tasteful/wholesome guard. Everything else (characters, setting, mood,
    camera, style guards, PRESERVE) is preserved."""
    stripped = _ACTION_RE.sub("", prompt, count=1)
    return stripped + SOFTEN


def _placeholder(settings: Settings, label: str) -> bytes:
    w, h = (int(x) for x in settings.image["size"].split("x"))
    img = Image.new("L", (w, h), 240)
    d = ImageDraw.Draw(img)
    d.rectangle([4, 4, w - 5, h - 5], outline=120, width=3)
    d.text((20, 20), label, fill=90)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _norm_setting(s: str) -> str:
    """Lowercase, strip, collapse internal whitespace — for scene-grouping comparison."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _load_settings_map(settings: Settings, chapter_number: int) -> dict[int, str]:
    """Best-effort map of panel_number -> setting from the parsed panels artifact."""
    path = settings.artifacts_dir / f"chapter-{chapter_number}.panels.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[int, str] = {}
    for d in data:
        try:
            out[int(d["panel_number"])] = str(d.get("setting", ""))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _scene_of(prompts, settings_map: dict[int, str], chapter_number: int) -> dict[int, int]:
    """Group panels into scenes — maximal runs of CONSECUTIVE panels (by panel_number) that
    share the same normalized `setting`. A setting whose first word is "same" (e.g.
    "Same carriage interior.") continues the PREVIOUS scene — but ONLY when there is a real
    established predecessor setting; a "same…" with no prior explicit setting (e.g. it follows
    an empty/unknown panel, or it leads the chapter) starts its own scene rather than anchoring
    on an un-established background. A substring like "sesame"/"samey" never continues a scene.
    An empty/unknown setting always starts its own scene (no grouping).
    Returns panel_number -> scene_id (scene_id = anchor's panel_number).
    """
    ordered = sorted((pp.panel_number for pp in prompts))
    scene: dict[int, int] = {}
    prev_norm: str | None = None
    cur_scene: int | None = None
    for pn in ordered:
        norm = _norm_setting(settings_map.get(pn, ""))
        is_same = bool(norm) and re.match(r"same\b", norm) is not None
        if cur_scene is None:
            cur_scene = pn                                  # first panel starts the first scene
        elif is_same and prev_norm:
            pass                                            # "same…" -> continue established scene
        elif (not norm) or norm != prev_norm:
            cur_scene = pn                                  # new/unknown setting -> its own scene
        scene[pn] = cur_scene
        if not is_same:                                     # remember the last *explicit* setting
            prev_norm = norm
    return scene


def _gen_panel(client, settings: Settings, tracker: CostTracker, cache: Cache,
               pp, sheets: dict[str, str], quality: str, anchor_path: str | None = None,
               state: dict | None = None) -> dict:
    ref_paths = [sheets[n] for n in pp.ref_sheets if n in sheets]
    # Identity-anchoring loss is silent otherwise: a requested sheet missing from `sheets` (it
    # failed/was budget-skipped in the charsheet stage) drops out above, so warn per panel and
    # tally it for the run summary (#3).
    missing = [n for n in (pp.ref_sheets or []) if n not in sheets]
    if missing:
        print(f"[warn] panel {pp.panel_number}: no sheet for {missing} -> identity not anchored",
              file=sys.stderr)
        if state is not None:
            state.setdefault("missing_sheet_panels", set()).add(pp.panel_number)
    prompt = pp.prompt + (PRESERVE if ref_paths else "")

    # Build the local reference bytes BEFORE any API call so a missing/corrupt sheet on disk is
    # reported as a file error (#bug-low) rather than mislabeled "(failed)" as an image-model
    # refusal. composite_sheets()/read_bytes() touch only local pipeline-written files.
    use_anchor = bool(anchor_path)
    try:
        if use_anchor:
            char_paths = ref_paths
            if len(char_paths) > MAX_REFS - 1:
                char_refs = [composite_sheets(char_paths)]  # tile extras into one ref
            else:
                char_refs = [Path(p).read_bytes() for p in char_paths]
            ref_bytes = [_read_anchor_bytes(anchor_path)] + char_refs  # anchor placed FIRST (cached read)
        elif ref_paths:
            if len(ref_paths) > MAX_REFS:
                ref_bytes = [composite_sheets(ref_paths)]       # tile extras into one ref
            else:
                ref_bytes = [Path(p).read_bytes() for p in ref_paths]
        else:
            ref_bytes = None
    except Exception as e:
        print(f"[warn] panel {pp.panel_number}: reference sheet unreadable ({e}) -> "
              f"generating without refs", file=sys.stderr)
        use_anchor = False
        ref_bytes = None                                         # fall back to no-ref generate
        prompt = pp.prompt                                       # drop PRESERVE: no refs to preserve

    def _attempt(prompt_text: str):
        """Run the 3-way generation call for `prompt_text` and return the cache Path. Shared by
        the first attempt and the softened-prompt retry so both take an identical code path
        (anchor appends BACKGROUND; ref/no-ref use prompt_text as-is)."""
        if use_anchor:
            # Scene-background anchor: the anchor image is an extra ref placed FIRST so gpt-image
            # preserves it (anchor + up to MAX_REFS-1 character sheets; overflow already tiled).
            return edit_image(client, settings, tracker, cache, stage="panels",
                              prompt=prompt_text + BACKGROUND, ref_bytes=ref_bytes, quality=quality)
        if ref_bytes is None:
            return generate_image(client, settings, tracker, cache,
                                  stage="panels", prompt=prompt_text, quality=quality)
        return edit_image(client, settings, tracker, cache,
                          stage="panels", prompt=prompt_text, ref_bytes=ref_bytes, quality=quality)

    # Never short-circuit before generate_image/edit_image: they check the content cache BEFORE
    # the budget guard, so a previously-generated panel costs $0 and is reused even after the
    # budget tripped (#12). A genuinely-uncached panel re-raises BudgetExceeded -> "(budget)" (#18).
    reason = None
    try:
        path = _attempt(prompt)
    except BudgetExceeded as e:
        # One actionable, de-duplicated line per run instead of a per-panel stderr storm (#1):
        # the first uncached panel that trips the cap explains the cause and the remedy; later
        # ones are tallied silently and summarized at the end of run(). The claim-the-warning
        # check-and-set runs under state["_warn_lock"] so it is atomic across the image workers —
        # otherwise several workers tripping the cap near-simultaneously each read budget_warned
        # as falsey before any sets it and all print the line.
        lock = state.get("_warn_lock") if state is not None else None
        if lock is not None:
            with lock:
                first = not state.get("budget_warned")
                if first:
                    state["budget_warned"] = True
        else:
            first = True                                         # no state -> always print
        if first:
            print(f"[warn] budget cap ${tracker.max_usd:.2f} reached; remaining uncached panels "
                  f"will be grey placeholders. Raise budget.max_usd or --budget; already-cached "
                  f"panels stay free. ({e})", file=sys.stderr)
        path = settings.cache_dir("panels") / f"panel_{pp.panel_number:04d}_placeholder.png"
        path.write_bytes(_placeholder(settings, f"panel {pp.panel_number} (budget)"))
        reason = "budget"
    except Exception as e:                                       # moderation refusal, etc.
        # A clear safety/moderation 400 is often the *Action:* clause, not the whole panel — retry
        # ONCE with that clause stripped and a tasteful guard appended before giving up. A recovered
        # retry is a real image (reason stays None); anything else still placeholders "(failed)".
        if _is_safety_reject(e):
            print(f"[warn] panel {pp.panel_number}: safety-rejected; retrying with softened prompt",
                  file=sys.stderr)
            try:
                path = _attempt(_sanitize_prompt(prompt))
            except Exception as e2:
                print(f"[warn] panel {pp.panel_number} failed even after softening "
                      f"({type(e2).__name__}): {str(e2)[:120]} -> placeholder", file=sys.stderr)
                path = settings.cache_dir("panels") / f"panel_{pp.panel_number:04d}_placeholder.png"
                path.write_bytes(_placeholder(settings, f"panel {pp.panel_number} (failed)"))
                reason = "failed"
        else:
            # Distinct cause -> keep the per-panel warning (no suppression).
            print(f"[warn] panel {pp.panel_number} failed ({type(e).__name__}): "
                  f"{str(e)[:120]} -> placeholder", file=sys.stderr)
            path = settings.cache_dir("panels") / f"panel_{pp.panel_number:04d}_placeholder.png"
            path.write_bytes(_placeholder(settings, f"panel {pp.panel_number} (failed)"))
            reason = "failed"

    # path is the single content-cache file (no duplicate stable copy) — dedup (#6).
    # "ok"/"reason" are internal: ok=False keeps PASS 2 from anchoring dependents on a placeholder
    # bg (#4); reason lets run() tally placeholders by cause. Both are popped before persisting.
    return {"panel_number": pp.panel_number, "path": str(path), "ok": reason is None,
            "reason": reason}


def run(client, settings: Settings, tracker: CostTracker, cache: Cache,
        prompts, sheets: dict[str, str], chapter_number: int) -> list[dict]:
    quality = settings.image["panel_quality"]
    workers = max(1, int(settings.concurrency.get("image", 3)))

    consistency = (getattr(settings, "scene", {}) or {}).get("background_consistency", True)
    # scene_id = anchor panel_number; a panel is an anchor iff scene[pn] == pn.
    if consistency:
        settings_map = _load_settings_map(settings, chapter_number)
        scene = _scene_of(prompts, settings_map, chapter_number)
    else:
        scene = {pp.panel_number: pp.panel_number for pp in prompts}  # every panel is its own anchor

    anchors = [pp for pp in prompts if scene.get(pp.panel_number) == pp.panel_number]
    others = [pp for pp in prompts if scene.get(pp.panel_number) != pp.panel_number]

    total = len(anchors) + len(others)
    # cross-panel run flags (#1) / tallies (#3); _warn_lock makes the budget-warning claim atomic
    # across the image workers so the cap message is printed exactly once.
    state: dict = {"_warn_lock": threading.Lock()}
    print(f"[panels] {total} panels, {workers} workers", file=sys.stderr)

    def _gen(pp, anchor_path=None):
        return _gen_panel(client, settings, tracker, cache, pp, sheets, quality,
                          anchor_path=anchor_path, state=state)

    def _run_pass(items, label, *, anchor_for=None):
        """Submit `items` and print a per-panel progress counter to stderr as each future
        returns, so a slow PASS isn't mistaken for a hung one (#5). Order no longer matters —
        the final manifest.sort restores panel order."""
        results = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_gen, pp, anchor_for(pp) if anchor_for else None) for pp in items]
            for k, fut in enumerate(as_completed(futures), start=1):
                results.append(fut.result())
                print(f"[panels] {k}/{len(items)} ({label})", file=sys.stderr)
        return results

    anchor_results = _run_pass(anchors, "anchor")               # PASS 1: anchors (char sheets only)

    # scene_id -> anchor image path (used as the extra reference for the rest of the scene).
    # Only REAL anchors qualify: a placeholdered/failed anchor (ok=False) is excluded so its
    # dependents generate WITHOUT a background ref instead of being told to match a grey box (#4).
    anchor_path_by_scene = {r["panel_number"]: r["path"] for r in anchor_results if r["ok"]}
    manifest = list(anchor_results)

    if others:                                                  # PASS 2: rest, anchored on scene img
        # Pre-warm the anchor-byte read cache ONCE, single-threaded, for every distinct anchor used
        # in PASS 2. Each dependent panel then reuses the cached bytes (a pure cache hit) instead of
        # re-reading the SAME anchor PNG per panel. Done here (not lazily in the workers) so the read
        # happens exactly once per anchor even though the workers run concurrently — the in-worker
        # lru_cache alone could race two threads into duplicate reads. Same file bytes -> the panel
        # content-cache key is unchanged.
        for ap in {anchor_path_by_scene.get(scene[pp.panel_number]) for pp in others}:
            if ap:
                _read_anchor_bytes(ap)
        manifest += _run_pass(
            others, "scene",
            anchor_for=lambda pp: anchor_path_by_scene.get(scene[pp.panel_number]))

    manifest.sort(key=lambda d: d["panel_number"])

    # Summarize placeholders by cause so a budget-stopped run isn't a green banner over grey pages
    # (#1/#2). Surfaced on stderr here (in-stage); the CLI summary line is a separate cross-file
    # change tracked in the deferred notes.
    budget = sorted(d["panel_number"] for d in manifest if d.get("reason") == "budget")
    failed = sorted(d["panel_number"] for d in manifest if d.get("reason") == "failed")
    bad = sorted(budget + failed)
    if bad:
        print(f"[warn] {len(bad)}/{total} panels are PLACEHOLDERS "
              f"(budget: {budget}, failed: {failed}) -> raise budget.max_usd or retry",
              file=sys.stderr)
    missing_panels = sorted(state.get("missing_sheet_panels", set()))
    if missing_panels:
        print(f"[warn] {len(missing_panels)}/{total} panels missing a character sheet "
              f"(identity not anchored): {missing_panels}", file=sys.stderr)

    for d in manifest:                                          # drop internal flags before persisting
        d.pop("ok", None)
        d.pop("reason", None)
    mpath = settings.artifacts_dir / f"chapter-{chapter_number}.panelimgs.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
