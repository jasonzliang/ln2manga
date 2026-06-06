"""Verify the consistency-critical image-call construction (model-gated input_fidelity, multi-ref)."""
import base64
import json
from types import SimpleNamespace

from ln2manga.artifacts import PanelPrompt
from ln2manga.cache import Cache
from ln2manga.cost import CostTracker
from ln2manga.imagegen import edit_image, generate_image
from ln2manga.stages import panels


def _tiny_png() -> bytes:
    """A small valid PNG distinct from the grey 'L' placeholder size (mirrors conftest.tiny_png)."""
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.new("L", (64, 96), 200).save(buf, "PNG")
    return buf.getvalue()


def _image_response(png: bytes):
    return SimpleNamespace(
        data=[SimpleNamespace(b64_json=base64.b64encode(png).decode(), url=None)],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
    )


class _SafetyBadRequest(Exception):
    """Stand-in for openai.BadRequestError: a 400 'rejected by the safety system'. Named so that
    type(e).__name__ contains 'BadRequest' (what _is_safety_reject keys on)."""

    def __init__(self, msg="Error code: 400 - your request was rejected by the safety system."):
        super().__init__(msg)


# Confidence guard: it really is a 400 the SDK reraises rather than retries.
_SafetyBadRequest.__name__ = "BadRequestError"


class _FlakyImages:
    """Records calls; raises a safety-style 400 on every call whose prompt still contains the
    'Action:' clause, and succeeds (returns a real image) once the prompt has been sanitized."""

    def __init__(self):
        self.calls = []

    def _maybe_raise(self, kw):
        if "Action:" in kw.get("prompt", ""):
            raise _SafetyBadRequest()
        return _image_response(_tiny_png())

    def edit(self, **kw):
        self.calls.append(("edit", kw))
        return self._maybe_raise(kw)

    def generate(self, **kw):
        self.calls.append(("generate", kw))
        return self._maybe_raise(kw)


class _AlwaysSafetyImages:
    """Records calls; raises the safety-style 400 on every call regardless of prompt."""

    def __init__(self):
        self.calls = []

    def edit(self, **kw):
        self.calls.append(("edit", kw))
        raise _SafetyBadRequest()

    def generate(self, **kw):
        self.calls.append(("generate", kw))
        raise _SafetyBadRequest()


class _NonSafetyError(RuntimeError):
    """A generic (non-safety) failure: must placeholder WITHOUT a softened retry."""


class _NonSafetyImages:
    def __init__(self):
        self.calls = []

    def edit(self, **kw):
        self.calls.append(("edit", kw))
        raise _NonSafetyError("boom")

    def generate(self, **kw):
        self.calls.append(("generate", kw))
        raise _NonSafetyError("boom")


class _FakeClient:
    def __init__(self, images):
        self.images = images


def _ctx(settings, model):
    settings.models["image"] = model
    return (CostTracker(10.0, settings.ledger_path, settings.prices_usd),
            Cache(settings.cache_dir))


def _is_panel_placeholder(settings, png) -> bool:
    """Placeholders are flat grey 'L' images of the configured panel size (see _placeholder).
    Accepts raw bytes OR a path (generate_image/edit_image now return the cache Path)."""
    import io

    from PIL import Image
    w, h = (int(x) for x in settings.image["size"].split("x"))
    img = Image.open(io.BytesIO(png)) if isinstance(png, (bytes, bytearray)) else Image.open(png)
    return img.size == (w, h) and img.mode == "L"


def test_edit_gpt_image_2_omits_input_fidelity(settings, recording_client):
    tracker, cache = _ctx(settings, "gpt-image-2")
    edit_image(recording_client, settings, tracker, cache,
               stage="panels", prompt="p", ref_bytes=[b"a", b"b"], quality="medium")
    op, kw = recording_client.images.calls[0]
    assert op == "edit"
    assert "input_fidelity" not in kw                 # gpt-image-2 rejects it
    assert isinstance(kw["image"], list) and len(kw["image"]) == 2  # multi-ref preserved
    assert "moderation" not in kw                     # edit() has no moderation param


def test_edit_gpt_image_15_sends_input_fidelity(settings, recording_client):
    tracker, cache = _ctx(settings, "gpt-image-1.5")
    edit_image(recording_client, settings, tracker, cache,
               stage="panels", prompt="p", ref_bytes=[b"a"], quality="medium")
    _, kw = recording_client.images.calls[0]
    assert kw.get("input_fidelity") == "high"


def test_single_ref_passed_as_tuple_not_list(settings, recording_client):
    tracker, cache = _ctx(settings, "gpt-image-2")
    edit_image(recording_client, settings, tracker, cache,
               stage="panels", prompt="p", ref_bytes=[b"only"], quality="low")
    _, kw = recording_client.images.calls[0]
    assert not isinstance(kw["image"], list)          # one ref -> single FileTypes


def test_cache_prevents_second_paid_call(settings, recording_client):
    tracker, cache = _ctx(settings, "gpt-image-2")
    for _ in range(2):
        edit_image(recording_client, settings, tracker, cache,
                   stage="panels", prompt="same", ref_bytes=[b"a"], quality="low")
    assert len(recording_client.images.calls) == 1     # 2nd call served from cache


def test_generate_sends_moderation(settings, recording_client):
    tracker, cache = _ctx(settings, "gpt-image-2")
    generate_image(recording_client, settings, tracker, cache,
                   stage="sheets", prompt="sheet", quality="high")
    op, kw = recording_client.images.calls[0]
    assert op == "generate"
    assert "moderation" in kw                          # generate() supports it


def test_generate_cache_key_includes_moderation(settings, recording_client):
    """#14: changing `moderation` must invalidate the generate cache (force regeneration)."""
    tracker, cache = _ctx(settings, "gpt-image-2")
    settings.image["moderation"] = "low"
    generate_image(recording_client, settings, tracker, cache,
                   stage="sheets", prompt="same", quality="high")
    # same prompt/params -> cache hit, no new call
    generate_image(recording_client, settings, tracker, cache,
                   stage="sheets", prompt="same", quality="high")
    assert len(recording_client.images.calls) == 1
    # flip moderation -> different cache key -> a fresh paid call
    settings.image["moderation"] = "auto"
    generate_image(recording_client, settings, tracker, cache,
                   stage="sheets", prompt="same", quality="high")
    assert len(recording_client.images.calls) == 2
    assert recording_client.images.calls[-1][1]["moderation"] == "auto"


def test_cached_panel_reused_after_budget_stop(settings, recording_client):
    """#12: once the budget trips, an already-cached panel is still emitted (free), not
    overwritten with a placeholder; only a genuinely-uncached panel becomes a placeholder.
    Both kinds carry the correct labeling (no synthetic '(failed)' RuntimeError -> #18)."""
    settings.models["image"] = "gpt-image-2"
    cache = Cache(settings.cache_dir)

    # Pre-warm the content cache for panel 1 exactly as panels.run would key it
    # (no refs -> generate, stage="panels", panel_quality, prompt unchanged).
    quality = settings.image["panel_quality"]
    warm = CostTracker(10.0, settings.ledger_path, settings.prices_usd)
    cached_png = generate_image(recording_client, settings, warm, cache,
                                stage="panels", prompt="cached panel", quality=quality)
    assert not _is_panel_placeholder(settings, cached_png)
    calls_after_warm = len(recording_client.images.calls)

    # Budget already exhausted: any *new* paid call must trip BudgetExceeded.
    broke = CostTracker(0.0, settings.ledger_path, settings.prices_usd)
    prompts = [
        PanelPrompt(panel_number=1, prompt="cached panel", ref_sheets=[]),   # in cache -> free
        PanelPrompt(panel_number=2, prompt="brand new panel", ref_sheets=[]),  # uncached -> budget
    ]
    manifest = panels.run(recording_client, settings, broke, cache,
                          prompts, sheets={}, chapter_number=1)

    # No new paid call happened for the cached panel (or the budget-blocked one).
    assert len(recording_client.images.calls) == calls_after_warm

    out1 = __import__("pathlib").Path(manifest[0]["path"]).read_bytes()
    out2 = __import__("pathlib").Path(manifest[1]["path"]).read_bytes()
    assert not _is_panel_placeholder(settings, out1)   # cached panel reused, not clobbered
    assert out1 == cached_png.read_bytes()             # manifest points at the content-cache file
    assert _is_panel_placeholder(settings, out2)        # uncached panel -> budget placeholder


# ── scene-background consistency ─────────────────────────────────────────────
def _seed_settings(settings, chapter_number, by_pn):
    """Write a minimal chapter-<n>.panels.json mapping panel_number -> setting."""
    data = [{"panel_number": pn, "setting": s} for pn, s in sorted(by_pn.items())]
    path = settings.artifacts_dir / f"chapter-{chapter_number}.panels.json"
    path.write_text(json.dumps(data), encoding="utf-8")


def _prompts(*panel_numbers):
    return [PanelPrompt(panel_number=pn, prompt=f"panel {pn}", ref_sheets=[]) for pn in panel_numbers]


def test_scene_of_groups_consecutive_and_same(settings):
    """Consecutive identical (and 'Same…') settings form one scene; a new setting starts a new one."""
    prompts = _prompts(1, 2, 3, 4, 5)
    settings_map = {
        1: "Interior of the dragon carriage.",
        2: "Interior of the dragon carriage.",
        3: "Same carriage interior.",          # 'same…' -> continues the carriage scene
        4: "A sunlit mansion garden.",          # distinct -> new scene
        5: "A sunlit mansion garden.",
    }
    scene = panels._scene_of(prompts, settings_map, 1)
    assert scene == {1: 1, 2: 1, 3: 1, 4: 4, 5: 4}


def test_scene_of_rejects_sesame_samey_as_continuation(settings):
    """#2: 'same' continuation is word-boundary/start-anchored — 'sesame'/'samey' do NOT
    continue the previous scene; only a real leading 'Same…' does."""
    prompts = _prompts(1, 2, 3, 4)
    settings_map = {
        1: "Interior of the carriage.",
        2: "A sesame field at dawn.",      # contains 'same' as a substring -> NOT a continuation
        3: "The samey corridor.",           # 'samey' -> NOT a continuation either
        4: "Same carriage interior.",       # real leading 'Same' -> continues scene 3
    }
    scene = panels._scene_of(prompts, settings_map, 1)
    # 1 alone; 2 distinct; 3 distinct; 4 continues 3
    assert scene == {1: 1, 2: 2, 3: 3, 4: 3}


def test_scene_of_empty_settings_each_panel_own_scene(settings):
    """#3: with an EMPTY/missing settings_map every panel is its own scene (no collapse into
    one chapter-wide scene anchored on panel 1)."""
    prompts = _prompts(1, 2, 3)
    scene = panels._scene_of(prompts, {}, 1)
    assert scene == {1: 1, 2: 2, 3: 3}
    # an explicit empty-string setting behaves the same (unknown -> own scene)
    scene2 = panels._scene_of(prompts, {1: "", 2: "", 3: ""}, 1)
    assert scene2 == {1: 1, 2: 2, 3: 3}


def test_placeholdered_anchor_not_used_as_ref_for_dependents(settings, recording_client, png_bytes):
    """#4: if a scene's anchor is a placeholder (budget/failed), its dependents must be
    generated WITHOUT a background reference — never told to match the grey placeholder."""
    settings.models["image"] = "gpt-image-2"
    _seed_settings(settings, 1, {1: "A quiet library.", 2: "A quiet library."})

    sheet = settings.artifacts_dir / "hero.png"
    sheet.write_bytes(png_bytes)
    prompts = [
        PanelPrompt(panel_number=1, prompt="anchor panel", ref_sheets=["hero"]),   # anchor
        PanelPrompt(panel_number=2, prompt="dependent panel", ref_sheets=["hero"]),  # dependent
    ]

    # Budget = 0 -> the anchor (PASS 1) trips BudgetExceeded and becomes a placeholder (ok=False),
    # but a content-cache prewarm lets the dependent (PASS 2) still be generated for free.
    quality = settings.image["panel_quality"]
    warm = CostTracker(10.0, settings.ledger_path, settings.prices_usd)
    cache = Cache(settings.cache_dir)
    # Prewarm exactly the call the dependent would make if it had NO anchor ref:
    # one char sheet -> edit with a single ref, prompt = "dependent panel" + PRESERVE.
    edit_image(recording_client, settings, warm, cache, stage="panels",
               prompt="dependent panel" + panels.PRESERVE, ref_bytes=[png_bytes], quality=quality)
    calls_after_warm = len(recording_client.images.calls)

    broke = CostTracker(0.0, settings.ledger_path, settings.prices_usd)
    manifest = panels.run(recording_client, settings, broke, cache,
                          prompts, sheets={"hero": str(sheet)}, chapter_number=1)

    # The anchor was placeholdered.
    assert _is_panel_placeholder(settings, manifest[0]["path"])
    # The dependent served the NO-anchor cached call (free) -> no new paid call, not a placeholder,
    # which proves it was generated without the placeholder anchor as a background reference.
    assert len(recording_client.images.calls) == calls_after_warm
    assert not _is_panel_placeholder(settings, manifest[1]["path"])
    # And the internal "ok" flag never leaks into the persisted manifest.
    assert all("ok" not in d for d in manifest)


def test_scene_of_normalizes_case_and_whitespace(settings):
    """lowercase / strip / collapsed-whitespace settings compare equal (one scene)."""
    prompts = _prompts(1, 2, 3)
    settings_map = {
        1: "Interior of the   carriage.",
        2: "  interior OF the carriage. ",       # same after normalization
        3: "A different street at night.",        # new scene
    }
    scene = panels._scene_of(prompts, settings_map, 1)
    assert scene == {1: 1, 2: 1, 3: 3}


def test_nonanchor_panel_gets_anchor_as_extra_ref(settings, recording_client, png_bytes):
    """Two panels in one scene, each with one char sheet:
       - the anchor (panel 1) edit-call has exactly 1 ref (the char sheet);
       - the non-anchor (panel 2) edit-call has 2 refs (anchor image FIRST + char sheet)."""
    settings.models["image"] = "gpt-image-2"
    tracker, cache = _ctx(settings, "gpt-image-2")
    _seed_settings(settings, 1, {1: "A quiet library.", 2: "A quiet library."})

    sheet = settings.artifacts_dir / "hero.png"
    sheet.write_bytes(png_bytes)
    prompts = [
        PanelPrompt(panel_number=1, prompt="panel 1", ref_sheets=["hero"]),
        PanelPrompt(panel_number=2, prompt="panel 2", ref_sheets=["hero"]),
    ]
    panels.run(recording_client, settings, tracker, cache,
               prompts, sheets={"hero": str(sheet)}, chapter_number=1)

    by_prompt = {}
    for op, kw in recording_client.images.calls:
        assert op == "edit"
        by_prompt[kw["prompt"]] = kw["image"]

    anchor_img = next(v for p, v in by_prompt.items() if p.startswith("panel 1"))
    other_img = next(v for p, v in by_prompt.items() if p.startswith("panel 2"))
    # anchor: single char sheet -> single FileTypes (not a list)
    assert not isinstance(anchor_img, list)
    # non-anchor: anchor image + char sheet = 2 refs (char sheets + 1)
    assert isinstance(other_img, list) and len(other_img) == 2
    # the anchor reference is placed FIRST
    assert other_img[0][0] == "ref0.png"
    # and the background instruction is appended to the non-anchor prompt
    other_prompt = next(p for p in by_prompt if p.startswith("panel 2"))
    assert "SAME background" in other_prompt


def test_toggle_off_adds_no_extra_ref(settings, recording_client, png_bytes):
    """With scene.background_consistency=False, the non-anchor panel gets NO anchor ref."""
    settings.models["image"] = "gpt-image-2"
    settings.scene = {"background_consistency": False}
    tracker, cache = _ctx(settings, "gpt-image-2")
    _seed_settings(settings, 1, {1: "A quiet library.", 2: "A quiet library."})

    sheet = settings.artifacts_dir / "hero.png"
    sheet.write_bytes(png_bytes)
    prompts = [
        PanelPrompt(panel_number=1, prompt="panel 1", ref_sheets=["hero"]),
        PanelPrompt(panel_number=2, prompt="panel 2", ref_sheets=["hero"]),
    ]
    panels.run(recording_client, settings, tracker, cache,
               prompts, sheets={"hero": str(sheet)}, chapter_number=1)

    for op, kw in recording_client.images.calls:
        assert op == "edit"
        assert not isinstance(kw["image"], list)        # one char sheet only, no anchor ref
        assert "SAME background" not in kw["prompt"]


# ── #6: 'Same…' with no established predecessor starts its own scene ──────────
def test_scene_of_same_after_empty_starts_own_scene(settings):
    """A 'Same…' panel following an empty/unknown setting must NOT attach to that
    un-established panel; it starts its own scene."""
    prompts = _prompts(1, 2)
    scene = panels._scene_of(prompts, {1: "", 2: "Same room."}, 1)
    assert scene == {1: 1, 2: 2}


def test_scene_of_leading_same_starts_own_scene(settings):
    """A chapter that LEADS with 'Same…' has no predecessor to continue -> own scene."""
    prompts = _prompts(1, 2)
    scene = panels._scene_of(prompts, {1: "Same place.", 2: "A new street."}, 1)
    assert scene == {1: 1, 2: 2}


def test_scene_of_same_after_real_setting_still_continues(settings):
    """Regression guard: a 'Same…' that DOES follow a real established setting still groups."""
    prompts = _prompts(1, 2)
    scene = panels._scene_of(prompts, {1: "A quiet library.", 2: "Same library."}, 1)
    assert scene == {1: 1, 2: 1}


def test_scene_of_empty_then_same_then_explicit(settings):
    """empty -> same (own scene) -> explicit setting that matches the 'same' panel groups onto
    the 'same' panel, not the empty one."""
    prompts = _prompts(1, 2, 3)
    settings_map = {1: "", 2: "Same room.", 3: "same room."}
    scene = panels._scene_of(prompts, settings_map, 1)
    # 1 alone (empty); 2 own scene (no predecessor); 3 'same…' with prev_norm still empty -> own
    assert scene == {1: 1, 2: 2, 3: 3}


# ── #4: a missing/unreadable sheet file falls back to a no-ref generate ───────
def test_unreadable_sheet_falls_back_to_generate(settings, recording_client):
    """#bug-low: a sheet listed in sheets but missing on disk must NOT be mislabeled '(failed)';
    the panel falls back to a no-ref generate instead of an edit-with-refs."""
    settings.models["image"] = "gpt-image-2"
    tracker, cache = _ctx(settings, "gpt-image-2")
    prompts = [PanelPrompt(panel_number=1, prompt="lonely panel", ref_sheets=["ghost"])]
    manifest = panels.run(recording_client, settings, tracker, cache,
                          prompts, sheets={"ghost": str(settings.artifacts_dir / "nope.png")},
                          chapter_number=1)
    # exactly one paid call, and it is a generate (no refs) — not an edit, not a placeholder
    assert len(recording_client.images.calls) == 1
    assert recording_client.images.calls[0][0] == "generate"
    assert not _is_panel_placeholder(settings, manifest[0]["path"])
    # PRESERVE must not be appended when we generate without refs
    assert "EXACTLY as in the reference sheet" not in recording_client.images.calls[0][1]["prompt"]


# ── #3: a panel whose requested sheet is absent warns + is tallied ───────────
def test_missing_sheet_warns_and_still_generates(settings, recording_client, capsys):
    """A requested ref_sheet absent from `sheets` is reported (identity not anchored) but the
    panel is still generated (anchor on survivors / no-ref)."""
    settings.models["image"] = "gpt-image-2"
    tracker, cache = _ctx(settings, "gpt-image-2")
    prompts = [PanelPrompt(panel_number=7, prompt="orphan", ref_sheets=["absent_hero"])]
    panels.run(recording_client, settings, tracker, cache,
               prompts, sheets={}, chapter_number=1)
    err = capsys.readouterr().err
    assert "panel 7: no sheet for ['absent_hero']" in err
    assert "1/1 panels missing a character sheet" in err


# ── #1 / #2: budget placeholders are de-duplicated and summarized ────────────
def test_budget_warning_deduped_and_summarized(settings, recording_client, capsys):
    """#1: only ONE actionable budget line is printed regardless of how many uncached panels
    trip the cap. #2: a run-level summary lists the placeholdered panels by cause."""
    settings.models["image"] = "gpt-image-2"
    broke = CostTracker(0.0, settings.ledger_path, settings.prices_usd)
    cache = Cache(settings.cache_dir)
    prompts = [PanelPrompt(panel_number=n, prompt=f"new {n}", ref_sheets=[]) for n in (1, 2, 3)]
    manifest = panels.run(recording_client, settings, broke, cache,
                          prompts, sheets={}, chapter_number=1)
    err = capsys.readouterr().err
    # all three are budget placeholders
    assert all(_is_panel_placeholder(settings, d["path"]) for d in manifest)
    # exactly one "budget cap" actionable line (no per-panel storm)
    assert err.count("budget cap") == 1
    # run-level summary names the placeholdered panels and their cause
    assert "3/3 panels are PLACEHOLDERS" in err
    assert "budget: [1, 2, 3]" in err
    # internal flags never leak into the persisted manifest
    assert all("ok" not in d and "reason" not in d for d in manifest)


# ── #5: per-panel progress counter reaches stderr ────────────────────────────
def test_progress_counter_printed(settings, recording_client, capsys):
    """A slow PASS isn't mistaken for a hung one: each completed panel prints a k/N counter."""
    settings.models["image"] = "gpt-image-2"
    tracker, cache = _ctx(settings, "gpt-image-2")
    prompts = [PanelPrompt(panel_number=n, prompt=f"p{n}", ref_sheets=[]) for n in (1, 2)]
    panels.run(recording_client, settings, tracker, cache,
               prompts, sheets={}, chapter_number=1)
    err = capsys.readouterr().err
    assert "[panels] 2 panels" in err
    assert "1/2 (anchor)" in err and "2/2 (anchor)" in err


# ── safety-reject -> retry once with a softened prompt ────────────────────────
_RISKY_PROMPT = (
    "Medium shot from the waist up. Characters in frame: Ram, a maid. "
    "Action: Ram licks her lips with a graceful, teasing gesture. "
    "Setting: a quiet library. Mood: playful. Camera angle: low. "
    "Leave clear empty space (sky, wall, or negative space) for speech bubbles."
)


def test_safety_reject_retries_softened_and_recovers(settings, capsys):
    """(a) A safety-style 400 on the FIRST generate triggers ONE retry with a sanitized prompt
    (no 'Action:' clause); the second call succeeds, so the panel is a real cached image with
    ok=True / reason=None — never a placeholder."""
    settings.models["image"] = "gpt-image-2"
    tracker, cache = _ctx(settings, "gpt-image-2")
    client = _FakeClient(_FlakyImages())
    prompts = [PanelPrompt(panel_number=1, prompt=_RISKY_PROMPT, ref_sheets=[])]  # no refs -> generate

    manifest = panels.run(client, settings, tracker, cache, prompts, sheets={}, chapter_number=1)

    # Two attempts: the rejected original, then the sanitized retry.
    assert len(client.images.calls) == 2
    assert all(op == "generate" for op, _ in client.images.calls)
    assert "Action:" in client.images.calls[0][1]["prompt"]          # original tripped the filter
    softened = client.images.calls[1][1]["prompt"]
    assert "Action:" not in softened                                 # clause stripped on retry
    assert "licks her lips" not in softened
    assert "Setting: a quiet library." in softened                   # everything else preserved
    assert "tasteful, wholesome, non-suggestive, and fully clothed" in softened
    # Recovered: a real image (not the grey placeholder).
    assert not _is_panel_placeholder(settings, manifest[0]["path"])
    err = capsys.readouterr().err
    assert "panel 1: safety-rejected; retrying with softened prompt" in err
    assert "PLACEHOLDERS" not in err                                 # no failure summary


def test_safety_reject_still_fails_after_softening_placeholders(settings, capsys):
    """(b) When the softened retry is ALSO rejected, the panel falls back to a grey placeholder
    with reason 'failed' and logs that it failed even after softening."""
    settings.models["image"] = "gpt-image-2"
    tracker, cache = _ctx(settings, "gpt-image-2")
    client = _FakeClient(_AlwaysSafetyImages())
    prompts = [PanelPrompt(panel_number=3, prompt=_RISKY_PROMPT, ref_sheets=[])]

    manifest = panels.run(client, settings, tracker, cache, prompts, sheets={}, chapter_number=1)

    assert len(client.images.calls) == 2                             # original + one softened retry
    assert _is_panel_placeholder(settings, manifest[0]["path"])
    err = capsys.readouterr().err
    assert "panel 3: safety-rejected; retrying with softened prompt" in err
    assert "panel 3 failed even after softening" in err
    assert "failed: [3]" in err                                      # summarized as a 'failed' cause


def test_nonsafety_error_placeholders_without_retry(settings, capsys):
    """(c) A NON-safety exception must placeholder immediately — exactly one generation attempt,
    no softened retry."""
    settings.models["image"] = "gpt-image-2"
    tracker, cache = _ctx(settings, "gpt-image-2")
    client = _FakeClient(_NonSafetyImages())
    prompts = [PanelPrompt(panel_number=5, prompt=_RISKY_PROMPT, ref_sheets=[])]

    manifest = panels.run(client, settings, tracker, cache, prompts, sheets={}, chapter_number=1)

    assert len(client.images.calls) == 1                             # NO retry for a generic error
    assert _is_panel_placeholder(settings, manifest[0]["path"])
    err = capsys.readouterr().err
    assert "safety-rejected" not in err
    assert "panel 5 failed (_NonSafetyError)" in err
    assert "failed: [5]" in err


# ── _sanitize_prompt keeps the style/preserve guards when Action: is the LAST labeled clause ──
def _trailing_action_prompt(ref_sheets):
    """A realistic prompt whose Action: clause is the LAST labeled segment (no Setting:/Mood:/
    Camera follows — emitted by build_prompt when setting=='', emotion=='neutral', angle=='eye-level'),
    followed only by the bubbles line, GLOBAL_STYLE and the NO-letters guard. Mirrors script.build_prompt."""
    from ln2manga.bible import GLOBAL_STYLE
    prompt = (
        "Medium shot from the waist up. Characters in frame: Ram, a maid. "
        "Action: Ram licks her lips with a graceful, teasing gesture. "
        "Leave clear empty space (sky, wall, or negative space) for speech bubbles. "
        + GLOBAL_STYLE + " "
        "NO letters, numbers, logos, brand marks or insignia on clothing, accessories or any object."
    )
    return PanelPrompt(panel_number=9, prompt=prompt, ref_sheets=ref_sheets)


def test_sanitize_trailing_action_keeps_style_preserve_and_noletters(settings, capsys, png_bytes):
    """Regression: when Action: is the LAST labeled clause, the softened retry must strip ONLY the
    Action sentence — GLOBAL_STYLE, the NO-letters guard and PRESERVE must all survive (they used to
    be swallowed when the lazy match ran to end-of-string). A char sheet is supplied so PRESERVE is
    appended to the prompt and we can assert it is retained too."""
    from ln2manga.bible import GLOBAL_STYLE
    settings.models["image"] = "gpt-image-2"
    tracker, cache = _ctx(settings, "gpt-image-2")
    client = _FakeClient(_FlakyImages())

    sheet = settings.artifacts_dir / "ram.png"
    sheet.write_bytes(png_bytes)
    prompts = [_trailing_action_prompt(["ram"])]

    manifest = panels.run(client, settings, tracker, cache,
                          prompts, sheets={"ram": str(sheet)}, chapter_number=1)

    # The first edit was rejected (Action: present); the softened retry succeeded.
    assert len(client.images.calls) == 2
    assert all(op == "edit" for op, _ in client.images.calls)
    softened = client.images.calls[1][1]["prompt"]
    # Action sentence gone…
    assert "Action:" not in softened
    assert "licks her lips" not in softened
    # …but every guard that follows it is preserved (this is the bug being fixed).
    assert GLOBAL_STYLE in softened
    assert "NO letters, numbers, logos" in softened
    assert "Leave clear empty space" in softened
    assert "EXACTLY as in the reference sheet" in softened          # PRESERVE retained
    assert "tasteful, wholesome, non-suggestive, and fully clothed" in softened
    # Recovered to a real image, not a placeholder.
    assert not _is_panel_placeholder(settings, manifest[0]["path"])


def test_sanitize_trailing_action_keeps_action_with_internal_period(settings):
    """An action sentence containing an internal period (e.g. 'Mr. Roswaal …') is removed in FULL,
    not just up to the first period; the trailing guards still survive."""
    from ln2manga.bible import GLOBAL_STYLE
    prompt = (
        "Medium shot. Characters in frame: Roswaal. "
        "Action: Mr. Roswaal bows deeply. "
        "Leave clear empty space (sky, wall, or negative space) for speech bubbles. "
        + GLOBAL_STYLE
    )
    out = panels._sanitize_prompt(prompt)
    assert "Roswaal bows" not in out                                # whole action removed
    assert "Mr." not in out
    assert "Leave clear empty space" in out
    assert GLOBAL_STYLE in out
    assert out.endswith(panels.SOFTEN)


# ── #1: the budget-cap warning is claimed atomically across image workers ─────
def test_budget_warning_printed_once_under_concurrency(settings, recording_client, capsys):
    """Regression for the racy check-then-set: with several image workers all tripping the cap in
    a widened window, the actionable 'budget cap' line must still be printed EXACTLY once. The race
    window is widened deterministically via a Lock wrapper that sleeps before the claim is granted —
    without the lock-guarded claim every worker would read budget_warned as falsey and print."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    class _SlowLock:
        """A real lock whose acquire path sleeps, widening the check-then-set window so a missing
        lock would let all workers print. Used as state['_warn_lock']."""

        def __init__(self):
            self._lock = threading.Lock()

        def __enter__(self):
            self._lock.acquire()
            time.sleep(0.02)
            return self

        def __exit__(self, *a):
            self._lock.release()

    settings.models["image"] = "gpt-image-2"
    broke = CostTracker(0.0, settings.ledger_path, settings.prices_usd)
    cache = Cache(settings.cache_dir)
    quality = settings.image["panel_quality"]
    state = {"_warn_lock": _SlowLock()}

    n = 8
    prompts = [PanelPrompt(panel_number=i, prompt=f"new {i}", ref_sheets=[]) for i in range(n)]

    def _one(pp):
        return panels._gen_panel(recording_client, settings, broke, cache, pp,
                                 sheets={}, quality=quality, state=state)

    with ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(_one, prompts))

    err = capsys.readouterr().err
    assert err.count("budget cap") == 1
    assert state["budget_warned"] is True


# ── scene-anchor bytes are read from disk ONCE per distinct anchor ────────────
def test_scene_anchor_bytes_read_once_per_anchor(settings, recording_client, png_bytes, monkeypatch):
    """PASS 2 reads each scene-anchor PNG from disk ONCE and reuses the bytes across every
    dependent panel of that scene (was: one read per dependent). The cached read returns the same
    file bytes, so the panel content-cache key is unchanged. Two dependents share one anchor here;
    the anchor file must be read exactly once."""
    import pathlib

    settings.models["image"] = "gpt-image-2"
    tracker, cache = _ctx(settings, "gpt-image-2")
    # One scene: panel 1 is the anchor; panels 2 and 3 are dependents on it.
    _seed_settings(settings, 1, {1: "A great hall.", 2: "A great hall.", 3: "A great hall."})

    sheet = settings.artifacts_dir / "hero.png"
    sheet.write_bytes(png_bytes)
    prompts = [
        PanelPrompt(panel_number=1, prompt="anchor", ref_sheets=["hero"]),
        PanelPrompt(panel_number=2, prompt="dep a", ref_sheets=["hero"]),
        PanelPrompt(panel_number=3, prompt="dep b", ref_sheets=["hero"]),
    ]

    # The module-level read cache persists across runs; clear it so this run starts cold.
    panels._read_anchor_bytes.cache_clear()

    # Count disk reads PER FILE so a shared anchor's reads are isolated from char-sheet reads.
    real_read = pathlib.Path.read_bytes
    reads: dict[str, int] = {}

    def _counting_read(self):
        reads[str(self)] = reads.get(str(self), 0) + 1
        return real_read(self)

    monkeypatch.setattr(pathlib.Path, "read_bytes", _counting_read)
    manifest = panels.run(recording_client, settings, tracker, cache,
                          prompts, sheets={"hero": str(sheet)}, chapter_number=1)

    # Identify the scene-anchor file from PASS 2's edit calls: the anchor ref is placed FIRST
    # ("ref0.png") and its bytes are the panel-1 (anchor) output. Both dependents must reuse it.
    anchor_out = pathlib.Path(manifest[0]["path"])
    anchor_path = str(anchor_out)
    assert reads.get(anchor_path, 0) == 1, (
        f"anchor read {reads.get(anchor_path, 0)} times (expected 1); reads={reads}")

    # And the bytes actually handed to the two dependents are byte-identical to the anchor file
    # (cache key unchanged): every dependent edit call leads with the same first ref = anchor bytes.
    anchor_bytes = anchor_out.read_bytes()
    dep_first_refs = [kw["image"][0][1] for op, kw in recording_client.images.calls
                      if op == "edit" and isinstance(kw["image"], list)]
    assert dep_first_refs, "expected dependent edit calls with a multi-ref anchor-first image list"
    assert all(b == anchor_bytes for b in dep_first_refs)


def test_gen_panel_budget_warns_without_state(settings, recording_client, capsys):
    """When state is None (no run-level dict), a budget trip must still print the cap line
    (first=True fallback) and placeholder, rather than crash on state.get()."""
    settings.models["image"] = "gpt-image-2"
    broke = CostTracker(0.0, settings.ledger_path, settings.prices_usd)
    cache = Cache(settings.cache_dir)
    quality = settings.image["panel_quality"]
    pp = PanelPrompt(panel_number=1, prompt="uncached", ref_sheets=[])

    result = panels._gen_panel(recording_client, settings, broke, cache, pp,
                               sheets={}, quality=quality, state=None)

    assert result["reason"] == "budget"
    assert _is_panel_placeholder(settings, result["path"])
    assert capsys.readouterr().err.count("budget cap") == 1
