"""CLI orchestration regression tests for run-all: cost estimate, cumulative-budget banner,
degraded-output summary, and the non-zero exit code on partial failure.

These drive `run-all` with the stage functions monkeypatched, so the tests exercise ONLY the
orchestration logic in cli.run_all (banner/estimate/summary/exit) without network or image gen.
"""
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from ln2manga import cli
from ln2manga.artifacts import PanelPrompt, PanelSpec

runner = CliRunner()


# ── helpers (#6 cost estimate) ───────────────────────────────────────────────
def test_flat_image_price_reads_config(settings):
    settings.models["image"] = "gpt-image-2"
    # from config/default.yaml: gpt-image-2 high=0.22, medium=0.12
    assert cli._flat_image_price(settings, "high") == pytest.approx(0.22)
    assert cli._flat_image_price(settings, "medium") == pytest.approx(0.12)


def test_flat_image_price_falls_back_when_model_or_quality_missing(settings):
    settings.models["image"] = "no-such-model"          # -> default row
    assert cli._flat_image_price(settings, "low") == pytest.approx(0.02)   # default.low
    # unknown quality on a known model falls back to that row's medium, never raises
    settings.models["image"] = "gpt-image-2"
    assert cli._flat_image_price(settings, "ultra") == pytest.approx(0.12)  # -> medium


def test_estimate_usd_sums_sheets_and_panels(settings):
    settings.models["image"] = "gpt-image-2"
    settings.image["sheet_quality"] = "high"            # 0.22
    settings.image["panel_quality"] = "medium"          # 0.12
    assert cli._estimate_usd(settings, n_sheets=2, n_panels=5) == pytest.approx(2 * 0.22 + 5 * 0.12)


# ── run-all orchestration harness ────────────────────────────────────────────
def _patch_pipeline(monkeypatch, settings, *, cast, prompts, pimgs, sheets, spent=0.0):
    """Patch _ctx + every stage so run-all executes purely on these fakes."""
    tracker = SimpleNamespace(
        spent=spent,
        image_calls=len(prompts),
        max_usd=settings.budget["max_usd"],
        remaining=lambda: max(0.0, settings.budget["max_usd"] - spent),
    )
    specs = [PanelSpec(panel_number=i + 1, characters_present=([cast[0]] if cast else []))
             for i in range(len(prompts))]

    monkeypatch.setattr(cli, "_ctx",
                        lambda *a, **k: (settings, object(), tracker, object()))
    monkeypatch.setattr(cli.scrape, "run",
                        lambda *a, **k: SimpleNamespace(paragraphs=["p"], scene_breaks=[]))
    monkeypatch.setattr(cli.parse, "run", lambda *a, **k: specs)
    monkeypatch.setattr(cli.script, "run", lambda *a, **k: prompts)
    monkeypatch.setattr(cli.charsheet, "characters_in", lambda s: list(cast))
    monkeypatch.setattr(cli.charsheet, "run", lambda *a, **k: dict(sheets))
    monkeypatch.setattr(cli.panels, "run", lambda *a, **k: pimgs)
    monkeypatch.setattr(cli.mangapost, "run", lambda *a, **k: pimgs)
    monkeypatch.setattr(cli.layout, "run", lambda *a, **k: ["page"])
    monkeypatch.setattr(cli.lettering, "run", lambda *a, **k: ["page"])
    monkeypatch.setattr(cli.export, "run",
                        lambda *a, **k: {"pdf": "out.pdf", "cbz": "out.cbz"})
    return tracker


def test_run_all_clean_run_exits_zero_no_warning(monkeypatch, settings):
    """#1/#4: a fully-generated run prints no WARNING and exits 0."""
    prompts = [PanelPrompt(panel_number=1, prompt="p1"), PanelPrompt(panel_number=2, prompt="p2")]
    pimgs = [{"panel_number": 1, "path": "/c/panel_0001.png"},
             {"panel_number": 2, "path": "/c/panel_0002.png"}]
    _patch_pipeline(monkeypatch, settings, cast=["Hero"], prompts=prompts, pimgs=pimgs,
                    sheets={"Hero": "/c/hero.png"})

    res = runner.invoke(cli.app, ["run-all", "--dry-run"])
    assert res.exit_code == 0, res.output
    assert "WARNING" not in res.output


def test_run_all_placeholders_warn_and_exit_2(monkeypatch, settings):
    """#1/#4: when panels are grey placeholders the summary warns and exit code is 2."""
    prompts = [PanelPrompt(panel_number=1, prompt="p1"), PanelPrompt(panel_number=2, prompt="p2")]
    pimgs = [{"panel_number": 1, "path": "/c/panel_0001.png"},
             {"panel_number": 2, "path": "/c/panel_0002_placeholder.png"}]
    _patch_pipeline(monkeypatch, settings, cast=["Hero"], prompts=prompts, pimgs=pimgs,
                    sheets={"Hero": "/c/hero.png"})

    res = runner.invoke(cli.app, ["run-all", "--dry-run"])
    assert res.exit_code == 2, res.output
    assert "1/2 panels are grey placeholders" in " ".join(res.output.split())


def test_run_all_missing_sheet_warn_and_exit_2(monkeypatch, settings):
    """#1/#4: a character with no reference sheet is named in the summary and trips exit 2."""
    prompts = [PanelPrompt(panel_number=1, prompt="p1")]
    pimgs = [{"panel_number": 1, "path": "/c/panel_0001.png"}]   # no placeholder
    _patch_pipeline(monkeypatch, settings, cast=["Hero", "Sidekick"], prompts=prompts,
                    pimgs=pimgs, sheets={"Hero": "/c/hero.png"})   # Sidekick missing

    res = runner.invoke(cli.app, ["run-all", "--dry-run"])
    assert res.exit_code == 2, res.output
    assert "Sidekick" in res.output
    # rich may wrap the message across lines in a narrow terminal; normalize whitespace first.
    assert "no reference sheet" in " ".join(res.output.split())


def test_run_all_prints_estimate(monkeypatch, settings):
    """#6: an upfront cost estimate prints in dry-run too (over-estimate, from config prices)."""
    settings.models["image"] = "gpt-image-2"
    settings.image["sheet_quality"] = "high"
    settings.image["panel_quality"] = "medium"
    prompts = [PanelPrompt(panel_number=i + 1, prompt=f"p{i}") for i in range(3)]
    pimgs = [{"panel_number": i + 1, "path": f"/c/panel_{i+1:04d}.png"} for i in range(3)]
    _patch_pipeline(monkeypatch, settings, cast=["Hero"], prompts=prompts, pimgs=pimgs,
                    sheets={"Hero": "/c/hero.png"})

    res = runner.invoke(cli.app, ["run-all", "--dry-run"])
    assert res.exit_code == 0, res.output
    norm = " ".join(res.output.split())
    assert "estimate :" in norm
    assert "1 sheets + 3 panels" in norm


def test_run_all_live_banner_shows_ledger(monkeypatch, settings):
    """#2: a live run surfaces already-spent / remaining BEFORE generation; dry-run does not."""
    prompts = [PanelPrompt(panel_number=1, prompt="p1")]
    pimgs = [{"panel_number": 1, "path": "/c/panel_0001.png"}]
    settings.budget["max_usd"] = 10.0
    _patch_pipeline(monkeypatch, settings, cast=["Hero"], prompts=prompts, pimgs=pimgs,
                    sheets={"Hero": "/c/hero.png"}, spent=7.5)

    # live path (no --dry-run): build_client is never reached because _ctx is patched
    res = runner.invoke(cli.app, ["run-all"])
    norm = " ".join(res.output.split())   # collapses the aligned "budget   :" padding to "budget :"
    assert "budget :" in norm
    assert "$7.50 already spent" in norm
    assert "$2.50 of $10.00 remaining" in norm


def test_run_all_dry_run_omits_ledger_banner(monkeypatch, settings):
    prompts = [PanelPrompt(panel_number=1, prompt="p1")]
    pimgs = [{"panel_number": 1, "path": "/c/panel_0001.png"}]
    _patch_pipeline(monkeypatch, settings, cast=["Hero"], prompts=prompts, pimgs=pimgs,
                    sheets={"Hero": "/c/hero.png"})

    res = runner.invoke(cli.app, ["run-all", "--dry-run"])
    assert "already spent" not in res.output


# ── clean command ────────────────────────────────────────────────────────────
def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x", encoding="utf-8")


def _seed_data(settings):
    _touch(settings.out_dir / "chapter-1_page_01.png")
    _touch(settings.artifacts_dir / "chapter-1.panels.json")
    _touch(settings.cache_dir("panels") / "p.png")
    _touch(settings.cache_dir("refs") / "subaru.png")    # downloaded ref
    _touch(settings.raw_dir / "chapter-1.json")
    settings.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    settings.ledger_path.write_text(
        '{"spent": 5.0, "image_calls": 3, "events": []}', encoding="utf-8")


def test_clean_default_clears_caches_keeps_output_refs_and_budget(monkeypatch, settings):
    import json as _json
    monkeypatch.setattr(cli, "load_settings", lambda *a, **k: settings)
    _seed_data(settings)
    _touch(settings.out_dir / "chapter-1.pdf")
    _touch(settings.out_dir / "chapter-1.cbz")

    res = runner.invoke(cli.app, ["clean", "--yes"])
    assert res.exit_code == 0, res.output
    # regenerable caches + scraped source gone
    assert not (settings.cache_dir("panels") / "p.png").exists()
    assert not (settings.raw_dir / "chapter-1.json").exists()
    # final rendered output + manifests PRESERVED by default (no --output)
    assert (settings.out_dir / "chapter-1_page_01.png").exists()
    assert (settings.out_dir / "chapter-1.pdf").exists()
    assert (settings.out_dir / "chapter-1.cbz").exists()
    assert (settings.artifacts_dir / "chapter-1.panels.json").exists()
    # downloaded refs + ledger PRESERVED by default
    assert (settings.cache_dir("refs") / "subaru.png").exists()
    assert _json.loads(settings.ledger_path.read_text())["spent"] == 5.0
    assert "reference images kept" in res.output


def test_clean_all_chapter_scoped_leaves_other_chapters(monkeypatch, settings):
    # The only way to remove rendered output is the explicit --all; --chapter N scopes that
    # output wipe to one chapter so iterating on one can't wreck another.
    monkeypatch.setattr(cli, "load_settings", lambda *a, **k: settings)
    # chapter 1 + chapter 2 both have finished output + manifests
    _touch(settings.out_dir / "chapter-1.pdf")
    _touch(settings.out_dir / "chapter-1.cbz")
    _touch(settings.out_dir / "chapter-1_page_01.png")
    _touch(settings.artifacts_dir / "chapter-1.panels.json")
    _touch(settings.out_dir / "chapter-2.pdf")
    _touch(settings.out_dir / "chapter-2.cbz")
    _touch(settings.out_dir / "chapter-2_page_01.png")
    _touch(settings.artifacts_dir / "chapter-2.panels.json")

    res = runner.invoke(cli.app, ["clean", "--all", "--chapter", "1", "--yes"])
    assert res.exit_code == 0, res.output
    # chapter 1 output + manifests removed
    assert not (settings.out_dir / "chapter-1.pdf").exists()
    assert not (settings.out_dir / "chapter-1.cbz").exists()
    assert not (settings.out_dir / "chapter-1_page_01.png").exists()
    assert not (settings.artifacts_dir / "chapter-1.panels.json").exists()
    # chapter 2 RENDER untouched (output wipe was scoped to chapter 1)
    assert (settings.out_dir / "chapter-2.pdf").exists()
    assert (settings.out_dir / "chapter-2.cbz").exists()
    assert (settings.out_dir / "chapter-2_page_01.png").exists()
    assert (settings.artifacts_dir / "chapter-2.panels.json").exists()


def test_clean_no_output_flag_exists(monkeypatch, settings):
    # The footgun --output flag is gone: it must error as an unknown option.
    monkeypatch.setattr(cli, "load_settings", lambda *a, **k: settings)
    res = runner.invoke(cli.app, ["clean", "--output", "--yes"])
    assert res.exit_code != 0
    assert "No such option" in res.output or "no such option" in res.output.lower()


def test_clean_all_no_chapter_removes_all_output(monkeypatch, settings):
    monkeypatch.setattr(cli, "load_settings", lambda *a, **k: settings)
    _touch(settings.out_dir / "chapter-1.pdf")
    _touch(settings.out_dir / "chapter-2.pdf")
    _touch(settings.artifacts_dir / "chapter-1.panels.json")
    _touch(settings.artifacts_dir / "chapter-2.panels.json")

    res = runner.invoke(cli.app, ["clean", "--all", "--yes"])
    assert res.exit_code == 0, res.output
    # all chapters' rendered output + manifests removed by the full reset
    assert not (settings.out_dir / "chapter-1.pdf").exists()
    assert not (settings.out_dir / "chapter-2.pdf").exists()
    assert not (settings.artifacts_dir / "chapter-1.panels.json").exists()
    assert not (settings.artifacts_dir / "chapter-2.panels.json").exists()
    assert "ALL chapters" in res.output


def test_clean_refs_and_budget_flags(monkeypatch, settings):
    import json as _json
    monkeypatch.setattr(cli, "load_settings", lambda *a, **k: settings)
    _seed_data(settings)

    res = runner.invoke(cli.app, ["clean", "--refs", "--budget", "--yes"])
    assert res.exit_code == 0, res.output
    assert not (settings.cache_dir("refs") / "subaru.png").exists()      # refs wiped
    assert _json.loads(settings.ledger_path.read_text())["spent"] == 0.0  # ledger reset
    # output NOT requested, so rendered pages remain
    assert (settings.out_dir / "chapter-1_page_01.png").exists()


def test_clean_dry_run_deletes_nothing(monkeypatch, settings):
    monkeypatch.setattr(cli, "load_settings", lambda *a, **k: settings)
    _seed_data(settings)

    res = runner.invoke(cli.app, ["clean", "--dry-run"])
    assert res.exit_code == 0, res.output
    assert (settings.out_dir / "chapter-1_page_01.png").exists()
    assert (settings.cache_dir("panels") / "p.png").exists()
    assert "nothing deleted" in res.output


def test_fetch_refs_missing_panels_is_clean_one_liner(monkeypatch, settings):
    # fetch-refs must surface a missing parsed-panels input as a clean one-liner naming the
    # producer (`parse`), BEFORE any credentials check — not a raw FileNotFoundError traceback.
    monkeypatch.setattr(cli, "load_settings", lambda *a, **k: settings)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)   # guard must fire before the key check
    res = runner.invoke(cli.app, ["fetch-refs", "-c", "9"])   # tmp data dir -> no chapter-9 panels
    assert res.exit_code == 1, res.output
    assert "parsed panels" in res.output
    assert "parse -c 9" in res.output
    assert "Traceback" not in res.output
    assert "OPENAI_API_KEY" not in res.output                # input checked before credentials


def test_clean_prune_removes_only_orphaned_cache(monkeypatch, settings):
    # --prune GCs content-cache files not referenced by any chapter manifest, keeping in-use ones.
    import json as _json
    monkeypatch.setattr(cli, "load_settings", lambda *a, **k: settings)
    panels = settings.cache_dir("panels")
    _touch(panels / "keep.png")
    _touch(panels / "orphan.png")
    sheets = settings.cache_dir("sheets")
    _touch(sheets / "keep_sheet.png")
    _touch(sheets / "orphan_sheet.png")
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (settings.artifacts_dir / "chapter-1.panelimgs.json").write_text(
        _json.dumps([{"panel_number": 1, "path": str(panels / "keep.png")}]), encoding="utf-8")
    (settings.artifacts_dir / "chapter-1.sheets.json").write_text(
        _json.dumps({"Hero": str(sheets / "keep_sheet.png")}), encoding="utf-8")

    res = runner.invoke(cli.app, ["clean", "--prune", "--yes"])
    assert res.exit_code == 0, res.output
    assert (panels / "keep.png").exists()              # referenced -> kept
    assert (sheets / "keep_sheet.png").exists()
    assert not (panels / "orphan.png").exists()        # orphaned -> pruned
    assert not (sheets / "orphan_sheet.png").exists()


def test_clean_prune_dry_run_keeps_everything(monkeypatch, settings):
    monkeypatch.setattr(cli, "load_settings", lambda *a, **k: settings)
    panels = settings.cache_dir("panels")
    _touch(panels / "orphan.png")
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    res = runner.invoke(cli.app, ["clean", "--prune", "--dry-run"])
    assert res.exit_code == 0, res.output
    assert (panels / "orphan.png").exists()            # dry-run deletes nothing
    assert "nothing deleted" in res.output


# ── error boundary (#1/#2/#3/#4) ─────────────────────────────────────────────
def test_missing_config_is_clean_one_liner(tmp_path):
    """#1: a missing --config yields the hand-written FileNotFoundError message with no rich
    Traceback panel and no locals/Settings dump."""
    missing = tmp_path / "nope.yaml"
    res = runner.invoke(cli.app, ["config", "--config", str(missing)])
    assert res.exit_code == 1, res.output
    assert "Config file not found" in res.output
    assert "Traceback" not in res.output
    assert "Settings(" not in res.output


def test_invalid_config_mapping_is_clean(tmp_path):
    """#1: a config that isn't a mapping surfaces config.py's ValueError cleanly (no traceback)."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("just a string\n", encoding="utf-8")
    res = runner.invoke(cli.app, ["config", "--config", str(bad)])
    assert res.exit_code == 1, res.output
    assert "not a valid config mapping" in res.output
    assert "Traceback" not in res.output


def test_missing_api_key_on_live_run_is_actionable(monkeypatch, settings):
    """#2: a live (non-dry) stage with OPENAI_API_KEY unset prints the README-pointing message
    and exits cleanly — the OpenAI SDK is never reached, so no raw credentials traceback."""
    monkeypatch.setattr(cli, "_load_settings", lambda *a, **k: settings)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # input exists, so the key check (not a missing-input error) is what trips
    _touch(settings.artifacts_dir / "chapter-1.panels.json")

    res = runner.invoke(cli.app, ["charsheet", "--chapter", "1"])
    assert res.exit_code == 1, res.output
    norm = " ".join(res.output.split())
    assert "OPENAI_API_KEY is not set" in norm
    assert "--dry-run" in norm
    assert "Traceback" not in res.output


def test_missing_input_names_producer_command(monkeypatch, settings):
    """#3: a stage whose prerequisite artifact is absent points at the producing command instead
    of dumping a raw pathlib FileNotFoundError."""
    monkeypatch.setattr(cli, "_load_settings", lambda *a, **k: settings)
    res = runner.invoke(cli.app, ["parse", "--chapter", "7", "--dry-run"])
    assert res.exit_code == 1, res.output
    norm = " ".join(res.output.split())
    assert "scraped chapter 7" in norm
    assert "ln2manga scrape -c 7" in norm
    assert "Traceback" not in res.output


def test_missing_input_beats_missing_key(monkeypatch, settings):
    """#4: on a live run that is BOTH missing its input AND missing the API key, the user's real
    mistake (no input) wins — the input check runs before the client is built."""
    monkeypatch.setattr(cli, "_load_settings", lambda *a, **k: settings)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    res = runner.invoke(cli.app, ["parse", "--chapter", "7"])   # live, no --dry-run
    assert res.exit_code == 1, res.output
    assert "scraped chapter 7" in " ".join(res.output.split())
    assert "OPENAI_API_KEY" not in res.output


def test_chapter_help_has_no_hardcoded_range(monkeypatch):
    """#5: the --chapter help guides the user without baking in a stale upper bound."""
    res = runner.invoke(cli.app, ["run-all", "--help"])
    norm = " ".join(res.output.split())
    assert "scrape -c N" in norm
    assert "1-86" not in norm
    assert "Arc 6 chapter number" not in norm
