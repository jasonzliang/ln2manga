"""ln2manga CLI — one subcommand per stage, plus `run-all`.

Each stage reads its inputs from disk artifacts and writes its outputs there, so any stage can
be re-run independently. Image/LLM results are content-addressed cached, so re-runs never re-pay.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from . import references
from .artifacts import (
    Chapter, PageLayout, PanelPrompt, PanelSpec, load_model, load_models, save_models,
)
from .cache import Cache
from .clients import build_client
from .config import load_settings
from .cost import CostTracker
from .stages import (
    charsheet, export, layout, lettering, mangapost, panels, parse, scrape, script,
)

# pretty_exceptions_show_locals=False: without it, any error (even the hand-written FileNotFound/
# ValueError/OpenAIError messages below) is buried under a rich Traceback panel that also dumps the
# entire Settings object and local config paths — noisy and a config-leak risk. The actionable
# one-liners the commands raise are far more useful on their own.
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False,
                  help="Light novel -> B&W manga pages (OpenAI API only).")
console = Console()

# --help is grouped into rich panels (rendered in first-seen order) so the one-shot entry points
# users actually reach for surface first, with per-stage re-runs and housekeeping below.
PANEL_PRIMARY = "Primary"
PANEL_ADVANCED = "Advanced (per-stage re-run)"
PANEL_UTILITIES = "Utilities"


def _load_settings(config: Optional[str]):
    """load_settings, but surface its hand-written FileNotFoundError/ValueError (missing or
    malformed config) as a clean one-line error instead of a rich traceback panel. The CLI
    entrypoint is the Typer app itself (no main() wrapper to catch these), so the boundary lives
    here, at the only place every command loads settings."""
    try:
        return load_settings(config)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


def _require(path, *, what: str, producer: str) -> Path:
    """Fail with a clean, actionable message (not a raw FileNotFoundError traceback) when a stage's
    input artifact is missing, naming the command that produces it. Called BEFORE the live client is
    built so a missing-input mistake surfaces ahead of any credentials error."""
    p = Path(path)
    if not p.exists():
        console.print(f"[red]Error:[/] No {what} ({p}). Run: {producer}")
        raise typer.Exit(1)
    return p


def _ctx(config: Optional[str], dry_run: bool, budget: Optional[float]):
    settings = _load_settings(config)
    references.load_references_config(settings, config)  # attach settings.references
    if not dry_run and not os.environ.get("OPENAI_API_KEY"):
        console.print("OPENAI_API_KEY is not set. Export it (see README section 1) "
                      "or use --dry-run for a free $0 run.")
        raise typer.Exit(1)
    try:
        client = build_client(dry_run)
    except Exception as e:                  # noqa: BLE001 — covers OpenAIError without importing the SDK
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)
    # Construct the tracker normally so it loads the persisted (cumulative) ledger first.
    tracker = CostTracker(settings.budget["max_usd"], settings.ledger_path,
                          settings.prices_usd,
                          int(settings.budget.get("max_image_calls", 10_000)),
                          dry_run=dry_run)
    if budget is not None:
        # Bug #2: --budget B means "$B of ADDITIONAL spend THIS run". The cap is the already
        # persisted spend plus B (dry-run starts clean at spent=0, so this is just B).
        tracker.max_usd = tracker.spent + float(budget)
        settings.budget["max_usd"] = tracker.max_usd
    cache = Cache(settings.cache_dir)
    return settings, client, tracker, cache


def _arts(settings, n):
    a = settings.artifacts_dir
    return {
        "chapter": settings.raw_dir / f"chapter-{n}.json",
        "panels": a / f"chapter-{n}.panels.json",
        "prompts": a / f"chapter-{n}.prompts.json",
        "sheets": a / f"chapter-{n}.sheets.json",
        "panelimgs": a / f"chapter-{n}.panelimgs.json",
        "mangaimgs": a / f"chapter-{n}.mangaimgs.json",
        "pages": a / f"chapter-{n}.pages.json",
        "lettered": a / f"chapter-{n}.lettered.json",
    }


def _flat_image_price(settings, quality: str) -> float:
    """Conservative flat per-image USD estimate, read straight from config prices (NOT via the
    dry-run tracker, whose estimate_image returns 0). Falls back through model -> default -> medium
    so a missing model/quality entry never raises."""
    model = settings.models.get("image", "default")
    table = settings.prices_usd.get("image", {})
    row = table.get(model) or table.get("default", {})
    return float(row.get(quality, row.get("medium", 0.0)))


def _estimate_usd(settings, n_sheets: int, n_panels: int) -> float:
    """Upfront cost estimate for a live run: one sheet per character + one image per panel."""
    sheet_price = _flat_image_price(settings, settings.image.get("sheet_quality", "high"))
    panel_price = _flat_image_price(settings, settings.image.get("panel_quality", "medium"))
    return n_sheets * sheet_price + n_panels * panel_price


# ── options ──────────────────────────────────────────────────────────────────
Chap = typer.Option(1, "--chapter", "-c",
                    help="Source chapter number (start with `ln2manga scrape -c N`; "
                         "an out-of-range value lists the nearest valid chapters).")
Cfg = typer.Option(None, "--config", help="Path to a config YAML (defaults to config/default.yaml).")
Dry = typer.Option(False, "--dry-run", help="Use the mock client; spend $0.")
Bud = typer.Option(None, "--budget",
                   help="Allow up to $B of ADDITIONAL spend this run (cap = already-spent + B).")
MaxPara = typer.Option(0, "--max-paragraphs", help="Truncate source paragraphs (0 = whole chapter).")
MaxPanels = typer.Option(0, "--max-panels", help="Cap rendered panels (0 = all) — direct cost lever.")


@app.command("scrape", rich_help_panel=PANEL_PRIMARY)
def scrape_cmd(chapter: int = Chap, config: Optional[str] = Cfg):
    """Fetch + clean a chapter."""
    settings, *_ = _ctx(config, True, None)
    ch = scrape.run(settings, chapter)
    console.print(f"[green]scraped[/] ch{chapter}: {len(ch.paragraphs)} paragraphs, "
                  f"{len(ch.scene_breaks)} scene breaks -> {_arts(settings, chapter)['chapter']}")


@app.command("parse", rich_help_panel=PANEL_ADVANCED)
def parse_cmd(chapter: int = Chap, config: Optional[str] = Cfg,
              dry_run: bool = Dry, budget: Optional[float] = Bud):
    """Prose -> structured panels (LLM)."""
    settings = _load_settings(config)
    src = _require(_arts(settings, chapter)["chapter"],
                   what=f"scraped chapter {chapter}", producer=f"ln2manga scrape -c {chapter}")
    settings, client, tracker, cache = _ctx(config, dry_run, budget)
    ch = load_model(Chapter, src)
    specs = parse.run(client, settings, tracker, cache, ch)
    console.print(f"[green]parsed[/] {len(specs)} panels  (spent ${tracker.spent:.3f})")


@app.command("script", rich_help_panel=PANEL_ADVANCED)
def script_cmd(chapter: int = Chap, config: Optional[str] = Cfg):
    """Panels -> image prompts (pure)."""
    settings = _load_settings(config)
    src = _require(_arts(settings, chapter)["panels"],
                   what=f"parsed panels for chapter {chapter}",
                   producer=f"ln2manga parse -c {chapter}")
    settings, *_ = _ctx(config, True, None)
    specs = load_models(PanelSpec, src)
    prompts = script.run(settings, specs, chapter)
    console.print(f"[green]scripted[/] {len(prompts)} prompts")


@app.command("charsheet", rich_help_panel=PANEL_ADVANCED)
def charsheet_cmd(chapter: int = Chap, config: Optional[str] = Cfg,
                  dry_run: bool = Dry, budget: Optional[float] = Bud):
    """Generate one reference sheet per character (image)."""
    settings = _load_settings(config)
    src = _require(_arts(settings, chapter)["panels"],
                   what=f"parsed panels for chapter {chapter}",
                   producer=f"ln2manga parse -c {chapter}")
    settings, client, tracker, cache = _ctx(config, dry_run, budget)
    specs = load_models(PanelSpec, src)
    sheets = charsheet.run(client, settings, tracker, cache, specs, chapter)
    console.print(f"[green]charsheets[/] {list(sheets)}  (spent ${tracker.spent:.3f})")


@app.command("panels", rich_help_panel=PANEL_ADVANCED)
def panels_cmd(chapter: int = Chap, config: Optional[str] = Cfg,
               dry_run: bool = Dry, budget: Optional[float] = Bud):
    """Generate reference-anchored panel art (image)."""
    settings = _load_settings(config)
    prompts_src = _require(_arts(settings, chapter)["prompts"],
                           what=f"image prompts for chapter {chapter}",
                           producer=f"ln2manga script -c {chapter}")
    sheets_src = _require(_arts(settings, chapter)["sheets"],
                          what=f"character sheets for chapter {chapter}",
                          producer=f"ln2manga charsheet -c {chapter}")
    settings, client, tracker, cache = _ctx(config, dry_run, budget)
    prompts = load_models(PanelPrompt, prompts_src)
    sheets = json.loads(Path(sheets_src).read_text())
    man = panels.run(client, settings, tracker, cache, prompts, sheets, chapter)
    console.print(f"[green]panels[/] {len(man)} images  (spent ${tracker.spent:.3f})")


@app.command("mangapost", rich_help_panel=PANEL_ADVANCED)
def mangapost_cmd(chapter: int = Chap, config: Optional[str] = Cfg):
    """Enforce B&W manga look (pure)."""
    settings = _load_settings(config)
    src = _require(_arts(settings, chapter)["panelimgs"],
                   what=f"panel images for chapter {chapter}",
                   producer=f"ln2manga panels -c {chapter}")
    settings, *_ = _ctx(config, True, None)
    man = json.loads(Path(src).read_text())
    out = mangapost.run(settings, man, chapter)
    console.print(f"[green]mangapost[/] {len(out)} panels")


@app.command("layout", rich_help_panel=PANEL_ADVANCED)
def layout_cmd(chapter: int = Chap, config: Optional[str] = Cfg):
    """Composite panels into RTL pages (pure)."""
    settings = _load_settings(config)
    src = _require(_arts(settings, chapter)["mangaimgs"],
                   what=f"B&W manga panels for chapter {chapter}",
                   producer=f"ln2manga mangapost -c {chapter}")
    settings, *_ = _ctx(config, True, None)
    man = json.loads(Path(src).read_text())
    pages = layout.run(settings, man, chapter)
    console.print(f"[green]layout[/] {len(pages)} pages")


@app.command("letter", rich_help_panel=PANEL_ADVANCED)
def letter_cmd(chapter: int = Chap, config: Optional[str] = Cfg,
               dry_run: bool = Dry, budget: Optional[float] = Bud):
    """Draw bubbles + dialogue (Pillow; or organic API bubble shapes if lettering.bubble_style=organic)."""
    settings = _load_settings(config)
    pages_src = _require(_arts(settings, chapter)["pages"],
                         what=f"page layouts for chapter {chapter}",
                         producer=f"ln2manga layout -c {chapter}")
    panels_src = _require(_arts(settings, chapter)["panels"],
                          what=f"parsed panels for chapter {chapter}",
                          producer=f"ln2manga parse -c {chapter}")
    settings, client, tracker, cache = _ctx(config, dry_run, budget)
    pages = load_models(PageLayout, pages_src)
    specs = load_models(PanelSpec, panels_src)
    out = lettering.run(settings, pages, specs, chapter, client=client, tracker=tracker, cache=cache)
    console.print(f"[green]lettered[/] {len(out)} pages  (spent ${tracker.spent:.3f})")


@app.command("export", rich_help_panel=PANEL_ADVANCED)
def export_cmd(chapter: int = Chap, config: Optional[str] = Cfg):
    """Bundle pages into PDF + CBZ (pure)."""
    settings = _load_settings(config)
    src = _require(_arts(settings, chapter)["lettered"],
                   what=f"lettered pages for chapter {chapter}",
                   producer=f"ln2manga letter -c {chapter}")
    settings, *_ = _ctx(config, True, None)
    pages = json.loads(Path(src).read_text())
    res = export.run(settings, pages, chapter)
    console.print(f"[green]exported[/] {res['pdf']}  +  {res['cbz']}")


@app.command("reset-budget", rich_help_panel=PANEL_UTILITIES)
def reset_budget(config: Optional[str] = Cfg,
                 yes: bool = typer.Option(False, "--yes", "-y",
                                          help="Skip the confirmation prompt.")):
    """Zero the persisted cost ledger (data/cost_ledger.json)."""
    settings = _load_settings(config)
    ledger = settings.ledger_path
    if not yes:
        prior = "0.000"
        if ledger.exists():
            try:
                prior = f"{float(json.loads(ledger.read_text()).get('spent', 0.0)):.3f}"
            except Exception:
                pass
        typer.confirm(f"Reset {ledger} (currently spent ${prior}) to spent=$0?", abort=True)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps({"spent": 0.0, "image_calls": 0,
                                  "max_usd": settings.budget["max_usd"], "events": []},
                                 indent=2), encoding="utf-8")
    console.print(f"[green]reset-budget[/] {ledger} -> spent $0.000, image_calls 0")


@app.command("clean", rich_help_panel=PANEL_UTILITIES)
def clean_cmd(
    config: Optional[str] = Cfg,
    chapter: Optional[int] = typer.Option(None, "--chapter", "-c",
        help="Scope the per-chapter targets (the rendered output removed by --all, and the "
             "scraped source) to ONLY this chapter's files — never other chapters'. Has no "
             "effect on the content-addressed caches, which are shared across chapters."),
    cache: bool = typer.Option(False, "--cache",
        help="Generated caches (parse/sheets/panels/manga/bubbles/html) + scraped source "
             "(data/raw). KEEPS downloaded reference images and the final PDF/CBZ. These "
             "caches are content-addressed and shared across chapters, so --cache is global "
             "even with --chapter."),
    refs: bool = typer.Option(False, "--refs",
        help="Downloaded reference images (data/cache/refs). Off unless asked — costly to refetch."),
    prune: bool = typer.Option(False, "--prune",
        help="Garbage-collect ONLY orphaned content-cache entries: panels/manga/sheets cache files "
             "no longer referenced by ANY chapter manifest (e.g. superseded by a re-anchor or an "
             "earlier run). Reclaims space without a full --cache wipe; everything still in use is "
             "kept so re-runs stay free."),
    budget: bool = typer.Option(False, "--budget",
        help="Reset the cost ledger to $0 (same as reset-budget)."),
    all_: bool = typer.Option(False, "--all",
        help="Full reset: the rendered output (PDF/CBZ/pages) + ALL caches (incl. refs) + "
             "budget. Scope the output wipe to one chapter with --chapter N."),
    dry_run: bool = typer.Option(False, "--dry-run",
        help="List what would be deleted; delete nothing."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
):
    """Delete regenerable data so the next run starts fresh.

    With NO flags this clears ONLY the regenerable caches + scraped source (the --cache target
    set) and KEEPS the rendered output (data/out: pages + PDF + CBZ), the per-chapter manifests
    (data/artifacts), downloaded reference images, and the budget ledger. So a bare `clean` (and
    `clean --cache`) can NEVER remove a finished PDF/CBZ.

    There is deliberately no standalone --output flag: the pipeline self-cleans its own stale
    per-chapter pages on every run and overwrites the rest, so the finished render is managed for
    you and never needs a manual wipe. A full reset that DOES delete the rendered output is the
    explicit --all; pair it with --chapter N to wipe only that chapter's render (so iterating on
    one chapter can't wreck another). Every target is resolved under data/ before deletion.
    """
    settings = _load_settings(config)
    data = settings.data_dir.resolve()
    cache_root = data / "cache"

    if prune:
        # GC orphaned content-cache entries: panels/manga/sheets cache files not referenced by
        # ANY chapter manifest (superseded by a re-anchor or an earlier run). Keeps in-use entries
        # so re-runs stay free. Content-addressed caches are append-only, so this reclaims the
        # accumulation without the blunt full --cache wipe.
        referenced: set[str] = set()
        for f in settings.artifacts_dir.glob("chapter-*.json"):
            try:
                data_j = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            items = data_j.values() if isinstance(data_j, dict) else (data_j or [])
            for it in items:
                p = it.get("path") if isinstance(it, dict) else (it if isinstance(it, str) else None)
                if p:
                    referenced.add(os.path.realpath(p))
        orphans: list[Path] = []
        for stage in ("panels", "manga", "sheets"):
            d = cache_root / stage
            if d.exists():
                orphans += [p for p in d.glob("*.png") if os.path.realpath(p) not in referenced]
        if not orphans:
            console.print("[green]prune[/] no orphaned cache entries — nothing to reclaim.")
            return
        total = sum(p.stat().st_size for p in orphans)
        console.print(f"[bold]Prune {len(orphans)} orphaned cache file(s)[/] (~{total/1e6:.1f} MB) "
                      f"not referenced by any chapter manifest:")
        for p in orphans[:8]:
            console.print(f"  • [dim]{p}[/]")
        if len(orphans) > 8:
            console.print(f"  • … and {len(orphans) - 8} more")
        if dry_run:
            console.print("[cyan]--dry-run: nothing deleted[/]")
            return
        if not yes:
            typer.confirm("Delete these orphaned cache entries?", abort=True)
        for p in orphans:
            rp = p.resolve()
            if data != rp and data not in rp.parents:   # safety: never touch anything outside data/
                continue
            rp.unlink(missing_ok=True)
            sidecar = rp.with_suffix(rp.suffix + ".json")     # drop the meta sidecar too
            if sidecar.exists():
                sidecar.unlink()
        console.print(f"[green]prune[/] reclaimed ~{total/1e6:.1f} MB "
                      f"({len(orphans)} orphaned cache file(s) removed)")
        return

    # --output is intentionally NOT a standalone flag: the render is self-managed (overwrite +
    # per-chapter orphan cleanup in the layout/lettering stages), so deleting it is only ever the
    # explicit full-reset --all (scope it to one chapter with --chapter).
    output = all_
    if all_:
        cache = refs = budget = True
    elif not (cache or refs or budget):
        cache = True                     # safe default: regenerable caches only, keep final output

    targets: list[tuple[str, Path]] = []
    if output:
        if chapter is None:
            console.print(f"[yellow]warning:[/] --all without --chapter removes ALL chapters' "
                          f"rendered output under {settings.out_dir} + {settings.artifacts_dir}")
            targets.append(("rendered output (data/out)", settings.out_dir))
            targets.append(("stage manifests (data/artifacts)", settings.artifacts_dir))
        else:
            # scope to this chapter's files only — never touch other chapters' output
            for p in sorted(settings.out_dir.glob(f"chapter-{chapter}*")):
                targets.append((f"rendered output (data/out/{p.name})", p))
            for p in sorted(settings.artifacts_dir.glob(f"chapter-{chapter}.*")):
                targets.append((f"stage manifest (data/artifacts/{p.name})", p))
    if cache and cache_root.exists():
        for sub in sorted(p for p in cache_root.iterdir() if p.is_dir() and p.name != "refs"):
            targets.append((f"cache/{sub.name}", sub))
        # scraped source IS per-chapter, so honor --chapter when given
        if chapter is None:
            targets.append(("scraped source (data/raw)", settings.raw_dir))
        else:
            targets.append((f"scraped source (data/raw/chapter-{chapter}.json)",
                            settings.raw_dir / f"chapter-{chapter}.json"))
    if refs:
        targets.append(("reference images (data/cache/refs)", cache_root / "refs"))

    existing = [(lbl, p) for lbl, p in targets if p.exists()]
    if not existing and not budget:
        console.print("[yellow]nothing to clean[/] (already empty)")
        return

    if existing:
        console.print("[bold]Will delete:[/]")
        for lbl, p in existing:
            n = sum(1 for _ in p.rglob("*")) if p.is_dir() else 1
            console.print(f"  • {lbl}  [dim]{p}[/]  ({n} item(s))")
    if budget:
        console.print(f"  • reset cost ledger  [dim]{settings.ledger_path}[/]")

    if dry_run:
        console.print("[cyan]--dry-run: nothing deleted[/]")
        return
    if not yes:
        typer.confirm("Delete the above?", abort=True)

    for lbl, p in existing:
        rp = p.resolve()
        if data != rp and data not in rp.parents:   # safety: never touch anything outside data/
            console.print(f"  [red]skip[/] {lbl}: resolved outside the data dir")
            continue
        if rp.is_dir():
            shutil.rmtree(rp)
            rp.mkdir(parents=True, exist_ok=True)    # keep the (now-empty) dir present
        else:
            rp.unlink()
    if budget:
        ledger = settings.ledger_path
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(json.dumps({"spent": 0.0, "image_calls": 0,
                                      "max_usd": settings.budget["max_usd"], "events": []},
                                     indent=2), encoding="utf-8")
    console.print(f"[green]clean[/] done — {len(existing)} target(s) cleared"
                  + (", ledger reset" if budget else "")
                  + ("" if refs else "; reference images kept"))


@app.command("fetch-refs", rich_help_panel=PANEL_UTILITIES)
def fetch_refs_cmd(chapter: int = Chap, config: Optional[str] = Cfg,
                   dry_run: bool = Dry, budget: Optional[float] = Bud):
    """Resolve reference images (explicit sources AND online search) for a chapter's characters."""
    settings = _load_settings(config)
    # Validate the input BEFORE building the client / loading the ledger, so a missing-panels
    # mistake surfaces as a clean one-liner (naming the producer) instead of a raw traceback or
    # a credentials error first — matching every other stage command.
    src = _require(_arts(settings, chapter)["panels"],
                   what=f"parsed panels for chapter {chapter}",
                   producer=f"ln2manga parse -c {chapter}")
    settings, client, tracker, cache = _ctx(config, dry_run, budget)
    cfg = references.references_config(settings)
    search_on = bool(cfg.get("search", {}).get("enabled"))
    if not cfg.get("enabled"):
        console.print("[yellow]references.enabled is false[/] — resolving anyway for inspection.")
    specs = load_models(PanelSpec, src)
    fetched, missing = [], []
    for name in charsheet.characters_in(specs):
        explicit = references.reference_source(name, settings)
        if not explicit and not search_on:
            continue
        path = references.resolve_reference(name, settings, client, tracker)
        origin = explicit if explicit else "online-search"
        (fetched if path is not None else missing).append((name, origin, str(path)))
    for name, origin, path in fetched:
        console.print(f"  [green]ok[/]   {name}: {origin} -> {path}")
    for name, origin, _ in missing:
        console.print(f"  [red]fail[/] {name}: {origin} (see stderr; will fall back to AI sheet)")
    if not fetched and not missing:
        console.print("[yellow]no references configured/searchable for this chapter's cast.[/]")
    console.print(f"[green]fetch-refs[/] resolved {len(fetched)}/{len(fetched) + len(missing)} "
                  f"references for ch{chapter}  (spent ${tracker.spent:.3f})")


@app.command("config", rich_help_panel=PANEL_UTILITIES)
def config_cmd(config: Optional[str] = Cfg):
    """Print the fully-resolved master configuration and data paths."""
    from .config import DEFAULT_CONFIG
    settings = _load_settings(config)
    data = settings.model_dump()
    data["references"] = references.references_config(settings)   # effective (defaults + search)
    console.print(f"[bold]config file :[/] {config or DEFAULT_CONFIG}")
    console.print_json(json.dumps(data, default=str))
    console.print(f"[bold]data dir    :[/] {settings.data_dir}")
    console.print(f"[bold]ledger      :[/] {settings.ledger_path}")
    console.print(f"[bold]fonts dir   :[/] {settings.fonts_dir}")


@app.command("run-all", rich_help_panel=PANEL_PRIMARY)
def run_all(chapter: int = Chap, config: Optional[str] = Cfg,
            dry_run: bool = Dry, budget: Optional[float] = Bud,
            max_paragraphs: int = MaxPara, max_panels: int = MaxPanels):
    """Run the entire pipeline end-to-end."""
    settings, client, tracker, cache = _ctx(config, dry_run, budget)
    if max_paragraphs:
        settings.scrape["max_paragraphs"] = max_paragraphs
    cap = settings.budget["max_usd"]
    mode = "[yellow]DRY-RUN ($0)[/]" if dry_run else f"[red]LIVE (cap ${cap:.2f})[/]"
    console.rule(f"ln2manga run-all  ch{chapter}  {mode}")
    if not dry_run:
        # Bug #2: the bare config cap is a CUMULATIVE ledger total, not a per-run allowance, so
        # surface what is already spent / still available before the user pays anything.
        console.print(f"  budget   : ${tracker.spent:.2f} already spent, "
                      f"${tracker.remaining():.2f} of ${tracker.max_usd:.2f} remaining "
                      f"(reset-budget for a clean slate)")

    ch = scrape.run(settings, chapter)
    console.print(f"  scrape   : {len(ch.paragraphs)} paragraphs")
    specs = parse.run(client, settings, tracker, cache, ch)
    if max_panels and len(specs) > max_panels:
        specs = specs[:max_panels]
        # Bug #4: parse.run wrote the FULL panel set to panels.json. Re-persist the truncated
        # set to the canonical artifact so every downstream stage and per-stage re-run agrees
        # (otherwise `letter`/`charsheet`/`script` would read the full set and defeat the cap).
        save_models(specs, _arts(settings, chapter)["panels"])
        console.print(f"  [yellow]capped to {max_panels} panels[/]")
    console.print(f"  parse    : {len(specs)} panels")
    prompts = script.run(settings, specs, chapter)
    console.print(f"  script   : {len(prompts)} prompts")
    # Upfront cost estimate (over-estimate, from flat config prices) before any image is generated.
    cast = charsheet.characters_in(specs)
    est = _estimate_usd(settings, len(cast), len(prompts))
    console.print(f"  estimate : ~${est:.2f} for {len(cast)} sheets + {len(prompts)} panels "
                  f"(cap ${cap:.2f})")
    if not dry_run and est > tracker.remaining():
        console.print("  [yellow]heads-up: estimate exceeds remaining budget; "
                      "expect grey placeholders. Raise --budget or run reset-budget.[/]")
    sheets = charsheet.run(client, settings, tracker, cache, specs, chapter)
    console.print(f"  charsheet: {len(sheets)} sheets  (${tracker.spent:.3f})")
    pimgs = panels.run(client, settings, tracker, cache, prompts, sheets, chapter)
    console.print(f"  panels   : {len(pimgs)} images  (${tracker.spent:.3f})")
    bw = mangapost.run(settings, pimgs, chapter)
    console.print(f"  mangapost: {len(bw)} panels")
    pages = layout.run(settings, bw, chapter)
    console.print(f"  layout   : {len(pages)} pages")
    lettered = lettering.run(settings, pages, specs, chapter,
                             client=client, tracker=tracker, cache=cache)
    console.print(f"  letter   : {len(lettered)} pages")
    res = export.run(settings, lettered, chapter)

    console.rule("[bold green]done")
    console.print(f"  total spent: [bold]${tracker.spent:.3f}[/] "
                  f"({tracker.image_calls} image calls)")
    console.print(f"  PDF : {res['pdf']}")
    console.print(f"  CBZ : {res['cbz']}")

    # Bug #1 + #4: a run that placeholdered most panels or dropped reference sheets currently
    # prints an identical cheerful 'done'. Compute the degraded-output summary from data already
    # in hand (no stage-return-signature change) and exit non-zero so `&&` chains / CI notice.
    placeholders = [d for d in pimgs if str(d.get("path", "")).endswith("_placeholder.png")]
    missing_sheets = [n for n in cast if n not in sheets]
    if placeholders or missing_sheets:
        msg = (f"WARNING: {len(placeholders)}/{len(pimgs)} panels are grey placeholders; "
               f"{len(missing_sheets)} character(s) have no reference sheet")
        if missing_sheets:
            msg += f" ({', '.join(missing_sheets)})"
        msg += (". Raise budget.max_usd / --budget or budget.max_image_calls and re-run "
                "(cached panels are reused for free).")
        console.print(f"[bold red]{msg}[/]")
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
