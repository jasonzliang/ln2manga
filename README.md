# ln2manga

Turn light-novel **prose** into **black-and-white, right-to-left manga pages** using **only the
OpenAI API** — the LLM parses prose into structured panels, and image generation draws
character-anchored panel art. Everything else (manga styling, page layout, speech-bubble
lettering, PDF/CBZ export) is plain local Python (Pillow + numpy).

```
scrape → parse(LLM) → script → charsheet(img) → panels(img) → mangapost → layout(RTL) → lettering → export
```

---

## 1. Install

```bash
pip install -e .
export OPENAI_API_KEY=sk-...        # required for any live (paid) run
```

Fonts ship vendored in `assets/fonts/` — nothing else to set up.

## 2. Quick start

```bash
# a) Scrape one chapter of source prose into data/raw/
ln2manga scrape --chapter 1

# b) FREE dry run — the whole pipeline against a mock client ($0). Great for
#    checking layout + lettering before you spend anything.
ln2manga run-all --chapter 1 --dry-run

# c) Real run, inside a hard spending cap (stops before exceeding it):
ln2manga run-all --chapter 1 --budget 10
```

Output lands in **`data/out/chapter-1.pdf`** and **`data/out/chapter-1.cbz`**.

If a run runs out of budget it does **not** crash — it fills the missing panels with grey
placeholders, prints a loud summary of what was skipped, and exits non-zero. Raise `--budget`
and re-run: cached panels are reused for free, so you only pay for the gaps.

## 3. Commands

Run `ln2manga --help` to see them grouped. The ones you'll use most:

| Command | What it does |
|---|---|
| `run-all` | The whole pipeline end-to-end (the main entry point). |
| `scrape` | Fetch + clean one chapter of source prose. |
| `config` | Print the fully-resolved configuration and data paths. |
| `fetch-refs` | Resolve/download character reference images ahead of time. |
| `reset-budget` | Zero the cost ledger (`data/cost_ledger.json`). |
| `clean` | Delete regenerable data (caches/scraped source) so the next run starts fresh (see §6). |

The per-stage commands — `parse`, `script`, `charsheet`, `panels`, `mangapost`, `layout`,
`letter`, `export` — let you re-run a single stage. Everything is content-addressed cached, so
re-running a stage never re-pays for work it already did. Each per-stage command reads the
previous stage's artifact from `data/artifacts/`, so run `run-all --dry-run -c N` once (or run
the stages in pipeline order) before re-running a single stage.

Common flags: `--chapter/-c N`, `--config PATH`, `--dry-run`, `--budget B`, `--max-panels N`,
`--max-paragraphs N` (truncate the work for a quick/cheap first run).

## 4. Character consistency

API-only — no LoRA, no fine-tuning. One canonical B&W **reference sheet** is generated per
character once, then passed as a reference image into **every** panel via `images.edit(image=[…])`,
so a character looks the same across the chapter. (For models that support `input_fidelity="high"`
it's sent automatically; models that reject it apply high fidelity on their own, so it's gated by
model.)

Generated sheets are also saved **by character name** under `data/out/reference-sheets/` (e.g.
`subaru.png`, `patrasche.png`) so you can see who's anchored on what. (The on-disk cache files are
content-addressed hashes for reuse; `data/artifacts/chapter-N.sheets.json` is the name→file map.)

### Anchoring on real reference images (opt-in)

By default each character is anchored on an **AI-generated** sheet. You can instead anchor on
**real images** — local files or online URLs — for stronger likeness. This is opt-in and fully
backward compatible (disabled = unchanged behavior).

1. **List the images.** Copy `config/references.example.yaml` to `config/references.yaml`
   (the default `references.external_file`), or put the same entries inline under
   `references.sources` in your config. Keys are character names (case-insensitive; aliases
   match). A value is a URL (downloaded + cached), a local path (used directly), **or a list of
   several images** that are synthesized into one consistent sheet:

   ```yaml
   # config/references.yaml
   Emilia:
     - "https://example.com/legal/emilia-front.png"   # online → downloaded
     - "config/refs/emilia-side.png"                  # local file → used directly
   ```

2. **Enable it** in your config:

   ```yaml
   references:
     enabled: true     # default: false
     mode: stylize     # raw     = use the image as-is ($0, no API call)
                       # stylize = one images.edit pass → clean B&W sheet in our style
   ```

3. **(Optional) automatic web search.** Set `references.search.enabled: true` to have ln2manga
   search the web for a character's official design and synthesize from what it finds. It's
   best-effort (many sources block bots or return promo/multi-character art that verification
   rejects); when it finds nothing usable it transparently falls back to the AI-generated sheet.
   For reliable, high-quality anchoring, provide your own images.

> ⚠️ **Supply only images you have the legal right to use.** The repo ships **no** copyrighted
> URLs — `config/references.example.yaml` contains only `example.com` placeholders, and your real
> `config/references.yaml` is git-ignored. Pre-download everything with `ln2manga fetch-refs -c 1`
> — but first run `ln2manga scrape -c 1` then `ln2manga parse -c 1` (or
> `ln2manga run-all --dry-run -c 1`), since `fetch-refs` reads the chapter's parsed panels.

## 5. Cost & safety

- **Disk cache** (`data/cache/`): identical inputs are never regenerated — re-runs are cheap.
- **USD ledger** (`data/cost_ledger.json`): a persistent spend record with a **hard cap checked
  before every paid call**, plus a backstop on the number of image calls
  (`budget.max_image_calls`). The cap is a *cumulative* total across runs.
- **`--budget B`** allows **$B of additional spend this run** (cap = already-spent + B). A live
  run prints what's already spent and what remains before generating anything, and an upfront
  estimate of the run's cost.
- **`--dry-run`** runs the entire pipeline against a mock client for **$0**.
- **`--max-panels N`** caps the panels generated this run — the cheapest way to do a tiny first
  live run, e.g. `ln2manga run-all -c 1 --budget 2 --max-panels 4`.
- **`ln2manga reset-budget`** zeroes the ledger for a clean slate.

## 6. Cleaning up / regenerating from scratch

`ln2manga clean` deletes **regenerable** data so the next run starts fresh. By design it is
**output-safe**: a plain `clean` (and `clean --cache`) clears only the content-addressed caches
(`parse/sheets/panels/manga/bubbles/html`) plus the scraped source (`data/raw`). It **keeps** the
rendered output (`data/out`: pages + PDF + CBZ), the per-chapter manifests (`data/artifacts`),
downloaded reference images, and the budget ledger — so a cache-clean can **never** wipe a finished
PDF/CBZ.

There is **no `--output` flag**. You never need to manually wipe the render to re-run: the pipeline
self-cleans its own stale per-chapter pages on every run and overwrites the rest, so re-running a
chapter always yields a correct, orphan-free output.

```bash
ln2manga clean --dry-run          # list exactly what would be deleted (deletes nothing)
ln2manga clean                    # default = --cache: regenerable caches + scraped source only
ln2manga clean --cache            # same as bare clean (keeps refs, output, manifests, ledger)
ln2manga clean --refs             # also remove downloaded reference images (costly to refetch)
ln2manga clean --budget           # reset the cost ledger to $0 (same as reset-budget)
ln2manga clean --all              # FULL reset: rendered output + ALL caches (incl. refs) + budget
ln2manga clean --all --chapter 1  # full reset scoped to chapter 1's output only (others untouched)
```

`--chapter N` scopes the **per-chapter** targets (the output removed by `--all`, and the scraped
source) to a single chapter; the shared content-addressed caches are global regardless. Add
`-y/--yes` to skip the confirmation prompt.

A genuine **from-scratch** chapter regen that keeps your reference images — clear the caches, then
re-run (the render is overwritten in place):

```bash
ln2manga clean --cache --yes
ln2manga run-all --chapter 1 --budget 12
```

## 7. Configuration

All knobs live in `config/default.yaml` (models, prices, sizes, budgets, layout, lettering,
references). Point any command at an alternate file with `--config path/to.yaml`; run
`ln2manga config` to print the fully-resolved settings. A few useful knobs:

- `concurrency.image` — number of parallel image-generation workers (default **10**).
- `budget.max_usd`, `budget.max_image_calls` — the hard spend / call caps.
- `lettering.bubble_style` — `organic` (cached AI bubble shapes + crisp text, the default) or
  `drawn` (deterministic Pillow vector bubbles, $0).
- `references.*` — see §4.

> **Local config files (why `run.yaml` / `references.yaml` aren't in the repo).** `ln2manga` reads
> `config/default.yaml` unless you pass `--config`. Your *personal* settings live in two
> **git-ignored** files — they're local-only because they may point at copyrighted reference images
> and reflect machine-specific choices, so the repo ships the **templates**, not your copies:
>
> | Git-ignored (yours) | Tracked template (shipped) | What it is |
> |---|---|---|
> | `config/run.yaml` | `config/default.yaml` | Your copy of the defaults with your overrides (e.g. `references.enabled: true`). Create with `cp config/default.yaml config/run.yaml`, edit, then `--config config/run.yaml`. *(Or just edit `default.yaml` directly — `run.yaml` is only a convention to keep your changes separate.)* |
> | `config/references.yaml` | `config/references.example.yaml` | Your character→image map (§4). |
>
> Both are intentionally not committed; the three shipped templates (`default.yaml`,
> `references.example.yaml`, `bible.example.yaml`) are everything a clone needs to start.

## 7a. Adapting to a different work (it's not Re:Zero-locked)

The pipeline is **work-agnostic** — the Re:Zero defaults are just defaults. To target a different
light novel, change config only (no code edits for the common case):

1. **Source site** — set `scrape.base_url`, `scrape.arc_path`, and `scrape.content_selector`. If
   the new site uses different markup, the cleaning rules are also config: `scrape.scene_break_chars`,
   `scrape.credit_keywords`, `scrape.chapter_text_pattern` / `chapter_link_pattern` / `nav_pattern`,
   `scrape.paragraph_tags`, `scrape.work_label` (each defaults to the current Re:Zero/WCT value).
2. **Cast + style** — point `bible.roster_file` at your own roster YAML (`{name, aliases,
   descriptor, ref_image}` per character) and optionally set `bible.global_style`. See
   **`config/bible.example.yaml`**. Omit it and the built-in Re:Zero roster is used. Characters not
   in the roster still work — the parser supplies a descriptor per discovered character.
3. **Reference art** — set `references.search.series` to the new work, and/or list official
   image URLs/paths per character in `config/references.yaml` (see §4).

With those YAML changes the same pipeline draws a different work; nothing is hardcoded to Re:Zero
in the stages.

## 8. ⚠️ Copyright / usage

The default test source is an **unofficial fan translation** of *Re:Zero* — a copyrighted work.
This project is for **personal, local, non-commercial** experimentation only:

- **Do not redistribute** scraped source text or generated manga pages.
- Scraping is rate-limited and cached to be polite to the source site.
- Anchor characters only on images you have the legal right to use (see §4).

You are responsible for using this tool in compliance with the copyright of any source material
and reference images you point it at.

## 9. License

The **source code** is released under the **MIT License** (see [`LICENSE`](LICENSE)). That license
covers this tool's code only — it grants **no** rights to any light-novel text, reference images,
or other third-party content you process with it, which remain their owners' property (see §8).
