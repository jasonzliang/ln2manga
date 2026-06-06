# ln2manga — Design & Decisions

Light-novel prose → black & white, right-to-left manga pages, using **only the OpenAI API**
for the LLM and image generation. Code-rendered lettering. Reference-sheet character anchoring.

This document records the design and the facts that were **verified against the live API / SDK**
(not just assumed from documentation), since several specifics were beyond the model's training
cutoff.

## Verified facts (live `OPENAI_API_KEY`, openai SDK 2.37.0, June 2026)

| Claim | How verified | Result |
|---|---|---|
| Model IDs exist | `client.models.list()` | `gpt-5.4`, `gpt-5.4-mini`, `gpt-image-2`, `gpt-image-1.5`, `gpt-image-1`, `gpt-image-1-mini` all present |
| `images.edit` takes multiple refs | SDK signature + live call | `image: Union[FileTypes, SequenceNotStr[FileTypes]]` — a list of refs works |
| `input_fidelity` support | live `images.edit` calls | `gpt-image-2` → **400 `invalid_input_fidelity_model`**; `gpt-image-1.5` accepts `"high"` |
| `images.generate` cannot take refs | SDK signature | no `image`/`input_fidelity` params → **panels must use `images.edit`** |
| `images.edit` has no `moderation` | SDK signature | only `images.generate` has `moderation` (and `style`) |
| Image responses are base64 | live calls | always `data[0].b64_json`, never a URL |
| Structured Outputs | live `responses.parse` | `gpt-5.4-mini` + Pydantic `text_format` → typed `.output_parsed` + `.usage` |
| Sizes | SDK signature | portrait `1024x1536` available |

**Consequence in code:** panels always go through `images.edit(image=[sheets...])`; `input_fidelity="high"`
is added only when the model is not `gpt-image-2*` (see `imagegen._supports_input_fidelity`).

## Pipeline

```
scrape → parse(LLM) → script → charsheet(img) → panels(img) → mangapost → layout(RTL) → lettering → export
```

| Stage | Module | API? | Responsibility |
|---|---|---|---|
| 1 scrape | `stages/scrape.py` | no | fetch + clean a chapter (drop credits between the first two `※` rows, drop `Translation Notes:` footnotes, record scene breaks) |
| 2 parse | `stages/parse.py` | `gpt-5.4-mini` | prose → ordered `PanelSpec`s via `responses.parse` (Structured Outputs), chunked on scene/size, roster injected |
| 3 script | `stages/script.py` | no | `PanelSpec` → natural-language image prompt; injects verbatim character descriptors + style |
| 4 charsheet | `stages/charsheet.py` | `gpt-image-2` | one canonical B&W reference sheet per character, once, cached |
| 5 panels | `stages/panels.py` | `gpt-image-2` | `images.edit` anchored on the panel's character sheet(s) |
| 6 mangapost | `stages/mangapost.py` | no | enforce B&W look: grayscale → autocontrast → posterize → Bayer halftone → ink edges |
| 7 layout | `stages/layout.py` | no | pack panels onto RTL page templates, return per-panel rects |
| 8 lettering | `stages/lettering.py` | no | speech/thought/shout/narration bubbles + wrapped dialogue, tail toward speaker |
| 9 export | `stages/export.py` | no | PDF + CBZ |

## Character consistency (the #1 risk, API-only, no LoRA)

1. Freeze a verbatim per-character descriptor in `bible.py` (never paraphrase).
2. Generate ONE reference sheet per character (`images.generate`, high quality), cache the bytes.
3. Every panel = `images.edit(image=[sheet(s)], prompt + "preserve identity, change only pose, no text")`,
   identity-critical character first. `gpt-image-2` auto-applies high fidelity; other models get
   `input_fidelity="high"`.
4. Stateless: the same canonical sheet bytes are re-pinned every call (never chain edits → avoids drift).
5. >5 characters in one panel → tile sheets into a single composite reference (`bible.composite_sheets`).

## Cost & safety

- **Content-addressed cache** (`cache.py`): image/LLM outputs keyed by inputs (incl. ref-image bytes);
  re-runs never re-pay.
- **Persistent budget ledger** (`cost.py`, `data/cost_ledger.json`): a conservative flat per-image
  estimate is checked *before* every paid call (over-estimates, so the cap trips early, never late);
  text cost is exact from `usage`. Hard caps: `budget.max_usd` and `budget.max_image_calls`.
- **`--dry-run`**: a `MockClient` runs the whole pipeline (incl. layout/lettering/export) for **$0**.
- **Resilience**: a per-panel moderation refusal becomes a placeholder (run continues); hitting the
  budget mid-run stops generation gracefully and still assembles what exists.
- **Retries**: `tenacity` backoff on 429/5xx; never on client errors.

## Tunable levers (`config/default.yaml` + CLI)

- `models.image` — swap `gpt-image-2` ↔ `gpt-image-1.5` (the only code effect is `input_fidelity` gating).
- `image.panel_quality` — `medium` (default) vs `high`.
- `--max-paragraphs` / `--max-panels` — bound source length and rendered panel count (cost levers).
- `--budget` — override the USD cap for a run.

## Known limitations

- **B&W fidelity**: `gpt-image-2` can return soft/tinted art; `mangapost` enforces the inked look
  deterministically, but it is a post-filter, not native monochrome.
- **Multi-character panels**: identity bleed is possible; mitigated by ordered + labeled sheets.
- **Bubble placement** is heuristic (top-down, speaker-side tail); it avoids stacking but is not
  layout-aware of the art content.
- **Per-image $**: exact `gpt-image-2` token pricing is not published here; estimates are deliberately
  conservative and `max_image_calls` is the hard backstop.
- **Copyright**: the test source is a fan translation of a copyrighted work — personal/local use only.
