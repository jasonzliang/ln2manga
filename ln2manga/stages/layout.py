"""Stage 7 — layout: composite B&W panels into right-to-left manga pages.

Panels are packed onto pages (3..6 per page), each page picking a row TEMPLATE from a
library of variants for that panel-count. A template defines, per row, how many columns it
has plus a relative row-height weight (and optionally per-cell width weights), so pages no
longer all look like the same evenly-split grid. Cells are emitted in MANGA reading order:
top row first, right-to-left within each row. Each panel image is cover-fitted into its cell.
Template choice (and panels-per-page) is DETERMINISTIC from (chapter_number, page_index) so
caching / reproduction holds — no randomness. Returns PageLayout objects carrying the chosen
template name plus per-panel rects so the lettering stage knows where each panel lives.
"""
from __future__ import annotations

import json
import math
import sys

from PIL import Image, ImageDraw

from ..artifacts import PageLayout, Rect
from ..config import Settings


# ── Template library ─────────────────────────────────────────────────────────
# A Template is (name, rows). Each row is (cols, row_height_weight, width_weights|None).
#   cols              number of cells in the row
#   row_height_weight relative vertical size of the row (heights are normalised)
#   width_weights     optional per-cell relative widths in RTL order (index 0 = rightmost);
#                     None means equal columns. Length must equal `cols`.
# Rows are listed top-to-bottom; cells within a row are emitted right-to-left.
Row = tuple[int, float, "list[float] | None"]
Template = tuple[str, "list[Row]"]

_TEMPLATES: dict[int, list[Template]] = {
    1: [
        ("solo", [(1, 1.0, None)]),
    ],
    2: [
        ("stack", [(1, 1.0, None), (1, 1.0, None)]),
        ("tall-top", [(1, 1.6, None), (1, 1.0, None)]),
        ("tall-bottom", [(1, 1.0, None), (1, 1.6, None)]),
    ],
    3: [
        ("hero-top", [(1, 1.4, None), (2, 1.0, None)]),
        ("hero-bottom", [(2, 1.0, None), (1, 1.4, None)]),
        ("triptych", [(1, 1.0, None), (1, 1.0, None), (1, 1.0, None)]),
        ("wide-left", [(2, 1.0, [1.0, 1.6]), (1, 1.0, None)]),
    ],
    4: [
        ("grid", [(2, 1.0, None), (2, 1.0, None)]),
        ("tall-top", [(2, 1.5, None), (2, 1.0, None)]),
        ("hero-top", [(1, 1.4, None), (3, 1.0, None)]),
        ("tower", [(1, 1.0, None), (2, 1.2, None), (1, 1.0, None)]),
    ],
    5: [
        ("classic", [(1, 1.0, None), (2, 1.0, None), (2, 1.0, None)]),
        ("tall-top", [(1, 1.6, None), (2, 1.0, None), (2, 1.0, None)]),
        ("mid-single", [(2, 1.0, None), (1, 1.0, None), (2, 1.0, None)]),
        ("bottom-single", [(2, 1.0, None), (2, 1.0, None), (1, 1.0, None)]),
    ],
    6: [
        ("grid", [(2, 1.0, None), (2, 1.0, None), (2, 1.0, None)]),
        ("hero-top", [(1, 1.5, None), (2, 1.0, None), (3, 1.0, None)]),
        ("hero-bottom", [(3, 1.0, None), (2, 1.0, None), (1, 1.5, None)]),
        ("twin-tall", [(2, 1.4, None), (2, 1.0, None), (2, 1.0, None)]),
    ],
}

# Top row that is a single, full-width cell — preferred when the page opens on a
# wide / establishing shot.
_BIG_TOP = {"tall-top", "hero-top", "hero-bottom", "classic", "solo", "stack",
            "tower"}


def _fallback_template(count: int) -> Template:
    """Generic stacked-pairs template for counts outside the library."""
    rows: list[Row] = [(2, 1.0, None)] * (count // 2)
    if count % 2:
        rows = [(1, 1.0, None)] + rows
    return ("auto-%d" % count, rows)


def templates_for_count(count: int) -> list[Template]:
    return _TEMPLATES.get(count) or [_fallback_template(count)]


def chunk_panels(n: int, target: int = 5, mx: int = 6, mn: int = 3,
                 chapter_number: int = 0) -> list[int]:
    """Partition n panels into pages of size mn..mx, deterministically varied.

    Page sizes cycle through a fixed reproducible pattern (seeded by chapter_number) so
    pages are not always ~target, while still covering every panel with no orphan page
    (no final page smaller than min, unless the whole chapter is that small).
    """
    if n <= 0:
        return []
    mn = max(1, mn)
    mx = max(mn, mx)
    if n <= mx:
        return [n]

    # Deterministic cycle of sizes spanning [mn, mx]; rotated by chapter so different
    # chapters break differently, but a given chapter is always identical.
    cycle = list(range(mn, mx + 1))                 # e.g. [3,4,5,6]
    rot = chapter_number % len(cycle)
    cycle = cycle[rot:] + cycle[:rot]

    groups, rem, i = [], n, 0
    while rem > 0:
        take = min(cycle[i % len(cycle)], rem)
        groups.append(take)
        rem -= take
        i += 1

    # Repair an undersized final page by borrowing from the previous one.
    while len(groups) >= 2 and groups[-1] < mn:
        need = mn - groups[-1]
        spare = groups[-2] - mn
        move = min(need, max(0, spare))
        if move == 0:                               # previous page can't spare panels;
            last = groups.pop()                     # merge the two pages instead and stop.
            groups[-1] += last                      # do NOT re-split: re-splitting would
            break                                   # recreate an undersized page and loop
                                                    # forever when mn==mx (or 2*mn > mx).
                                                    # the merged page is <= mn+mx by design.
        groups[-2] -= move
        groups[-1] += move
    return groups


def rows_for_count(count: int) -> list[int]:
    """Column-count per row for the FIRST (canonical) template of a count.

    Kept for backwards compatibility; new code should use a Template.
    """
    _, rows = templates_for_count(count)[0]
    return [cols for cols, _h, _w in rows]


def rects_for_rows(rows, pw: int, ph: int, margin: int, gutter: int) -> list[Rect]:
    """Tile a page into cell rects from a row spec.

    `rows` accepts either:
      * a list[int] of column-counts (legacy: equal row heights, equal columns), or
      * a list[Row] of (cols, height_weight, width_weights|None).
    RTL order is preserved (cell index 0 = rightmost in each row); margins/gutters honoured.
    """
    norm: list[Row] = []
    for r in rows:
        if isinstance(r, (tuple, list)):
            cols = int(r[0])
            hw = float(r[1]) if len(r) > 1 else 1.0
            ww = r[2] if len(r) > 2 else None
        else:                                       # bare int column count
            cols, hw, ww = int(r), 1.0, None
        norm.append((cols, hw, ww))

    inner_h = ph - 2 * margin - gutter * (len(norm) - 1)
    total_hw = sum(hw for _c, hw, _w in norm) or 1.0

    rects: list[Rect] = []
    y = float(margin)
    for cols, hw, ww in norm:
        row_h = inner_h * (hw / total_hw)
        avail_w = pw - 2 * margin - gutter * (cols - 1)
        weights = list(ww) if ww else [1.0] * cols
        total_ww = sum(weights) or 1.0
        col_ws = [avail_w * (w / total_ww) for w in weights]

        # c=0 is the RIGHTMOST cell (RTL): walk leftward, consuming each cell's width.
        x_right = float(pw - margin)
        for c in range(cols):
            w = col_ws[c]
            x1 = x_right
            x0 = x1 - w
            rects.append(Rect(x0=int(x0), y0=int(y), x1=int(x1), y1=int(y + row_h)))
            x_right = x0 - gutter
        y += row_h + gutter
    return rects


def _rects_for_template(tpl: Template, pw, ph, margin, gutter) -> list[Rect]:
    return rects_for_rows(tpl[1], pw, ph, margin, gutter)


def choose_template(count: int, chapter_number: int, page_index: int,
                    big_top: bool = False) -> Template:
    """Deterministically pick a template variant for a page.

    Reproducible: indexed purely by (chapter_number, page_index, count). When `big_top`
    is set (page opens on an establishing/wide shot), prefer a variant whose top row is a
    single large panel, if any such variant exists for this count.
    """
    candidates = templates_for_count(count)
    if big_top:
        preferred = [t for t in candidates if t[0] in _BIG_TOP]
        if preferred:
            candidates = preferred
    idx = (chapter_number * 131 + page_index * 17) % len(candidates)
    return candidates[idx]


# Vertical cover-crop bias: fraction of the vertical overflow removed from the TOP.
# 0.5 = centred crop (shaves equal top+bottom); 0.0 = keep the entire top. A small
# value keeps ~the top 80% of the source, so when a tall PORTRAIT panel is cover-fit
# into a WIDE cell we crop mostly off the BOTTOM (feet/legs) and protect the TOP,
# which is exactly where characters' heads sit. Horizontal cropping stays centred
# (0.5) since left-right framing is usually symmetric.
_COVER_TOP_BIAS = 0.2


def _fit_cover(img: Image.Image, w: int, h: int) -> Image.Image:
    scale = max(w / img.width, h / img.height)
    # Round up and floor at the target so the resized image always fully covers
    # (w, h); int() truncation could leave it one pixel short, yielding a
    # negative crop offset and a black sliver on the panel edge.
    nw, nh = max(w, math.ceil(img.width * scale)), max(h, math.ceil(img.height * scale))
    # LANCZOS high-quality downscale: panels are now continuous-tone greyscale manga art (default
    # mangapost has no 1-bit halftone), so smooth interpolation is correct and avoids aliasing.
    img = img.resize((nw, nh), Image.LANCZOS)
    # Horizontal crop centred; vertical crop biased toward the top (see _COVER_TOP_BIAS).
    left = max(0, (nw - w) // 2)
    top = max(0, int((nh - h) * _COVER_TOP_BIAS))
    return img.crop((left, top, left + w, top + h))


def _load_shot_types(settings: Settings, chapter_number: int) -> dict[int, str]:
    """Best-effort map of panel_number -> shot_type from the parsed panels artifact."""
    path = settings.artifacts_dir / f"chapter-{chapter_number}.panels.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[int, str] = {}
    for d in data:
        try:
            out[int(d["panel_number"])] = str(d.get("shot_type", ""))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def run(settings: Settings, bw_manifest: list[dict], chapter_number: int) -> list[PageLayout]:
    cfg = settings.layout
    pw, ph = int(cfg["page_width"]), int(cfg["page_height"])
    margin, gutter = int(cfg["margin"]), int(cfg["gutter"])
    mn = int(cfg.get("panels_per_page_min", 3))
    mx = int(cfg.get("panels_per_page_max", 6))
    target = max(mn, mx - 1)

    ordered = sorted(bw_manifest, key=lambda d: d["panel_number"])
    sizes = chunk_panels(len(ordered), target=target, mx=mx, mn=mn,
                         chapter_number=chapter_number)

    shot_types = _load_shot_types(settings, chapter_number)
    wide_shots = {"establishing", "wide"}

    # Self-clean this chapter's stale BASE pages first so a re-run that produces FEWER pages
    # (e.g. 10 -> 8) never leaves orphan page_09/page_10 PNGs behind. Lettered pages belong to
    # the lettering stage; the PDF/CBZ are rebuilt from the manifest, so this only tidies loose
    # PNGs — but those orphans are exactly what made stale output look like current output.
    settings.out_dir.mkdir(parents=True, exist_ok=True)
    for stale in settings.out_dir.glob(f"chapter-{chapter_number}_page_*.png"):
        if not stale.name.endswith("_lettered.png"):
            try:
                stale.unlink()
            except OSError:
                pass

    pages: list[PageLayout] = []
    failed: list[tuple[int, str]] = []
    idx = 0
    for pi, size in enumerate(sizes, start=1):
        group = ordered[idx:idx + size]
        idx += size

        first_pn = group[0]["panel_number"] if group else None
        big_top = shot_types.get(first_pn, "") in wide_shots
        name, _rows = choose_template(size, chapter_number, pi, big_top=big_top)
        tpl: Template = (name, _rows)
        rects = _rects_for_template(tpl, pw, ph, margin, gutter)

        page = Image.new("L", (pw, ph), 255)
        draw = ImageDraw.Draw(page)
        for item, rect in zip(group, rects):
            try:
                panel = Image.open(item["path"]).convert("L")
                page.paste(_fit_cover(panel, rect.w, rect.h), (rect.x0, rect.y0))
            except (OSError, Image.UnidentifiedImageError) as e:
                failed.append((item["panel_number"], f"{type(e).__name__}: {item['path']}"))
                box = Image.new("L", (rect.w, rect.h), 235)
                ImageDraw.Draw(box).text((8, 8), f"panel {item['panel_number']} missing", fill=90)
                page.paste(box, (rect.x0, rect.y0))
            draw.rectangle([rect.x0, rect.y0, rect.x1, rect.y1], outline=0, width=4)

        out_path = settings.out_dir / f"chapter-{chapter_number}_page_{pi:02d}.png"
        page.save(out_path)
        pages.append(PageLayout(
            page_number=pi, template=name,
            panel_numbers=[g["panel_number"] for g in group],
            rects=rects, image_path=str(out_path),
        ))

    if failed:
        print(f"[warn] layout ch{chapter_number}: {len(failed)} panel image(s) failed to "
              f"load; pages degraded with placeholders:", file=sys.stderr)
        for pn, why in failed:
            print(f"  - panel {pn}: {why}", file=sys.stderr)

    ppath = settings.artifacts_dir / f"chapter-{chapter_number}.pages.json"
    ppath.write_text(json.dumps([p.model_dump() for p in pages], indent=2), encoding="utf-8")
    return pages
