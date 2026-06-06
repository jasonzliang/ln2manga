import numpy as np
from PIL import Image

from ln2manga.stages import layout
from ln2manga.stages.layout import (
    _fit_cover,
    choose_template,
    chunk_panels,
    rects_for_rows,
    rows_for_count,
    templates_for_count,
)


def test_fit_cover_uses_lanczos(monkeypatch):
    # Panels are continuous-tone greyscale manga art (default mangapost has no 1-bit halftone),
    # so _fit_cover must downscale with smooth Image.LANCZOS (not NEAREST, which aliases tone).
    captured = {}
    orig_resize = Image.Image.resize

    def spy_resize(self, size, resample=None, *a, **kw):
        captured["resample"] = resample
        return orig_resize(self, size, resample, *a, **kw)

    monkeypatch.setattr(Image.Image, "resize", spy_resize)
    out = _fit_cover(Image.new("L", (50, 70), 128), 300, 400)
    assert captured["resample"] == Image.LANCZOS
    assert out.size == (300, 400)


def test_fit_cover_exact_size_and_fully_covered():
    # 333x333 cover-fit into 700x1000 used to truncate the non-governing axis to
    # 999 (one px short), giving a negative crop offset and a black sliver (#15).
    src = Image.new("L", (333, 333), 200)
    out = _fit_cover(src, 700, 1000)
    assert out.size == (700, 1000)
    a = np.asarray(out)
    # No padded-in black rows/cols: every pixel is real image content (200).
    assert a.min() == 200

    # Sweep a range of source/target ratios; output must always equal (w, h)
    # and never expose a fill-padded sliver.
    for iw, ih, w, h in [(333, 333, 700, 1000), (101, 99, 200, 200),
                         (640, 480, 300, 700), (51, 50, 1000, 1000)]:
        s = Image.new("L", (iw, ih), 123)
        o = _fit_cover(s, w, h)
        assert o.size == (w, h)
        assert np.asarray(o).min() == 123


def test_fit_cover_biases_vertical_crop_toward_top():
    # A tall PORTRAIT source cover-fit into a WIDE cell scales by width and crops the
    # vertical overflow. The crop must be biased toward the TOP (heads) so that more of
    # the top survives than a centred crop would keep, sacrificing the bottom instead.
    # Mark the source with horizontal bands: top third = 250 (heads), then 150, then 50.
    iw, ih = 100, 300
    arr = np.full((ih, iw), 50, dtype=np.uint8)
    arr[: ih // 3, :] = 250          # top band (e.g. heads)
    arr[ih // 3 : 2 * ih // 3, :] = 150
    src = Image.fromarray(arr, mode="L")

    w, h = 300, 100
    out = _fit_cover(src, w, h)
    assert out.size == (w, h)

    a = np.asarray(out)
    # Full coverage: no padded/black sliver — min pixel is real content, not 0.
    assert a.min() >= 50

    # With scale = max(300/100, 100/300) = 3, the source becomes 300x900 and we crop a
    # 100-tall window from a 900-tall image (800px overflow). A CENTRED crop would take
    # top=400, landing entirely in the 150 band (top band ends at 300) -> the 250 head
    # band would be completely gone. The top-bias keeps the head band visible.
    assert (a == 250).any(), "top (head) band was cropped away — crop is not top-biased"

    # The bias offset is strictly above centre (keeps more of the top than centring).
    nh = max(h, -(-ih * max(w / iw, h / ih) // 1))  # ceil-scaled height (== 900 here)
    centred_top = (int(nh) - h) // 2
    biased_top = int((int(nh) - h) * layout._COVER_TOP_BIAS)
    assert biased_top < centred_top


def test_rtl_ordering_single_row():
    rects = rects_for_rows([2], pw=1000, ph=1000, margin=40, gutter=20)
    # cell 0 must be the RIGHTMOST one in manga reading order
    assert rects[0].x0 > rects[1].x0
    assert rects[0].x1 > rects[1].x1


def test_rtl_ordering_two_rows():
    rects = rects_for_rows([2, 2], pw=1000, ph=1200, margin=40, gutter=20)
    # row 1 is above row 2
    assert rects[0].y0 < rects[2].y0
    # within each row, index 0 is rightmost
    assert rects[0].x0 > rects[1].x0
    assert rects[2].x0 > rects[3].x0


def test_chunk_panels_no_orphan_and_bounds():
    # Small chapters fit on one page.
    assert chunk_panels(6, mx=6, mn=3) == [6]
    assert chunk_panels(3, mx=6, mn=3) == [3]
    assert chunk_panels(0) == []

    # For any n>=3 across several chapters: pages cover all panels, every page is
    # within [min, max], and there is no orphan (undersized) final page.
    for ch in range(8):
        for n in range(3, 120):
            g = chunk_panels(n, mn=3, mx=6, chapter_number=ch)
            assert sum(g) == n, (n, ch, g)
            assert all(3 <= x <= 6 for x in g), (n, ch, g)


def test_chunk_panels_terminates_when_min_equals_max():
    # Regression (#5): with mn==mx the orphan-repair loop used to merge then re-split
    # the leftover forever. It must terminate, cover every panel, and keep each page
    # within [mn, mn+mx] (a merged final page may be up to mn+mx).
    mn = mx = 4
    g = chunk_panels(10, mn=mn, mx=mx)
    assert sum(g) == 10
    assert all(mn <= x <= mn + mx for x in g), g


def test_chunk_panels_terminates_when_two_min_exceeds_max():
    # Regression (#5): 2*mn > mx (3+3 > 4) also drove the merge/re-split into an
    # infinite loop. Must terminate with full coverage and bounded page sizes.
    mn, mx = 3, 4
    g = chunk_panels(7, mn=mn, mx=mx)
    assert sum(g) == 7
    assert all(mn <= x <= mn + mx for x in g), g


def test_chunk_panels_terminates_for_all_min_le_max():
    # No mn<=mx config should hang: every page covers all panels and stays <= mn+mx.
    for mn in range(1, 7):
        for mx in range(mn, 9):
            for n in range(0, 60):
                for ch in range(3):
                    g = chunk_panels(n, mn=mn, mx=mx, chapter_number=ch)
                    assert sum(g) == n, (n, mn, mx, ch, g)
                    assert all(mn <= x <= mn + mx for x in g if n >= mn), (n, mn, mx, ch, g)


def test_chunk_panels_varies_page_size():
    # The packing must NOT be a uniform ~target on every page (problem A.4).
    g = chunk_panels(40, mn=3, mx=6, chapter_number=0)
    assert len(set(g)) > 1


def test_chunk_panels_reproducible():
    a = chunk_panels(37, mn=3, mx=6, chapter_number=2)
    b = chunk_panels(37, mn=3, mx=6, chapter_number=2)
    assert a == b


def test_rows_for_count_sums():
    # rows_for_count now returns the column counts of the canonical template.
    assert sum(rows_for_count(5)) == 5
    assert sum(rows_for_count(6)) == 6
    for n in (1, 2, 3, 4):
        assert sum(rows_for_count(n)) == n


def test_template_library_cells_match_count():
    # Every template variant must place exactly `count` cells.
    for count in range(1, 7):
        for name, rows in templates_for_count(count):
            assert sum(cols for cols, _h, _w in rows) == count, (count, name)


def test_templates_differ_across_pages():
    # Across several page indices a count must yield more than one distinct template
    # (no longer the single hardcoded "1+2+2"). Problem A.1/A.3.
    names = [choose_template(5, chapter_number=1, page_index=pi)[0] for pi in range(8)]
    assert len(set(names)) > 1
    # And it is deterministic / reproducible.
    again = [choose_template(5, chapter_number=1, page_index=pi)[0] for pi in range(8)]
    assert names == again


def test_choose_template_big_top_prefers_single_large_top_row():
    name, rows = choose_template(5, chapter_number=3, page_index=4, big_top=True)
    # A wide/establishing opener should get a top row that is one full-width cell.
    top_cols = rows[0][0]
    assert top_cols == 1, name


def test_weighted_rows_produce_unequal_heights():
    # tall-top: first row weight 1.6 should be visibly taller than the equal rows.
    rows = [(1, 1.6, None), (2, 1.0, None), (2, 1.0, None)]
    rects = rects_for_rows(rows, pw=1000, ph=1400, margin=40, gutter=20)
    top_h = rects[0].h
    mid_h = rects[1].h
    assert top_h > mid_h * 1.4


def test_weighted_columns_produce_unequal_widths_keeping_rtl():
    # Per-cell width weights: rightmost (index 0) narrower than the weighted-left cell.
    rects = rects_for_rows([(2, 1.0, [1.0, 1.6])], pw=1000, ph=800, margin=40, gutter=20)
    assert rects[1].w > rects[0].w          # left cell wider
    assert rects[0].x0 > rects[1].x0        # RTL: cell 0 still rightmost


def test_rects_tile_within_page_with_margins():
    rows = [(1, 1.6, None), (2, 1.0, None), (2, 1.0, None)]
    pw, ph, m, g = 1000, 1400, 40, 20
    rects = rects_for_rows(rows, pw, ph, m, g)
    for r in rects:
        assert r.x0 >= m - 1 and r.x1 <= pw - m + 1
        assert r.y0 >= m - 1 and r.y1 <= ph - m + 1
        assert r.w > 0 and r.h > 0


def test_layout_produces_pages(settings, png_bytes, tmp_path):
    # write a few panel pngs and run layout
    paths = []
    for i in range(1, 6):
        p = settings.cache_dir("manga") / f"panel_{i:04d}.png"
        p.write_bytes(png_bytes)
        paths.append({"panel_number": i, "path": str(p)})
    pages = layout.run(settings, paths, chapter_number=1)
    assert len(pages) == 1
    assert len(pages[0].rects) == len(pages[0].panel_numbers) == 5
    assert (settings.out_dir / "chapter-1_page_01.png").exists()

    # The chosen template NAME is a real library name (not just "1+2+2") and is
    # persisted both on the model and into pages.json.
    names = {name for name, _rows in templates_for_count(5)}
    assert pages[0].template in names
    import json
    saved = json.loads(
        (settings.artifacts_dir / "chapter-1.pages.json").read_text(encoding="utf-8")
    )
    assert saved[0]["template"] == pages[0].template


def test_layout_tolerates_missing_or_corrupt_panel(settings, png_bytes, capsys):
    # Regression: one missing/corrupt manga-cache PNG must NOT abort the whole chapter
    # with a raw traceback. The page is degraded with a placeholder, the rest render,
    # and a loud warning naming the bad panel is printed to stderr.
    paths = []
    for i in range(1, 6):
        p = settings.cache_dir("manga") / f"panel_{i:04d}.png"
        if i == 2:
            # never written -> FileNotFoundError (OSError)
            pass
        elif i == 4:
            p.write_bytes(b"not a real png")  # -> UnidentifiedImageError
        else:
            p.write_bytes(png_bytes)
        paths.append({"panel_number": i, "path": str(p)})

    pages = layout.run(settings, paths, chapter_number=1)

    # Whole chapter still renders: one page with all 5 cells, image saved.
    assert len(pages) == 1
    assert len(pages[0].rects) == len(pages[0].panel_numbers) == 5
    assert (settings.out_dir / "chapter-1_page_01.png").exists()

    # A single loud summary on stderr names both bad panels.
    err = capsys.readouterr().err
    assert "2 panel image(s) failed to load" in err
    assert "panel 2" in err
    assert "panel 4" in err
