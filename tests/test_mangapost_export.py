import zipfile

import numpy as np
import pytest
from PIL import Image

from ln2manga.stages import export
from ln2manga.stages.mangapost import mangaize


def test_mangaize_is_bw_and_same_size():
    img = Image.new("RGB", (128, 192), (180, 90, 40))  # tinted -> must become grayscale
    out = mangaize(img, tones=4, halftone=True, ink_lines=True)
    assert out.mode == "L"
    assert out.size == (128, 192)


def test_mangaize_no_black_border_ring():
    # A uniform mid-gray image has no real interior edges; the FIND_EDGES border
    # artifact (bug #7) previously forced a solid black frame on every panel.
    img = Image.new("RGB", (128, 192), (128, 128, 128))
    out = mangaize(img, tones=4, halftone=True, ink_lines=True)
    a = np.asarray(out)
    # None of the four border rows/cols may be entirely black (all-zero).
    assert a[0, :].max() > 0
    assert a[-1, :].max() > 0
    assert a[:, 0].max() > 0
    assert a[:, -1].max() > 0


def test_mangaize_keeps_interior_ink_lines():
    # A real interior edge (left half black, right half white) must still ink in.
    a = np.zeros((192, 128), dtype=np.uint8)
    a[:, 64:] = 255
    out = mangaize(Image.fromarray(a, "L").convert("RGB"),
                   tones=4, halftone=True, ink_lines=True)
    oa = np.asarray(out)
    # The vertical seam region should contain genuine black ink pixels.
    assert oa[:, 60:68].min() == 0


def _isolated_black_count(a):
    """Count black pixels (==0) whose 4-neighbours are all non-black (lone-pixel speckle/grain)."""
    black = a == 0
    interior = black[1:-1, 1:-1]
    up = ~black[:-2, 1:-1]
    down = ~black[2:, 1:-1]
    left = ~black[1:-1, :-2]
    right = ~black[1:-1, 2:]
    isolated = interior & up & down & left & right
    return int(isolated.sum())


def test_mangaize_coarse_cell_reduces_grain():
    # A uniform mid-gray panel screentones into the halftone band. With a 1px Bayer cell the
    # dither is salt-and-pepper grain (many isolated single black pixels); with a 4px cell the
    # dots are large connected blocks, so isolated-1px speckle drops sharply.
    img = Image.new("RGB", (256, 256), (128, 128, 128))
    speckly = np.asarray(mangaize(img, tones=4, halftone=True, ink_lines=False,
                                  halftone_cell=1, denoise=False))
    coarse = np.asarray(mangaize(img, tones=4, halftone=True, ink_lines=False,
                                 halftone_cell=4, denoise=False))
    fine_isolated = _isolated_black_count(speckly)
    coarse_isolated = _isolated_black_count(coarse)
    # Both must actually produce screentone (some black), but the coarse cell must be far cleaner.
    assert speckly.min() == 0
    assert coarse.min() == 0
    assert fine_isolated > 100               # the 1px baseline is genuinely speckly
    assert coarse_isolated < fine_isolated / 10


def test_export_pdf_and_cbz(settings, png_bytes):
    paths = []
    for i in range(3):
        p = settings.out_dir / f"p{i}.png"
        p.write_bytes(png_bytes)
        paths.append(str(p))
    res = export.run(settings, paths, chapter_number=1)
    assert res["pdf"].endswith(".pdf")
    pdf_bytes = open(res["pdf"], "rb").read(5)
    assert pdf_bytes[:4] == b"%PDF"
    with zipfile.ZipFile(res["cbz"]) as z:
        names = z.namelist()
        assert len([x for x in names if x.startswith("page_")]) == 3


def test_export_cbz_has_rtl_comicinfo(settings, png_bytes):
    # The CBZ must carry the right-to-left manga flag so Komga/Kavita/Tachiyomi/YACReader
    # render two-page spreads RTL (the project promises right-to-left manga).
    paths = []
    for i in range(2):
        p = settings.out_dir / f"p{i}.png"
        p.write_bytes(png_bytes)
        paths.append(str(p))
    res = export.run(settings, paths, chapter_number=1)
    with zipfile.ZipFile(res["cbz"]) as z:
        assert "ComicInfo.xml" in z.namelist()
        info = z.read("ComicInfo.xml").decode("utf-8")
    assert "<Manga>YesAndRightToLeft</Manga>" in info
    assert "<PageCount>2</PageCount>" in info


def test_export_missing_page_raises_with_context(settings, png_bytes):
    # A deleted/renamed page must fail fast with stage context and a suggested remedy,
    # before the expensive image-open work, naming the offending path.
    good = settings.out_dir / "p0.png"
    good.write_bytes(png_bytes)
    missing = settings.out_dir / "does_not_exist.png"
    with pytest.raises(FileNotFoundError) as exc:
        export.run(settings, [str(good), str(missing)], chapter_number=7)
    msg = str(exc.value)
    assert "does_not_exist.png" in msg
    assert "letter" in msg
    assert "chapter 7" in msg
