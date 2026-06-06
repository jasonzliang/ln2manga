"""Stage 9 — export: bundle lettered pages into a PDF and a CBZ (comic archive)."""
from __future__ import annotations

import zipfile
from pathlib import Path

from PIL import Image

from ..config import Settings


def to_pdf(page_paths: list[str], out_path: str | Path) -> None:
    # NOTE: PDF carries no reading-direction metadata — Pillow's PDF writer exposes no
    # ViewerPreferences/Direction hook. The right-to-left manga intent is signalled in the
    # CBZ via ComicInfo.xml (see to_cbz); the PDF page sequence is identical for LTR/RTL.
    imgs = [Image.open(p).convert("L") for p in page_paths]
    if not imgs:
        raise ValueError("no pages to export")
    imgs[0].save(out_path, "PDF", save_all=True, append_images=imgs[1:], resolution=150.0)


def to_cbz(page_paths: list[str], out_path: str | Path) -> None:
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for i, p in enumerate(page_paths, start=1):
            z.write(p, arcname=f"page_{i:03d}{Path(p).suffix}")
        # ComicInfo.xml signals right-to-left reading so Komga/Kavita/Tachiyomi/YACReader
        # render two-page spreads RTL; without it those readers default to LTR spreads.
        n = len(page_paths)
        z.writestr(
            "ComicInfo.xml",
            '<?xml version="1.0" encoding="utf-8"?>'
            '<ComicInfo xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xmlns:xsd="http://www.w3.org/2001/XMLSchema">'
            f"<Manga>YesAndRightToLeft</Manga><PageCount>{n}</PageCount>"
            "</ComicInfo>",
        )


def run(settings: Settings, page_paths: list[str], chapter_number: int) -> dict[str, str]:
    missing = [p for p in page_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"export: {len(missing)} page image(s) missing: {missing}; "
            f"re-run the `letter` stage for chapter {chapter_number}"
        )
    pdf = settings.out_dir / f"chapter-{chapter_number}.pdf"
    cbz = settings.out_dir / f"chapter-{chapter_number}.cbz"
    to_pdf(page_paths, pdf)
    to_cbz(page_paths, cbz)
    return {"pdf": str(pdf), "cbz": str(cbz)}
