"""Stage 1 — scrape: fetch & clean one chapter of a web-serialized work.

Polite: custom UA, inter-request delay, and on-disk HTML cache so re-runs never re-fetch.

The cleaning + index heuristics are configurable via ``settings.scrape.*`` so the
scraper is site-agnostic; every knob defaults to the value verified against the
original target (witchculttranslation.com, Re:Zero Arc 6), so behavior is
byte-identical when no extra config is supplied:
  - chapter prose lives in `.entry-content` <p> tags
  - a translator credit block sits between the first two `※ ※ ※` separator rows -> dropped
  - footnotes begin at a `Translation Notes:` paragraph -> dropped
  - remaining `※` rows are scene breaks -> recorded as indices, removed from prose

Configurable keys (with their built-in defaults — see ``_rules``):
  scene_break_chars   "※"        characters that mark a scene-break / separator row
  credit_keywords     [...]       leading words that mark a translator-credit line
  notes_header        "translation notes:"   prose row that begins the footnote block
  footnote_pattern    r"^\\s*\\[\\d+\\]\\s*[–-]"   per-footnote row pattern
  chapter_text_pattern r"chapter\\s+(\\d+)\\b"     chapter number from anchor TEXT
  chapter_link_pattern r"chapter-(\\d+)\\b"        chapter number from href slug
  nav_pattern         r"\\b(previous|prev|next)\\s+chapter\\b"   nav-anchor skip
  paragraph_tags      ["p"]       tag names holding chapter prose inside the container
  work_label          "arc-6"     Chapter.arc value + label in the not-found error
  encoding            "utf-8"     forced response encoding
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from ..artifacts import Chapter
from ..config import Settings

# ── built-in defaults (the verified witchcult / Re:Zero Arc 6 behavior) ──────
_DEFAULT_SCENE_BREAK_CHARS = "※"
_DEFAULT_CREDIT_KEYWORDS = [
    "translated", "proofread", "realigned", "edited",
    "translation", "tlc", "tl", "by",
]
_DEFAULT_NOTES_HEADER = "translation notes:"
_DEFAULT_FOOTNOTE_PATTERN = r"^\s*\[\d+\]\s*[–-]"
_DEFAULT_CHAPTER_TEXT_PATTERN = r"chapter\s+(\d+)\b"
_DEFAULT_CHAPTER_LINK_PATTERN = r"chapter-(\d+)\b"
_DEFAULT_NAV_PATTERN = r"\b(previous|prev|next)\s+chapter\b"
_DEFAULT_PARAGRAPH_TAGS = ["p"]
_DEFAULT_WORK_LABEL = "arc-6"
_DEFAULT_ENCODING = "utf-8"


def _credit_regex(keywords: list[str]) -> re.Pattern[str]:
    """Build the translator-credit line matcher from the keyword list: a label
    line ("Proofread By:") or a leading credit word followed by anything."""
    alt = "|".join(re.escape(k) for k in keywords)
    return re.compile(rf"(?i)^\s*({alt})\b.*?:?\s*$")


def _sep_regex(scene_break_chars: str) -> re.Pattern[str]:
    """A row made up solely of whitespace + the configured scene-break chars."""
    cls = re.escape(scene_break_chars)
    return re.compile(rf"^[\s　{cls}]+$")


@dataclass(frozen=True)
class _Rules:
    """Compiled, config-driven cleaning + index heuristics. Built once per call
    via ``_rules(settings)`` so the values pick up ``settings.scrape.*`` overrides
    while defaulting to the verified literals."""
    scene_break_chars: str
    sep_re: re.Pattern[str]
    credit_re: re.Pattern[str]
    notes_re: re.Pattern[str]
    footnote_re: re.Pattern[str]
    chapter_text_re: re.Pattern[str]
    chapter_link_re: re.Pattern[str]
    nav_re: re.Pattern[str]
    paragraph_tags: list[str]
    work_label: str
    encoding: str

    def is_sep(self, p: str) -> bool:
        return bool(self.sep_re.match(p)) and any(
            c in p for c in self.scene_break_chars
        )

    def looks_like_credit(self, p: str) -> bool:
        """True for translator-credit rows: a label line ("Proofread By:") or a
        short bare name ("Remonwater"). Prose paragraphs are long and/or end with
        sentence punctuation, so they fail both tests."""
        if self.credit_re.match(p):
            return True
        return (
            len(p) <= 40
            and p.count(" ") <= 3
            and not p.endswith((".", "。", "”", "\"", "！", "？", "!", "?"))
        )


def _rules(settings: Settings) -> _Rules:
    s = settings.scrape
    scene_break_chars = s.get("scene_break_chars", _DEFAULT_SCENE_BREAK_CHARS)
    credit_keywords = s.get("credit_keywords", _DEFAULT_CREDIT_KEYWORDS)
    notes_header = s.get("notes_header", _DEFAULT_NOTES_HEADER)
    return _Rules(
        scene_break_chars=scene_break_chars,
        sep_re=_sep_regex(scene_break_chars),
        credit_re=_credit_regex(credit_keywords),
        notes_re=re.compile(rf"(?i)^\s*{re.escape(notes_header.rstrip(':'))}\s*:"),
        footnote_re=re.compile(s.get("footnote_pattern", _DEFAULT_FOOTNOTE_PATTERN)),
        chapter_text_re=re.compile(
            s.get("chapter_text_pattern", _DEFAULT_CHAPTER_TEXT_PATTERN), re.I
        ),
        chapter_link_re=re.compile(
            s.get("chapter_link_pattern", _DEFAULT_CHAPTER_LINK_PATTERN), re.I
        ),
        nav_re=re.compile(
            s.get("nav_pattern", _DEFAULT_NAV_PATTERN), re.I
        ),
        paragraph_tags=list(s.get("paragraph_tags", _DEFAULT_PARAGRAPH_TAGS)),
        work_label=s.get("work_label", _DEFAULT_WORK_LABEL),
        encoding=s.get("encoding", _DEFAULT_ENCODING),
    )


# ── module-level defaults, for callers/tests that want the verified behavior ─
# (mirror the built-in defaults; kept so importers don't need a Settings object.)
_DEFAULT_RULES = _Rules(
    scene_break_chars=_DEFAULT_SCENE_BREAK_CHARS,
    sep_re=_sep_regex(_DEFAULT_SCENE_BREAK_CHARS),
    credit_re=_credit_regex(_DEFAULT_CREDIT_KEYWORDS),
    notes_re=re.compile(rf"(?i)^\s*{re.escape(_DEFAULT_NOTES_HEADER.rstrip(':'))}\s*:"),
    footnote_re=re.compile(_DEFAULT_FOOTNOTE_PATTERN),
    chapter_text_re=re.compile(_DEFAULT_CHAPTER_TEXT_PATTERN, re.I),
    chapter_link_re=re.compile(_DEFAULT_CHAPTER_LINK_PATTERN, re.I),
    nav_re=re.compile(_DEFAULT_NAV_PATTERN, re.I),
    paragraph_tags=_DEFAULT_PARAGRAPH_TAGS,
    work_label=_DEFAULT_WORK_LABEL,
    encoding=_DEFAULT_ENCODING,
)
_SEP_RE = _DEFAULT_RULES.sep_re
_NOTES_RE = _DEFAULT_RULES.notes_re
_FOOTNOTE_RE = _DEFAULT_RULES.footnote_re
_CHAPTER_TEXT_RE = _DEFAULT_RULES.chapter_text_re
_CHAPTER_LINK_RE = _DEFAULT_RULES.chapter_link_re
_NAV_RE = _DEFAULT_RULES.nav_re
_CREDIT_RE = _DEFAULT_RULES.credit_re


def _is_sep(p: str, rules: _Rules = _DEFAULT_RULES) -> bool:
    return rules.is_sep(p)


def _looks_like_credit(p: str, rules: _Rules = _DEFAULT_RULES) -> bool:
    return rules.looks_like_credit(p)


def _get(url: str, settings: Settings, *, cache: bool = True) -> str:
    slug = re.sub(r"\W+", "_", url.split("//", 1)[-1]).strip("_")[:120]
    cache_path = settings.cache_dir("html") / f"{slug}.html"
    if cache and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    time.sleep(float(settings.scrape.get("delay_s", 3.0)))
    headers = {"User-Agent": settings.scrape.get("user_agent", "ln2manga/0.1")}
    # A single transient 429/503/timeout should not abort with a raw traceback.
    # Retry a few times with exponential backoff for 5xx/429 (honouring
    # Retry-After); fail fast on other 4xx, which won't recover on retry.
    resp = None
    for attempt in range(4):
        try:
            resp = requests.get(url, headers=headers, timeout=40)
            resp.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            status = getattr(e.response, "status_code", None)
            if attempt == 3 or (status is not None and status < 500 and status != 429):
                raise SystemExit(f"Fetch failed for {url}: {e}")
            retry_after = getattr(e.response, "headers", {}).get("Retry-After")
            try:
                delay = float(retry_after)
            except (TypeError, ValueError):
                delay = min(2 ** attempt, 8)
            time.sleep(delay)
    # The site declares charset=UTF-8 in <head>; requests only inspects the HTTP
    # header and falls back to ISO-8859-1 for header-less text/html, which mangles
    # the ※ separators and Japanese footnotes. Force the configured encoding so the
    # cache (and every re-run that reads it) stays clean.
    resp.encoding = settings.scrape.get("encoding", _DEFAULT_ENCODING)
    html = resp.text
    cache_path.write_text(html, encoding="utf-8")
    return html


def chapter_index(settings: Settings) -> dict[int, tuple[str, str]]:
    """Return {chapter_number: (title, url)} for the configured work."""
    rules = _rules(settings)
    base_url = settings.scrape["base_url"]
    url = base_url.rstrip("/") + settings.scrape["arc_path"]
    base_host = urlparse(base_url).netloc
    soup = BeautifulSoup(_get(url, settings), "html.parser")
    out: dict[int, tuple[str, str]] = {}
    for a in soup.select("a[href]"):
        # Resolve relative hrefs against base_url so the stored URL is absolute
        # and _get() can fetch it; a same-site relative link resolves to base_host.
        href = urljoin(base_url, a["href"])
        # Restrict to the configured site so off-host mirrors don't shadow the
        # canonical chapter links.
        link_host = urlparse(href).netloc
        if base_host and link_host and link_host != base_host:
            continue
        text = a.get_text(strip=True)
        # A "Previous/Next Chapter N" nav link names a neighbouring chapter and
        # would otherwise win via setdefault — skip it so the real anchor stands.
        if rules.nav_re.search(text):
            continue
        m = rules.chapter_text_re.search(text) or rules.chapter_link_re.search(href)
        if m:
            n = int(m.group(1))
            out.setdefault(n, (text, href))
    return out


def clean_paragraphs(
    raw: list[str], rules: _Rules = _DEFAULT_RULES
) -> tuple[list[str], list[int]]:
    raw = [p for p in (s.strip() for s in raw) if p]
    seps = [i for i, p in enumerate(raw) if rules.is_sep(p)]
    start = 0
    # Only strip the leading credit block when it really is one: a separator near
    # the top, a second separator close behind, and the rows between them all
    # looking like credit lines. Otherwise the leading ※ is a genuine scene break
    # (handled by the main loop) and the opening prose must be preserved.
    if len(seps) >= 2 and seps[0] <= 3 and seps[1] <= 8:
        credit_rows = raw[seps[0] + 1:seps[1]]
        # Require at least one real credit keyword ("Translated By :", etc.); a
        # block of only short name-callout lines (e.g. a cold-open list of
        # character names) is genuine prose, not a credit block.
        if (
            credit_rows
            and any(rules.credit_re.match(r) for r in credit_rows)
            and all(rules.looks_like_credit(r) for r in credit_rows)
        ):
            start = seps[1] + 1        # genuine credit block -> drop it
    body = raw[start:]

    for i, p in enumerate(body):       # cut footnotes
        if rules.notes_re.match(p) or rules.footnote_re.match(p):
            body = body[:i]
            break

    paragraphs: list[str] = []
    scene_breaks: list[int] = []
    for p in body:
        if rules.is_sep(p):
            if paragraphs:
                scene_breaks.append(len(paragraphs))   # break BEFORE the next paragraph
            continue
        paragraphs.append(p)
    return paragraphs, scene_breaks


def run(settings: Settings, number: int) -> Chapter:
    rules = _rules(settings)
    index = chapter_index(settings)
    if number not in index:
        nearest = sorted(index, key=lambda c: abs(c - number))[:5]
        raise SystemExit(
            f"Chapter {number} not found in {rules.work_label} index "
            f"(have {len(index)} chapters; nearest available: {nearest})."
        )
    title, url = index[number]
    soup = BeautifulSoup(_get(url, settings), "html.parser")
    el = soup.select_one(settings.scrape.get("content_selector", ".entry-content"))
    if el is None:
        raise SystemExit("Could not find chapter content container.")
    raw = [p.get_text(" ", strip=True) for p in el.find_all(rules.paragraph_tags)]
    paragraphs, scene_breaks = clean_paragraphs(raw, rules)

    cap = int(settings.scrape.get("max_paragraphs", 0) or 0)
    if cap:
        paragraphs = paragraphs[:cap]
        scene_breaks = [b for b in scene_breaks if b < cap]

    if not paragraphs:
        selector = settings.scrape.get("content_selector", ".entry-content")
        raise SystemExit(
            f"Scraped 0 paragraphs from {url} via selector {selector!r} "
            f"({len(raw)} <p> tags found before cleaning). "
            f"Site markup may have changed — check content_selector "
            f"and the credit/footnote filters."
        )

    chapter = Chapter(arc=rules.work_label, number=number, title=title, url=url,
                      paragraphs=paragraphs, scene_breaks=scene_breaks)
    out = settings.raw_dir / f"chapter-{number}.json"
    chapter.model_dump_json()  # validate
    Path(out).write_text(chapter.model_dump_json(indent=2), encoding="utf-8")
    return chapter
