"""Tests for ln2manga.stages.scrape.

No network: clean_paragraphs is exercised with synthetic paragraph lists, and the
chapter-link extraction is exercised against sample anchor HTML strings.
"""
import re

from bs4 import BeautifulSoup

from ln2manga.stages.scrape import (
    _CHAPTER_LINK_RE,
    _CHAPTER_TEXT_RE,
    _NAV_RE,
    _looks_like_credit,
    clean_paragraphs,
)

# ── separator helpers ──────────────────────────────────────────────────────
SEP = "※　※　※　※　※"   # ideographic-space-joined ※ row, as the live site renders it


# ── clean_paragraphs: canonical two-fence credit block ─────────────────────
def test_canonical_credit_block_is_dropped():
    raw = [
        SEP,
        "Translated By :",
        "Remonwater",
        "Proofread By:",
        "Realigned by:",
        SEP,
        "The journey began at dawn.",
        "Subaru looked east.",
    ]
    paras, breaks = clean_paragraphs(raw)
    assert paras == ["The journey began at dawn.", "Subaru looked east."]
    assert breaks == []                       # both fences consumed, no scene break


def test_chapter1_layout_yields_no_break_for_simple_credits():
    # Mirrors the verified ch.1 head: credit block stripped, prose preserved,
    # and a later separator becomes the single scene break.
    raw = [SEP, "Translated By :", "Remonwater", "Proofread By:", "Realigned by:", SEP]
    raw += [f"prose {i}." for i in range(5)]
    raw += [SEP]
    raw += [f"more {i}." for i in range(3)]
    paras, breaks = clean_paragraphs(raw)
    assert paras == [f"prose {i}." for i in range(5)] + [f"more {i}." for i in range(3)]
    assert breaks == [5]                       # the mid-chapter ※ is the scene break


# ── clean_paragraphs: NO credit block (the bug #8 regression) ──────────────
def test_cold_open_with_early_scene_break_is_preserved():
    # Opening scene, then a real scene break, then more prose. There is NO credit
    # block, so nothing before the second separator may be discarded.
    raw = ["p0.", "p1.", "p2.", SEP, "q0.", "q1.", SEP, "tail."]
    paras, breaks = clean_paragraphs(raw)
    assert paras == ["p0.", "p1.", "p2.", "q0.", "q1.", "tail."]
    assert breaks == [3, 5]                    # both separators are scene breaks


def test_simplest_cold_open_not_collapsed():
    raw = [
        "Subaru opened his eyes.",
        SEP,
        "Three days earlier...",
        "The journey began.",
        SEP,
        "And so on.",
    ]
    paras, breaks = clean_paragraphs(raw)
    assert paras == [
        "Subaru opened his eyes.",
        "Three days earlier...",
        "The journey began.",
        "And so on.",
    ]
    assert breaks == [1, 3]


def test_lone_leading_separator_is_a_scene_break_not_a_fence():
    raw = [SEP, "First prose line.", "Second prose line."]
    paras, breaks = clean_paragraphs(raw)
    # leading ※ before any prose produces no scene break (nothing precedes it)
    assert paras == ["First prose line.", "Second prose line."]
    assert breaks == []


def test_two_fences_but_inter_fence_is_prose_kept():
    # Two early separators but the rows between them are full sentences, not credits.
    raw = [SEP, "This is clearly a long opening sentence of prose.", SEP, "Next."]
    paras, breaks = clean_paragraphs(raw)
    assert paras == ["This is clearly a long opening sentence of prose.", "Next."]
    # first ※ (no preceding prose) -> no break; second ※ after one para -> break at 1
    assert breaks == [1]


def test_short_name_callouts_between_fences_are_not_dropped():
    # A cold-open list of short character names sits between two early fences but
    # contains NO credit keyword, so it is genuine prose and must be preserved.
    raw = [SEP, "Subaru", "Emilia", "Beatrice", SEP, "The morning was quiet."]
    paras, breaks = clean_paragraphs(raw)
    assert paras == ["Subaru", "Emilia", "Beatrice", "The morning was quiet."]
    # leading ※ (nothing before it) -> no break; second ※ after 3 paras -> break at 3
    assert breaks == [3]


# ── clean_paragraphs: footnotes still cut ──────────────────────────────────
def test_footnotes_are_cut():
    raw = [
        SEP, "Translated By :", "Remonwater", SEP,
        "Body paragraph one.",
        "Translation Notes:",
        "[1] – a footnote",
    ]
    paras, breaks = clean_paragraphs(raw)
    assert paras == ["Body paragraph one."]
    assert breaks == []


# ── _looks_like_credit unit checks ─────────────────────────────────────────
def test_looks_like_credit_matches_labels_and_names():
    assert _looks_like_credit("Translated By :")
    assert _looks_like_credit("Proofread By:")
    assert _looks_like_credit("Realigned by:")
    assert _looks_like_credit("Edited by:")
    assert _looks_like_credit("Remonwater")          # short bare name


def test_looks_like_credit_rejects_prose():
    assert not _looks_like_credit("The journey began at dawn.")
    assert not _looks_like_credit("Subaru opened his eyes and looked around slowly.")
    assert not _looks_like_credit("Three days earlier...")  # ends with period


# ── chapter-link regex against sample URL/anchor strings ───────────────────
def _index_from_html(html: str, base_host: str = "witchculttranslation.com"):
    """Mini reimplementation of chapter_index's link loop over sample HTML,
    so we can assert the regex/host behavior without a Settings object or network."""
    from urllib.parse import urlparse

    soup = BeautifulSoup(html, "html.parser")
    out = {}
    for a in soup.select("a[href]"):
        href = a["href"]
        link_host = urlparse(href).netloc
        if base_host and link_host and link_host != base_host:
            continue
        text = a.get_text(strip=True)
        if _NAV_RE.search(text):
            continue
        m = _CHAPTER_TEXT_RE.search(text) or _CHAPTER_LINK_RE.search(href)
        if m:
            out.setdefault(int(m.group(1)), (text, href))
    return out


def test_link_regex_canonical_slug():
    m = _CHAPTER_LINK_RE.search(
        "https://witchculttranslation.com/2023/12/04/arc-6-chapter-1-the-way-back/"
    )
    assert m and int(m.group(1)) == 1


def test_link_regex_deviant_slug_via_text():
    # Slug deviates from /arc-6-chapter-N-, but the anchor text carries "Chapter 31".
    html = (
        '<a href="https://witchculttranslation.com/2020/03/11/'
        'arc-6-hall-of-memories-chapter-31-foo/">Arc 6 Chapter 31</a>'
    )
    idx = _index_from_html(html)
    assert 31 in idx


def test_offsite_mirror_is_excluded():
    html = (
        '<a href="https://translationchicken.com/arc-6-chapter-33/">Chapter 33</a>'
        '<a href="https://witchculttranslation.com/arc-6-chapter-34-x/">Chapter 34</a>'
    )
    idx = _index_from_html(html)
    assert 33 not in idx          # off-host mirror excluded
    assert 34 in idx              # same-site kept


def test_text_takes_precedence_over_href_number():
    # Anchor text number wins when both are present.
    html = '<a href="https://witchculttranslation.com/x-chapter-99-y/">Chapter 7</a>'
    idx = _index_from_html(html)
    assert 7 in idx and 99 not in idx


def test_relative_href_is_treated_as_same_site():
    html = '<a href="/arc-6-chapter-12-title/">Chapter 12</a>'
    idx = _index_from_html(html)
    assert 12 in idx


def test_prev_next_nav_links_do_not_shadow_real_chapter():
    # A "Previous Chapter 30" nav link precedes the real Chapter 30 anchor.
    # The nav link must be skipped so the real (correct) href wins.
    html = (
        '<a href="https://witchculttranslation.com/arc-6-chapter-29-x/">'
        'Previous Chapter 30</a>'
        '<a href="https://witchculttranslation.com/arc-6-chapter-30-real/">'
        'Arc 6 Chapter 30</a>'
        '<a href="https://witchculttranslation.com/arc-6-chapter-31-y/">'
        'Next Chapter 32</a>'
    )
    idx = _index_from_html(html)
    assert idx[30] == ("Arc 6 Chapter 30",
                       "https://witchculttranslation.com/arc-6-chapter-30-real/")
    assert 32 not in idx          # "Next Chapter 32" nav link skipped entirely


def test_nav_re_matches_nav_words_only():
    assert _NAV_RE.search("Previous Chapter 30")
    assert _NAV_RE.search("Next Chapter 32")
    assert _NAV_RE.search("Prev Chapter 5")
    assert not _NAV_RE.search("Arc 6 Chapter 31")
    assert not _NAV_RE.search("Chapter 33")


def test_nav_re_ignores_nav_words_used_as_content():
    # "Next"/"Previous" used as a content word in a real chapter title must NOT
    # be treated as a navigation anchor (the nav word only counts before "chapter").
    assert not _NAV_RE.search("Chapter 50 - What Comes Next")
    assert not _NAV_RE.search("Chapter 12 - The Next Step")
    assert not _NAV_RE.search("Chapter 7 - The Previous Day")


def test_content_title_with_nav_word_is_still_indexed():
    # A real chapter whose title happens to contain "Next" must still be indexed,
    # not silently dropped as a navigation link.
    html = (
        '<a href="https://witchculttranslation.com/arc-6-chapter-50-x/">'
        'Chapter 50 - What Comes Next</a>'
    )
    idx = _index_from_html(html)
    assert idx[50] == ("Chapter 50 - What Comes Next",
                       "https://witchculttranslation.com/arc-6-chapter-50-x/")


# ── run(): hard error when cleaning yields zero paragraphs ─────────────────
def test_run_raises_when_zero_paragraphs(monkeypatch):
    import pytest

    from ln2manga.config import load_settings
    from ln2manga.stages import scrape as scrape_mod

    settings = load_settings()
    # Chapter exists in the index, but the fetched content container has no <p>
    # tags (e.g. the site moved prose into <div>s) -> cleaning yields nothing.
    monkeypatch.setattr(
        scrape_mod, "chapter_index",
        lambda s: {7: ("Chapter 7", "https://witchculttranslation.com/x/")},
    )
    selector = settings.scrape.get("content_selector", ".entry-content")
    cls = selector.lstrip(".")
    monkeypatch.setattr(
        scrape_mod, "_get",
        lambda url, s, **kw: f'<div class="{cls}">no paragraphs here</div>',
    )
    with pytest.raises(SystemExit) as exc:
        scrape_mod.run(settings, 7)
    assert "0 paragraphs" in str(exc.value)


# ── encoding helper: ensure ※ survives UTF-8 round trip used in _is_sep ─────
def test_separator_detection_unicode():
    from ln2manga.stages.scrape import _is_sep

    assert _is_sep(SEP)
    assert not _is_sep("ordinary prose")
    # a mojibake-decoded ※ (latin-1 of its UTF-8 bytes) must NOT count as a sep
    mojibake = "※".encode("utf-8").decode("latin-1")
    assert not _is_sep(mojibake)


# ── site-agnostic config: overrides change cleaning/index behavior ──────────
def _settings(**scrape_overrides):
    """Load the shipped settings, then patch settings.scrape with overrides."""
    from ln2manga.config import load_settings

    s = load_settings()
    s.scrape.update(scrape_overrides)
    return s


def test_scene_break_chars_override_changes_cleaning():
    from ln2manga.stages.scrape import _rules, clean_paragraphs

    rules = _rules(_settings(scene_break_chars="#"))
    # With "#" configured as the separator, a "#" row is now a scene break and the
    # default ※ row is just (short, non-credit-keyword) prose.
    raw = ["p0.", "#", "p1.", "p2."]
    paras, breaks = clean_paragraphs(raw, rules)
    assert paras == ["p0.", "p1.", "p2."]
    assert breaks == [1]
    # ※ is no longer a separator under this config
    assert not rules.is_sep(SEP)
    assert rules.is_sep("#　#　#")


def test_credit_keywords_override_changes_credit_stripping():
    from ln2manga.stages.scrape import _rules, clean_paragraphs

    SEPH = "#"
    rules = _rules(_settings(scene_break_chars="#", credit_keywords=["staff", "編集"]))
    raw = [
        SEPH,
        "Staff:",
        "Alice",
        SEPH,
        "The story opens here.",
    ]
    paras, breaks = clean_paragraphs(raw, rules)
    # "Staff:" matches the custom credit keyword -> the leading block is dropped.
    assert paras == ["The story opens here."]
    assert breaks == []
    # The DEFAULT keyword "Translated By:" is NOT a credit line under this config,
    # so a block led by it is preserved as prose.
    raw2 = [SEPH, "Translated By :", "Bob", SEPH, "Body."]
    paras2, _ = clean_paragraphs(raw2, rules)
    assert "Translated By :" in paras2


def test_chapter_text_pattern_override_changes_index():
    from ln2manga.stages.scrape import _rules

    rules = _rules(_settings(chapter_text_pattern=r"episode\s+(\d+)\b"))
    # The default "Chapter N" wording no longer matches; "Episode N" does.
    assert rules.chapter_text_re.search("Episode 12")
    assert not rules.chapter_text_re.search("Chapter 12")


def test_work_label_flows_into_chapter_arc(monkeypatch, tmp_path):
    from ln2manga.stages import scrape as scrape_mod

    settings = _settings(work_label="book-1", paths={"data": str(tmp_path)})
    monkeypatch.setattr(
        scrape_mod, "chapter_index",
        lambda s: {7: ("Chapter 7", "https://witchculttranslation.com/x/")},
    )
    monkeypatch.setattr(
        scrape_mod, "_get",
        lambda url, s, **kw: '<div class="entry-content"><p>Hello there.</p></div>',
    )
    chapter = scrape_mod.run(settings, 7)
    assert chapter.arc == "book-1"
    assert chapter.paragraphs == ["Hello there."]


def test_work_label_flows_into_not_found_error(monkeypatch):
    import pytest

    from ln2manga.stages import scrape as scrape_mod

    settings = _settings(work_label="book-1")
    monkeypatch.setattr(scrape_mod, "chapter_index", lambda s: {1: ("c1", "u1")})
    with pytest.raises(SystemExit) as exc:
        scrape_mod.run(settings, 99)
    assert "book-1 index" in str(exc.value)


# ── absent config reproduces the current Re:Zero / WCT behavior ─────────────
def test_default_rules_reproduce_rezero_behavior():
    from ln2manga.stages.scrape import _rules, clean_paragraphs

    rules = _rules(_settings())  # no scrape overrides == shipped default
    # default work label
    assert rules.work_label == "arc-6"
    # default ※ separator detection
    assert rules.is_sep(SEP)
    # default credit-stripping (canonical two-fence credit block)
    raw = [
        SEP, "Translated By :", "Remonwater", "Proofread By:", "Realigned by:", SEP,
        "The journey began at dawn.", "Subaru looked east.",
    ]
    paras, breaks = clean_paragraphs(raw, rules)
    assert paras == ["The journey began at dawn.", "Subaru looked east."]
    assert breaks == []
    # default chapter index regexes
    assert rules.chapter_text_re.search("Chapter 31") and not rules.nav_re.search(
        "Arc 6 Chapter 31"
    )
    assert rules.chapter_link_re.search("arc-6-chapter-1-the-way-back")
    # footnote cut
    raw2 = ["Body.", "Translation Notes:", "[1] – note"]
    paras2, _ = clean_paragraphs(raw2, rules)
    assert paras2 == ["Body."]


def test_default_clean_paragraphs_matches_explicit_rules():
    # clean_paragraphs() with no rules arg must equal clean_paragraphs(raw, default).
    from ln2manga.stages.scrape import _rules, clean_paragraphs

    raw = [SEP, "Translated By :", "Remonwater", SEP, "Body one.", SEP, "Body two."]
    default_rules = _rules(_settings())
    assert clean_paragraphs(raw) == clean_paragraphs(raw, default_rules)
