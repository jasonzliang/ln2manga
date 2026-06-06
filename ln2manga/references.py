"""Real reference images for characters — including ONLINE SEARCH for official designs.

OPT-IN, fully backward compatible. When `references.enabled` is false (the default) nothing here
is used and charsheet behaves exactly as before.

A character can be anchored on a REAL image instead of (or as the basis for) an AI-generated
sheet. A source is found in priority order:
  1. settings.references["sources"]              — inline map in the master config YAML
  2. references.external_file (a separate YAML)  — optional extra map (name -> url/path)
  3. bible.Character.ref_image                   — a source set in code
  4. ONLINE SEARCH (references.search.enabled)   — the OpenAI web-search tool finds the official
       character design / key visual and returns direct image URLs, which are validated and
       downloaded. Requires an OpenAI client (passed through from charsheet).

Lookups are case-insensitive and alias-aware. Downloads are validated (HTTP 200, a minimum byte
size, decodable image, min dimension) and cached as PNG under data/cache/refs/<name>.png. Any
error logs to stderr and yields None, so callers transparently fall back to the AI sheet.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import requests
import yaml
from PIL import Image

from . import bible
from .config import PROJECT_ROOT, Settings
from .cost import BudgetExceeded
from .retry import with_retry

_IMG_URL_RE = re.compile(r'https?://[^\s"\'<>)]+?\.(?:png|jpe?g|webp)(?:\?[^\s"\'<>)]*)?', re.I)


def _link_or_copy(src, dest) -> None:
    """Materialize `dest` as a HARDLINK to `src` (same inode, zero extra bytes), falling back to a
    real byte copy on a filesystem that can't hardlink (cross-device / unsupported / Windows).

    `src` is a write-once content-addressed cache file, so the hardlink never goes stale; the
    unlink-then-link re-points `dest` at the current target if it changed.
    """
    dest = Path(dest)
    try:
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        os.link(src, dest)
    except OSError:
        shutil.copyfile(src, dest)

_SEARCH_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "series": "",            # e.g. "Re:Zero" — disambiguates the search
    "model": "gpt-5.4-mini",  # any model that supports the web-search tool
    "tool": "web_search_preview",  # or "web_search"
    "user_agent": "",        # blank -> a realistic browser UA (image hosts block bot UAs)
    "max_urls": 8,           # candidate URLs to try
    "max_keep": 3,           # verified official images to KEEP and synthesize the sheet from
    "min_bytes": 3000,       # reject 404 placeholders / icons
    "cost_per_call": 0.03,   # budget estimate per web-search / agentic round
    "verify": True,          # vision-check that each image is the RIGHT character's official art
    "verify_model": "gpt-5.4-mini",
    "agentic": True,         # iterate like a subagent: model drives web_search + a verify_image
                             #   tool (download + vision-check) until it has max_keep verified images
    "agentic_rounds": 6,     # max tool-loop rounds
}
_REF_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "mode": "stylize",       # "raw" = use the image as-is; "stylize" = redraw as a B&W manga sheet
    "external_file": "config/references.yaml",
    "sources": {},
    "search": dict(_SEARCH_DEFAULTS),
}


def _slug(name: str) -> str:
    # Append a short hash of the name so DISTINCT characters can never collide on the same
    # cache/refs/<slug>.* files under parallel generation (audit: slug-collision races).
    s = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name).strip("_").lower()
    h = hashlib.sha1((name or "").strip().lower().encode("utf-8")).hexdigest()[:6]
    return f"{s or 'ref'}_{h}"


def references_config(settings: Settings) -> dict[str, Any]:
    """Effective references config: defaults merged under settings.references (a declared
    Settings field populated straight from the master config YAML)."""
    cfg: dict[str, Any] = {**_REF_DEFAULTS}
    raw = getattr(settings, "references", None)
    if isinstance(raw, dict):
        cfg.update(raw)
    search_raw = raw.get("search") if isinstance(raw, dict) else None
    cfg["search"] = {**_SEARCH_DEFAULTS, **(search_raw if isinstance(search_raw, dict) else {})}
    if not isinstance(cfg.get("sources"), dict):
        cfg["sources"] = {}
    return cfg


def load_references_config(settings: Settings, config_path: str | Path | None = None) -> dict[str, Any]:
    """Back-compat shim. `references` is now a declared Settings field loaded from the master
    config, so no attaching is needed; this just returns the effective config."""
    return references_config(settings)


def _external_sources(cfg: dict[str, Any]) -> dict[str, str]:
    rel = cfg.get("external_file")
    if not rel:
        return {}
    p = Path(rel)
    if not p.is_absolute():
        p = PROJECT_ROOT / rel
    if not p.exists():
        return {}
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:
        print(f"[refs] could not read external_file {p}: {e}", file=sys.stderr)
        return {}
    if isinstance(data, dict) and isinstance(data.get("sources"), dict):
        data = data["sources"]
    if not isinstance(data, dict):
        return {}
    return {str(k): v for k, v in data.items() if v}   # value may be a str OR a list


def _as_list(v: Any) -> list[str]:
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if x]
    return [str(v)] if v else []


def reference_sources(name: str, settings: Settings) -> list[str]:
    """All EXPLICITLY-configured reference sources for a character (URLs/paths). A configured
    value may be a single string OR a list (several official images to synthesize the sheet
    from). Online search is handled separately, in resolve_reference_set."""
    if not name:
        return []
    cfg = references_config(settings)
    char = bible.get_character(name, settings)
    candidates = [name] + ([char.name] + list(char.aliases) if char else [])

    def _lookup(table: dict) -> Any:
        lowered = {str(k).strip().lower(): v for k, v in table.items()}
        for cand in candidates:
            if cand.strip().lower() in lowered:
                return lowered[cand.strip().lower()]
        return None

    inline = cfg.get("sources") or {}
    v = _lookup(inline) if isinstance(inline, dict) else None
    if v:
        return _as_list(v)
    v = _lookup(_external_sources(cfg))
    if v:
        return _as_list(v)
    if char and getattr(char, "ref_image", None):
        return _as_list(char.ref_image)
    return []


def reference_source(name: str, settings: Settings) -> str | None:
    """The first explicitly-configured reference source for a character, or None."""
    srcs = reference_sources(name, settings)
    return srcs[0] if srcs else None


def _is_url(src: str) -> bool:
    s = src.lower()
    return s.startswith("http://") or s.startswith("https://")


# Image hosts and wikis often block non-browser User-Agents; use a realistic browser UA for
# reference image/page fetches (configurable via references.search.user_agent).
_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


def _ref_headers(settings: Settings, *, html: bool = False) -> dict[str, str]:
    sc = references_config(settings).get("search", {})
    ua = sc.get("user_agent") or _BROWSER_UA
    accept = ("text/html,application/xhtml+xml,*/*;q=0.8" if html
              else "image/avif,image/webp,image/png,image/*,*/*;q=0.8")
    return {"User-Agent": ua, "Accept": accept}


def _save_png(img_bytes: bytes, dest: Path, *, min_bytes: int = 0) -> Path | None:
    # min_bytes>0 marks an UNTRUSTED (online-searched) image, which also gets a min-dimension
    # gate to reject 404 placeholders/icons. Explicit user-provided refs (min_bytes=0) are trusted.
    if min_bytes and len(img_bytes) < min_bytes:
        print(f"[refs] image too small ({len(img_bytes)}B < {min_bytes}B) -> skip", file=sys.stderr)
        return None
    try:
        im = Image.open(io.BytesIO(img_bytes))
        im.load()
    except Exception as e:
        print(f"[refs] not a valid image: {e}", file=sys.stderr)
        return None
    # Reject low-res thumbnails for SEARCHED images (min_bytes>0 only): a 200px floor keeps a
    # correct-but-smallish official design while dropping icons/thumbnails (a 260x260 thumbnail
    # was wrongly accepted before — audit BUG 2). Explicit user refs (min_bytes=0) stay trusted.
    if min_bytes and min(im.size) < 200:
        print(f"[refs] image dimensions too small {im.size} -> skip", file=sys.stderr)
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.convert("RGB" if im.mode not in ("L", "RGB", "RGBA") else im.mode).save(dest, "PNG")
    return dest


def _fetch_to(src: str, settings: Settings, dest: Path, *, min_bytes: int = 0) -> Path | None:
    """Download a URL (polite UA + timeout) or read a local path; validate + cache as PNG."""
    if _is_url(src):
        try:
            timeout = float(settings.image.get("timeout_s", 240))
            resp = requests.get(src, headers=_ref_headers(settings), timeout=timeout)
            resp.raise_for_status()
            return _save_png(resp.content, dest, min_bytes=min_bytes)
        except Exception as e:
            print(f"[refs] download failed <{src[:80]}>: {type(e).__name__}: {str(e)[:120]}",
                  file=sys.stderr)
            return None
    p = Path(src).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        print(f"[refs] local reference not found: {p}", file=sys.stderr)
        return None
    try:
        return _save_png(p.read_bytes(), dest, min_bytes=min_bytes)
    except Exception as e:
        print(f"[refs] could not read local reference <{p}>: {type(e).__name__}: {str(e)[:120]}",
              file=sys.stderr)
        return None


def _search_cache(settings: Settings, canon: str) -> Path:
    return settings.cache_dir("refs") / f"{_slug(canon)}.search.json"


def _page_image_urls(page_url: str, settings: Settings, limit: int = 6) -> list[str]:
    """Fetch a page and extract real image URLs (og:image first, then <img> sources). This
    yields genuine downloadable images, unlike model-fabricated direct URLs."""
    from urllib.parse import urljoin

    from bs4 import BeautifulSoup
    try:
        timeout = float(settings.image.get("timeout_s", 240))
        resp = requests.get(page_url, headers=_ref_headers(settings, html=True), timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        print(f"[refs] page fetch failed <{page_url[:70]}>: {type(e).__name__}", file=sys.stderr)
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    out: list[str] = []
    og = soup.find("meta", attrs={"property": "og:image"})
    if og and og.get("content"):
        out.append(urljoin(page_url, og["content"]))
    for im in soup.find_all("img"):
        src = im.get("src") or im.get("data-src") or ""
        if src:
            u = urljoin(page_url, src)
            if _IMG_URL_RE.match(u):
                out.append(u)
    seen: set[str] = set()
    return [u for u in out if not (u in seen or seen.add(u))][:limit]


def search_reference_urls(name: str, settings: Settings, client, tracker=None) -> list[str]:
    """Use the OpenAI web-search tool to find OFFICIAL-design image URLs for a character.
    Results are cached per character so repeat runs don't re-search."""
    cfg = references_config(settings)
    sc = cfg.get("search", {})
    char = bible.get_character(name, settings)
    canon = char.name if char else name
    cache_p = _search_cache(settings, canon)
    if cache_p.exists():
        try:
            return list(json.loads(cache_p.read_text()).get("urls", []))
        except Exception:
            pass

    series = sc.get("series") or ""
    desc = char.descriptor if char else name
    query = (
        f"Find OFFICIAL character design / key-visual art for {canon}"
        + (f" from {series}" if series else "") + f" (the character is: {desc}). "
        "List, one per line: (a) any DIRECT image URLs ending in .png/.jpg/.jpeg/.webp, and "
        "(b) the official website and fandom-wiki PAGE URLs where the official art appears. "
        "Prefer official sources. No other commentary."
    )
    model = sc.get("model", "gpt-5.4-mini")
    tool = {"type": sc.get("tool", "web_search")}
    if tracker is not None:
        tracker.check(float(sc.get("cost_per_call", 0.03)))
    try:
        r = with_retry(client.responses.create)(model=model, tools=[tool], input=query)
    except Exception as e:
        print(f"[refs] online search failed for {canon}: {type(e).__name__}: {str(e)[:140]}",
              file=sys.stderr)
        return []

    if tracker is not None:
        usd = float(sc.get("cost_per_call", 0.03))
        try:
            u = getattr(r, "usage", None)
            if u is not None:
                usd += tracker.estimate_text(model, getattr(u, "input_tokens", 0) or 0,
                                             getattr(u, "output_tokens", 0) or 0)
        except Exception:
            pass
        tracker.record("search", model, usd, {"char": canon})

    # Collect both direct image URLs and PAGE URLs (from the model text AND the search's real
    # citation annotations). Models often fabricate direct image URLs, so we also scrape the
    # cited pages for genuine <img>/og:image URLs.
    text = getattr(r, "output_text", "") or ""
    img_urls: list[str] = list(_IMG_URL_RE.findall(text))
    page_urls: list[str] = [u for u in re.findall(r'https?://[^\s"\'<>)]+', text)
                            if not _IMG_URL_RE.match(u)]
    for item in getattr(r, "output", []) or []:
        for ce in getattr(item, "content", []) or []:
            for ann in getattr(ce, "annotations", []) or []:
                u = getattr(ann, "url", None)
                if u:
                    (img_urls if _IMG_URL_RE.match(u) else page_urls).append(u)

    seen_pages: set[str] = set()
    for pg in [p for p in page_urls if not (p in seen_pages or seen_pages.add(p))][:4]:
        img_urls.extend(_page_image_urls(pg, settings))

    seen: set[str] = set()
    urls = [u for u in img_urls if not (u in seen or seen.add(u))]
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    cache_p.write_text(json.dumps({"query": query, "urls": urls}, indent=2), encoding="utf-8")
    return urls


def _verify_match(img_bytes: bytes, canon: str, desc: str, sc: dict[str, Any],
                  client, tracker=None) -> bool:
    """Ask a vision model whether the image actually depicts this character. Web search can
    return the wrong character's art, so this gate rejects mismatches. Fails CLOSED: rejects on a
    clear 'no' AND when the verify call errors, so an unverifiable image is never anchored."""
    model = sc.get("verify_model") or sc.get("model", "gpt-5.4-mini")
    # Honor the same budget cap as the rest of the pipeline: guard BEFORE this billable vision
    # call (verify spend otherwise bypasses the cap until the next round's coarse check). Let
    # BudgetExceeded propagate so the agentic/single-shot loops stop cleanly.
    if tracker is not None:
        tracker.check(float(sc.get("cost_per_call", 0.03)))
    b64 = base64.b64encode(img_bytes).decode()
    try:
        r = with_retry(client.responses.create)(model=model, input=[{"role": "user", "content": [
            {"type": "input_text", "text":
                f"This should be a clean OFFICIAL character design of {canon} ({desc}). "
                f"Answer 'yes' ONLY if the image clearly shows {canon} as the single main "
                "subject of a clean character design or portrait. Answer 'no' if it shows a "
                "different character, is a promotional poster with several characters, or is "
                "dominated by logos/text. Answer only 'yes' or 'no'."},
            {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"}]}])
    except Exception as e:
        print(f"[refs] verify call failed for {canon}: {type(e).__name__} -> rejecting unverified",
              file=sys.stderr)
        return False
    if tracker is not None:
        try:
            u = getattr(r, "usage", None)
            if u is not None:
                tracker.record("verify", model, tracker.estimate_text(
                    model, getattr(u, "input_tokens", 0) or 0,
                    getattr(u, "output_tokens", 0) or 0), {"char": canon})
        except Exception:
            pass
    ans = (getattr(r, "output_text", "") or "").strip().lower()
    if not ans.startswith("y"):
        print(f"[refs] {canon}: searched image rejected by vision check (answer: {ans[:24]!r})",
              file=sys.stderr)
        return False
    return True


def _record_search_cost(tracker, model: str, resp: Any, sc: dict[str, Any], char: str = "") -> None:
    if tracker is None:
        return
    usd = float(sc.get("cost_per_call", 0.03))
    try:
        u = getattr(resp, "usage", None)
        if u is not None:
            usd += tracker.estimate_text(model, getattr(u, "input_tokens", 0) or 0,
                                         getattr(u, "output_tokens", 0) or 0)
    except Exception:
        pass
    tracker.record("search", model, usd, {"char": char})


def _verify_candidate(url: str, canon: str, desc: str, sc: dict[str, Any], settings: Settings,
                      client, tracker, kept: list, refs_dir: Path, slug: str) -> dict:
    """Download + validate + vision-verify one candidate URL; on success record it in `kept`."""
    if not url or any(u == url for (u, _p) in kept):
        return {"ok": False, "reason": "empty or duplicate url"}
    cand = refs_dir / f"{slug}.cand.png"
    p = _fetch_to(url, settings, cand, min_bytes=int(sc.get("min_bytes", 3000)))
    if p is None:
        return {"ok": False, "reason": "download failed, too small, or not an image"}
    if sc.get("verify", True) and not _verify_match(p.read_bytes(), canon, desc, sc, client, tracker):
        p.unlink(missing_ok=True)
        return {"ok": False, "reason": f"not a clean official design of {canon}"}
    member = refs_dir / f"{slug}.set{len(kept)}.png"
    p.replace(member)
    kept.append((url, member))
    return {"ok": True, "reason": f"verified official design of {canon}"}


def _agentic_search_urls(name: str, settings: Settings, client, tracker, want: int) -> list[Path]:
    """Agentic loop (a subagent inside the pipeline): the model drives the web_search tool and a
    verify_image function tool (download + vision-check) until it has `want` verified official
    images. This is far more reliable than a single search call because every candidate is
    actually fetched and confirmed, and the model keeps searching when candidates fail."""
    cfg = references_config(settings)
    sc = cfg.get("search", {})
    char = bible.get_character(name, settings)
    canon = char.name if char else name
    desc = char.descriptor if char else name
    series = sc.get("series") or ""
    model = sc.get("model", "gpt-5.4-mini")
    refs_dir = settings.cache_dir("refs")
    slug = _slug(canon)
    tools = [
        {"type": sc.get("tool", "web_search")},
        {"type": "function", "name": "list_page_images",
         "description": ("Fetch a web PAGE and return the REAL image URLs found on it (og:image and "
                         "<img> sources). Use this on an official/wiki/database page to get genuine "
                         "downloadable image URLs instead of guessing. Returns {image_urls: [...]}."),
         "parameters": {"type": "object", "additionalProperties": False,
                        "properties": {"url": {"type": "string"}}, "required": ["url"]}},
        {"type": "function", "name": "verify_image",
         "description": ("Download a candidate DIRECT image URL and check it is a CLEAN official "
                         "character design of the target character. Returns {ok, reason}."),
         "parameters": {"type": "object", "additionalProperties": False,
                        "properties": {"url": {"type": "string"}}, "required": ["url"]}},
    ]
    instr = (
        f"Find {want} distinct DIRECT image URLs of clean OFFICIAL character-design art of {canon}"
        + (f" from {series}" if series else "") + f". The character: {desc}.\n"
        "Method (do NOT guess image URLs — guessed URLs 404):\n"
        "1) Use web search to find the official site / wiki / character-database PAGE for the character.\n"
        "2) Call list_page_images(page_url) to get the REAL image URLs on that page.\n"
        "3) For each promising direct image URL, call verify_image(url).\n"
        f"Repeat (try other pages/sources) until {want} URLs are verified ok, or you have exhausted "
        "good options. If a verify fails, go back to searching/listing other pages — do not give up "
        "after one failure. Prefer a clean single-character full-body design or portrait; avoid promo "
        "posters with several characters or heavy text/logos."
    )
    kept: list = []
    if tracker is not None:
        tracker.check(float(sc.get("cost_per_call", 0.03)))
    try:
        resp = with_retry(client.responses.create)(model=model, tools=tools, input=instr)
    except Exception as e:
        print(f"[refs] agentic search failed for {canon}: {type(e).__name__}: {str(e)[:140]}",
              file=sys.stderr)
        # Cache this errored attempt exactly like a fruitless completed search so it is never
        # re-paid on later runs (the recurring re-search leak). A BudgetExceeded from the check
        # above propagates uncaught and stays UNMARKED, so it re-runs once budget is raised.
        (refs_dir / f"{slug}.searched").write_text("", encoding="utf-8")
        return []
    _record_search_cost(tracker, model, resp, sc, canon)        # round 1 cost
    try:
        for _ in range(int(sc.get("agentic_rounds", 6))):
            fcs = [o for o in (getattr(resp, "output", []) or [])
                   if getattr(o, "type", None) == "function_call"]
            if not fcs:
                break
            outputs = []
            for fc in fcs:
                try:
                    url = json.loads(fc.arguments).get("url", "")
                except Exception:
                    url = ""
                if getattr(fc, "name", "") == "list_page_images":
                    res = {"image_urls": _page_image_urls(url, settings, limit=15)}
                else:                                  # verify_image
                    res = _verify_candidate(url, canon, desc, sc, settings, client, tracker,
                                            kept, refs_dir, slug)
                outputs.append({"type": "function_call_output", "call_id": fc.call_id,
                                "output": json.dumps(res)})
            if len(kept) >= want:
                break
            if tracker is not None:                # pre-check budget before each extra round (#4)
                # A BudgetExceeded here must PROPAGATE (matching the round-1 and verify-time
                # guards) so the search stays UNMARKED and re-runs once budget is raised; only a
                # bare `break` would swallow it and then mark the search complete below (audit).
                tracker.check(float(sc.get("cost_per_call", 0.03)))
            try:
                resp = with_retry(client.responses.create)(
                    model=model, previous_response_id=resp.id, tools=tools, input=outputs)
            except Exception as e:
                print(f"[refs] agentic continue failed for {canon}: {type(e).__name__}", file=sys.stderr)
                break
            _record_search_cost(tracker, model, resp, sc, canon)   # record each continuation (#12)
    except BudgetExceeded:
        # A budget breach (per-round guard above OR verify-time inside _verify_candidate) leaves
        # NO `.searched` marker, so the search re-runs once budget is raised. But _verify_candidate
        # may already have materialized partial `{slug}.set*.png` members; if left on disk they make
        # resolve_reference_set short-circuit on `existing` and never resume (audit). Unlink them.
        for f in refs_dir.glob(f"{slug}.set*.png"):
            f.unlink(missing_ok=True)
        raise
    (refs_dir / f"{slug}.searched").write_text("", encoding="utf-8")   # mark completed (cache the outcome)
    if kept:
        print(f"[refs] {canon}: agentic search verified {len(kept)} official image(s)",
              file=sys.stderr)
    return [p for (_u, p) in kept][:want]


def resolve_reference_set(name: str, settings: Settings, client=None, tracker=None,
                          limit: int | None = None) -> list[Path]:
    """Return a list of validated PNG references for a character (cached under data/cache/refs/).

    - An explicit source (inline/external/bible) yields exactly that one trusted image.
    - Otherwise, if references.search.enabled and a client is given, the web is searched for
      OFFICIAL designs; each candidate is downloaded, size-validated, and (when search.verify)
      vision-checked to confirm it is the RIGHT character. Up to `limit` (search.max_keep)
      verified images are returned, so the sheet can be SYNTHESIZED from several official refs.
    """
    char = bible.get_character(name, settings)
    canon = char.name if char else name
    refs_dir = settings.cache_dir("refs")
    slug = _slug(canon)

    srcs = reference_sources(name, settings)
    if srcs:                       # explicit (trusted) sources — may be several to synthesize from
        keep_n = int(limit) if limit is not None else len(srcs)
        out: list[Path] = []
        for src in srcs:
            if len(out) >= keep_n:
                break
            member = refs_dir / f"{slug}.set{len(out)}.png"
            p = member if member.exists() else _fetch_to(src, settings, member)
            if p:
                out.append(p)
        return out

    sc = references_config(settings).get("search", {})
    if not (sc.get("enabled") and client is not None):
        return []
    keep = int(limit if limit is not None else sc.get("max_keep", 3))
    # Reuse already-searched + verified images so a re-run NEVER re-pays the expensive search/verify
    # loop (audit H1). Clear data/cache/refs/<slug>.set*.png to force a fresh search.
    existing = sorted(refs_dir.glob(f"{slug}.set*.png"),
                      key=lambda p: (int(m.group(1)) if (m := re.search(r"set(\d+)$", p.stem)) else 0))
    # '.searched' marks a COMPLETED search (even one that found nothing) so a fruitless search is
    # never re-paid on later runs. Delete data/cache/refs/<slug>.* to force a fresh search.
    if existing or (refs_dir / f"{slug}.searched").exists():
        if existing:
            return existing[:keep]
        # Reuse-check robust to a base-name-only cache: an earlier resolve_reference materialized
        # the verified design as the canonical `{slug}.png` (no set*.png remains). Reuse it as the
        # anchor instead of re-treating the character as "searched, found nothing" (audit BUG 1).
        base = refs_dir / f"{slug}.png"
        if base.exists():
            return [base]
        return []
    if sc.get("agentic"):      # subagent-style iterate-and-verify loop (most reliable)
        return _agentic_search_urls(name, settings, client, tracker, keep)
    # else: single-shot search + page-scrape + per-URL verify
    desc = char.descriptor if char else canon
    verify = bool(sc.get("verify", True))
    min_bytes = int(sc.get("min_bytes", 3000))

    # Verify every candidate (capped at 2*keep so a flood of hits can't blow the budget), then
    # PREFER LARGER images: keep the `keep` largest by pixel area, so a full-size official design
    # wins over a thumbnail when both verify (audit BUG 2). (The agentic path stops at the first
    # `want` verified images and does not re-rank by size — threading "prefer largest" through the
    # model-driven tool loop would be invasive; this single-shot path is where it applies.)
    verified: list[tuple[int, Path]] = []   # (pixel_area, staged_path)
    try:
        for i, u in enumerate(search_reference_urls(name, settings, client, tracker)):
            if len(verified) >= max(keep * 2, keep):
                break
            stage = refs_dir / f"{slug}.cand{i}.png"
            p = _fetch_to(u, settings, stage, min_bytes=min_bytes)
            if p is None:
                continue
            if verify and not _verify_match(p.read_bytes(), canon, desc, sc, client, tracker):
                p.unlink(missing_ok=True)
                continue
            try:
                with Image.open(p) as im:
                    area = im.size[0] * im.size[1]
            except Exception:
                area = 0
            verified.append((area, p))
            print(f"[refs] {canon}: verified official design <{u[:64]}> ({area}px)", file=sys.stderr)
    except BudgetExceeded:
        # A verify-time budget breach propagates out of resolve_reference_set WITHOUT marking the
        # search complete. The just-fetched plus earlier-verified-but-not-yet-materialized
        # `{slug}.cand*.png` staging files would otherwise be left as orphaned disk litter (they
        # don't match the set*.png reuse glob, so they're harmless but accumulate). Sweep them.
        for f in refs_dir.glob(f"{slug}.cand*.png"):
            f.unlink(missing_ok=True)
        raise
    # Largest first; materialize the top `keep` as the canonical set*.png and drop the rest.
    verified.sort(key=lambda t: t[0], reverse=True)
    kept: list[Path] = []
    for area, p in verified[:keep]:
        member = refs_dir / f"{slug}.set{len(kept)}.png"
        p.replace(member)
        kept.append(member)
    for _area, p in verified[keep:]:
        p.unlink(missing_ok=True)
    (refs_dir / f"{slug}.searched").write_text("", encoding="utf-8")   # cache the outcome (incl. empty)
    return kept


def resolve_reference(name: str, settings: Settings, client=None, tracker=None) -> Path | None:
    """Return ONE validated reference Path for the character (or None). Convenience wrapper over
    resolve_reference_set — explicit source, else the first verified online-searched design."""
    char = bible.get_character(name, settings)
    canon = char.name if char else name
    dest = settings.cache_dir("refs") / f"{_slug(canon)}.png"
    if dest.exists():
        return dest
    src = reference_source(name, settings)
    if src:
        return _fetch_to(src, settings, dest)
    members = resolve_reference_set(name, settings, client, tracker, limit=1)
    if members:
        # COPY (not move) so the canonical `{slug}.set*.png` survives on disk for reuse by a
        # later resolve_reference_set (charsheet stage). Moving would rename set0 -> base and
        # leave the set*.png glob empty, so a re-run that sees `.searched` would treat the
        # character as "searched, found nothing" and leave it UNANCHORED (audit BUG 1). HARDLINK
        # (not byte-copy): set0 is a write-once content-addressed cache file, so the canonical
        # `{slug}.png` can share its inode (zero extra bytes) while both paths still exist.
        _link_or_copy(members[0], dest)   # promote to the canonical single-ref path
        return dest
    return None
