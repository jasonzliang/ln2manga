from PIL import Image, ImageDraw

from ln2manga.artifacts import DialogueLine, PageLayout, PanelSpec, Rect
from ln2manga.stages import lettering


def test_wrap_breaks_long_text():
    img = Image.new("L", (400, 400), 255)
    draw = ImageDraw.Draw(img)
    from ln2manga.stages.lettering import _font
    font = _font(None, 16)
    lines = lettering._wrap(draw, "word " * 40, font, max_w=120)
    assert len(lines) > 1


def test_fit_text_shrinks_to_fit(settings):
    img = Image.new("L", (400, 400), 255)
    draw = ImageDraw.Draw(img)
    fp = lettering._font_path(settings, "dialogue_font")
    lines, font, tw, th, lh = lettering._fit_text(
        draw, "The quick brown fox jumps over the lazy dog again and again.",
        fp, max_w=160, max_h=160, max_font=38, min_font=14)
    assert tw <= 160 + 1
    assert lines


def test_letter_panel_runs_without_error(settings):
    img = Image.new("L", (800, 600), 255)
    draw = ImageDraw.Draw(img)
    rect = Rect(x0=20, y0=20, x1=780, y1=580)
    spec = PanelSpec(panel_number=1, speaker_side="right",
                     dialogue=[DialogueLine(speaker="Subaru", text="I won't give up!",
                                            style="shout"),
                               DialogueLine(speaker=None, text="He clenched his fists.",
                                            style="narration")])
    lettering.letter_panel(draw, rect, spec, settings)  # should not raise
    # something was drawn (page no longer all-white)
    assert img.getextrema()[0] < 255


def test_wrap_hard_breaks_unbreakable_word():
    img = Image.new("L", (400, 400), 255)
    draw = ImageDraw.Draw(img)
    from ln2manga.stages.lettering import _font
    font = _font(None, 16)
    word = "Supercalifragilisticexpialidocious"
    max_w = 80
    lines = lettering._wrap(draw, word, font, max_w=max_w)
    # the over-long token must be split across multiple lines, each within max_w
    assert len(lines) > 1
    for ln in lines:
        assert draw.textlength(ln, font=font) <= max_w


def test_fit_text_caps_width_for_long_token(settings):
    img = Image.new("L", (400, 400), 255)
    draw = ImageDraw.Draw(img)
    fp = lettering._font_path(settings, "dialogue_font")
    # an unbreakable token far wider than max_w must still produce lines that fit
    lines, font, tw, th, lh = lettering._fit_text(
        draw, "Supercalifragilisticexpialidocious", fp,
        max_w=80, max_h=200, max_font=38, min_font=14)
    assert tw <= 80 + 1
    assert lines


def test_long_word_bubble_stays_inside_panel(settings):
    img = Image.new("L", (800, 600), 255)
    draw = ImageDraw.Draw(img)
    # a narrow panel that cannot fit the token on one line
    rect = Rect(x0=50, y0=50, x1=180, y1=400)
    spec = PanelSpec(panel_number=1, speaker_side="left",
                     dialogue=[DialogueLine(speaker="A",
                                            text="Supercalifragilisticexpialidocious",
                                            style="speech")])
    lettering.letter_panel(draw, rect, spec, settings)
    # no ink may be drawn outside the panel rect (would mean bubble bled into
    # the gutter / neighboring art). getbbox() on an "L" image boxes non-black
    # pixels, so invert first to box the drawn (dark) ink.
    from PIL import ImageChops
    bbox = ImageChops.invert(img).getbbox()  # bounding box of non-white ink
    assert bbox is not None
    left, upper, right, lower = bbox
    assert left >= rect.x0
    assert right <= rect.x1
    assert upper >= rect.y0
    assert lower <= rect.y1


def test_bubbles_do_not_overlap_vertically(settings):
    img = Image.new("L", (800, 1200), 255)
    draw = ImageDraw.Draw(img)
    rect = Rect(x0=20, y0=20, x1=780, y1=1180)
    spec = PanelSpec(panel_number=1, speaker_side="left",
                     dialogue=[DialogueLine(speaker="A", text="First line here.",
                                            style="speech"),
                               DialogueLine(speaker="B", text="Second line here.",
                                            style="speech")])

    # capture the (cy, bubble_h, tail_target) of each bubble that gets drawn
    drawn = []
    real_draw_bubble = lettering._draw_bubble

    def spy_draw_bubble(d, cx, cy, tw, th, style, tail_target):
        drawn.append((cy, th, tail_target))
        return real_draw_bubble(d, cx, cy, tw, th, style, tail_target)

    lettering._draw_bubble = spy_draw_bubble
    try:
        lettering.letter_panel(draw, rect, spec, settings)
    finally:
        lettering._draw_bubble = real_draw_bubble

    assert len(drawn) == 2
    (cy0, th0, tail0), (cy1, th1, tail1) = drawn
    # bottom-most extent of the first bubble: its tail tip if tailed, else body
    body_bottom0 = cy0 + th0 / 2 + lettering.PAD
    first_bottom = max(body_bottom0, tail0[1] if tail0 else body_bottom0)
    # top of the second bubble's body
    second_top = cy1 - (th1 / 2 + lettering.PAD)
    assert second_top >= first_bottom


def test_stacked_bubbles_in_short_panel_never_overlap(settings):
    # BUG 10 regression: in a SHORT panel a later (stacked) bubble must never be
    # pulled UP past the previous bubble's bottom (body OR tail) by the vertical-fit
    # clamp. A second bubble that does not fit below the first must be SKIPPED, not
    # nudged up over it. The dimensions/fonts below deterministically force the
    # second bubble to overflow the panel bottom — exactly the clamp path the bug
    # lived in. With the fix the second bubble is dropped; the old code dragged it
    # up over the first bubble's tail.
    settings.lettering = dict(settings.lettering)
    settings.lettering["min_font"] = 34   # can't shrink -> 2nd bubble overflows
    settings.lettering["max_font"] = 40

    img = Image.new("L", (300, 240), 255)
    draw = ImageDraw.Draw(img)
    rect = Rect(x0=20, y0=20, x1=280, y1=220)   # short panel, two bubbles won't both fit
    spec = PanelSpec(panel_number=1, speaker_side="left",
                     dialogue=[DialogueLine(speaker="A",
                                            text="First short line here for the panel.",
                                            style="speech"),
                               DialogueLine(speaker="B",
                                            text="This second line is long enough to "
                                                 "wrap into several rows here.",
                                            style="speech")])

    drawn = []
    real_draw_bubble = lettering._draw_bubble

    def spy_draw_bubble(d, cx, cy, tw, th, style, tail_target):
        rx, ry = lettering._bubble_size(tw, th, style)
        drawn.append((cy, ry, tail_target))
        return real_draw_bubble(d, cx, cy, tw, th, style, tail_target)

    lettering._draw_bubble = spy_draw_bubble
    try:
        lettering.letter_panel(draw, rect, spec, settings)
    finally:
        lettering._draw_bubble = real_draw_bubble

    # at least the first bubble draws; the second may legitimately be skipped.
    assert 1 <= len(drawn) <= 2
    if len(drawn) == 2:
        (cy0, ry0, tail0), (cy1, ry1, _tail1) = drawn
        # bottom-most extent of the first bubble: its tail tip if tailed, else body.
        first_body_bottom = cy0 + ry0
        first_bottom = max(first_body_bottom, tail0[1] if tail0 else first_body_bottom)
        second_top = cy1 - ry1
        # the second bubble must sit fully BELOW the first (never nudged up over it).
        assert second_top >= first_bottom, (
            f"second bubble top {second_top} is above first bubble bottom "
            f"{first_bottom} (overlap)")


def test_stacked_bubbles_drop_second_when_no_room(settings):
    # Companion to the BUG 10 regression: under the same deterministic, too-short
    # panel the fix must DROP the second bubble rather than pull it up. (Pinning
    # the skip makes the regression unambiguous: the old code drew two and
    # overlapped; the fix draws one.)
    settings.lettering = dict(settings.lettering)
    settings.lettering["min_font"] = 34
    settings.lettering["max_font"] = 40

    img = Image.new("L", (300, 240), 255)
    draw = ImageDraw.Draw(img)
    rect = Rect(x0=20, y0=20, x1=280, y1=220)
    spec = PanelSpec(panel_number=1, speaker_side="left",
                     dialogue=[DialogueLine(speaker="A",
                                            text="First short line here for the panel.",
                                            style="speech"),
                               DialogueLine(speaker="B",
                                            text="This second line is long enough to "
                                                 "wrap into several rows here.",
                                            style="speech")])

    drawn = []
    real_draw_bubble = lettering._draw_bubble

    def spy_draw_bubble(d, cx, cy, tw, th, style, tail_target):
        drawn.append(cy)
        return real_draw_bubble(d, cx, cy, tw, th, style, tail_target)

    lettering._draw_bubble = spy_draw_bubble
    try:
        lettering.letter_panel(draw, rect, spec, settings)
    finally:
        lettering._draw_bubble = real_draw_bubble

    assert len(drawn) == 1, "second bubble must be skipped, not nudged up over the first"


def _run_organic(settings, interior, fit_return=None):
    """Drive letter_panel's ORGANIC branch with a forced `interior` Rect (and an
    optional forced `_fit_text` return) and capture what `_draw_text_block` got."""
    from ln2manga import bubbles

    img = Image.new("L", (800, 600), 255)
    draw = ImageDraw.Draw(img)
    rect = Rect(x0=20, y0=20, x1=780, y1=580)
    spec = PanelSpec(panel_number=1, speaker_side="left",
                     dialogue=[DialogueLine(
                         speaker="A",
                         text=("This is a long line of dialogue that wants to wrap "
                               "across many rows but the interior is very short."),
                         style="speech")])

    captured = {}
    real_place = bubbles.place_bubble
    real_block = lettering._draw_text_block
    real_fit = lettering._fit_text

    def fake_place(page, shape_path, cx, cy, bw, bh):
        return interior

    def spy_block(d, lines, font, cx, top, line_h):
        captured["lines"] = list(lines)
        captured["top"] = top
        captured["line_h"] = line_h
        return real_block(d, lines, font, cx, top, line_h)

    # Force the INTERIOR _fit_text call (the second positional arg differs: the
    # interior pass receives `iw` as max_w) to return a multi-line, tall result so
    # the ilh > ih clamp branch is exercised; all other calls use the real fitter.
    def maybe_fake_fit(d, text, font_path, max_w, max_h, max_font, min_font):
        if fit_return is not None and max_w <= interior.w:
            return fit_return(real_fit, d, text, font_path)
        return real_fit(d, text, font_path, max_w, max_h, max_font, min_font)

    shape_lib = {"speech": ["dummy_shape.png"]}
    page_live = Image.new("L", (800, 600), 255)

    bubbles.place_bubble = fake_place
    lettering._draw_text_block = spy_block
    if fit_return is not None:
        lettering._fit_text = maybe_fake_fit
    try:
        lettering.letter_panel(draw, rect, spec, settings,
                               page_img=page_live, page_live=page_live,
                               shape_lib=shape_lib)
    finally:
        bubbles.place_bubble = real_place
        lettering._draw_text_block = real_block
        lettering._fit_text = real_fit
    return captured


def test_organic_short_interior_single_line_not_stacked(settings):
    # BUG 7 regression: when the shape's interior is shorter than one line of text
    # (ilh > ih), the ORGANIC branch must draw a SINGLE (possibly clipped) line and
    # never stack multiple lines below — which would spill far past the interior.
    short_interior = Rect(x0=300, y0=300, x1=560, y1=312)   # h=12 -> ih=max(8,4)=8

    # force the interior fitter to hand back a TALL, multi-line block: line_h far
    # exceeds the interior height and there are several lines. Without the clamp the
    # whole multi-line block (line_h * N) would be drawn, overflowing the interior.
    def tall_multiline(real_fit, d, text, font_path):
        font = lettering._font(font_path, 30)
        lines = ["First", "Second", "Third", "Fourth"]
        line_h = 35                                   # >> ih (=8): one line won't fit
        tw = max(d.textlength(l, font=font) for l in lines)
        return lines, font, tw, line_h * len(lines), line_h

    captured = _run_organic(settings, short_interior, fit_return=tall_multiline)
    assert captured.get("lines"), "no text drawn in organic branch"
    ih = max(8, short_interior.h - 8)
    # ilh (35) > ih (8): the fix keeps exactly ONE line (never the 4-line stack).
    assert len(captured["lines"]) == 1, (
        f"expected a single clipped line for a too-short interior, got "
        f"{captured['lines']}")
    # the total drawn block must equal one line height, not N * line_h.
    assert captured["line_h"] * len(captured["lines"]) == captured["line_h"]


def test_organic_text_fits_normal_interior(settings):
    # Companion: with a roomy interior the drawn block must never exceed the
    # interior height (the normal imax-capped path).
    interior = Rect(x0=200, y0=200, x1=560, y1=360)   # h=160 -> ih=152
    captured = _run_organic(settings, interior)
    assert captured.get("lines"), "no text drawn in organic branch"
    ih = max(8, interior.h - 8)
    drawn_block_h = captured["line_h"] * len(captured["lines"])
    assert drawn_block_h <= ih, (
        f"organic text block height {drawn_block_h} exceeds interior height {ih}")


def test_organic_text_block_stays_within_bottom_margin(settings):
    # DEFECT 1 regression: a multi-line SPEECH bubble's drawn text block must end
    # STRICTLY ABOVE the reserved bottom margin (interior.y1 - margin), so the last
    # line's descenders clear the rounded bottom + tail notch instead of crossing the
    # bubble outline onto the artwork. Centering happens in the REDUCED region.
    interior = Rect(x0=200, y0=200, x1=560, y1=400)   # tall -> multi-line wrap
    captured = _run_organic(settings, interior)
    assert captured.get("lines"), "no text drawn in organic branch"
    assert len(captured["lines"]) > 1, "test needs a multi-line block to be meaningful"
    # the helper's dialogue is style="speech" -> the larger speech bottom reserve applies.
    margin = round(interior.h * lettering._ORGANIC_BOTTOM_MARGIN_FRAC_SPEECH)
    assert margin > 0, "speech must reserve a non-zero bottom margin"
    block_bottom = captured["top"] + captured["line_h"] * len(captured["lines"])
    assert block_bottom <= interior.y1 - margin, (
        f"text block bottom {block_bottom} crosses the reserved interior bottom "
        f"{interior.y1 - margin} (margin {margin})")
    # speech must reserve MORE at the bottom than narration/box (it has a tail).
    assert (lettering._ORGANIC_BOTTOM_MARGIN_FRAC_SPEECH
            > lettering._ORGANIC_BOTTOM_MARGIN_FRAC)


def test_short_panel_three_speech_lines_all_render(settings):
    # DEFECT 2 regression: a reasonably-sized SHORT panel with 3 speech bubbles must
    # stack all 3 without the geometric stacking-skip dropping the 3rd. The organic
    # path is used (where the over-tall _ORGANIC_INFLATE caused the real-chapter
    # skips); a fake place_bubble returns a plausible inner interior. With the reduced
    # inflate all 3 fit; an inflated bubble (the old 1.3) would skip the 3rd here.
    from ln2manga import bubbles

    rect = Rect(x0=20, y0=20, x1=480, y1=540)   # short panel, 460x520 usable
    dialogue = [DialogueLine(speaker="A", text="Hi there!", style="speech"),
                DialogueLine(speaker="B", text="Yes, what is it?", style="speech"),
                DialogueLine(speaker="C", text="Okay, let us go.", style="speech")]
    img = Image.new("L", (520, 580), 255)
    draw = ImageDraw.Draw(img)
    page = Image.new("L", (520, 580), 255)
    spec = PanelSpec(panel_number=1, speaker_side="left", dialogue=dialogue)

    blocks = []
    real_block = lettering._draw_text_block
    real_place = bubbles.place_bubble

    def spy_block(d, lines, font, cx, top, line_h):
        blocks.append(top)
        return real_block(d, lines, font, cx, top, line_h)

    def fake_place(pg, shape_path, cx, cy, bw, bh):
        # plausible interior: inner 80% of the requested bubble box.
        return Rect(x0=int(cx - bw * 0.4), y0=int(cy - bh * 0.4),
                    x1=int(cx + bw * 0.4), y1=int(cy + bh * 0.4))

    lettering._draw_text_block = spy_block
    bubbles.place_bubble = fake_place
    dropped: list[str] = []
    try:
        lettering.letter_panel(draw, rect, spec, settings, page_img=page,
                               page_live=page, shape_lib={"speech": ["x.png"]},
                               dropped=dropped)
    finally:
        lettering._draw_text_block = real_block
        bubbles.place_bubble = real_place

    assert len(blocks) == 3, f"expected all 3 speech bubbles to render, got {len(blocks)}"
    assert dropped == [], f"nothing should drop in a panel sized for 3, got {dropped}"


def test_drop_protects_narration_over_excess_speech(settings):
    # DEFECT 2 (drop priority): in a panel that can hold only 2 of its 3 lines, the
    # NARRATION/caption must be protected — a lower-priority SPEECH line is dropped
    # instead, even though the narration is LAST in reading order (so a naive
    # reading-order placement would have dropped it). The KEPT lines render in their
    # original on-page reading order (speech above narration).
    rect = Rect(x0=20, y0=20, x1=400, y1=340)   # holds only 2 of the 3 lines
    dialogue = [
        DialogueLine(speaker="A", text="What was that noise just now?", style="speech"),
        DialogueLine(speaker="B", text="I really cannot say for certain.", style="speech"),
        DialogueLine(speaker=None, text="The room fell silent.", style="narration")]
    img = Image.new("L", (440, 380), 255)
    draw = ImageDraw.Draw(img)
    spec = PanelSpec(panel_number=1, speaker_side="left", dialogue=dialogue)

    seq = []   # (cy, style) of each drawn bubble, in draw order
    real = lettering._draw_bubble

    def spy(d, cx, cy, tw, th, style, tail_target):
        seq.append((cy, style))
        return real(d, cx, cy, tw, th, style, tail_target)

    lettering._draw_bubble = spy
    dropped: list[str] = []
    try:
        lettering.letter_panel(draw, rect, spec, settings, dropped=dropped)
    finally:
        lettering._draw_bubble = real

    styles = [st for _cy, st in seq]
    assert "narration" in styles, "the narration/caption must be kept, not dropped"
    assert styles.count("speech") == 1, "exactly one speech line should be kept"
    # a SPEECH line (not the narration) was the one dropped.
    assert len(dropped) == 1 and dropped[0] in (
        "What was that noise just now?", "I really cannot say for certain.")
    # reading order of the KEPT lines is preserved: the speech (earlier) draws ABOVE
    # the narration (later) -> smaller cy first.
    assert seq == sorted(seq), f"kept lines not in reading order: {seq}"
    assert styles[0] == "speech" and styles[-1] == "narration"


def test_center_speaker_draws_no_tail(settings):
    # 'center' (the schema default) must not get a misleading straight-down tail.
    # Compare ink against an identical 'right' panel which DOES get a tail.
    rect = Rect(x0=20, y0=20, x1=780, y1=580)
    line = DialogueLine(speaker="A", text="Hello there friend.", style="speech")

    img_center = Image.new("L", (800, 600), 255)
    lettering.letter_panel(ImageDraw.Draw(img_center),
                           rect, PanelSpec(panel_number=1, speaker_side="center",
                                           dialogue=[line]), settings)

    img_right = Image.new("L", (800, 600), 255)
    lettering.letter_panel(ImageDraw.Draw(img_right),
                           rect, PanelSpec(panel_number=1, speaker_side="right",
                                           dialogue=[line]), settings)

    # the tailed (right) bubble must extend lower than the tail-less (center) one.
    # invert so getbbox() boxes the drawn (dark) ink rather than non-black pixels.
    from PIL import ImageChops
    assert ImageChops.invert(img_center).getbbox()[3] < ImageChops.invert(img_right).getbbox()[3]


def _spy_bubbles(monkeyable_settings, draw, rect, spec, page_img):
    """Run letter_panel capturing each (cx, cy, tw, th, style) the bubble drawer got."""
    drawn = []
    real = lettering._draw_bubble

    def spy(d, cx, cy, tw, th, style, tail_target):
        drawn.append((cx, cy, tw, th, style))
        return real(d, cx, cy, tw, th, style, tail_target)

    lettering._draw_bubble = spy
    try:
        lettering.letter_panel(draw, rect, spec, monkeyable_settings, page_img=page_img)
    finally:
        lettering._draw_bubble = real
    return drawn


def test_bubble_avoids_dark_face_block(settings):
    # synthetic page: a panel rect with a dark "face" block in the UPPER-CENTER.
    # the content-aware placement must drop the bubble into an empty (lower) band.
    page = Image.new("L", (800, 600), 255)
    pdraw = ImageDraw.Draw(page)
    rect = Rect(x0=20, y0=20, x1=780, y1=580)
    # face block: upper-center of the panel, ~top third, centered horizontally.
    fx0, fy0, fx1, fy1 = 300, 40, 500, 260
    pdraw.rectangle([fx0, fy0, fx1, fy1], fill=0)

    img = Image.new("L", (800, 600), 255)
    draw = ImageDraw.Draw(img)
    spec = PanelSpec(panel_number=1, speaker_side="right", shot_type="close_up",
                     dialogue=[DialogueLine(speaker="A", text="Look out behind you!",
                                            style="speech")])
    drawn = _spy_bubbles(settings, draw, rect, spec, page)
    assert len(drawn) == 1
    cx, cy, tw, th, style = drawn[0]
    rx, ry = lettering._bubble_size(tw, th, style)
    # the bubble's bounding box must NOT overlap the face block.
    b_left, b_right = cx - rx, cx + rx
    b_top, b_bottom = cy - ry, cy + ry
    overlaps = not (b_right <= fx0 or b_left >= fx1 or b_bottom <= fy0 or b_top >= fy1)
    assert not overlaps, f"bubble {(b_left, b_top, b_right, b_bottom)} overlaps face {(fx0, fy0, fx1, fy1)}"


def _small_face_page():
    """A page whose EMPTIEST full-width rows are the MIDDLE band (dense top + dense
    bottom), with a SMALL off-center face inside that middle band. The naive
    emptiest-ROW heuristic averages the small face across the wide middle row and so
    its full-width band sits ON the face; a footprint-scored placement can dodge
    sideways within the band to the empty side and miss the face entirely.
    Returns (page, rect, face_box)."""
    page = Image.new("L", (800, 600), 255)
    pd = ImageDraw.Draw(page)
    rect = Rect(x0=20, y0=20, x1=780, y1=580)
    pd.rectangle([40, 40, 760, 180], fill=0)     # dense TOP band
    pd.rectangle([40, 400, 760, 560], fill=0)    # dense BOTTOM band
    face = (120, 250, 230, 360)                  # small face in the empty MIDDLE, left of center
    pd.rectangle(list(face), fill=0)
    return page, rect, face


def _overlaps(box, face):
    bl, bt, br, bb = box
    fx0, fy0, fx1, fy1 = face
    return not (br <= fx0 or bl >= fx1 or bb <= fy0 or bt >= fy1)


def test_footprint_avoids_small_face_naive_band_would_bury():
    # REGRESSION: a SMALL off-center face (chibi / secondary / background) sits in the
    # panel's emptiest ROW band. The naive emptiest-row heuristic returns a FULL-WIDTH
    # band that vertically overlaps the face -> it would BURY the face. The footprint
    # scorer must dodge SIDEWAYS within the band and choose a placement that does NOT
    # overlap the small face.
    page, rect, face = _small_face_page()
    fx0, fy0, fx1, fy1 = face
    bubble_w, bubble_h = 300, 110

    # naive emptiest-ROW band: full-width, so it covers the face if it overlaps it vertically.
    band_top, band_dens = lettering._emptiest_band(page, rect, bubble_h)
    assert band_top is not None
    naive_box = (rect.x0, band_top, rect.x1, band_top + bubble_h)
    assert _overlaps(naive_box, face), (
        "test precondition: the naive full-width emptiest-row band must cover the small "
        f"face (band rows {band_top}-{band_top + bubble_h} vs face rows {fy0}-{fy1})")

    # footprint-scored placement must AVOID the small face.
    cx, top, ink = lettering._best_footprint(page, rect, bubble_w, bubble_h,
                                             prefer_side="left")
    assert cx is not None
    fp_box = (cx - bubble_w / 2, top, cx + bubble_w / 2, top + bubble_h)
    assert not _overlaps(fp_box, face), (
        f"footprint {fp_box} buried the small face {face} that it should have dodged "
        f"sideways (naive band would have covered it)")


def test_small_face_penalty_steers_off_concentrated_cluster():
    # The small-face penalty must measurably change scoring: with the penalty disabled
    # the scorer may cover MORE of the small concentrated cluster than with it enabled.
    # Assert the production (penalty-on) footprint covers no MORE cluster ink than the
    # penalty-off footprint, and avoids the face outright here.
    import numpy as np

    page, rect, face = _small_face_page()
    fx0, fy0, fx1, fy1 = face
    fm = np.zeros((page.height, page.width), dtype=np.float32)
    fm[fy0:fy1, fx0:fx1] = 1.0
    bubble_w, bubble_h = 300, 110

    def covered(penalty):
        saved = lettering._SMALL_FACE_PENALTY
        lettering._SMALL_FACE_PENALTY = penalty
        try:
            cx, top, _ = lettering._best_footprint(page, rect, bubble_w, bubble_h,
                                                   prefer_side="left")
        finally:
            lettering._SMALL_FACE_PENALTY = saved
        bl, bt = int(cx - bubble_w / 2), int(top)
        return int(fm[bt:bt + bubble_h, bl:bl + bubble_w].sum())

    off = covered(0.0)
    on = covered(lettering._SMALL_FACE_PENALTY)
    assert on <= off, (
        f"penalty-on footprint covered MORE small-face ink ({on}) than penalty-off ({off})")
    assert on == 0, f"production placement should fully avoid the small face, covered {on}px"


def test_letter_panel_first_bubble_avoids_small_face(settings):
    # END-TO-END: through letter_panel, the FIRST speech bubble of a close-up must not
    # overlap a SMALL off-center face that lives in the panel's emptiest row band (where
    # the naive full-width emptiest-row band would have buried it).
    page, rect, face = _small_face_page()
    fx0, fy0, fx1, fy1 = face
    img = Image.new("L", (800, 600), 255)
    draw = ImageDraw.Draw(img)
    spec = PanelSpec(panel_number=1, speaker_side="right", shot_type="close_up",
                     dialogue=[DialogueLine(speaker="A", text="Look out behind you!",
                                            style="speech")])
    drawn = _spy_bubbles(settings, draw, rect, spec, page)
    assert drawn, "no bubble drawn"
    cx, cy, tw, th, style = drawn[0]
    rx, ry = lettering._bubble_size(tw, th, style)
    box = (cx - rx, cy - ry, cx + rx, cy + ry)
    assert not _overlaps(box, face), (
        f"first bubble {box} buried the small off-center face {face}")


def test_large_face_not_treated_as_small_cluster():
    # GUARDRAIL: a LARGE dominant face (fills much of the panel, the close-up case) must
    # NOT be flagged wholesale as a "small face" cluster — that regime is handled by the
    # existing full-face bottom-anchor fallback, not by the small-cluster penalty. The
    # cluster mask must therefore flag only a SMALL fraction of a big solid face's cells
    # (its wide surround is just as dark, so it fails the concentration test), while a
    # genuinely small blob is flagged in full.
    import numpy as np

    def cluster_fraction(face_side):
        page = Image.new("L", (800, 600), 255)
        ImageDraw.Draw(page).rectangle([200, 120, 200 + face_side, 120 + face_side], fill=0)
        rect = Rect(x0=20, y0=20, x1=780, y1=580)
        # drive the detector through _best_footprint and read back via the mask it builds:
        # reconstruct the same mask the function uses to assert on its selectivity.
        crop = np.asarray(page.crop((20, 20, 780, 580)), dtype=np.float32)[::4, ::4]
        ink = (crop < 128).astype(np.float32)
        sh, sw = ink.shape
        sat = np.zeros((sh + 1, sw + 1), dtype=np.float32)
        sat[1:, 1:] = ink.cumsum(0).cumsum(1)
        win = min(sh, sw, max(1, round(lettering._SMALL_FACE_WIN_FRAC * min(sh, sw))))
        surround = min(min(sh, sw), max(win + 1, round(win * 2.5)))

        def wm(side):
            half = side // 2
            ri, ci = np.arange(sh), np.arange(sw)
            r0 = np.clip(ri - half, 0, sh); r1 = np.clip(ri - half + side, 0, sh)
            c0 = np.clip(ci - half, 0, sw); c1 = np.clip(ci - half + side, 0, sw)
            R0, C0 = np.meshgrid(r0, c0, indexing="ij")
            R1, C1 = np.meshgrid(r1, c1, indexing="ij")
            area = np.maximum(1, (R1 - R0) * (C1 - C0)).astype(np.float32)
            return (sat[R1, C1] - sat[R0, C1] - sat[R1, C0] + sat[R0, C0]) / area

        local, wide = wm(win), wm(surround)
        cluster = ((local >= lettering._SMALL_FACE_LOCAL_INK)
                   & (local >= lettering._SMALL_FACE_CONCENTRATION * wide)
                   & (ink > 0))
        ink_cells = int(ink.sum())
        return (int(cluster.sum()) / ink_cells) if ink_cells else 0.0

    small_frac = cluster_fraction(60)    # ~60px chibi face
    large_frac = cluster_fraction(300)   # ~300px dominant face
    assert small_frac > 0.8, f"small face under-detected as a cluster: {small_frac:.2f}"
    assert large_frac < 0.2, f"large face wrongly flagged as a small cluster: {large_frac:.2f}"
    assert large_frac < small_frac


def test_shot_aware_max_h():
    # tighter shots get a smaller per-bubble vertical budget than wider ones.
    assert lettering._shot_max_h_frac("wide") == 0.45
    assert lettering._shot_max_h_frac("medium") == 0.40
    assert lettering._shot_max_h_frac("close_up") == 0.30
    assert lettering._shot_max_h_frac("extreme_close_up") == 0.24
    # unknown / missing shot falls back to the medium default.
    assert lettering._shot_max_h_frac(None) == 0.40
    assert lettering._shot_max_h_frac("nonsense") == 0.40
    assert (lettering._shot_max_h_frac("extreme_close_up")
            < lettering._shot_max_h_frac("close_up")
            < lettering._shot_max_h_frac("medium")
            < lettering._shot_max_h_frac("wide"))


def test_emptiest_band_finds_clear_region():
    # a page with a dark block up top: the emptiest band of a short bubble must land
    # below the block (larger top-y), not on it.
    page = Image.new("L", (400, 400), 255)
    ImageDraw.Draw(page).rectangle([100, 10, 300, 150], fill=0)  # dark upper block
    rect = Rect(x0=0, y0=0, x1=400, y1=400)
    top_y, density = lettering._emptiest_band(page, rect, bubble_h=80)
    assert top_y is not None
    # the chosen band must sit below the dark block, and be near-empty.
    assert top_y >= 150 - 80  # band does not overlap the block's lower edge meaningfully
    assert density < lettering._DENSE_BAND


def test_abridge_long_line_fits_budget_at_word_boundary():
    # a long multi-sentence line must be abridged to <= budget at a sentence/word
    # boundary, never mid-word, with no overflow past the budget.
    text = ("I will protect you no matter what happens to either of us. "
            "Even if it costs me everything, I refuse to back down now. "
            "So please, just trust me on this one thing.")
    budget = 60
    out = lettering._abridge(text, budget)
    assert len(out) <= budget
    # result is non-empty and not the full original (it was shortened)
    assert out and len(out) < len(text)
    # every word in the result (minus a trailing ellipsis) is a real word from the
    # source -> no word was split mid-token.
    src_words = set(text.replace(",", " ").replace(".", " ").split())
    for w in out.rstrip("…").split():
        assert w.strip(".,;:…") in src_words


def test_abridge_short_line_unchanged():
    short = "Hi there, friend."
    assert lettering._abridge(short, 50) == short
    # leading/trailing/collapsible whitespace is normalized but content unchanged.
    assert lettering._abridge("  Hi there,   friend.  ", 50) == short


def test_abridge_never_splits_a_word():
    text = "Supercalifragilistic expialidocious antidisestablishmentarianism words"
    out = lettering._abridge(text, 25)
    assert len(out) <= 25
    # strip a possible trailing ellipsis, then confirm each token is intact.
    tokens = out.rstrip("…").split()
    src = set(text.split())
    for tok in tokens:
        assert tok in src, f"{tok!r} is not a whole source word"


def test_letter_panel_abridges_overlong_dialogue(settings):
    # an over-long line in a small bubble must be abridged to coherent text that
    # fits the char budget (no mid-word clip), then drawn without raising.
    img = Image.new("L", (400, 300), 255)
    draw = ImageDraw.Draw(img)
    rect = Rect(x0=20, y0=20, x1=240, y1=200)  # modest bubble
    long_text = ("I have waited my entire life for this single moment to arrive. "
                 "Now that it finally has, I will not let anyone stand in my way. "
                 "Remember everything that I taught you, alright?")
    spec = PanelSpec(panel_number=1, speaker_side="right",
                     dialogue=[DialogueLine(speaker="A", text=long_text, style="speech")])

    drawn_text = {}
    real_block = lettering._draw_text_block

    def spy(d, lines, font, cx, top, line_h):
        drawn_text["lines"] = list(lines)
        return real_block(d, lines, font, cx, top, line_h)

    lettering._draw_text_block = spy
    try:
        lettering.letter_panel(draw, rect, spec, settings)
    finally:
        lettering._draw_text_block = real_block

    assert drawn_text.get("lines"), "no text was drawn"
    rendered = " ".join(drawn_text["lines"])
    # the rendered text is a shortened version of the original (abridged).
    assert len(rendered) < len(long_text)
    # no word in the rendered text was split mid-token (every token, minus a
    # trailing ellipsis, is a whole source word).
    src_words = set(long_text.replace(",", " ").replace(".", " ").split())
    for w in rendered.rstrip("…").split():
        cleaned = w.strip(".,;:…")
        if cleaned:
            assert cleaned in src_words, f"{w!r} looks like a mid-word clip"


def test_letter_panel_backward_compat_no_page_img(settings):
    # no page_img -> keep the original top-anchored behavior, still draws, no crash.
    img = Image.new("L", (800, 600), 255)
    draw = ImageDraw.Draw(img)
    rect = Rect(x0=20, y0=20, x1=780, y1=580)
    spec = PanelSpec(panel_number=1, speaker_side="left",
                     dialogue=[DialogueLine(speaker="A", text="Backward compatible!",
                                            style="speech")])
    lettering.letter_panel(draw, rect, spec, settings)  # no page_img, must not raise
    assert img.getextrema()[0] < 255  # something was drawn

    # the no-page_img bubble stays top-anchored (its body sits in the upper half).
    drawn = []
    real = lettering._draw_bubble

    def spy(d, cx, cy, tw, th, style, tail_target):
        drawn.append(cy)
        return real(d, cx, cy, tw, th, style, tail_target)

    lettering._draw_bubble = spy
    try:
        img2 = Image.new("L", (800, 600), 255)
        lettering.letter_panel(ImageDraw.Draw(img2), rect, spec, settings)
    finally:
        lettering._draw_bubble = real
    assert drawn and drawn[0] < (rect.y0 + rect.y1) / 2


def test_default_font_scales_with_size():
    # Missing-font fallback must honor the requested size (Pillow 10+ load_default(size)),
    # not collapse every size to one tiny ~13px bitmap (which made _fit_text a no-op).
    img = Image.new("L", (200, 200), 255)
    draw = ImageDraw.Draw(img)
    big = lettering._font(None, 38)
    small = lettering._font(None, 15)
    assert draw.textlength("Hello", font=big) > draw.textlength("Hello", font=small)


def test_dropped_dialogue_beyond_cap_is_surfaced(settings):
    # Lines beyond the per-panel cap (default 4) must be reported via `dropped`,
    # not silently lost. No on-page marker is drawn for them.
    img = Image.new("L", (800, 1200), 255)
    draw = ImageDraw.Draw(img)
    rect = Rect(x0=20, y0=20, x1=780, y1=1180)   # tall: all 4 within-cap bubbles fit
    spec = PanelSpec(panel_number=1, speaker_side="right",
                     dialogue=[DialogueLine(speaker="A", text="One.", style="speech"),
                               DialogueLine(speaker="B", text="Two.", style="speech"),
                               DialogueLine(speaker="C", text="Three.", style="speech"),
                               DialogueLine(speaker="D", text="Four.", style="speech"),
                               DialogueLine(speaker="E", text="Five dropped.",
                                            style="speech")])
    dropped: list[str] = []
    lettering.letter_panel(draw, rect, spec, settings, dropped=dropped)
    assert dropped == ["Five dropped."]


def test_max_bubbles_per_panel_is_configurable(settings):
    # The per-panel bubble cap honors settings.lettering["max_bubbles_per_panel"].
    settings.lettering = dict(settings.lettering)
    settings.lettering["max_bubbles_per_panel"] = 2
    img = Image.new("L", (800, 600), 255)
    draw = ImageDraw.Draw(img)
    rect = Rect(x0=20, y0=20, x1=780, y1=580)
    spec = PanelSpec(panel_number=1, speaker_side="right",
                     dialogue=[DialogueLine(speaker="A", text="One.", style="speech"),
                               DialogueLine(speaker="B", text="Two.", style="speech"),
                               DialogueLine(speaker="C", text="Three.", style="speech")])
    dropped: list[str] = []
    lettering.letter_panel(draw, rect, spec, settings, dropped=dropped)
    assert dropped == ["Three."]


def test_dropped_dialogue_on_panel_overflow_is_surfaced(settings):
    # A later stacked bubble that overflows the panel bottom is skipped (never nudged
    # up over the prior bubble) AND its text is reported as dropped.
    settings.lettering = dict(settings.lettering)
    settings.lettering["min_font"] = 34   # can't shrink -> 2nd bubble overflows
    settings.lettering["max_font"] = 40
    img = Image.new("L", (300, 240), 255)
    draw = ImageDraw.Draw(img)
    rect = Rect(x0=20, y0=20, x1=280, y1=220)   # short panel, two bubbles won't both fit
    spec = PanelSpec(panel_number=1, speaker_side="left",
                     dialogue=[DialogueLine(speaker="A",
                                            text="First short line here for the panel.",
                                            style="speech"),
                               DialogueLine(speaker="B",
                                            text="This second line is long enough to "
                                                 "wrap into several rows here.",
                                            style="speech")])
    dropped: list[str] = []
    lettering.letter_panel(draw, rect, spec, settings, dropped=dropped)
    assert len(dropped) == 1
    assert dropped[0].startswith("This second line")


def test_no_dropped_when_all_fit(settings):
    # The happy path reports nothing dropped.
    img = Image.new("L", (800, 600), 255)
    draw = ImageDraw.Draw(img)
    rect = Rect(x0=20, y0=20, x1=780, y1=580)
    spec = PanelSpec(panel_number=1, speaker_side="right",
                     dialogue=[DialogueLine(speaker="A", text="Just one line.",
                                            style="speech")])
    dropped: list[str] = []
    lettering.letter_panel(draw, rect, spec, settings, dropped=dropped)
    assert dropped == []


def test_narration_does_not_disable_face_avoidance(settings):
    # Regression: adding a narration caption must NOT disable the single-speech-bubble
    # face-avoidance band-anchoring (which used to require n_lines == 1, counting the
    # caption). The speech bubble must still drop into an empty band off the face.
    page = Image.new("L", (800, 600), 255)
    pdraw = ImageDraw.Draw(page)
    rect = Rect(x0=20, y0=20, x1=780, y1=580)
    fx0, fy0, fx1, fy1 = 300, 40, 500, 260   # dark face block, upper-center
    pdraw.rectangle([fx0, fy0, fx1, fy1], fill=0)

    img = Image.new("L", (800, 600), 255)
    draw = ImageDraw.Draw(img)
    spec = PanelSpec(panel_number=1, speaker_side="right", shot_type="close_up",
                     dialogue=[DialogueLine(speaker="A", text="Look out behind you!",
                                            style="speech"),
                               DialogueLine(speaker=None, text="He spun around.",
                                            style="narration")])
    drawn = _spy_bubbles(settings, draw, rect, spec, page)
    # the FIRST (speech) bubble must not overlap the face block.
    cx, cy, tw, th, style = drawn[0]
    assert style == "speech"
    rx, ry = lettering._bubble_size(tw, th, style)
    b_left, b_right = cx - rx, cx + rx
    b_top, b_bottom = cy - ry, cy + ry
    overlaps = not (b_right <= fx0 or b_left >= fx1 or b_bottom <= fy0 or b_top >= fy1)
    assert not overlaps, (
        f"speech bubble {(b_left, b_top, b_right, b_bottom)} overlaps face "
        f"{(fx0, fy0, fx1, fy1)} despite a narration caption being present")


def _one_page_chapter(settings, tmp_path, style="shout"):
    """A minimal pages/specs pair (one page, one panel) writing a real page PNG."""
    page_png = tmp_path / "page.png"
    Image.new("L", (800, 600), 255).save(page_png)
    rect = Rect(x0=20, y0=20, x1=780, y1=580)
    page = PageLayout(page_number=1, template="x", panel_numbers=[1], rects=[rect],
                      image_path=str(page_png))
    spec = PanelSpec(panel_number=1, speaker_side="right",
                     dialogue=[DialogueLine(speaker="A", text="Hi!", style=style)])
    return [page], [spec]


def test_run_warns_on_partial_organic_library(settings, tmp_path, monkeypatch, capsys):
    # Regression: when ensure_shape_library returns a PARTIAL dict (a style it could
    # not build is omitted), run() must surface that the style fell back to drawn
    # bubbles instead of silently routing it to the drawn branch.
    settings.lettering = dict(settings.lettering)
    settings.lettering["bubble_style"] = "organic"
    pages, specs = _one_page_chapter(settings, tmp_path, style="shout")

    from ln2manga import bubbles
    # partial library: everything EXCEPT "shout" is available.
    partial = {s: ["dummy.png"] for s in bubbles.STYLES if s != "shout"}
    monkeypatch.setattr(bubbles, "ensure_shape_library",
                        lambda *a, **k: dict(partial))

    lettering.run(settings, pages, specs, chapter_number=1,
                  client=object(), tracker=object(), cache=object())
    err = capsys.readouterr().err
    assert "shout" in err and "WARNING" in err


def test_run_warns_on_dropped_dialogue(settings, tmp_path, capsys):
    # Regression: dialogue dropped because it exceeded the per-panel cap must be
    # reported to stderr at the end of run(), not silently lost.
    page_png = tmp_path / "page.png"
    Image.new("L", (800, 600), 255).save(page_png)
    rect = Rect(x0=20, y0=20, x1=780, y1=580)
    pages = [PageLayout(page_number=1, template="x", panel_numbers=[1], rects=[rect],
                        image_path=str(page_png))]
    specs = [PanelSpec(panel_number=1, speaker_side="right",
                       dialogue=[DialogueLine(speaker="A", text="One.", style="speech"),
                                 DialogueLine(speaker="B", text="Two.", style="speech"),
                                 DialogueLine(speaker="C", text="Three.", style="speech"),
                                 DialogueLine(speaker="D", text="Four.", style="speech"),
                                 DialogueLine(speaker="E", text="Five dropped.",
                                              style="speech")])]
    lettering.run(settings, pages, specs, chapter_number=1)
    err = capsys.readouterr().err
    assert "WARNING" in err and "dropped" in err and "Five dropped." in err


def _render_first_line_text(settings, rect, dialogue):
    """Run letter_panel (drawn path) on `dialogue` and return the rendered text of
    the FIRST drawn bubble, as a single collapsed string."""
    img = Image.new("L", (int(rect.x1) + 40, int(rect.y1) + 40), 255)
    draw = ImageDraw.Draw(img)
    spec = PanelSpec(panel_number=1, speaker_side="right", dialogue=dialogue)

    captured: list[str] = []
    real_block = lettering._draw_text_block

    def spy(d, lines, font, cx, top, line_h):
        captured.append(" ".join(lines))
        return real_block(d, lines, font, cx, top, line_h)

    lettering._draw_text_block = spy
    try:
        lettering.letter_panel(draw, rect, spec, settings)
    finally:
        lettering._draw_text_block = real_block
    assert captured, "no text drawn"
    return captured[0]


def test_dense_panel_crowd_trim_shortens_line(settings):
    # At >=4 spoken lines the crowd-trim frac FIRES: rendering the long first line with a
    # harsh frac is strictly shorter than with no trim (frac 1.0) on the same crowded panel.
    # (Production frac is a GENTLE 0.90; the test uses a harsher value only to exercise the
    # mechanism deterministically without depending on exact pixel budgets.)
    rect = Rect(x0=20, y0=20, x1=500, y1=720)
    long_text = ("I have waited my entire life for this single moment to finally "
                 "arrive, and now that it has, I will not let anyone at all stand "
                 "in my way no matter the cost to me or to anyone else here today.")
    four = [DialogueLine(speaker="A", text=long_text, style="speech"),
            DialogueLine(speaker="B", text="A second spoken line.", style="speech"),
            DialogueLine(speaker="C", text="A third spoken line.", style="speech"),
            DialogueLine(speaker="D", text="A fourth spoken line.", style="speech")]
    settings.lettering = dict(settings.lettering); settings.lettering["dense_abridge_frac"] = 1.0
    untrimmed = _render_first_line_text(settings, rect, four)
    settings.lettering = dict(settings.lettering); settings.lettering["dense_abridge_frac"] = 0.6
    trimmed = _render_first_line_text(settings, rect, four)
    assert len(trimmed) < len(untrimmed), (
        f"crowd-trim did not shorten the line: trimmed {len(trimmed)} vs untrimmed {len(untrimmed)}")
    # ...but never gutted: the floor keeps it readable (a short clause, not a stub).
    assert len(trimmed) >= 12, f"crowd-trim too harsh: {len(trimmed)} chars"


def test_three_line_panel_ignores_dense_frac(settings):
    # GUARDRAIL ("not too harsh"): the >=4 threshold means a 3-line panel must NOT apply the
    # crowd-trim frac at all — so changing dense_abridge_frac (1.0 vs a harsh 0.5) leaves a
    # 3-line render byte-identical. (At 4 lines, by contrast, the frac would bite.)
    rect = Rect(x0=20, y0=20, x1=500, y1=720)
    long_text = ("I have waited my entire life for this single moment to finally "
                 "arrive, and now that it has, I will not let anyone at all stand "
                 "in my way no matter the cost to me or to anyone else here today.")
    three = [DialogueLine(speaker="A", text=long_text, style="speech"),
             DialogueLine(speaker="B", text="A second spoken line.", style="speech"),
             DialogueLine(speaker="C", text="A third spoken line.", style="speech")]
    settings.lettering = dict(settings.lettering); settings.lettering["dense_abridge_frac"] = 1.0
    full = _render_first_line_text(settings, rect, three)
    settings.lettering = dict(settings.lettering); settings.lettering["dense_abridge_frac"] = 0.5
    harsh = _render_first_line_text(settings, rect, three)
    assert full == harsh, "dense_abridge_frac must have NO effect on a 3-line panel"


def test_four_spoken_lines_render_without_drop_fifth_dropped(settings):
    # A 4-spoken-line panel renders 4 bubbles with nothing recorded as dropped (the
    # raised cap of 4). A 5-line panel drops exactly the 5th (beyond the cap).
    rect = Rect(x0=20, y0=20, x1=780, y1=1180)   # tall panel: all 4 bubbles fit

    def run(dialogue):
        img = Image.new("L", (820, 1220), 255)
        draw = ImageDraw.Draw(img)
        spec = PanelSpec(panel_number=1, speaker_side="right", dialogue=dialogue)
        drawn = []
        real = lettering._draw_bubble

        def spy(d, cx, cy, tw, th, style, tail_target):
            drawn.append(style)
            return real(d, cx, cy, tw, th, style, tail_target)

        lettering._draw_bubble = spy
        dropped: list[str] = []
        try:
            lettering.letter_panel(draw, rect, spec, settings, dropped=dropped)
        finally:
            lettering._draw_bubble = real
        return drawn, dropped

    four = [DialogueLine(speaker="A", text="One.", style="speech"),
            DialogueLine(speaker="B", text="Two.", style="speech"),
            DialogueLine(speaker="C", text="Three.", style="speech"),
            DialogueLine(speaker="D", text="Four.", style="speech")]
    drawn4, dropped4 = run(four)
    assert len(drawn4) == 4, f"expected 4 bubbles, drew {len(drawn4)}"
    assert dropped4 == [], f"nothing should drop within the cap of 4, got {dropped4}"

    five = four + [DialogueLine(speaker="E", text="Five dropped.", style="speech")]
    drawn5, dropped5 = run(five)
    assert len(drawn5) == 4, f"only 4 bubbles may draw under the cap, drew {len(drawn5)}"
    assert dropped5 == ["Five dropped."], (
        f"the 5th line must be dropped beyond the cap, got {dropped5}")


def test_max_bubbles_per_panel_defaults_to_four_when_key_absent(settings):
    # Regression: a user config that omits `max_bubbles_per_panel` must fall back to
    # the documented/shipped default of 4 (not the old in-code 3), so the 4th line is
    # NOT silently dropped. load_settings does no merge over default.yaml, so a partial
    # override would otherwise hit the code fallback.
    settings.lettering = dict(settings.lettering)
    settings.lettering.pop("max_bubbles_per_panel", None)   # simulate a partial override
    rect = Rect(x0=20, y0=20, x1=780, y1=1180)   # tall panel: all 4 bubbles fit
    img = Image.new("L", (820, 1220), 255)
    draw = ImageDraw.Draw(img)
    spec = PanelSpec(panel_number=1, speaker_side="right",
                     dialogue=[DialogueLine(speaker="A", text="One.", style="speech"),
                               DialogueLine(speaker="B", text="Two.", style="speech"),
                               DialogueLine(speaker="C", text="Three.", style="speech"),
                               DialogueLine(speaker="D", text="Four.", style="speech")])
    drawn = []
    real = lettering._draw_bubble

    def spy(d, cx, cy, tw, th, style, tail_target):
        drawn.append(style)
        return real(d, cx, cy, tw, th, style, tail_target)

    lettering._draw_bubble = spy
    dropped: list[str] = []
    try:
        lettering.letter_panel(draw, rect, spec, settings, dropped=dropped)
    finally:
        lettering._draw_bubble = real
    # the default fallback must keep the 4th bubble (cap=4), dropping nothing.
    assert len(drawn) == 4, f"expected 4 bubbles with default cap, drew {len(drawn)}"
    assert dropped == [], f"nothing should drop with default cap of 4, got {dropped}"


def test_organic_unbreakable_token_no_midword_space(settings):
    # Regression: in the ORGANIC branch a hard-broken over-long token (URL / katakana
    # run / hyphen-less compound) must NOT pick up a spurious mid-word SPACE. _wrap
    # hard-breaks the token into adjacent chunks carrying no separating space; the
    # interior re-fit must abridge the coherent SOURCE string, not " ".join(lines)
    # (which would inject a space the interior re-wrap then draws as a word break).
    from ln2manga import bubbles

    img = Image.new("L", (800, 600), 255)
    draw = ImageDraw.Draw(img)
    page = Image.new("L", (800, 600), 255)
    rect = Rect(x0=20, y0=20, x1=780, y1=580)
    spec = PanelSpec(panel_number=1, speaker_side="left",
                     dialogue=[DialogueLine(
                         speaker="A",
                         text="Supercalifragilisticexpialidocious",
                         style="speech")])

    captured = {}
    real_block = lettering._draw_text_block
    real_place = bubbles.place_bubble

    def spy_block(d, lines, font, cx, top, line_h):
        captured["lines"] = list(lines)
        return real_block(d, lines, font, cx, top, line_h)

    def fake_place(pg, shape_path, cx, cy, bw, bh):
        # plausible interior wide enough to break the token across two lines.
        return Rect(x0=int(cx - 130), y0=int(cy - 60),
                    x1=int(cx + 130), y1=int(cy + 60))

    lettering._draw_text_block = spy_block
    bubbles.place_bubble = fake_place
    try:
        lettering.letter_panel(draw, rect, spec, settings, page_img=page,
                               page_live=page, shape_lib={"speech": ["x.png"]})
    finally:
        lettering._draw_text_block = real_block
        bubbles.place_bubble = real_place

    assert captured.get("lines"), "no text drawn in organic branch"
    # the token must split across multiple lines (the interior is narrower than it)...
    assert len(captured["lines"]) > 1, "token should hard-break across lines"
    # ...and NO drawn line may contain an interior space (that would be a mid-word break).
    for ln in captured["lines"]:
        assert " " not in ln, f"spurious mid-word space in hard-broken line {ln!r}"
    # the concatenation must reconstruct a contiguous prefix of the source token
    # (no characters lost or reordered by the re-join).
    joined = "".join(captured["lines"]).rstrip("…")
    assert "Supercalifragilisticexpialidocious".startswith(joined), (
        f"hard-broken chunks {captured['lines']} do not reconstruct the source token")
