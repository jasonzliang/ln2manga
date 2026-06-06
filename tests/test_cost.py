import pytest

from ln2manga.cost import BudgetExceeded, CostTracker

PRICES = {
    "image": {"gpt-image-2": {"low": 0.02, "medium": 0.08, "high": 0.19}},
    "text_per_mtok": {"gpt-5.4-mini": {"input": 0.75, "output": 4.50}},
}


def test_estimate_and_record(tmp_path):
    t = CostTracker(1.0, tmp_path / "led.json", PRICES)
    assert t.estimate_image("gpt-image-2", "high") == 0.19
    t.record("image", "gpt-image-2", 0.19)
    assert abs(t.spent - 0.19) < 1e-9
    assert t.image_calls == 1


def test_budget_cap_blocks_before_spending(tmp_path):
    t = CostTracker(0.10, tmp_path / "led.json", PRICES)
    with pytest.raises(BudgetExceeded):
        t.check(t.estimate_image("gpt-image-2", "high"), is_image=True)  # 0.19 > 0.10


def test_image_call_backstop(tmp_path):
    t = CostTracker(100.0, tmp_path / "led.json", PRICES, max_image_calls=1)
    t.check(0.02, is_image=True)
    t.record("image", "gpt-image-2", 0.02)
    with pytest.raises(BudgetExceeded):
        t.check(0.02, is_image=True)


def test_ledger_persists_across_instances(tmp_path):
    led = tmp_path / "led.json"
    t1 = CostTracker(10.0, led, PRICES)
    t1.record("image", "gpt-image-2", 0.5)
    t2 = CostTracker(10.0, led, PRICES)
    assert abs(t2.spent - 0.5) < 1e-9
    assert t2.image_calls == 1


def test_text_cost_from_usage(tmp_path):
    from types import SimpleNamespace
    t = CostTracker(10.0, tmp_path / "led.json", PRICES)
    usd = t.record_text_from_usage("gpt-5.4-mini",
                                   SimpleNamespace(input_tokens=1_000_000, output_tokens=0))
    assert abs(usd - 0.75) < 1e-6


def test_corrupt_ledger_does_not_crash(tmp_path, capsys):
    """#11: a truncated/invalid ledger must not raise; warn and start from a clean ledger."""
    led = tmp_path / "led.json"
    led.write_text("{ this is not valid json", encoding="utf-8")
    t = CostTracker(10.0, led, PRICES)          # must not raise
    assert t.spent == 0.0
    assert t.image_calls == 0
    out = capsys.readouterr().out
    assert "unreadable" in out
    # bad copy preserved for inspection
    assert (tmp_path / "led.json.corrupt").exists()
    # the fresh tracker can still record + persist atomically and reload cleanly
    t.record("image", "gpt-image-2", 0.05)
    t2 = CostTracker(10.0, led, PRICES)
    assert abs(t2.spent - 0.05) < 1e-9


def test_estimate_image_from_usage_uses_per_token_price(tmp_path):
    """#3: when a per-image-token price is configured, real usage yields a token-based cost."""
    prices = dict(PRICES)
    prices["image_per_mtok"] = {"gpt-image-2": {"input": 10.0, "output": 40.0}}
    t = CostTracker(10.0, tmp_path / "led.json", prices)
    usd = t.estimate_image_from_usage(
        "gpt-image-2", {"input_tokens": 1_000_000, "output_tokens": 1_000_000})
    assert abs(usd - 50.0) < 1e-6
    # no usage / no configured rate -> 0.0 (so max(flat, this) keeps the flat estimate)
    assert t.estimate_image_from_usage("gpt-image-2", {}) == 0.0
    assert CostTracker(10.0, tmp_path / "l2.json", PRICES).estimate_image_from_usage(
        "gpt-image-2", {"input_tokens": 5, "output_tokens": 5}) == 0.0


def test_in_memory_events_are_capped(tmp_path, monkeypatch):
    """#17: self.events must not grow unbounded; it is capped to the last 2000 (on-disk cap).

    This targets the IN-MEMORY trim, so the per-record fsync'd ledger save is mocked out:
    recording 2500 events with a real fsync + re-read each is O(n^2) disk I/O (~27s — it
    dominated the whole suite). Disk durability/atomicity is covered by the save/recover and
    concurrent-processes tests."""
    t = CostTracker(1_000_000.0, tmp_path / "led.json", PRICES)
    monkeypatch.setattr(t, "_save", lambda: None)   # exercise the in-memory cap, not 2500 fsyncs
    n = 2500
    for _ in range(n):
        t.record("image", "gpt-image-2", 0.01)
    assert len(t.events) <= 2000
    # spend still reflects every recorded event, not just the retained ones
    assert abs(t.spent - n * 0.01) < 1e-6
    assert t.image_calls == n


def test_record_text_fallback_estimate_charges_when_usage_missing(tmp_path):
    """#13: a billed text call with no usage object still advances spend via the estimate."""
    t = CostTracker(10.0, tmp_path / "led.json", PRICES)
    est = t.estimate_text("gpt-5.4-mini", 4000, 16000)
    assert est > 0.0
    t.record("text", "gpt-5.4-mini", est, {"note": "usage_missing_fallback_estimate"})
    assert abs(t.spent - est) < 1e-9


def _hammer_ledger(led_path, n, usd):
    # Run in a separate process: a fresh tracker (own in-memory baseline) records repeatedly
    # against the shared ledger, exactly like a second concurrent ln2manga invocation.
    t = CostTracker(1_000_000.0, led_path, PRICES)
    for _ in range(n):
        t.record("image", "gpt-image-2", usd)


def test_concurrent_processes_accumulate_spend(tmp_path):
    """Concurrent CLI invocations sharing one ledger must not lose spend (last-writer-wins).

    Two processes each record their own deltas against the same ledger; the cross-process
    file lock + re-read-then-merge in record() means their spend accumulates instead of one
    clobbering the other, so the hard cap can't be silently overshot.
    """
    import multiprocessing as mp

    led = tmp_path / "led.json"
    n, usd = 60, 0.01
    procs = [mp.Process(target=_hammer_ledger, args=(str(led), n, usd)) for _ in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0

    final = CostTracker(1_000_000.0, led, PRICES)
    expected = 2 * n * usd
    assert abs(final.spent - expected) < 1e-6, final.spent
    assert final.image_calls == 2 * n


def test_check_reflects_spend_committed_after_construction(tmp_path):
    """check() refreshes from disk, so it sees spend another tracker committed *after* this
    one was constructed and blocks a call that would breach the cap — its in-memory baseline
    alone (0.0 at init) would have wrongly let the call through."""
    led = tmp_path / "led.json"
    guard = CostTracker(0.10, led, PRICES)        # constructed first: in-memory spent == 0.0
    assert guard.spent == 0.0

    other = CostTracker(10.0, led, PRICES)        # a "second process" against the same ledger
    other.record("image", "gpt-image-2", 0.19)    # commits 0.19 to disk after guard was built

    # Without the disk refresh in check(), guard.spent is still 0.0 and 0.0 + 0.05 <= 0.10
    # would pass; with the refresh it sees 0.19 already spent and blocks.
    with pytest.raises(BudgetExceeded):
        guard.check(0.05, is_image=True)
