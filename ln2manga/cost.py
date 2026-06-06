"""Budget-capped cost tracking.

Image models bill per token, but exact per-token rates drift and differ by model, so for
the *budget guard* we use conservative flat per-image estimates from config (they over-
estimate, so the cap trips early). Text cost is computed precisely from token usage.

The running total is persisted to disk so the cap survives across separate CLI invocations.
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


class BudgetExceeded(RuntimeError):
    pass


class CostTracker:
    def __init__(self, max_usd: float, ledger_path: str | Path, prices: dict[str, Any],
                 max_image_calls: int = 10_000, dry_run: bool = False):
        self.max_usd = float(max_usd)
        self.max_image_calls = int(max_image_calls)
        self.ledger_path = Path(ledger_path)
        # Sibling lockfile for the cross-process advisory lock (see record()). The "+ .lock"
        # suffix keeps it next to the ledger and out of the way of the ledger's own atomic
        # temp/.corrupt siblings.
        self._lockfile = self.ledger_path.with_suffix(self.ledger_path.suffix + ".lock")
        self.prices = prices
        self.dry_run = bool(dry_run)
        self.spent: float = 0.0
        self.image_calls: int = 0
        self.events: list[dict] = []
        self._lock = threading.Lock()   # guards spent/image_calls/ledger under parallel generation
        if not self.dry_run:          # dry-run starts clean and never touches the real ledger
            self._load()

    # ── persistence ────────────────────────────────────────────────────────
    def _load(self) -> None:
        if not self.ledger_path.exists():
            return
        try:
            d = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValueError) as e:
            # A run killed mid-write (Ctrl-C / OOM / disk-full) can leave a truncated or
            # otherwise unparseable ledger. Don't crash every subsequent CLI invocation:
            # move the bad file aside for inspection, warn loudly, and start clean (#11).
            bad = self.ledger_path.with_suffix(self.ledger_path.suffix + ".corrupt")
            try:
                self.ledger_path.replace(bad)
            except OSError:
                bad = self.ledger_path
            print(f"[warn] cost ledger {self.ledger_path} was unreadable ({e}); "
                  f"starting from a clean ledger (bad copy at {bad}).")
            return
        self.spent = float(d.get("spent", 0.0))
        self.image_calls = int(d.get("image_calls", 0))
        self.events = d.get("events", [])

    def _save(self) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({
            "spent": round(self.spent, 6),
            "image_calls": self.image_calls,
            "max_usd": self.max_usd,
            "events": self.events[-2000:],
        }, indent=2)
        # Write atomically (temp file + os.replace in the same dir) so an interrupted write
        # can never truncate the live ledger and corrupt the budget guard (#11).
        fd, tmp = tempfile.mkstemp(dir=str(self.ledger_path.parent),
                                   prefix=".cost_ledger.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.ledger_path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ── estimates ──────────────────────────────────────────────────────────
    def estimate_image(self, model: str, quality: str) -> float:
        if self.dry_run:
            return 0.0
        table = self.prices.get("image", {})
        row = table.get(model) or table.get("default", {})
        return float(row.get(quality, row.get("medium", 0.10)))

    def estimate_image_from_usage(self, model: str, usage: dict | None) -> float:
        """Token-based image cost from real API usage, if a per-token price is configured.

        Returns 0.0 when dry-run, when no `image_per_mtok` rate is configured, or when the
        response carried no usable token counts. Callers record max(flat_estimate, this) so
        the flat estimate stays the floor (and the budget guard) while a real call that bills
        higher than the flat guess is never under-counted in the ledger (#3).
        """
        if self.dry_run or not usage:
            return 0.0
        table = self.prices.get("image_per_mtok", {})
        row = table.get(model) or table.get("default")
        if not row:
            return 0.0
        in_tok = int(usage.get("input_tokens") or 0)
        out_tok = int(usage.get("output_tokens") or 0)
        return (in_tok * float(row.get("input", 0.0))
                + out_tok * float(row.get("output", 0.0))) / 1_000_000

    def _text_rate(self, model: str) -> dict[str, float]:
        table = self.prices.get("text_per_mtok", {})
        return table.get(model) or table.get("default", {"input": 1.0, "output": 8.0})

    def estimate_text(self, model: str, in_tokens: int, out_tokens: int) -> float:
        if self.dry_run:
            return 0.0
        r = self._text_rate(model)
        return (in_tokens * r["input"] + out_tokens * r["output"]) / 1_000_000

    # ── guard + record ─────────────────────────────────────────────────────
    def remaining(self) -> float:
        return max(0.0, self.max_usd - self.spent)

    def check(self, estimate: float, *, is_image: bool = False) -> None:
        """Raise BEFORE issuing a call that would breach the cap. Thread-safe; with parallel
        generation the overshoot is bounded by (workers-1) x per-image estimate."""
        if self.dry_run:
            return
        with self._lock:
            # Refresh from disk so the pre-call guard accounts for spend committed by other
            # processes sharing this ledger; a stale baseline would let the cap be overshot.
            if self.ledger_path.exists():
                with open(self._lockfile, "w") as lf:
                    fcntl.flock(lf, fcntl.LOCK_EX)
                    try:
                        self._load()
                    finally:
                        fcntl.flock(lf, fcntl.LOCK_UN)
            if self.spent + estimate > self.max_usd + 1e-9:
                raise BudgetExceeded(
                    f"Would spend ${self.spent + estimate:.3f} > cap ${self.max_usd:.2f} "
                    f"(already ${self.spent:.3f}). Raise budget.max_usd or use --dry-run."
                )
            # check() and record() are separate locked ops, so under parallel generation this
            # backstop can be overshot by up to (workers-1) calls (TOCTOU); bounded and acceptable (#18).
            if is_image and self.image_calls + 1 > self.max_image_calls:
                raise BudgetExceeded(
                    f"Image-call backstop hit ({self.max_image_calls}). Raise budget.max_image_calls."
                )

    def record(self, kind: str, model: str, usd: float, meta: dict | None = None) -> None:
        if self.dry_run:
            if kind == "image":
                with self._lock:
                    self.image_calls += 1
            return
        with self._lock:
            # Cross-process advisory lock: two concurrent ln2manga processes can legitimately
            # share one ledger (it is per data_dir). Without serialising read-modify-write each
            # would add its delta to the same baseline and the last _save() would clobber the
            # other's spend (lost update), silently breaching the hard cap. So we hold an
            # exclusive flock and re-read the freshest on-disk totals before adding this call's
            # delta, making concurrent deltas accumulate instead of overwrite. In a single
            # process the on-disk value equals the in-memory value, so behaviour is unchanged.
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._lockfile, "w") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                try:
                    self._load()    # rebase on whatever other processes have already committed
                    self.spent += float(usd)
                    if kind == "image":
                        self.image_calls += 1
                    self.events.append({
                        "t": round(time.time(), 1), "kind": kind, "model": model,
                        "usd": round(float(usd), 6), "meta": meta or {},
                    })
                    # Cap the in-memory list to match the on-disk cap (events[-2000:] in _save),
                    # otherwise self.events grows unbounded over a long-running process (#17).
                    if len(self.events) > 2000:
                        del self.events[: len(self.events) - 2000]
                    self._save()
                finally:
                    fcntl.flock(lf, fcntl.LOCK_UN)

    def record_text_from_usage(self, model: str, usage: Any) -> float:
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        usd = self.estimate_text(model, in_tok, out_tok)
        self.record("text", model, usd, {"in": in_tok, "out": out_tok})
        return usd
