"""Settings: single source of truth for model IDs, prices, sizes, budgets, paths."""
from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

PKG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_DIR.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
FONTS_DIR = PROJECT_ROOT / "assets" / "fonts"


class Settings(BaseModel):
    # extra="allow": any NEW top-level section added to the master config YAML is preserved on
    # the Settings object (forward-compatible) instead of being silently dropped by pydantic.
    model_config = ConfigDict(extra="allow")

    models: dict[str, Any]
    image: dict[str, Any]
    prices_usd: dict[str, Any]
    budget: dict[str, Any]
    concurrency: dict[str, Any]
    scrape: dict[str, Any]
    parse: dict[str, Any]
    layout: dict[str, Any]
    mangapost: dict[str, Any]
    lettering: dict[str, Any]
    paths: dict[str, Any]
    references: dict[str, Any] = {}     # opt-in real-reference-image feature (see references.py)

    # ── path helpers ───────────────────────────────────────────────────────
    @cached_property
    def data_dir(self) -> Path:
        d = (PROJECT_ROOT / self.paths.get("data", "data")).resolve()
        return d

    def _sub(self, *parts: str) -> Path:
        p = self.data_dir.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def raw_dir(self) -> Path:
        return self._sub("raw")

    @property
    def artifacts_dir(self) -> Path:
        return self._sub("artifacts")

    @property
    def out_dir(self) -> Path:
        return self._sub("out")

    def cache_dir(self, stage: str) -> Path:
        return self._sub("cache", stage)

    @property
    def ledger_path(self) -> Path:
        return self.data_dir / "cost_ledger.json"

    @property
    def fonts_dir(self) -> Path:
        return FONTS_DIR


def load_settings(path: str | Path | None = None) -> Settings:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {cfg_path}. "
            f"Pass a valid --config or omit it to use the default ({DEFAULT_CONFIG})."
        )
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(
            f"{cfg_path} is not a valid config mapping "
            f"(got {type(data).__name__}); see {DEFAULT_CONFIG}."
        )
    return Settings(**data)
