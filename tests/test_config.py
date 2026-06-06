"""Tests for load_settings validation and error messages."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# make the package importable without an editable install
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ln2manga.config import DEFAULT_CONFIG, Settings, load_settings  # noqa: E402


def test_default_config_loads_as_settings():
    s = load_settings()
    assert isinstance(s, Settings)


def test_missing_config_raises_clear_filenotfound(tmp_path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError) as exc:
        load_settings(missing)
    msg = str(exc.value)
    assert str(missing) in msg
    assert str(DEFAULT_CONFIG) in msg


def test_empty_config_raises_clear_valueerror(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_settings(empty)
    msg = str(exc.value)
    assert str(empty) in msg
    assert "NoneType" in msg


def test_comment_only_config_raises_clear_valueerror(tmp_path):
    comment = tmp_path / "comment.yaml"
    comment.write_text("# just a comment\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_settings(comment)
    assert str(comment) in str(exc.value)


def test_list_config_raises_clear_valueerror(tmp_path):
    listy = tmp_path / "list.yaml"
    listy.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_settings(listy)
    msg = str(exc.value)
    assert str(listy) in msg
    assert "list" in msg
