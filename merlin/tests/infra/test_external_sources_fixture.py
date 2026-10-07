"""The external-checkout skip helper skips on ABSENCE only, never on a present checkout."""

from __future__ import annotations

import external_sources


def test_an_unset_checkout_is_missing(monkeypatch):
    monkeypatch.delenv("MERLIN_EXT_FIXTURE_SOURCE", raising=False)
    monkeypatch.setattr("merlin.common.paths._dotenv", lambda: {})
    assert external_sources.missing("fixture_source") == ["MERLIN_EXT_FIXTURE_SOURCE"]


def test_a_checkout_pointing_at_nothing_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("MERLIN_EXT_FIXTURE_SOURCE", str(tmp_path / "absent"))
    assert external_sources.missing("fixture_source") == ["MERLIN_EXT_FIXTURE_SOURCE"]


def test_a_present_checkout_is_not_skipped(monkeypatch, tmp_path):
    monkeypatch.setenv("MERLIN_EXT_FIXTURE_SOURCE", str(tmp_path))
    assert external_sources.missing("fixture_source") == []
    assert not external_sources.requires_ext("fixture_source").args[0]
