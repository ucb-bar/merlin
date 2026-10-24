"""Run labels must expose source edits and unavailable cleanliness observations."""

import pytest

from merlin.common import provenance


@pytest.mark.parametrize(
    "status,suffix,dirty_files",
    [("", "", 0), (" M source.py", "+dirty", 1), ("?? new.py", "+dirty", 1), (None, "+unverified", -1)],
)
def test_record_distinguishes_dirty_and_unknown_source(monkeypatch, status, suffix, dirty_files):
    revision = "a" * 40
    monkeypatch.setattr(provenance, "_git", lambda root, *args: revision if args == ("rev-parse", "HEAD") else status)
    result = provenance.record()
    assert result["merlin_source_label"] == revision + suffix
    assert result["merlin"]["source_label"] == revision + suffix
    assert result["merlin"]["dirty_files"] == dirty_files
    assert result["merlin"]["commit"] == revision
