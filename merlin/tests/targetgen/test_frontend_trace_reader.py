"""Frontend trace capacity is separate from ordinary strict authority records."""

from __future__ import annotations

import inspect

import pytest

from merlin.common import strict_json
from merlin.targetgen import frontend_trace as T


def test_trace_uses_dedicated_capacity_without_changing_global(monkeypatch):
    original = strict_json.loads
    observed = []

    def record(raw, *, max_bytes):
        observed.append(max_bytes)
        return original(raw, max_bytes=max_bytes)

    monkeypatch.setattr(strict_json, "loads", record)
    assert T.loads_frontend_trace(b'{"graphs": {}}') == {"graphs": {}}
    assert observed == [256 * 1024 * 1024]
    assert strict_json.MAX_RECORD_BYTES == 128 * 1024 * 1024
    assert inspect.signature(original).parameters["max_bytes"].default == 128 * 1024 * 1024


def test_trace_above_generic_bound_is_complete_and_bounded(monkeypatch):
    raw = b'{"complete":true}' + b" " * 64
    with pytest.raises(ValueError, match="bounded reader"):
        strict_json.loads(raw, max_bytes=64)
    monkeypatch.setattr(T, "MAX_FRONTEND_TRACE_BYTES", len(raw))
    assert T.loads_frontend_trace(raw) == {"complete": True}
    with pytest.raises(ValueError, match="bounded reader"):
        T.loads_frontend_trace(raw + b" ")
    with pytest.raises(ValueError, match="bounded reader"):
        T.loads_frontend_trace(raw.decode() + "é")


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"nested":{"a":1,"a":2}}',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"a":-Infinity}',
        b'{"a":1e999}',
    ],
)
def test_trace_preserves_duplicate_and_finite_checks(raw):
    with pytest.raises(ValueError, match="duplicate|non-finite"):
        T.loads_frontend_trace(raw)
