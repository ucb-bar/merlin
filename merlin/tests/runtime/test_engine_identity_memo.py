"""An engine identity is reused only while its pinned files keep their stat identity."""

from __future__ import annotations

import json
import time

import pytest

from merlin.runtime.backends import engine_identity_memo as M


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    M.clear()
    monkeypatch.setattr(M, "SETTLED_NS", -(10**12))  # file times cannot be aged in a test
    yield
    M.clear()


def _receipt(tmp_path):
    binary = tmp_path / "engine"
    binary.write_bytes(b"engine")
    pinned = tmp_path / "rtl"
    pinned.mkdir()
    (pinned / "top.fir").write_text("circuit top")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"artifacts": {"binary": {"path": str(binary), "sha256": "a" * 64},
                                                  "rtl": {"path": str(pinned), "sha256": "b" * 64}}}))
    return binary, pinned, receipt


def test_unchanged_pins_reuse_and_any_change_recomputes(tmp_path):
    binary, pinned, receipt = _receipt(tmp_path)
    paths = [str(binary), *M.receipt_paths(receipt)]
    assert str(pinned) in paths
    calls = []
    compute = lambda: calls.append(1) or {"identity": len(calls)}
    assert M.cached(("k",), paths, compute) == {"identity": 1}
    assert M.cached(("k",), paths, compute) == {"identity": 1} and len(calls) == 1
    time.sleep(0.05)
    (pinned / "top.fir").write_text("circuit TOP")  # a file inside a pinned directory changed
    assert M.cached(("k",), paths, compute) == {"identity": 2}


def test_values_expire_and_callers_get_copies(tmp_path, monkeypatch):
    binary, _, receipt = _receipt(tmp_path)
    paths = [str(binary), *M.receipt_paths(receipt)]
    calls = []
    compute = lambda: calls.append(1) or {"nested": {"n": len(calls)}}
    first = M.cached(("k",), paths, compute)
    first["nested"]["n"] = 99
    assert M.cached(("k",), paths, compute) == {"nested": {"n": 1}}
    monkeypatch.setattr(M, "MAX_AGE_S", 0.0)
    assert M.cached(("k",), paths, compute) == {"nested": {"n": 2}}


def test_unreadable_or_recent_files_are_never_cached(tmp_path, monkeypatch):
    calls = []
    compute = lambda: calls.append(1) or len(calls)
    M.cached(("k",), [str(tmp_path / "missing")], compute)
    M.cached(("k",), [str(tmp_path / "missing")], compute)
    assert len(calls) == 2
    monkeypatch.setattr(M, "SETTLED_NS", 10**12)  # everything looks freshly written
    binary, _, receipt = _receipt(tmp_path / "x") if (tmp_path / "x").mkdir() is None else None
    paths = [str(binary), *M.receipt_paths(receipt)]
    M.cached(("j",), paths, compute)
    M.cached(("j",), paths, compute)
    assert len(calls) == 4
