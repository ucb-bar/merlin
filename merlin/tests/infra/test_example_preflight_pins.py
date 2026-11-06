"""A runnable package stage must not hide a refused matrix source pin."""

from __future__ import annotations

import importlib.util

import pytest

from merlin.common.paths import repo_root


@pytest.mark.parametrize(
    ("required", "expected"),
    [("saturn_opu_int8", 0), ("saturn_vme_kodiak", 1), ("unknown_pin", 1)],
)
def test_required_pin_gates_preflight_even_when_host_package_stage_passes(monkeypatch, required, expected):
    path = repo_root() / "examples/lib/preflight.py"
    spec = importlib.util.spec_from_file_location("example_preflight_pins", path)
    assert spec and spec.loader
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)

    monkeypatch.setattr(preflight, "_rows", lambda _board, _sdk: ([("package", True, "ready")], False, True))
    monkeypatch.setattr(preflight, "_components", lambda: [("clang", True, "MERLIN_CLANG", "/bin/clang")])
    monkeypatch.setattr(
        preflight,
        "_pin_rows",
        lambda _board: [
            ("ok", "saturn_opu_int8", "exact source"),
            ("stop", "saturn_vme_kodiak", "wrong source"),
        ],
    )
    monkeypatch.setattr(preflight, "_settings", lambda _keys: [])

    assert preflight.main(["--board", "chipyard_kodiak", "--require", "package", "--require-pin", required]) == expected
