"""The Triton toolchain pin and its probe.

merlin.triton drives triton's compiler-internal frontend (ASTSource/make_ir), which is not a stable
public API, so the version is exact-pinned and checked before anything touches it. These tests keep
the pin honest in the two ways it can rot: the declared version drifting from the installed one, and
the pin drifting from the `triton` extra that installs it.
"""
from __future__ import annotations

import pytest
import tomllib

from merlin.common.paths import repo_root
from merlin.triton import toolchain


def test_pin_matches_the_pyproject_extra():
    """One version, two places that must agree — there is deliberately no third lock file."""
    data = tomllib.loads((repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
    extra = data["project"]["optional-dependencies"]["triton"]
    assert extra == [f"triton=={toolchain.PINNED_TRITON}"], (
        f"pyproject `triton` extra {extra} disagrees with toolchain.PINNED_TRITON "
        f"{toolchain.PINNED_TRITON!r}; bump both together")


def test_probe_reports_without_raising():
    """probe() is for diagnostics, so it answers even when the toolchain is unusable."""
    p = toolchain.probe()
    assert p.pinned == toolchain.PINNED_TRITON
    assert isinstance(p.compatible, bool)
    assert p.compatible or p.reason, "an incompatible probe must say why"
    assert set(p.as_dict()) == {"triton", "pinned", "compatible", "reason", "notes"}


def test_probe_is_compatible_when_the_pinned_wheel_is_installed():
    pytest.importorskip("triton")
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("triton")
    except PackageNotFoundError:
        pytest.skip("triton importable but not installed as a distribution")
    p = toolchain.probe()
    if installed != toolchain.PINNED_TRITON:
        assert not p.compatible and "mismatch" in p.reason
        pytest.skip(f"installed triton {installed} != pin {toolchain.PINNED_TRITON}")
    assert p.compatible, p.reason
    assert toolchain.require().installed == toolchain.PINNED_TRITON


def test_require_raises_with_both_versions_named(monkeypatch):
    """A mismatch must name expected AND found; that is the whole point of failing early."""
    monkeypatch.setattr(toolchain, "_installed_version", lambda: "0.0.1-not-a-real-release")
    with pytest.raises(toolchain.TritonToolchainError) as exc:
        toolchain.require()
    msg = str(exc.value)
    assert toolchain.PINNED_TRITON in msg and "0.0.1-not-a-real-release" in msg


def test_a_stripped_install_is_reported_separately_from_a_version_mismatch(monkeypatch):
    """A Python-only tree reports a plausible version, then fails deep inside make_ir.

    There is such a tree in this repo (under an experiment workspace) whose dist-info carries
    neither METADATA nor RECORD and which has no triton/_C. Detecting it here turns a confusing
    late failure into a clear early one.
    """
    monkeypatch.setattr(toolchain, "_installed_version", lambda: toolchain.PINNED_TRITON)
    monkeypatch.setattr(toolchain, "_native_library_present", lambda: False)
    p = toolchain.probe()
    assert not p.compatible
    assert any("libtriton" in n for n in p.notes), p.notes
    assert "mismatch" not in p.reason      # not a version problem, and must not be reported as one
