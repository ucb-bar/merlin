"""`merlin-compile --run none` must not report `compiled` for a workload it never looked at.

WHY THIS EXISTS. `compile_oot` built the BACKEND PACKAGE, set `status: compiled`, and returned on
`run == "none"` -- and the check that the named workload is actually a capsule sat BELOW that early
return. So the compile-only path never read `workload` at all.

MEASURED 2026-09-10, both reporting `status: compiled`:
  --workload definitely_not_a_real_workload_xyz --target gemmini --run none
  --workload tiny_llama                          --target gemmini --run none

The second is the damaging one. `tiny_llama` is a whole MODEL, and this function's own docstring says
"Accelerators run capsules/kernels, not whole VLA models" -- so a caller asking for a model got
`compiled` back, wrote no files, and had every reason to believe a model had been compiled for
gemmini. A status that cannot be false is not a status.
"""

from __future__ import annotations

import pytest

from merlin.common.paths import repo_root

CORPUS = repo_root() / "merlin/contract/capsules/isa"


def _compile_oot(workload: str, run: str = "none"):
    from merlin.compile_cli import compile_oot

    return compile_oot(workload, target="gemmini", run=run, verify=False, package=None, timeout=600)


def test_a_nonexistent_workload_is_not_reported_as_compiled():
    """THE REGRESSION. Moving the capsule check back below the early return fails here."""
    out = _compile_oot("definitely_not_a_real_workload_xyz")
    assert out["status"] == "not_run"
    assert "no capsule" in out["reason"]


def test_a_whole_model_name_is_refused_and_says_where_to_go():
    """`tiny_llama` is a model, not a capsule; the refusal must name the right path, not just fail."""
    out = _compile_oot("tiny_llama")
    assert out["status"] == "not_run"
    assert "whole models" in out["reason"]
    assert "bundle-pack" in out["reason"]


def test_the_refusal_happens_before_any_package_build(monkeypatch):
    """The check must precede the build: validating after it wastes a full toolchain build to
    produce a refusal, and (the original defect) never runs at all on the compile-only path."""
    from merlin.targetgen import oot_runner

    built: list = []
    monkeypatch.setattr(oot_runner, "build_package", lambda *a, **k: built.append(1))
    out = _compile_oot("definitely_not_a_real_workload_xyz")
    assert out["status"] == "not_run"
    assert built == [], "the backend package was built before the workload was validated"


@pytest.mark.skipif(
    not (CORPUS / "A0_config_smoke").is_dir(), reason="isa corpus capsule A0_config_smoke not present in this checkout"
)
def test_a_real_capsule_still_compiles():
    """The guard must not break the legitimate compile-only path."""
    out = _compile_oot("A0_config_smoke")
    assert out["status"] == "compiled"
