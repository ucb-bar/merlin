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
import yaml

from merlin.common.paths import repo_root

CORPUS = repo_root() / "merlin/contract/capsules/isa"


@pytest.fixture
def compiler_package(tmp_path, monkeypatch):
    """A schema-checked local package; building is observed, not delegated to host tools."""
    from merlin.targetgen import oot_runner

    package = tmp_path / "compiler"
    package.mkdir()
    manifest = {
        "artifact_type": "mlir_oot_target_backend",
        "target": "fixture",
        "language": "python",
        "authoring": {"mode": "hand_curated"},
        "integrity_exempt": False,
        "entrypoints": {"tool": "compiler.py"},
        "commands": {
            command: {"argv": ["python3", "compiler.py"]}
            for command in ("parse", "lower_interface_to_target", "emit_command_buffer", "lower_target_to_llvm")
        },
    }
    (package / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    (package / "compiler.py").write_text("raise RuntimeError('this unit fixture must never be executed')\n")
    built = []
    monkeypatch.setattr(oot_runner, "build_package", lambda pkg, **kwargs: built.append((pkg, kwargs)))
    return package, built


def _compile_oot(workload: str, compiler_package, run: str = "none"):
    from merlin.compile_cli import compile_oot

    package, _ = compiler_package
    return compile_oot(workload, target="fixture", run=run, verify=False, package=str(package), timeout=600)


def test_a_nonexistent_workload_is_not_reported_as_compiled(compiler_package):
    """THE REGRESSION. Moving the capsule check back below the early return fails here."""
    out = _compile_oot("definitely_not_a_real_workload_xyz", compiler_package)
    assert out["status"] == "not_run"
    assert "no capsule" in out["reason"]


def test_a_whole_model_name_is_refused_and_says_where_to_go(compiler_package):
    """`tiny_llama` is a model, not a capsule; the refusal must name the right path, not just fail."""
    out = _compile_oot("tiny_llama", compiler_package)
    assert out["status"] == "not_run"
    assert "whole models" in out["reason"]
    assert "bundle-pack" in out["reason"]


def test_the_refusal_happens_before_any_package_build(compiler_package):
    """The check must precede the build: validating after it wastes a full toolchain build to
    produce a refusal, and (the original defect) never runs at all on the compile-only path."""
    _, built = compiler_package
    out = _compile_oot("definitely_not_a_real_workload_xyz", compiler_package)
    assert out["status"] == "not_run"
    assert "no capsule" in out["reason"], "an absent default package must not short-circuit this test"
    assert built == [], "the backend package was built before the workload was validated"


@pytest.mark.skipif(
    not (CORPUS / "A0_config_smoke").is_dir(), reason="isa corpus capsule A0_config_smoke not present in this checkout"
)
def test_a_real_capsule_still_compiles(compiler_package):
    """A valid workload reaches the package build, without requiring a real compiler toolchain."""
    package, built = compiler_package
    out = _compile_oot("A0_config_smoke", compiler_package)
    assert out["status"] == "compiled"
    assert len(built) == 1
    assert built[0][0].directory == package
    assert built[0][1] == {"timeout": 600}
