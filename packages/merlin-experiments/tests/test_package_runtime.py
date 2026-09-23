"""The compiler transport stays core-only; trusted evaluation is a lazy optional consumer."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from merlin.common.paths import python_import_roots
from merlin.targetgen import oot_runner, package_runtime


def test_legacy_module_and_error_identity_and_internal_patching(tmp_path, monkeypatch):
    assert oot_runner is package_runtime
    assert oot_runner.CertFailure is package_runtime.CertFailure
    package = package_runtime.Package(tmp_path, {"language": "python"}, tmp_path / "unused.py")
    monkeypatch.setattr(oot_runner, "_resolve_argv", lambda *args: [sys.executable, "-c", "print('same module')"])
    result = package_runtime.run_entrypoint(package, "parse", tmp_path / "input.mlir")
    assert result.returncode == 0 and result.stdout.strip() == "same module"


def test_core_transport_import_and_failure_without_aet_or_experiments(tmp_path):
    code = """
import importlib.abc, json, sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "aet" or fullname.startswith("aet.") or fullname == "merlin.targetgen.package_certification":
            raise ModuleNotFoundError("optional package blocked", name=fullname)
sys.meta_path.insert(0, BlockOptional())
from merlin.targetgen import oot_runner, package_runtime
from merlin.compile import mesh_backend
assert oot_runner is package_runtime
assert not any(name == "aet" or name.startswith("aet.") for name in sys.modules)
try:
    package_runtime.load_package(sys.argv[1])
except package_runtime.CertFailure as exc:
    assert exc.plane == "contract"
    assert exc.category == "structural_invariant_violation"
    assert exc.category.value == "structural_invariant_violation"
else:
    raise AssertionError("missing package did not fail closed")
try:
    oot_runner.certify
except ModuleNotFoundError as exc:
    assert "merlin-experiments" in str(exc)
else:
    raise AssertionError("certification silently available")
print(json.dumps({"core_transport": True}))
"""
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(str(path) for path in python_import_roots()))
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "missing")],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == {"core_transport": True}


def test_failure_is_canonical_aet_category_when_adapter_installed():
    from aet.core.failures import FailureCategory

    failure = package_runtime.CertFailure("contract", "structural_invariant_violation", "invalid input")
    assert failure.category is FailureCategory.STRUCTURAL_INVARIANT_VIOLATION
    assert failure.category.value == "structural_invariant_violation"
    infra = package_runtime.InfraFailure(
        package_runtime.INFRASTRUCTURE_PLANE,
        package_runtime.InfraCategory.COHORT_NOT_MATERIALIZED,
        "missing cohort",
    )
    assert isinstance(infra, oot_runner.CertFailure)
    assert str(infra.category) == "cohort_not_materialized"


def test_optional_certification_consumes_patched_runtime_and_preserves_failure_record(tmp_path, monkeypatch):
    from merlin.targetgen import provenance

    source = tmp_path / "input.interface.mlir"
    source.write_text("module {}\n")
    recorded = []
    monkeypatch.setattr(provenance, "toolchain_shas", lambda target: {})
    failure = package_runtime.CertFailure("integrity", "forbidden_pattern", "blocked test input")

    def refuse(*args, **kwargs):
        raise failure

    monkeypatch.setattr(oot_runner, "load_package", refuse)
    monkeypatch.setattr(oot_runner, "_record", lambda *args: recorded.append(args))
    result = package_runtime.certify(
        tmp_path / "package", source, runs_root=tmp_path / "runs", run_id="failure-probe", target="synthetic"
    )
    assert result["status"] == "fail"
    assert result["failure"] == {"plane": "integrity", "category": "forbidden_pattern", "detail": "blocked test input"}
    assert len(recorded) == 1
    assert recorded[0][11] == result["failure"]
    assert "package_certification" in package_runtime.certify.__module__


def test_missing_provider_metadata_does_not_create_authority(tmp_path):
    package = package_runtime.Package(tmp_path, {}, tmp_path / "unused")
    assert package.provider is None


def test_integrity_scan_does_not_ignore_packages_below_a_build_ancestor(tmp_path):
    candidate = tmp_path / "build" / "packages" / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "tool.py").write_text("from merlin.targetgen import capsule_golden\n")
    package = package_runtime.Package(candidate, {}, candidate / "tool.py")
    with pytest.raises(package_runtime.CertFailure, match="integrity violation"):
        package_runtime.integrity_scan(package)


def test_integrity_scan_retains_own_generated_build_subtree_exclusion(tmp_path):
    candidate = tmp_path / "build" / "packages" / "candidate"
    generated = candidate / "build"
    generated.mkdir(parents=True)
    (generated / "generated.py").write_text("from merlin.targetgen import capsule_golden\n")
    (candidate / "tool.py").write_text("print('ordinary candidate')\n")
    package = package_runtime.Package(candidate, {}, candidate / "tool.py")
    package_runtime.integrity_scan(package)
