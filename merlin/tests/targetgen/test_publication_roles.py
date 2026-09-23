"""Role-aware preserved exports and scoped build observations; no hardware/network."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from merlin.common.yaml import load_yaml, write_yaml
from merlin.targetgen import package_records, package_runtime, publish


@pytest.fixture
def compiler(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    source = artifacts / "targets" / "fixture" / "slot"
    source.mkdir(parents=True)
    write_yaml(
        source / "manifest.yaml",
        {
            "package_id": "original-identity",
            "target": "fixture",
            "artifact_type": "mlir_oot_target_backend",
            "language": "python",
            "authoring": {"mode": "hand_curated"},
            "integrity_exempt": False,
            "entrypoints": {"tool": "compiler.py"},
            "commands": {
                name: {"argv": ["{tool}", "{input_mlir}"]}
                for name in ("parse", "lower_interface_to_target", "emit_command_buffer", "lower_target_to_llvm")
            },
        },
    )
    (source / "compiler.py").write_text("from helper import VALUE\nprint(VALUE)\n")
    (source / "helper.py").write_text("VALUE = 'module {}'\n")
    (source / "README.md").write_text("Original author documentation, never replaced.\n")
    (source / "build").mkdir()
    (source / "build" / "input.txt").write_text("This is a declared source member, not disposable.\n")
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    return source, artifacts


def export(compiler, **kwargs):
    source, artifacts = compiler
    return publish.publish(
        "fixture",
        package_id=source.name,
        artifacts_root=artifacts,
        remote="file:///not-contacted",
        gate=False,
        **kwargs,
    )


def test_export_preserves_every_original_member_and_actual_python_layout(compiler, tmp_path):
    source, _ = compiler
    before = package_records.payload_inventory(source)
    result = export(compiler, dry_run=True)
    for member in before["members"]:
        if member["kind"] == "file":
            assert (result.repo_dir / member["path"]).read_bytes() == (source / member["path"]).read_bytes()
    assert package_records.payload_inventory(source) == before
    assert (result.repo_dir / "MERLIN_PUBLICATION.md").is_file()
    assert (result.repo_dir / "manifest.yaml").read_bytes() == (source / "manifest.yaml").read_bytes()
    assert json.loads(result.build_verification_path.read_text())["reason"] == "dry-run"
    from merlin.targetgen.publication_verification import verify_export

    receipt = verify_export(result.repo_dir, result.repo_dir.parent, timeout=7)
    assert receipt["status"] == "pass", receipt
    assert receipt["build_invoked"] is False
    assert receipt["scope"] == "build-artifact-presence"
    copied = Path(receipt["execution_path"])
    input_path = tmp_path / "input.mlir"
    input_path.write_text("module {}")
    output = package_runtime.run_entrypoint(
        package_runtime.load_package(copied), "parse", input_path, timeout=7, write_bytecode=False
    )
    assert output.returncode == 0 and output.stdout.strip() == "module {}"
    assert package_records.payload_inventory(result.repo_dir) == receipt["exported_input"]


@pytest.mark.parametrize("reserved", [".merlin", "MERLIN_PUBLICATION.md", ".git"])
def test_reserved_publication_collisions_refuse_without_payload_writes(compiler, reserved):
    source, _ = compiler
    (source / reserved).write_text("operator-owned")
    before = package_records.payload_inventory(source)
    with pytest.raises(publish.PublishError, match="reserved"):
        export(compiler, dry_run=True)
    assert package_records.payload_inventory(source) == before


def test_explicit_support_role_cannot_become_compiler_through_abi_shape(compiler):
    source, _ = compiler
    (source / "contracts").mkdir()
    write_yaml(source / "contracts" / "target_contract.yaml", {"name": "fixture"})
    write_yaml(
        source / "provider.yaml",
        {"schema": "merlin.provider.v1", "id": "support", "target": "fixture", "role": "support"},
    )
    with pytest.raises(publish.PublishError, match="not support"):
        export(compiler, dry_run=True)


def test_candidate_missing_real_tool_cannot_gain_generated_wrapper(compiler):
    source, _ = compiler
    (source / "compiler.py").unlink()
    with pytest.raises(publish.PublishError, match="existing tool"):
        export(compiler, dry_run=True)


@pytest.mark.parametrize("failure", ["error", "timeout"])
def test_build_failure_retains_receipt_and_prevents_git(compiler, monkeypatch, tmp_path, failure):
    source, _ = compiler
    manifest = load_yaml(source / "manifest.yaml")
    manifest["build"] = {"command": [sys.executable, "synthetic-not-executed"], "tool_output": "compiler.py"}
    write_yaml(source / "manifest.yaml", manifest)
    before = package_records.payload_inventory(source)

    def build(pkg, *, timeout):
        assert pkg.directory != source and timeout == 3
        if failure == "timeout":
            raise subprocess.TimeoutExpired("synthetic", timeout)
        raise OSError("synthetic build failure")

    monkeypatch.setattr(package_runtime, "build_package", build)
    monkeypatch.setattr(publish, "_git", lambda *a, **kw: pytest.fail("Git must not run after failed build"))
    with pytest.raises(publish.PublishError, match="build verification failed"):
        export(compiler, dry_run=False, build_timeout=3)
    receipts = list((tmp_path / "out").rglob("build_verification.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["status"] == "fail" and receipt["timeout_per_step_s"] == 3
    assert receipt["error"]["type"] == ("TimeoutExpired" if failure == "timeout" else "OSError")
    assert package_records.payload_inventory(source) == before


def test_dry_run_never_invokes_build(compiler, monkeypatch):
    monkeypatch.setattr(package_runtime, "build_package", lambda *a, **kw: pytest.fail("dry-run build"))
    export(compiler, dry_run=True)


def test_actual_python_build_runs_only_in_retained_verification_copy(compiler, monkeypatch, tmp_path):
    from merlin.targetgen.contract import toolchain
    from merlin.targetgen.publication_verification import verify_export

    source, _ = compiler
    manifest = load_yaml(source / "manifest.yaml")
    manifest["build"] = {
        "command": [
            sys.executable,
            "-c",
            "from pathlib import Path; Path('package_build').mkdir(exist_ok=True); "
            "Path('package_build/generated.py').write_text('print(42)\\n')",
        ],
        "tool_output": "package_build/generated.py",
    }
    write_yaml(source / "manifest.yaml", manifest)
    # Python-only build: prevent unrelated CMake probing, retain real runtime build execution.
    monkeypatch.setattr(package_runtime, "_usable_cmake", lambda: "cmake")
    monkeypatch.setattr(toolchain, "mlir_cmake_dir", lambda: tmp_path)
    monkeypatch.setattr(toolchain, "mlir_install", lambda: tmp_path)
    before = package_records.payload_inventory(source)
    result = export(compiler, dry_run=True)
    receipt = verify_export(result.repo_dir, result.repo_dir.parent, timeout=5)
    assert receipt["status"] == "pass", receipt
    assert receipt["build_invoked"] is True
    assert receipt["built_execution"] != receipt["exported_input"]
    assert (Path(receipt["execution_path"]) / "package_build" / "generated.py").read_text() == "print(42)\n"
    assert not (source / "package_build" / "generated.py").exists()
    assert not (result.repo_dir / "package_build" / "generated.py").exists()
    assert package_records.payload_inventory(source) == before


@pytest.mark.parametrize("verify", [True, False])
def test_execute_wires_explicit_verification_before_git(compiler, monkeypatch, verify):
    source, _ = compiler
    called = []

    def build(pkg, *, timeout):
        assert verify and timeout == 9 and pkg.directory != source
        called.append("build")

    def commit(*args):
        assert called == (["build"] if verify else [])
        called.append("git")
        return "synthetic-commit", "synthetic-tag", False

    monkeypatch.setattr(package_runtime, "build_package", build)
    monkeypatch.setattr(publish, "_git_publish", commit)
    result = export(compiler, dry_run=False, verify_build=verify, build_timeout=9)
    assert called == (["build", "git"] if verify else ["git"])
    receipt = json.loads(result.build_verification_path.read_text())
    assert receipt["status"] == ("pass" if verify else "not-run")
    if not verify:
        assert receipt["reason"] == "not-requested"


def test_index_guidance_is_role_aware_without_invented_fetch_or_compile_workflow():
    entries = [
        {
            "branch": "stable/example",
            "package_id": "example",
            "dtype": "declared",
            "status": "observed",
            "role": "host_schedule",
        }
    ]
    text = publish._index_readme("fixture", entries)
    assert "host_schedule" in text and "candidate_compiler" in text
    assert "not-attested" in text and "MERLIN_PUBLICATION.md" in text
    for forbidden in (
        "merlin-target-fetch",
        "merlin-compile --workload",
        "load_rvv_package",
        "payload/schedule.mlir",
        "standalone, buildable",
        "RISC-V toolchain",
    ):
        assert forbidden not in text


def test_publication_instructions_show_actual_recipe_without_invented_commands(compiler):
    source, _ = compiler
    manifest = load_yaml(source / "manifest.yaml")
    manifest["build"] = {"command": ["custom-builder", "--declared"], "tool_output": "build/actual-tool"}
    write_yaml(source / "manifest.yaml", manifest)
    result = export(compiler, dry_run=True)
    text = (result.repo_dir / "MERLIN_PUBLICATION.md").read_text()
    assert "custom-builder" in text and "--declared" in text
    assert "cmake -S" not in text and "--help" not in text
    selected = publish.select_champion("fixture", package_id="slot", artifacts_root=compiler[1])
    assert selected.layout_kind == "preserved_payload"
