"""A current native checker cannot excuse missing or stale auditor-only proof objects."""

import sys

import pytest

from merlin.common.owned_process import run_owned
from merlin.verify import pass_validation
from merlin.verify.lean import check


@pytest.fixture
def isolated_checker(tmp_path, pass_gate_tools, monkeypatch):
    project = tmp_path / "project"
    proof = project / "MerlinVerify" / "AuditOnly.lean"
    proof.parent.mkdir(parents=True)
    (project / "lakefile.toml").write_text(
        'name = "MerlinVerify"\n'
        'defaultTargets = ["MerlinVerify"]\n'
        '[[lean_lib]]\nname = "MerlinVerify"\nglobs = ["MerlinVerify.+"]\n'
        '[[lean_exe]]\nname = "merlin-lean"\nroot = "Main"\n'
        '[[lean_exe]]\nname = "merlin-lean-audit"\nroot = "Audit"\n'
    )
    for name in ("Main", "Audit"):
        (project / (name + ".lean")).write_text("def main : IO Unit := pure ()\n")
    proof.write_text("def MerlinVerify.auditOnly : Bool := true\n")
    sdk = pass_gate_tools.sdk_bin

    def lake(*targets, no_build=False):
        return run_owned(
            [str(sdk / "lake"), "build", *(["--no-build"] if no_build else []), *targets],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    built = lake("MerlinVerify", "merlin-lean", "merlin-lean-audit")
    assert built.returncode == 0, built.stdout + built.stderr
    monkeypatch.setattr(check, "lean_root", lambda: project)
    monkeypatch.setattr(check, "lean_exe", lambda: project / ".lake/build/bin/merlin-lean")
    return sdk, proof, project / ".lake/build/lib/lean/MerlinVerify/AuditOnly.olean", lake


def test_current_library_and_executables_are_accepted(isolated_checker):
    sdk, _, _, _ = isolated_checker
    assert pass_validation.CheckerTools(sdk, sys.executable).sdk_bin == sdk


@pytest.mark.parametrize("mutation", ["stale", "missing"])
def test_audit_only_object_must_be_current(isolated_checker, mutation):
    sdk, source, obj, lake = isolated_checker
    original_object = obj.read_bytes()
    if mutation == "stale":
        source.write_text("unsafe def MerlinVerify.auditOnly : Bool := true\n")
    else:
        obj.unlink()
    # Neither executable imports AuditOnly: the old executable-only check passes.
    executables = lake("merlin-lean", "merlin-lean-audit", no_build=True)
    assert executables.returncode == 0, executables.stdout + executables.stderr
    with pytest.raises(ValueError, match="out of date"):
        pass_validation.CheckerTools(sdk, sys.executable)
    # Admission must refuse rather than silently rebuilding a proof artifact.
    if mutation == "stale":
        assert obj.read_bytes() == original_object
    else:
        assert not obj.exists()
