"""The explicit claim-root audit retains unrelated declaration safety checks."""

import json
import os
from pathlib import Path

import pytest

from merlin.common import owned_process
from merlin.common.paths import repo_root
from merlin.verify import compiler_macc
from merlin.verify.lean.check import lean_exe


def invoke(tools, args, *, extra_path=None):
    environment = {**os.environ, "PATH": str(tools.sdk_bin) + os.pathsep + os.environ.get("PATH", "")}
    if extra_path is not None:
        environment["LEAN_PATH"] = str(extra_path) + os.pathsep + environment.get("LEAN_PATH", "")
    return owned_process.run(
        [str(tools.sdk_bin / "lake"), "env", str(lean_exe().with_name("merlin-lean-audit")), *args],
        cwd=repo_root() / "merlin/lean",
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("roster", ["", "missing.root", compiler_macc.THEOREM + "," + compiler_macc.THEOREM])
def test_unknown_empty_and_duplicate_roots_refuse(pass_gate_tools, roster):
    result = invoke(pass_gate_tools, ["--roots", roster, "MerlinVerify.Check.Bf16F32AccBoundary"])
    assert result.returncode == 2
    assert not result.stdout.strip()


def fixture_module(tools, work: Path, *, unsafe=False):
    name = "MaccAuditControl"
    source = work / (name + ".lean")
    source.write_text(
        "namespace MerlinVerify.MaccAuditControl\n"
        "theorem claim : True := True.intro\n"
        + ("unsafe def unrelated : Nat := 0\n" if unsafe else "def unrelated : Nat := 0\n")
        + "end MerlinVerify.MaccAuditControl\n"
    )
    built = owned_process.run(
        [str(tools.sdk_bin / "lean"), "-o", str(source.with_suffix(".olean")), str(source)],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert built.returncode == 0, built.stderr + built.stdout
    return name, "MerlinVerify.MaccAuditControl.claim"


def test_safe_claim_does_not_excuse_unsafe_unrelated_declaration(pass_gate_tools, tmp_path):
    module, claim = fixture_module(pass_gate_tools, tmp_path, unsafe=True)
    result = invoke(pass_gate_tools, ["--roots", claim, module], extra_path=tmp_path)
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["root_axioms"] == {claim: []}
    assert any("unsafe:" in violation for violation in report["violations"])
    assert report["audit_scope"] == "exact_roots_and_all_imported_project_declaration_safety"


def test_original_default_full_audit_retains_its_report(pass_gate_tools, tmp_path):
    module, claim = fixture_module(pass_gate_tools, tmp_path)
    full = invoke(pass_gate_tools, [module], extra_path=tmp_path)
    assert full.returncode == 0
    assert json.loads(full.stdout) == {"theorems_audited": 1, "defined_in": [], "violations": []}
    scoped = invoke(pass_gate_tools, ["--roots", claim, module], extra_path=tmp_path)
    report = json.loads(scoped.stdout)
    assert scoped.returncode == 0 and report["roots"] == [claim]
    assert report["root_axioms"] == {claim: []}
    assert set(report["statements"]) == {claim}
    assert report["project_theorems_scanned"] == 1
