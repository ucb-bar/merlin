"""Adversarial host controls; simulated audit output never supplies proof authority."""

import json
import subprocess

import pytest
from test_compiler_formal_transforms import pair

from merlin.verify import compiler_loop_intake as intake
from merlin.verify import loop_transform_gate as gate


@pytest.mark.parametrize("mutation", ["replace", "add", "remove", "symlink"])
def test_private_tree_refuses_mutation(tmp_path, mutation):
    library = tmp_path / "lib"
    library.mkdir()
    obj = library / "Query.olean"
    obj.write_bytes(b"original")
    references = gate._library_refs(library)
    if mutation == "replace":
        obj.write_bytes(b"changed")
    elif mutation == "add":
        (library / "new.olean").write_bytes(b"new")
    elif mutation == "remove":
        obj.unlink()
    else:
        (library / "alias.olean").symlink_to(obj)
    with pytest.raises(intake.Unavailable, match="private proof"):
        gate._validate_library(library, references)


@pytest.mark.parametrize("mutation", ["replace", "add", "redirect", "snapshot"])
def test_audit_time_mutation_never_issues_live_completion(tmp_path, monkeypatch, mutation):
    prepared = intake.prepare(pair(intake.TILE), kinds=(intake.TILE,))
    tool = tmp_path / "lean"
    tool.write_bytes(b"test-only tool placeholder")
    destination = tmp_path / "proof"
    monkeypatch.setattr(gate, "_closure", lambda _: [])

    def simulated_process(command, cwd, env, directory, label, timeout):
        obj = directory / "lib/LoopTransformQuery.olean"
        if label == "query":
            obj.write_bytes(b"test-only object placeholder")
            output = ""
        elif label == "audit":
            imported = obj
            if mutation == "replace":
                obj.write_bytes(b"audit-time replacement")
            elif mutation == "add":
                (obj.parent / "Injected.olean").write_bytes(b"unlisted object")
            elif mutation == "redirect":
                imported = obj.parent / "Other.olean"
                imported.write_bytes(obj.read_bytes())
            else:
                (directory / "stage0.mlir").write_text("changed during audit")
            output = json.dumps(
                {
                    "violations": [],
                    "assurance": "lean_kernel_std3_no_native_reflection",
                    "theorems_audited": 1,
                    "axioms_used": [],
                    "imports": [{"module": "LoopTransformQuery", "olean": str(imported)}],
                }
            )
        else:
            output = json.dumps({"statements": {root: "SIMULATED TEST TYPE" for root in prepared.roots}})
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(gate, "_run", simulated_process)
    before = set(gate._LIVE)
    with pytest.raises(intake.Unavailable, match="private proof|escaped|artifact changed"):
        gate.check(prepared, lean=tool, destination=destination)
    assert set(gate._LIVE) == before
