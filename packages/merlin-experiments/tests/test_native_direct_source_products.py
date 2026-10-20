"""Real host subprocess custody, neutral bytes; no compiler/target qualification."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from merlin_experiments.phase1.feedback import candidate_observations as PRODUCTS
from merlin_experiments.phase2 import _archive_source_domain as DOMAIN
from merlin_experiments.phase2 import _native_direct_source_domain as DIRECT
from test_checked_source_model_diagnostic import selected as selected
from test_native_direct_source_domain import publish, receipt_fixture

from merlin.common import invocation_record as I


def _pin(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size}


def native_fixture(tmp_path, selected):
    policy, member, workspace, original, cb, evidence = receipt_fixture(tmp_path, selected)
    source = member.source_dir / "capsule.interface.mlir"
    consumed = workspace / "consumed.mlir"
    consumed.write_bytes(source.read_bytes())
    emitted, buffer = workspace / "lowered.llvm.mlir", workspace / "command_buffer.json"
    emitted.write_bytes(b"neutral emitted source; never a target executable")
    buffer.write_text(json.dumps(cb))
    tool = workspace / "copy_stage.py"
    tool.write_text("import pathlib,sys\npathlib.Path(sys.argv[2]).write_bytes("
                    "sys.argv[3].encode()+pathlib.Path(sys.argv[1]).read_bytes())\n")
    prior = emitted
    for stage, name in (("llvm_translation", "kernel.ll"), ("object", "kernel.o"), ("elf", "original.elf")):
        output = workspace / name
        argv = [str(Path(sys.executable).resolve()), str(tool), str(prior), str(output), stage]
        with I.observe(workspace, stage=stage, argv=argv, inputs=(prior,), outputs=(output,),
                       dependencies=(tool,), env=dict(os.environ)) as receipt:
            result = subprocess.run(argv, capture_output=True, check=True, env=dict(os.environ))
            receipt.complete(result)
        prior = output
    emission = {"source_interface": _pin(source), "command_buffer": _pin(buffer), "lowered_mlir": _pin(emitted)}
    native = {"schema": "merlin_candidate_native_pointer_source_v1", "status": "numeric_match_diagnostic",
              "source_execution_protocol": DIRECT.SELECTOR.POINTER_PROTOCOL,
              "source": {"interface": _pin(consumed)}, "elf": _pin(prior),
              "candidate": {"command_buffer_sha256": hashlib.sha256(
                                json.dumps(cb, sort_keys=True).encode()).hexdigest(),
                            "lowered_mlir_sha256": _pin(emitted)["sha256"]},
              "build_artifacts": {name: _pin(workspace / name) for name in
                                  ("lowered.llvm.mlir", "kernel.ll", "kernel.o", "original.elf")}}
    original["original_grade"].update(candidate_emission=emission, candidate_native_execution=native)
    products, producers = PRODUCTS.reopen_native_products(workspace, emission=emission, elf_pin=native["elf"],
        source=native["source"], candidate=native["candidate"], build_artifacts=native["build_artifacts"])
    compiled = original["compiled"]
    compiled.update(elf_sha256=native["elf"]["sha256"], llvm_mlir_sha256=_pin(emitted)["sha256"],
        artifact_join={"kind": "original_native_model_products", "source_sha256": _pin(source)["sha256"],
                       "command_buffer_sha256": _pin(buffer)["sha256"],
                       "producer_sha256s": [_pin(path)["sha256"] for path in producers]})
    compiled["candidate_diagnostics"].update(source_emission_join={"status": "UNKNOWN"},
        artifacts={name: _pin(path) for name, path in products.items()})
    return policy, member, workspace, original, cb, evidence


def test_original_native_producer_join_does_not_need_a_generic_plan_verdict(tmp_path, selected):
    policy, member, workspace, original, cb, evidence = native_fixture(tmp_path, selected)
    value = publish(workspace, original, cb)
    with pytest.raises(ValueError, match="source/emission binding"):
        DOMAIN.emitted_source(member, value["compiled"], workspace)
    before = (workspace / "ordinary_observation.json").read_bytes()
    result = DIRECT.current(policy, member, workspace, value, evidence)
    assert result["status"] == "source_domain_supported"
    assert result["training_source_groups"] == ["training-group"]
    assert (workspace / "ordinary_observation.json").read_bytes() == before
    assert "qualification" in result["authority"]  # Explicitly denies qualification authority.


@pytest.mark.parametrize("fault", ["source", "consumed_source", "emitted", "object", "elf", "producer",
                                  "join_kind", "join_source", "join_producer", "compiled_elf", "role", "protocol"])
def test_native_join_refuses_changed_or_foreign_consumed_roles(tmp_path, selected, fault):
    policy, member, workspace, original, cb, evidence = native_fixture(tmp_path, selected)
    compiled = original["compiled"]
    if fault in {"source", "consumed_source", "emitted", "object", "elf"}:
        path = {"source": member.source_dir / "capsule.interface.mlir", "consumed_source": workspace / "consumed.mlir",
                "emitted": workspace / "lowered.llvm.mlir", "object": workspace / "kernel.o",
                "elf": workspace / "original.elf"}[fault]
        path.write_bytes(path.read_bytes() + b"\n")
    elif fault == "producer":
        path = next(workspace.glob("invocations/*/invocation.json"))
        path.write_bytes(path.read_bytes() + b" ")
    elif fault.startswith("join_"):
        key = {"join_kind": "kind", "join_source": "source_sha256", "join_producer": "producer_sha256s"}[fault]
        compiled["artifact_join"][key] = "foreign"
    elif fault == "compiled_elf":
        compiled["elf_sha256"] = "0" * 64
    elif fault == "role":
        compiled["candidate_diagnostics"]["artifacts"]["object"]["sha256"] = "0" * 64
    else:
        original["original_grade"]["candidate_native_execution"]["source_execution_protocol"] = "other"
    value = publish(workspace, original, cb)
    assert DIRECT.current(policy, member, workspace, value, evidence)["status"] == "UNKNOWN"


def test_final_reopening_catches_mutation_after_profile_support(tmp_path, selected, monkeypatch):
    policy, member, workspace, original, cb, evidence = native_fixture(tmp_path, selected)
    value = publish(workspace, original, cb)
    support = DOMAIN.train_support

    def change(catalog, profile):
        result = support(catalog, profile)
        (workspace / "original.elf").write_bytes(b"changed after actual structural check")
        return result

    monkeypatch.setattr(DOMAIN, "train_support", change)
    assert DIRECT.current(policy, member, workspace, value, evidence)["status"] == "UNKNOWN"


def test_exact_source_geometry_still_required_after_original_join(tmp_path, selected):
    policy, member, workspace, original, cb, evidence = native_fixture(tmp_path, selected)
    evidence["sources"]["source"]["semantic_profile"]["m"] = 32
    value = publish(workspace, original, cb)
    result = DIRECT.current(policy, member, workspace, value, evidence)
    assert result["status"] == "UNKNOWN" and "exact joint" in result["reason"]
