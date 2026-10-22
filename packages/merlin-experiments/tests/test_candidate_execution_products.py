"""Neutral retained-role custody controls; no accelerator execution is claimed."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy

import pytest
from merlin_experiments.phase1.feedback import candidate_execution_products as P
from merlin_experiments.phase1.feedback import candidate_observations as O
from merlin_experiments.phase1.feedback.candidate_execution_products import reopen_execution_products

from merlin.common import invocation_record as I
from merlin.common.tree_hash import hash_tree
from merlin.targetgen import executed_program as E
from merlin.targetgen.contract.interface_emit import parse_interface_mlir
from merlin.targetgen.contract.readback_policy import canonical_sha256

SOURCE = '''module attributes {merlin_iface.version = "0.1", merlin_iface.target = "fixture"} {
%w = merlin_iface.tensor {name = "w", role = "weight"} : tensor<3x4xi8>
%a = merlin_iface.tensor {name = "a", role = "input"} : tensor<2x3xi8>
%r = merlin_iface.resident_pack %w {layout = "packed_rhs"} : (tensor<3x4xi8>) -> !merlin_iface.resident
%acc = merlin_iface.matmul %a, %r : (tensor<2x3xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
%y = merlin_iface.commit %acc {name = "y", epilogue = [], output_dtype = "i32"} :
 (!merlin_iface.acc<i32>) -> tensor<2x4xi32>
merlin_iface.evict %r : (!merlin_iface.resident) -> ()
}'''
LOWERED = "module { llvm.func @kernel_entry() { llvm.return } }"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def neutral_call():
    return b"neutral fixture transport\n"


@pytest.fixture(params=["subprocess", "python_call"])
def retained(tmp_path, request):
    root = tmp_path / "generated"
    root.mkdir()
    for name, raw in {"package_kernel.elf": b"neutral opaque ELF fixture", "harness.c": b"neutral caller",
                      "kernel.o": b"neutral kernel object"}.items():
        (root / name).write_bytes(raw)
    (root / "input.interface.mlir").write_text(SOURCE)
    (root / "lowered.llvm.mlir").write_text(LOWERED)
    raw = parse_interface_mlir(SOURCE)
    effective = deepcopy(raw)
    effective["tensors"]["y"] = dict(shape=[2, 4], dtype="i32", role="output")
    (root / "command_buffer.json").write_text(json.dumps(raw))
    build = dict(status="completed", elf_sha256=sha(root / "package_kernel.elf"),
                 harness_sha256=sha(root / "harness.c"), kernel_object_name="kernel.o",
                 kernel_object_sha256=sha(root / "kernel.o"), command_buffer_sha256=canonical_sha256(effective))
    build["build_identity_sha256"] = canonical_sha256(build)
    # Real subprocess/receipt transport over neutral bytes; it does not execute
    # the opaque fixture or establish any source/numerical/accelerator verdict.
    if request.param == "subprocess":
        run = I.run([sys.executable, "-I", "-B", "-c", "print('neutral fixture transport')"],
                    directory=root, stage="execution", inputs=(root / "package_kernel.elf",),
                    capture_output=True, env={})
    else:
        with I.observe_call(root, stage="execution", function=neutral_call, arguments={},
                            inputs=(root / "package_kernel.elf",)) as receipt:
            console = neutral_call()
            receipt.returned(stdout=console)
        run = subprocess.CompletedProcess([], 0, stdout=console)
    invocation = next((root / "invocations").glob("*/invocation.json"))
    result = dict(elf=str(root / "package_kernel.elf"), readback_build=build, console=run.stdout,
                  oracle=dict(provenance=dict(sha256="1" * 64)),
                  execution_invocation=dict(path=str(invocation), sha256=sha(invocation)))
    E.retain(result, sim="fixture", workdir=root,
             compiler_artifacts=dict(command_buffer=effective, lowered_mlir=LOWERED))
    compiler = tmp_path / "compiler"
    compiler.mkdir()
    (compiler / "compiler.py").write_text("raise RuntimeError('neutral source custody fixture')\n")
    candidate_sha = hash_tree(compiler)["sha256"]
    shutil.copytree(compiler, root / "compiler-source" / candidate_sha)
    grade = dict(kind="layer", label="public", tiers=dict(L2=dict(status="pass", measured_now=True,
                 sim_provenance=dict(engine="fixture", sha256="1" * 64))),
                 compiler_source=dict(status="stable", candidate_sha256=candidate_sha, package=str(compiler),
                     snapshot="compiler-source/" + candidate_sha,
                     artifacts={name: sha(root / name) for name in
                                ("input.interface.mlir", "lowered.llvm.mlir", "command_buffer.json")},
                     executed_programs=E.collect(root)))
    assert grade["compiler_source"]["executed_programs"][0]["compiler_artifacts"]["status"] == "retained"
    return root, grade, candidate_sha


def test_exact_effective_roles_replace_ambiguous_basename_without_authority(retained):
    root, grade, expected = retained
    legacy = O.observe_generated_artifacts(root)
    assert "command_buffer" in legacy["product_collection"]["unavailable_roles"]
    report = O.observe_generated_artifacts(root, original_grade=grade, candidate_sha256=expected)
    assert report["product_collection"]["unavailable_roles"] == []
    assert report["candidate_features"]["features"]["source_macs"] == 24
    assert report["representations"]["tensors"]["y"]["shape"] == [2, 4]
    assert report["product_collection"]["execution_join"]["effective_command_buffer"].startswith("completed-call")
    assert report["authority"] == "none"
    assert report["dynamic_execution"] == "UNKNOWN" and report["timing_cycles"] is None
    assert str(root) not in json.dumps(report)


@pytest.mark.parametrize("damage", ["source", "lowered", "retained_cb", "retained_elf", "invocation",
                                  "source_pin", "engine", "not_fresh", "ambiguous", "private", "candidate"])
def test_changed_original_custody_refuses_without_basename_fallback(retained, damage):
    root, grade, expected = retained
    execution = grade["compiler_source"]["executed_programs"][0]
    if damage in {"source", "lowered"}:
        name = "input.interface.mlir" if damage == "source" else "lowered.llvm.mlir"
        (root / name).write_text("changed")
    elif damage in {"retained_cb", "retained_elf"}:
        pin = (execution["elf"] if damage == "retained_elf"
               else execution["compiler_artifacts"]["members"]["command_buffer"])
        (root / pin["path"]).write_text("changed")
    elif damage == "invocation":
        (root / "invocations").joinpath(next((root / "invocations").iterdir()).name, "stdout.bin").write_text("changed")
    elif damage == "source_pin":
        grade["compiler_source"]["artifacts"]["input.interface.mlir"] = "0" * 64
    elif damage == "engine":
        grade["tiers"]["L2"]["sim_provenance"]["sha256"] = "2" * 64
    elif damage == "not_fresh":
        grade["tiers"]["L2"]["measured_now"] = False
    elif damage == "ambiguous":
        grade["compiler_source"]["executed_programs"].append(deepcopy(execution))
    elif damage == "private":
        grade["label"] = "hidden"
    else:
        grade["compiler_source"]["candidate_sha256"] = "b" * 64
    report = O.observe_generated_artifacts(root, original_grade=grade, candidate_sha256=expected)
    assert "original_retained_execution_join" in report["product_collection"]["unavailable_roles"]
    assert report["source"]["status"] == "UNKNOWN"
    with pytest.raises((ValueError, KeyError)):
        reopen_execution_products(root, grade, candidate_sha256=expected)


def replace_retained_role(root, grade, role, payload):
    """Keep collection coherent so the deeper original-producer join is tested."""
    record_path = next((root / "executed").glob("*/*/executed_program.json"))
    record = json.loads(record_path.read_bytes())
    pin = record["compiler_artifacts"]["members"][role]
    path = root / pin["path"]
    path.write_bytes(payload)
    pin.update(sha256=sha(path), bytes=len(payload))
    record_path.write_text(json.dumps(record))
    grade["compiler_source"]["executed_programs"] = E.collect(root)


@pytest.mark.parametrize("role,payload,reason", [
    ("console", b"different producer output", "console differs"),
    ("build", b"[]", "must be a mapping"),
    ("command_buffer", b"[]", "must be a mapping"),
])
def test_coherent_retention_cannot_replace_original_producer(retained, role, payload, reason):
    root, grade, expected = retained
    replace_retained_role(root, grade, role, payload)
    with pytest.raises(ValueError, match=reason):
        reopen_execution_products(root, grade, candidate_sha256=expected)
    report = O.observe_generated_artifacts(root, original_grade=grade, candidate_sha256=expected)
    assert report["source"]["status"] == "UNKNOWN"


@pytest.mark.parametrize("damage", ["identity", "readback_policy"])
def test_completed_build_identity_and_readback_are_independent_joins(retained, damage):
    root, grade, expected = retained
    execution = grade["compiler_source"]["executed_programs"][0]
    path = root / execution["compiler_artifacts"]["members"]["build"]["path"]
    build = json.loads(path.read_bytes())
    if damage == "identity":
        build["build_identity_sha256"] = "0" * 64
    else:
        build["readback_policy"] = {"different": "policy"}
        build["build_identity_sha256"] = canonical_sha256(
            {key: value for key, value in build.items() if key != "build_identity_sha256"})
        record_path = next((root / "executed").glob("*/*/executed_program.json"))
        record = json.loads(record_path.read_bytes())
        record["build_identity_sha256"] = build["build_identity_sha256"]
        record_path.write_text(json.dumps(record))
    replace_retained_role(root, grade, "build", json.dumps(build).encode())
    with pytest.raises(ValueError, match="identity or readback"):
        reopen_execution_products(root, grade, candidate_sha256=expected)


@pytest.mark.parametrize("damage", ["source", "snapshot_missing", "snapshot_link", "raw_buffer"])
def test_original_source_snapshot_and_raw_buffer_remain_mandatory(retained, damage):
    root, grade, expected = retained
    snapshot = root / grade["compiler_source"]["snapshot"]
    if damage == "source":
        (snapshot / "compiler.py").write_text("changed")
    elif damage in {"snapshot_missing", "snapshot_link"}:
        shutil.rmtree(snapshot)
        if damage == "snapshot_link":
            snapshot.symlink_to(grade["compiler_source"]["package"], target_is_directory=True)
    else:
        (root / "command_buffer.json").write_text("{}")
    report = O.observe_generated_artifacts(root, original_grade=grade, candidate_sha256=expected)
    assert report["source"]["status"] == "UNKNOWN"
    assert report["product_collection"]["unavailable_roles"] == ["original_retained_execution_join"]


@pytest.mark.parametrize("damage", ["large_payload", "large_record", "escaping_member", "linked_directory",
                                  "unselected_hardlink"])
def test_namespace_preflight_refuses_before_original_payload_collector(retained, monkeypatch, damage):
    root, grade, expected = retained
    record_path = next((root / "executed").glob("*/*/executed_program.json"))
    if damage == "large_payload":
        with (record_path.parent / "unselected.bin").open("wb") as stream:
            stream.truncate((64 << 20) + 1)
    elif damage == "large_record":
        record_path.write_text(" " * ((64 << 10) + 1))
    elif damage == "escaping_member":
        record = json.loads(record_path.read_bytes())
        record["elf"]["path"] = "../outside.elf"
        record_path.write_text(json.dumps(record))
    elif damage == "linked_directory":
        (root / "executed" / "linked").symlink_to(root / "invocations", target_is_directory=True)
    else:
        os.link(root / "kernel.o", record_path.parent / "unselected.o")

    def forbidden(_root):
        pytest.fail("unbounded original collector must not run")

    monkeypatch.setattr(E, "collect", forbidden)
    with pytest.raises(ValueError):
        reopen_execution_products(root, grade, candidate_sha256=expected)


def test_original_four_field_source_receipt_needs_no_invented_lowered_pin(retained):
    root, grade, expected = retained
    grade["compiler_source"]["artifacts"].pop("lowered.llvm.mlir")
    paths, join = reopen_execution_products(root, grade, candidate_sha256=expected)
    assert paths["emitted"].read_text() == LOWERED
    assert join["qualification"].startswith("byte custody only")


@pytest.mark.parametrize("role", ["tiers", "L2", "simulator", "artifacts"])
def test_malformed_original_nested_roles_are_unavailable(retained, role):
    root, grade, expected = retained
    if role == "tiers":
        grade["tiers"] = ["malformed"]
    elif role == "L2":
        grade["tiers"]["L2"] = ["malformed"]
    elif role == "simulator":
        grade["tiers"]["L2"]["sim_provenance"] = ["malformed"]
    else:
        grade["compiler_source"]["artifacts"] = ["malformed"]
    report = O.observe_generated_artifacts(root, original_grade=grade, candidate_sha256=expected)
    assert report["source"]["status"] == "UNKNOWN"


@pytest.mark.parametrize("tree", ["snapshot", "package"])
def test_source_tree_budget_covers_root_bytecode_before_unbounded_digest(retained, monkeypatch, tree):
    root, grade, expected = retained
    package = grade["compiler_source"]["package"]
    target = root / grade["compiler_source"]["snapshot"] if tree == "snapshot" else root.parent / "compiler"
    with (target / "rogue.pyc").open("wb") as stream:
        stream.truncate((32 << 20) + 1)
    original_hash_tree = P.hash_tree

    def bounded_first(path):
        if path == target:
            pytest.fail("oversized source namespace must be rejected before hash_tree")
        return original_hash_tree(path)

    monkeypatch.setattr(P, "hash_tree", bounded_first)
    with pytest.raises(ValueError, match="source inventory"):
        reopen_execution_products(root, grade, compiler=package, candidate_sha256=expected)


def test_legacy_public_emission_reader_is_unchanged(retained, monkeypatch):
    root, grade, expected = retained
    grade["compiler_source"]["executed_programs"][0]["schema"] = E.SCHEMA
    monkeypatch.setattr(E, "collect", lambda _root: pytest.fail("legacy grade is emission-only"))
    report = O.observe_generated_artifacts(root, original_grade=grade, candidate_sha256=expected)
    assert "command_buffer" in report["product_collection"]["unavailable_roles"]
    assert report["product_collection"]["consumption_join"] == "UNKNOWN"


@pytest.mark.parametrize("damage", [None, "private", "source", "invocation", "engine"])
def test_model_slice_uses_same_exact_retained_custody(retained, damage):
    root, grade, expected = retained
    grade["kind"] = "model_slice"
    if damage == "private":
        grade["label"] = "hidden"
    elif damage == "source":
        grade["compiler_source"]["candidate_sha256"] = "b" * 64
    elif damage == "invocation":
        next((root / "invocations").iterdir()).joinpath("stdout.bin").write_text("changed")
    elif damage == "engine":
        grade["tiers"]["L2"]["sim_provenance"]["sha256"] = "2" * 64
    report = O.observe_generated_artifacts(root, original_grade=grade, candidate_sha256=expected)
    if damage is None:
        assert report["product_collection"]["consumption_join"] == "reopened_original_retained_functional_layer_products"
        assert report["authority"] == "none"
        assert report["timing_cycles"] is None
    else:
        assert "original_retained_execution_join" in report["product_collection"]["unavailable_roles"]
        with pytest.raises(ValueError):
            reopen_execution_products(root, grade, candidate_sha256=expected)
