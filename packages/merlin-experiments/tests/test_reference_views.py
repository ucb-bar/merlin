"""Original source/ABI/artifact joins remain distinct from execution authority."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import reference_analysis as analysis
from merlin_experiments.phase1.feedback import reference_experiments as selected
from merlin_experiments.phase1.feedback.reference_views import artifact_views, compare_views, source_view
from test_reference_analysis_tool import _request, _world

from merlin.kernels.kernel_evidence import ArtifactPin, KernelEvidence
from merlin.targetgen.contract.interface_emit import emit_interface_mlir
from merlin.verify.lean_contract import document_digest


def inputs():
    tensors = dict(w=dict(shape=[4, 3], dtype="i8", role="weight"))
    commands = [dict(opcode="RES_PACK", operands=dict(src="w", dst="resident"),
                     attributes=dict(layout="packed_rhs"))]
    for i in range(2):
        tensors["a" + str(i)] = dict(shape=[2, 4], dtype="i8", role="input")
        tensors["y" + str(i)] = dict(shape=[2, 3], dtype="i32", role="output")
        commands.extend([
            dict(opcode="MATMUL_RESIDENT", operands=dict(lhs="a" + str(i), rhs="resident", dst="acc" + str(i))),
            dict(opcode="COMMIT", operands=dict(src="acc" + str(i), dst="y" + str(i)),
                 attributes=dict(epilogue=[], output_dtype="i32")),
        ])
    commands.append(dict(opcode="EVICT", operands=dict(handle="resident")))
    cb = dict(abi_version="0.1", target="fixture", tensors=tensors, commands=commands)
    source = emit_interface_mlir(cb).encode()
    args = ", ".join("%arg" + str(i) + ": !llvm.ptr" for i in range(5))
    emitted = ("builtin.module { llvm.func @kernel_entry(" + args + ") { llvm.return } }").encode()
    return source, cb, emitted


@pytest.mark.parametrize("role", ["source", "command_buffer", "emitted"])
def test_live_artifact_analysis_reopens_every_candidate_leaf_without_changing_byte_comparisons(tmp_path, role):
    fixture = _world(tmp_path)
    directory = tmp_path / "variants/trial"
    directory.mkdir(parents=True)
    source, cb, emitted = inputs()
    raw = dict(source=source, command_buffer=json.dumps(cb).encode(), emitted=emitted)
    bundle = dict(schema="merlin.reference_artifact_views.v1")
    for name, payload in raw.items():
        path = directory / name
        path.write_bytes(payload)
        bundle[name] = dict(path=path.relative_to(tmp_path).as_posix(), sha256=hashlib.sha256(payload).hexdigest())
    path = directory / "views.json"
    path.write_text(json.dumps(bundle))
    with selected.invocation(fixture["path"], fixture["sha"], tasks=False):
        response = analysis.handle(
            _request("candidate", path.relative_to(tmp_path).as_posix(), format="artifact_views"),
            workspace=tmp_path, compiler_sha256=fixture["compiler"])
        assert response["status"] == "analyzed"
        report = response["report"]
        # Keep source byte equivalence independent of the files' visible names.
        assert report["inputs"] == artifact_views(
            source_mlir=source, command_buffer=raw["command_buffer"], emitted_mlir=emitted)["inputs"]
        assert all("path" not in pin for pin in report["inputs"].values())
        assert report["artifact_inputs"][role]["path"] == bundle[role]["path"]
        before = selected.project(current_compiler_sha256=fixture["compiler"])
        assert before["status"] == "pending_experiment"
        # The manifest and compiler stay unchanged; only one referenced leaf changes.
        (directory / role).write_bytes(raw[role] + b"\n")
        after = selected.project(current_compiler_sha256=fixture["compiler"])
        assert after["status"] == "unknown"
        assert after["reason"] == "selected_reference_input_stale_or_unavailable"


def observe(source=None, cb=None, emitted=None, **kw):
    original_source, original_cb, original_emitted = inputs()
    return artifact_views(source_mlir=source or original_source,
        command_buffer=json.dumps(cb or original_cb).encode(), emitted_mlir=emitted or original_emitted, **kw)


def test_complete_source_operands_outputs_numeric_and_abi_join_without_execution_credit():
    report = observe()
    assert report["source_command_join"] == "matches_declared_source"
    group, = report["source"]["groups"]
    assert group["rhs"] == "w" and len(group["members"]) == 2
    assert list(report["source"]["outputs"]) == ["y0", "y1"]
    assert report["emitted"]["abi_arity"] == 5
    assert report["native"]["command_counts"] is None
    assert report["numerical_status"] == report["timing_status"] == report["mechanism_status"] == "unknown"
    assert report["gaps"][0]["goals"] == ["evidence.native_issue"]


def test_direct_whole_operation_output_uses_original_typed_abi_without_product_capture_credit():
    from merlin_experiments.phase1.feedback.reference_capture_views import candidate_view

    from merlin.compile.logical_abi import kernel_abi_from_commands

    cb = dict(abi_version="0.1", target="fixture", tensors=dict(
        image=dict(shape=[2, 7, 7, 2], dtype="i8", role="input"),
        weight=dict(shape=[18, 3], dtype="i8", role="weight"),
        bias=dict(shape=[3], dtype="i32", role="bias"),
        result=dict(shape=[98, 3], dtype="i8", role="output")), commands=[
            dict(opcode="RES_PACK", operands=dict(src="weight", dst="packed"),
                 attributes=dict(layout="packed_conv_rhs")),
            dict(opcode="CONV2D", operands=dict(ifm="image", weight="packed", dst="result"),
                 attributes=dict(kernel=[3, 3, 2, 3], stride=[1, 1], padding=[1, 1, 1, 1],
                                 dilation=[1, 1], bias="bias", epilogue=["bias_add"],
                                 output_dtype="i8", layout="nhwc")),
            dict(opcode="EVICT", operands=dict(handle="packed")),
        ])
    source = emit_interface_mlir(cb).encode()
    cb["kernel_abi"] = kernel_abi_from_commands(cb)
    emitted = (b"builtin.module { llvm.func @kernel_entry(%a: !llvm.ptr, %b: !llvm.ptr, "
               b"%c: !llvm.ptr, %d: !llvm.ptr) { llvm.return } }")
    report = observe(source=source, cb=cb, emitted=emitted)
    assert report["source_command_join"] == "matches_declared_source"
    assert report["source"]["outputs"] == dict(result=dict(shape=[98, 3], dtype="i8", role="output"))
    assert report["source"]["argument_order"] == ["image", "weight", "bias", "result"]
    assert report["source"]["mechanism_observations"]["shared_input_coverage"]["status"] == "partial"
    capture, cca = candidate_view(dict(source=source, emitted=emitted), entry_symbol="kernel_entry", decoder={})
    assert cca is None and capture["status"] == "unknown" and report["source"]["products"] == 0
    assert report["semantic_equivalence"] == report["numerical_status"] == "unknown"
    cb["kernel_abi"]["outputs"] = []
    refused = observe(source=source, cb=cb, emitted=emitted)
    assert refused["source_command_join"] == "mismatch"
    assert dict(axis="source.abi_roster", status="mismatch") in refused["mismatches"]


@pytest.mark.parametrize("mutation,axis", [
    ("input_shape", "source.tensor"), ("output_shape", "source.tensor"),
    ("output_dtype", "source.tensor"), ("numeric", "source.commands"),
    ("missing_liveout", "source.tensor"), ("missing_numeric", "source.commands"),
    ("operand", "source.commands"), ("abi", "source.abi_roster"),
])
def test_candidate_declarations_cannot_replace_original_source_contract(mutation, axis):
    source, cb, _ = inputs()
    if mutation == "input_shape":
        cb["tensors"]["a0"]["shape"] = [2, 5]
    elif mutation == "output_shape":
        cb["tensors"]["y0"]["shape"] = [2, 4]
    elif mutation == "output_dtype":
        cb["tensors"]["y0"]["dtype"] = "i8"
    elif mutation == "numeric":
        cb["commands"][2]["attributes"]["epilogue"] = ["relu"]
    elif mutation == "missing_numeric":
        del cb["commands"][2]["attributes"]["output_dtype"]
    elif mutation == "missing_liveout":
        del cb["tensors"]["y1"]
    elif mutation == "operand":
        cb["commands"][1]["operands"]["lhs"] = "a1"
    else:
        cb["kernel_abi"] = dict(args=[dict(tensor="w")], outputs=["y0"])
    report = observe(source=source, cb=cb)
    assert report["source_command_join"] == "mismatch"
    assert axis in {row["axis"] for row in report["mismatches"]}
    assert compare_views(observe(), report)["issue_comparison"]["fence_count"]["status"] == "unknown"


def test_wrong_actual_entry_arity_is_a_mismatch_not_zero_work_or_success():
    _, _, emitted = inputs()
    report = observe(emitted=emitted.replace(b", %arg4: !llvm.ptr", b""))
    assert report["mismatches"] == [dict(axis="emitted.abi_arity", expected=5, actual=4, status="mismatch")]


def capture_for(source, cb, emitted, *, arity=5):
    raw = b'{"funct":1,"rs1":0,"rs2":0}\n{"kind":"fence"}\n'
    isa = dict(FUNCT_CLASS={1: "MOVE"})
    artifacts = tuple(ArtifactPin(kind, "original/" + kind, hashlib.sha256(data).hexdigest())
        for kind, data in dict(source=source, command_buffer=json.dumps(cb).encode(), emitted=emitted,
                              capture=raw).items())
    artifacts += (ArtifactPin("isa", "original/selected-isa", document_digest(isa)),)
    evidence = KernelEvidence("original-native", capture_kind="native_issue", artifacts=artifacts)
    return dict(raw=raw, evidence=evidence, abi_arity=arity), isa


def test_concrete_native_records_are_decoded_but_pins_do_not_authenticate_execution():
    source, cb, emitted = inputs()
    capture, isa = capture_for(source, cb, emitted)
    report = observe(native_capture=capture, isa=isa)
    assert report["native"]["command_counts"] == dict(MOVE=1)
    assert report["native"]["fence_count"] == 1
    assert report["native"]["status"] == "recorded_unverified_execution"
    assert report["native"]["original_producer_custody"] == "unverified"
    assert report["mechanism_status"] == "unknown"
    capture["abi_arity"] = 4
    assert observe(native_capture=capture, isa=isa)["native"]["reason"] == "native_capture_abi_arity_mismatch"


def test_wrong_capture_source_pin_and_incomplete_isa_stay_unknown():
    source, cb, emitted = inputs()
    capture, isa = capture_for(source, cb, emitted)
    bad = copy.copy(capture)
    bad["evidence"] = KernelEvidence("wrong-source", capture_kind="native_issue", artifacts=tuple(
        ArtifactPin(pin.kind, pin.uri, "a" * 64) if pin.kind == "source" else pin
        for pin in capture["evidence"].artifacts))
    assert observe(native_capture=bad, isa=isa)["native"]["reason"] == "native_capture_artifact_binding_mismatch"
    assert observe(native_capture=capture, isa={})["native"]["command_counts"] is None


def test_raw_event_counts_survive_recurrence_compaction_and_partial_scope_cannot_close_region_goal():
    from dataclasses import replace

    source, cb, emitted = inputs()
    capture, isa = capture_for(source, cb, emitted)
    capture["raw"] *= 10
    capture["evidence"] = replace(capture["evidence"], artifacts=tuple(
        replace(pin, sha256=hashlib.sha256(capture["raw"]).hexdigest()) if pin.kind == "capture" else pin
        for pin in capture["evidence"].artifacts))
    report = observe(native_capture=capture, isa=isa)
    assert report["native"]["command_counts"] == dict(MOVE=10)
    assert report["native"]["fence_count"] == 10
    comparison = compare_views(report, report)
    assert comparison["issue_comparison"]["fence_count"]["status"] == "recorded_partial_or_incompatible_scope"
    assert comparison["gaps"][-1]["kind"] == "adapter_gap"


def test_source_rhs_identity_does_not_authorize_use_after_evict():
    _, cb, _ = inputs()
    evict = cb["commands"].pop()
    cb["commands"].insert(1, evict)
    with pytest.raises(ValueError, match="source product operands are unresolved"):
        source_view(emit_interface_mlir(cb).encode())


@pytest.mark.parametrize("stages", [["acc_scale"], ["relu"], ["acc_scale", "relu"], ["requant"]])
def test_shape_preserving_source_epilogues_cannot_hide_wrong_result_geometry(stages):
    _, cb, _ = inputs()
    cb["commands"][2]["attributes"]["epilogue"] = stages
    source = emit_interface_mlir(cb).encode()
    assert source_view(source)["outputs"]["y0"]["shape"] == [2, 3]
    # Mutate the actual source declaration and matching candidate metadata:
    # agreeing declarations cannot make a geometrically invalid source valid.
    source = source.replace(b"tensor<2x3xi32>", b"tensor<2x5xi32>")
    cb["tensors"]["y0"]["shape"] = [2, 5]
    with pytest.raises(ValueError, match="analysis_source_readout_geometry_mismatch"):
        observe(source=source, cb=cb)


def test_existing_selected_endpoint_reopens_actual_inputs_and_enforces_off_on_and_changed_pins(tmp_path):
    fixture = _world(tmp_path)
    folder = tmp_path / "variants/trial"
    folder.mkdir(parents=True)
    source, cb, emitted = inputs()
    bundle = dict(schema="merlin.reference_artifact_views.v1")
    for role, data in dict(source=source, command_buffer=json.dumps(cb).encode(), emitted=emitted).items():
        path = folder / (role + ".txt")
        path.write_bytes(data)
        bundle[role] = dict(path=path.relative_to(tmp_path).as_posix(), sha256=hashlib.sha256(data).hexdigest())
    path = folder / "views.json"
    path.write_text(json.dumps(bundle))
    request = _request("candidate", path.relative_to(tmp_path).as_posix(), format="artifact_views")
    assert analysis.handle(request, workspace=tmp_path)["reason"] == "reference_guidance_disabled"
    with selected.invocation(fixture["path"], fixture["sha"]):
        response = analysis.handle(request, workspace=tmp_path, compiler_sha256=fixture["compiler"])
        (folder / "source.txt").write_bytes(source + b"\n")
        changed = analysis.handle(request, workspace=tmp_path, compiler_sha256=fixture["compiler"])
    assert response["status"] == "analyzed" and response["report"]["experiment"]["tasks"]
    assert response["qualification_issued"] is False
    assert changed["reason"] == "analysis_artifact_pin_mismatch"


def test_broker_uses_host_descriptor_capability_not_capsule_contract_directory(tmp_path, monkeypatch):
    from merlin.targetgen import target_experiment
    from merlin.targetgen.rocc import semantics

    fixture = _world(tmp_path)
    capsule_contract = tmp_path / "capsule-contract"
    capsule_contract.mkdir()
    capability = tmp_path / "target.yaml"
    capability.write_text('name: fixture\n')
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: fixture\n")
    experiment = SimpleNamespace(target="fixture", declared_contract="target.yaml",
                                 declared_contract_path=lambda: capability)
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda path: experiment)
    class Observer:
        def isa_constants(self, target):
            return dict(CUSTOM_OPCODE=1, FUNCT3=0, FUNCT_CLASS={})

        def decode_instruction(self, *args):
            return "UNKNOWN", {}
    def bind(**kw):
        assert kw["contract"] == dict(name="fixture")  # read actual capability YAML, not directory
        return SimpleNamespace(operand_observer=Observer())
    monkeypatch.setattr(semantics, "bind", bind)
    folder = tmp_path / "variants/current"
    folder.mkdir(parents=True)
    source, cb, emitted = inputs()
    emitted = emitted.replace(b"llvm.return", b'"llvm.inline_asm"() <{asm_string = "fence", '
        b'constraints = "", has_side_effects}> : () -> () llvm.return')
    bundle = dict(schema="merlin.reference_artifact_views.v1")
    for role, data in dict(source=source, command_buffer=json.dumps(cb).encode(), emitted=emitted).items():
        path = folder / role
        path.write_bytes(data)
        bundle[role] = dict(path=path.relative_to(tmp_path).as_posix(), sha256=hashlib.sha256(data).hexdigest())
    path = folder / "views.json"
    path.write_text(json.dumps(bundle))
    with selected.invocation(fixture["path"], fixture["sha"]):
        response = analysis.broker_response(_request("candidate", path.relative_to(tmp_path).as_posix(),
            format="artifact_views"), workspace=tmp_path, facts=fixture["facts"],
            compiler_sha256=fixture["compiler"], rid="broker-capability", started_ns=time.time_ns(),
            contract=capsule_contract, descriptor=descriptor, target="fixture")
    assert response["status"] == "analyzed"
    assert response["report"]["emitted"]["selected_decode"]["command_counts"] == dict(FENCE=1)
    assert response["report"]["native"]["command_counts"] is None


@pytest.mark.skipif(not os.environ.get("MERLIN_TEST_ATTENTION_ARTIFACT_ROOT"), reason="operator-retained source")
@pytest.mark.parametrize("arm", ["fresh7ref", "fresh7refcca"])
@pytest.mark.parametrize("case", ["X_att_qk_h32_gqa4_m16k64n16", "X_att_pv_h32_gqa4_m16k16n64"])
def test_real_retained_grouped_attention_source_command_and_full_abi(arm, case):
    root = Path(os.environ["MERLIN_TEST_ATTENTION_ARTIFACT_ROOT"]) / arm / case
    source = (root / "input.mlir").read_bytes()
    cb = json.loads((root / "command_buffer.json").read_bytes())
    emitted = (root / "kernel.llvm.mlir").read_bytes()
    report = artifact_views(source_mlir=source, command_buffer=json.dumps(cb).encode(), emitted_mlir=emitted)
    assert report["source_command_join"] == "matches_declared_source"
    assert report["source"]["products"] == len(report["source"]["outputs"]) == 32
    assert len(report["source"]["groups"]) == 4 and report["emitted"]["abi_arity"] == 68
    assert all(len(group["members"]) == 8 for group in report["source"]["groups"])
    wrong = emitted.replace(b"%67: !llvm.ptr", b"%67: i64", 1)
    if wrong != emitted:
        with pytest.raises(ValueError, match="analysis_emitted_invalid"):
            artifact_views(source_mlir=source, command_buffer=json.dumps(cb).encode(), emitted_mlir=wrong)
    assert source_view(source)["outputs"] == report["source"]["outputs"]
