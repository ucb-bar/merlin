"""The candidate-native model diagnostic must refuse incomplete bindings."""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from merlin.runtime.route_quality import (
    HostComputeUnverified,
    HostComputeViolation,
    require_clean_host_compute,
)
from merlin.targetgen.native_dispatch_accounting import (
    _completed_eligible_tasks,
    _kernel_command_inventory,
    _mandatory_command_blocks,
    _verified_work_functs_by_family,
)
from merlin.targetgen.native_model_execution import (
    NativeModelExecutionError,
    _build_artifacts,
    _digest,
    _frozen_model_policy,
    _functional_engine,
    _host_compute_report,
    _logical_values,
    audit_candidate_source_placement,
    audit_emitted_host_compute,
    execute_candidate_model,
)


def test_logical_leaf_binding_preserves_dtype_shape_and_values():
    raw = np.asarray([[1.25, -2.5], [0.0, 3.0]], dtype=np.float32).tobytes()
    assert _logical_values(raw, tensor="arg0", dtype="f32", shape=(2, 2)) == [[1.25, -2.5], [0.0, 3.0]]
    with pytest.raises(NativeModelExecutionError, match="captured 4 element"):
        _logical_values(raw, tensor="arg0", dtype="f32", shape=(2, 3))


def test_non_whole_program_never_builds_and_leaves_failure_receipt(tmp_path):
    source = tmp_path / "source"
    capture = tmp_path / "capture"
    source.mkdir()
    capture.mkdir()
    out = tmp_path / "output"
    result = execute_candidate_model(
        command_buffer={"kernel_abi": {"kind": "tile"}},
        lowered_mlir_text="module {}",
        capsule_dir=source,
        capture_bundle=capture,
        target="example",
        out_dir=out,
    )
    assert result["status"] == "incomplete"
    assert "whole_program" in result["failure"]["detail"]
    assert json.loads((out / "result.json").read_text()) == result


def test_existing_artifact_directory_is_not_overwritten(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    (out / "prior.txt").write_text("do not overwrite")
    with pytest.raises(NativeModelExecutionError, match="fresh artifact directory"):
        execute_candidate_model(
            command_buffer={},
            lowered_mlir_text="",
            capsule_dir=tmp_path,
            capture_bundle=tmp_path,
            target="example",
            out_dir=out,
        )
    assert (out / "prior.txt").read_text() == "do not overwrite"


def test_partial_build_artifacts_are_pinned_after_failure(tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    (build / "kernel.ll").write_text("define void @candidate() { ret void }\n")
    inventory = _build_artifacts(tmp_path)
    assert set(inventory) == {"kernel.ll"}
    assert inventory["kernel.ll"]["size_bytes"] == (build / "kernel.ll").stat().st_size
    assert inventory["kernel.ll"]["sha256"]


def test_selected_functional_engine_binds_binary_and_extension_bytes(tmp_path, monkeypatch):
    from merlin.runtime.backends import base as backends
    from merlin.targetgen import oracle_policy

    binary = tmp_path / "spike"
    binary.write_bytes(b"functional engine")
    extension = tmp_path / "libmodel.so"
    extension.write_bytes(b"selected extension")
    backend = SimpleNamespace(
        available=lambda engine: engine == "spike",
        run_elf=lambda *_args, **_kwargs: "OUT x 1\nDONE\n",
        spike_path=lambda: binary,
        spike_extension=lambda: ((f"--extlib={extension}", "--extension=model"), tmp_path),
    )
    monkeypatch.setattr(backends, "get_backend", lambda target: backend)
    monkeypatch.setattr(oracle_policy, "selected_sim_via", lambda target: "chipyard")
    monkeypatch.setattr(oracle_policy, "oracle_tier_plan", lambda target, via: SimpleNamespace(tiers=("L2", "L3")))
    chosen, citation, revalidate = _functional_engine("synthetic")
    assert chosen is backend
    assert citation["binary"]["sha256"] == _digest(binary)["sha256"]
    assert citation["extension"]["sha256"] == _digest(extension)["sha256"]
    revalidate()
    extension.write_bytes(b"different extension")
    with pytest.raises(NativeModelExecutionError, match="changed"):
        revalidate()


def test_candidate_l2_pinned_full_output_mismatch_is_fail_not_unverified(tmp_path, monkeypatch):
    """Reopen a synthetic frozen golden and actual OUT/DONE bytes, not a claimed numeric flag."""
    import hashlib

    from merlin.runtime.backends import spike
    from merlin.targetgen import native_model_execution as native
    from merlin.targetgen.capsule_golden import compare
    from merlin.targetgen.golden_store import load_golden, write_golden

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "capsule.yaml").write_text("numeric_policy:\n  compare: exact_int\n")
    write_golden(capsule, {"golden_source": "synthetic_test_only", "outputs": {"out": [[1, 2]]}})
    elf = tmp_path / "candidate.elf"
    elf.write_bytes(b"synthetic candidate ELF for binding test")
    console = tmp_path / "console_l2.txt"
    console.write_text("OUT out 1 2 1 3\nDONE\n")
    cb = tmp_path / "command_buffer.json"
    cb.write_text(json.dumps({"tensors": {"out": {"dtype": "i8", "role": "output"}}}))
    lowered = tmp_path / "lowered.llvm.mlir"
    lowered.write_text("builtin.module {}")
    observed, _ = spike.parse_output(console.read_text())
    numeric = compare(
        load_golden(capsule)["outputs"], observed, {"compare": "exact_int"}, golden_source="synthetic_test_only"
    )
    assert numeric["status"] == "fail" and numeric["mismatch_count"] == 1
    policy = {"required_tiers": ["L0", "L1", "L2", "L3"], "test": "synthetic"}
    citation = {"binary": {"sha256": "synthetic_test_only"}}
    receipt = {
        "source": {
            "capsule_declaration": _digest(capsule / "capsule.yaml"),
            "golden": _digest(capsule / "golden.yaml"),
            "golden_arrays": _digest(capsule / "golden.npz"),
        },
        "frozen_policy": policy,
        "elf": _digest(elf),
        "tiers": {
            "L2": {
                "status": "fail",
                "engine": "spike",
                "engine_citation": citation,
                "elf": _digest(elf),
                "console": _digest(console),
                "numeric": numeric,
            }
        },
    }
    emission = {"command_buffer": _digest(cb), "lowered_mlir": _digest(lowered)}
    monkeypatch.setattr(
        native, "audit_candidate_static_tiers", lambda *a, **k: {"L0": {"status": "pass"}, "L1": {"status": "pass"}}
    )
    monkeypatch.setattr(native, "_frozen_model_policy", lambda *a, **k: policy)
    monkeypatch.setattr(
        native,
        "_functional_engine",
        lambda *a, **k: (SimpleNamespace(parse_output=spike.parse_output), citation, lambda: None),
    )

    def l2_status():
        return native.audit_candidate_tiers(
            emission,
            {},
            receipt,
            target="synthetic",
            entry_symbol="kernel",
            completed_dispatch={"status": "unverified"},
        )["tiers"]["L2"]["status"]

    assert l2_status() == "fail"
    from merlin.runtime.backends import base as backends
    from merlin.targetgen.capsule_grade import candidate_native_model_check

    expected = {
        "command_buffer_sha256": hashlib.sha256(
            json.dumps(json.loads(cb.read_text()), sort_keys=True).encode()
        ).hexdigest(),
        "lowered_mlir_sha256": _digest(lowered)["sha256"],
    }
    receipt["candidate"] = expected
    monkeypatch.setattr(
        backends,
        "harness_build_recipe",
        lambda target: SimpleNamespace(require_kernel_stack_frame=lambda: SimpleNamespace(entry_symbol="kernel")),
    )
    monkeypatch.setattr(
        native, "audit_emitted_host_compute", lambda *a, **k: {"status": "clean", "candidate": expected}
    )
    monkeypatch.setattr(
        native,
        "audit_candidate_source_placement",
        lambda *a, **k: {
            "status": "clean",
            "source_sha256": "s" * 64,
            "n_source_operations": 1,
            "eligible_source_op_indices": [0],
        },
    )
    monkeypatch.setattr(native, "audit_candidate_completed_dispatch", lambda *a, **k: {"status": "unverified"})
    check = candidate_native_model_check(
        {
            "candidate_emission": emission,
            "candidate_source_eligibility": {},
            "candidate_native_execution": receipt,
        },
        target="synthetic",
    )
    assert check["status"] == "fail"
    assert "candidate_verified_numeric_mismatch" in check["violations"]
    assert "candidate_full_model_native_unverified" in check["violations"]  # L3 did not run
    receipt["console"] = _digest(console)
    receipt["numeric"] = numeric
    receipt["simulator"] = "synthetic_rtl"
    receipt["simulator_provenance"] = {
        "selection": {"engine": "synthetic_rtl"},
        "citation": {"binary": "synthetic_rtl"},
    }
    receipt["tiers"]["L3"] = {
        "status": "fail",
        "engine": "synthetic_rtl",
        "elf": _digest(elf),
        "console": _digest(console),
        "numeric": numeric,
        "selection": receipt["simulator_provenance"]["selection"],
        "engine_citation": receipt["simulator_provenance"]["citation"],
    }
    dispatch = {"status": "verified", "numeric_status": "fail", "linked_elf": _digest(elf), "console": _digest(console)}
    both = native.audit_candidate_tiers(
        emission, {}, receipt, target="synthetic", entry_symbol="kernel", completed_dispatch=dispatch
    )
    assert both["status"] == "fail"
    assert both["failed_tiers"] == ["L2", "L3"]
    assert "derived_from_rtl" not in both["tiers"]["L3"]  # selection declared no fidelity
    receipt["simulator_provenance"]["selection"]["fidelity"] = "elaborated_rtl"
    both = native.audit_candidate_tiers(
        emission, {}, receipt, target="synthetic", entry_symbol="kernel", completed_dispatch=dispatch
    )
    assert both["tiers"]["L3"]["derived_from_rtl"] is True
    assert "cycle_accurate" not in both["tiers"]["L3"]
    receipt["tiers"]["L3"]["engine_citation"] = {"binary": "forged"}
    assert (
        native.audit_candidate_tiers(
            emission, {}, receipt, target="synthetic", entry_symbol="kernel", completed_dispatch=dispatch
        )["tiers"]["L3"]["status"]
        == "unverified"
    )
    receipt["tiers"]["L3"]["engine_citation"] = receipt["simulator_provenance"]["citation"]
    console.write_text("OUT out 1 2 1 4\nDONE\n")  # stale content pin
    assert l2_status() == "unverified"
    console.write_text("OUT out 1 2 1 3\n")  # bound but no completion witness
    receipt["tiers"]["L2"]["console"] = _digest(console)
    assert l2_status() == "unverified"
    console.write_text("OUT out 1 2 1 2\nDONE\n")  # forged failure claim against correct output
    receipt["tiers"]["L2"]["console"] = _digest(console)
    assert l2_status() == "unverified"
    console.write_text("OUT out 1 2 1 3\nDONE\n")
    receipt["tiers"]["L2"]["console"] = _digest(console)
    receipt["tiers"]["L2"]["elf"] = {**_digest(elf), "sha256": "0" * 64}
    assert l2_status() == "unverified"
    receipt["tiers"]["L2"]["elf"] = _digest(elf)
    del receipt["source"]["golden_arrays"]  # archive exists but its pin was omitted
    assert l2_status() == "unverified"
    receipt["source"]["golden_arrays"] = _digest(capsule / "golden.npz")
    del receipt["source"]["golden"]
    assert l2_status() == "unverified"


def test_frozen_tier_policy_allows_explicit_empty_prohibitions_but_not_empty_tiers(tmp_path, monkeypatch):
    from merlin.targetgen import target_experiment

    descriptor = tmp_path / "target_experiment.yaml"
    descriptor.write_text("target: synthetic\n")
    corpus = tmp_path / "corpus"
    (corpus / "isa").mkdir(parents=True)
    (corpus / "MANIFEST.yaml").write_text(
        "instruction_policy:\n  status: resolved\n  prohibited_instruction_roles: []\n"
    )
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    declaration = capsule / "capsule.yaml"
    declaration.write_text("expected:\n  instruction_classes: [MAC]\nrequired_oracle_tiers: [L0, L1, L2, L3]\n")
    (capsule / "expected_instruction_coverage.yaml").write_text("instruction_classes: [MAC]\n")
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(descriptor))
    monkeypatch.setattr(
        target_experiment,
        "load_target_experiment",
        lambda path: SimpleNamespace(target="synthetic", capsule_corpus=corpus / "isa"),
    )
    policy = _frozen_model_policy(capsule, target="synthetic")
    assert policy["prohibited_instruction_roles"] == []
    assert policy["required_tiers"] == ["L0", "L1", "L2", "L3"]
    declaration.write_text("expected:\n  instruction_classes: [MAC]\nrequired_oracle_tiers: []\n")
    with pytest.raises(NativeModelExecutionError, match="required oracle tiers"):
        _frozen_model_policy(capsule, target="synthetic")


def test_only_family_matched_roles_count_as_device_work():
    endpoint = {
        "engine": "spatial",
        "exposure": "rocc",
        "roles": {"config": ["CFG"], "sync": ["FLUSH"], "loop_descriptor": ["LOOP_CONFIG"], "accumulate": ["MAC"]},
    }
    table = {"names": {0: "CFG", 1: "FLUSH", 2: "LOOP_CONFIG", 3: "MAC"}}
    work = _verified_work_functs_by_family([endpoint], table)
    assert work["contraction"] == {3}
    assert work["elementwise_map"] == set()
    assert work["movement"] == set()
    with pytest.raises(NativeModelExecutionError, match="verified spatial"):
        _verified_work_functs_by_family([{"engine": "host", "exposure": "rocc"}], table)


def test_candidate_compute_command_must_be_on_every_returning_cfg_path():
    from xdsl.dialects.llvm import LLVM

    from merlin.frontends.linalg_mlir import make_context, parse_mlir_text

    context = make_context()
    context.load_dialect(LLVM)
    artifact = """builtin.module {
      llvm.func @kernel(%p: !llvm.ptr, %c: i1) {
        llvm.cond_br %c, ^work, ^exit {merlin.global_task = 0 : i64}
      ^work:
        %v = llvm.load %p {merlin.global_task = 0 : i64} : !llvm.ptr -> i64
        llvm.br ^exit {merlin.global_task = 0 : i64}
      ^exit:
        llvm.return
      }
    }"""
    module = parse_mlir_text(artifact, context)
    function = next(op for op in module.body.block.ops if op.name == "llvm.func")
    command = next(op for op in function.walk() if op.name == "llvm.load")
    assert not _mandatory_command_blocks(function, {command})  # zero-trip branch bypasses work
    entry_command = next(op for op in function.walk() if op.name == "llvm.cond_br")
    assert _mandatory_command_blocks(function, {entry_command})


def test_kernel_inventory_requires_symbol_scoped_derived_command_words(tmp_path, monkeypatch):
    import subprocess

    from merlin.llvmlower import toolchain

    path = tmp_path / "kernel.o"
    path.write_bytes(b"object bytes are read by the disassembler")
    disassembler = tmp_path / "llvm-objdump"
    disassembler.write_bytes(b"selected tool pin")
    monkeypatch.setattr(toolchain, "objdump", lambda: disassembler)
    commands = []

    def disassemble(argv, **kwargs):
        commands.append(argv)
        return subprocess.CompletedProcess(
            argv, 0, "Disassembly of section .text:\n00000000 <kernel>:\n  0: 0800002b <unknown>\n", ""
        )

    monkeypatch.setattr(subprocess, "run", disassemble)
    table = {"custom_opcode": 0x2B, "legal_funct": [4], "names": {"4": "MAC"}}
    words, functs = _kernel_command_inventory(path, symbol="kernel", table=table)
    assert words == [0x0800002B] and functs == [4]
    assert "--disassemble-symbols=kernel" in commands[0]
    table["legal_funct"] = []
    with pytest.raises(NativeModelExecutionError, match="unrecognized custom command"):
        _kernel_command_inventory(path, symbol="kernel", table=table)


def test_grouped_eligible_regions_each_need_distinct_mandatory_work_command():
    from xdsl.dialects.llvm import LLVM

    from merlin.frontends.linalg_mlir import make_context, parse_mlir_text

    context = make_context()
    context.load_dialect(LLVM)
    module = parse_mlir_text(
        """builtin.module { llvm.func @kernel(%p: !llvm.ptr) {
      %a = llvm.load %p : !llvm.ptr -> i64
      %b = llvm.load %p : !llvm.ptr -> i64
      llvm.return
    } }""",
        context,
    )
    function = next(op for op in module.body.block.ops if op.name == "llvm.func")
    first, second = [op for op in function.walk() if op.name == "llvm.load"]
    placement = {
        "eligible_source_op_indices": [0, 1],
        "eligible_source_regions": [
            {"source_op_index": 0, "region_id": "r0", "semantic_family": "contraction"},
            {"source_op_index": 1, "region_id": "r1", "semantic_family": "contraction"},
        ],
        "task_source_regions": [{"task_index": 0, "source_op_indices": [0, 1]}],
    }
    with pytest.raises(NativeModelExecutionError, match="operation 1 in region r1"):
        _completed_eligible_tasks(function, placement, {0: {"contraction": {first}}}, {0: {"r0": {first}}})
    completed = _completed_eligible_tasks(
        function, placement, {0: {"contraction": {first, second}}}, {0: {"r0": {first}, "r1": {second}}}
    )
    assert completed[0]["eligible_source_region_ids"] == ["r0", "r1"]
    placement["eligible_source_regions"][1]["semantic_family"] = "elementwise_map"
    with pytest.raises(NativeModelExecutionError, match="elementwise_map device command"):
        _completed_eligible_tasks(
            function, placement, {0: {"contraction": {first, second}}}, {0: {"r0": {first}, "r1": {second}}}
        )
    placement["eligible_source_regions"][1]["semantic_family"] = "contraction"
    placement["eligible_source_regions"][1]["region_id"] = "r0"
    with pytest.raises(NativeModelExecutionError, match="operation 0 in region r0"):
        _completed_eligible_tasks(
            function, placement, {0: {"contraction": {first, second}}}, {0: {"r0": {first, second}}}
        )
    assert _completed_eligible_tasks(
        function,
        placement,
        {0: {"contraction": {first, second}}},
        {0: {"r0": {first, second}}},
        {0: {0: {first}, 1: {second}}},
    )[0]["eligible_source_region_ids"] == ["r0", "r0"]


def test_task_scoped_host_compute_distinguishes_addressing_host_island_and_violation():
    cb = {
        "kernel_abi": {"kind": "whole_program", "args": [{"tensor": "arg0", "access": "read"}]},
        "params": {
            "global_program_plan": {
                "tasks": [
                    {"task_index": 0, "kind": "contraction"},
                    {"task_index": 1, "kind": "host"},
                ]
            }
        },
    }
    body = """
        %one = llvm.mlir.constant(1 : i64) : i64
        %index = llvm.add %n, %one {merlin.global_task = 0 : i64} : i64
        %ptr = llvm.getelementptr %t[%index] {merlin.global_task = 0 : i64}
            : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %value = llvm.load %ptr {merlin.global_task = 1 : i64} : !llvm.ptr -> i64
        %sum = llvm.add %value, %one {merlin.global_task = 1 : i64} : i64
        llvm.store %sum, %ptr {merlin.global_task = 1 : i64} : i64, !llvm.ptr
        llvm.return
    """

    def artifact(ir):
        return "builtin.module { llvm.func @gemmini_kernel(%t: !llvm.ptr, %n: i64) {" + ir + "} }"

    clean = _host_compute_report(cb, artifact(body), entry_symbol="gemmini_kernel")
    assert require_clean_host_compute(clean) is clean
    assert clean.coverage == 1  # host island is legitimate, not hidden accelerator compute

    illegal = body.replace(
        "%sum = llvm.add %value, %one {merlin.global_task = 1 : i64}",
        "%sum = llvm.add %value, %one {merlin.global_task = 0 : i64}",
    )
    with pytest.raises(HostComputeViolation):
        require_clean_host_compute(_host_compute_report(cb, artifact(illegal), entry_symbol="gemmini_kernel"))

    unscoped = body.replace("{merlin.global_task = 1 : i64}", "")
    with pytest.raises(HostComputeUnverified):
        require_clean_host_compute(_host_compute_report(cb, artifact(unscoped), entry_symbol="gemmini_kernel"))


def test_emitted_host_audit_reopens_exact_pinned_artifacts(tmp_path):
    cb = {
        "kernel_abi": {"kind": "whole_program", "args": [{"tensor": "arg0", "access": "read"}]},
        "params": {"global_program_plan": {"tasks": [{"task_index": 0, "kind": "contraction"}]}},
    }
    artifact = (
        "builtin.module { llvm.func @kernel(%t: !llvm.ptr, %n: i64) { "
        "%one = llvm.mlir.constant(1 : i64) : i64 "
        "%index = llvm.add %n, %one {merlin.global_task = 0 : i64} : i64 "
        "%ptr = llvm.getelementptr %t[%index] {merlin.global_task = 0 : i64} "
        ": (!llvm.ptr, i64) -> !llvm.ptr, i64 llvm.return } }"
    )
    cb_path, artifact_path = tmp_path / "command_buffer.json", tmp_path / "lowered.llvm.mlir"
    cb_path.write_text(json.dumps(cb))
    artifact_path.write_text(artifact)
    pins = {"command_buffer": _digest(cb_path), "lowered_mlir": _digest(artifact_path)}
    assert audit_emitted_host_compute(pins, entry_symbol="kernel")["status"] == "clean"
    violating = (
        "builtin.module { llvm.func @kernel(%t: !llvm.ptr, %n: i64) { "
        "%one = llvm.mlir.constant(1 : i64) : i64 "
        "%value = llvm.load %t {merlin.global_task = 0 : i64} : !llvm.ptr -> i64 "
        "%sum = llvm.add %value, %one {merlin.global_task = 0 : i64} : i64 "
        "llvm.store %sum, %t {merlin.global_task = 0 : i64} : i64, !llvm.ptr "
        "llvm.return } }"
    )
    artifact_path.write_text(violating)
    pins["lowered_mlir"] = _digest(artifact_path)
    assert audit_emitted_host_compute(pins, entry_symbol="kernel")["status"] == "violation"
    artifact_path.write_text(artifact.replace("llvm.getelementptr", "llvm.load"))
    assert audit_emitted_host_compute(pins, entry_symbol="kernel")["status"] == "unverified"


def test_device_required_model_without_candidate_native_evidence_is_incomplete():
    from merlin.targetgen.capsule_grade import model_execution_check

    checked = model_execution_check({}, {"semantic": {"must_accelerate": True}})
    assert "candidate_emitted_host_compute_unverified" in checked["violations"]
    assert "candidate_full_model_native_unverified" in checked["violations"]


def test_score_numeric_status_never_borrows_legacy_model_numeric():
    from merlin.targetgen.capsule_grade import _score_numeric_status

    row = {
        "numeric": {"status": "pass"},
        "candidate_native_model_check": {"status": "incomplete", "violations": ["candidate_required_tiers_unverified"]},
    }
    assert _score_numeric_status(row) is None
    row["candidate_native_model_check"].update(status="fail", violations=["candidate_verified_numeric_mismatch"])
    assert _score_numeric_status(row) == "fail"
    row["candidate_native_model_check"].update(status="pass", violations=[])
    assert _score_numeric_status(row) == "pass"
    del row["candidate_native_model_check"]
    assert _score_numeric_status(row) == "pass"


def test_real_grade_rollup_withholds_legacy_tiers_cost_and_numeric_for_candidate(tmp_path, monkeypatch):
    from merlin.targetgen import capsule_grade as grade_module

    package, capsules = tmp_path / "package", tmp_path / "capsules"
    package.mkdir()
    capsules.mkdir()
    capsule = {"name": "M", "kind": "model", "label": "public", "semantic": {"must_accelerate": True}}
    row = {
        "capsule": "M",
        "kind": "model",
        "label": "public",
        "status": "fail",
        "numeric": {"status": "pass"},
        "tiers": {
            tier: {"status": "pass", "cycles": 5000, "derived_from_rtl": tier == "L3", "timing": {"sim_active_s": 8.0}}
            for tier in ("L0", "L1", "L2", "L3")
        },
        "mesh_execution": {"matmul_layers_on_mesh": 99},
        "candidate_native_model_check": {
            "schema": "merlin_candidate_native_model_check_v1",
            "status": "fail",
            "violations": ["candidate_verified_numeric_mismatch"],
            "candidate_required_tiers": {
                "status": "fail",
                "required_tiers": ["L0", "L1", "L2", "L3"],
                "tiers": {
                    "L0": {"status": "pass"},
                    "L1": {"status": "pass"},
                    "L2": {"status": "fail"},
                    "L3": {"status": "unverified"},
                },
            },
        },
    }
    monkeypatch.setattr(grade_module, "load_package", lambda *a, **k: SimpleNamespace(integrity_exempt=False))
    monkeypatch.setattr(grade_module, "integrity_scan", lambda *a, **k: None)
    monkeypatch.setattr(grade_module, "build_package", lambda *a, **k: None)
    monkeypatch.setattr(grade_module.CR, "discover_capsules", lambda *a, **k: [capsule])
    monkeypatch.setattr(grade_module.CR, "run_suite", lambda *a, **k: [row])
    monkeypatch.setattr(grade_module, "enforce_model_execution_check", lambda result, *a, **k: result)
    trace = tmp_path / "runs" / "runs" / "gemmini-capsule-bench" / "M" / "generated" / "instruction_trace.json"
    trace.parent.mkdir(parents=True)
    trace.write_text(json.dumps({"legacy_host_graph_trace": True}))
    aggregate_inputs = {}

    def aggregate(rows, *, traces, **kwargs):
        aggregate_inputs["rows"] = rows
        aggregate_inputs["traces"] = traces
        return {
            "by_tier_reached": {},
            "instruction_class_coverage": {},
            "mode_coverage": {},
            "unavailable": {},
            "acceleratable_coverage": {},
        }

    monkeypatch.setattr(grade_module.CV, "aggregate", aggregate)
    score = grade_module.grade(
        package, capsules_root=capsules, runs_root=tmp_path / "runs", target="gemmini", oracle_adapters={}
    )
    assert score["n_capsules"] == 1 and score["n_passed"] == 0
    assert score["tier_reached"]["L3"] == 0 and score["highest_tier"] == "L1"
    assert score["pass_evidence"]["rtl_backed"] == 0
    assert score["cycles_diagnostic"] == {} and score["timing_diagnostic"] == {}
    assert score["numeric_all_exact"] is False
    model = score["per_capsule"][0]
    assert model["numeric"] == "fail" and model["tiers"]["L2"] == "fail"
    assert model["tiers"]["L3"] == "unverified"
    assert "mesh_execution" not in model and "cost_plane" not in model
    assert aggregate_inputs["rows"][0]["tiers"]["L3"]["status"] == "unverified"
    assert aggregate_inputs["traces"] == {}  # legacy generated trace never describes candidate ELF
    assert row["tiers"]["L3"]["cycles"] == 5000  # diagnostic source is preserved, not scored

    # Even a candidate pass cannot inherit RTL identity from an L3 name when
    # the selected-engine audit supplied no fidelity declaration.
    row["status"] = "pass"
    check = row["candidate_native_model_check"]
    check["status"] = "pass"
    check["violations"] = []
    check["candidate_required_tiers"]["status"] = "pass"
    check["candidate_required_tiers"]["tiers"]["L2"] = {"status": "pass"}
    check["candidate_required_tiers"]["tiers"]["L3"] = {"status": "pass"}
    score = grade_module.grade(
        package, capsules_root=capsules, runs_root=tmp_path / "runs", target="gemmini", oracle_adapters={}
    )
    assert score["n_passed"] == 1
    assert score["pass_evidence"]["rtl_backed"] == 0
    assert score["pass_evidence"]["rtl_tiers_seen"] == []
    assert aggregate_inputs["traces"] == {}


def test_candidate_score_view_carries_only_audited_rtl_fidelity():
    from merlin.targetgen.capsule_grade import _candidate_score_view

    row = {
        "kind": "model",
        "candidate_native_model_check": {
            "status": "pass",
            "candidate_required_tiers": {"required_tiers": ["L3"], "tiers": {"L3": {"status": "pass"}}},
        },
    }
    tier = _candidate_score_view(row)["tiers"]["L3"]
    assert "derived_from_rtl" not in tier and "cycle_accurate" not in tier
    row["candidate_native_model_check"]["candidate_required_tiers"]["tiers"]["L3"].update(
        fidelity="elaborated_rtl", derived_from_rtl=True
    )
    tier = _candidate_score_view(row)["tiers"]["L3"]
    assert tier["fidelity"] == "elaborated_rtl" and tier["derived_from_rtl"] is True
    assert "cycle_accurate" not in tier


def test_model_grade_accepts_real_native_candidate_receipt_extra_fields(monkeypatch):
    from merlin.runtime.backends import base as backends
    from merlin.targetgen import native_model_execution as native_module
    from merlin.targetgen.capsule_grade import (
        cycles_by_tier,
        enforce_model_execution_check,
        model_execution_check,
    )

    monkeypatch.setattr(
        backends,
        "harness_build_recipe",
        lambda target: SimpleNamespace(require_kernel_stack_frame=lambda: SimpleNamespace(entry_symbol="kernel")),
    )
    expected = {"command_buffer_sha256": "a" * 64, "lowered_mlir_sha256": "b" * 64}
    monkeypatch.setattr(
        native_module, "audit_emitted_host_compute", lambda *a, **k: {"status": "clean", "candidate": expected}
    )
    monkeypatch.setattr(
        native_module,
        "audit_candidate_source_placement",
        lambda *a, **k: {
            "status": "clean",
            "source_sha256": "s" * 64,
            "n_source_operations": 2,
            "eligible_source_op_indices": [0],
        },
    )
    monkeypatch.setattr(
        native_module,
        "audit_candidate_completed_dispatch",
        lambda *a, **k: {
            "status": "verified",
            "source_sha256": "s" * 64,
            "numeric_status": "pass",
            "eligible_tasks": [{"source_op_indices": [0, 1]}],
        },
    )
    tier_proof = {
        "status": "pass",
        "required_tiers": ["L0", "L1", "L2", "L3"],
        "tiers": {tier: {"status": "pass"} for tier in ("L0", "L1", "L2", "L3")},
    }
    monkeypatch.setattr(native_module, "audit_candidate_tiers", lambda *a, **k: tier_proof)
    record = {
        "candidate": {**expected, "entry_arity": 35, "abi_args": 35},
        "status": "numeric_match_diagnostic",
        "numeric": {"status": "pass"},
        "elf": {"path": "/exact/elf"},
        "console": {"path": "/exact/console"},
    }
    row = {
        "candidate_native_execution": record,
        "candidate_emission": {},
        "coverage_certificate": {},
        "operation": {"target": "synthetic"},
    }
    checked = model_execution_check(row, {"semantic": {"must_accelerate": True}}, target="synthetic")
    assert "candidate_full_model_native_unverified" not in checked["violations"]
    assert checked["candidate_native_model_check"]["status"] == "pass"
    assert checked["status"] == "pass"  # the independently proven candidate is authoritative
    tier_proof["status"] = "unverified"
    tier_proof["missing_tiers"] = ["L2"]
    checked = model_execution_check(row, {"semantic": {"must_accelerate": True}}, target="synthetic")
    assert "candidate_required_tiers_unverified" in checked["violations"]
    assert checked["candidate_native_model_check"]["status"] == "incomplete"
    tier_proof["status"] = "pass"
    tier_proof["missing_tiers"] = []
    row["status"] = "incomplete"
    row["failure"] = {"plane": "legacy_model", "detail": "runner-owned graph failed"}
    row["legacy_model_diagnostic"] = {"status": "incomplete"}
    row["coverage_certificate"] = {"n_eligible_accelerated": 99, "scope": "legacy"}
    row["mesh_execution"] = {"matmul_layers_on_mesh": 99, "scope": "legacy"}
    row["tiers"] = {"L3": {"status": "pass", "cycles": 42_000_000, "scope": "legacy"}}
    enforce_model_execution_check(row, {"semantic": {"must_accelerate": True}}, target="synthetic")
    assert row["status"] == "pass"
    assert "failure" not in row
    assert set(row["tiers"]) == {"L0", "L1", "L2", "L3"}
    assert all(row["tiers"][tier]["status"] == "pass" for tier in row["tiers"])
    assert row["legacy_model_diagnostic"]["status"] == "incomplete"
    assert row["legacy_model_diagnostic"]["artifacts"]["coverage_certificate"]["n_eligible_accelerated"] == 99
    assert row["legacy_model_diagnostic"]["artifacts"]["mesh_execution"]["matmul_layers_on_mesh"] == 99
    assert row["legacy_model_diagnostic"]["artifacts"]["tiers"]["L3"]["cycles"] == 42_000_000
    assert "coverage_certificate" not in row and "mesh_execution" not in row
    assert row["candidate_source_coverage"]["n_completed_eligible"] == 1
    assert cycles_by_tier(row["tiers"]) == {}
    record["candidate"]["lowered_mlir_sha256"] = "c" * 64
    checked = model_execution_check(row, {"semantic": {"must_accelerate": True}}, target="synthetic")
    assert "candidate_full_model_native_unverified" in checked["violations"]
    assert checked["candidate_native_model_check"]["status"] == "incomplete"

    # A separately verified functional-tier counterexample is a candidate
    # failure even if the unrelated legacy host-graph path was incomplete.
    record["candidate"]["lowered_mlir_sha256"] = expected["lowered_mlir_sha256"]
    tier_proof["status"] = "fail"
    tier_proof["failed_tiers"] = ["L2"]
    tier_proof["tiers"]["L2"] = {"status": "fail"}
    row["status"] = "incomplete"
    enforce_model_execution_check(row, {"semantic": {"must_accelerate": True}}, target="synthetic")
    assert row["status"] == "fail"
    assert row["failure"]["plane"] == "candidate_model_numeric"
    assert row["failure"]["category"] == "FUNCTIONAL_MISMATCH"
    assert "candidate_verified_numeric_mismatch" in row["candidate_native_model_check"]["violations"]


def test_known_emitted_accelerator_task_tensor_math_is_failure_even_without_native_run(tmp_path, monkeypatch):
    from merlin.runtime.backends import base as backends
    from merlin.targetgen.capsule_grade import enforce_model_execution_check

    cb_path, artifact_path = tmp_path / "command_buffer.json", tmp_path / "lowered.llvm.mlir"
    cb_path.write_text(
        json.dumps(
            {
                "kernel_abi": {"kind": "whole_program", "args": [{"tensor": "arg0", "access": "read"}]},
                "params": {"global_program_plan": {"tasks": [{"task_index": 0, "kind": "contraction"}]}},
            }
        )
    )
    artifact_path.write_text(
        "builtin.module { llvm.func @kernel(%t: !llvm.ptr) { "
        "%one = llvm.mlir.constant(1 : i64) : i64 "
        "%value = llvm.load %t {merlin.global_task = 0 : i64} : !llvm.ptr -> i64 "
        "%sum = llvm.add %value, %one {merlin.global_task = 0 : i64} : i64 "
        "llvm.store %sum, %t {merlin.global_task = 0 : i64} : i64, !llvm.ptr "
        "llvm.return } }"
    )
    monkeypatch.setattr(
        backends,
        "harness_build_recipe",
        lambda target: SimpleNamespace(require_kernel_stack_frame=lambda: SimpleNamespace(entry_symbol="kernel")),
    )
    row = {
        "status": "pass",
        "tiers": {},
        "operation": {"target": "synthetic"},
        "candidate_emission": {"command_buffer": _digest(cb_path), "lowered_mlir": _digest(artifact_path)},
    }
    result = enforce_model_execution_check(row, {"semantic": {"must_accelerate": True}}, target="synthetic")
    assert result["status"] == "fail"
    assert result["failure"]["plane"] == "model_host_compute"
    assert "candidate_host_tensor_compute_violation" in result["model_execution_check"]["violations"]


def test_all_host_relabel_cannot_hide_independently_eligible_source_operation(tmp_path, monkeypatch):
    import copy
    import hashlib

    from merlin.targetgen import native_model_execution as native_module
    from merlin.targetgen import target_registry

    source = (
        "module { func.func @main(%x: tensor<1xf32>) -> tensor<1xf32> { "
        '%a = "test.eligible"(%x) {prov.region_id = "r0"} '
        ": (tensor<1xf32>) -> tensor<1xf32> "
        '%b = "test.glue"(%a) {prov.region_id = "r1"} '
        ": (tensor<1xf32>) -> tensor<1xf32> "
        "func.return %b : tensor<1xf32> } }"
    )
    source_path = tmp_path / "model.mlir"
    source_path.write_text(source)
    (tmp_path / "capsule.yaml").write_text("interface_mlir: model.mlir\n")
    contract_path = tmp_path / "target_contract.yaml"
    contract_path.write_text("name: synthetic\n")
    monkeypatch.setattr(
        target_registry, "resolve", lambda target: SimpleNamespace(capability_contract_path=contract_path)
    )
    cb_path = tmp_path / "command_buffer.json"
    artifact_path = tmp_path / "lowered.llvm.mlir"
    artifact_path.write_text("builtin.module {}")
    plan = {
        "schema": "mixed_program_plan_v1",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "source_op_count": 2,
        "tasks": [
            {"task_index": 0, "kind": "host", "source_op_indices": [0]},
            {"task_index": 1, "kind": "host", "source_op_indices": [1]},
        ],
    }
    cb_path.write_text(json.dumps({"kernel_abi": {"kind": "whole_program"}, "params": {"global_program_plan": plan}}))
    emission = {
        "capsule_declaration": _digest(tmp_path / "capsule.yaml"),
        "source_interface": _digest(source_path),
        "command_buffer": _digest(cb_path),
        "lowered_mlir": _digest(artifact_path),
    }
    certificate = {
        "target": "synthetic",
        "source_mlir_sha256": plan["source_sha256"],
        "denominator_source": "semantic_capabilities (independent eligibility oracle)",
        "source_inventory_scope": "frozen_source_only_no_legacy_execution",
        "eligibility_capability_contract": _digest(contract_path),
        "n_unknown_capture_formats": 0,
        "n_precision_transform_obligations": 0,
        "precision_transform_verification": {"status": "not_required"},
        "n_regions": 2,
        "regions": [
            {
                "region_id": "r0",
                "carrier_op": "test.eligible",
                "target_eligible": True,
                "semantic_family": "contraction",
                "precision_transform_required": False,
                "captured_input_format": "int8",
                "captured_weight_format": "int8",
                "eligibility_input_format": "int8",
                "eligibility_weight_format": "int8",
            },
            {
                "region_id": "r1",
                "carrier_op": "test.glue",
                "target_eligible": False,
                "precision_transform_required": False,
            },
        ],
    }
    monkeypatch.setattr(native_module, "independent_frozen_source_eligibility", lambda *_args, **_kwargs: certificate)
    verdict = audit_candidate_source_placement(emission, certificate, target="synthetic")
    assert verdict["status"] == "violation"
    assert verdict["eligible_host_source_op_indices"] == [0]
    from merlin.targetgen.capsule_grade import enforce_model_execution_check

    graded = enforce_model_execution_check(
        {
            "status": "pass",
            "tiers": {},
            "operation": {"target": "synthetic"},
            "candidate_emission": emission,
            "candidate_source_eligibility": certificate,
        },
        {"semantic": {"must_accelerate": True}},
        target="synthetic",
    )
    assert graded["status"] == "fail"
    assert graded["failure"]["plane"] == "model_source_placement"
    assert "candidate_source_placement_violation" in graded["model_execution_check"]["violations"]

    plan["tasks"][0]["kind"] = "contraction"
    cb_path.write_text(json.dumps({"kernel_abi": {"kind": "whole_program"}, "params": {"global_program_plan": plan}}))
    emission["command_buffer"] = _digest(cb_path)
    assert audit_candidate_source_placement(emission, certificate, target="synthetic")["status"] == "clean"

    certificate["regions"][0]["precision_transform_required"] = None
    certificate["n_unknown_capture_formats"] = 1
    certificate["precision_transform_verification"]["status"] = "unknown_capture_format"
    unknown = audit_candidate_source_placement(emission, certificate, target="synthetic")
    assert unknown["status"] == "unverified"
    assert "unknown capture precision" in unknown["detail"]
    certificate["regions"][0]["precision_transform_required"] = False
    certificate["n_unknown_capture_formats"] = 0
    certificate["precision_transform_verification"]["status"] = "not_required"
    assert audit_candidate_source_placement(emission, certificate, target="synthetic")["status"] == "clean"

    certificate["regions"][0]["captured_weight_format"] = None
    missing_weight = audit_candidate_source_placement(emission, certificate, target="synthetic")
    assert missing_weight["status"] == "unverified"
    assert "weight-format admission" in missing_weight["detail"]
    certificate["regions"][0]["captured_weight_format"] = "int8"
    assert audit_candidate_source_placement(emission, certificate, target="synthetic")["status"] == "clean"

    plan["tasks"] = plan["tasks"][:1]
    cb_path.write_text(json.dumps({"kernel_abi": {"kind": "whole_program"}, "params": {"global_program_plan": plan}}))
    emission["command_buffer"] = _digest(cb_path)
    missing = audit_candidate_source_placement(emission, certificate, target="synthetic")
    assert missing["status"] == "violation"
    assert "candidate tasks omit source operations" in missing["problems"]
    plan["tasks"].append({"task_index": 1, "kind": "host", "source_op_indices": [1]})

    plan["tasks"][1]["source_op_indices"] = [0]
    cb_path.write_text(json.dumps({"kernel_abi": {"kind": "whole_program"}, "params": {"global_program_plan": plan}}))
    emission["command_buffer"] = _digest(cb_path)
    assert audit_candidate_source_placement(emission, certificate, target="synthetic")["status"] == "violation"
    frozen_certificate = copy.deepcopy(certificate)
    monkeypatch.setattr(
        native_module, "independent_frozen_source_eligibility", lambda *_args, **_kwargs: frozen_certificate
    )
    certificate["regions"][0]["target_eligible"] = False
    assert audit_candidate_source_placement(emission, certificate, target="synthetic")["status"] == "unverified"
    certificate["regions"][0]["target_eligible"] = True
    monkeypatch.setattr(native_module, "independent_frozen_source_eligibility", lambda *_args, **_kwargs: certificate)

    certificate["source_mlir_sha256"] = "1" * 64
    assert audit_candidate_source_placement(emission, certificate, target="synthetic")["status"] == "unverified"
    certificate["source_mlir_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    contract_path.write_text("name: synthetic\nsemantic_capabilities: []\n")
    assert audit_candidate_source_placement(emission, certificate, target="synthetic")["status"] == "unverified"
    certificate["eligibility_capability_contract"] = _digest(contract_path)

    plan["tasks"][1]["source_op_indices"] = [1]
    plan["source_sha256"] = "0" * 64
    cb_path.write_text(json.dumps({"kernel_abi": {"kind": "whole_program"}, "params": {"global_program_plan": plan}}))
    emission["command_buffer"] = _digest(cb_path)
    assert audit_candidate_source_placement(emission, certificate, target="synthetic")["status"] == "violation"


@pytest.mark.parametrize("legacy_status", ["pass", "incomplete"])
def test_model_runner_emits_submitted_whole_program_before_pass_admission(tmp_path, monkeypatch, legacy_status):
    from merlin.targetgen import capsule_runner as runner
    from merlin.targetgen import native_model_execution as native

    generated, run_path = tmp_path / "generated", tmp_path / "run"
    generated.mkdir()
    (tmp_path / "capsule.interface.mlir").write_text("module {}")
    (tmp_path / "capsule.yaml").write_text("name: model\nkind: model\n")
    monkeypatch.setattr(
        runner, "make_run_paths", lambda *a, **k: SimpleNamespace(generated=generated, run_path=run_path)
    )
    monkeypatch.setattr(
        runner,
        "_grade_model_capsule",
        lambda capsule, **kw: {
            "capsule": capsule["name"],
            "kind": "model",
            "status": legacy_status,
            "tiers": {},
            "failure": {"plane": "legacy_model", "category": "DIAGNOSTIC", "detail": "legacy path"}
            if legacy_status != "pass"
            else None,
        },
    )
    calls = []

    def emit(pkg, package_dir, capsule, paths, **kwargs):
        calls.append("submitted_four_stage_entrypoints")
        cb = {"kernel_abi": {"kind": "whole_program"}}
        (generated / "command_buffer.json").write_text(json.dumps(cb))
        (generated / "lowered.llvm.mlir").write_text("builtin.module {}")
        return object(), cb, "builtin.module {}"

    @contextmanager
    def bundle(capsule, *, timeout):
        yield tmp_path, {"construction": "frozen"}, lambda: None

    def execute(**kwargs):
        calls.append("submitted_native_build")
        assert kwargs["simulator"] is None
        return {"status": "compiled_not_run"}

    monkeypatch.setattr(runner, "run_entrypoints", emit)
    monkeypatch.setattr(runner, "_model_runtime_bundle", bundle)
    monkeypatch.setattr(native, "execute_candidate_model", execute)
    for key in ("MERLIN_MODEL_NATIVE_SIMULATOR", "MERLIN_MODEL_NATIVE_RTL_FACTS", "MERLIN_MODEL_NATIVE_BOARD_CONFIG"):
        monkeypatch.delenv(key, raising=False)
    config = SimpleNamespace(
        target="synthetic",
        suite="synthetic-capsules",
        dtype="i8",
        force_match_policy=None,
        fourth_output_name="lowered.llvm.mlir",
    )
    capsule = {"name": "model", "kind": "model", "__dir__": str(tmp_path), "semantic": {"must_accelerate": True}}
    result = runner.run_capsule(
        capsule, tmp_path, runs_root=tmp_path, target="synthetic", config=config, oracle_adapters={}
    )
    assert calls == ["submitted_four_stage_entrypoints", "submitted_native_build"]
    assert result["status"] == "incomplete"
    assert result["candidate_emission"]["lowered_mlir"]["sha256"]
    assert result["legacy_model_diagnostic"]["status"] == legacy_status
    assert result["candidate_source_eligibility_failure"]  # malformed synthetic source is not borrowed
    assert "candidate_full_model_native_unverified" in result["model_execution_check"]["violations"]
