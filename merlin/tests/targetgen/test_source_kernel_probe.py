"""A projected kernel must remain tied to one exact captured MLIR body."""

# ruff: noqa: E501 -- the multiline fixture preserves exact emitted MLIR lines.

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from subprocess import CompletedProcess
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root
from merlin.runtime.simulator import simulate
from merlin.targetgen.contract.interface_emit import emit_interface_mlir, parse_interface_mlir
from merlin.targetgen.contract.model_kernel_outline import outline_integer_matmuls
from merlin.targetgen.contract.model_kernel_route import probe_integer_model_kernels
from merlin.targetgen.contract.model_stitching import stitching_inventory
from merlin.targetgen.contract.resident_interface_abi import bind_single_resident_matmul
from merlin.targetgen.source_kernel_probe import derive_kernel_window

pytestmark = pytest.mark.target("gemmini", "atlas")

_INTEGER_BODY = """builtin.module {
  func.func @forward(%a: tensor<4x19xi8>, %b: tensor<19x8xi8>) -> tensor<4x8xi32> {
    %zero = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %init = "tensor.splat"(%zero) : (i32) -> tensor<4x8xi32>
    %result = "linalg.generic"(%a, %b, %init) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%lhs: i8, %rhs: i8, %acc: i32):
      %lhs32 = "arith.extsi"(%lhs) : (i8) -> i32
      %rhs32 = "arith.extsi"(%rhs) : (i8) -> i32
      %product = "arith.muli"(%lhs32, %rhs32) : (i32, i32) -> i32
      %sum = "arith.addi"(%acc, %product) : (i32, i32) -> i32
      "linalg.yield"(%sum) : (i32) -> ()
    }) {prov.op = "int_matmul", prov.source_node_ids = ["g:prepared:root:n7"]} : (tensor<4x19xi8>, tensor<19x8xi8>, tensor<4x8xi32>) -> tensor<4x8xi32>
    func.return %result : tensor<4x8xi32>
  }
}"""


def _integer_capture(tmp_path, *, body=_INTEGER_BODY):
    from merlin.common import mlir_query as query

    model = body.encode()
    (tmp_path / "model.mlir").write_bytes(model)
    module = query.parse(body)
    matches = [(index, op) for index, op in enumerate(query.walk(module))
               if query.op_name(op) == "linalg.generic"]
    assert len(matches) == 1
    ordinal, op = matches[0]
    trace = {
        "status": "complete",
        "graphs": {"prepared": {"nodes": [{"id": "g:prepared:root:n7", "target": "aten._int_mm.default"}]}},
        "mlir": {"sha256": hashlib.sha256(model).hexdigest(), "operations": [{
            "ordinal": ordinal, "operation": "linalg.generic",
            "source_node_ids": ["g:prepared:root:n7"],
            "origin_node_ids": ["g:original:root:n6"],
            "operand_types": [str(value.type) for value in op.operands],
            "result_types": [str(value.type) for value in op.results],
        }]},
    }
    (tmp_path / "frontend-trace.json").write_text(json.dumps(trace), encoding="utf-8")
    return tmp_path


def test_exact_model_route_probe_keeps_ssa_bindings_and_separates_kernel_from_model(tmp_path, monkeypatch):
    from merlin.targetgen.contract import model_kernel_route as route

    source = repo_root() / "examples/gemmini/target"
    package = tmp_path / "selected-package"
    package.mkdir()
    (package / "manifest.yaml").write_text("target: gemmini\n", encoding="utf-8")
    monkeypatch.setattr(
        route.package_runtime, "load_package",
        lambda _path: SimpleNamespace(target="gemmini", package_id="test-package", directory=package),
    )

    def emit(_package, _name, input_mlir, output_json, **_kw):
        text = input_mlir.read_text(encoding="utf-8")
        if "merlin_iface." in text:
            command = parse_interface_mlir(text)
        else:
            command = {"target": "gemmini", "commands": [], "declined": {"op": "model", "reason": "unrouted"}}
        output_json.write_text(json.dumps(command), encoding="utf-8")
        return CompletedProcess([], 0, "", "")

    monkeypatch.setattr(route.package_runtime, "run_entrypoint", emit)
    report = probe_integer_model_kernels(
        _INTEGER_BODY.encode(),
        target="gemmini",
        software_spec=(source / "software-spec.yaml").read_bytes(),
        capability_contract=(source / "contracts/target_contract.yaml").read_bytes(),
        package_dir=package,
    )
    assert report["model_sha256"] == hashlib.sha256(_INTEGER_BODY.encode()).hexdigest()
    assert report["candidate_count"] == report["distinct_interfaces"] == 1
    assert report["emission_counts"] == {"isolated_kernel_emitted": 1}
    candidate = report["candidates"][0]
    assert candidate["operand_bindings"] == [
        {"source": "function_argument", "argument_index": 0},
        {"source": "function_argument", "argument_index": 1},
    ]
    assert candidate["output_result_id"] == candidate["operation_id"] + ":result:0"
    assert [command["opcode"] for command in candidate["emission"]["commands"]] == [
        "RES_PACK", "MATMUL_RESIDENT", "COMMIT"
    ]
    assert report["complete_model_direct_emission"] == {
        "status": "declined", "declined": {"op": "model", "reason": "unrouted"}
    }
    assert report["stitching"]["composition_status"] == "unlowered"
    assert report["whole_model_offload_verified"] is False


def test_staged_admission_binds_development_evidence_without_promoting_unknown_sw(tmp_path, monkeypatch):
    from merlin.llvmlower import staged_admission as staged
    from merlin.llvmlower.exact_offload import ExactOffloadSelection

    source = repo_root() / "examples/gemmini/target"
    model = _INTEGER_BODY.encode()
    spec = (source / "software-spec.yaml").read_bytes()
    contract = (source / "contracts/target_contract.yaml").read_bytes()
    outline = outline_integer_matmuls(model, target="gemmini", software_spec=spec, capability_contract=contract)
    candidate = outline["candidates"][0]
    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.yaml").write_text("target: gemmini\n", encoding="utf-8")
    monkeypatch.setattr(staged.package_runtime, "load_package", lambda _path: SimpleNamespace(
        target="gemmini", package_id="test-package", directory=package,
    ))
    monkeypatch.setattr(staged, "validate_facts", lambda _doc, *, target: [])
    rtl_facts = json.dumps({"inputs": {"target": "gemmini"}, "facts": {
        "arrays": [{"rows": 16, "cols": 16}],
    }}).encode()
    seen = []

    def emit(_package, name, input_mlir, output_json=None, **_kw):
        interface = input_mlir.read_text(encoding="utf-8")
        seen.append((name, hashlib.sha256(interface.encode()).hexdigest()))
        if name == "emit_command_buffer":
            output_json.write_text(json.dumps(parse_interface_mlir(interface)), encoding="utf-8")
            return CompletedProcess([], 0, "", "")
        assert name == "emit_target_artifact"
        return CompletedProcess([], 0, "module { llvm.func @gemmini_kernel(%arg0: !llvm.ptr) {} }", "")

    monkeypatch.setattr(staged.package_runtime, "run_entrypoint", emit)
    report = staged.stage_integer_model_admission(
        model, target="gemmini", software_spec=spec, capability_contract=contract,
        package_dir=package, operation_id=candidate["operation_id"], rtl_facts=rtl_facts,
    )
    binding = report["exact_binding"]
    assert report["candidate_count"] == 1 and report["candidate_admission_counts"] == {"unknown": 1}
    assert binding["model_sha256"] == hashlib.sha256(model).hexdigest()
    assert binding["operation_id"] == candidate["operation_id"]
    assert binding["interface_sha256"] == candidate["interface_sha256"]
    assert hashlib.sha256(report["selected_interface_mlir"].encode()).hexdigest() == binding["interface_sha256"]
    assert binding["rtl_facts_sha256"] == hashlib.sha256(rtl_facts).hexdigest()
    assert seen == [("emit_command_buffer", binding["interface_sha256"]),
                    ("emit_target_artifact", binding["interface_sha256"])]
    assert report["source_facts"]["logical_shapes"] == {"A": [4, 19], "B": [19, 8], "Y": [4, 8]}
    assert report["source_facts"]["physical_layout"] == "not_observed_in_tensor_ssa"
    assert report["source_facts"]["physical_aliasing"] == "not_observed_in_tensor_ssa"
    assert report["compiler_evidence"]["status"] == "emitted_unverified"
    assert report["compiler_evidence"]["binding_sha256"] == binding["binding_sha256"]
    assert report["shim_evidence"]["status"] == "generated_unexecuted"
    assert report["shim_evidence"]["binding_sha256"] == binding["binding_sha256"]
    assert report["shim_evidence"]["rtl_facts_provenance"]["status"] == "unverified"
    assert report["shim_evidence"]["rtl_facts_provenance"]["reported_source_consistency"] == "not_reported"
    assert set(report["codegen_obligations"]) == {"layouts", "tails", "aliasing"}
    assert all(row["status"] == "requires_review" for row in report["codegen_obligations"].values())
    assert report["software_admission"]["status"] == "unknown"
    boundary = report["admission_boundary"]
    assert boundary["phase0_semantic_screen"]["status"] == "diagnostic_observation_not_phase0_admission"
    assert boundary["phase0_semantic_screen"]["unresolved_physical_constraints"] == [
        "aliasing", "layouts", "tails"
    ]
    assert boundary["phase1_physical_plan"]["status"] == "generated_not_executed"
    assert "cannot upgrade" in boundary["promotion_policy"]
    assert report["review_required"] is True and report["whole_model_offload_verified"] is False
    without_facts = staged.stage_integer_model_admission(
        model, target="gemmini", software_spec=spec, capability_contract=contract,
        package_dir=package, operation_id=candidate["operation_id"],
    )
    assert without_facts["shim_evidence"]["status"] == "declined"
    assert without_facts["exact_binding"]["rtl_facts_sha256"] is None
    assert without_facts["exact_binding"]["binding_sha256"] != binding["binding_sha256"]
    wrong_facts = json.dumps({"inputs": {"target": "atlas"}, "facts": {
        "arrays": [{"rows": 16, "cols": 16}],
    }}).encode()
    with pytest.raises(ValueError, match="exact candidate target"):
        staged.stage_integer_model_admission(
            model, target="gemmini", software_spec=spec, capability_contract=contract,
            package_dir=package, operation_id=candidate["operation_id"], rtl_facts=wrong_facts,
        )
    rectangular_facts = json.dumps({"inputs": {"target": "gemmini"}, "facts": {
        "arrays": [{"rows": 16, "cols": 32}],
    }}).encode()
    with pytest.raises(ValueError, match="non-square array"):
        staged.stage_integer_model_admission(
            model, target="gemmini", software_spec=spec, capability_contract=contract,
            package_dir=package, operation_id=candidate["operation_id"], rtl_facts=rectangular_facts,
        )
    reported_verified_facts = json.dumps({"inputs": {"target": "gemmini"}, "facts": {
        "arrays": [{"rows": 16, "cols": 16}],
    }, "source_consistency": {"status": "verified"}}).encode()
    verified_report = staged.stage_integer_model_admission(
        model, target="gemmini", software_spec=spec, capability_contract=contract,
        package_dir=package, operation_id=candidate["operation_id"], rtl_facts=reported_verified_facts,
    )
    assert verified_report["shim_evidence"]["rtl_facts_provenance"]["status"] == (
        "reported_verified_not_rechecked"
    )
    with pytest.raises(ValueError, match="reviewed SW admission"):
        ExactOffloadSelection.from_outline(
            outline, model=model, target="gemmini", software_spec=spec, capability_contract=contract,
            package_dir=package, operation_ids=(candidate["operation_id"],),
        )
    with pytest.raises(ValueError, match="not a candidate"):
        staged.stage_integer_model_admission(
            model + b"\n", target="gemmini", software_spec=spec, capability_contract=contract,
            package_dir=package, operation_id=candidate["operation_id"], rtl_facts=rtl_facts,
        )


def test_resident_pointer_binding_uses_interface_dataflow_and_validates_contract_order():
    import yaml

    commands = [
        {"opcode": "RES_PACK", "operands": {"src": "weights", "dst": "resident"},
         "attributes": {"layout": "packed_rhs"}},
        {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "activation", "rhs": "resident", "dst": "sum"}},
        {"opcode": "COMMIT", "operands": {"src": "sum", "dst": "result"},
         "attributes": {"output_dtype": "i32", "epilogue": []}},
    ]
    tensors = {
        "weights": {"shape": [19, 8], "dtype": "i8", "role": "weight"},
        "activation": {"shape": [4, 19], "dtype": "i8", "role": "input"},
    }
    cb = {"abi_version": "0.1", "target": "atlas", "tensors": tensors, "commands": commands}
    interface = emit_interface_mlir(cb)
    bound = bind_single_resident_matmul(interface, target="atlas")
    assert bound.pointer_order == ("weights", "activation", "result")
    assert (bound.m, bound.n, bound.k) == (4, 8, 19)
    assert bound.kernel_symbol == "atlas_kernel"
    wrong_order = emit_interface_mlir({**cb, "tensors": dict(reversed(list(tensors.items())))})
    with pytest.raises(ValueError, match="pointer ABI"):
        bind_single_resident_matmul(wrong_order, target="atlas")
    wrong_result_type = interface.replace("tensor<4x8xi32>", "tensor<4x8xi8>")
    with pytest.raises(ValueError, match="fully typed round trip"):
        bind_single_resident_matmul(wrong_result_type, target="atlas")
    contract = yaml.safe_load((repo_root() / "merlin/contract/legacy/kernel_abi_v1.yaml").read_bytes())
    rows = contract["kernel_abi"]["arg_order_by_command_shape"]
    resident = next(row for row in rows if row["shape"] == "resident_matmul")
    resident["order"] = list(reversed(resident["order"]))
    with pytest.raises(ValueError, match="unsupported by the rank-2 shim"):
        bind_single_resident_matmul(interface, target="atlas", abi_contract=yaml.safe_dump(contract).encode())


def test_integerized_source_matrix_body_can_supply_a_bounded_window(tmp_path):
    capture = _integer_capture(tmp_path)
    projected = derive_kernel_window(
        capture, "g:prepared:root:n7", tile_dim=16, projection_types=("i8", "i8", "i32")
    )
    assert projected["source"]["mlir_operation"] == "linalg.generic"
    assert projected["source"]["geometry"] == {"M": 4, "K": 19, "N": 8}
    assert projected["projection"]["geometry"] == {"M": 4, "K": 19, "N": 8}
    assert projected["projected_type_body_observed"] is True


def test_exact_integer_model_body_outlines_a_compilable_interface_kernel(tmp_path):
    from merlin.targetgen.tool_cli import main

    model = _INTEGER_BODY.encode()
    (tmp_path / "model.mlir").write_bytes(model)
    source_root = repo_root() / "examples/gemmini/target"
    software_spec = source_root / "software-spec.yaml"
    capability_contract = source_root / "contracts/target_contract.yaml"
    output = tmp_path / "kernels"
    assert main(
        [
            "outline-int-mm", "--target", "gemmini", "--mlir", str(tmp_path / "model.mlir"),
            "--software-spec", str(software_spec), "--capability-contract", str(capability_contract),
            "--out", str(output),
        ]
    ) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["model_sha256"] == hashlib.sha256(model).hexdigest()
    assert manifest["software_spec_sha256"] == hashlib.sha256(software_spec.read_bytes()).hexdigest()
    assert manifest["capability_contract_sha256"] == hashlib.sha256(capability_contract.read_bytes()).hexdigest()
    assert manifest["interface_class"]["status"] == "declared"
    assert len(manifest["candidates"]) == 1 and manifest["refused"] == []
    candidate = manifest["candidates"][0]
    assert candidate["software_admission"]["status"] == "unknown"
    assert candidate["compiler_support"] == "not_evaluated"
    assert candidate["operation_id"].startswith(f"mlir:{manifest['model_sha256']}:")
    assert candidate["operand_bindings"] == [
        {"source": "function_argument", "argument_index": 0},
        {"source": "function_argument", "argument_index": 1},
    ]
    stitching = manifest["stitching"]
    assert stitching["model_sha256"] == manifest["model_sha256"]
    assert stitching["candidate_operation_ids"] == [candidate["operation_id"]]
    assert stitching["status"] == "diagnostic_unexecutable"
    assert [edge["direction"] for edge in stitching["candidate_boundary_crossings"]] == [
        "host_to_candidate", "host_to_candidate", "candidate_to_host"
    ]
    assert [edge["dtype"] for edge in stitching["candidate_boundary_crossings"]] == [
        "i8", "i8", "i32"
    ]
    assert stitching["functions"][0]["return_values"][0]["value_id"] == (
        candidate["operation_id"] + ":result:0"
    )
    assert stitching["functions"][0]["unlowered_host_operations"]
    plan = stitching["composition_plan"]
    assert plan["model_sha256"] == manifest["model_sha256"]
    assert plan["status"] == "unlowered" and plan["executable"] is False
    assert plan["functions"][0]["order"] == "single_block_lexical"
    steps = plan["functions"][0]["steps"]
    assert [step["kind"] for step in steps] == [
        "host", "host", "transfer", "transfer", "kernel", "transfer", "return"
    ]
    kernel = next(step for step in steps if step["kind"] == "kernel")
    assert kernel["interface_sha256"] == candidate["interface_sha256"]
    assert kernel["compiler_support"] == "not_evaluated"
    assert kernel["software_admission"]["status"] == "unknown"
    assert kernel["operand_value_ids"] == [edge["source_value_id"] for edge in stitching["candidate_boundary_crossings"][:2]]
    assert [step["edge_id"] for step in steps if step["kind"] == "transfer"] == [
        edge["edge_id"] for edge in stitching["candidate_boundary_crossings"]
    ]
    assert {obligation["kind"] for obligation in plan["obligations"]} >= {
        "host_operation_lowering", "memory_placement_and_typed_transfers",
        "kernel_dispatch_and_pointer_abi", "kernel_compilation_and_admission",
        "whole_model_numerical_equivalence",
    }
    interface = (output / candidate["interface_file"]).read_text()
    assert hashlib.sha256(interface.encode()).hexdigest() == candidate["interface_sha256"]
    parsed = parse_interface_mlir(interface)
    assert list(parsed["tensors"]) == ["B", "A"]  # weight, lhs; output is the produced commit
    assert parsed["tensors"]["B"]["role"] == "input"
    commands = parsed["commands"]
    assert [command["opcode"] for command in commands] == ["RES_PACK", "MATMUL_RESIDENT", "COMMIT"]
    assert commands[-1]["attributes"]["output_dtype"] == "i32"

    # Numerically check this isolated interface's signed, untiled K-tail
    # semantics against an independent scalar loop. This is not device code.
    a = [[((i * 7 + k * 3) % 17) - 8 for k in range(19)] for i in range(4)]
    b = [[((k * 5 + j * 11) % 23) - 11 for j in range(8)] for k in range(19)]
    expected = [[sum(a[i][k] * b[k][j] for k in range(19)) for j in range(8)] for i in range(4)]
    assert simulate(parsed, {"A": a, "B": b})["outputs"] == {"Y": expected}

    # Neither same-typed wrong wiring nor a nonzero seed is a standalone matmul.
    wrong_product = _INTEGER_BODY.replace('"arith.muli"(%lhs32, %rhs32)', '"arith.muli"(%lhs32, %lhs32)')
    selection = {"target": "gemmini", "software_spec": software_spec.read_bytes(),
                 "capability_contract": capability_contract.read_bytes()}
    assert outline_integer_matmuls(wrong_product.encode(), **selection)["candidates"] == []
    nonzero_init = _INTEGER_BODY.replace('value = 0 : i32', 'value = 1 : i32')
    assert outline_integer_matmuls(nonzero_init.encode(), **selection)["candidates"] == []


def test_exact_model_selection_reaches_device_rewrite_without_shape_redecision(tmp_path, monkeypatch):
    """JSON edits cannot admit a candidate; a re-derived and certified byte/ID can route."""
    import yaml

    from merlin.llvmlower import exact_offload
    from merlin.llvmlower.device_offload import load_sidecar, rewrite_prepared_file
    from merlin.llvmlower.exact_offload import ExactOffloadSelection, ReleaseBinding
    from merlin.targetgen import oot_runner

    root = repo_root() / "examples/gemmini/target"
    model = _INTEGER_BODY.encode()
    original_spec = (root / "software-spec.yaml").read_bytes()
    contract = (root / "contracts/target_contract.yaml").read_bytes()
    outline = outline_integer_matmuls(
        model, target="gemmini", software_spec=original_spec, capability_contract=contract,
    )
    operation_id = outline["candidates"][0]["operation_id"]
    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.yaml").write_text("target: gemmini\n")
    with pytest.raises(ValueError, match="reviewed SW admission"):
        ExactOffloadSelection.from_outline(
            outline, model=model, target="gemmini", software_spec=original_spec,
            capability_contract=contract, package_dir=package, operation_ids=(operation_id,),
        )

    doctored = json.loads(json.dumps(outline))
    doctored["candidates"][0]["software_admission"]["status"] = "admitted"
    with pytest.raises(ValueError, match="deterministic re-derivation"):
        ExactOffloadSelection.from_outline(
            doctored, model=model, target="gemmini", software_spec=original_spec,
            capability_contract=contract, package_dir=package, operation_ids=(operation_id,),
        )

    # Synthetic reviewed declaration and oracle stand-in test plumbing only;
    # they are not evidence that today's Gemmini inputs were certified.
    spec_doc = yaml.safe_load(original_spec)
    spec_doc["status"] = "reviewed"
    for absent_observation in ("layouts", "tails", "aliasing"):
        spec_doc["operations"]["contraction"].pop(absent_observation)
    spec = yaml.safe_dump(spec_doc).encode()
    reviewed = outline_integer_matmuls(model, target="gemmini", software_spec=spec, capability_contract=contract)
    selection = ExactOffloadSelection.from_outline(
        reviewed, model=model, target="gemmini", software_spec=spec,
        capability_contract=contract, package_dir=package, operation_ids=(operation_id,),
    )
    materialized = json.loads(json.dumps(reviewed))
    interface_text = materialized["candidates"][0].pop("interface_mlir")
    materialized["candidates"][0]["interface_file"] = "kernel-000004.interface.mlir"
    (tmp_path / "kernel-000004.interface.mlir").write_text(interface_text)
    assert ExactOffloadSelection.from_outline(
        materialized, model=model, target="gemmini", software_spec=spec,
        capability_contract=contract, package_dir=package, operation_ids=(operation_id,),
        interface_root=tmp_path,
    ) == selection
    prepared = tmp_path / "prepared.mlir"
    prepared.write_text(_INTEGER_BODY)
    with pytest.raises(ValueError, match="no independent accelerator certification"):
        rewrite_prepared_file(prepared, tmp_path / "before_cert", "gemmini", exact_selection=selection)
    with pytest.raises(ValueError, match="verified reviewed Phase 0 release"):
        rewrite_prepared_file(
            prepared, tmp_path / "tuple_only", "gemmini",
            exact_selection=replace(selection, certification_sha256=("0" * 64,)),
        )
    with pytest.raises(ValueError, match="verified reviewed Phase 0 release"):
        selection.certify(package, runs_root=tmp_path / "unbound_runs", simulator="test_oracle", timeout=3)

    # Test-only installed-owner stand-in: the host release adapter has its own tests.
    class SyntheticReleaseBinding:
        review_digest = "a" * 64

        def verify(self, _selection):
            return None

    with pytest.raises(ValueError, match="verified reviewed Phase 0 release"):
        replace(selection, release_binding=SyntheticReleaseBinding()).check_release()
    binding = ReleaseBinding(tmp_path / "seal.json", tmp_path / "descriptor.yaml", "app", "a" * 64)
    monkeypatch.setattr(exact_offload, "_release_verifier", lambda: lambda _binding, _selection: None)
    with pytest.raises(ValueError, match="did not return the selected review identity"):
        replace(selection, release_binding=binding).check_release()
    monkeypatch.setattr(exact_offload, "_release_verifier", lambda: lambda _binding, _selection: binding.review_digest)
    selection = replace(selection, release_binding=binding)

    def skipped_certify(_package, _interface, **_kwargs):
        return {"status": "pass", "oracle": {"result": "skipped"}}

    monkeypatch.setattr(oot_runner, "certify", skipped_certify, raising=False)
    with pytest.raises(ValueError, match="running accelerator oracle"):
        selection.certify(package, runs_root=tmp_path / "skipped_runs", simulator="test_oracle", timeout=3)

    def fake_certify(_package, interface, **kwargs):
        assert interface.read_text() == reviewed["candidates"][0]["interface_mlir"]
        assert kwargs["require_accelerator_trace"] is True
        return {
            "status": "pass", "oracle": {"result": "pass"},
            "trace_check": {"status": "pass", "drives_accelerator": True},
            "test_only": True,
        }

    monkeypatch.setattr(oot_runner, "certify", fake_certify, raising=False)
    selection = selection.certify(package, runs_root=tmp_path / "cert_runs", simulator="test_oracle", timeout=3)
    monkeypatch.setattr(
        "merlin.system.offload.device_dtype_triples", lambda _target: (("i8", "i8", "i32"),)
    )
    rewrite = rewrite_prepared_file(prepared, tmp_path / "build", "gemmini", exact_selection=selection)
    assert rewrite.moved == 1
    sidecar = load_sidecar(tmp_path / "build")
    assert sidecar["routed"][0]["operation_id"] == operation_id
    assert sidecar["release_review_digest"] == binding.review_digest
    assert sidecar["software_spec_sha256"] == hashlib.sha256(spec).hexdigest()
    assert sidecar["capability_contract_sha256"] == hashlib.sha256(contract).hexdigest()
    assert next(iter(sidecar["expected_interfaces"].values()))["sha256"] == (
        reviewed["candidates"][0]["interface_sha256"]
    )

    # Exercise the actual whole-model preparation seam. Its normalized MLIR,
    # not the earlier raw capture text, is the selected byte identity.
    from merlin.llvmlower.device_build import DeviceRouting
    from merlin.runtime.backends.zephyr_model import prepare_for_lowering

    source = tmp_path / "source.mlir"
    source.write_bytes(model)
    preflight = tmp_path / "preflight"
    preflight.mkdir()
    normalized, _ = prepare_for_lowering(source, preflight, blocking=False)
    normalized_bytes = normalized.read_bytes()
    normalized_outline = outline_integer_matmuls(
        normalized_bytes, target="gemmini", software_spec=spec, capability_contract=contract
    )
    normalized_id = normalized_outline["candidates"][0]["operation_id"]
    normalized_selection = ExactOffloadSelection.from_outline(
        normalized_outline, model=normalized_bytes, target="gemmini", software_spec=spec,
        capability_contract=contract, package_dir=package, operation_ids=(normalized_id,),
    )
    normalized_selection = replace(normalized_selection, release_binding=binding)
    monkeypatch.setattr(
        oot_runner, "certify",
        lambda _package, interface, **_kwargs: {
            "status": "pass", "oracle": {"result": "pass"},
            "trace_check": {"status": "pass", "drives_accelerator": True},
            "interface_sha256": hashlib.sha256(interface.read_bytes()).hexdigest(), "test_only": True,
        },
    )
    normalized_selection = normalized_selection.certify(
        package, runs_root=tmp_path / "normalized_cert", simulator="test_oracle", timeout=3
    )
    routed_work = tmp_path / "routed_work"
    routed_work.mkdir()
    routed_model, _ = prepare_for_lowering(
        source, routed_work, blocking=False,
        device=DeviceRouting("gemmini", package, "int8", "i32", exact_selection=normalized_selection),
    )
    assert "func.call" in routed_model.read_text()
    assert load_sidecar(routed_work)["routed"][0]["operation_id"] == normalized_id

    changed = tmp_path / "changed.mlir"
    changed.write_text(_INTEGER_BODY + "\n")
    with pytest.raises(ValueError, match="prepared model bytes changed"):
        rewrite_prepared_file(changed, tmp_path / "other", "gemmini", exact_selection=selection)
    (package / "manifest.yaml").write_text("target: another\n")
    with pytest.raises(ValueError, match="package tree changed"):
        selection.check_package(package)


def test_model_stitching_keeps_host_producer_and_consumer_edges_explicit():
    root = repo_root() / "examples/gemmini/target"
    model = _INTEGER_BODY.replace(
        '    %result = "linalg.generic"(%a, %b, %init)',
        '    %prepared = "tensor.cast"(%a) : (tensor<4x19xi8>) -> tensor<4x19xi8>\n'
        '    %result = "linalg.generic"(%prepared, %b, %init)',
    )
    result = outline_integer_matmuls(
        model.encode(), target="gemmini",
        software_spec=(root / "software-spec.yaml").read_bytes(),
        capability_contract=(root / "contracts/target_contract.yaml").read_bytes(),
    )
    assert len(result["candidates"]) == 1
    stitching = result["stitching"]
    producer = stitching["candidate_boundary_crossings"][0]
    assert producer["direction"] == "host_to_candidate"
    assert producer["source_value_id"] == result["candidates"][0]["operand_bindings"][0]["source_value_id"]
    assert producer["producer_operation_id"] in stitching["functions"][0]["unlowered_host_operations"]
    assert stitching["candidate_boundary_crossings"][-1]["consumer_operation"] == "func.return"
    plan = stitching["composition_plan"]
    steps = plan["functions"][0]["steps"]
    host = next(step for step in steps if step.get("operation_id") == producer["producer_operation_id"])
    incoming = next(step for step in steps if step.get("edge_id") == producer["edge_id"])
    kernel = next(step for step in steps if step["kind"] == "kernel")
    assert steps.index(host) < steps.index(incoming) < steps.index(kernel)


def test_composition_refuses_to_linearize_multiple_blocks():
    from merlin.common import mlir_query as query

    model = """builtin.module {
      func.func @forward(%a: tensor<2xi8>) -> tensor<2xi8> {
        cf.br ^next(%a : tensor<2xi8>)
      ^next(%b: tensor<2xi8>):
        func.return %b : tensor<2xi8>
      }
    }"""
    plan = stitching_inventory(query.parse(model), "0" * 64, [])["composition_plan"]
    assert plan["executable"] is False
    assert plan["functions"][0]["order"] == "unestablished"
    assert plan["functions"][0]["steps"] == []
    assert plan["functions"][0]["unplanned_operation_ids"]
    assert any(obligation["kind"] == "control_flow_lowering" for obligation in plan["obligations"])


def test_outline_requires_selected_same_target_resident_class_and_sw_declaration():
    root = repo_root() / "examples"
    gemmini_spec = (root / "gemmini/target/software-spec.yaml").read_bytes()
    gemmini_contract = (root / "gemmini/target/contracts/target_contract.yaml").read_bytes()
    atlas_spec = (root / "atlas/target/software-spec.yaml").read_bytes()
    atlas_contract = (root / "atlas/target/contracts/target_contract.yaml").read_bytes()
    model = _INTEGER_BODY.encode()

    with pytest.raises(ValueError, match="exact bytes"):
        outline_integer_matmuls(model, target="gemmini", software_spec=gemmini_spec, capability_contract=None)
    with pytest.raises(ValueError, match="target"):
        outline_integer_matmuls(model, target="another_target", software_spec=gemmini_spec,
                                capability_contract=gemmini_contract)
    with pytest.raises(ValueError, match="features list"):
        outline_integer_matmuls(model, target="gemmini", software_spec=gemmini_spec,
                                capability_contract=b"name: gemmini\nfamily: tensor_resident\n")
    unsupported = outline_integer_matmuls(model, target="atlas", software_spec=atlas_spec,
                                          capability_contract=atlas_contract)
    assert unsupported["candidates"] == []
    assert unsupported["stitching"]["composition_plan"]["executable"] is False
    assert not any(step["kind"] == "kernel" for step in unsupported["stitching"]["composition_plan"]["functions"][0]["steps"])
    assert unsupported["interface_class"]["status"] == "unsupported"
    assert len(unsupported["refused"]) == 1
    assert "selected contract lacks" in unsupported["refused"][0]["reason"]
    no_command_buffer = gemmini_contract.replace(
        b"accumulator_commit, command_buffer, metrics", b"accumulator_commit, metrics", 1
    )
    no_interface = outline_integer_matmuls(model, target="gemmini", software_spec=gemmini_spec,
                                           capability_contract=no_command_buffer)
    assert no_interface["candidates"] == []
    assert "command_buffer" in no_interface["refused"][0]["reason"]
    host_only = gemmini_spec.replace(b"placement: accelerator", b"placement: host", 1)
    refused = outline_integer_matmuls(model, target="gemmini", software_spec=host_only,
                                      capability_contract=gemmini_contract)
    assert refused["candidates"] == []
    assert "placement" in refused["refused"][0]["reason"]

    renamed_spec = gemmini_spec.replace(b"target: gemmini", b"target: another_target", 1)
    renamed_contract = gemmini_contract.replace(b"name: gemmini", b"name: another_target", 1)
    renamed = outline_integer_matmuls(model, target="another_target", software_spec=renamed_spec,
                                      capability_contract=renamed_contract)
    assert len(renamed["candidates"]) == 1
    assert renamed["candidates"][0]["software_admission"]["status"] == "unknown"


@pytest.mark.parametrize("changed", [
    _INTEGER_BODY.replace('prov.op = "int_matmul"', 'prov.op = "elementwise"'),
    _INTEGER_BODY.replace('"arith.addi"(%acc, %product)', '"arith.subi"(%acc, %product)'),
])
def test_generic_without_exact_integer_matmul_structure_is_not_a_source_window(tmp_path, changed):
    capture = _integer_capture(tmp_path, body=changed)
    with pytest.raises(ValueError, match="integer matmul|matrix body"):
        derive_kernel_window(capture, "g:prepared:root:n7", tile_dim=16, projection_types=("i8", "i8", "i32"))


def _capture(tmp_path, *, parent_k=147):
    model = b"module { one selected operation }\n"
    (tmp_path / "model.mlir").write_bytes(model)
    trace = {
        "status": "diagnostic",
        "graphs": {"prepared": {"nodes": [{"id": "g:prepared:root:n7", "target": "aten.convolution.default"}]}},
        "mlir": {
            "sha256": hashlib.sha256(model).hexdigest(),
            "operations": [
                {
                    "ordinal": 29,
                    "operation": "linalg.matmul",
                    "source_node_ids": ["g:prepared:root:n7"],
                    "origin_node_ids": ["g:original:root:n6"],
                    "operand_types": [
                        f"tensor<64x{parent_k}xf32>",
                        f"tensor<{parent_k}x12544xf32>",
                        "tensor<64x12544xf32>",
                    ],
                    "result_types": ["tensor<64x12544xf32>"],
                }
            ],
        },
    }
    (tmp_path / "frontend-trace.json").write_text(json.dumps(trace), encoding="utf-8")
    return tmp_path


def test_source_geometry_and_k_tail_survive_bounded_integer_projection(tmp_path):
    capture = _capture(tmp_path)
    projected = derive_kernel_window(
        capture, "g:prepared:root:n7", tile_dim=16, projection_types=("i8", "i8", "i32")
    )
    assert projected["source"]["geometry"] == {"M": 64, "K": 147, "N": 12544}
    assert projected["source"]["operand_types"][0] == "tensor<64x147xf32>"
    assert projected["projection"]["geometry"] == {"M": 16, "K": 19, "N": 16}
    assert projected["projected_type_body_observed"] is False
    assert projected["model_equivalence_claim"] == "none_synthetic_operands"


def test_changed_model_bytes_refuse_stale_trace(tmp_path):
    capture = _capture(tmp_path)
    (capture / "model.mlir").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind"):
        derive_kernel_window(capture, "g:prepared:root:n7", tile_dim=16, projection_types=("i8", "i8", "i32"))


def test_projection_precision_is_selected_not_built_into_shared_geometry(tmp_path):
    capture = _capture(tmp_path)
    projected = derive_kernel_window(
        capture, "g:prepared:root:n7", tile_dim=16, projection_types=("f32", "f32", "f32")
    )
    assert projected["projection"]["dtype"] == {"lhs": "f32", "rhs": "f32", "result": "f32"}
    assert projected["projected_type_body_observed"] is True
    assert projected["model_equivalence_claim"] == "none_synthetic_operands"
