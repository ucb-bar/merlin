"""Saved source programs are reusable offline, byte-bound, and not target certificates."""

import hashlib
import json

import numpy as np
import pytest
import yaml
from merlin_experiments.corpus.coverage import selected_cohort_coverage
from merlin_experiments.phase0 import coverage_commitment as CC
from merlin_experiments.phase0 import writer
from merlin_experiments.phase0.evidence import _materialize_evidence
from merlin_experiments.phase0.profiles import synthesis_input_identity
from merlin_experiments.phase0.provenance import _scrub_capsule_dir
from merlin_experiments.phase0.requirements import _materialized_iteration_capsules, _validate_capture_recipes
from merlin_experiments.phase0.writer import _integer_reference_bound, _source_integer_reference_bound, _write_capsule

from merlin.targetgen import capsule_source as source
from merlin.targetgen.capsule_common import load_capsule
from merlin.targetgen.corpus_spec import CorpusBinding, build
from merlin.targetgen.golden_store import load_golden


def _bundle(root):
    root.mkdir()
    weights = root / "weights.safetensors"
    weights.write_bytes(b"diagnostic-fixture-weights")
    (root / "weights.safetensors.manifest.json").write_text(
        json.dumps(
            {
                "0": {"kind": "param", "name": "weight"},
                "1": {"kind": "input", "name": "x"},
            }
        )
    )
    program = (
        f'builtin.module attributes {{prov.weights_file = "{weights}"}} {{ '
        "func.func @forward(%weight: tensor<2xf32>, %x: tensor<2xf32>) -> tensor<2xf32> { "
        "func.return %x : tensor<2xf32> } }"
    )
    (root / "model.mlir").write_text(program)
    raw = program.encode()
    trace = {"mlir": {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}}
    (root / "frontend-trace.json").write_text(json.dumps(trace))
    meta = {
        "ok": True,
        "opaque": 0,
        "dtype": "fp32",
        "torch_seed": 0,
        "weights": str(weights),
        "input_abi": [{"shape": [2], "dtype": "f32"}],
        "output_abi": [{"shape": [2], "dtype": "f32"}],
        "frontend_trace": {
            "path": "frontend-trace.json",
            "sha256": hashlib.sha256((root / "frontend-trace.json").read_bytes()).hexdigest(),
        },
    }
    (root / "meta.json").write_text(json.dumps(meta))
    np.savez(root / "inputs.npz", in0=np.array([1, 2], dtype=np.float32))
    np.save(root / "golden.npy", np.array([1, 2], dtype=np.float32))
    (root / "input_order.json").write_text('{"x": 0}')
    receipt = {
        "schema": "m2m.capture-receipt.v1",
        "materialized_abi": {"complete": True},
        "source_closure_verified": False,
        "source": {"path": "/unavailable/loader.py"},
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in root.iterdir()
        },
    }
    (root / "capture_receipt.json").write_text(json.dumps(receipt))
    return {
        "capture_source_path": str(root / "model.mlir"),
        "capture_sha256": hashlib.sha256(raw).hexdigest(),
        "capture_receipt": {"receipt_sha256": hashlib.sha256((root / "capture_receipt.json").read_bytes()).hexdigest()},
        "n_operations": 3,
    }


def test_source_capsule_reuse_is_offline_exact_and_fail_closed(tmp_path, monkeypatch):
    application = _bundle(tmp_path / "original")
    entries, outputs = _materialized_iteration_capsules({"applications": {"iteration": application}}, "0" * 64)
    saved = tmp_path / "installed"
    _materialize_evidence(saved, outputs)
    selected = entries[0]
    selected["materialized_capture"]["path"] = str(saved / selected["materialized_capture"]["path"])
    # Deliberately remove the original source; only the immutable selected members may be read.
    for path in (tmp_path / "original").iterdir():
        path.unlink()
    monkeypatch.setattr(source, "PytorchRefSource", lambda *a, **kw: pytest.fail("unexpected recapture"))
    binding = CorpusBinding("fixture", 2, "f32", "f32", False, ["L0"], "tolerance_float", atol=1e-5, rtol=1e-5)
    result = _write_capsule(selected, binding, tmp_path / "corpus")
    capsule = yaml.safe_load((result / "capsule.yaml").read_text())
    assert load_capsule(result)["source_role"] == "model_derived"
    assert capsule["materialized_capture"]["source_closure_verified"] is False
    assert capsule["materialized_capture"]["coverage_scope"] == "full_capture"
    assert not (result / "capsule.pytorch.py").exists()
    assert capsule["expected"]["instruction_classes"] == []
    receipt = json.loads((result / "frontend-evidence.json").read_text())
    assert receipt["packaging"] == "weights_reference_relocation"
    assert receipt["raw_source_mlir_sha256"] == application["capture_sha256"]
    assert receipt["raw_source_trace_bound"] is True
    assert receipt["schema"] == "merlin.capsule_frontend_evidence.v2"
    relocation = receipt["weights_reference_relocation"]
    assert "from" not in relocation
    assert (
        relocation["from_reference_sha256"]
        == hashlib.sha256(str(tmp_path / "original/weights.safetensors").encode("utf-8")).hexdigest()
    )
    assert relocation["to"] == "capsule.weights.safetensors"
    assert str(tmp_path / "original/weights.safetensors") not in (result / "capsule.yaml").read_text()
    assert str(tmp_path / "original/weights.safetensors") not in (result / "frontend-evidence.json").read_text()
    assert 'prov.weights_file = "capsule.weights.safetensors"' in (result / "capsule.interface.mlir").read_text()
    _scrub_capsule_dir(result)
    receipt = json.loads((result / "frontend-evidence.json").read_text())
    assert receipt["raw_source_mlir_sha256"] == application["capture_sha256"]
    assert receipt["source_mlir_sha256"] == hashlib.sha256((result / receipt["source_mlir"]).read_bytes()).hexdigest()
    assert (
        receipt["packaged_mlir_sha256"] == hashlib.sha256((result / receipt["packaged_mlir"]).read_bytes()).hexdigest()
    )
    assert (
        receipt["interface_mlir_sha256"]
        == hashlib.sha256((result / receipt["interface_mlir"]).read_bytes()).hexdigest()
    )
    assert receipt["source_portability"]["kind"] == "weights_reference_relocation"
    assert receipt["source_portability"]["edit_count"] == 1
    assert 'prov.weights_file = "capsule.weights.safetensors"' in (result / "frontend-source.mlir").read_text()
    for name in (
        "frontend-source.mlir",
        "capsule.interface.mlir",
        "frontend-trace.json",
        "capsule.weights.safetensors",
        "capsule.weights.safetensors.manifest.json",
        "source-capture-receipt.json",
    ):
        path = result / name
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        with pytest.raises(ValueError, match="frontend evidence member changed"):
            _scrub_capsule_dir(result)
        path.write_bytes(original)
    evidence_path = result / "frontend-evidence.json"
    original_evidence = evidence_path.read_bytes()
    modified_evidence = json.loads(original_evidence)
    modified_evidence["source_mlir_sha256"] = "0" * 64
    evidence_path.write_text(json.dumps(modified_evidence))
    with pytest.raises(ValueError, match="differs from capsule declaration"):
        _scrub_capsule_dir(result)
    evidence_path.write_bytes(original_evidence)
    _scrub_capsule_dir(result)  # idempotent after restoring the exact evidence-bearing bytes
    assert load_golden(result)["outputs"] == {"Y0": [1.0, 2.0]}
    with pytest.raises(source.M2MUnavailable, match="held-out validation"):
        source.materialized_model_artifacts({**selected["materialized_capture"], "workload_role": "validation"})
    source_receipt = saved / "materialized/iteration/capture_receipt.json"
    original_receipt = source_receipt.read_bytes()
    renamed_receipt = json.loads(original_receipt)
    renamed_receipt["source"]["path"] = "/missing/workloads/tiny_llama/loader.py"
    source_receipt.write_text(json.dumps(renamed_receipt))
    with pytest.raises(source.M2MUnavailable, match="renamed"):
        source.materialized_model_artifacts(
            {
                **selected["materialized_capture"],
                "receipt_sha256": hashlib.sha256(source_receipt.read_bytes()).hexdigest(),
            }
        )
    source_receipt.write_bytes(original_receipt)
    selected_model = saved / "materialized/iteration/model.mlir"
    selected_model.write_text(selected_model.read_text() + "\n")
    with pytest.raises(source.M2MUnavailable, match="receipt"):
        source.materialized_model_artifacts(selected["materialized_capture"])


def test_preselected_loader_is_copied_into_a_gradeable_source_capsule(tmp_path):
    application = _bundle(tmp_path / "original")
    snapshot = tmp_path / "snapshot" / "loader.py"
    snapshot.parent.mkdir()
    snapshot.write_text("def load_model():\n    return None\n")
    loader_sha256 = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    entries, outputs = _materialized_iteration_capsules(
        {"applications": {"iteration": application}},
        "0" * 64,
        loader_snapshots={"iteration": (snapshot, loader_sha256)},
    )
    assert outputs["materialized/iteration/loader.py"] == snapshot.read_bytes()
    saved = tmp_path / "installed"
    _materialize_evidence(saved, outputs)
    selected = entries[0]
    selected["materialized_capture"]["path"] = str(saved / selected["materialized_capture"]["path"])
    selected["materialized_capture"]["loader_path"] = str(saved / selected["materialized_capture"]["loader_path"])
    binding = CorpusBinding("fixture", 2, "f32", "f32", False, ["L0"], "tolerance_float", atol=1e-5, rtol=1e-5)
    result = _write_capsule(selected, binding, tmp_path / "corpus")
    capsule = yaml.safe_load((result / "capsule.yaml").read_text())
    assert capsule["pytorch_ref"]["loader"] == "capsule.pytorch.py"
    assert (result / "capsule.pytorch.py").read_bytes() == snapshot.read_bytes()
    materialized_loader = saved / "materialized/iteration/loader.py"
    materialized_loader.write_text("def load_model():\n    raise RuntimeError('changed')\n")
    with pytest.raises(source.M2MUnavailable, match="loader differs"):
        source.materialized_model_artifacts(selected["materialized_capture"])


def test_derived_micro_model_reads_only_explicit_frozen_captures(tmp_path, monkeypatch):
    from merlin.targetgen import micro_model

    saved = tmp_path / "frozen" / "model.mlir"
    saved.parent.mkdir()
    saved.write_text("module {}\n")
    monkeypatch.setattr(writer, "_roster_captures", lambda: pytest.fail("ambient recaptures were read"))

    def spec(target, captures, *, software_spec, capture_dtype):
        assert target == "fixture"
        assert captures == {"iteration": saved}
        assert software_spec == {"frozen": "selected"}
        assert capture_dtype == "fp32"
        return micro_model.MicroModelSpec(target="fixture")

    monkeypatch.setattr(micro_model, "spec", spec)
    monkeypatch.setattr(micro_model, "emit_pytorch", lambda spec: "# frozen inventory\n")
    entry = {
        "cat": "model",
        "name": "SY_micro_model",
        "_frozen_application_captures": {"iteration": saved},
        "_frozen_software_spec": {"frozen": "selected"},
    }
    assert writer._emit_micro_model_loader(entry, "fixture", tmp_path / "out", capture_dtype="fp32")
    assert entry["loader"].endswith("capsule.pytorch.py")
    assert "_frozen_application_captures" not in entry
    assert "_frozen_software_spec" not in entry


def test_quantized_capture_recipe_must_match_selected_provider(tmp_path):
    bundle = tmp_path / "capture"
    _bundle(bundle)
    meta = json.loads((bundle / "meta.json").read_text())
    meta["quantization_stats"] = {"recipe_sha256": "a" * 64}
    (bundle / "meta.json").write_text(json.dumps(meta))
    captures = {"iteration": bundle / "model.mlir"}
    with pytest.raises(ValueError, match="iteration.*different quantization recipe"):
        _validate_capture_recipes(captures, {"b" * 64})
    _validate_capture_recipes(captures, {"a" * 64})
    meta["quantization_stats"]["recipe_sha256"] = []
    (bundle / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="iteration.*different quantization recipe"):
        _validate_capture_recipes(captures, {"a" * 64})


def test_external_capture_contract_must_match_selected_spec(tmp_path):
    bundle = tmp_path / "capture"
    _bundle(bundle)
    contract_sha = "a" * 64
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text("selected: host\n")
    policy_sha = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    manifest = {
        "schema": "m2m.quantization_manifest.v1",
        "contract_sha256": contract_sha,
        "policy_sha256": policy_sha,
        "sites": [{"site_id": "one", "status": "host"}],
    }
    manifest_sha = hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest_path = bundle / "quantization-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    mlir_path = bundle / "model.mlir"
    mlir_path.write_text(
        mlir_path.read_text().replace(
            "prov.weights_file =",
            f'prov.quantization_manifest_sha256 = "{manifest_sha}", prov.weights_file =',
        )
    )
    meta_path = bundle / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["quantization_manifest"] = {
        "path": manifest_path.name,
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "manifest_sha256": manifest_sha,
    }
    meta_path.write_text(json.dumps(meta))
    receipt_path = bundle / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    for path in (mlir_path, meta_path, manifest_path):
        receipt["artifacts"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    receipt_path.write_text(json.dumps(receipt))
    captures = {"iteration": mlir_path}
    selections = {"iteration": (policy_path, policy_sha)}
    assert _validate_capture_recipes(
        captures, set(), software_spec_sha256=contract_sha, policy_selections=selections
    ) == {"iteration": policy_sha}
    with pytest.raises(ValueError, match="external quantization contract differs"):
        _validate_capture_recipes(captures, set(), software_spec_sha256="b" * 64, policy_selections=selections)
    with pytest.raises(ValueError, match="independent policy selection"):
        _validate_capture_recipes(captures, set(), software_spec_sha256=contract_sha)
    with pytest.raises(ValueError, match="selected quantization policy differs"):
        _validate_capture_recipes(
            captures, set(), software_spec_sha256=contract_sha, policy_selections={"iteration": (policy_path, "0" * 64)}
        )


def test_selected_policy_sidecar_is_rechecked_for_synthesis(tmp_path):
    member = "selected-quantization-policies/iteration.input"
    sidecar = tmp_path / member
    sidecar.parent.mkdir()
    sidecar.write_bytes(b"selected policy\n")
    digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    requirement = tmp_path / "requirements.yaml"
    requirement.write_text(
        yaml.safe_dump(
            {
                "quantization_policy_selections": {
                    "schema": "merlin.phase0.quantization_policy_selections.v1",
                    "status": "byte_selected_not_numerically_reviewed",
                    "applications": {"iteration": {"artifact": member, "sha256": digest}},
                }
            }
        )
    )
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("capsules: []\n")
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("workload_spec: {applications: [iteration]}\n")
    selected = dict(conformance_spec=requirement, recipe=recipe, descriptor=descriptor)
    assert synthesis_input_identity(**selected)["conformance_spec_sha256"]
    sidecar.write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="selected quantization policy sidecar differs"):
        synthesis_input_identity(**selected)


def test_integer_golden_bound_uses_concrete_reduction_and_internal_width():
    semantics = {
        "internal_arithmetic": {
            "full_operation_overflow_policy": "bounded_exact_requires_each_partial_sum",
            "mac_result_bits": 20,
            "signed_operand_bits": 8,
        }
    }
    capsule = {
        "operation": {"op": "matmul", "attributes": {"lhs": "A", "weight": "W"}},
        "inputs": [
            {"name": "A", "role": "input", "shape": [2, 4], "dtype": "i8"},
            {"name": "W", "role": "weight", "shape": [4, 2], "dtype": "i8"},
        ],
    }
    proof = _integer_reference_bound({"numerical_semantics": semantics}, capsule)
    assert proof["status"] == "proven_safe" and proof["reduction_extent"] == 4
    capsule["stimulus_range"] = [127, 127]
    capsule["inputs"][0]["shape"] = [2, 64]
    capsule["inputs"][1]["shape"] = [64, 2]
    with pytest.raises(ValueError, match="may_overflow"):
        _integer_reference_bound({"numerical_semantics": semantics}, capsule)


def test_fused_integer_matmul_bias_cannot_bypass_internal_width_bound():
    semantics = {
        "internal_arithmetic": {
            "full_operation_overflow_policy": "bounded_exact_requires_each_partial_sum",
            "mac_result_bits": 20,
            "signed_operand_bits": 8,
        }
    }
    binding = CorpusBinding("fixture", 16, "int8", "i32", True, ["L0"], "exact_int")
    entry = {
        "name": "fused",
        "kind": "layer",
        "source_role": "derived_sweep",
        "source_reference": "fixture",
        "op": "fused_matmul_bias",
        "M": 16,
        "K": 64,
        "N": 16,
    }
    capsule, _ = build(entry, binding)
    capsule["stimulus_range"] = [127, 127]
    with pytest.raises(ValueError, match="may_overflow"):
        _integer_reference_bound({"numerical_semantics": semantics}, capsule)
    capsule["stimulus_range"] = [1, 1]
    proof = _integer_reference_bound({"numerical_semantics": semantics}, capsule)
    assert proof["status"] == "proven_safe"
    assert proof["reduction_extent"] == 64
    assert proof["maximum_absolute_initial_addend"] == 1


def test_resident_reuse_bounds_each_integer_matmul(monkeypatch):
    from merlin.runtime.tensor import Tensor

    semantics = {
        "internal_arithmetic": {
            "full_operation_overflow_policy": "bounded_exact_requires_each_partial_sum",
            "mac_result_bits": 20,
            "signed_operand_bits": 8,
        }
    }
    binding = CorpusBinding("fixture", 16, "int8", "i32", True, ["L0"], "exact_int")
    entry = {
        "name": "reuse",
        "kind": "layer",
        "source_role": "derived_sweep",
        "source_reference": "fixture",
        "op": "resident_reuse",
        "K": 64,
        "N": 16,
        "matmuls": [{"lhs": "A0", "out": "Y0", "M": 16}, {"lhs": "A1", "out": "Y1", "M": 16}],
    }
    capsule, _ = build(entry, binding)
    leaves = {
        "W": Tensor((64, 16), [127] * (64 * 16), "i8"),
        "A0": Tensor((16, 64), [0] * (16 * 64), "i8"),
        "A1": Tensor((16, 64), [127] * (16 * 64), "i8"),
    }
    monkeypatch.setattr(writer.CG, "materialize_capsule_leaves", lambda _: leaves)
    with pytest.raises(ValueError, match="may_overflow"):
        _integer_reference_bound({"numerical_semantics": semantics}, capsule)
    leaves["A1"] = Tensor((16, 64), [1] * (16 * 64), "i8")
    proof = _integer_reference_bound({"numerical_semantics": semantics}, capsule)
    assert proof["status"] == "proven_safe"
    assert proof["bound"] == 64 * 127
    assert [(member["lhs"], member["partial_sum_bound"]["reduction_extent"]) for member in proof["members"]] == [
        ("A0", 64),
        ("A1", 64),
    ]
    capsule["operation"]["attributes"]["matmuls"][1]["lhs"] = "missing"
    with pytest.raises(ValueError, match="requires concrete lhs"):
        _integer_reference_bound({"numerical_semantics": semantics}, capsule)


@pytest.mark.parametrize("role,transform", [("island", "xor_low_bit"), ("no_island", "none")])
def test_host_island_bounds_both_concrete_integer_contractions(role, transform):
    semantics = {
        "internal_arithmetic": {
            "full_operation_overflow_policy": "bounded_exact_requires_each_partial_sum",
            "mac_result_bits": 20,
            "signed_operand_bits": 8,
        }
    }
    binding = CorpusBinding("fixture", 16, "int8", "i32", True, ["L0"], "exact_int")
    entry = {
        "name": "seam",
        "kind": "model_slice",
        "source_role": "derived_sweep",
        "source_reference": "fixture",
        "op": "host_island_seam",
        "M": 1,
        "K": 1,
        "H": 16,
        "N": 1,
        "comparison_role": role,
        "host_transform": transform,
        "xor_mask": 1,
    }
    capsule, _ = build(entry, binding)
    capsule["stimulus_range"] = [127, 127]
    proof = _integer_reference_bound({"numerical_semantics": semantics}, capsule)
    assert proof["status"] == "proven_safe"
    assert [member["region"] for member in proof["members"]] == ["contraction_0", "contraction_1"]
    assert proof["members"][0]["partial_sum_bound"]["bound"] == 127 * 127
    assert proof["members"][1]["partial_sum_bound"]["bound"] == 16 * (126 if role == "island" else 127) * 127
    with pytest.raises(ValueError, match="selected internal-width bound policy"):
        _integer_reference_bound({}, capsule)

    # Only the derived second input is dangerous; the first contraction remains within i20.
    wider, _ = build({**entry, "H": 64}, binding)
    wider["stimulus_range"] = [127, 127]
    with pytest.raises(ValueError, match="may_overflow"):
        _integer_reference_bound({"numerical_semantics": semantics}, wider)

    capsule["operation"]["attributes"]["shared_accelerator_epilogue"] = "unknown"
    with pytest.raises(ValueError, match="declared two-contraction i8 seam"):
        _integer_reference_bound({"numerical_semantics": semantics}, capsule)
    capsule["operation"]["attributes"]["shared_accelerator_epilogue"] = "saturating_i32_to_i8"
    capsule["inputs"][2]["shape"] = [17, 1]
    with pytest.raises(ValueError, match="operand ABI differs"):
        _integer_reference_bound({"numerical_semantics": semantics}, capsule)


def _exact_source_integer_member(root, *, reduction_extent):
    semantics = {
        "model": {"engine": "integer_reference"},
        "operand_dtype": "int8",
        "accumulator_dtype": "i32",
        "readout_dtype": "i32",
        "internal_arithmetic": {
            "full_operation_overflow_policy": "bounded_exact_requires_each_partial_sum",
            "mac_result_bits": 20,
            "signed_operand_bits": 8,
        },
    }
    binding = CorpusBinding("fixture", 16, "int8", "i32", True, ["L0"], "exact_int")
    entry = {
        "name": "source_mm",
        "cat": "isa",
        "kind": "isa",
        "source_role": "model_derived",
        "source_reference": "selected integer operation",
        "source": "pytorch",
        "capture_op": "int_matmul",
        "op": "matmul",
        "M": 1,
        "K": reduction_extent,
        "N": 1,
        "numerical_semantics": semantics,
    }
    capsule, _ = build(entry, binding)
    capsule["application_signature_match"] = {
        "status": "verified_capture_match",
        "source_quantization": "int8_dyn_act_int8_weight",
    }
    root.mkdir()
    (root / "capsule.yaml").write_text(yaml.safe_dump(capsule))
    values = [127] * reduction_extent
    golden = {
        "golden_source": "host_torch_eager",
        "oracle_provenance": {
            "inputs": {
                "A0": {
                    "shape": [1, reduction_extent],
                    "dtype": "i8",
                    "decoded": values,
                    "integer_bytes_hex": bytes(values).hex(),
                },
                "W": {
                    "shape": [reduction_extent, 1],
                    "dtype": "i8",
                    "decoded": values,
                    "integer_bytes_hex": bytes(values).hex(),
                },
            }
        },
        "outputs": {"Y0": [[reduction_extent * 127 * 127]]},
    }
    (root / "golden.yaml").write_text(yaml.safe_dump(golden))
    return entry, binding, capsule, golden


def test_source_integer_bound_uses_captured_bytes_and_stamps_both_artifacts(tmp_path, monkeypatch):
    directory = tmp_path / "source"
    entry, binding, capsule, golden = _exact_source_integer_member(directory, reduction_extent=2)
    proof = _source_integer_reference_bound(entry, capsule, directory)
    assert proof["status"] == "proven_safe"
    assert proof["reduction_extent"] == 2
    assert proof["maximum_absolute_operands"] == [127, 127]
    assert proof["maximum_absolute_initial_addend"] == 0
    monkeypatch.setattr(writer, "_write_capsule_inner", lambda *_: directory)
    _write_capsule(entry, binding, tmp_path)
    saved_cap = yaml.safe_load((directory / "capsule.yaml").read_text())
    saved_golden = load_golden(directory)
    assert saved_cap["integer_partial_sum_bound"] == saved_golden["integer_partial_sum_bound"] == proof
    assert saved_golden["golden_source"] == "host_torch_eager"
    assert saved_golden["outputs"] == golden["outputs"]


@pytest.mark.parametrize("op,dtype,result", [("permute", "i8", "i8"), ("dtype_cast", "i32", "f32")])
def test_native_storage_writer_preserves_result_policy_without_mac_qualification(
    tmp_path, monkeypatch, op, dtype, result
):
    signature = {
        "provenance_op": op,
        "frontend_op": "selected.frontend",
        "semantic_family": "movement" if op == "permute" else "elementwise_map",
        "disposition": "support_required" if op == "permute" else "host_required",
        "ordered_operand_types": [{"dtype": dtype, "shape": [2, 3]}],
        "ordered_result_types": [{"dtype": result, "shape": [3, 2] if op == "permute" else [2, 3]}],
    }
    entry = {
        "name": "native_storage", "op": op, "source": "pytorch", "operand_dtype": dtype,
        "capture_dtype": "fp32", "frontend_op": "selected.frontend",
        "native_input_dtype": dtype, "native_output_dtype": result,
        "generalization": {"must_accelerate": False, "eligible": False},
        "native_source_abi": {
            "full_inventory_sha256": "a" * 64, "capture_sha256": "b" * 64,
            "signature_sha256": hashlib.sha256(
                json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "signature": signature,
        },
    }
    policy = {"compare": "exact_int" if result == "i8" else "tolerance_float", "dtype": result}
    capsule = {"name": entry["name"], "operation": {"op": op}, "inputs": [], "numeric_policy": policy}
    golden = {"golden_source": "host_torch_eager", "outputs": {"Y": [[1]]}}
    (tmp_path / "capsule.yaml").write_text(yaml.safe_dump(capsule))
    (tmp_path / "golden.yaml").write_text(yaml.safe_dump(golden))
    monkeypatch.setattr(writer, "_write_capsule_inner", lambda *a, **kw: tmp_path)
    binding = CorpusBinding("synthetic", 4, dtype, "i32", True, ["L0"], "exact_int")
    _write_capsule(entry, binding, tmp_path)
    assert yaml.safe_load((tmp_path / "capsule.yaml").read_text())["numeric_policy"] == policy
    assert "integer_partial_sum_bound" not in yaml.safe_load((tmp_path / "capsule.yaml").read_text())
    assert load_golden(tmp_path) == golden
    # A source model cannot bypass its existing qualifier by copying storage declarations.
    from merlin_experiments.phase0 import model_qualification

    def model_refuses(*args):
        raise ValueError("model qualification still required")

    monkeypatch.setattr(model_qualification, "qualify", model_refuses)
    with pytest.raises(ValueError, match="model qualification still required"):
        _write_capsule({**entry, "kind": "model"}, binding, tmp_path)
    with pytest.raises(ValueError, match="complete typed declaration"):
        _write_capsule({key: value for key, value in entry.items() if key != "native_output_dtype"}, binding, tmp_path)


def test_source_integer_bound_refuses_overflow_and_incomplete_capture(tmp_path):
    safe = tmp_path / "safe"
    entry, _, capsule, golden = _exact_source_integer_member(safe, reduction_extent=2)
    with pytest.raises(ValueError, match="selected internal-width bound policy"):
        _source_integer_reference_bound({**entry, "numerical_semantics": {}}, capsule, safe)
    capsule["application_signature_match"] = {}
    with pytest.raises(ValueError, match="verified isolated i8 matmul"):
        _source_integer_reference_bound(entry, capsule, safe)
    capsule["application_signature_match"] = {
        "status": "verified_capture_match",
        "source_quantization": "int8_dyn_act_int8_weight",
    }
    with pytest.raises(ValueError, match="verified isolated i8 matmul"):
        _source_integer_reference_bound({**entry, "kind": "model"}, capsule, safe)
    capsule["kind"] = "model"
    with pytest.raises(ValueError, match="verified isolated i8 matmul"):
        _source_integer_reference_bound(entry, capsule, safe)
    capsule["kind"] = "isa"
    golden["oracle_provenance"]["inputs"]["A0"].pop("integer_bytes_hex")
    (safe / "golden.yaml").write_text(yaml.safe_dump(golden))
    with pytest.raises(ValueError, match="captured bytes"):
        _source_integer_reference_bound(entry, capsule, safe)
    golden["oracle_provenance"]["inputs"]["A0"]["integer_bytes_hex"] = bytes([127, 127]).hex()
    golden["outputs"]["Y0"][0][0] += 1
    (safe / "golden.yaml").write_text(yaml.safe_dump(golden))
    with pytest.raises(ValueError, match="host-eager output differs"):
        _source_integer_reference_bound(entry, capsule, safe)

    overflowing = tmp_path / "overflowing"
    entry, _, capsule, _ = _exact_source_integer_member(overflowing, reduction_extent=64)
    with pytest.raises(ValueError, match="may_overflow"):
        _source_integer_reference_bound(entry, capsule, overflowing)


def _exact_spec_integer_member(root, *, reduction_extent):
    entry, binding, capsule, golden = _exact_source_integer_member(root, reduction_extent=reduction_extent)
    entry.update(source="spec", spec_ref="gemmini:op.matmul")
    capsule["spec_ref"] = entry["spec_ref"]
    capsule.pop("application_signature_match")
    golden["golden_source"] = "specir_program_gemmini"
    golden["oracle_provenance"]["spec_ref"] = entry["spec_ref"]
    golden["oracle_provenance"]["inputs"] = {
        "A0": {"shape": [1, reduction_extent], "decoded": [[127] * reduction_extent]},
        "W": {"shape": [reduction_extent, 1], "decoded": [[127] for _ in range(reduction_extent)]},
    }
    (root / "capsule.yaml").write_text(yaml.safe_dump(capsule))
    (root / "golden.yaml").write_text(yaml.safe_dump(golden))
    return entry, binding, capsule, golden


def test_spec_integer_bound_uses_exact_program_operands_and_refuses_incomplete_provenance(tmp_path, monkeypatch):
    safe = tmp_path / "spec"
    entry, binding, capsule, golden = _exact_spec_integer_member(safe, reduction_extent=2)
    bound = _source_integer_reference_bound(entry, capsule, safe)
    assert bound["status"] == "proven_safe"
    assert bound["maximum_absolute_operands"] == [127, 127]
    assert bound["maximum_absolute_initial_addend"] == 0
    assert [member["operand_stream"] for member in bound["members"]] == [
        "spec_program",
        "capsule_materialized",
    ]
    monkeypatch.setattr(writer, "_write_capsule_inner", lambda *_: safe)
    _write_capsule(entry, binding, tmp_path)
    assert yaml.safe_load((safe / "capsule.yaml").read_text())["integer_partial_sum_bound"] == bound
    assert load_golden(safe)["integer_partial_sum_bound"] == bound

    golden["golden_source"] = "specir_program_other"
    (safe / "golden.yaml").write_text(yaml.safe_dump(golden))
    with pytest.raises(ValueError, match="exact program operand provenance"):
        _source_integer_reference_bound(entry, capsule, safe)
    golden["golden_source"] = "specir_program_gemmini"
    golden["oracle_provenance"]["inputs"]["A0"].pop("decoded")
    (safe / "golden.yaml").write_text(yaml.safe_dump(golden))
    with pytest.raises(ValueError, match="incomplete exact i8 operand values"):
        _source_integer_reference_bound(entry, capsule, safe)

    overflowing = tmp_path / "spec_overflow"
    entry, _, capsule, _ = _exact_spec_integer_member(overflowing, reduction_extent=64)
    with pytest.raises(ValueError, match="may_overflow"):
        _source_integer_reference_bound(entry, capsule, overflowing)

    # The spec program can be safe while the capsule's integer grader uses
    # independently materialized operands that overflow the selected MAC width.
    separate = tmp_path / "spec_separate_stimulus"
    entry, _, capsule, golden = _exact_spec_integer_member(separate, reduction_extent=64)
    golden["oracle_provenance"]["inputs"]["A0"]["decoded"] = [[1] * 64]
    golden["oracle_provenance"]["inputs"]["W"]["decoded"] = [[1] for _ in range(64)]
    golden["outputs"]["Y0"] = [[64]]
    (separate / "golden.yaml").write_text(yaml.safe_dump(golden))
    capsule["stimulus_range"] = [127, 127]
    with pytest.raises(ValueError, match="may_overflow"):
        _source_integer_reference_bound(entry, capsule, separate)


def test_rectangular_attention_score_uses_the_selected_internal_width():
    semantics = {
        "internal_arithmetic": {
            "full_operation_overflow_policy": "bounded_exact_requires_each_partial_sum",
            "mac_result_bits": 20,
            "signed_operand_bits": 8,
        }
    }
    capsule = {
        "operation": {"op": "attention_qk", "attributes": {"q": "Q", "k": "K"}},
        "inputs": [
            {"name": "Q", "role": "input", "shape": [16, 32], "dtype": "i8"},
            {"name": "K", "role": "input", "shape": [1040, 32], "dtype": "i8"},
        ],
    }
    proof = _integer_reference_bound({"numerical_semantics": semantics}, capsule)
    assert proof["status"] == "proven_safe" and proof["reduction_extent"] == 32
    capsule["inputs"][1]["shape"] = [1040, 31]
    with pytest.raises(ValueError, match="reduction extents differ"):
        _integer_reference_bound({"numerical_semantics": semantics}, capsule)


def test_scope_chain_bounds_its_embedded_integer_contraction():
    semantics = {
        "internal_arithmetic": {
            "full_operation_overflow_policy": "bounded_exact_requires_each_partial_sum",
            "mac_result_bits": 20,
            "signed_operand_bits": 8,
        }
    }
    binding = CorpusBinding("fixture", 4, "int8", "i32", True, ["L0"], "exact_int")
    entry = {
        "name": "selected_scope",
        "kind": "model_slice",
        "source_role": "derived_sweep",
        "source_reference": "selected requirement",
        "op": "scope_chain",
        "M": 4,
        "K": 8,
        "N": 4,
        "scope_families": ["movement", "contraction", "elementwise_map"],
    }
    capsule, _ = build(entry, binding)
    proof = _integer_reference_bound({"numerical_semantics": semantics}, capsule)
    assert proof["status"] == "proven_safe"
    assert proof["reduction_extent"] == 8

    capsule["stimulus_range"] = [127, 127]
    capsule["inputs"][0]["shape"] = [4, 64]
    capsule["inputs"][1]["shape"] = [4, 64]  # scope_chain stores W transposed
    with pytest.raises(ValueError, match="may_overflow"):
        _integer_reference_bound({"numerical_semantics": semantics}, capsule)


def test_exact_conformance_cohort_cannot_borrow_unselected_siblings(tmp_path):
    binding = CorpusBinding("fixture", 4, "int8", "i32", True, ["L0"], "exact_int")
    members = {}
    for name, size in (("selected", 4), ("unselected", 7)):
        entry = {
            "name": name,
            "cat": "isa",
            "kind": "isa",
            "op": "matmul",
            "label": "public",
            "M": size,
            "K": size,
            "N": size,
            "source_role": "derived_sweep",
            "source_reference": "fixture",
        }
        capsule, program = build(entry, binding)
        directory = tmp_path / "isa" / name
        directory.mkdir(parents=True)
        (directory / "capsule.yaml").write_text(yaml.safe_dump(capsule))
        (directory / "capsule.interface.mlir").write_text(program)
        members[name] = directory
    requirement = {
        "cells": [{"cell": "contraction/i8/partial"}],
        "boundaries": {"tile_edge": 4},
        "composition": {"required": {"A": 1}},
    }
    selected = selected_cohort_coverage(requirement, [members["selected"]])
    assert selected["n_covered"] == 0
    assert selected["uncovered"] == ["contraction/i8/partial"]
    assert selected["composition"]["status"] == "not_measured"
    assert selected["composition"]["phase"] == "phase0"
    broad = selected_cohort_coverage(requirement, [tmp_path / "isa"])
    assert broad["n_covered"] == 1
    inputs = {"schema": CC.INPUT_SCHEMA, "target": "fixture", "capability_contract": {}}
    assert (
        CC._selected_program({"kind": "layer", "interface_mlir": "capsule.interface.mlir"}, members["selected"])
        is not None
    )
    assert (
        CC._selected_program(
            {"kind": "layer", "interface_mlir": "capsule.interface.mlir", "linalg_mlir": "missing.mlir"},
            members["selected"],
        )
        is None
    )
    assert (
        CC._selected_program(
            {"kind": "layer", "interface_mlir": "capsule.interface.mlir", "linalg_mlir": None}, members["selected"]
        )
        is None
    )
    report = CC.observe_cohort(inputs, [members["selected"]], target="fixture")
    assert not [row for row in report["blockers"] if row["component"] == "capsule"]
    CC.verify_cohort_binding(report, inputs, [members["selected"]])
    interface = members["selected"] / "capsule.interface.mlir"
    interface.write_text(interface.read_text() + "\n")
    with pytest.raises(ValueError, match="exact admitted-cohort bytes"):
        CC.verify_cohort_binding(report, inputs, [members["selected"]])
    interface.write_text("malformed IR")
    malformed = CC.observe_cohort(inputs, [members["selected"]], target="fixture")
    assert any(row["reason"] == "capsule program is not inventoried" for row in malformed["blockers"])


def test_selected_memory_regime_uses_frozen_facts_not_ambient_target(tmp_path, monkeypatch):
    from merlin.targetgen.rtl import facts as rtl_facts

    binding = CorpusBinding("fixture", 4, "int8", "i32", True, ["L0"], "exact_int")
    capsule, program = build(
        {
            "name": "selected",
            "cat": "isa",
            "kind": "isa",
            "op": "matmul",
            "label": "public",
            "M": 4,
            "K": 4,
            "N": 4,
            "source_role": "derived_sweep",
            "source_reference": "fixture",
        },
        binding,
    )
    member = tmp_path / "isa" / "selected"
    member.mkdir(parents=True)
    (member / "capsule.yaml").write_text(yaml.safe_dump(capsule))
    (member / "capsule.interface.mlir").write_text(program)
    artifact = {
        "schema_version": "2.0",
        "inputs": {},
        "facts": {
            "arrays": [{"name": "mesh", "rows": 16, "cols": 16}],
            "datapaths": [{"name": "input", "dtype": "i8", "evidence": "scratchpad smem"}],
            "memories": [{"name": "scratchpad", "bytes": 4096, "depth": 64}],
        },
    }
    raw = json.dumps(artifact, sort_keys=True).encode()
    digest = hashlib.sha256(raw).hexdigest()
    requirement = {
        "target": "fixture",
        "cells": [],
        "memory_mapping": {"required": {"fits_double": ["model"]}},
        "derivation": {"phase0_execution": {"raw_facts_sha256": digest}},
    }
    inputs = {"evidence": {"raw_facts_sha256": digest}, "raw_facts_utf8": raw.decode()}
    monkeypatch.setattr(rtl_facts, "load_facts", lambda *_: pytest.fail("ambient RTL facts were read"))
    coverage = selected_cohort_coverage(requirement, [member], inputs=inputs)
    assert coverage["memory_mapping"]["status"] == "ok"
    assert coverage["memory_mapping"]["n_covered"] == 1
    assert coverage["memory_mapping"]["covered_by"]["fits_double"] == ["selected"]
    with pytest.raises(ValueError, match="selected RTL facts differ"):
        selected_cohort_coverage(requirement, [member], inputs={**inputs, "raw_facts_utf8": raw.decode() + " "})


def test_selected_host_axes_do_not_resolve_an_ambient_contract(tmp_path, monkeypatch):
    from merlin.targetgen import target_registry

    member = tmp_path / "model_slices" / "host_matmul"
    member.mkdir(parents=True)
    (member / "capsule.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "host_matmul",
                "kind": "model_slice",
                "label": "public",
                "linalg_mlir": "capsule.interface.mlir",
                "semantic": {"semantic_family": "contraction"},
                "inputs": [
                    {"name": "A", "role": "input", "shape": [4, 4], "dtype": "f32"},
                    {"name": "B", "role": "weight", "shape": [4, 4], "dtype": "f32"},
                ],
                "operation": {"op": "matmul", "attributes": {"dtype": "f32"}},
            }
        )
    )
    (member / "capsule.interface.mlir").write_text(
        'module attributes {prov.level = "linalg-on-tensors"} { '
        "func.func @forward(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> { "
        "%0 = tensor.empty() : tensor<4x4xf32> "
        "%1 = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) "
        "outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32> "
        "return %1 : tensor<4x4xf32> } }"
    )
    contract = {"name": "fixture", "compute_units": []}
    raw_contract = (json.dumps(contract, sort_keys=True, indent=2) + "\n").encode()
    digest = hashlib.sha256(raw_contract).hexdigest()
    requirement = {
        "target": "fixture",
        "cells": [],
        "host_only": {"families": ["contraction"]},
        "host_lane": {"required": [{"family": "contraction", "dtype": "f32"}]},
        "derivation": {"phase0_execution": {"contract_sha256": digest}},
    }
    monkeypatch.setattr(target_registry, "load_contract", lambda *_: pytest.fail("ambient contract was read"))
    coverage = selected_cohort_coverage(requirement, [member], inputs={"capability_contract": contract})
    assert coverage["host_only"]["status"] == "ok"
    assert coverage["host_only"]["covered_by"] == {"contraction": ["host_matmul"]}
    assert coverage["host_lane"]["status"] == "ok"
    assert coverage["host_lane"]["covered_by"] == {"contraction/f32": ["host_matmul"]}
    with pytest.raises(ValueError, match="selected capability contract differs"):
        selected_cohort_coverage(
            requirement,
            [member],
            inputs={"capability_contract": {**contract, "compute_units": [{}]}},
        )
