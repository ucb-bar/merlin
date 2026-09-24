"""One captured W8A8 integer operation becomes one inspectable, verified PyTorch slice."""

from __future__ import annotations

import hashlib
import json

import pytest
import yaml

from merlin.targetgen.application_inventory import exact_int_mm_geometry
from merlin.targetgen.capsule_source import PytorchRefSource, write_pytorch_capsule
from merlin.targetgen.corpus_synth import SynthesisError, _application_operation_plan, exact_int_mm_entries


def _demand_and_inventory():
    row = {
        "operation": "aten._int_mm.default",
        "mlir_operation": "linalg.generic",
        "frontend_op": "aten._int_mm.default",
        "provenance_op": "int_matmul",
        "semantic_family": "contraction",
        "operand_format": "int8",
        "accumulator_dtypes": ["i32"],
        "ordered_operand_types": [
            {"shape": [2, 32], "dtype": "i8"},
            {"shape": [32, 64], "dtype": "i8"},
            {"shape": [2, 64], "dtype": "i32"},
        ],
        "ordered_result_types": [{"shape": [2, 64], "dtype": "i32"}],
        "result_shapes": [[2, 64]],
        "contraction_shape": {"M": 2, "K": 32, "N": 64, "rank": 3},
        "indexing_maps": [
            "affine_map<(d0, d1, d2) -> (d0, d2)>",
            "affine_map<(d0, d1, d2) -> (d2, d1)>",
            "affine_map<(d0, d1, d2) -> (d0, d1)>",
        ],
        "iterator_types": [
            "#linalg.iterator_type<parallel>",
            "#linalg.iterator_type<parallel>",
            "#linalg.iterator_type<reduction>",
        ],
        "body_operations": ["arith.extsi", "arith.extsi", "arith.muli", "arith.addi", "linalg.yield"],
        "quant_evidence": {"prov.quant_inner_1": "weights.int_data"},
        "disposition": "hardware_admitted",
        "count": 1,
        "ordinals": [4],
    }
    app = {
        "capture_sha256": "b" * 64,
        "capture_normalization": {"normalized_sha256": "c" * 64},
        "capture_quantization": "int8_dyn_act_int8_weight",
        "signatures": [row],
    }
    full = {"schema_version": 1, "applications": {"app": app}}
    digest = hashlib.sha256(json.dumps(full, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    demands = {
        "full_inventory_sha256": digest,
        "status": "inventoried",
        "n_operations": 1,
        "applications": {
            "app": {
                **{key: app[key] for key in ("capture_sha256", "capture_normalization", "capture_quantization")},
                "counts": {"hardware_admitted": 1},
            }
        },
        "operation_groups": [
            {
                "operation": "aten._int_mm.default",
                "mlir_operation": "linalg.generic",
                "semantic_family": "contraction",
                "operand_format": "int8",
                "disposition": "hardware_admitted",
                "shape_class": "contraction:rank_3",
                "count": 1,
                "sources": {"app": {"capture_sha256": app["capture_sha256"], "count": 1}},
            }
        ],
    }
    return demands, full


def test_exact_integer_matmul_bridge_refuses_weaker_signatures():
    demands, full = _demand_and_inventory()
    entries, refused = exact_int_mm_entries(demands, full)
    assert len(entries) == 1 and not refused
    entry = entries[0]
    assert (entry["M"], entry["K"], entry["N"]) == (2, 32, 64)
    assert entry["capture_op"] == "int_matmul" and entry["output_dtype"] == "i32"
    assert entry["application_signature_match"]["status"] == "candidate_unverified"

    row = full["applications"]["app"]["signatures"][0]
    assert exact_int_mm_geometry(row) == (2, 32, 64)
    for changed in ("ordered_operand_types", "indexing_maps", "body_operations", "quant_evidence"):
        broken = {**row, changed: None}
        assert exact_int_mm_geometry(broken) is None, changed
    with pytest.raises(SynthesisError, match="digest differs"):
        exact_int_mm_entries({**demands, "full_inventory_sha256": "a" * 64}, full)

    plan = _application_operation_plan(demands, exact_entries=entries)
    assert plan["status"] == "obligations_pending" and plan["blocked_operations"] == 0
    obligation = plan["obligations"][0]
    assert obligation["status"] == "candidate_unverified"
    assert obligation["coverage_status"] == "unverified"
    assert obligation["capsule_candidates"] == [entry["name"]]
    entry["application_signature_match"]["sources"][0]["ordinals"] = []
    partial = _application_operation_plan(demands, exact_entries=entries)
    assert partial["status"] == "blocked" and partial["blocked_operations"] == 1
    assert partial["obligations"][0]["status"] == "refused"


def test_integer_matmul_capsule_has_matching_linalg_and_independent_golden(tmp_path):
    source = PytorchRefSource()
    if not source.available():
        pytest.skip("model2MLIR capture interpreter is unavailable")
    from merlin.targetgen.corpus_spec import CorpusBinding

    demands, full = _demand_and_inventory()
    entry = exact_int_mm_entries(demands, full)[0][0]
    binding = CorpusBinding(
        target="test",
        tile_dim=16,
        operand_dtype="i8",
        accum_dtype="i32",
        integer=True,
        tiers=["L0", "L1"],
        compare="exact_int",
        classes_for=lambda **_: [],
    )
    directory = write_pytorch_capsule(entry, binding, tmp_path, source=source)
    capsule = yaml.safe_load((directory / "capsule.yaml").read_text())
    golden = yaml.safe_load((directory / "golden.yaml").read_text())
    assert capsule["application_signature_match"]["status"] == "verified_capture_match"
    assert capsule["source_role"] == "model_derived"
    assert capsule["numeric_policy"] == {"compare": "exact_int", "dtype": "i32"}
    assert "tensor<2x64xi32>" in (directory / "capsule.interface.mlir").read_text()
    assert 'prov.aten = "aten._int_mm.default"' in (directory / "capsule.linalg.mlir").read_text()
    assert golden["golden_source"] == "host_torch_eager"
    assert len(golden["outputs"]["Y0"]) == 2
