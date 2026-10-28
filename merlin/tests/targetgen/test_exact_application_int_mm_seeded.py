"""One captured m2m_q.int_mm_seeded contraction becomes one inspectable, verified PyTorch slice (a matmul + seed bias)."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from merlin.targetgen import capsule_golden, capsule_inputs
from merlin.targetgen.application_inventory import INT_MM_SEEDED_OPERATION, exact_int_mm_geometry
from merlin.targetgen.capsule_common import load_capsule
from merlin.targetgen.capsule_source import M2MUnavailable, PytorchRefSource, verify_exact_application_int_mm, write_pytorch_capsule
from merlin.targetgen.corpus_synth import _application_operation_plan, exact_int_mm_entries, exact_int_mm_seeded_entries
from merlin.targetgen.golden_store import load_golden

M, K, N = 2, 32, 64


def _demand_and_inventory(operation=INT_MM_SEEDED_OPERATION):
    row = {
        "operation": operation,
        "mlir_operation": "linalg.generic",
        "frontend_op": operation,
        "provenance_op": "int_matmul",
        "semantic_family": "contraction",
        "operand_format": "int8",
        "accumulator_dtypes": ["i32"],
        "ordered_operand_types": [
            {"shape": [M, K], "dtype": "i8"},
            {"shape": [K, N], "dtype": "i8"},
            {"shape": [M, N], "dtype": "i32"},
        ],
        "ordered_result_types": [{"shape": [M, N], "dtype": "i32"}],
        "result_shapes": [[M, N]],
        "contraction_shape": {"M": M, "K": K, "N": N, "rank": 3},
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
                "operation": operation,
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


def test_seeded_and_unseeded_operations_never_cross_candidates():
    demands, full = _demand_and_inventory()
    seeded, refused = exact_int_mm_seeded_entries(demands, full)
    assert len(seeded) == 1 and not refused
    assert exact_int_mm_entries(demands, full) == ([], [])
    unseeded_demands, unseeded_full = _demand_and_inventory("aten._int_mm.default")
    assert exact_int_mm_seeded_entries(unseeded_demands, unseeded_full) == ([], [])
    assert len(exact_int_mm_entries(unseeded_demands, unseeded_full)[0]) == 1


def test_seeded_candidate_is_a_matmul_whose_only_stage_is_its_seed_bias():
    demands, full = _demand_and_inventory()
    entry = exact_int_mm_seeded_entries(demands, full)[0][0]
    assert (entry["M"], entry["K"], entry["N"]) == (M, K, N)
    assert entry["op"] == "matmul" and entry["capture_op"] == "int_mm_seeded"
    assert entry["epilogue"] == ["bias_add"] and entry["bias"] == "B" and entry["output_dtype"] == "i32"
    assert entry["name"].startswith("SY_int_mm_seeded_m2_k32_n64_")
    match = entry["application_signature_match"]
    assert match["status"] == "candidate_unverified" and match["operation"] == INT_MM_SEEDED_OPERATION
    row = full["applications"]["app"]["signatures"][0]
    assert exact_int_mm_geometry(row, operation=INT_MM_SEEDED_OPERATION) == (M, K, N)
    assert exact_int_mm_geometry(row) is None  # not an aten._int_mm
    plan = _application_operation_plan(demands, exact_entries=[entry])
    obligation = plan["obligations"][0]
    assert obligation["status"] == "candidate_unverified"
    assert obligation["writer_candidate"]["capture_op"] == "int_mm_seeded"
    assert obligation["capsule_candidates"] == [entry["name"]]
    # An unseeded entry cannot stand in for a seeded group.
    assert _application_operation_plan(demands, exact_entries=[])["obligations"][0].get("capsule_candidates") in (None, [])


def test_verification_refuses_the_unseeded_lowering_and_a_missing_guard():
    demands, full = _demand_and_inventory()
    match = exact_int_mm_seeded_entries(demands, full)[0][0]["application_signature_match"]
    with pytest.raises(M2MUnavailable):
        verify_exact_application_int_mm("module {}", match)


def _binding():
    from merlin.targetgen.corpus_spec import CorpusBinding

    return CorpusBinding(
        target="test",
        tile_dim=16,
        operand_dtype="i8",
        accum_dtype="i32",
        integer=True,
        tiers=["L0", "L1"],
        compare="exact_int",
        classes_for=lambda **_: [],
    )


def test_seeded_capsule_has_matching_linalg_guard_and_independent_exact_golden(tmp_path):
    source = PytorchRefSource()
    if not source.available():
        pytest.skip("model2MLIR capture interpreter is unavailable")
    demands, full = _demand_and_inventory()
    entry = exact_int_mm_seeded_entries(demands, full)[0][0]
    directory = write_pytorch_capsule(entry, _binding(), tmp_path, source=source)
    capsule = load_capsule(directory)
    golden = load_golden(directory)
    match = capsule["application_signature_match"]
    assert match["status"] == "verified_capture_match" and match["operation"] == INT_MM_SEEDED_OPERATION
    assert capsule["numeric_policy"] == {"compare": "exact_int", "dtype": "i32"}
    assert [row["name"] for row in capsule["inputs"]] == ["W", "A0", "B"] or {row["name"] for row in capsule["inputs"]} == {"A0", "W", "B"}
    bias = next(row for row in capsule["inputs"] if row["name"] == "B")
    assert bias["role"] == "bias" and bias["dtype"] == "i32" and bias["shape"] == [N]
    linalg = (directory / "capsule.linalg.mlir").read_text()
    assert 'prov.accumulator_init = "int32_seed"' in linalg and "seeded contraction may overflow" in linalg
    assert golden["golden_source"] == "host_torch_eager"
    assert capsule_inputs.is_exact_pytorch_integer_source(capsule)
    leaves = capsule_inputs.materialize_capsule_leaves(capsule)
    assert leaves["B"].dtype == "i32" and leaves["A0"].dtype == "i8"
    saved = golden["oracle_provenance"]["inputs"]
    assert saved["B"]["dtype"] == "i32"
    assert bytes.fromhex(saved["B"]["integer_bytes_hex"]) == b"".join(
        (value & 0xFFFFFFFF).to_bytes(4, "little") for value in leaves["B"].data
    )
    # The seed is nonzero and varied, so a dropped or mis-broadcast seed cannot match the golden.
    assert len(set(leaves["B"].data)) > 1 and any(leaves["B"].data)
    # Independent recomputation: exact int64 A @ W + seed (broadcast over rows), narrowed to i32 with no wrap.
    a, w, seed = (leaves[n] for n in ("A0", "W", "B"))
    rows = [
        [sum(a.data[m * K + k] * w.data[k * N + n] for k in range(K)) + seed.data[n] for n in range(N)]
        for m in range(M)
    ]
    assert all(-(1 << 31) <= value < (1 << 31) for row in rows for value in row)
    assert capsule_golden.golden(capsule, directory) == {"Y0": rows}
