"""Coarse operation-accounting checks: complete partition, honest provenance, frozen inputs."""

import json
from collections import Counter
from copy import deepcopy

import pytest

from merlin.targetgen.operation_accounting import admit_operation_row, build_operation_accounting


def test_parsed_application_sources_are_capture_bound(tmp_path):
    from merlin.targetgen.application_inventory import application_demand_inventory

    raw = b"builtin.module { func.func @forward(%x: tensor<2xf32>) -> tensor<2xf32> { func.return %x : tensor<2xf32> } }"
    capture = tmp_path / "model.mlir"
    capture.write_bytes(raw)
    inventory = application_demand_inventory(
        {"sample": capture}, "test_device", detailed=True,
        capability_contract={"name": "test_device", "compute_units": []}, include_graph=True,
    )
    report = build_operation_accounting(inventory, application_sources={"sample": raw})
    assert report["applications"]["sample"]["capture_sha256"] == inventory["applications"]["sample"]["capture_sha256"]
    with pytest.raises(ValueError, match="selected capture bytes"):
        build_operation_accounting(inventory, application_sources={"sample": raw + b"\n"})
    with pytest.raises(ValueError, match="exact selected application roster"):
        build_operation_accounting(inventory, application_sources={})


@pytest.mark.parametrize(
    "carrier, form, status",
    [
        ("merlin_iface.movement", "copy", "admitted"),
        ("linalg.copy", "copy", "admitted"),
        ("memref.copy", "copy", "admitted"),
        ("linalg.transpose", "permutation", "unsupported"),
        ("linalg.generic", None, "unsupported"),
    ],
)
def test_movement_form_comes_from_the_ir_carrier_not_a_provenance_label(carrier, form, status):
    _, _, contract = _inputs()
    contract["compute_units"][0]["semantic_capabilities"] = [
        {"family": "movement", "dtypes": ["int8"], "forms": ["copy"], "layouts": ["row_major_contiguous"]}
    ]
    row = {
        "operation": "movement",
        "mlir_operation": carrier,
        "semantic_family": "movement",
        "disposition": "unclassified",
        "operand_format": "int8",
        "shape_confidence": "observed",
        "layout": "row_major_contiguous",
        "ordered_operand_types": [{"shape": [2, 3], "dtype": "i8"}],
        "ordered_result_types": [{"shape": [2, 3], "dtype": "i8"}],
        "result_dtypes": ["i8"],
    }
    result = admit_operation_row(row, software_spec=None, capability_contract=contract)
    assert result["observed_admission_signature"].get("form") == form
    assert result["hardware_admission"]["status"] == status
    row.pop("layout")
    unresolved = admit_operation_row(row, software_spec=None, capability_contract=contract)
    assert unresolved["hardware_admission"]["status"] == "unknown"


def test_exact_frontend_roster_does_not_hide_a_family_only_declaration():
    row = {
        "operation": "aten.synthetic_elementwise",
        "frontend_op": "aten.synthetic_elementwise",
        "mlir_operation": "linalg.generic",
        "semantic_family": "elementwise_map",
        "disposition": "host_required",
        "ordered_operand_types": [{"shape": [2, 3], "dtype": "f32"}],
        "ordered_result_types": [{"shape": [2, 3], "dtype": "f32"}],
    }
    spec = {
        "status": "reviewed",
        "operations": [
            {
                "id": "aten.synthetic_elementwise",
                "ops": ["aten.explicit"],
                "families": ["elementwise_map"],
                "placement": "host",
                "signature": {"ranks": [2]},
            },
            {
                "id": "family_only",
                "families": ["elementwise_map"],
                "placement": "host",
                "signature": {"ranks": [2]},
            },
        ],
    }
    result = admit_operation_row(row, software_spec=spec, capability_contract=None)
    assert result["matching_declarations"] == ["family_only"]
    assert result["software_admissions"][0]["status"] == "admitted"


def _inputs():
    rows = []
    for name, disposition, family, fmt, frontend, positions in (
        ("builtin.module", "structural", None, None, None, [0]),
        ("linalg.generic", "hardware_admitted", "contraction", "int8", "aten.mm.default", [1, 2]),
        ("arith.addi", "component", "elementwise_map", "int8", "aten.mm.default", [3, 4]),
        ("tensor.empty", "support_required", "movement", "int8", None, [5]),
        ("linalg.generic", "host_required", "normalization", "fp32", "aten.layer_norm.default", [6]),
        ("func.call", "unclassified", None, None, None, [7]),
    ):
        rows.append(
            {
                "operation": frontend or name,
                "mlir_operation": name,
                "frontend_op": frontend,
                "provenance_op": None,
                "semantic_family": family,
                "operand_format": fmt,
                "disposition": disposition,
                "reason": "old inventory annotation",
                "count": len(positions),
                "ordinals": positions,
                "ordered_operand_types": [{"shape": [4, 8], "dtype": "i8"}],
                "ordered_result_types": [{"shape": [4, 4], "dtype": "i32"}],
                "result_dtypes": ["i32"],
                "accumulator_dtypes": ["i32"] if family == "contraction" else None,
                "contraction_shape": {"M": 4, "K": 8, "N": 4, "rank": 3} if family == "contraction" else None,
            }
        )
    counts = Counter()
    for row in rows:
        counts[row["disposition"]] += row["count"]
    inventory = {
        "schema_version": 1,
        "status": "incomplete",
        "n_operations": 8,
        "applications": {
            "small_model": {
                "capture": "capture/model.mlir",
                "capture_sha256": "a" * 64,
                "n_operations": 8,
                "n_signatures": len(rows),
                "counts": dict(counts),
                "signatures": rows,
            }
        },
    }
    spec = {
        "status": "reviewed",
        "operations": [
            {
                "id": "matrix",
                "families": ["contraction"],
                "placement": "accelerator",
                "signature": {"operand_dtypes": ["int8"], "accumulator_dtype": "i32", "ranks": [2]},
            },
            {
                "id": "norm",
                "families": ["normalization"],
                "placement": "host",
                "signature": {"operand_dtypes": ["fp32"]},
            },
            {
                "id": "absent",
                "ops": ["aten.absent.default"],
                "placement": "accelerator",
                "signature": {"operand_dtypes": ["int8"]},
            },
        ],
    }
    contract = {
        "name": "test_device",
        "compute_units": [
            {
                "name": "matrix_unit",
                "kind": "systolic",
                "dtypes": ["int8"],
                "ops": ["matmul"],
                "semantic_capabilities": [{"family": "contraction", "dtypes": ["int8"], "ranks": [2]}],
            }
        ],
    }
    return inventory, spec, contract


def test_complete_partition_preserves_provenance_and_selected_admission_without_lowering_claims():
    inventory, spec, contract = _inputs()
    originals = deepcopy((inventory, spec, contract))
    catalog = {
        "schema": "merlin.pytorch_opset.v1",
        "status": "observed",
        "torch": "test_version",
        "all_ops": ["aten.mm.default", "aten.other.default"],
        "n_all_aten": 2,
        "ops": ["aten.mm.default"],
        "decomposed": [],
    }
    report = build_operation_accounting(inventory, spec, capability_contract=contract, framework_catalog=catalog)
    assert (inventory, spec, contract) == originals
    overall = report["overall"]
    assert overall["n_mlir_operations"] == sum(overall["classification_counts"].values()) == 8
    assert overall["classification_counts"] == {
        "accelerator_candidate": 2,
        "host_required": 1,
        "nested_component": 2,
        "structural": 1,
        "support_lowering_required": 1,
        "unresolved": 1,
    }
    provenance = overall["pytorch_provenance"]
    assert provenance["original_pytorch_invocation_count"] is None
    mm = next(row for row in provenance["groups"] if row["frontend_op"] == "aten.mm.default")
    assert mm["annotated_mlir_operation_count"] == 4  # includes two nested children, not four Torch calls
    assert provenance["unattributed_mlir_operations"] == 3
    matrix = report["applications"]["small_model"]["signatures"][1]
    assert matrix["observed_admission_signature"]["rank"] == 2  # not the 3-loop contraction rank
    assert matrix["ordinals"] == [1, 2]
    assert matrix["hardware_admission"]["status"] == "admitted"
    assert matrix["hardware_admission"]["units"] == ["matrix_unit"]
    assert overall["hardware_admission_counts"]["admitted"] == 2
    assert matrix["matching_declarations"] == ["matrix"]
    assert all(row["lowering_status"] == "unverified" for row in report["applications"]["small_model"]["signatures"])
    absent = next(row for row in report["declared_support_universe"]["operations"] if row["id"] == "absent")
    assert absent["observation_status"] == "absent_from_selected_workloads"
    assert absent["observed_mlir_operation_count"] == 0
    universe = report["framework_universe"]
    assert universe["observed_registered_aten_operators"] == ["aten.mm.default"]
    assert universe["unobserved_registered_aten_operators"] == ["aten.other.default"]
    assert universe["unlisted_frontend_operators"] == ["aten.layer_norm.default"]
    assert universe["accelerator_candidate_frontend_operators"] == ["aten.mm.default"]
    assert json.dumps(report, sort_keys=True) == json.dumps(
        build_operation_accounting(
            deepcopy(inventory),
            deepcopy(spec),
            capability_contract=deepcopy(contract),
            framework_catalog=deepcopy(catalog),
        ),
        sort_keys=True,
    )
    stale = build_operation_accounting(inventory, spec)
    assert stale["applications"]["small_model"]["signatures"][1]["classification"] == "unresolved"
    spec["status"] = "unreviewed"
    diagnostic = build_operation_accounting(inventory, spec, capability_contract=contract)
    assert diagnostic["overall"]["classification_counts"].get("accelerator_candidate", 0) == 0


def test_missing_inputs_are_distinct_from_an_explicitly_undeclared_workload():
    missing = build_operation_accounting(None)
    assert missing["status"] == "not_available"
    assert missing["overall"]["n_mlir_operations"] is None
    empty = build_operation_accounting(
        {"schema_version": 1, "status": "not_declared", "n_operations": 0, "applications": {}}
    )
    assert empty["status"] == "not_declared"
    assert empty["overall"]["n_mlir_operations"] == 0
    with pytest.raises(ValueError, match="detailed application inventory"):
        build_operation_accounting({})


def test_inconsistent_counts_and_lost_ordinals_cannot_create_a_clean_report():
    inventory, _, _ = _inputs()
    for mutate in (
        lambda app: app["signatures"][1].update(count=-1),
        lambda app: app["signatures"][1].update(ordinals=[1, 1]),
        lambda app: app.update(n_operations=9),
        lambda app: app["counts"].update(hardware_admitted=1),
        lambda app: app.update(capture_sha256="not-a-digest"),
    ):
        changed = deepcopy(inventory)
        mutate(changed["applications"]["small_model"])
        with pytest.raises(ValueError):
            build_operation_accounting(changed)


def test_capture_execution_attestation_is_carried_per_application_and_never_judged():
    inventory, spec, contract = _inputs()
    (label,) = inventory["applications"]
    attestation = {"schema": "fixture", "issuer": "anything"}
    report = build_operation_accounting(
        inventory, spec, capability_contract=contract, capture_execution_attestations={label: attestation}
    )
    assert report["applications"][label]["capture_execution_attestation"] == attestation
    assert report["applications"][label]["capture_execution_attestation"] is not attestation
    absent = build_operation_accounting(inventory, spec, capability_contract=contract)
    assert absent["applications"][label]["capture_execution_attestation"] is None
