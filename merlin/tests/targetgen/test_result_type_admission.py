"""A legal operand does not by itself license a dtype-changing result."""

from merlin.targetgen.application_inventory import application_demand_inventory
from merlin.targetgen.compute_units import SemanticCapability
from merlin.targetgen.eligibility import RegionDescriptor, is_eligible
from merlin.targetgen.operation_accounting import admit_operation_row


def test_new_source_fields_preserve_positional_descriptor_arguments():
    capability = SemanticCapability("contraction", ("int8",), (2, 3))
    assert capability.ranks == (2, 3) and capability.operand_pairs is None
    region = RegionDescriptor("source", "matmul", "contraction", "int8", "int8", 4, 8, 2)
    assert (region.m, region.k, region.n) == (4, 8, 2)
    assert region.captured_input_formats is None


def test_conversion_requires_result_format_evidence() -> None:
    region = RegionDescriptor(family="elementwise_map", in_dtype="i8", out_dtype="f32")
    unknown = is_eligible(region, {"elementwise_map": SemanticCapability("elementwise_map", dtypes=("int8",))})
    assert not unknown.eligible and unknown.undetermined and unknown.refusal == "result_dtype_unknown"

    supported = is_eligible(
        region,
        {"elementwise_map": SemanticCapability("elementwise_map", dtypes=("int8",), result_dtypes=("fp32",))},
    )
    assert supported.eligible
    refused = is_eligible(
        region,
        {"elementwise_map": SemanticCapability("elementwise_map", dtypes=("int8",), result_dtypes=("int8",))},
    )
    assert not refused.eligible and not refused.undetermined and refused.refusal == "result_dtype"


def test_operation_accounting_uses_the_same_result_check() -> None:
    row = {
        "operation": "quantized_decomposed.dequantize_per_tensor.default",
        "mlir_operation": "quant_ext.dequantize_per_tensor",
        "semantic_family": "elementwise_map",
        "disposition": "hardware_admitted",
        "operand_format": "int8",
        "result_dtypes": ["f32"],
        "ordered_operand_types": [{"dtype": "i8", "shape": [1, 4]}],
        "ordered_result_types": [{"dtype": "f32", "shape": [1, 4]}],
        "shape_confidence": "result_type",
    }
    contract = {
        "compute_units": [
            {
                "name": "vector",
                "kind": "vector",
                "dtypes": ["int8"],
                "ops": ["add"],
                "semantic_capabilities": [{"family": "elementwise_map", "dtypes": ["int8"]}],
            }
        ]
    }
    decision = admit_operation_row(row, software_spec=None, capability_contract=contract)
    assert decision["hardware_admission"]["status"] == "unknown"
    assert decision["hardware_admission"]["refusal"] == "result_dtype_unknown"


def test_reviewed_host_placement_resolves_only_the_selected_application_demand(tmp_path) -> None:
    capture = tmp_path / "iteration" / "model.mlir"
    capture.parent.mkdir()
    capture.write_text(
        "builtin.module { func.func @forward(%x: tensor<1x2x3xi8>, %s: tensor<f32>, "
        "%z: tensor<i64>) -> tensor<1x2x3xf32> { "
        '%y = "quant_ext.dequantize_per_tensor"(%x, %s, %z) '
        '{prov.aten = "quantized_decomposed.dequantize_per_tensor.default", '
        'prov.op = "dequantize_per_tensor", prov.family = "quantize"} : '
        "(tensor<1x2x3xi8>, tensor<f32>, tensor<i64>) -> tensor<1x2x3xf32> "
        "func.return %y : tensor<1x2x3xf32> } }",
        encoding="utf-8",
    )
    operation = "quantized_decomposed.dequantize_per_tensor.default"
    contract = {
        "compute_units": [
            {
                "name": "vector",
                "kind": "vector",
                "dtypes": ["int8"],
                "ops": ["add"],
                "semantic_capabilities": [{"family": "elementwise_map", "dtypes": ["int8"]}],
            }
        ]
    }
    declaration = {
        "id": "exact_host_conversion",
        "status": "reviewed",
        "ops": [operation],
        "families": ["elementwise_map"],
        "placement": "host",
        "signature": {
            "ordered_operand_dtypes": ["i8", "f32", "i64"],
            "ordered_result_dtypes": ["f32"],
            "ranks": [3],
        },
    }
    software = {"status": "reviewed", "operations": [declaration]}
    host_document = {
        "schema": "merlin.host_capabilities.v1",
        "status": "reviewed",
        "compiler": {"package_sha256": "a" * 64, "dtype_strategy": "int8"},
        "operations": [declaration],
        "evidence": {},
    }
    host = {
        "selected": {
            "package_sha256": "a" * 64,
            "capability_spec_sha256": "b" * 64,
            "dtype_strategy": "int8",
            "capability_spec": host_document,
        }
    }
    options = {"detailed": True, "capability_contract": contract, "software_spec": software, "host_capabilities": host}
    admitted = application_demand_inventory({"iteration": capture}, "synthetic", **options)
    converted = next(
        row
        for row in admitted["applications"]["iteration"]["signatures"]
        if row["mlir_operation"] == "quant_ext.dequantize_per_tensor"
    )
    assert converted["disposition"] == "host_required"
    assert admitted["status"] == "inventoried"

    # Accelerator software semantics need not duplicate the separately selected
    # platform host declaration. Its exact typed review remains mandatory.
    software["operations"] = []
    separated = application_demand_inventory({"iteration": capture}, "synthetic", **options)
    converted = next(
        row
        for row in separated["applications"]["iteration"]["signatures"]
        if row["mlir_operation"] == "quant_ext.dequantize_per_tensor"
    )
    assert converted["disposition"] == "host_required"
    assert separated["status"] == "inventoried"

    host_document["operations"][0] = {**declaration, "status": "unreviewed"}
    unknown = application_demand_inventory({"iteration": capture}, "synthetic", **options)
    converted = next(
        row
        for row in unknown["applications"]["iteration"]["signatures"]
        if row["mlir_operation"] == "quant_ext.dequantize_per_tensor"
    )
    assert converted["disposition"] == "unclassified"
    assert unknown["status"] == "incomplete"
