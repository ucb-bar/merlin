"""Lossless composition into the established numerical and ISA record formats."""

from __future__ import annotations

from merlin.target.spec.source import Composition, SpecError, fields

_NUMERICAL = {
    "model",
    "operand_dtype",
    "accumulator_dtype",
    "readout_dtype",
    "subnormal_operand_flush",
    "input_domain",
    "output_domain",
    "rounding",
    "reduction_order",
    "reduction_cadence",
    "product_rounding",
    "overflow",
    "internal_arithmetic",
    "readout",
    "block_size",
    "scale_encoding",
    "operand_rounding",
    "scale_rule",
    "bf16_source_subnormal_flush",
    "zero_block_scale_e8m0",
}
_OPERATION = {
    "id",
    "ops",
    "families",
    "family",
    "placement",
    "numerical_contract",
    "evidence",
    "status",
    "description",
    "hardware",
    "signature",
}
_FORMAT = {
    "id",
    "operand_dtype",
    "accumulator_dtype",
    "eligible_operations",
    "unit",
    "numerical_semantics",
    "framework",
    "evidence",
    "description",
    "activation_zero_point",
    "weight_zero_point",
    "block_size",
    "subnormal_operand_flush",
    "activation_mode",
    "weight_mode",
    "site_modes",
    "scale_encoding",
    "activation_granularity",
    "weight_granularity",
}


def check_record(value: dict, kind: str) -> None:
    """Close lossy v1 boundaries before invoking their existing normalizers."""
    from merlin.targetgen.software_spec import _SIGNATURE_FIELDS, validate_numerical_semantics

    if kind == "numerical_semantics":
        fields(value, _NUMERICAL, kind)
        fields(value.get("model"), {"engine", "source_root_env", "source_root_path"}, "numerical model", {"engine"})
        if "internal_arithmetic" in value:
            internal = fields(
                value["internal_arithmetic"],
                {
                    "signed_operand_bits",
                    "mac_result_bits",
                    "mac_result_overflow",
                    "full_operation_overflow_policy",
                    "operand_domain",
                    "addend_domain",
                    "output_overflow",
                    "product",
                    "accumulator",
                    "cross_tile",
                },
                "internal arithmetic",
            )
            if "operand_domain" in internal:
                fields(internal["operand_domain"], {"exponent_range", "signed_zero"}, "operand domain")
        if "readout" in value:
            fields(
                value["readout"],
                {"acc_scale_rounding", "narrowing", "requant_rounding", "hardware_integer_shift_rounding", "shift"},
                "numerical readout",
            )
        validate_numerical_semantics(value)
    elif kind == "signature":
        fields(value, _SIGNATURE_FIELDS, kind)
    elif kind == "operation":
        fields(value, _OPERATION | set(_SIGNATURE_FIELDS), kind)
        if "signature" in value:
            check_record(value["signature"], "signature")
    elif kind == "quantization_format":
        fields(value, _FORMAT, kind)
        if "numerical_semantics" in value:
            check_record(value["numerical_semantics"], "numerical_semantics")
    elif kind == "quantization_element_formats":
        from merlin.target.spec.quantization import validate_element_formats

        validate_element_formats(value)
    elif kind == "quantization_output_requantization":
        from merlin.target.spec.quantization import validate_output_requantization

        validate_output_requantization(value)
    elif kind == "isa":
        from merlin.targetgen.isa_spec import validate_isa_spec

        validate_isa_spec(value)
    elif kind == "transfer_contracts":
        from merlin.targetgen.transfer_contracts import validate_transfer_contracts

        fields(value, {"status", "declarations"}, kind, {"declarations"})
        declarations = value["declarations"]
        rows = (
            declarations.values()
            if isinstance(declarations, dict)
            else declarations
            if isinstance(declarations, list)
            else []
        )
        for row in rows:
            fields(row, {"id", "from", "to", "copy", "valid_window", "evidence", "signature", "semantics"}, "transfer")
        validate_transfer_contracts(value)
    elif kind == "computation":
        if value.get("status") == "UNKNOWN":
            fields(value, {"status", "reason"}, kind, {"status", "reason"})
        else:
            fields(value, {"kind", "indexing_maps", "indexing_maps_explicit", "iterator_types", "scalar_body"}, kind)
            check_record(value.get("scalar_body"), "scalar_body")
        from merlin.target.spec.computation import _computation

        _computation(value, label="computation", unknowns=[])
    elif kind == "scalar_body":
        fields(value, {"arguments", "captures", "operations", "yields", "effects"}, kind)
        from merlin.target.spec.computation import _scalar_body

        _scalar_body(value, label="scalar body", unknowns=[])
        for row in value.get("operations", []):
            fields(row, {"op", "operands", "results", "attributes", "effects", "regions"}, "scalar operation")
        for row in value.get("captures", []):
            fields(row, {"source", "shape", "dtype", "result_index", "const_value"}, "scalar capture")


def normalize(composition: Composition) -> tuple[dict, dict, list[str], list[dict]]:
    from merlin.targetgen.software_spec import SCHEMA as V1_SCHEMA
    from merlin.targetgen.software_spec import validate_software_spec

    source = fields(
        composition.document,
        {
            "schema",
            "target",
            "status",
            "imports",
            "definitions",
            "format_registry_sha256",
            "numerical_semantics",
            "operations",
            "quantization",
            "restrictions",
            "transfer_contracts",
            "isa",
            "capability_contract",
            "evidence",
            "computations",
        },
        "software spec v2",
        {"schema", "target", "numerical_semantics", "operations"},
    )
    candidate = {"schema": V1_SCHEMA, "target": source["target"], "status": source.get("status", "unreviewed")}
    for field in ("evidence", "restrictions", "capability_contract"):
        if field in source:
            candidate[field] = source[field]
    for field in ("numerical_semantics", "isa", "transfer_contracts"):
        if field in source:
            candidate[field] = composition.expand(source[field], field)
            check_record(candidate[field], field)
    operations = source["operations"]
    if not isinstance(operations, dict) or not operations:
        raise SpecError("software spec v2 operations must be a nonempty mapping")
    candidate["operations"] = {}
    for name, row in operations.items():
        expanded = composition.expand(row, "operation")
        check_record(expanded, "operation")
        candidate["operations"][name] = expanded
    if "quantization" in source:
        quantization = fields(
            source["quantization"],
            {"formats", "element_formats", "output_requantization"},
            "v2 quantization (application methodology must be explicitly selected separately)",
            {"formats"},
        )
        if not isinstance(quantization["formats"], list):
            raise SpecError("quantization.formats must be a list")
        formats = [composition.expand(row, "quantization_format") for row in quantization["formats"]]
        for row in formats:
            check_record(row, "quantization_format")
        candidate["quantization"] = {"formats": formats}
        for field in ("element_formats", "output_requantization"):
            if field in quantization:
                kind = f"quantization_{field}"
                expanded = composition.expand(quantization[field], kind)
                check_record(expanded, kind)
                candidate["quantization"][field] = expanded
    candidate = validate_software_spec(candidate, source="compact expansion")
    selected = source.get("computations", {})
    if not isinstance(selected, dict):
        raise SpecError("computations must be a mapping of typed computation records")
    computations, obligations = {}, []
    for name, row in selected.items():
        from merlin.target.spec.computation import _computation

        expanded = composition.expand(row, "computation")
        check_record(expanded, "computation")
        unknowns = []
        computations[name] = {
            "computation": _computation(expanded, label=str(name), unknowns=unknowns),
            "unknowns": unknowns,
        }
        obligations.append(
            f"computations.{name}: no admitted v1 execution/proof projection for an authored typed computation"
        )
    # Validate every imported definition as part of the visible review closure.
    # Merely importing a definition does not select it as an operation or scheme.
    selected_names = set(composition.used)
    definitions = []
    for name, (kind, value, scope) in sorted(composition.definitions.items()):
        expanded = composition.expand(value, kind, scope=scope, stack=(name,))
        check_record(expanded, kind)
        definitions.append({"name": name, "kind": kind, "used": name in selected_names, "value": expanded})
    return candidate, computations, obligations, definitions
