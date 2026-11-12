"""Closed target quantization declarations and structural semantic extensions.

These validators describe authored numerical records. Capability matching,
application preparation and capture orchestration remain separate consumers.
Structural intake does not select or qualify an execution implementation.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping

from merlin.common import quant_formats

_INTEGER_PARAMETERS = frozenset({"activation_zero_point", "weight_zero_point", "block_size"})

_BOOL_PARAMETERS = frozenset({"subnormal_operand_flush"})

SEMANTIC_EXTENSION_SCHEMAS = {
    "element_formats": "merlin.quantization.floating_elements.v1",
    "output_requantization": "merlin.quantization.output_requantization.v1",
}


def validate_element_formats(value: object) -> dict:
    """Validate explicit finite floating encodings, including named intermediates.

    This describes codes, not an encode/decode algorithm or a physical packing.
    Registered names must agree with the selected registry's element geometry.
    """
    if not isinstance(value, Mapping) or not value:
        raise ValueError("quantization.element_formats must be a nonempty mapping")
    fields = {"exponent_bits", "fraction_bits", "exponent_bias", "max_finite", "max_positive_code"}
    for name, row in value.items():
        label = f"quantization.element_formats.{name}"
        if not isinstance(name, str) or not name.isascii() or not name.isidentifier():
            raise ValueError("element format names must be ASCII identifiers")
        if not isinstance(row, Mapping) or set(row) != fields:
            raise ValueError(f"{label} requires exactly {sorted(fields)}")
        if any(type(row[key]) is not int for key in fields - {"max_finite"}):
            raise ValueError(f"{label} encoding fields must be exact integers")
        exponent, fraction = row["exponent_bits"], row["fraction_bits"]
        if exponent < 1 or fraction < 0 or exponent + fraction > 63:
            raise ValueError(f"{label} requires a positive exponent and at most 64 signed element bits")
        bias, code = row["exponent_bias"], row["max_positive_code"]
        if not 0 <= bias < 1 << exponent or not 0 < code < 1 << (exponent + fraction):
            raise ValueError(f"{label} bias or positive code exceeds its encoding")
        maximum = row["max_finite"]
        if type(maximum) not in (int, float) or maximum <= 0 or (type(maximum) is float and not math.isfinite(maximum)):
            raise ValueError(f"{label}.max_finite must be finite and positive")
        encoded_exponent, encoded_fraction = divmod(code, 1 << fraction)
        significand = (1 if encoded_exponent else 0) + encoded_fraction / (1 << fraction)
        try:
            decoded_maximum = math.ldexp(significand, (encoded_exponent or 1) - bias)
        except (OverflowError, TypeError) as exc:
            raise ValueError(f"{label} maximum is outside the supported finite encoding") from exc
        if decoded_maximum != maximum:
            raise ValueError(f"{label}.max_finite differs from its max_positive_code")
        try:
            registered = quant_formats.get(name)
        except KeyError:
            continue  # A local named intermediate does not register a public dtype.
        if (registered.exp_bits, registered.mant_bits, registered.element_bits, registered.signed) != (
            exponent,
            fraction,
            1 + exponent + fraction,
            True,
        ):
            raise ValueError(f"{label} differs from the registered element geometry")
    return copy.deepcopy(dict(value))


def validate_output_requantization(value: object) -> dict:
    """Validate the legacy source record without granting output conversion support.

    bf16_readout_code is an authored selector carried for an OOT adapter; its
    relation to an instruction encoding requires separate source-bound evidence.
    """
    fields = {"status", "source", "formats", "scale_resident", "bf16_readout_code", "requires_direct_consumer"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"quantization.output_requantization requires exactly {sorted(fields)}")
    if value["status"] not in {"reviewed", "unreviewed"} or value["source"] != "accumulator":
        raise ValueError("output requantization requires explicit review status and accumulator source")
    formats = value["formats"]
    if (
        not isinstance(formats, list)
        or not formats
        or any(not isinstance(name, str) or not name for name in formats)
        or len(set(formats)) != len(formats)
    ):
        raise ValueError("output requantization formats must be a nonempty unique dtype list")
    for name in formats:
        try:
            fmt = quant_formats.get(name)
        except KeyError as exc:
            raise ValueError(f"output requantization references unknown format {name!r}") from exc
        if not fmt.is_float or not fmt.is_block_scaled:
            raise ValueError("output requantization requires declared block-scaled floating formats")
    if any(type(value[key]) is not bool for key in ("scale_resident", "requires_direct_consumer")):
        raise ValueError("output requantization residency and consumer requirements must be Boolean")
    if type(value["bf16_readout_code"]) is not int or value["bf16_readout_code"] < 0:
        raise ValueError("output requantization readout selector must be a nonnegative exact integer")
    return copy.deepcopy(dict(value))


def semantic_extensions(spec: Mapping) -> list[dict]:
    """Expose preserved, closed records and unresolved execution obligations.

    An ordinary quant_recipe does not implement these declarations. Selection
    and qualification of an OOT realization remain separate from this intake.
    """
    quantization = spec.get("quantization") or {}
    records = []
    for key, validator in (
        ("element_formats", validate_element_formats),
        ("output_requantization", validate_output_requantization),
    ):
        if key in quantization:
            records.append(
                {
                    "schema": SEMANTIC_EXTENSION_SCHEMAS[key],
                    "source_field": f"quantization.{key}",
                    "value": validator(quantization[key]),
                    "structural_status": "supported",
                    "execution_status": "unknown",
                    "qualification": "not_established",
                    "obligations": [
                        "select and bind an implementation that preserves the complete numerical declaration",
                        "establish framework realization, reference arithmetic and hardware correspondence separately",
                    ],
                }
            )
    return records


def _unknown(value) -> bool:
    return value is None or (
        isinstance(value, str) and (value == "unknown" or value.startswith("unknown ") or " " in value)
    )


def _dtype(value: str) -> str:
    if value == "unknown":
        return value
    return quant_formats.get(value).name


def validate_quantization_declarations(spec: Mapping) -> list[dict]:
    """Validate references and normalize identities without rewriting authored bytes.

    Id-less legacy declarations have a stable identity from their canonical
    operand/accumulator pairing. Same-pair variants must have distinct explicit
    ids. Prose parameters remain visible unknowns, not executable constraints.
    """
    declaration = spec.get("quantization")
    if declaration is None:
        return []
    if not isinstance(declaration, Mapping) or not isinstance(declaration.get("formats"), list):
        raise ValueError("quantization.formats must be a list")
    unknown = set(declaration) - {"formats", "methodology", *SEMANTIC_EXTENSION_SCHEMAS}
    if unknown:
        raise ValueError(f"quantization has unknown fields {sorted(unknown)}")
    semantic_extensions(spec)
    if "methodology" in declaration:
        from merlin.compiler.quantization.methodology import validate as validate_methodology

        validate_methodology(declaration["methodology"])
    operation_ids = {row.get("id") for row in spec.get("operations") or [] if isinstance(row, Mapping)}
    result, seen = [], set()
    for index, original in enumerate(declaration["formats"]):
        label = f"quantization.formats[{index}]"
        if not isinstance(original, Mapping):
            raise ValueError(f"{label} must be a mapping")
        row = copy.deepcopy(dict(original))
        for field in ("operand_dtype", "accumulator_dtype"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label}.{field} must be an explicit registered dtype or unknown")
            try:
                row[field] = _dtype(value)
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{label}.{field}: unknown numeric format {value!r}") from exc
        identity = row.get("id", f"{row['operand_dtype']}__{row['accumulator_dtype']}")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError(f"{label}.id must be a nonempty string")
        if identity in seen:
            raise ValueError(f"duplicate quantization format id {identity!r}")
        seen.add(identity)
        row["id"] = identity
        references = row.get("eligible_operations")
        if references == "from_facts":
            # Filled at Phase 0 selection from the derived quantization recipes
            # (merlin.development.targetgen.spec_fact_drift.resolve_spec).
            result.append(row)
            continue
        if (
            not isinstance(references, list)
            or not references
            or any(not isinstance(value, str) or not value.strip() for value in references)
        ):
            raise ValueError(f"{label}.eligible_operations must be a nonempty operation-id list")
        if len(set(references)) != len(references):
            raise ValueError(f"{label}.eligible_operations contains duplicate references")
        foreign = set(references) - operation_ids
        if foreign:
            raise ValueError(f"{label}.eligible_operations references undeclared operations: {sorted(foreign)!r}")
        if "unit" in row and (not isinstance(row["unit"], str) or not row["unit"].strip()):
            raise ValueError(f"{label}.unit must be a nonempty compute-unit name")
        if "framework" in row and not isinstance(row["framework"], Mapping):
            raise ValueError(f"{label}.framework must be a mapping of explicit adapter parameters")
        for field in _INTEGER_PARAMETERS:
            if field in row and not _unknown(row[field]) and type(row[field]) is not int:
                raise ValueError(f"{label}.{field} must be an integer or unknown")
        if type(row.get("block_size")) is int and row["block_size"] < 1:
            raise ValueError(f"{label}.block_size must be positive")
        for field in _BOOL_PARAMETERS:
            if field in row and not _unknown(row[field]) and type(row[field]) is not bool:
                raise ValueError(f"{label}.{field} must be Boolean or unknown")
        for field in ("activation_mode", "weight_mode"):
            if field in row and not _unknown(row[field]) and row[field] not in {"static", "dynamic"}:
                raise ValueError(f"{label}.{field} must be static, dynamic, or unknown")
        if "site_modes" in row:
            if "activation_mode" in row or "weight_mode" in row:
                raise ValueError(f"{label}.site_modes cannot coexist with global activation/weight modes")
            sites = row["site_modes"]
            if not isinstance(sites, Mapping) or not sites:
                raise ValueError(f"{label}.site_modes must name at least one operation site")
            for site, modes in sites.items():
                if not isinstance(site, str) or not site.strip():
                    raise ValueError(f"{label}.site_modes requires nonempty site names")
                if not isinstance(modes, Mapping) or set(modes) != {"lhs", "rhs"}:
                    raise ValueError(f"{label}.site_modes[{site!r}] must declare lhs and rhs modes")
                if any(mode not in {"static", "dynamic", "unknown"} for mode in modes.values()):
                    raise ValueError(f"{label}.site_modes[{site!r}] modes must be static, dynamic, or unknown")
        if "numerical_semantics" in row:
            from merlin.development.targetgen.software_spec import validate_numerical_semantics

            semantics = validate_numerical_semantics(row["numerical_semantics"])
            if (
                _dtype(semantics["operand_dtype"]) != row["operand_dtype"]
                or _dtype(semantics["accumulator_dtype"]) != row["accumulator_dtype"]
            ):
                raise ValueError(f"{label}.numerical_semantics must match this format's operand/accumulator pairing")
        result.append(row)
    if "element_formats" in declaration:
        operands = {row["operand_dtype"] for row in result}
        if not operands <= set(declaration["element_formats"]):
            raise ValueError("element_formats must describe every declared quantization operand")
    if "output_requantization" in declaration:
        outputs = {_dtype(name) for name in declaration["output_requantization"]["formats"]}
        if not outputs <= {row["operand_dtype"] for row in result}:
            raise ValueError("output requantization references undeclared quantization operands")
    return result
