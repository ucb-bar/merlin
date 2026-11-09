"""Local typed hardware expressions create mandatory source-compatibility gaps.

Widths become conditional numeric requirements only for compatible independently
reviewed signed integer contraction semantics. No local expression grants a
complete contraction or imposes a fabricated accumulation order/K bound.
"""

from merlin.targetgen.rtl.hw_arithmetic import LAW, SCHEMA

from .component_generation import digest


def _bits(dtype):
    if not isinstance(dtype, str):
        return None
    prefix = next((name for name in ("int", "i") if dtype.startswith(name)), None)
    suffix = dtype[len(prefix) :] if prefix is not None else ""
    return int(suffix) if suffix.isdecimal() and int(suffix) > 0 else None


def required_unknowns(arithmetic, *, spec, contraction_owners):
    """Return required gaps, never pass witnesses or new software permissions."""
    if arithmetic.get("schema") != SCHEMA or arithmetic.get("complete_arithmetic_domain") is not False:
        raise ValueError("automatic numeric obligations need original incomplete local HW arithmetic facts")
    numeric = spec["numerical_semantics"]
    operand, accumulator = (_bits(numeric.get(key)) for key in ("operand_dtype", "accumulator_dtype"))
    compatible = (
        (numeric.get("model") or {}).get("engine") == "integer_reference"
        and numeric.get("overflow") == "bounded_exact"
        and operand is not None
        and accumulator is not None
    )
    rows = []
    for relation in arithmetic["relations"]:
        eligible = (
            sorted(contraction_owners)
            if (
                compatible
                and relation["law"] == LAW
                and all(arg["width"] == operand for arg in relation["operands"])
                and relation["addend"]["width"] == accumulator
            )
            else []
        )
        requirements = {
            "relation_sha256": digest(relation),
            "compatible_reviewed_owners": eligible,
            "signed_operand_bits": [arg["width"] for arg in relation["operands"]],
            "signed_addend_input_bits": relation["addend"]["width"],
            "modular_result_bits": relation["result_bits"],
            "scope": "conditional local numeric compatibility; no selected contraction path",
        }
        rows.append(
            {
                "id": "auto_missing_" + digest(requirements),
                "kind": "numeric_datapath",
                "selector": digest(relation),
                "reason": (
                    "local modular result requires source interval and actual selected accumulation-path proof"
                    if eligible
                    else "local arithmetic has no compatible reviewed signed integer semantic link"
                ),
                "requirements": requirements,
            }
        )
    domain = {
        "kind": "numeric_domain",
        "selector": "local_hw_arithmetic",
        "reason": "unrecognized outputs and cross-state/instance arithmetic retain mandatory UNKNOWN",
    }
    rows.append({"id": "auto_missing_" + digest(domain), **domain})
    return rows
