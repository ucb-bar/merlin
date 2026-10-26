"""Explicit, target-owned software semantics, separate from extracted hardware facts.

Loading a spec does not discover targets, import a reference model, or certify its
claims. The selected capability declaration remains a prototype until reviewed
against the evidence recorded by Phase 0.
"""

from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path

import yaml

from merlin.targetgen.admission_conditions import (
    screen_conditions,
    validate_emission_obligations,
    validate_source_conditions,
)
from merlin.targetgen.operator_review import STATUSES as REVIEW_STATUSES
from merlin.targetgen.operator_review import review_accepted, review_record
from merlin.targetgen.transfer_contracts import screen_transfer_contract as screen_transfer_contract
from merlin.targetgen.transfer_contracts import validate_transfer_contracts

SCHEMA = "merlin.software_spec.v1"
_NUMERICAL_CONTRACTS = frozenset(
    {
        "operand_sum_exhaustive_i8_v1",
        "operand_sum_per_operand_exhaustive_i8_v1",
    }
)
_IEEE_ROUNDING = frozenset({"rne", "rmm", "rtz", "rdn", "rup"})
_REDUCTION_ORDERS = frozenset({"index_sequential", "tree", "pairwise"})
_REDUCTION_CADENCES = frozenset({"per_step", "single_final"})
_SIGNATURE_FIELDS = frozenset(
    {
        "operand_dtypes",
        "dtypes",
        "accumulator_dtype",
        "readout_dtype",
        "ranks",
        "layouts",
        "tails",
        "broadcasting",
        "aliasing",
        "scale_granularity",
        "epilogues",
        "composed_with",
        "shape_bounds",
        "restriction",
        "semantics",
        "family",
        "families",
        "ordered_operand_dtypes",
        "ordered_result_dtypes",
        "compute_dtypes",
        "quantization_parameters",
        "kinds",
        "max_tensor_operands",
        "expressions",
        "numerical_family",
    }
)
#: The arithmetic a declaration performs on the payload it admits. Each class names the executable
#: signature constraints that make the claim screenable, so a row cannot claim arithmetic the screen
#: never checks. ``none`` is byte movement or a literal fill: it may not declare arithmetic fields.
ARITHMETIC_CLASSES = {
    "none": (),
    "integer_multiply_accumulate": ("accumulator_dtype",),
    "integer_sum": ("kinds",),
    "integer_max": ("kinds",),
    "integer_affine": ("expressions",),
    "readout_stages": (),
    "integer_components": ("numerical_family",),
}
_ARITHMETIC_FIELDS = frozenset({"accumulator_dtype", "kinds", "expressions", "epilogues"})
#: The domain axes a declaration states explicitly when the spec selects ``operation_domains``.
#: An axis the declaration deliberately leaves open is named under ``unconstrained`` with a reason.
DOMAIN_AXES = ("ranks", "layouts", "tails", "aliasing")
_DOMAIN_PLACEMENTS = frozenset({"accelerator", "fused_accelerator"})


def validate_quantization_parameters(parameters: object, *, source: str) -> dict[str, int]:
    """Validate exact authored per-tensor quantization parameter constraints."""
    if (
        not isinstance(parameters, dict)
        or not parameters
        or set(parameters) - {"zero_point", "quant_min", "quant_max"}
        or any(type(value) is not int for value in parameters.values())
    ):
        raise ValueError(
            f"{source}: quantization_parameters must contain exact integer zero_point/quant_min/quant_max constraints"
        )
    if "quant_min" in parameters and "quant_max" in parameters and parameters["quant_min"] > parameters["quant_max"]:
        raise ValueError(f"{source}: inverted quantization bounds")
    return parameters


def software_spec_path_for_recipe(recipe: str | Path, document: dict | None = None) -> Path | None:
    """Resolve one explicit recipe reference; supplied bytes avoid reopening live input."""
    source = Path(recipe).expanduser().absolute()
    selected = document if document is not None else yaml.safe_load(source.read_bytes())
    if not isinstance(selected, dict):
        raise ValueError(f"{source}: recipe must be a mapping")
    declaration = selected.get("software_spec")
    if declaration is None:
        return None
    if isinstance(declaration, dict):
        if set(declaration) != {"provider", "resource"}:
            raise ValueError(f"{source}: provider software_spec needs provider and resource")
        provider_name, member = declaration["provider"], declaration["resource"]
        if not isinstance(provider_name, str) or not provider_name.strip() or not isinstance(member, str):
            raise ValueError(f"{source}: invalid software_spec provider selection")
        from merlin.targetgen.providers import ProviderRole, contained_resource
        from merlin.targetgen.target_registry import resolve

        selected_provider = resolve(provider_name)
        provider = selected_provider.provider
        if selected_provider.kind != "external" or provider is None or provider.role != ProviderRole.SUPPORT:
            raise ValueError(f"{source}: software_spec requires an explicitly selected OOT support provider")
        if provider.target != provider_name:
            raise ValueError(f"{source}: software_spec provider target differs from selection")
        return contained_resource(provider.root, member)
    if not isinstance(declaration, str) or not declaration.strip():
        raise ValueError(f"{source}: software_spec must be an explicit path or provider resource")
    member = Path(declaration).expanduser()
    selected_path = member if member.is_absolute() else source.parent / member
    return Path(os.path.abspath(selected_path))


def validate_numerical_semantics(document: dict) -> dict:
    """Validate the explicit engine selection without importing its implementation."""
    if not isinstance(document, dict):
        raise ValueError("numerical_semantics must be a mapping")
    reviews = document.get("policy_reviews", {})
    if not isinstance(reviews, dict):
        raise ValueError("numerical_semantics.policy_reviews must be a mapping")
    for field, review in reviews.items():
        if field not in document or not isinstance(review, dict) or review.get("status") not in REVIEW_STATUSES:
            raise ValueError(f"invalid numerical policy review for {field!r}")
    domain_review = reviews.get("calibration_domain")
    if domain_review is not None and (
        domain_review.get("scope") != "op_validation_only"
        or domain_review.get("composition_observers") != "independent_scheme_policy"
    ):
        raise ValueError("calibration_domain review must preserve independent composition observer policies")
    domain = document.get("calibration_domain")
    if domain is not None and (not isinstance(domain, dict) or domain.get("status") not in REVIEW_STATUSES):
        raise ValueError("calibration_domain requires an explicit review status")
    mesh = document.get("native_unit_contract", {}).get("mesh_mac", {})
    if "component_execution_constraints" in mesh:
        raise ValueError("component_execution_constraints belong in emission_obligations")
    if "emission_obligations" in mesh:
        validate_emission_obligations(mesh["emission_obligations"])
    model = document.get("model")
    if not isinstance(model, dict) or model.get("engine") not in {
        "specir_fp_reduce",
        "integer_reference",
        "mx_block_reference",
    }:
        raise ValueError("numerical_semantics.model.engine must select a supported independent model")
    for field in ("operand_dtype", "accumulator_dtype", "readout_dtype"):
        if not isinstance(document.get(field), str) or not document[field].strip():
            raise ValueError(f"numerical_semantics.{field} must be an explicit dtype")
    if type(document.get("subnormal_operand_flush")) is not bool:
        raise ValueError("numerical_semantics.subnormal_operand_flush must be explicit Boolean")
    for field in ("input_domain", "output_domain"):
        domain = document.get(field)
        if domain is not None and (
            not isinstance(domain, dict)
            or set(domain) != {"nonfinite"}
            or domain["nonfinite"] not in {"allow", "forbid"}
        ):
            raise ValueError(f"numerical_semantics.{field} requires an explicit allow/forbid nonfinite policy")
    readout = document.get("readout")
    if readout is not None:
        validate_readout(readout)
    if model["engine"] == "specir_fp_reduce":
        if document.get("rounding") not in _IEEE_ROUNDING:
            raise ValueError("numerical_semantics.rounding must select a supported IEEE mode")
        if document.get("reduction_order") not in _REDUCTION_ORDERS:
            raise ValueError("numerical_semantics.reduction_order must be explicitly supported")
        if document.get("reduction_cadence") not in _REDUCTION_CADENCES:
            raise ValueError("numerical_semantics.reduction_cadence must be explicitly supported")
        if document.get("product_rounding") != "accumulator_format":
            raise ValueError("the selected float model currently requires product_rounding: accumulator_format")
        if document["accumulator_dtype"] != document["readout_dtype"]:
            raise ValueError("the selected float model does not implement a distinct readout conversion")
    if model["engine"] == "mx_block_reference":
        from merlin.common.quant_formats import get as quant_format

        operand = quant_format(document["operand_dtype"])
        if operand.kind != "mx_block" or operand.scale.kind != "block_e8m0" or operand.scale.block is None:
            raise ValueError("mx_block_reference requires a registered block-scaled MX operand")
        if document.get("block_size") != operand.scale.block:
            raise ValueError("mx_block_reference block_size differs from the registered format")
        if document.get("scale_encoding") != "e8m0":
            raise ValueError("mx_block_reference requires E8M0 scale encoding")
        if document["accumulator_dtype"] != document["readout_dtype"]:
            raise ValueError("mx_block_reference requires the selected readout dtype")
        if document.get("operand_rounding") not in _IEEE_ROUNDING:
            raise ValueError("mx_block_reference requires an explicit operand rounding mode")
        for field in ("scale_rule", "product_rounding", "reduction_order", "reduction_cadence"):
            if not isinstance(document.get(field), str) or not document[field].strip():
                raise ValueError(f"mx_block_reference requires explicit {field}")
        if not isinstance(document.get("internal_arithmetic"), dict) or not document["internal_arithmetic"]:
            raise ValueError("mx_block_reference requires explicit internal arithmetic")
    if "source_root_env" in model and "source_root_path" in model:
        raise ValueError("model source_root_env and source_root_path are mutually exclusive")
    for field in ("source_root_env", "source_root_path"):
        if field in model and (not isinstance(model[field], str) or not model[field].strip()):
            raise ValueError(f"model.{field} must be a nonempty string")
    return copy.deepcopy(document)


def numerical_policy_review(spec: dict) -> dict:
    """Review the selected quantized model's observer and host IEEE policies."""
    numeric = spec.get("numerical_semantics", {})
    documents = {
        f"numerical_semantics.policy_reviews.{name}": review
        for name, review in numeric.get("policy_reviews", {}).items()
    }
    # Historical recipes include the original domain's review metadata in their
    # byte identity. A scoped sibling review owns current approval without
    # rewriting those already selected arithmetic artifacts.
    if "calibration_domain" in numeric and "calibration_domain" not in numeric.get("policy_reviews", {}):
        documents["numerical_semantics.calibration_domain"] = numeric["calibration_domain"]
    return {
        **review_record(*documents.values(), default="unreviewed"),
        "rows": {path: review_record(document, default="unreviewed") for path, document in documents.items()},
    }


def validate_readout(readout: object) -> dict:
    """Keep rational readouts valid and require the complete f32 arithmetic definition."""
    if not isinstance(readout, dict):
        raise ValueError("numerical_semantics.readout must be a mapping")
    strategy = readout.get("strategy")
    if strategy == "f32_product_rne":
        from merlin.targetgen.contract.schemas import validate

        validate(readout, "f32_readout")
    elif strategy == "exact_rational_host":
        if readout.get("requant_rounding") != "half_even":
            raise ValueError("exact_rational_host requires readout.requant_rounding: half_even (RNE)")
    else:
        raise ValueError(f"unknown readout strategy {strategy!r}")
    return copy.deepcopy(readout)


def readout_exactness_gate(strategy: str) -> dict:
    """Describe admission obligations, without claiming that any proof has run."""
    if strategy == "f32_product_rne":
        return {"definitional": strategy, "requires": "decoded_multiplier_bits_equal_formed_multiplier"}
    if strategy == "exact_rational_host":
        return {"proof": "prove_fp32_products", "requires": "rational_equality_proof"}
    raise ValueError(f"unknown readout strategy {strategy!r}")


def validate_compositions(block: object) -> dict:
    """Validate family chains; operation rosters cannot enter this declaration surface."""
    from merlin.targetgen.contract.schemas import validate

    validate(block, "software_compositions")
    seen = set()
    for chain in block.get("chains", []):
        if chain["id"] in seen:
            raise ValueError(f"duplicate composition id {chain['id']!r}")
        seen.add(chain["id"])
        _validate_composition_constraints(chain)
        _validate_composition_schemes(chain)
    return copy.deepcopy(block)


def _validate_composition_schemes(chain: dict) -> None:
    """A scheme is selected once: as a primary scheme or as one justified cross-check."""
    selection = chain.get("quantization_schemes")
    if selection is None:
        return
    selected = [(identity, ()) for identity in selection["primary"]]
    selected += [
        (row["scheme"], tuple(sorted(row.get("axes", {}).items()))) for row in selection.get("cross_checks", [])
    ]
    if len(set(selected)) != len(selected):
        raise ValueError(f"composition {chain['id']!r} selects a quantization scheme more than once")


def _validate_composition_constraints(chain: dict) -> None:
    constraints = chain["constraints"]
    names = {item if isinstance(item, str) else next(iter(item)) for item in constraints}
    if len(names) != len(constraints):
        raise ValueError("composition constraints must have distinct names")
    if (chain.get("extension") or chain.get("multiplicity")) and chain["producer"] != "contraction":
        raise ValueError("operand extension and shared-input multiplicity require a contraction producer")
    if chain["producer"] == "operand_sum" and not {"operand_multiplier_in", "load_minimum_not_negated"} <= names:
        raise ValueError("operand sum requires multiplier and minimum-load constraints")
    if "dataflows" in chain and "os_dataflow_requires" not in names:
        raise ValueError("dataflows require a constant-operand mesh partial-sum bound")
    if chain.get("extension") and not {"coefficient_integer_exact", "k_extension_within"} <= names:
        raise ValueError("constant integer operand extension requires coefficient and extended-K bounds")
    if chain.get("multiplicity") and not {"per_output_readout_independent", "shared_input_resident"} <= names:
        raise ValueError("shared-input multiplicity requires independent readouts and residency bounds")
    if (
        "max" in chain.get("reduction_kinds", [])
        and not {"max_requires_store_path_pooling", "scratchpad_source_pooling_fenced", "padded_max_requires"} <= names
    ):
        raise ValueError("max reduction requires store pooling, fence and padding constraints")


def load_software_spec(path: str | Path, target: str | None = None) -> dict:
    """Read one versioned authored spec, refusing malformed or foreign declarations."""
    source = Path(path)
    document = yaml.safe_load(source.read_bytes())
    return validate_software_spec(document, target=target, source=source)


def validate_software_spec(document: dict, target: str | None = None, *, source: str | Path = "software spec") -> dict:
    """Validate already-observed bytes without reopening a mutable input file."""
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError(f"{source}: software spec must declare schema {SCHEMA}")
    document = copy.deepcopy(document)
    document.setdefault("status", "unreviewed")
    name = document.get("target")
    if not isinstance(name, str) or not name.strip() or (target is not None and name != target):
        raise ValueError(f"{source}: software spec target differs from selected target {target!r}")
    if document.get("status") not in REVIEW_STATUSES:
        raise ValueError(f"{source}: software spec has an invalid review status")
    contract = document.get("capability_contract")
    if contract is not None and (not isinstance(contract, dict) or contract.get("name") != name):
        raise ValueError(f"{source}: optional capability_contract must match the selected target")
    validate_numerical_semantics(document.get("numerical_semantics"))
    quantization = document.get("quantization_contract")
    if quantization is not None and (
        not isinstance(quantization, dict)
        or quantization.get("status") not in REVIEW_STATUSES
        or not isinstance(quantization.get("storage"), list)
        or not quantization["storage"]
        or any(not isinstance(dtype, str) or not dtype for dtype in quantization["storage"])
    ):
        raise ValueError(f"{source}: quantization_contract requires review status and storage dtypes")
    operations = document.get("operations")
    if isinstance(operations, dict):
        from merlin.targetgen.semantic_families import FAMILIES

        normalized = []
        for identity, declaration in operations.items():
            if not isinstance(identity, str) or not identity.strip() or not isinstance(declaration, dict):
                raise ValueError(f"{source}: named operations require nonempty names and mappings")
            row = copy.deepcopy(declaration)
            if "id" in row and row["id"] != identity:
                raise ValueError(f"{source}: named operation id conflicts with its name")
            row["id"] = identity
            if not row.get("ops") and not row.get("families"):
                if identity not in FAMILIES:
                    raise ValueError(f"{source}: operation {identity!r} requires explicit ops or families")
                row["families"] = [identity]
            normalized.append(row)
        operations = normalized
        document["operations"] = operations
    if not isinstance(operations, list) or not operations:
        raise ValueError(f"{source}: operations must be a nonempty mapping or list of signature constraints")
    seen = set()
    for row in operations:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
            raise ValueError(f"{source}: operation requires an explicit id")
        if row["id"] in seen:
            raise ValueError(f"{source}: duplicate operation id {row['id']!r}")
        seen.add(row["id"])
        if "admission_conditions" in row:
            validate_source_conditions(row["admission_conditions"])
        if row.get("status", "reviewed") not in REVIEW_STATUSES:
            raise ValueError(f"{source}: operation {row['id']!r} has an invalid review status")
        if "numerical_contract" in row and (
            not isinstance(row["numerical_contract"], (str, dict))
            or (isinstance(row["numerical_contract"], str) and row["numerical_contract"] not in _NUMERICAL_CONTRACTS)
        ):
            raise ValueError(f"{source}: operation {row['id']!r} has an unsupported numerical contract")
        if "emission_obligations" in row:
            validate_emission_obligations(row["emission_obligations"])
        # The author names constraints directly; consumers keep one canonical
        # signature representation. Family selectors remain declaration fields.
        flat_constraints = set(row) & (_SIGNATURE_FIELDS - {"family", "families"})
        if flat_constraints:
            if "signature" in row:
                raise ValueError(f"{source}: operation {row['id']!r} mixes flat and nested signature constraints")
            unknown = (
                set(row)
                - flat_constraints
                - {
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
                    "arithmetic",
                    "unconstrained",
                    "emission_obligations",
                    "admission_conditions",
                }
            )
            if unknown:
                raise ValueError(f"{source}: operation {row['id']!r} has unknown fields {sorted(unknown)}")
            row["signature"] = {field: row.pop(field) for field in sorted(flat_constraints)}
        for selector in ("ops", "families"):
            if selector in row and (
                not isinstance(row[selector], list)
                or any(not isinstance(value, str) or not value.strip() for value in row[selector])
            ):
                raise ValueError(f"{source}: operation {row['id']!r} {selector} must be a string list")
        if "hardware" in row:
            # A fact-derived declaration: the selected facts fill its hardware-shaped fields at Phase 0
            # selection (merlin.targetgen.spec_fact_drift.resolve_spec). Authoring one of them as well
            # would leave two authorities for the same value.
            from merlin.targetgen.spec_fact_drift import DERIVED_FIELDS, FORMS, semantic_family

            if row["hardware"] not in FORMS:
                raise ValueError(f"{source}: operation {row['id']!r} hardware must be one of {list(FORMS)}")
            authored = sorted((set(row) | set(row.get("signature") or {})) & set(DERIVED_FIELDS))
            if authored:
                raise ValueError(
                    f"{source}: operation {row['id']!r} derives {authored} from facts; "
                    "narrow a derived value with a reasoned restriction instead"
                )
            if semantic_family(row) is None:
                raise ValueError(f"{source}: operation {row['id']!r} hardware form needs exactly one semantic family")
            row.setdefault("signature", {})
        elif row.get("placement") not in {"accelerator", "fused_accelerator", "host", "unknown"}:
            raise ValueError(f"{source}: operation {row['id']!r} has no explicit placement")
        signature = row.get("signature")
        if not isinstance(signature, dict) or (not signature and "hardware" not in row):
            raise ValueError(f"{source}: operation {row['id']!r} requires a signature mapping")
        for field in (
            "operand_dtypes",
            "dtypes",
            "epilogues",
            "composed_with",
            "families",
            "ordered_operand_dtypes",
            "ordered_result_dtypes",
            "compute_dtypes",
        ):
            if field in signature and (
                not isinstance(signature[field], list)
                or any(not isinstance(value, str) or not value for value in signature[field])
            ):
                raise ValueError(f"{source}: operation {row['id']!r} {field} must be a string list")
        if "kinds" in signature:
            from merlin.targetgen.compute_units import REDUCTION_KINDS

            if (
                not isinstance(signature["kinds"], list)
                or not signature["kinds"]
                or any(kind not in REDUCTION_KINDS for kind in signature["kinds"])
            ):
                raise ValueError(
                    f"{source}: operation {row['id']!r} kinds must be reduction kinds from {sorted(REDUCTION_KINDS)}"
                )
        if "max_tensor_operands" in signature and (
            type(signature["max_tensor_operands"]) is not int or signature["max_tensor_operands"] < 0
        ):
            raise ValueError(f"{source}: max_tensor_operands must be a nonnegative integer")
        if "expressions" in signature and (
            not isinstance(signature["expressions"], list)
            or not signature["expressions"]
            or any(value not in {"affine"} for value in signature["expressions"])
        ):
            raise ValueError(f"{source}: expressions must list supported expression classes (affine)")
        if "ranks" in signature and (
            not isinstance(signature["ranks"], list)
            or any(type(value) is not int or value < 0 for value in signature["ranks"])
        ):
            raise ValueError(f"{source}: operation {row['id']!r} ranks must be nonnegative integers")
        if "quantization_parameters" in signature:
            validate_quantization_parameters(
                signature["quantization_parameters"], source=f"{source}: operation {row['id']!r}"
            )
        if "shape_bounds" in signature:
            bounds = signature["shape_bounds"]
            if not isinstance(bounds, dict) or any(not isinstance(value, dict) for value in bounds.values()):
                raise ValueError(f"{source}: operation {row['id']!r} shape_bounds must map axes to constraints")
        if "execution_constraints" in signature:
            raise ValueError("execution_constraints belong in emission_obligations")
        _validate_row_domain(row, source)
    if "operation_domains" in document:
        validate_operation_domains(document, source=source)
    if "evidence" in document and not isinstance(document.get("evidence"), dict):
        raise ValueError(f"{source}: evidence must record the declaration's provenance and unknowns")
    from merlin.targetgen.quantization_spec import validate_quantization_declarations

    validate_quantization_declarations(document)
    if "compositions" in document:
        validate_compositions(document["compositions"])
    fallback = document["numerical_semantics"].get("quantized_native_fallback")
    if fallback is not None:
        if not isinstance(fallback, dict) or fallback.get("readout") not in {
            "full_int32_then_exact_host_rne",
            "full_int32_then_host_readout_per_scheme",
        }:
            raise ValueError("quantized_native_fallback.readout must select a supported host readout")
    from merlin.targetgen.spec_fact_drift import validate_restrictions

    validate_restrictions(document)
    if "transfer_contracts" in document:
        document["transfer_contracts"] = validate_transfer_contracts(document["transfer_contracts"])
    if "isa" in document:
        from merlin.targetgen.isa_spec import validate_isa_spec

        validate_isa_spec(document["isa"])
    return document


def _validate_row_domain(row: dict, source) -> None:
    """A declared arithmetic class must be backed by the constraints that screen it."""
    label = f"{source}: operation {row['id']!r}"
    signature = row.get("signature") or {}
    unconstrained = row.get("unconstrained")
    if unconstrained is not None:
        if not isinstance(unconstrained, dict) or not unconstrained:
            raise ValueError(f"{label} unconstrained must map domain axes to reasons")
        for axis, reason in unconstrained.items():
            if axis not in DOMAIN_AXES:
                raise ValueError(f"{label} unconstrained names {axis!r}, not a domain axis {list(DOMAIN_AXES)}")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"{label} unconstrained {axis} requires a written reason")
            if axis in signature:
                raise ValueError(f"{label} both constrains and leaves {axis} unconstrained")
    if "arithmetic" not in row:
        return
    arithmetic = row["arithmetic"]
    if arithmetic not in ARITHMETIC_CLASSES:
        raise ValueError(f"{label} arithmetic must be one of {sorted(ARITHMETIC_CLASSES)}")
    if arithmetic == "none":
        claimed = sorted(_ARITHMETIC_FIELDS & set(signature))
        if claimed:
            raise ValueError(f"{label} declares arithmetic none but constrains {claimed}")
        return
    # Fact-derived fields are filled at selection; a ``hardware`` row is checked on its authored ones.
    missing = [field for field in ARITHMETIC_CLASSES[arithmetic] if field not in signature]
    if missing and "hardware" not in row:
        raise ValueError(f"{label} arithmetic {arithmetic} requires constraints {missing}")
    kinds = set(signature.get("kinds") or ())
    if arithmetic == "integer_sum" and "kinds" in signature and kinds != {"sum"}:
        raise ValueError(f"{label} arithmetic integer_sum admits only sum reductions")
    if arithmetic == "integer_max" and "kinds" in signature and kinds != {"max"}:
        raise ValueError(f"{label} arithmetic integer_max admits only max reductions")


def validate_operation_domains(document: dict, *, source="software spec") -> dict:
    """Every device declaration states its shapes, layouts, tails, aliasing and arithmetic.

    Shape of the top-level block (schema ``merlin.software_spec.operation_domains.v1``)::

        operation_domains:
          schema: merlin.software_spec.operation_domains.v1
          status: reviewed | unreviewed | authored_pending_operator_review
          required: [ranks, layouts, tails, aliasing, arithmetic]   # any nonempty subset

    The domains themselves live on each operation declaration, where admission screens them:
    ``ranks`` (shapes), ``layouts``, ``tails`` and ``aliasing`` as signature constraints,
    ``placement`` (or a fact-derived ``hardware`` form), and ``arithmetic`` from
    :data:`ARITHMETIC_CLASSES`. An axis a declaration deliberately leaves open is named under
    ``unconstrained: {axis: reason}``. Every accelerator, fused or fact-derived declaration must
    satisfy each required axis; host rows are out of scope. Selected by the spec itself, so a spec
    without the block keeps its meaning. :func:`operation_domains` returns the resolved view.
    """
    block = document["operation_domains"]
    if not isinstance(block, dict) or block.get("schema") != "merlin.software_spec.operation_domains.v1":
        raise ValueError(f"{source}: operation_domains must declare schema merlin.software_spec.operation_domains.v1")
    required = block.get("required")
    allowed = {*DOMAIN_AXES, "arithmetic"}
    if not isinstance(required, list) or not required or set(required) - allowed or len(set(required)) != len(required):
        raise ValueError(f"{source}: operation_domains.required must list axes from {sorted(allowed)}")
    if block.get("status", "reviewed") not in REVIEW_STATUSES:
        raise ValueError(f"{source}: operation_domains has an invalid review status")
    for row in document["operations"]:
        if row.get("placement") not in _DOMAIN_PLACEMENTS and "hardware" not in row:
            continue
        signature, unconstrained = row.get("signature") or {}, row.get("unconstrained") or {}
        missing = [
            axis
            for axis in required
            if (axis == "arithmetic" and axis not in row)
            or (axis != "arithmetic" and axis not in signature and axis not in unconstrained)
        ]
        if missing:
            raise ValueError(f"{source}: device operation {row['id']!r} leaves domain axes {missing} undeclared")
    return copy.deepcopy(block)


def operation_domains(spec: dict) -> dict[str, dict]:
    """Per device declaration, its declared domain: one record per row for ledgers and reports.

    Each record has ``placement``, ``arithmetic`` and, per domain axis, the declared values or
    ``{"unconstrained": reason}``; an undeclared axis is ``None``. Read from a validated (and, for
    fact-derived rows, resolved) spec; it adds no admission rule of its own.
    """
    result = {}
    for row in spec.get("operations") or []:
        if row.get("placement") not in _DOMAIN_PLACEMENTS and "hardware" not in row:
            continue
        signature, unconstrained = row.get("signature") or {}, row.get("unconstrained") or {}
        record = {"placement": row.get("placement", "from_facts"), "arithmetic": row.get("arithmetic")}
        for axis in DOMAIN_AXES:
            if axis in signature:
                record[axis] = copy.deepcopy(signature[axis])
            elif axis in unconstrained:
                record[axis] = {"unconstrained": unconstrained[axis]}
            else:
                record[axis] = None
        result[row["id"]] = record
    return result


def capability_contract(spec: dict, *, base_contract: dict | None = None) -> dict:
    """Select backend/ISA policy independently from minimal authored semantics.

    Legacy inline declarations retain their exact ownership and precedence. A
    minimal spec requires an explicitly selected provider/config contract with
    the same target identity; this pure function never searches the filesystem or
    invents runtime, geometry, protocol or operation support from a target name.
    SW admission still screens the spec's operations independently of backend
    declarations, so a broad legacy provider cannot widen reviewed SW support.
    """
    selected = spec.get("capability_contract")
    if selected is None:
        selected = base_contract
    if not isinstance(selected, dict) or selected.get("name") != spec.get("target"):
        raise ValueError("minimal software spec requires an explicit same-target backend capability contract")
    return copy.deepcopy(selected)


def numerical_datapath(spec: dict) -> dict:
    """The selected numerical axes consumed by corpus bindings and golden engines."""
    semantics = validate_numerical_semantics(spec["numerical_semantics"])
    return {
        "operand_dtype": semantics["operand_dtype"],
        "accum_dtype": semantics["accumulator_dtype"],
        "subnormal_operand_flush": semantics["subnormal_operand_flush"],
        "numerical_semantics": semantics,
    }


def software_spec_identity(path: str | Path, spec: dict | None = None) -> dict:
    """Byte identity, distinct from any generated contract or hardware identity."""
    document = spec if spec is not None else load_software_spec(path)
    return {
        "schema": document["schema"],
        "target": document["target"],
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "status": document.get("status", "unreviewed"),
    }


def software_spec_references(path: str | Path, document: dict | None = None) -> dict[str, Path]:
    """Resolve only declared external model roots; never infer a sibling checkout.

    A tooling environment variable is a location selector, not an immutable source
    identity. The caller must inventory and freeze the actual selected source bytes.
    """
    source = Path(path).absolute()
    spec = document if document is not None else load_software_spec(source)
    model = spec["numerical_semantics"]["model"]
    value = model.get("source_root_path")
    if "source_root_env" in model:
        value = os.environ.get(model["source_root_env"])
        if not value:
            raise ValueError(f"{source}: selected model requires {model['source_root_env']}")
    if value is None:
        return {}
    root = Path(value).expanduser()
    if not root.is_absolute():
        root = source.parent / root
    return {"numerical_model": root.resolve()}


#: The operand/result types that identify WHICH typed instance of an operation a row declares, each
#: paired with the observed key it is compared to. An ordered list is compared element-wise; a
#: choice list against one observed format. Accumulator, readout and compute precision are not
#: here: they are how one typed instance is computed, so a mismatch there is that row's refusal.
_ORDERED_PRECISION_FIELDS = ("ordered_operand_dtypes", "ordered_result_dtypes")
_CHOICE_PRECISION_FIELDS = (("operand_dtypes", "operand_dtype"), ("dtypes", "operand_dtype"))


def _precision(value):
    """The registered format name, or the spelling itself (as ``admit_operation`` compares them)."""
    from merlin.common.quant_formats import get as quant_format

    try:
        return quant_format(value).name
    except (KeyError, ValueError, TypeError):
        return value


def _executable(value) -> bool:
    return isinstance(value, str) and bool(value) and value != "unknown" and " " not in value


def precision_disjoint(row: dict, signature: dict) -> list[str]:
    """Why ``row``'s declared precision provably excludes the observed ``signature`` (empty if it may not).

    Exactly the precision comparisons :func:`admit_operation` itself refuses on, restricted to
    observed values: an absent observation, prose or ``unknown`` never proves disjointness, so the
    row then keeps its exact-roster priority and screens the signature itself.
    """
    declared = row.get("signature") or {}
    reasons = []
    for key in _ORDERED_PRECISION_FIELDS:
        expected, actual = declared.get(key), signature.get(key)
        if not isinstance(expected, list) or not isinstance(actual, list):
            continue
        if not all(_executable(value) for value in [*expected, *actual]):
            continue
        expected_formats, actual_formats = [_precision(v) for v in expected], [_precision(v) for v in actual]
        if expected_formats != actual_formats:
            reasons.append(f"{key} {actual_formats} differs from declared {expected_formats}")
    observed_operand = signature.get("operand_dtype")
    expected = declared.get("ordered_operand_dtypes")
    if (
        _executable(observed_operand)
        and isinstance(expected, list)
        and expected
        and all(_executable(value) for value in expected)
        and _precision(observed_operand) not in [_precision(v) for v in expected]
    ):
        reasons.append(f"operand_dtype {_precision(observed_operand)!r} is no declared ordered operand format")
    for key, observed_key in _CHOICE_PRECISION_FIELDS:
        if key not in declared:
            continue
        choices = declared[key] if isinstance(declared[key], list) else [declared[key]]
        observed = signature.get(observed_key)
        if not choices or not _executable(observed) or not all(_executable(choice) for choice in choices):
            continue
        if _precision(observed) not in [_precision(choice) for choice in choices]:
            reasons.append(f"{observed_key} {_precision(observed)!r} not in declared {key}")
    return reasons


def _names_operation(row: dict, op: str) -> bool:
    """A row's explicit frontend roster names ``op``; a family-name query never matches an id."""
    from merlin.targetgen.semantic_families import FAMILIES

    if op in row.get("ops", []):
        return True
    # A list-form row without a roster may still be identified by the op it declares. A family
    # name is not such an op: a row whose id merely equals a family name is a family declaration
    # and is selected with that family's other rows, never in place of them.
    return not row.get("ops") and row.get("id") == op and op not in FAMILIES


def exact_declarations(spec: dict, op: str, signature: dict) -> tuple[list[dict], list[str]]:
    """The exact-op rows that narrow admission of ``op`` at ``signature``, and the ids set aside.

    A row is set aside only when it is a reviewed, fully executable HOST declaration (in a reviewed
    spec) whose declared precision provably excludes the observed signature. Such a row describes a
    different typed instance of the operation and cannot admit this one, so it must not shadow the
    family declarations. Every other exact row -- an accelerator restriction, an unreviewed or
    prose-qualified row, a row whose precision is unobserved or matches -- keeps its priority.
    """
    retained, ignored = [], []
    for row in spec.get("operations") or []:
        if not _names_operation(row, op):
            continue
        declared = row.get("signature") or {}
        if (
            row.get("placement") == "host"
            and review_accepted(spec, default="unreviewed")
            and review_accepted(row)
            and "hardware" not in row
            and not (set(declared) - _SIGNATURE_FIELDS)
            and not ({"restriction", "semantics"} & set(declared))
            and precision_disjoint(row, signature)
        ):
            ignored.append(row["id"])
        else:
            retained.append(row)
    return retained, ignored


def admit_operation(spec: dict, op: str, signature: dict, placement: str) -> dict:
    """Screen a concrete signature against authored constraints, never against target names.

    This is SW admission, not proof that hardware or a compiler implements the
    operation. Missing observations, prose-only constraints, unreviewed selections
    and explicitly unknown placement remain diagnostic unknowns.
    """
    from merlin.common.quant_formats import get as quant_format
    from merlin.targetgen.semantic_families import FAMILIES, from_op

    if not isinstance(signature, dict) or placement not in {"accelerator", "fused_accelerator", "host", "unknown"}:
        raise ValueError("operation admission requires an observed signature and explicit placement")
    family = signature.get("family") or from_op(op)
    rows = spec.get("operations") or []
    # An explicit frontend roster narrows this declaration, even when it also names a semantic
    # family. A roster written for a provably different typed instance of the operation (for
    # example the reviewed f32 host form of an op observed at int8) does not: it cannot admit this
    # signature, so the operation falls through to its family declarations.
    direct, ignored = exact_declarations(spec, op, signature)
    candidates = direct or [
        row
        for row in rows
        if not row.get("ops")
        and family is not None
        and (
            family in row.get("families", [])
            or family == row.get("family")
            or (row.get("id") in FAMILIES and family == row["id"])
        )
    ]
    if not candidates:
        decision = {"status": "unsupported", "reason": f"no SW operation declaration admits {op!r}", "op": op}
        if ignored:
            decision["ignored_disjoint_declarations"] = ignored
        return decision
    # A proved representation uses its own declaration, rather than bypassing its
    # narrowed execution domain through an ordinary storage-format declaration.
    represented = [
        row
        for row in candidates
        if signature.get("numerical_family") in (row.get("signature", {}).get("numerical_family") or [])
        and signature.get("numerical_family") is not None
    ]
    candidates = represented or candidates
    unresolved = sorted(row["id"] for row in candidates if "hardware" in row)
    if unresolved:
        raise ValueError(
            f"software declarations {unresolved} derive their hardware fields from facts; "
            "screen against the Phase 0 selection's resolved spec, not the authored bytes"
        )

    def canonical(value):
        try:
            return quant_format(value).name
        except (KeyError, ValueError, TypeError):
            return value

    decisions = []
    for row in candidates:
        constraints = row["signature"]
        missing, refused = [], []
        dependencies = []
        numeric_review = None
        arithmetic = placement in _DOMAIN_PLACEMENTS and row.get("arithmetic") not in {None, "none"}
        if arithmetic:
            numeric = spec.get("numerical_semantics", {})
            if numeric.get("policy_reviews") or numeric.get("calibration_domain"):
                numeric_review = numerical_policy_review(spec)
                dependencies.append({"status": numeric_review["review_status"]})
            quantization = spec.get("quantization_contract")
            if quantization is not None:
                dependencies.append(quantization)
                operand = signature.get("operand_dtype")
                if operand is None:
                    missing.append("quantization_contract.storage")
                elif canonical(operand) not in [canonical(dtype) for dtype in quantization["storage"]]:
                    refused.append("operand_dtype is outside quantization_contract.storage")
        special = (spec.get("native_ieee_special_values")
                   if signature.get("native_ieee_special_values") is True else None)
        for declaration in (row, special):
            if not isinstance(declaration, dict) or "admission_conditions" not in declaration:
                continue
            state, paths = screen_conditions(declaration["admission_conditions"], signature)
            if state == "unknown":
                missing.extend(paths)
            elif state == "refused":
                refused.extend(f"admission condition violated: {path}" for path in paths)
        if row.get("admission_basis") == "declared_unchecked" and not review_accepted(row, default="unreviewed"):
            missing.append("unchecked declaration requires explicit operation review")
        missing.extend(sorted(set(constraints) - _SIGNATURE_FIELDS))
        for declaration_key in ("restriction", "semantics"):
            if declaration_key in constraints:
                missing.append(f"{declaration_key}: prose obligation requires explicit review evidence")
        if "family" in constraints:
            if family is None:
                missing.append("family")
            elif constraints["family"] != family:
                refused.append("semantic family is not admitted")
        if "families" in constraints:
            if family is None:
                missing.append("families")
            elif family not in constraints["families"]:
                refused.append("semantic family is not admitted")
        for key in ("ordered_operand_dtypes", "ordered_result_dtypes", "compute_dtypes"):
            if key not in constraints:
                continue
            expected, actual = constraints[key], signature.get(key)
            if not isinstance(expected, list) or not isinstance(actual, list) or any(value is None for value in actual):
                missing.append(key)
            elif [canonical(value) for value in expected] != [canonical(value) for value in actual]:
                refused.append(f"{key} differs from the declared ordered precision signature")
        if "quantization_parameters" in constraints:
            actual = signature.get("quantization_parameters")
            if not isinstance(actual, dict):
                missing.append("quantization_parameters")
            else:
                for key, expected in sorted(constraints["quantization_parameters"].items()):
                    observed = actual.get(key)
                    if type(observed) is not int:
                        missing.append(f"quantization_parameters.{key}")
                    elif observed != expected:
                        refused.append(f"quantization_parameters.{key}={observed} differs from {expected}")
        if "max_tensor_operands" in constraints:
            count = signature.get("tensor_operands")
            if type(count) is not int:
                missing.append("tensor_operands")
            elif count > constraints["max_tensor_operands"]:
                refused.append(f"tensor_operands {count} exceeds {constraints['max_tensor_operands']}")
        declared_placement = row["placement"]
        if declared_placement == "unknown" or placement == "unknown":
            missing.append("placement")
        elif declared_placement != placement:
            refused.append(f"placement {placement!r} differs from declared {declared_placement!r}")
        # The declared arithmetic class is screened through its executable constraints; an observer
        # that also names the class is held to it, so movement never admits an arithmetic construction.
        observed_arithmetic = signature.get("arithmetic")
        if "arithmetic" in row and observed_arithmetic is not None and observed_arithmetic != row["arithmetic"]:
            refused.append(f"arithmetic {observed_arithmetic!r} differs from declared {row['arithmetic']!r}")
        # A standalone declaration does not authorize using the same family as an epilogue of a
        # different operation. The composition must be selected and reviewed explicitly, even if
        # the hardware also exposes a standalone path for this family.
        if signature.get("composed_with") and "composed_with" not in constraints:
            refused.append("composition is not admitted by this standalone declaration")
        for key, observed_key in (
            ("operand_dtypes", "operand_dtype"),
            ("dtypes", "operand_dtype"),
            ("accumulator_dtype", "accum_dtype"),
            ("readout_dtype", "readout_dtype"),
            ("ranks", "rank"),
            ("layouts", "layout"),
            ("tails", "tails"),
            ("broadcasting", "broadcasting"),
            ("aliasing", "aliasing"),
            ("scale_granularity", "scale_granularity"),
            ("kinds", "kind"),
            ("expressions", "expression"),
            ("numerical_family", "numerical_family"),
        ):
            if key not in constraints:
                continue
            expected, actual = constraints[key], signature.get(observed_key)
            if key == "numerical_family" and actual is None:
                # Representation-only rows cannot relax the source primitive constraints.
                # Their family is established by an exact construction proof, never inferred
                # from a missing observation in an untransformed source operation.
                refused.append("numerical_family requires an explicit representation proof")
                continue
            if actual is None or expected == "unknown":
                missing.append(key)
                continue
            # Lists and scalar equality are executable constraints. Authored
            # prose cannot be silently discarded or treated as an enum value.
            choices = expected if isinstance(expected, list) else [expected]
            if any(isinstance(choice, str) and " " in choice for choice in choices):
                missing.append(key)
                continue
            if "dtype" in key:
                actual, choices = canonical(actual), [canonical(choice) for choice in choices]
            if actual not in choices:
                refused.append(f"{observed_key} {actual!r} not in {choices!r}")
        if family == "elementwise_map" and signature.get("epilogues") and "epilogues" not in constraints:
            # An unconstrained standalone elementwise declaration is not a wildcard fused-stage
            # licence. A contraction's own declaration remains independent of its fused stage form.
            refused.append("epilogue is not admitted by this declaration")
        if "epilogues" in constraints:
            observed = signature.get("epilogues")
            from merlin.runtime.commandbuffer import BIAS_STAGES

            # The existing command-buffer ABI defines these as the same bias
            # stage; sharing its vocabulary avoids target-owned spelling forks.
            def normalize_stage(stage):
                return "bias_add" if stage in BIAS_STAGES else stage

            if observed is None:
                missing.append("epilogues")
            elif not isinstance(observed, list) or not {normalize_stage(stage) for stage in observed} <= {
                normalize_stage(stage) for stage in constraints["epilogues"]
            }:
                refused.append("epilogue is not admitted")
        if "composed_with" in constraints:
            composed = signature.get("composed_with")
            if composed is None:
                missing.append("composed_with")
            elif not set(composed) <= set(constraints["composed_with"]) or not composed:
                refused.append("required composition is not admitted")
        if isinstance(row.get("numerical_contract"), str) and row["numerical_contract"] in _NUMERICAL_CONTRACTS:
            witness = signature.get("numeric_screen")
            parameters = witness.get("parameters") if isinstance(witness, dict) else None
            valid = (
                isinstance(witness, dict)
                and witness.get("schema") == "merlin.operand_sum_numeric_screen.v1"
                and witness.get("form") == "fused_operand_sum"
                and witness.get("qualification") == "exhaustive_software_model_not_rtl_execution"
                and witness.get("pairs_checked") == 65536
                and isinstance(parameters, dict)
                and parameters.get("operand_dtype") == "i8"
                and parameters.get("output_dtype") == "i8"
                and type(parameters.get("bound_lsb")) is int
                and 0 <= parameters["bound_lsb"] <= 255
                and type(witness.get("max_error_lsb")) is int
                and witness["max_error_lsb"] >= 0
                and type(witness.get("n_over_bound")) is int
                and witness["n_over_bound"] >= 0
                and isinstance(witness.get("witness_sha256"), str)
                and len(witness["witness_sha256"]) == 64
                and all(char in "0123456789abcdef" for char in witness["witness_sha256"])
            )
            contract = row["numerical_contract"]
            if valid:
                valid = witness.get("numerical_contract", "operand_sum_exhaustive_i8_v1") == contract
            if valid and contract == "operand_sum_per_operand_exhaustive_i8_v1":
                valid = parameters.get("operand_sum_rounding") == "per_operand" and parameters["bound_lsb"] == 0
            if not valid or witness.get("status") not in {"within_bound", "exceeds_bound"}:
                missing.append("numerical_contract")
            elif (
                witness["status"] == "exceeds_bound"
                or witness["n_over_bound"] != 0
                or (witness["max_error_lsb"] > parameters["bound_lsb"])
            ):
                refused.append("numerical_contract exceeds the selected fact-derived error bound")
        for axis, bounds in (constraints.get("shape_bounds") or {}).items():
            dimensions = signature.get("dimensions") or {}
            observed = dimensions.get(axis)
            if type(observed) is not int or not isinstance(bounds, dict):
                missing.append(f"shape_bounds.{axis}")
                continue
            if set(bounds) - {"min", "max", "multiple_of"}:
                missing.append(f"shape_bounds.{axis}")
            for comparison, bound in bounds.items():
                if type(bound) is not int or bound < 1:
                    missing.append(f"shape_bounds.{axis}.{comparison}")
                elif (
                    (comparison == "min" and observed < bound)
                    or (comparison == "max" and observed > bound)
                    or (comparison == "multiple_of" and observed % bound)
                ):
                    refused.append(f"dimension {axis}={observed} violates {comparison}={bound}")
        review = review_record(
            {"status": spec.get("status", "unreviewed"), "evidence": spec.get("evidence", {})},
            row,
            *dependencies,
            *([special] if isinstance(special, dict) else []),
        )
        if signature.get("native_ieee_special_values") is True and (
            not isinstance(special, dict) or special.get("enabled") is not True
        ):
            missing.append("native_ieee_special_values")
        if refused:
            status, reason = "unsupported", "; ".join(refused)
        elif missing or not review["operator_accepted"]:
            status = "unknown"
            reason = "unresolved SW constraints: " + ", ".join(missing or ["operation review"])
        else:
            status = "admitted"
            reason = (
                "reviewed SW constraints match the observed signature"
                if review["review_status"] == "reviewed"
                else "authored SW constraints match under explicit operator-acceptance policy"
            )
        decisions.append(
            {
                "status": status,
                "reason": reason,
                "op": op,
                "declaration": row["id"],
                "admission_basis": row.get("admission_basis", "authored"),
                **({"emission_obligations": copy.deepcopy(row["emission_obligations"])}
                   if row.get("emission_obligations") else {}),
                "constraints_status": "refused" if refused else "unknown" if missing else "matched",
                "unresolved_constraints": missing,
                **({"numerical_policy_review": numeric_review} if numeric_review is not None else {}),
                **review,
            }
        )
    decision = next(
        (row for row in decisions if row["status"] == "admitted"),
        next((row for row in decisions if row["status"] == "unknown"), decisions[0]),
    )
    if ignored:
        decision = {**decision, "ignored_disjoint_declarations": ignored}
    return decision


def admit_contraction_operation(spec: dict, op: str, signature: dict, placement: str) -> dict:
    """Exact source-op priority for a separately typed integer MAC, as a view of :func:`admit_operation`.

    The disjoint-precision rule now lives in :func:`exact_declarations` and applies to every family;
    this wrapper keeps the contraction callers' record of which host rows were set aside and whether
    an exact source declaration still governs the decision. The authored specification is unchanged.
    """
    decision = admit_operation(spec, op, signature, placement)
    retained, ignored = exact_declarations(spec, op, signature)
    return {
        **decision,
        "ignored_disjoint_host_precision_declarations": ignored,
        "applicable_exact_source_declaration": op != signature.get("family") and bool(retained),
    }
