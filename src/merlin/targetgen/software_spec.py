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

from merlin.targetgen.transfer_contracts import screen_transfer_contract as screen_transfer_contract
from merlin.targetgen.transfer_contracts import validate_transfer_contracts

SCHEMA = "merlin.software_spec.v1"
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
    }
)


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
    model = document.get("model")
    if not isinstance(model, dict) or model.get("engine") not in {
        "specir_fp_reduce", "integer_reference", "mx_block_reference"
    }:
        raise ValueError("numerical_semantics.model.engine must select a supported independent model")
    for field in ("operand_dtype", "accumulator_dtype", "readout_dtype"):
        if not isinstance(document.get(field), str) or not document[field].strip():
            raise ValueError(f"numerical_semantics.{field} must be an explicit dtype")
    if type(document.get("subnormal_operand_flush")) is not bool:
        raise ValueError("numerical_semantics.subnormal_operand_flush must be explicit Boolean")
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
    if document.get("status") not in {"unreviewed", "reviewed"}:
        raise ValueError(f"{source}: software spec has an invalid review status")
    contract = document.get("capability_contract")
    if contract is not None and (not isinstance(contract, dict) or contract.get("name") != name):
        raise ValueError(f"{source}: optional capability_contract must match the selected target")
    validate_numerical_semantics(document.get("numerical_semantics"))
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
        if row.get("placement") not in {"accelerator", "fused_accelerator", "host", "unknown"}:
            raise ValueError(f"{source}: operation {row['id']!r} has no explicit placement")
        signature = row.get("signature")
        if not isinstance(signature, dict) or not signature:
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
        if "ranks" in signature and (
            not isinstance(signature["ranks"], list)
            or any(type(value) is not int or value < 1 for value in signature["ranks"])
        ):
            raise ValueError(f"{source}: operation {row['id']!r} ranks must be positive integers")
        if "shape_bounds" in signature:
            bounds = signature["shape_bounds"]
            if not isinstance(bounds, dict) or any(not isinstance(value, dict) for value in bounds.values()):
                raise ValueError(f"{source}: operation {row['id']!r} shape_bounds must map axes to constraints")
    if "evidence" in document and not isinstance(document.get("evidence"), dict):
        raise ValueError(f"{source}: evidence must record the declaration's provenance and unknowns")
    from merlin.targetgen.quantization_spec import validate_quantization_declarations

    validate_quantization_declarations(document)
    if "transfer_contracts" in document:
        document["transfer_contracts"] = validate_transfer_contracts(document["transfer_contracts"])
    return document


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
    direct = [row for row in rows if op in row.get("ops", []) or row.get("id") == op]
    candidates = direct or [
        row
        for row in rows
        if family is not None
        and (
            family in row.get("families", [])
            or family == row.get("family")
            or (row.get("id") in FAMILIES and family == row["id"])
        )
    ]
    if not candidates:
        return {"status": "unsupported", "reason": f"no SW operation declaration admits {op!r}", "op": op}

    def canonical(value):
        try:
            return quant_format(value).name
        except (KeyError, ValueError, TypeError):
            return value

    decisions = []
    for row in candidates:
        constraints = row["signature"]
        missing, refused = [], []
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
        declared_placement = row["placement"]
        if declared_placement == "unknown" or placement == "unknown":
            missing.append("placement")
        elif declared_placement != placement:
            refused.append(f"placement {placement!r} differs from declared {declared_placement!r}")
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
        ):
            if key not in constraints:
                continue
            expected, actual = constraints[key], signature.get(observed_key)
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
        if refused:
            status, reason = "unsupported", "; ".join(refused)
        elif missing or spec.get("status") != "reviewed":
            status = "unknown"
            reason = "unresolved SW constraints: " + ", ".join(missing or ["software spec review"])
        else:
            status, reason = "admitted", "reviewed SW constraints match the observed signature"
        decisions.append(
            {
                "status": status,
                "reason": reason,
                "op": op,
                "declaration": row["id"],
                "constraints_status": "refused" if refused else "unknown" if missing else "matched",
                "unresolved_constraints": missing,
                "review_status": spec.get("status", "unknown"),
            }
        )
    return next(
        (row for row in decisions if row["status"] == "admitted"),
        next((row for row in decisions if row["status"] == "unknown"), decisions[0]),
    )
