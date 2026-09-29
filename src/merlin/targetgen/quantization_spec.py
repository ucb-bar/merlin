"""Frozen quantization declarations and their operation-scoped hardware candidates.

This is an auditable input contract, not another quantizer. Existing readout
derivation supplies :mod:`quant_recipe` candidates; existing layer planning and
TorchAO public extension adapters remain responsible for realization. A format
name, operand width, or declared operation never proves a framework route.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping

from merlin.common import quant_formats

SCHEMA = "merlin.phase0.quantization_contract.v1"
_IDENTITY_FIELDS = frozenset(
    {
        "id",
        "operand_dtype",
        "accumulator_dtype",
        "eligible_operations",
        "unit",
        "numerical_semantics",
        "framework",
        "evidence",
        "description",
    }
)
_INTEGER_PARAMETERS = frozenset({"activation_zero_point", "weight_zero_point", "block_size"})
_BOOL_PARAMETERS = frozenset({"subnormal_operand_flush"})


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
            from merlin.targetgen.software_spec import validate_numerical_semantics

            semantics = validate_numerical_semantics(row["numerical_semantics"])
            if (
                _dtype(semantics["operand_dtype"]) != row["operand_dtype"]
                or _dtype(semantics["accumulator_dtype"]) != row["accumulator_dtype"]
            ):
                raise ValueError(f"{label}.numerical_semantics must match this format's operand/accumulator pairing")
        result.append(row)
    return result


def _record(value, basis: str, *, not_applicable=False) -> dict:
    return {
        "status": "not_applicable" if not_applicable else "unknown" if _unknown(value) else "known",
        "value": copy.deepcopy(value),
        "basis": basis,
    }


def _unresolved(value, prefix: str) -> list[str]:
    if isinstance(value, Mapping):
        return [name for key, member in value.items() for name in _unresolved(member, f"{prefix}.{key}")]
    if isinstance(value, list):
        return [name for index, member in enumerate(value) for name in _unresolved(member, f"{prefix}[{index}]")]
    return [prefix] if _unknown(value) else []


def _parameters(
    declaration: dict, candidate: dict | None, facet: dict | None, semantics: dict
) -> tuple[dict, list[str], list[str]]:
    """Project existing recipe values; unknown authoring cannot be overwritten by a recipe."""
    recipe = (candidate or {}).get("recipe") or {}
    weight, activation = recipe.get("weight") or {}, recipe.get("activation") or {}
    scale = (facet or {}).get("scale") or {}
    fmt = quant_formats.get(declaration["operand_dtype"]) if declaration["operand_dtype"] != "unknown" else None
    floating = fmt is not None and fmt.is_float
    parameters = {
        "operand_dtype": _record(declaration["operand_dtype"], "authored format declaration"),
        "accumulator_dtype": _record(declaration["accumulator_dtype"], "authored format declaration"),
        "weight_granularity": _record(weight.get("granularity"), "selected quant_recipe.weight"),
        "activation_granularity": _record(activation.get("granularity"), "selected quant_recipe.activation"),
        "weight_mode": _record(weight.get("mode"), "selected quant_recipe.weight"),
        "activation_mode": _record(activation.get("mode"), "selected quant_recipe.activation"),
        "weight_zero_point": _record(
            0 if weight.get("symmetric") is True else None, "selected recipe symmetry", not_applicable=floating
        ),
        "activation_zero_point": _record(
            0 if activation.get("symmetric") is True else None, "selected recipe symmetry", not_applicable=floating
        ),
        "block_size": _record(
            weight.get("block"),
            "selected recipe block geometry",
            not_applicable=weight.get("granularity") in {"tensor", "channel"},
        ),
        "scale_encoding": _record(scale.get("dtype"), "selected readout scale carrier dtype"),
        "subnormal_operand_flush": _record(
            semantics.get("subnormal_operand_flush"),
            "selected numerical semantics",
            not_applicable=fmt is not None and not floating,
        ),
    }
    conflicts = []
    for key, value in declaration.items():
        if key in _IDENTITY_FIELDS:
            continue
        prior = parameters.get(key)
        if prior is not None and prior["status"] == "not_applicable" and _unknown(value):
            # An authored unknown cannot make block geometry required after the
            # selected per-tensor/per-channel recipe proved it has no block axis.
            continue
        if prior is not None and prior["status"] == "known" and not _unknown(value):
            authored, derived = value, prior["value"]
            if key == "scale_encoding":
                # A readout carrier and an authored SW declaration can spell the
                # same registered numeric format differently (fp32 versus f32).
                # Unknown/custom scale encodings retain exact comparison.
                try:
                    authored, derived = _dtype(authored), _dtype(derived)
                except (KeyError, ValueError, TypeError):
                    pass
            if authored != derived:
                conflicts.append(f"{key}: authored value {value!r} differs from derived value {prior['value']!r}")
        parameters[key] = _record(value, "authored quantization parameter")
    # The legacy shared granularity constrains both tensors. It is not a way to
    # silently request a finer weight scale than the selected readout can carry.
    granularity = declaration.get("scale_granularity")
    if granularity is not None and not _unknown(granularity):
        for role in ("weight", "activation"):
            observed = parameters[f"{role}_granularity"]
            if observed["status"] == "known" and observed["value"] != granularity:
                conflicts.append(f"scale_granularity: {role} recipe holds {observed['value']!r}, not {granularity!r}")
    unknowns = [key for key, value in parameters.items() if value["status"] == "unknown"]
    unknowns.extend(
        name
        for key, value in parameters.items()
        if value["status"] != "not_applicable"
        for name in _unresolved(value["value"], key)
    )
    unknowns.extend(_unresolved(semantics, "numerical_semantics"))
    return parameters, sorted(set(unknowns)), conflicts


def _hardware_matches(declaration: dict, snapshot: Mapping, semantics: dict) -> list[dict]:
    matches = []
    for candidate in snapshot.get("quantization_candidates") or []:
        if not isinstance(candidate, Mapping):
            raise ValueError("quantization candidates must be mappings")
        try:
            candidate_format = _dtype(candidate.get("format"))
        except (KeyError, ValueError, TypeError):
            continue
        if candidate_format != declaration["operand_dtype"]:
            continue
        if declaration.get("unit") and declaration["unit"] != candidate.get("unit"):
            continue
        facets = [
            facet
            for facet in snapshot.get("readout_facets") or []
            if isinstance(facet, Mapping) and facet.get("unit") == candidate.get("unit")
        ]
        facet = facets[0] if len(facets) == 1 else None
        parameters, unknowns, conflicts = _parameters(declaration, dict(candidate), facet, semantics)
        try:
            semantic_pair = (_dtype(semantics.get("operand_dtype")), _dtype(semantics.get("accumulator_dtype")))
        except (KeyError, ValueError, TypeError):
            semantic_pair = None
        if semantic_pair != (declaration["operand_dtype"], declaration["accumulator_dtype"]):
            unknowns.append("numerical_semantics.format_specific_selection")
        accumulator = (facet or {}).get("accumulator_dtype")
        if accumulator is None:
            unknowns.append("hardware.accumulator_dtype")
        else:
            try:
                if _dtype(accumulator) != declaration["accumulator_dtype"]:
                    conflicts.append("authored accumulator format differs from selected readout accumulator")
            except (KeyError, ValueError, TypeError):
                unknowns.append("hardware.accumulator_dtype")
        recipe = candidate.get("recipe") or {}
        if candidate.get("status") != "derived" or recipe.get("status") != "derived":
            unknowns.append("hardware.quantization_recipe")
        if not recipe.get("weight") or not recipe.get("activation"):
            unknowns.append("hardware.quantization_tensor_specs")
        for role in ("weight", "activation"):
            tensor = recipe.get(role) or {}
            try:
                if _dtype(tensor.get("dtype")) != declaration["operand_dtype"]:
                    conflicts.append(f"selected recipe {role} dtype differs from authored operand format")
            except (KeyError, ValueError, TypeError):
                unknowns.append(f"hardware.quantization_recipe.{role}.dtype")
        matches.append(
            {
                "unit": candidate.get("unit"),
                "status": "incompatible" if conflicts else "unknown" if unknowns else "candidate",
                "candidate": copy.deepcopy(dict(candidate)),
                "parameters": parameters,
                "unknowns": sorted(set(unknowns)),
                "conflicts": conflicts,
            }
        )
    return matches


def _observations(accounting: Mapping | None, declaration_id: str) -> list[dict]:
    observations = []
    applications = (accounting or {}).get("applications") or []
    applications = applications.values() if isinstance(applications, Mapping) else applications
    for application in applications:
        for signature in application.get("signatures") or []:
            if declaration_id in signature.get("matching_declarations", []):
                observations.append({"application": application.get("id", application.get("name")), **signature})
    return observations


def capture_recipe_candidates(spec: Mapping, quantization_contract: Mapping) -> list[dict]:
    """Project saved hardware recipes through explicit SW format/operation scopes.

    These are diagnostic transformation inputs, not reviewed accelerator support.
    Framework observer policy is recorded from the existing adapter, not inferred
    from RTL or added to the authored SW spec. Actual calibration and transformed
    precision must still be observed by the capture producer.
    """
    from merlin.targetgen._recipe_quantizer import DEFAULT_OBSERVER_EPSILON, _observer_name
    from merlin.targetgen.quant_recipe import digest
    from merlin.targetgen.semantic_families import from_op

    operations = {row["id"]: row for row in spec.get("operations") or []}
    result = []
    for declaration in quantization_contract.get("formats") or []:
        selected = declaration["declaration"]
        rows = [
            operations[identity]
            for identity in selected["eligible_operations"]
            if operations[identity]["placement"] == "accelerator"
        ]
        families = {family for row in rows for family in row.get("families", [])}
        families.update(family for row in rows for op in row.get("ops", []) if (family := from_op(op)))
        for match in declaration.get("hardware_matches") or []:
            if match["status"] != "candidate" or not rows:
                continue
            recipe = copy.deepcopy(match["candidate"]["recipe"])
            # Dynamic module transforms cannot enforce FX signature constraints.
            # Preserve them in the diagnostic contract, not as executable scoped recipes.
            if recipe["activation"]["mode"] != "static":
                continue
            allowed = [family for family in recipe["families"] if family in families]
            if not allowed:
                continue
            for family in set(recipe["families"]) - set(allowed):
                recipe.setdefault("unquantized", {})[family] = "not selected by the SW quantization operation scope"
            recipe["families"] = allowed
            recipe["accumulator_dtype"] = selected["accumulator_dtype"]
            recipe["software_admission"] = {
                "target": spec["target"],
                "status": spec["status"],
                "operations": copy.deepcopy(rows),
            }
            # This authored numerical choice determines which independent
            # framework reference may judge the integer rewrite. Hardware
            # facts derive the format, not the reference semantics.
            recipe["software_numerical_engine"] = spec["numerical_semantics"]["model"]["engine"]
            framework = selected.get("framework") or {}
            unsupported = set(framework) - {"activation_observer", "weight_observer", "observer_epsilon"}
            if unsupported:
                raise ValueError(f"capture adapter cannot enforce framework parameters: {sorted(unsupported)!r}")
            for role in ("activation", "weight"):
                tensor = recipe[role]
                tensor["observer"] = framework.get(
                    f"{role}_observer", _observer_name(tensor, is_weight=role == "weight")
                )
            recipe["framework_capture_policy"] = {
                "observer_epsilon": framework.get("observer_epsilon", DEFAULT_OBSERVER_EPSILON),
                "basis": "selected framework parameters or existing deterministic adapter policy; not RTL semantics",
            }
            recipe["recipe_sha256"] = digest(recipe)
            result.append(
                {
                    "format_id": selected["id"],
                    "unit": match["unit"],
                    "recipe": recipe,
                    "qualification": "diagnostic prospective transformation; no target admission",
                }
            )
    return result


def _operation_eligibility(
    spec: Mapping, declaration: dict, matches: list[dict], accounting: Mapping | None
) -> list[dict]:
    from merlin.targetgen.software_spec import admit_operation

    eligible_ids = set(declaration["eligible_operations"])
    decisions = []
    for operation in spec.get("operations") or []:
        identity, placement = operation["id"], operation["placement"]
        observations = _observations(accounting, identity)
        results = []
        for observation in observations:
            # Rescreen against only this declaration; another row must not license
            # an operation this format's eligible_operations did not select.
            selected = {**spec, "operations": [operation]}
            signature = observation.get("observed_admission_signature") or {}
            admission = admit_operation(selected, identity, signature, placement)
            hardware = observation.get("hardware_admission") or {}
            family = signature.get("family")
            units = hardware.get("units") or []
            recipes = [
                match
                for match in matches
                if match["status"] == "candidate"
                and match["unit"] in units
                and family in ((match["candidate"].get("recipe") or {}).get("families") or [])
            ]
            if identity not in eligible_ids or placement not in {"accelerator", "fused_accelerator"}:
                status, reason = "ineligible", "not an accelerator operation selected by this quantization declaration"
            elif admission["status"] == "unsupported" or hardware.get("status") == "unsupported":
                status, reason = (
                    "ineligible",
                    "observed SW signature or selected hardware admission refuses the operation",
                )
            elif observation.get("classification") in {
                "host_required",
                "unsupported",
                "structural",
                "component",
                "support_adapter",
                "nested_component",
                "support_lowering_required",
            }:
                status, reason = (
                    "ineligible",
                    "host, unsupported, nested or support-only work is not an "
                    "independent accelerator quantization site",
                )
            elif admission["status"] != "admitted" or hardware.get("status") != "admitted":
                status, reason = "unknown", "observed SW and selected hardware admission are not both established"
            elif observation.get("classification") != "accelerator_candidate":
                status, reason = "unknown", "independent accelerator operation classification is not established"
            elif not recipes:
                status, reason = "unknown", "no complete matching format-specific recipe absorbs this observed family"
            else:
                status, reason = (
                    "candidate",
                    "observed admission and selected format-specific recipe agree; "
                    "framework/compiler routes remain unevaluated",
                )
            results.append(
                {
                    "application": observation.get("application"),
                    "operation": observation.get("operation"),
                    "count": observation.get("count"),
                    "ordinals": copy.deepcopy(observation.get("ordinals")),
                    "capture_sha256": observation.get("capture_sha256"),
                    "classification": observation.get("classification"),
                    "observed_signature": copy.deepcopy(signature),
                    "software_admission": admission,
                    "hardware_admission": copy.deepcopy(hardware),
                    "status": status,
                    "reason": reason,
                    "candidate_units": [row["unit"] for row in recipes] if status == "candidate" else [],
                }
            )
        if identity not in eligible_ids or placement not in {"accelerator", "fused_accelerator"}:
            status, reason = "ineligible", "not an accelerator operation selected by this quantization declaration"
        elif not results:
            status, reason = "unknown", "no observed application signature establishes operation or layer eligibility"
        elif all(row["status"] == "candidate" for row in results):
            status, reason = (
                "candidate",
                "all recorded signatures are candidates, not unobserved signatures or model layers",
            )
        elif all(row["status"] == "ineligible" for row in results):
            status, reason = "ineligible", "all recorded signatures are refused"
        else:
            status, reason = "unknown", "recorded signatures are unresolved or have mixed eligibility"
        decisions.append(
            {
                "operation_id": identity,
                "placement": placement,
                "declared_eligible": identity in eligible_ids,
                "status": status,
                "reason": reason,
                "observed_signatures": results,
                "layer_eligibility": "not_evaluated",
            }
        )
    return decisions


def build_quantization_contract(
    software_spec: Mapping | None, hardware_snapshot: Mapping, operation_accounting: Mapping | None = None
) -> dict:
    """Build JSON-only input for inspection and future TorchAO adapter selection.

    ``hardware_snapshot`` contains selected ``quantization_candidates`` produced
    by quant_recipe.derive_candidates and their ``readout_facets``. This function
    never resolves a target provider, reruns RTL extraction, or imports TorchAO.
    """
    software_spec = software_spec or {}
    declarations = validate_quantization_declarations(software_spec)
    formats = []
    for declaration in declarations:
        semantics = copy.deepcopy(
            declaration.get("numerical_semantics", software_spec.get("numerical_semantics")) or {}
        )
        matches = _hardware_matches(declaration, hardware_snapshot, semantics)
        parameters, unknowns, _ = _parameters(declaration, None, None, semantics)
        decisions = _operation_eligibility(software_spec, declaration, matches, operation_accounting)
        framework = declaration.get("framework") or {}
        formats.append(
            {
                "id": declaration["id"],
                "declaration": declaration,
                "numerical_semantics": semantics,
                "status": "candidate" if any(row["status"] == "candidate" for row in matches) else "unknown",
                "unselected_parameters": parameters,
                "unselected_unknowns": unknowns,
                "hardware_matches": matches,
                "operation_eligibility": decisions,
                "framework_route": "not_evaluated",
                "compiler_route": "not_evaluated",
                "applied_transformation": "not_evaluated",
                "framework_parameters": {
                    key: _record(framework.get(key), "authored framework parameter; not a quantizer default")
                    for key in sorted(
                        set(framework)
                        | {
                            "activation_observer",
                            "weight_observer",
                            "observer_epsilon",
                            "calibration_dataset",
                            "calibration_count",
                        }
                    )
                },
            }
        )
    body = {
        "schema": SCHEMA,
        "target": software_spec.get("target")
        or hardware_snapshot.get("target")
        or (hardware_snapshot.get("contract") or {}).get("name"),
        "software_review": software_spec.get("status"),
        "status": "diagnostic" if declarations else "not_configured",
        "formats": formats,
        "summary": {
            "declared_formats": len(formats),
            "format_candidates": sum(row["status"] == "candidate" for row in formats),
            "operation_candidate_pairs": sum(
                decision["status"] == "candidate" for row in formats for decision in row["operation_eligibility"]
            ),
        },
        "torchao_adapter": {
            "status": "not_evaluated",
            "source_modification": False,
            "recipe_schema": "quant_recipe_v1",
            "layer_plan_schema": "quant_layer_plan_v1",
            "extension_points": ["torchao.quantization.pt2e.quantizer.Quantizer", "torchao.core.config.AOBaseConfig"],
            "existing_adapter": "merlin.targetgen._recipe_quantizer",
            "layer_planner": "merlin.targetgen.quant_layer_plan.plan",
            "requirements": [
                "complete operation signatures and explicit accelerator placement",
                "matching format-specific recipe and layer shape plan",
                "framework-version qualification and explicit observer/calibration configuration",
                "transformation receipt and compiler-route validation",
            ],
        },
        "scope": "operation candidates only; no blanket dtype support, "
        "model-layer admission, or TorchAO realization proof",
    }
    body["sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return body
