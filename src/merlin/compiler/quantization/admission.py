"""Operation-scoped numerical candidate admission from selected frozen inputs.

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
from merlin.target.spec.quantization import (
    _BOOL_PARAMETERS as _BOOL_PARAMETERS,
)
from merlin.target.spec.quantization import (
    _INTEGER_PARAMETERS as _INTEGER_PARAMETERS,
)
from merlin.target.spec.quantization import (
    SEMANTIC_EXTENSION_SCHEMAS as SEMANTIC_EXTENSION_SCHEMAS,
)
from merlin.target.spec.quantization import (
    _dtype as _dtype,
)
from merlin.target.spec.quantization import (
    _unknown as _unknown,
)
from merlin.target.spec.quantization import (
    semantic_extensions as semantic_extensions,
)
from merlin.target.spec.quantization import (
    validate_element_formats as validate_element_formats,
)
from merlin.target.spec.quantization import (
    validate_output_requantization as validate_output_requantization,
)
from merlin.target.spec.quantization import (
    validate_quantization_declarations as validate_quantization_declarations,
)

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
            # IDs select authored declarations; admission screens the observed
            # computation spelling, independently of its MLIR carrier or family.
            admission = admit_operation(selected, observation.get("operation"), signature, placement)
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
    software_spec: Mapping | None,
    hardware_snapshot: Mapping,
    operation_accounting: Mapping | None = None,
    *,
    preparation_profile=None,
) -> dict:
    """Build JSON-only input for inspection and future TorchAO adapter selection.

    ``hardware_snapshot`` contains selected ``quantization_candidates`` produced
    by quant_recipe.derive_candidates and their ``readout_facets``. This function
    never resolves a target provider, reruns RTL extraction, or imports TorchAO.
    """
    software_spec = software_spec or {}
    declarations = validate_quantization_declarations(software_spec)
    extensions = semantic_extensions(software_spec)
    formats = []
    for declaration in declarations:
        semantics = copy.deepcopy(
            declaration.get("numerical_semantics", software_spec.get("numerical_semantics")) or {}
        )
        matches = _hardware_matches(declaration, hardware_snapshot, semantics)
        if extensions:
            for match in matches:
                match["unknowns"] = sorted(
                    set(match["unknowns"]) | {f"{record['source_field']}.execution_adapter" for record in extensions}
                )
                if match["status"] != "incompatible":
                    match["status"] = "unknown"
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
    methodology = (software_spec.get("quantization") or {}).get("methodology")
    if extensions:
        body["semantic_extensions"] = extensions
    if preparation_profile is not None:
        from merlin.compiler.quantization import methodology as QM
        from merlin.compiler.quantization.preparation import PreparationProfile

        if not isinstance(preparation_profile, PreparationProfile):
            raise TypeError("preparation_profile must be an explicitly selected PreparationProfile")
        path = QM.output_path_of(hardware_snapshot.get("contract"))
        body["preparation_profile"] = {
            "identity": preparation_profile.identity(),
            "output_path": path,
            "status": "selected" if path is not None else "unresolved",
            "reason": None if path is not None else "the selected contract declares no derived output path",
        }
    elif methodology is not None:
        from merlin.compiler.quantization import methodology as QM

        path = QM.output_path_of(hardware_snapshot.get("contract"))
        body["methodology"] = {
            "declaration": QM.validate(methodology),
            "output_path": path,
            "status": "selected" if path is not None else "unresolved",
            "reason": None if path is not None else "the selected contract declares no derived output path",
        }
    body["sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return body


def __getattr__(name):
    """Expose the original projection callable without importing its framework owner at intake."""
    if name == "capture_recipe_candidates":
        from .torchao.recipes import capture_recipe_candidates

        return capture_recipe_candidates
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), "capture_recipe_candidates"})
