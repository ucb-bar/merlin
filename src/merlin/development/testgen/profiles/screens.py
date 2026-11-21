"""Intersect backend candidates with authored SW intent, without certifying either."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

from merlin.development.targetgen.applications.interface_observations import epilogue_carrier
from merlin.development.testgen.profiles.loading import compare_public_profiles, from_definition, load_profile
from merlin.experiments.definition import SpecError
from merlin.target.capabilities.semantic_families import from_op
from merlin.target.spec.software import admit_operation

# ---- software_screen ---------------------------------------------------------------------------


def _screen_scope_chain(spec: dict, entry: dict, capsule: dict | None) -> dict:
    """Admit an emitted region chain one operation at a time, not by container name.

    The builder emits separate movement, contraction and map regions. Adjacency
    does not establish fused placement: each region needs an accelerator SW
    declaration of its own. The pre-emission entry lacks an operation inventory.
    """
    attrs = ((capsule or {}).get("operation") or {}).get("attributes") or {}
    families, operations = attrs.get("scope_families"), attrs.get("scope_region_ops")
    if capsule is None or not isinstance(families, list) or not isinstance(operations, list):
        return {
            "status": "unknown",
            "constraints_status": "unknown",
            "decisions": [],
            "reason": "scope chain requires the emitted per-region operation inventory",
        }
    selected = entry.get("scope_families")
    if (
        len(families) < 3
        or len(families) != len(operations)
        or families != selected
        or attrs.get("scope_signature") != " -> ".join(families)
        or any(from_op(op) != family for op, family in zip(operations, families, strict=True))
    ):
        return {
            "status": "unsupported",
            "constraints_status": "refused",
            "decisions": [],
            "reason": "emitted scope regions differ from the selected operation-family chain",
        }
    operands = [row for row in capsule.get("inputs") or [] if row.get("role") in {"input", "weight"}]
    operand_dtypes = {row.get("dtype") for row in operands}
    operand_dtype = next(iter(operand_dtypes)) if len(operand_dtypes) == 1 else None
    accumulator_dtype = attrs.get("output_dtype")
    dimensions = {axis: attrs[axis] for axis in ("M", "K", "N") if type(attrs.get(axis)) is int}
    decisions = []
    for index, (op, family) in enumerate(zip(operations, families, strict=True)):
        signature = {
            "family": family,
            "operand_dtype": operand_dtype if family != "elementwise_map" else accumulator_dtype,
            "accum_dtype": accumulator_dtype,
            "rank": 2,
            "dimensions": dimensions,
        }
        decision = admit_operation(spec, op, signature, "accelerator")
        decisions.append({"role": "region", "index": index, "family": family, **decision})
    refused = [row for row in decisions if row["status"] == "unsupported"]
    unknown = [row for row in decisions if row["status"] == "unknown"]
    return {
        "status": "unsupported" if refused else "unknown" if unknown else "admitted",
        "constraints_status": "refused"
        if refused
        else "unknown"
        if any(row.get("constraints_status") != "matched" for row in decisions)
        else "matched",
        "reason": "; ".join(f"region {row['index']} ({row['op']}): {row['reason']}" for row in refused or unknown)
        or "all emitted region SW constraints match",
        "decisions": decisions,
    }


def screen_entry(
    spec: dict,
    entry: dict,
    *,
    defaults: dict | None = None,
    capsule: dict | None = None,
    host_capabilities: dict | None = None,
) -> dict:
    """Screen a selected carrier and every explicitly carried epilogue separately.

    Precision comes from the selected entry/binding or emitted capsule, not the
    backend's full format list. Missing layout/placement observations stay unknown.
    Whole-program containers need their per-operation inventory, not a model-name
    admission invented here.
    """
    defaults = defaults or {}
    if capsule is None and entry.get("component_coverage"):
        return {
            "status": "unknown",
            "constraints_status": "unknown",
            "decisions": [],
            "reason": "component obligation requires the concrete emitted per-operation program screen",
        }
    operation = (capsule or {}).get("operation") or {}
    op = operation.get("op") or entry.get("op", "unknown")
    attrs = operation.get("attributes") or {}
    kind = (capsule or {}).get("kind") or entry.get("kind")
    if kind == "model_slice" and op == "scope_chain":
        return _screen_scope_chain(spec, entry, capsule)
    performance = (capsule or {}).get("performance") or entry.get("performance") or {}
    knobs = (performance.get("emitter") or {}).get("knobs") or {}
    declared_regions = knobs.get("accelerator_regions") if knobs.get("operation") == op else None
    observed_regions = attrs.get("accelerator_contractions")
    # A registered emitter's multi-region program shape is not a single unknown
    # operation. A model_slice label or unfamiliar name alone is insufficient:
    # ordinary slices still undergo the same carrier and epilogue refusals below.
    compound_program = (
        kind == "model_slice"
        and from_op(op) is None
        and any(type(count) is int and count > 1 for count in (declared_regions, observed_regions))
    )
    if compound_program:
        from merlin.development.targetgen.corpus.model_builders import BUILDERS

        compound_program = op in BUILDERS
    if (
        kind == "model"
        or op in {"model", "micro_model"}
        # A typed tensor program states its operations in its nodes; the entry carries no per-node
        # observation, so its written program's per-operation screen decides it.
        or (capsule is None and op == "component_program")
        or entry.get("micro_model") is True
        or entry.get("materialized_capture")
        or compound_program
    ):
        return {
            "status": "unknown",
            "constraints_status": "unknown",
            "decisions": [],
            "reason": "whole-program container requires the saved per-operation admission inventory",
        }
    signature = {
        "operand_dtype": entry.get("operand_dtype") or entry.get("capture_dtype") or defaults.get("operand_dtype"),
        "accum_dtype": entry.get("accum_dtype") or defaults.get("accumulator_dtype"),
        "dimensions": {key: value for key, value in entry.items() if type(value) is int},
    }
    signature.update(entry.get("operation_signature") or {})
    # An entry's layout request is explicit; its presence is not execution proof.
    if entry.get("layout") is not None:
        signature["layout"] = entry["layout"]
    if capsule is not None:
        operands = [row for row in capsule.get("inputs") or [] if row.get("role") == "input"]
        dtypes = {row.get("dtype") for row in operands}
        if len(dtypes) == 1:
            signature["operand_dtype"] = dtypes.pop()
        if attrs.get("output_dtype"):
            signature["readout_dtype"] = attrs["output_dtype"]
    placement = entry.get("placement", "unknown")
    generalization = (capsule or {}).get("semantic") or entry.get("generalization") or {}
    host_only = generalization.get("must_accelerate") is False and generalization.get("eligible") is False
    if host_only:
        from merlin.development.targetgen.policies.host_capabilities import admit_host_operation

        # A probe built from an observed host operation names the frontend operator it reproduces;
        # declarations that name exact operators can then recognize it before it is written.
        decision = admit_host_operation(
            host_capabilities, {"mlir_operation": op, "frontend_op": entry.get("frontend_op")}, signature
        )
        if decision["status"] == "unsupported" and from_op(op) == "movement":
            # Data movement is a support lowering, not a host compute operation, and the written
            # program's own admission (program_admission.screen_written) reads it that way. Before
            # writing, the absence of a host COMPUTE declaration for it decides nothing.
            decision = {
                **decision,
                "status": "unknown",
                "reason": "data movement: decided by the written program's admission, not a host compute declaration",
            }
        return {
            **decision,
            "constraints_status": "refused" if decision["status"] == "unsupported" else "unknown",
            "scope": "explicit host-only probe; no accelerator support claim",
            "decisions": [{"role": "host", **decision}],
        }
    carrier = admit_operation(spec, op, signature, placement)
    decisions = [{"role": "carrier", **carrier}]
    stages = attrs.get("epilogue", entry.get("epilogue") or [])
    family = from_op(op)
    producer = epilogue_carrier(op, family)
    for stage in stages:
        # Pre-emission screening must project the same carrier and data type as
        # the command-buffer observation. Missing readout type stays unknown;
        # an accumulator type is not evidence of an elementwise stage's input.
        stage_dtype = (
            signature.get("operand_dtype")
            if family == "contraction"
            else attrs.get("output_dtype") or entry.get("readout_dtype") or signature.get("operand_dtype")
        )
        stage_signature = {
            **signature,
            "operand_dtype": stage_dtype,
            "epilogues": [stage],
            "composed_with": [producer] if producer is not None else None,
        }
        # An epilogue is explicitly part of the accelerator carrier, not a
        # standalone host operation with a coincidentally matching family.
        decision = admit_operation(spec, stage, stage_signature, "fused_accelerator")
        decisions.append({"role": "epilogue", "stage": stage, **decision})
    refused = [row for row in decisions if row["status"] == "unsupported"]
    unknown = [row for row in decisions if row["status"] == "unknown"]
    status = "unsupported" if refused else "unknown" if unknown else "admitted"
    return {
        "status": status,
        "constraints_status": "refused"
        if refused
        else "unknown"
        if any(row.get("constraints_status") != "matched" for row in decisions)
        else "matched",
        "reason": "; ".join(row["reason"] for row in refused or unknown) or "all selected SW constraints match",
        "decisions": decisions,
    }


def diagnostic_entry(entry: dict, decision: dict) -> dict:
    """Retain an explicitly refused probe, but outside functional/performance roles."""
    result = copy.deepcopy(entry)
    result["cat"] = "_diagnostic"
    result["software_screen"] = copy.deepcopy(decision)
    prefix = "Backend probe explicitly refused by the selected SW spec; diagnostic only. "
    source_reference = str(entry.get("source_reference") or "")
    result["source_reference"] = source_reference if source_reference.startswith(prefix) else prefix + source_reference
    return result


def intersect_requirement(requirement: dict, spec: dict, contract: dict) -> dict:
    """Remove explicit SW refusals from required device cells, never from the census.

    Backend-declared capabilities remain inspectable in diagnostic obligations.
    Unknown typed constraints remain required obligations rather than being
    silently omitted or treated as reviewed support.
    """
    from merlin.target.capabilities.eligibility import capability_map_from_contract

    result = copy.deepcopy(requirement)
    capabilities = capability_map_from_contract(contract) or {}
    defaults = spec["numerical_semantics"]
    rejected, retained = [], []
    for cell in result.get("cells") or []:
        family = cell.get("family")
        capability = capabilities.get(family)
        composed = list(getattr(capability, "composed_with", ()) or ())
        signature = {"family": family, "operand_dtype": cell.get("dtype"), "accum_dtype": defaults["accumulator_dtype"]}
        if composed:
            signature["composed_with"] = composed
        # Exact-op declarations cannot be disproved by a hypothetical family
        # label; the real synthesized carrier is screened independently below.
        exact = [op for row in spec["operations"] for op in row.get("ops", []) if from_op(op) == family]
        decisions = [
            admit_operation(spec, op, signature, "fused_accelerator" if composed else "accelerator")
            for op in exact or [str(family)]
        ]
        if all(row["status"] == "unsupported" for row in decisions):
            rejected.append({"kind": "cell", "backend_requirement": cell, "software_decisions": decisions})
        else:
            retained.append(cell)
    result["cells"] = retained
    epilogue = result.get("epilogue") or {}
    required = []
    for row in epilogue.get("required") or []:
        decision = screen_entry(spec, {"op": "matmul", "epilogue": [row["stage"]]}, defaults=defaults)
        if decision["status"] == "unsupported":
            rejected.append({"kind": "epilogue", "backend_requirement": row, "software_decision": decision})
        else:
            required.append(row)
    if "epilogue" in result:
        result["epilogue"]["required"] = required
    result["software_intersection"] = {
        "status": "diagnostic",
        "rejected_backend_capabilities": rejected,
        "qualification": "authored constraint screen; retained unknowns are not certified support",
    }
    return result


# ---- comparison_screen -------------------------------------------------------------------------
# Withdraw a comparison family whose unfused member the selected declarations refuse.
#
# A comparison group measures a fused member against the standalone parts it replaces. When one of
# those parts cannot exist on the target -- the declarations admit its operation only fused onto
# another -- the group has nothing to compare against, and writing the part produces a capsule that
# verified generation must refuse. The family is withdrawn at derivation instead, with the refusal
# recorded, rather than emitted as a group that can never be completed.
#
# A part is screened as what the group declares it to be: one operation, standalone, so it is observed
# as composed with nothing and carrying no epilogue. Only a definite refusal withdraws the family; an
# unresolved constraint is left to the written program's own screen.


def refused_part(variants: list[dict], base: dict, *, software_spec: dict | None, binding) -> dict | None:
    """The first ``part`` member every matching declaration refuses, with the decisions, or ``None``."""
    from merlin.target.capabilities.semantic_families import from_op
    from merlin.target.spec.software import admit_operation

    if not software_spec:
        return None
    for variant in variants:
        group = variant.get("comparison_group")
        role = group.get("role") if isinstance(group, dict) else None
        op = variant.get("op") or base.get("op")
        family = from_op(op) if op else None
        if role != "part" or family is None:
            continue
        signature: dict[str, Any] = {
            "family": family,
            "operand_dtype": variant.get("operand_dtype") or base.get("operand_dtype") or binding.operand_dtype,
            "accum_dtype": binding.accum_dtype,
            "composed_with": [],
            "epilogues": [],
        }
        if family == "elementwise_map":
            # The unfused part of an epilogue fusion runs in the accumulator domain it would be fused in.
            signature["operand_dtype"] = binding.accum_dtype
        decisions = []
        for declaration in software_spec.get("operations") or []:
            decision = admit_operation(
                {**software_spec, "operations": [declaration]}, str(op), signature, declaration["placement"]
            )
            if "declaration" in decision:
                decisions.append(decision)
        if decisions and all(decision["status"] == "unsupported" for decision in decisions):
            return {
                "member": op,
                "family": family,
                "reason": "; ".join(sorted({decision["reason"] for decision in decisions})),
                "decisions": decisions,
            }
    return None


def refused_emitted_part(entries: list[dict], *, binding, evidence) -> dict | None:
    """Definite refusal of any comparison-group member from its builder's typed interface.

    Every member of a comparison group is needed for the comparison -- a fused member and its parts,
    an island member and its matched no-island member -- so a member the selected declarations
    definitely refuse withdraws the family, whatever its role. This is a pre-write SW/host admission
    screen, not a numerical or execution proof. Unknown admissions remain for the written capsule's
    full screen.
    """
    from merlin.development.targetgen.corpus import binding as CS
    from merlin.development.targetgen.corpus import model_builders as _merlin_model_builders_1
    from merlin.development.testgen.evidence.program_admission import account_interface_text, summarize
    from merlin.target.capabilities.semantic_families import from_op

    for entry in entries:
        group = entry.get("comparison_group")
        if not isinstance(group, dict) or not group.get("role"):
            continue
        if entry.get("source") not in (None, "direct"):
            continue  # A non-builder source has no interface to screen here.
        op = entry.get("op")
        if op not in _merlin_model_builders_1.BUILDERS:
            continue
        try:
            _, selected_binding = CS.entry_binding(entry, binding)
            capsule, mlir = _merlin_model_builders_1.build(entry, selected_binding)
            if "linalg_mlir" in capsule:
                # A builder that emits a source program is screened the way its written capsule
                # will be: per operation, as for a captured application.
                observed = _account_source_program(mlir, str(entry.get("name") or op), binding, evidence)
            else:
                observed = account_interface_text(mlir, target=binding.target, evidence=evidence)
        except (KeyError, TypeError, ValueError):
            # The ordinary writer reports a malformed or unavailable builder;
            # it must not become an inapplicability skip at this early screen.
            continue
        semantic = capsule.get("semantic") or {}
        host_only = semantic.get("must_accelerate") is False and semantic.get("eligible") is False
        decision = summarize(observed, host_only=host_only, scope="builder-emitted typed comparison part")
        if decision["status"] == "unsupported":
            return {
                "member": op,
                "name": entry.get("name"),
                "family": from_op(op),
                "reason": decision["reason"],
                "decisions": decision["decisions"],
                "interface_sha256": hashlib.sha256(mlir.encode()).hexdigest(),
                "basis": "builder_emitted_typed_interface",
            }
    return None


def _account_source_program(mlir: str, name: str, binding, evidence) -> list[dict]:
    import tempfile
    from pathlib import Path

    from merlin.development.testgen.evidence.program_admission import account_program

    with tempfile.TemporaryDirectory(prefix="merlin-comparison-part-") as scratch:
        program = Path(scratch) / "capsule.linalg.mlir"
        program.write_text(mlir, encoding="utf-8")
        return account_program(program, name=name, target=binding.target, evidence=evidence)


# ---- comparison --------------------------------------------------------------------------------
# Public cross-target comparison from explicitly selected experiment definitions.


def build(definitions: list[Path]) -> dict:
    declarations = [from_definition(path) for path in definitions]
    targets = [item.target for item in declarations]
    if len(targets) < 2 or len(set(targets)) != len(targets):
        raise SpecError("comparison requires at least two definitions with distinct targets")
    profiles = []
    for item in declarations:
        try:
            profile = load_profile(item.profile, include_holdouts=False, **item.profile_inputs())
        except ValueError as exc:
            raise SpecError(f"invalid public comparison input: {exc}") from exc
        profiles.append((item.target, profile))
    result = compare_public_profiles(profiles)
    result["scope"] = "current-declared-public-recipes"
    result["definitions"] = [str(item.definition) for item in declarations]
    return result
