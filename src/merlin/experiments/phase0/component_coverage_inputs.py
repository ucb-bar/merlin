"""Bounded deterministic independent input construction through normal builders."""

from __future__ import annotations

import copy
import itertools
import json

from .component_coverage import BUDGETED_REPORT_SCHEMA, REPORT_SCHEMA, CoverageState
from .component_generation import digest


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def covering_points(axes, interactions, *, max_members, max_cells):
    """Enumerate declared interactions; cover all remaining pairs deterministically.

    Greedy extension operates on missing projections, so it does not enumerate
    the full Cartesian domain. Budget exhaustion reports exact missing cells.
    """
    names = sorted(axes)
    domains = [tuple(dict.fromkeys(_json(value) for value in axes[name])) for name in names]
    if any(not values for values in domains):
        return [], {"status": "incomplete", "reason": "axis has no available values", "missing_cells": None}
    groups = {tuple([index]) for index in range(len(names))}
    groups |= set(itertools.combinations(range(len(names)), 2))
    groups |= {tuple(sorted(names.index(name) for name in group)) for group in interactions}
    required_count = sum(_product(len(domains[index]) for index in group) for group in groups)
    if required_count > max_cells:
        return [], {
            "status": "incomplete",
            "reason": "declared interaction cells exceed selected budget",
            "required_cells": required_count,
            "missing_cells": required_count,
        }
    missing = {
        (group, values)
        for group in groups
        for values in itertools.product(*(range(len(domains[index])) for index in group))
    }
    if not names:
        return [{}], {"status": "complete", "required_cells": 0, "missing_cells": 0}
    points = []
    while missing and len(points) < max_members:
        group, values = min(missing)
        selected = dict(zip(group, values, strict=True))
        for index in range(len(names)):
            if index in selected:
                continue

            def score(value):
                trial = {**selected, index: value}
                return sum(all(i in trial for i in g) and tuple(trial[i] for i in g) == v for g, v in missing)

            selected[index] = max(range(len(domains[index])), key=lambda value: (score(value), -value))
        point = tuple(selected[index] for index in range(len(names)))
        missing -= {(g, tuple(point[index] for index in g)) for g in groups}
        points.append({name: json.loads(domains[index][point[index]]) for index, name in enumerate(names)})
    return points, {
        "status": "complete" if not missing else "incomplete",
        "required_cells": required_count,
        "missing_cells": len(missing),
        "members": len(points),
        "algorithm": "deterministic_projection_extension.v1",
        "reason": "selected member budget exhausted" if missing else "all declared interactions and pairs covered",
    }


def _product(values):
    result = 1
    for value in values:
        result *= value
    return result


def _substitute(value, point):
    if isinstance(value, dict):
        if set(value) == {"axis"}:
            if value["axis"] not in point:
                raise ValueError("component base refers to an undeclared axis")
            return copy.deepcopy(point[value["axis"]])
        return {key: _substitute(child, point) for key, child in value.items()}
    if isinstance(value, list):
        return [_substitute(child, point) for child in value]
    return value


def expand(plan, *, binding, evidence):
    """Produce ordinary independent entries and a private immutable-input report."""
    from merlin.targetgen import corpus_spec

    from .sweeps import resolve_extent

    document = plan.to_dict()
    effect_owners = {row["id"]: row for row in document["effects"]}
    entries, obligations = [], []
    used = 0
    for obligation in plan.obligations:
        row = obligation.to_dict()
        domains, boundaries, errors = {}, {}, []
        for name, axis in row["axes"].items():
            if axis["kind"] == "resource":
                continue
            domains[name] = [
                resolve_extent(value, binding.tile_dim) if axis["kind"] == "extent" else value
                for value in axis["values"]
            ]
        for name, axis in row["axes"].items():
            if axis["kind"] != "resource":
                continue
            from .resource_boundaries import BoundaryUnavailable, derive

            declaration = axis["declaration"]
            try:
                if declaration.get("derive") != "resident_allocation_boundary":
                    from .resource_frontiers import derive as derive
                values, _, record = derive(
                    declaration,
                    owner=obligation.id,
                    axis=name,
                    target=binding.target,
                    tile=binding.tile_dim,
                    dtype=binding.operand_dtype,
                    fixed=domains,
                    evidence=evidence,
                    resolve_extent=resolve_extent,
                )
                domains[name], boundaries[name] = values, record
                if record.get("missing_scenarios"):
                    errors.append("declared resource scenarios unavailable: " + ", ".join(record["missing_scenarios"]))
            except BoundaryUnavailable as exc:
                domains[name], boundaries[name] = [], exc.record
                errors.append(str(exc))
        representations = 2 if "graph_family" in row["base"] else 1
        available = document["budget"]["max_members"] - used
        points, coverage = covering_points(
            domains,
            row["interactions"],
            max_members=max(available // representations, 0),
            max_cells=document["budget"]["max_interaction_cells"],
        )
        if available < representations:
            points, coverage = (
                [],
                {"status": "incomplete", "reason": "global member budget exhausted", "missing_cells": None},
            )
        if coverage["status"] != "complete":
            errors.append(coverage["reason"])
        if representations != 1:
            coverage.update(points=len(points), members=len(points) * representations)
        members = []
        for ordinal, point in enumerate(points):
            entry = _substitute(row["base"], point)
            for name, axis in row["axes"].items():
                if axis["kind"] in {"extent", "resource"} and name not in entry:
                    entry[name] = point[name]
            if "graph_family" in entry:
                from .component_graph_variants import expand_entry

                variants = expand_entry(entry, binding=binding, policy=document.get("execution_budget"))
            else:
                variants = [entry]
            for entry in variants:
                stem = obligation.id
                if stem.isascii() and all(c.isalnum() or c == "_" for c in stem):
                    name = "CP_" + stem + "_" + str(ordinal).zfill(4)
                else:
                    # Keep reviewed ids lossless in a distinct capsule namespace;
                    # the normal capsule ABI admits only ASCII letters/digits/_ .
                    name = "CPhex_" + stem.encode().hex() + "_" + str(ordinal).zfill(4)
                variant = entry.pop("graph_variant", None)
                if variant is not None:
                    name += "_" + variant["representation"]
                cohort = obligation.cohort
                entry.update(
                    name=name,
                    kind=entry.get("kind", "isa"),
                    cat="_perf"
                    if cohort == "development"
                    else "hidden"
                    if cohort == "withheld_transfer"
                    else entry.get("kind", "isa"),
                    label="dev" if cohort == "development" else "hidden" if cohort == "withheld_transfer" else "public",
                    source_role="derived_sweep",
                    source_reference="independent reviewed component coverage:" + plan.source_sha256,
                )
                if row["frontend"] == "pytorch":
                    from merlin.targetgen.capsule_source import supported_ops

                    if entry["op"] not in supported_ops():
                        errors.append("normal PyTorch generator has no selected operation builder")
                        continue
                    entry["source"] = "pytorch"
                elif entry["op"] not in corpus_spec.BUILDERS:
                    errors.append("normal MLIR generator has no selected operation builder")
                    continue
                if entry.get("_component_source_unavailable"):
                    errors.append(entry["_component_source_unavailable"])
                    observed_effects = set()
                else:
                    observed_effects = generated_effects(entry, binding=binding)
                missing_effects = [
                    effect_owners[identity]["kind"]
                    for identity in row["effects"]
                    if effect_owners[identity]["kind"] not in observed_effects
                ]
                from .component_input_witnesses import NUMERIC_INPUT_EFFECTS
                from .component_source_witnesses import SOURCE_NUMERIC_EFFECTS

                if entry.get("input_palette") is not None:
                    missing_effects = [kind for kind in missing_effects if kind not in NUMERIC_INPUT_EFFECTS]
                if entry["op"] == "producer_quantizer_observer" and entry.get("source") == "pytorch":
                    missing_effects = [kind for kind in missing_effects if kind not in SOURCE_NUMERIC_EFFECTS]
                if missing_effects:
                    errors.append("no concrete generated witness for declared effects: " + ", ".join(missing_effects))
                observed_effects = sorted(observed_effects)
                stamp = {
                    "plan_sha256": plan.source_sha256,
                    "obligation": obligation.id,
                    "cohort": cohort,
                    "expectation": row["expectation"],
                    "point_sha256": digest(point),
                    "operation_owners": row["operations"],
                    "effect_owners": row["effects"],
                    "resource_boundaries": copy.deepcopy(boundaries),
                    "generated_effects": observed_effects,
                }
                if variant is not None:
                    stamp["graph_variant"] = variant
                entry["component_coverage"] = stamp
                member = {"name": name, "requested_member": entry["cat"] + "/" + name, "point_sha256": digest(point)}
                if variant is not None:
                    member["graph_variant"] = copy.deepcopy(variant)
                members.append(member)
                entries.append(entry)
        used += len(points) * representations
        obligations.append(
            {
                "id": obligation.id,
                "mandatory": obligation.mandatory,
                "cohort": obligation.cohort,
                "expectation": row["expectation"],
                "declaration_sha256": digest(row),
                "coverage": coverage,
                "resource_boundaries": boundaries,
                "errors": list(dict.fromkeys(errors)),
                "state": CoverageState.UNAVAILABLE.value,
                "members": members,
            }
        )
    report = {
        "schema": BUDGETED_REPORT_SCHEMA if "execution_budget" in document else REPORT_SCHEMA,
        "status": "incomplete",
        "plan": {"path": plan.source_path, "sha256": plan.source_sha256},
        "declaration": document,
        "hardware": document["hardware"],
        "software_spec_sha256": document["software_spec_sha256"],
        "numerical_semantics_sha256": document["numerical_semantics_sha256"],
        "obligations": obligations,
        "qualification": "finite generated input coverage; candidate semantics, execution and timing unverified",
    }
    return entries, report


def generated_effects(entry, *, binding):
    """Only concrete generated source structures discharge effect obligations.

    These are source semantics. Target aliasing, invocation epochs, placement,
    lifetime and synchronization still require the Phase 1 execution owner.
    """
    op = entry["op"]
    if op == "component_program":
        from merlin.targetgen.component_program import analyze

        program = analyze(
            entry.get("program"),
            operand_dtype=binding.mlir_dtype(binding.operand_dtype),
            accumulator_dtype=binding.mlir_dtype(binding.accum_dtype),
        )
        return set(program["effects"])
    effects = {"output_publication"}
    if entry.get("source") == "pytorch":
        from merlin.targetgen.component_sources import source_effects

        effects |= source_effects(entry)
    elif op == "conv2d":
        effects.add("convolution_source_window")
    if op == "resident_reuse" and len(entry.get("matmuls") or []) > 1:
        effects |= {"immutable_reuse", "multiple_consumers", "shared_producer"}
    return effects
