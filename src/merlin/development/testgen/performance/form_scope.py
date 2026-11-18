"""Derived Phase 0 stage: the FORM-PERF tier of the Phase-2 performance cohort.

A whole model is too slow to iterate on, and its cycles are not spread evenly: they concentrate in a
handful of op FORMS. This stage names those forms from the ITERATION workloads' own compute groups --
through the one grouping (``compute_groups`` + ``group_command``) -- prices array contractions
from the target's own facts, and records unpriced forms rather than assigning them array cycles.
The template's form-perf family then mints a model-shaped capsule at the extents of each class's
costliest group. One additional capsule may witness an uncovered joint M/K/N extent from an observed
iteration group. Both carry a vendor-reference arm spec: the vendor bar may use any instruction
role the target has (it is the bar), the candidate arm is held to the experiment's declared
``prohibited_instruction_roles``.

Two halves, split on purpose. :func:`derive_form_scope` runs at derivation, reads the captures and
writes ``scope.performance.forms`` into the byte-bound requirement. :func:`form_perf_entries` runs at
generation and reads only that frozen scope and the shared template. :func:`form_perf_coverage`
reports, per class above the template's share threshold, whether a form-perf capsule and a vendor bar
exist. When an application contains an unpriced form, every class it contributes is required:
a work proxy cannot justify pruning a performance obligation. :func:`claim_model_form_statistics`
is the owner-side, after-freeze claim-model check -- a
statistic, never a capsule.

Nothing here names a target or a model. Array geometry comes from the RTL facts
(:func:`merlin.analysis.perf.bounds.per_shape_ceiling.machine_from_facts`), the issue count from the sequencer's own loop
(:func:`merlin.analysis.perf.bounds.headroom.tile_issue_cycles`); an unpriceable form is recorded as such and
shares fall back to MACs, said so, rather than to an invented rate.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from math import prod
from pathlib import Path
from typing import Any

# ---- form_perf ---------------------------------------------------------------------------------

SCHEMA = "merlin.phase0.form_perf_scope.v1"


COVERAGE_SCHEMA = "merlin.phase0.form_perf_coverage.v1"


STATISTICS_SCHEMA = "merlin.phase0.claim_model_form_statistics.v1"


FORM_BLOCK = "form"


_EXTENT_KEYS = ("M", "K", "N", "ci", "Himg", "Wimg", "kh", "kw", "stride", "padding", "dilation")


_TEMPLATE_OWNED = frozenset({"name", "kind", "cat", "label", "source_role", "source_reference"})


def _key_text(key: Mapping[str, Any]) -> str:
    return json.dumps(key, sort_keys=True)


def class_id(key: Mapping[str, Any]) -> str:
    return hashlib.sha256(_key_text(key).encode()).hexdigest()[:12]


def device_class_key(entry: Mapping[str, Any], *, activation: str) -> dict[str, Any]:
    """A device group's form class: its model form plus where its streamed operand comes from."""
    from merlin.development.testgen.performance.model_forms import form_key

    return {"placement": "device", **form_key(entry), "activation_source": activation}


def reduction_class_key(form: Mapping[str, Any], *, operand_dtype: str) -> dict[str, Any]:
    """A host region's trailing-window reduction, as a class of its device form."""
    return {
        "placement": "host_region",
        "op": str(form["kind"]),
        "operand_dtype": str(operand_dtype),
        "input_rank": int(form["input_rank"]),
        "reduced_dims": int(form["reduced_dims"]),
    }


def class_label(key: Mapping[str, Any]) -> str:
    """A readable name for a class, for capsule names and reports; the class id stays the identity."""
    if key.get("placement") == "host_region":
        return f"{key['op']}_r{key['input_rank']}d{key['reduced_dims']}"
    stages = "_".join(key.get("epilogue") or ()) or "raw"
    geometry = key.get("geometry") or {}
    tag = (
        f"k{geometry.get('kh')}x{geometry.get('kw')}s{(geometry.get('stride') or [0])[0]}"
        if key.get("op") == "conv2d"
        else "_".join(f"{a.lower()}1" for a in geometry.get("degenerate_axes") or ())
    )
    source = "input" if key.get("activation_source") == "model_input" else ""
    return "_".join(p for p in (str(key.get("op")), tag, stages, str(key.get("scale_class") or ""), source) if p)


def gemm_extents(entry: Mapping[str, Any]) -> tuple[int, int, int]:
    """``(rows, depth, cols)`` the unit streams, reduces and holds for one device-form entry."""
    from merlin.compiler.partition.groups import group_command as GC

    if str(entry.get("op")) == "conv2d":
        positions, features = GC.device_output_shape(entry)
        return int(positions), int(entry["ci"]) * int(entry["kh"]) * int(entry["kw"]), int(features)
    if str(entry.get("op")) == "residual_add":
        return int(entry["M"]), 1, int(entry["N"])
    return int(entry["M"]), int(entry.get("K") or 1), int(entry["N"])


def predict(
    entry: Mapping[str, Any], machine, *, placement: str | None = None, batch_slices: int = 1
) -> dict[str, Any]:
    """Price every disjoint slice of a group, retaining the one-slice device form."""
    from merlin.analysis.perf.bounds.headroom import tile_issue_cycles
    from merlin.analysis.perf.bounds.per_shape_ceiling import is_unknown

    rows, depth, cols = gemm_extents(entry)
    if type(batch_slices) is not int or batch_slices < 1:
        raise ValueError("batch_slices must be a positive integer")
    out: dict[str, Any] = {
        "macs": int(rows) * int(depth) * int(cols) * batch_slices,
        "rows": rows,
        "depth": depth,
        "cols": cols,
        "batch_slices": batch_slices,
    }
    if placement == "host" or entry.get("op") not in ("matmul", "conv2d"):
        out["predicted_cycles"] = None
        out["unpriced_reason"] = (
            "host form has no selected host-cycle model"
            if placement == "host"
            else f"{entry.get('op')!s} does not execute on the modeled contraction array"
        )
        return out
    if machine is None or is_unknown(machine.array_rows) or is_unknown(machine.array_cols):
        refusals = getattr(machine, "refusals", {}) or {}
        out["predicted_cycles"] = None
        out["unpriced_reason"] = refusals.get("array_rows") or "the array geometry is not derivable from the facts"
        return out
    per_slice_cycles = tile_issue_cycles(
        rows, depth, cols, array_rows=int(machine.array_rows), array_cols=int(machine.array_cols)
    )
    out["predicted_cycles"] = per_slice_cycles * batch_slices
    out["pricing"] = "merlin.analysis.perf.bounds.headroom.tile_issue_cycles over the RTL-derived array geometry"
    return out


def application_members(target: str, module, binding, *, weight_args=None, oracle=None, machine=None) -> dict:
    """Every group of one capture that has a device form, priced, with the groups that have none counted."""
    from merlin.compiler.partition.groups import compute_groups as CG
    from merlin.compiler.partition.groups import (
        formation_compute_groups as _merlin_formation_compute_groups_1,
    )
    from merlin.compiler.partition.groups import group_command as GC
    from merlin.development.targetgen.corpus import group_entries as G

    groups = _merlin_formation_compute_groups_1.form_groups(module, target, oracle=oracle)
    members: list[dict[str, Any]] = []
    unstated: dict[str, int] = {}
    for group in groups:
        if group.placement == CG.HOST or group.root is None:
            continue
        try:
            stated = GC.program(group, weight_args=weight_args)
        except CG.NoCapsuleForm as error:
            unstated[str(error)] = unstated.get(str(error), 0) + 1
            continue
        entry = {
            k: v
            for k, v in GC.device_orientation(stated.entry, {"stored_operand": stated.stored_operand}).items()
            if k not in ("name", "scale_granularity")
        }
        key = device_class_key(entry, activation=G.activation_source(group, stated.stored_operand))
        members.append(
            {
                "group": group.index,
                "placement": group.placement,
                "key": key,
                "entry": entry,
                "batch_shape": list(stated.batch_shape),
                "batch_slices": prod(stated.batch_shape) if stated.batch_shape else 1,
            }
        )
    operand = str(binding.operand_dtype)
    for form in G.host_reduction_forms(groups):
        stated_reduction = G.reduction_entry(form, operand_dtype=operand)
        oriented = GC.device_orientation(stated_reduction, {"stored_operand": None})
        entry = {k: v for k, v in oriented.items() if k not in ("name", "kind")}
        entry.pop("scale_granularity", None)
        members.append(
            {
                "group": form["group"],
                "placement": "host",
                "key": reduction_class_key(form, operand_dtype=operand),
                "entry": entry,
                "host_region": {k: form[k] for k in ("kind", "rows", "window", "divisor", "region_dtype")},
            }
        )
    for member in members:
        member["price"] = predict(
            member["entry"], machine, placement=member["placement"], batch_slices=member.get("batch_slices", 1)
        )
    return {"members": members, "unstated": unstated, "groups": len(groups)}


MIN_TOLERANCE_GRADED_ELEMENTS = 2


SKIPPED_UNFALSIFIABLE = "skipped_unfalsifiable"


MERLIN_SUPPLIED_HOST = "merlin_supplied_host"


def _host_placed(member: Mapping[str, Any]) -> bool:
    return member.get("placement") == "host"


def _output_elements(entry: Mapping[str, Any]) -> int | None:
    """Elements the device writes for ``entry`` (``group_command.device_output_shape``), or None."""
    from merlin.compiler.partition.groups import group_command as GC

    try:
        shape = GC.device_output_shape(entry)
    except GC.NoDeviceShape:
        return None
    count = 1
    for extent in shape:
        count *= int(extent)
    return count


def _choose_representative(row: dict[str, Any], *, tolerance_graded: bool) -> None:
    """The class's costliest member; under a tolerance, only among members a grade can falsify."""
    candidates = range(len(row["members"]))
    if tolerance_graded:
        sizes = {i: _output_elements(row["members"][i]["entry"]) for i in candidates}
        candidates = [i for i, n in sizes.items() if n is not None and n >= MIN_TOLERANCE_GRADED_ELEMENTS]
        if not candidates:
            row["representative"] = None
            row["status"] = SKIPPED_UNFALSIFIABLE
            row["reason"] = (
                f"every member writes fewer than {MIN_TOLERANCE_GRADED_ELEMENTS} device-output elements "
                f"({sorted({n for n in sizes.values()}, key=lambda n: (n is None, n))}); a tolerance-graded "
                "golden with no spread cannot be falsified"
            )
            return
    row["representative"] = max(candidates, key=lambda i: (row["members"][i]["share_weight"], -i))


def aggregate(applications: Mapping[str, Mapping[str, Any]], *, tolerance_graded: bool = False) -> dict[str, Any]:
    """Classes over every application, with cycle shares only for fully priced applications.

    ``tolerance_graded`` is the target's compare policy (anything but exact integer comparison): it
    restricts the representative to members whose device output has at least two elements, and marks
    a class with none ``skipped_unfalsifiable`` instead of minting a capsule no grade can falsify.

    Host-placed members are Merlin-supplied host compute (``docs/design/merlin_supplied_host_placement.md``).
    They neither enter an application's accelerator total nor make it unpriced, and a class made of
    them is ``merlin_supplied_host``: recorded with its host work, never a Phase 2 obligation.
    """
    classes: dict[str, dict[str, Any]] = {}
    shares: dict[str, dict[str, Any]] = {}
    for label in sorted(applications):
        members = applications[label]["members"]
        accelerator = [m for m in members if not _host_placed(m)]
        priced = all(m["price"].get("predicted_cycles") is not None for m in accelerator)
        basis = "predicted_cycles" if priced else "macs"
        total = sum(float(m["price"]["predicted_cycles"] if priced else m["price"]["macs"]) for m in accelerator)
        shares[label] = {
            "basis": basis,
            "total": total,
            "qualification": (
                "RTL-derived contraction-array issue cycles"
                if priced
                else "unpriced mixed-form work proxy; not a cycle share"
            ),
            "scope": "accelerator-placed groups only; host-placed groups are Merlin-supplied and excluded",
            "merlin_supplied_host_groups": len(members) - len(accelerator),
        }
        for member in members:
            cid = class_id(member["key"])
            row = classes.setdefault(
                cid, {"class_id": cid, "label": class_label(member["key"]), "key": member["key"], "members": []}
            )
            if _host_placed(member):
                weight = 0.0
            else:
                weight = float(member["price"]["predicted_cycles"] if priced else member["price"]["macs"])
            row["members"].append(
                {
                    "application": label,
                    "group": member["group"],
                    "placement": member["placement"],
                    "entry": member["entry"],
                    "batch_shape": member.get("batch_shape", []),
                    "batch_slices": member.get("batch_slices", 1),
                    "price": member["price"],
                    "share_weight": weight,
                    **({"host_region": member["host_region"]} if "host_region" in member else {}),
                }
            )
    for row in classes.values():
        if all(_host_placed(member) for member in row["members"]):
            row.update(
                share_by_application={},
                share_basis_by_application={},
                unpriced_applications=[],
                max_share=0.0,
                occurrences=len(row["members"]),
                representative=None,
                status=MERLIN_SUPPLIED_HOST,
                reason=(
                    "every member is host-placed: Merlin's host package compiles it, so it is excluded from "
                    "the accelerator shares and is not a Phase 2 performance obligation"
                ),
                merlin_supplied_host_macs={
                    label: sum(int(m["price"]["macs"]) for m in row["members"] if m["application"] == label)
                    for label in sorted({m["application"] for m in row["members"]})
                },
            )
            continue
        per_app: dict[str, float] = {}
        for member in row["members"]:
            total = shares[member["application"]]["total"]
            per_app[member["application"]] = per_app.get(member["application"], 0.0) + (
                member["share_weight"] / total if total else 0.0
            )
        row["share_by_application"] = {k: round(v, 6) for k, v in sorted(per_app.items())}
        row["share_basis_by_application"] = {k: shares[k]["basis"] for k in sorted(per_app)}
        row["unpriced_applications"] = sorted(
            label for label in per_app if shares[label]["basis"] != "predicted_cycles"
        )
        row["max_share"] = max(row["share_by_application"].values()) if per_app else 0.0
        row["occurrences"] = len(row["members"])
        _choose_representative(row, tolerance_graded=tolerance_graded)
    ordered = sorted(classes.values(), key=lambda r: (-r["max_share"], r["class_id"]))
    return {"classes": ordered, "application_totals": shares}


def derive_form_scope(
    target: str,
    captures: Mapping[str, Path],
    binding,
    *,
    iteration_roster: Sequence[str],
    held_out: Sequence[str],
    facts: Mapping[str, Any] | None = None,
    oracle=None,
    performance_scale: Mapping[str, Path] | None = None,
    performance_scale_roster: Sequence[str] = (),
) -> dict[str, Any]:
    """``scope.performance.forms``: form classes of the ITERATION captures, priced and shared.

    ``performance_scale`` optionally adds independent captures declared ONLY as Phase 2 sources
    (``workload_spec.performance_applications``): the same forms at the extents where a target's
    on-chip capacities, tiling and movement decide the cost. They join this scope and nothing else --
    the Phase 1 model forms and source capsules still read the iteration roster alone -- so the two
    phases get deliberately different cohorts. They are refused exactly like any other source when
    they name a held-out model, and may not reuse an iteration label.
    """
    from merlin.analysis.perf.bounds.per_shape_ceiling import machine_from_facts
    from merlin.compiler.ir import mlir_query as mq
    from merlin.compiler.partition import memory_plan as stream_plan
    from merlin.development.testgen.generation.claim_boundary import is_held_out

    if set(captures) != set(iteration_roster):
        raise ValueError(
            f"form-perf classes derive from the iteration roster only: got {sorted(captures)}, "
            f"declared {sorted(iteration_roster)}"
        )
    scale = dict(performance_scale or {})
    if set(scale) != set(performance_scale_roster):
        raise ValueError(
            f"performance-scale captures must match the declared performance roster: got {sorted(scale)}, "
            f"declared {sorted(performance_scale_roster)}"
        )
    if overlap := sorted(set(scale) & set(captures)):
        raise ValueError(f"performance-scale applications {overlap} reuse iteration labels")
    sources = {**{label: Path(path) for label, path in captures.items()}, **{k: Path(v) for k, v in scale.items()}}
    if len({str(path.resolve()) for path in sources.values()}) != len(sources):
        raise ValueError("an iteration and a performance-scale application select the same capture")
    for label in sources:
        if is_held_out(label, held_out) is not None:
            raise ValueError(f"{label!r} is a held-out model; its forms may be evaluated, never derived")
    machine = machine_from_facts(target, facts=dict(facts) if facts is not None else None, measure_fill=False)
    applications: dict[str, dict[str, Any]] = {}
    for label in sorted(sources):
        path = Path(sources[label])
        manifest = path.with_name("weights.safetensors.manifest.json")
        weights = (
            stream_plan.weight_args_of(json.loads(manifest.read_text(encoding="utf-8"))) if manifest.is_file() else None
        )
        found = application_members(
            target, mq.parse(str(path)), binding, weight_args=weights, oracle=oracle, machine=machine
        )
        found["capture_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        applications[label] = found
    summary = aggregate(applications, tolerance_graded=str(getattr(binding, "compare", "")) != "exact_int")
    return {
        "schema": SCHEMA,
        "grouping": "merlin.compiler.partition.groups.formation_compute_groups.form_groups + group_command.program",
        "workload_role": "iteration_and_performance_scale" if scale else "iteration",
        "applications": {
            label: {
                "capture_sha256": app["capture_sha256"],
                "groups": app["groups"],
                "device_form_members": len(app["members"]),
                "unstated": app["unstated"],
                "workload_role": "performance_scale" if label in scale else "iteration",
                **summary["application_totals"][label],
            }
            for label, app in applications.items()
        },
        **({"performance_scale_roster": sorted(scale)} if scale else {}),
        "machine": {key: machine.to_dict()[key] for key in ("array_rows", "array_cols")},
        "classes": summary["classes"],
        "qualification": (
            "Array contractions priced from RTL-derived geometry; applications with non-array or host "
            "forms use a work proxy and require every form for coverage. No candidate or vendor arm was measured."
        ),
    }


def validate_form_scope_declaration(sweep: Mapping[str, Any], *, owner: str) -> float:
    """The template family's declared share threshold, checked; the family owes no axes."""
    declared = sweep.get("requires_form_scope")
    threshold = declared.get("min_predicted_cycle_share") if isinstance(declared, dict) else None
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not 0.0 < float(threshold) <= 1.0:
        raise ValueError(f"{owner}: requires_form_scope.min_predicted_cycle_share must be in (0, 1]")
    if "stratify_geometry" in declared and not isinstance(declared["stratify_geometry"], bool):
        raise ValueError(f"{owner}: requires_form_scope.stratify_geometry must be a boolean")
    if sweep.get("axes"):
        raise ValueError(f"{owner}: a form-scope family takes its extents from the derived forms, not from axes")
    arms = ((sweep.get("base") or {}).get("performance") or {}).get("arms") or {}
    if set(arms) != {"candidate", "vendor_reference"}:
        raise ValueError(f"{owner}: a form-scope family declares exactly a candidate and a vendor_reference arm")
    return float(threshold)


def _joint_extent_witness_index(row: Mapping[str, Any]) -> int | None:
    """Pick one observed group that best extends the representative's joint M/K/N envelope.

    The extra capsule is bounded by an actual iteration group, not a synthetic
    combination of independent axis maxima. At most one extra measurement is
    admitted per class; any remaining gaps stay visible in the coverage report.
    """
    representative = row.get("representative")
    if type(representative) is not int:
        return None
    members = row["members"]
    primary = members[representative]
    if primary.get("placement") == "host" or primary["entry"].get("op") not in ("matmul", "conv2d"):
        return None
    primary_extents = gemm_extents(primary["entry"])
    gaps = []
    for index, member in enumerate(members):
        entry = member.get("entry") or {}
        if member.get("placement") == "host" or entry.get("op") not in ("matmul", "conv2d"):
            continue
        # A tolerance-graded one-element golden has no spread. Restrict an
        # optional witness to shapes that can be falsified under either policy.
        if (_output_elements(entry) or 0) < MIN_TOLERANCE_GRADED_ELEMENTS:
            continue
        extents = gemm_extents(entry)
        if not all(need <= have for need, have in zip(extents, primary_extents)):
            gaps.append((index, member, extents))
    if not gaps:
        return None
    return max(
        gaps,
        key=lambda candidate: (
            sum(
                float(other[1]["share_weight"])
                for other in gaps
                if all(need <= have for need, have in zip(other[2], candidate[2]))
            ),
            float(candidate[1]["share_weight"]),
            -candidate[0],
        ),
    )[0]


GEOMETRY_WITNESS = "observed_geometry_stratum_witness"


def member_geometry(member: Mapping[str, Any]) -> str | None:
    """The GEMM geometry stratum of one device contraction member, or ``None`` when it has none."""
    from merlin.capture.demands import classify_geometry

    entry = member.get("entry") or {}
    if member.get("placement") == "host" or entry.get("op") not in ("matmul", "conv2d"):
        return None
    rows, depth, cols = gemm_extents(entry)
    return classify_geometry(int(rows), int(cols), int(depth))


def geometry_witness_indices(row: Mapping[str, Any], *, exclude: Sequence[int | None] = ()) -> list[tuple[str, int]]:
    """``(stratum, member index)`` for the costliest observed member of every geometry stratum the
    class's representative does not already sit in, in stratum order.

    Only falsifiable device contractions qualify; an excluded index (an already-emitted witness) and
    any member whose GEMM extents repeat an emitted one are skipped, so no paired measurement is
    minted twice for the same work.
    """
    representative = row.get("representative")
    if type(representative) is not int:
        return []
    members = row["members"]
    emitted = {gemm_extents(members[i]["entry"]) for i in (representative, *exclude) if type(i) is int}
    covered = {member_geometry(members[i]) for i in (representative, *exclude) if type(i) is int}
    best: dict[str, int] = {}
    for index, member in enumerate(members):
        stratum = member_geometry(member)
        if stratum is None or stratum in covered or index in exclude:
            continue
        if (_output_elements(member["entry"]) or 0) < MIN_TOLERANCE_GRADED_ELEMENTS:
            continue
        if gemm_extents(member["entry"]) in emitted:
            continue
        current = best.get(stratum)
        if current is None or (float(member["share_weight"]), -index) > (
            float(members[current]["share_weight"]),
            -current,
        ):
            best[stratum] = index
    return sorted(best.items())
