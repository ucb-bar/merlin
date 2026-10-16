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
(:func:`merlin.perf.derived_bound.machine_from_facts`), the issue count from the sequencer's own loop
(:func:`merlin.perf.mesh_occupancy.tile_issue_cycles`); an unpriceable form is recorded as such and
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

SCHEMA = "merlin.phase0.form_perf_scope.v1"
COVERAGE_SCHEMA = "merlin.phase0.form_perf_coverage.v1"
STATISTICS_SCHEMA = "merlin.phase0.claim_model_form_statistics.v1"
#: The block a form-perf member carries under ``performance``.
FORM_BLOCK = "form"
_EXTENT_KEYS = ("M", "K", "N", "ci", "Himg", "Wimg", "kh", "kw", "stride", "padding", "dilation")
#: Entry keys the template family owns; a derived group entry never overrides them.
_TEMPLATE_OWNED = frozenset({"name", "kind", "cat", "label", "source_role", "source_reference"})


# ------------------------------------------------------------------------------------------------
# classes and prices
# ------------------------------------------------------------------------------------------------
def _key_text(key: Mapping[str, Any]) -> str:
    return json.dumps(key, sort_keys=True)


def class_id(key: Mapping[str, Any]) -> str:
    return hashlib.sha256(_key_text(key).encode()).hexdigest()[:12]


def device_class_key(entry: Mapping[str, Any], *, activation: str) -> dict[str, Any]:
    """A device group's form class: its model form plus where its streamed operand comes from."""
    from .model_forms import form_key

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
    from merlin.xdsl_dialects.lowering import group_command as GC

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
    from merlin.perf.derived_bound import is_unknown
    from merlin.perf.mesh_occupancy import tile_issue_cycles

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
    out["pricing"] = "merlin.perf.mesh_occupancy.tile_issue_cycles over the RTL-derived array geometry"
    return out


# ------------------------------------------------------------------------------------------------
# derivation (reads the captures)
# ------------------------------------------------------------------------------------------------
def application_members(target: str, module, binding, *, weight_args=None, oracle=None, machine=None) -> dict:
    """Every group of one capture that has a device form, priced, with the groups that have none counted."""
    from merlin.targetgen import group_capsule_entries as G
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin.xdsl_dialects.lowering import group_command as GC

    groups = CG.form_groups(module, target, oracle=oracle)
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


#: A class whose every member writes fewer device-output elements than this cannot be graded under a
#: tolerance: a single element has no spread, so a constant answer can always be placed inside it.
MIN_TOLERANCE_GRADED_ELEMENTS = 2
SKIPPED_UNFALSIFIABLE = "skipped_unfalsifiable"


def _output_elements(entry: Mapping[str, Any]) -> int | None:
    """Elements the device writes for ``entry`` (``group_command.device_output_shape``), or None."""
    from merlin.xdsl_dialects.lowering import group_command as GC

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
    """
    classes: dict[str, dict[str, Any]] = {}
    shares: dict[str, dict[str, Any]] = {}
    for label in sorted(applications):
        members = applications[label]["members"]
        priced = all(m["price"].get("predicted_cycles") is not None for m in members)
        basis = "predicted_cycles" if priced else "macs"
        total = sum(float(m["price"]["predicted_cycles"] if priced else m["price"]["macs"]) for m in members)
        shares[label] = {
            "basis": basis,
            "total": total,
            "qualification": (
                "RTL-derived contraction-array issue cycles"
                if priced
                else "unpriced mixed-form work proxy; not a cycle share"
            ),
        }
        for member in members:
            cid = class_id(member["key"])
            row = classes.setdefault(
                cid, {"class_id": cid, "label": class_label(member["key"]), "key": member["key"], "members": []}
            )
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
    from merlin.common import mlir_query as mq
    from merlin.perf.derived_bound import machine_from_facts
    from merlin.xdsl_dialects.lowering import stream_plan

    from .claim_boundary import is_held_out

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
        "grouping": "merlin.xdsl_dialects.lowering.compute_groups.form_groups + group_command.program",
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


# ------------------------------------------------------------------------------------------------
# generation (reads only the frozen scope and the template)
# ------------------------------------------------------------------------------------------------
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
    from merlin.capture.shape_taxonomy import classify_geometry

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


def form_perf_entries(
    sweep: Mapping[str, Any],
    requirement: Mapping[str, Any] | None,
    requirement_sha256: str | None,
    *,
    source_window_entries: Sequence[Mapping[str, Any]] = (),
    skipped: list | None = None,
    blocked: list | None = None,
) -> list[dict[str, Any]]:
    """Observed form representatives and at most one joint-extent witness per class."""
    family = str(sweep.get("id") or "")
    validate_form_scope_declaration(sweep, owner=f"performance sweep {family}")
    scope = (((requirement or {}).get("scope") or {}).get("performance") or {}).get("forms")
    if not isinstance(scope, dict) or scope.get("schema") != SCHEMA:
        if blocked is not None:
            blocked.append(
                {
                    "family": family,
                    "sweep": family,
                    "status": "blocked_unimplemented",
                    "reason": "selected requirement lacks a derived iteration-workload form scope",
                    "requirement_sha256": requirement_sha256,
                }
            )
        return []
    classes = scope.get("classes") or []
    windows = ((requirement or {}).get("conv_geometry") or {}).get("required") or []
    if not classes and not windows:
        if skipped is not None:
            skipped.append(
                {
                    "family": family,
                    "sweep": family,
                    "status": "skipped_inapplicable",
                    "reason": "the iteration workloads form no device-form group on this target",
                    "requirement_sha256": requirement_sha256,
                }
            )
        return []
    base = copy.deepcopy(sweep.get("base") or {})
    out: list[dict[str, Any]] = []

    def append_form_entry(row: Mapping[str, Any], member: Mapping[str, Any], *, name: str, witness: bool | str) -> None:
        entry = copy.deepcopy(base)
        # The member's arithmetic comes from the derived group; its identity, category and role are
        # the template family's, so a group's own statement cannot relabel what kind of capsule it is.
        entry.update({k: copy.deepcopy(v) for k, v in member["entry"].items() if k not in _TEMPLATE_OWNED})
        if entry.get("op") == "residual_add" and "bound_lsb" in entry and "stimulus_range" not in entry:
            # A bounded integer tolerance is an arithmetic contract, not a tunable grading
            # threshold. Span the operand format so the golden can falsify that declared bound;
            # otherwise a narrow default stimulus can make a legitimate device fail after the
            # generic float-policy safeguard tightens its tolerance to zero.
            from .model_forms import element_range

            entry["stimulus_range"] = list(element_range(str(entry["operand_dtype"])))
        entry["name"] = name
        performance = entry["performance"]
        performance[FORM_BLOCK] = {
            "class_id": row["class_id"],
            "label": row["label"],
            "key": copy.deepcopy(row["key"]),
            "max_share": row["max_share"],
            "share_by_application": dict(row["share_by_application"]),
            "occurrences": row["occurrences"],
            "representative": {
                "application": member["application"],
                "group": member["group"],
                "placement": member["placement"],
                "price": dict(member["price"]),
                **({"host_region": dict(member["host_region"])} if "host_region" in member else {}),
            },
            "members": [
                {"application": m["application"], "group": m["group"], "placement": m["placement"]}
                for m in row["members"]
            ],
            "requirement_basis": {"sha256": requirement_sha256, "axis": "scope.performance.forms"},
        }
        if witness == GEOMETRY_WITNESS:
            performance[FORM_BLOCK]["mechanism"] = GEOMETRY_WITNESS
            performance[FORM_BLOCK]["geometry"] = member_geometry(member)
        elif witness:
            performance[FORM_BLOCK]["mechanism"] = "observed_joint_extent_witness"
        vendor = performance["arms"]["vendor_reference"]
        vendor["demand_equal_entry"] = {k: member["entry"][k] for k in _EXTENT_KEYS if k in member["entry"]}
        entry["source_role"] = "derived_sweep"
        entry["source_reference"] = (
            f"{sweep.get('source_reference') or 'form-perf family'}; class {row['label']} "
            f"({row['occurrences']} group(s), max per-application "
            f"{'work-proxy' if row.get('unpriced_applications') else 'predicted-cycle'} "
            f"share {row['max_share']:.4f}), "
            f"at the extents of {member['application']} group {member['group']}"
            + ("; one additional observed joint M/K/N extent witness" if witness else "")
        )
        out.append(entry)

    index = 0
    for row in classes:
        if row.get("representative") is None:
            if skipped is not None:
                skipped.append(
                    {
                        "family": family,
                        "sweep": family,
                        "class_id": row.get("class_id"),
                        "label": row.get("label"),
                        "status": row.get("status") or SKIPPED_UNFALSIFIABLE,
                        "reason": row.get("reason") or "the class has no gradeable representative",
                        "requirement_sha256": requirement_sha256,
                    }
                )
            continue
        member = row["members"][int(row["representative"])]
        append_form_entry(row, member, name=f"{family}{index:02d}_{row['label']}", witness=False)
        index += 1
    # Keep the historical primary names stable; append only one observed
    # interaction witness per class after all primary entries have been named.
    for row in classes:
        witness_index = _joint_extent_witness_index(row)
        if witness_index is None:
            continue
        member = row["members"][witness_index]
        append_form_entry(row, member, name=f"{family}J{len(out):02d}_{row['label']}", witness=True)
    # Optional, declared by the template: one observed member per further GEMM geometry stratum of a
    # class. A form class is keyed by operation and readout, so a tall convolution-as-GEMM, a wide
    # projection and a one-row product can share one class and one representative; their costs are
    # decided by different tiling and movement regimes. Each stratum's member is an observed group.
    if (sweep.get("requires_form_scope") or {}).get("stratify_geometry"):
        for row in classes:
            taken = [_joint_extent_witness_index(row)]
            for stratum, index in geometry_witness_indices(row, exclude=taken):
                member = row["members"][index]
                append_form_entry(
                    row, member, name=f"{family}G{len(out):02d}_{row['label']}_{stratum}", witness=GEOMETRY_WITNESS
                )
    # Integerization can erase a source convolution's window into im2col + matmul. The regular
    # device-form census then cannot represent its padding and stride. Reuse the independently
    # synthesized *functional* window member, rather than inventing a form from a claim model.
    primary_count = index
    window_index = 0
    by_signature: dict[str, Mapping[str, Any]] = {}
    for source in source_window_entries:
        generalization = source.get("generalization") or {}
        if (
            source.get("source_role") != "derived_sweep"
            or generalization.get("generalization_axis") != "conv_window"
            or source.get("op") != "conv2d"
        ):
            continue
        signature = str(generalization.get("conv_window") or "")
        if signature in by_signature:
            raise ValueError(f"multiple derived functional members claim source window {signature!r}")
        by_signature[signature] = source
    for window in windows:
        signature = str(window.get("signature") or "")
        source = by_signature.get(signature)
        if not signature or source is None or not _matches_source_window(source, window):
            if blocked is not None:
                blocked.append(
                    {
                        "family": family,
                        "status": "blocked_unimplemented",
                        "reason": (
                            f"independent source convolution window {signature or '<unnamed>'} "
                            "has no exact derived functional member"
                        ),
                        "requirement_sha256": requirement_sha256,
                    }
                )
            continue
        entry = copy.deepcopy(base)
        entry.update({k: copy.deepcopy(v) for k, v in source.items() if k not in _TEMPLATE_OWNED})
        entry["name"] = f"{family}W{primary_count + window_index:02d}_{class_id({'source_window': signature})}"
        entry["source_role"] = "derived_sweep"
        entry["source_reference"] = (
            f"independent iteration source window {signature}; bounded functional member "
            f"{source['name']} reused for a paired performance measurement"
        )
        entry["performance"] = copy.deepcopy(base["performance"])
        entry["performance"][FORM_BLOCK] = {
            "source_window_signature": signature,
            "source_member": source["name"],
            "requirement_basis": {"sha256": requirement_sha256, "axis": "conv_geometry.required"},
            "representative_scope": "bounded source-window mechanism; model-scale cost is a separate claim",
        }
        entry["performance"]["arms"]["vendor_reference"]["demand_equal_entry"] = {
            k: source[k] for k in _EXTENT_KEYS if k in source
        }
        out.append(entry)
        window_index += 1
    return out


def _matches_source_window(entry: Mapping[str, Any], window: Mapping[str, Any]) -> bool:
    kernel = window.get("kernel") or []
    expected = {
        "kh": kernel[0] if len(kernel) == 2 else None,
        "kw": kernel[1] if len(kernel) == 2 else None,
        "stride": window.get("stride"),
        "padding": [*(window.get("pad_before") or []), *(window.get("pad_after") or [])],
        "dilation": window.get("dilation"),
    }
    attrs = ((entry.get("operation") or {}).get("attributes") or {}) if "operation" in entry else entry
    return entry.get("op", (entry.get("operation") or {}).get("op")) == "conv2d" and all(
        expected[key] is not None and attrs.get(key) == expected[key] for key in expected
    )


# ------------------------------------------------------------------------------------------------
# coverage and the owner-side claim-model statistic
# ------------------------------------------------------------------------------------------------
def _emitted_contraction_extents(capsule: Mapping[str, Any]) -> tuple[int, int, int] | None:
    """Read an emitted capsule's operands, not its copied representative price."""
    operation = capsule.get("operation")
    if not isinstance(operation, Mapping):
        # Generation passes flat entries before writing them in focused callers.
        return gemm_extents(capsule) if capsule.get("op") in ("matmul", "conv2d") else None
    op, attrs = operation.get("op"), operation.get("attributes") or {}
    inputs = {row.get("name"): row.get("shape") for row in capsule.get("inputs") or []}
    if op == "matmul":
        lhs, weight = inputs.get(attrs.get("lhs")), inputs.get(attrs.get("weight"))
        if not (isinstance(lhs, list) and isinstance(weight, list) and len(lhs) == len(weight) == 2):
            return None
        if lhs[1] != weight[0]:
            return None
        extents = (lhs[0], lhs[1], weight[1])
    elif op == "conv2d" and attrs.get("layout") == "nhwc":
        image, weight = inputs.get(attrs.get("ifm")), inputs.get(attrs.get("weight"))
        if not (isinstance(image, list) and len(image) == 4 and isinstance(weight, list) and len(weight) == 2):
            return None
        depth = int(attrs.get("ci") or 0) * int(attrs.get("kh") or 0) * int(attrs.get("kw") or 0)
        if image[3] != attrs.get("ci") or weight[0] != depth:
            return None
        extents = gemm_extents({**attrs, "op": "conv2d", "Himg": image[1], "Wimg": image[2], "N": weight[1]})
    else:
        return None
    return extents if all(type(value) is int and value > 0 for value in extents) else None


def _geometry_strata(row: Mapping[str, Any], capsules: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Observed GEMM geometry strata of a class against those its emitted members exercise.

    Diagnostic only: a stratum without a member is reported, never required, because the template
    decides whether strata are minted (``requires_form_scope.stratify_geometry``).
    """
    from merlin.capture.shape_taxonomy import classify_geometry

    observed = sorted({stratum for stratum in map(member_geometry, row.get("members") or []) if stratum})
    represented = set()
    for capsule in capsules:
        extents = _emitted_contraction_extents(capsule)
        if extents is not None:
            rows, depth, cols = extents
            represented.add(classify_geometry(int(rows), int(cols), int(depth)))
    return {
        "observed": observed,
        "represented": sorted(represented & set(observed)),
        "unrepresented": sorted(set(observed) - represented),
    }


def _joint_extent_diagnostic(row: Mapping[str, Any], capsules: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check the *iteration* groups' joint size envelope, without consulting claim models.

    An M/K/N envelope is only a conservative stress diagnostic: it is neither
    necessary for correctness nor sufficient for model-scale performance. In
    particular, the most expensive group of a form need not bound its siblings
    on all three axes. Keep that distinction visible in the generated report.
    """
    bounds = [extents for capsule in capsules if (extents := _emitted_contraction_extents(capsule)) is not None]
    observed = []
    for member in row.get("members") or []:
        entry = member.get("entry") or {}
        if member.get("placement") == "host" or entry.get("op") not in ("matmul", "conv2d"):
            continue
        observed.append((member, gemm_extents(entry)))
    unwitnessed = [
        {"application": member.get("application"), "group": member.get("group")}
        for member, extents in observed
        if not any(all(need <= have for need, have in zip(extents, bound)) for bound in bounds)
    ]
    status = "not_applicable" if not observed else "observed_no_witness" if unwitnessed else "observed_all_bounded"
    return {
        "status": status,
        "iteration_groups": len(observed),
        "bounded_groups": len(observed) - len(unwitnessed),
        "unbounded_groups": unwitnessed,
        "qualification": (
            "iteration-workload M/K/N size envelope only; neither a correctness condition nor a "
            "model-scale performance guarantee"
        ),
    }


def form_perf_coverage(
    requirement: Mapping[str, Any] | None, capsules: Sequence[Mapping[str, Any]], *, threshold: float | None
) -> dict[str, Any]:
    """Every required form and source window needs a performance representative; ratios pending."""
    scope = (((requirement or {}).get("scope") or {}).get("performance") or {}).get("forms") or {}
    classes = scope.get("classes") or [] if isinstance(scope, dict) else []
    by_class: dict[str, list[Mapping[str, Any]]] = {}
    for capsule in capsules:
        form = ((capsule.get("performance") or {}).get(FORM_BLOCK) or {}) if isinstance(capsule, Mapping) else {}
        if form.get("class_id"):
            by_class.setdefault(str(form["class_id"]), []).append(capsule)
    rows, missing, unfalsifiable = [], [], []
    for row in classes:
        unpriced_applications = row.get("unpriced_applications") or []
        required = threshold is not None and (
            bool(unpriced_applications) or float(row.get("max_share") or 0.0) >= float(threshold)
        )
        found = by_class.get(str(row["class_id"]), [])
        bar = any(
            isinstance(((c.get("performance") or {}).get("arms") or {}).get("vendor_reference"), Mapping) for c in found
        )
        record = {
            "class_id": row["class_id"],
            "label": row.get("label"),
            "max_share": row.get("max_share"),
            "share_by_application": row.get("share_by_application"),
            "required": required,
            "required_reason": (
                "unpriced_application_requires_all_forms"
                if required and unpriced_applications
                else "predicted_cycle_share_threshold"
                if required
                else None
            ),
            "unpriced_applications": list(unpriced_applications),
            "capsules": sorted(str(c.get("name")) for c in found),
            "vendor_bar": bar,
            "ours_over_vendor": None,
            "ratio_status": "unmeasured",
            "joint_extent_diagnostic": _joint_extent_diagnostic(row, found),
            "geometry_strata": _geometry_strata(row, found),
        }
        if row.get("status") == SKIPPED_UNFALSIFIABLE:
            # A declared skip with its reason, not a silent gap: listed apart from `missing`.
            unfalsifiable.append(row["class_id"])
            record["status"] = SKIPPED_UNFALSIFIABLE
            record["reason"] = row.get("reason")
        elif required and (not found or not bar):
            missing.append(row["class_id"])
            record["status"] = "missing"
        else:
            record["status"] = "covered" if found and bar else ("not_required" if not required else "missing")
        rows.append(record)
    # Integerization may turn an independently captured convolution into im2col + int_mm. The
    # device-form grouping then sees a matmul, but a matmul-only perf capsule cannot exercise the
    # source window's padding, movement and store path. This is an iteration-source obligation,
    # never a held-out model shape or a request to manufacture a target-specific implementation.
    source_windows_without_form = []
    source_window_rows = []
    for window in ((requirement or {}).get("conv_geometry") or {}).get("required") or []:
        if not isinstance(window, Mapping):
            continue
        expected = {
            "kh": (window.get("kernel") or [None, None])[0],
            "kw": (window.get("kernel") or [None, None])[1],
            "stride": window.get("stride"),
            "padding": [*(window.get("pad_before") or []), *(window.get("pad_after") or [])],
            "dilation": window.get("dilation"),
        }
        direct = [
            capsule
            for capsule in capsules
            if ((capsule.get("performance") or {}).get(FORM_BLOCK) or {}).get("source_window_signature")
            == window.get("signature")
            and _matches_source_window(capsule, window)
        ]
        represented = any(
            isinstance(((capsule.get("performance") or {}).get("arms") or {}).get("vendor_reference"), Mapping)
            for capsule in direct
        )
        matched = list(direct)
        for row in classes:
            key = row.get("key") or {}
            geometry = key.get("geometry") or {}
            if key.get("op") != "conv2d" or any(geometry.get(k) != v for k, v in expected.items() if k != "dilation"):
                continue
            if expected["dilation"] != [1, 1] and geometry.get("dilation") != expected["dilation"]:
                continue
            found = by_class.get(str(row.get("class_id"))) or []
            matched.extend(found)
            represented = represented or any(
                isinstance(((capsule.get("performance") or {}).get("arms") or {}).get("vendor_reference"), Mapping)
                for capsule in found
            )
            if represented:
                break
        if not represented:
            source_windows_without_form.append(str(window.get("signature") or "unknown"))
        source_window_rows.append(
            {
                "signature": str(window.get("signature") or "unknown"),
                "capsules": sorted({str(capsule.get("name")) for capsule in matched}),
                "status": "covered" if represented else "missing",
                "scope": "bounded source-window mechanism; model-scale cost is a separate claim",
            }
        )
    status = (
        "no_form_scope"
        if not isinstance(scope, dict) or scope.get("schema") != SCHEMA
        else (
            "threshold_undeclared"
            if threshold is None
            else ("incomplete" if missing or source_windows_without_form else "complete")
        )
    )
    return {
        "schema": COVERAGE_SCHEMA,
        "status": status,
        "threshold": {
            "min_predicted_cycle_share": threshold,
            "basis": (
                "per-application predicted issue cycles where priced; all forms required for unpriced applications"
            ),
        },
        "classes": rows,
        "missing": missing,
        "source_windows_without_form": sorted(set(source_windows_without_form)),
        "source_windows": source_window_rows,
        "skipped_unfalsifiable": unfalsifiable,
        "claim_model_check": {
            "visibility": "owner_only_after_phase1_freeze",
            "mode": "statistics_only",
            "writes_capsules": False,
            "status": "deferred_until_phase1_freeze",
        },
    }


def attach_to_phase2_report(report: dict[str, Any], coverage: Mapping[str, Any]) -> None:
    """A required form gap must affect the cohort verdict, not just its nested diagnostic."""
    report["form_perf_coverage"] = dict(coverage)
    if coverage["threshold"]["min_predicted_cycle_share"] is None or coverage["status"] == "complete":
        return
    report["status"] = "incomplete"
    report["blockers"].append(
        {
            "component": "form_perf_coverage",
            "reason": "required independent-workload performance forms are not represented",
            "missing_classes": coverage["missing"],
            "source_windows_without_form": coverage["source_windows_without_form"],
        }
    )


def claim_model_form_statistics(
    target: str,
    capture: str | Path,
    scope: Mapping[str, Any],
    binding,
    *,
    phase1_freeze_receipt: str | Path,
    weight_args=None,
    facts: Mapping[str, Any] | None = None,
    oracle=None,
) -> dict[str, Any]:
    """Owner-side: claim cost in known forms and inside observed iteration-group extents.

    Refuses to run before the Phase-1 compiler is frozen. The selected freeze record must agree with
    the current submission and the harness-owned ``frozen`` OOT tag; an arbitrary existing file is
    not a freeze. Returns aggregate numbers only: nothing here can write, name or propose a
    capsule, so a claim model can be measured against the corpus without ever shaping it.
    """
    from merlin.common import mlir_query as mq
    from merlin.perf.derived_bound import machine_from_facts

    freeze_digest = _verified_phase1_freeze_digest(Path(phase1_freeze_receipt))
    if not isinstance(scope, Mapping) or scope.get("schema") != SCHEMA:
        raise ValueError("claim-model statistics need the derived iteration form scope")
    machine = machine_from_facts(target, facts=dict(facts) if facts is not None else None, measure_fill=False)
    found = application_members(
        target, mq.parse(str(capture)), binding, weight_args=weight_args, oracle=oracle, machine=machine
    )
    public_classes = {str(row["class_id"]): row for row in scope.get("classes") or []}
    covered = set(public_classes)
    priced = all(m["price"].get("predicted_cycles") is not None for m in found["members"])
    basis = "predicted_cycles" if priced else "macs"
    total = sum(float(m["price"][basis]) for m in found["members"]) or 0.0
    inside = sum(float(m["price"][basis]) for m in found["members"] if class_id(m["key"]) in covered)
    # A form-class match does not imply that the public iteration workloads exercised the scale of
    # this claim model. Keep this owner-only and aggregate: claim extents never become capsule inputs.
    within_extents, outside_extents = _iteration_extent_shares(found["members"], public_classes, basis)
    return {
        "schema": STATISTICS_SCHEMA,
        "visibility": "owner_only_after_phase1_freeze",
        "freeze_receipt_sha256": freeze_digest,
        "capture_sha256": hashlib.sha256(Path(capture).read_bytes()).hexdigest(),
        "basis": basis,
        "cost_qualification": (
            "RTL-derived contraction-array issue cycles"
            if priced
            else "mixed-form work proxy, not predicted cycles or measured performance"
        ),
        "device_form_groups": sum(m["placement"] != "host" for m in found["members"]),
        "host_form_groups": sum(m["placement"] == "host" for m in found["members"]),
        "covered_share": (inside / total) if total else None,
        "within_iteration_extent_share": (within_extents / total) if total else None,
        "outside_iteration_extent_share": (outside_extents / total) if total else None,
        "extent_qualification": (
            "One public group of the same form class must dominate all three per-slice GEMM extents "
            "and the disjoint batch-slice multiplicity. "
            "This is only a conservative scale diagnostic, not a performance or correctness proof."
        ),
        "uncovered_classes": sorted({class_id(m["key"]) for m in found["members"]} - covered),
        "writes_capsules": False,
    }


def _iteration_extent_shares(
    claim_members: Sequence[Mapping[str, Any]], public_classes: Mapping[str, Mapping[str, Any]], basis: str
) -> tuple[float, float]:
    """Cost in/out of the public iteration groups' observed extent envelope, without leaking shapes.

    A per-axis maximum assembled from different examples would falsely cover a combination that no
    example exercised. Require one actual member to dominate all three per-slice extents and the
    batch-slice multiplicity instead.
    """
    within = outside = 0.0
    for member in claim_members:
        row = public_classes.get(class_id(member["key"]))
        if row is None:
            continue  # Already reported by uncovered_classes / covered_share.
        extent = gemm_extents(member["entry"])
        exemplars = row.get("members") or []
        witnessed = any(
            all(observed >= required for observed, required in zip(gemm_extents(public["entry"]), extent))
            and int(public.get("batch_slices", 1)) >= int(member.get("batch_slices", 1))
            for public in exemplars
            if isinstance(public, Mapping) and isinstance(public.get("entry"), Mapping)
        )
        if witnessed:
            within += float(member["price"][basis])
        else:
            outside += float(member["price"][basis])
    return within, outside


def _verified_phase1_freeze_digest(receipt: Path) -> str:
    """Bind the owner-side claim check to one real harness freeze, not a path's existence."""
    from merlin.common import oot_repo
    from merlin.common.tree_hash import hash_tree

    if receipt.name != "freeze.json" or receipt.is_symlink() or not receipt.is_file():
        raise ValueError("the claim-model form check runs only after the Phase-1 freeze; no freeze receipt found")
    raw = receipt.read_bytes()
    try:
        record = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("the Phase-1 freeze receipt is unreadable") from exc
    if not isinstance(record, dict):
        raise ValueError("the Phase-1 freeze receipt is not a record")
    run = receipt.parent
    submission, repo = run / "submission", run / "oot"
    if submission.is_symlink() or not submission.is_dir() or repo.is_symlink() or not repo.is_dir():
        raise ValueError("the Phase-1 freeze lacks its submission or harness OOT repository")
    history = record.get("oot")
    if not isinstance(history, dict):
        raise ValueError("the Phase-1 freeze receipt has no bound frozen OOT submission")
    digest, commit = record.get("submission_sha256"), history.get("frozen_commit")
    if (
        history.get("repo") != str(repo)
        or not isinstance(digest, str)
        or len(digest) != 64
        or not isinstance(commit, str)
        or len(commit) != 40
        or history.get("package_digest") != digest
    ):
        raise ValueError("the Phase-1 freeze receipt has no bound frozen OOT submission")
    observed = hash_tree(submission)
    if observed.get("sha256") != digest or observed.get("n_files") != record.get("submission_files"):
        raise ValueError("the Phase-1 submission changed after its freeze")
    try:
        if oot_repo.resolve(repo, oot_repo.FROZEN_TAG) != commit:
            raise ValueError("the Phase-1 frozen tag differs from its freeze receipt")
        oot_repo.verify(repo, oot_repo.FROZEN_TAG, digest)
    except oot_repo.OotRepoError as exc:
        raise ValueError("the Phase-1 frozen OOT package does not match its freeze receipt") from exc
    return hashlib.sha256(raw).hexdigest()
