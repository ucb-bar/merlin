"""Compare candidate mechanism, numerical and emitted-stream evidence without compiling or grading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .candidate_artifacts import (  # noqa: F401 -- preserve established imports at this boundary
    _read,
    owned_path,
)
from .candidate_effects import (  # noqa: F401 -- preserve established imports at this boundary
    _storage_status,
    capacity_conditions,
    effects,
    traffic,
)

APPLIED, FALLBACK, UNKNOWN = "applied", "optimization_did_not_apply_emitted_fallback", "unknown"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def stream_facts(commands, structure, evidence_id, reason=None):
    """Width, buffering and overlap facts of one build's proved command stream (else explicit unknown)."""
    from merlin.kernels.mechanism_applicability import NO_STREAM, STREAM_FACTS, STREAM_SCOPE

    unknown = {name: dict(status="unknown", reason=reason or NO_STREAM, scope=STREAM_SCOPE) for name in STREAM_FACTS}
    if commands is None or structure is None:
        return unknown
    from merlin.kernels.mechanism_applicability import EXTRA_FACTS, stream_effects

    derived = stream_effects({}, structure, commands, evidence_id=evidence_id)
    extra = {name: derived[name] for name, *_ in EXTRA_FACTS if name in derived}  # present only when derived
    return {**{name: derived.get(name) or unknown[name] for name in STREAM_FACTS}, **extra}


def stream_changes(before, after):
    """Per-role width, buffering and overlap of ``after`` relative to ``before`` (factual directions)."""
    from merlin.kernels.mechanism_applicability import _stream_rows

    def record(build):
        return dict(derived_effects=dict(facts=build.get("stream_facts") or {}))

    rows = _stream_rows(
        record(before),
        record(after),
        before.get("evidence_id", "unattributed"),
        after.get("evidence_id", "unattributed"),
    )
    return [{k: v for k, v in row.items() if k not in ("evidence", "basis")} for row in rows]


def _flag(rows, reason, change="increases"):
    if not rows:
        return dict(status="unknown", reason=reason)
    hit = [
        dict(role=r.get("role"), before=r.get("reference"), after=r.get("candidate"))
        for r in rows
        if r["direction"] == "candidate_differs" and r.get("needed_change") == change
    ]
    known = [r for r in rows if r["direction"] != "unknown"]
    if not known:
        return dict(status="unknown", reason=rows[0].get("reason") or reason)
    return dict(status=bool(hit), roles=hit, unknown_roles=[r.get("role") for r in rows if r["direction"] == "unknown"])


_EXTRA_FLAGS = (
    ("issue_rate_regressed", ("host_issue_rate",), "increases"),
    ("reconfiguration_increased", ("reconfiguration", "reconfiguration_in_loop"), "reduces"),
    ("capacity_underused", ("local_capacity_use",), "increases"),
    ("bank_conflict_introduced", ("same_bank_halves",), "reduces"),
    ("contiguity_lost", ("dram_contiguity",), "increases"),
    ("epilogue_moved_off_accelerator", ("epilogue_placement",), "increases"),
)


def regressions(before, after, changes=None):
    """Facts, not advice: traffic per compute rose, overlap fell, or a role's transfers narrowed."""
    changes = stream_changes(before, after) if changes is None else changes
    ratio = [((b.get("traffic") or {}).get("ratios") or {}).get("dynamic") or {} for b in (before, after)]
    traffic_changes = {}
    for key in ("data_movement_loads_per_compute", "preloads_per_compute"):
        b, a = (r.get(key) for r in ratio)
        traffic_changes[key] = (
            dict(status="unknown", reason="dynamic loads per compute not proved on both builds")
            if None in (b, a)
            else dict(status=a > b, before=b, after=a)
        )
    statuses = [v["status"] for v in traffic_changes.values()]
    traffic_flag = dict(
        status="unknown" if "unknown" in statuses and True not in statuses else True in statuses, **traffic_changes
    )
    width = [r for r in changes if r["effect"] == "role_transfer_width"]
    overlap = [r for r in changes if r["effect"] == "issue_overlap"]
    return dict(
        traffic_regressed=traffic_flag,
        overlap_regressed=_flag(overlap, "issue_overlap unknown"),
        width_narrowed=_flag(width, "role_transfer_width unknown"),
        output_group_narrowed=_flag(
            [r for r in changes if r["effect"] == "output_group_extent"], "output_group_extent unknown"
        ),
        prefetch_coarsened=_flag(
            [r for r in changes if r["effect"] == "prefetch_interleave"],
            "prefetch_interleave unknown",
            change="reduces",
        ),
        prefetch_distance_shortened=_flag(
            [r for r in changes if r["effect"] == "prefetch_distance"], "prefetch_distance unknown"
        ),
        compute_ahead_reduced=_flag(
            [r for r in changes if r["effect"] == "prefetch_compute_ahead"], "prefetch_compute_ahead unknown"
        ),
        **{
            flag: _flag(
                [r for r in changes if r["effect"] in effect_names], effect_names[0] + " not reported", change=change
            )
            for flag, effect_names, change in _EXTRA_FLAGS
        },
        basis="after relative to the compared build; a regression flag states a fact, not a cause",
    )


def _routes(record):
    return [
        dict(family=r["family"], schema=r.get("schema"), routes=r.get("routes", []), refusals=r.get("refusals", []))
        for r in record.get("receipts", [])
        if r.get("routes") or r.get("refusals")
    ][:16]


def _applied(before, after, mechanism):
    if mechanism and mechanism in _stream_fact_names():
        return _stream_applied(before, after, mechanism)
    sb, sa = before.get("stream") or {}, after.get("stream") or {}
    if sb.get("status") != "decoded" or sa.get("status") != "decoded":
        return dict(status=UNKNOWN, reason="emitted stream not decoded on both builds")
    if sb["sha256"] == sa["sha256"]:
        return dict(
            status=FALLBACK, reason="emitted_stream_identical_to_before", evidence=dict(stream_sha256=sa["sha256"])
        )
    if not mechanism:
        return _undeclared_applied(before, after, dict(before=sb["sha256"], after=sa["sha256"]))
    fb = (before.get("effects") or {}).get(mechanism) or {}
    fa = (after.get("effects") or {}).get(mechanism) or {}
    evidence = dict(mechanism=mechanism, basis="effects", before=_brief(fb), after=_brief(fa))
    if fb.get("status") not in ("observed", "partial") or fa.get("status") not in ("observed", "partial"):
        return dict(status=UNKNOWN, reason="declared mechanism effect unknown on a build", evidence=evidence)
    if fb.get("value") == fa.get("value"):
        return dict(status=FALLBACK, reason="declared_mechanism_effect_unchanged", evidence=evidence)
    return dict(status=APPLIED, reason="emitted stream and declared mechanism effect both changed", evidence=evidence)


def _stream_applied(before, after, mechanism):
    """A proved-stream mechanism: judged on the builds' proved command streams and that stream fact.

    The static IR site list can be identical while loop structure, and so the issued stream, differs;
    identity is therefore read from the proved streams themselves."""
    pb, pa = before.get("proved_stream_sha256"), after.get("proved_stream_sha256")
    fb = (before.get("stream_facts") or {}).get(mechanism) or {}
    fa = (after.get("stream_facts") or {}).get(mechanism) or {}
    evidence = dict(
        mechanism=mechanism,
        basis="stream_facts",
        before=_brief(fb),
        after=_brief(fa),
        proved_stream_sha256=dict(before=pb, after=pa),
    )
    if before.get("emitted_sha256") and before.get("emitted_sha256") == after.get("emitted_sha256"):
        return dict(status=FALLBACK, reason="emitted_code_identical_to_before", evidence=evidence)
    if pb is None or pa is None:
        return dict(status=UNKNOWN, reason="proved command stream not established on both builds", evidence=evidence)
    if pb == pa:
        return dict(status=FALLBACK, reason="proved_command_stream_identical_to_before", evidence=evidence)
    if fb.get("status") not in ("observed", "partial") or fa.get("status") not in ("observed", "partial"):
        return dict(status=UNKNOWN, reason="declared mechanism effect unknown on a build", evidence=evidence)
    if fb.get("value") == fa.get("value"):
        return dict(status=FALLBACK, reason="declared_mechanism_effect_unchanged", evidence=evidence)
    return dict(
        status=APPLIED, reason="proved command stream and declared mechanism effect both changed", evidence=evidence
    )


_DIFFERS = ("candidate_differs", "candidate_already_exceeds")


def _undeclared_applied(before, after, emitted):
    """No declared mechanism: judged on every stream mechanism measured on both builds' proved streams.

    The measures are :func:`stream_changes` (before as reference, after as candidate), so a mechanism
    counts as changed only when its own projection differs, not when an incidental count does.
    ``applied`` needs a changed emitted stream and at least one mechanism whose measure differs; no single
    mechanism is attributed. An identical proved stream is a fallback. Otherwise ``unknown`` names each
    mechanism not measured on both builds and why."""
    pb, pa = before.get("proved_stream_sha256"), after.get("proved_stream_sha256")
    evidence = dict(
        basis="stream mechanisms measured on both builds' proved command streams",
        emitted_stream_sha256=emitted,
        proved_stream_sha256=dict(before=pb, after=pa),
    )
    if pb is not None and pb == pa:
        return dict(status=FALLBACK, reason="proved_command_stream_identical_to_before", evidence=evidence)
    rows = {}
    for row in stream_changes(before, after):
        rows.setdefault(row["effect"], []).append(row)
    changed = sorted(e for e, rs in rows.items() if any(r["direction"] in _DIFFERS for r in rs))
    unproved = {
        e: (next((r.get("reason") for r in rs if r.get("reason")), None) or "not measured")
        for e, rs in rows.items()
        if e not in changed and any(r["direction"] == "unknown" for r in rs)
    }
    unchanged = sorted(set(rows) - set(changed) - set(unproved))
    evidence.update(changed_mechanisms=changed, unchanged_mechanisms=unchanged, unmeasured_mechanisms=unproved)
    if changed:
        return dict(
            status=APPLIED,
            reason="emitted stream changed and these stream mechanisms differ on the "
            "proved streams: " + ", ".join(changed) + " (no mechanism declared; none attributed)",
            evidence=evidence,
        )
    if pb is None or pa is None:
        side = " and ".join(s for s, v in (("before", pb), ("after", pa)) if v is None)
        return dict(
            status=UNKNOWN,
            reason=f"emitted stream changed; no mechanism declared and no proved command stream on {side}",
            evidence=evidence,
        )
    return dict(
        status=UNKNOWN,
        reason="emitted stream changed; no mechanism declared and no stream mechanism "
        "measured on both builds differs" + (f"; not measured: {', '.join(sorted(unproved))}" if unproved else ""),
        evidence=evidence,
    )


def _stream_fact_names():
    from merlin.kernels.mechanism_applicability import EXTRA_FACTS, STREAM_FACTS

    return STREAM_FACTS + tuple(name for name, *_ in EXTRA_FACTS)


def _brief(fact):
    if not fact:
        return None
    value = json.dumps(fact.get("value"), sort_keys=True, default=str)
    return dict(
        status=fact.get("status"),
        reason=fact.get("reason"),
        value=fact.get("value") if len(value) <= 2048 else dict(sha256=_digest(fact.get("value"))),
    )


def verdict_rows(workspace, paths):
    """``{capsule: row}`` from owned selfcheck response documents (``per_capsule`` rows)."""
    rows = {}
    for value in paths or ():
        path, label = owned_path(workspace, value)
        raw, pin = _read(path, label)
        document = json.loads(raw)
        for row in document.get("per_capsule") or ():
            if isinstance(row, dict) and isinstance(row.get("capsule"), str):
                rows[row["capsule"]] = dict(source=pin, submission_sha256=document.get("submission_sha256"), row=row)
    return rows


def numerical(row):
    if row is None:
        return dict(status="unknown", reason="no selfcheck verdict row for this capsule")
    value = row["row"]
    executed = value.get("numerical_execution_passed")
    status = (
        "correct"
        if executed is True and value.get("numeric_consistency_passed") is not False
        else "incorrect"
        if executed is False and value.get("pass") is False
        else "unknown"
    )
    return dict(
        status=status,
        certified=value.get("pass"),
        numerical_execution_passed=executed,
        numeric_consistency_passed=value.get("numeric_consistency_passed"),
        source=value.get("capsule"),
        record=row["source"],
        submission_sha256=row["submission_sha256"],
        binding="author-supplied selfcheck response; the build-to-response binding is not proved",
    )


def declared_shape(capsules_root, capsule, record):
    """Shape conditions from the public capsule declaration, else the build's emitted tensors."""
    if capsules_root is not None:
        path = Path(capsules_root) / capsule / "capsule.yaml"
        if path.is_file() and not path.is_symlink():
            import yaml

            document = yaml.safe_load(path.read_text()) or {}
            tensors = {
                row.get("name"): dict(shape=row.get("shape"), dtype=row.get("dtype"), role=row.get("role"))
                for row in document.get("inputs") or ()
                if isinstance(row, dict)
            }
            return dict(
                basis="public_capsule_declaration",
                operation=(document.get("operation") or {}).get("op"),
                tensors=tensors,
            )
    if record.get("declared_tensors"):
        return dict(basis="emitted_command_buffer_tensors", operation=None, tensors=record["declared_tensors"])
    return dict(basis="unknown", operation=None, tensors={})


def _baseline_entry(baseline_cases, capsule):
    if capsule in baseline_cases:
        return baseline_cases[capsule]
    return next(iter(baseline_cases.values())) if len(baseline_cases) == 1 else None


def _identical(after, before, baseline):
    """``fell_back_to_seed_or_before``: the emitted kernel is byte-identical to the before or seed build."""
    digest = after.get("emitted_sha256")
    if digest is None:
        return dict(status="unknown", reason="no emitted lowered.llvm.mlir in this build")
    same = [
        name
        for name, other in (("before", before), ("seed_baseline", baseline))
        if other is not None and other.get("emitted_sha256") == digest
    ]
    compared = [
        name
        for name, other in (("before", before), ("seed_baseline", baseline))
        if other is not None and other.get("emitted_sha256")
    ]
    if not compared:
        return dict(status="unknown", reason="no before or seed build with emitted code to compare")
    return dict(
        status=bool(same),
        identical_to=same,
        compared=compared,
        emitted_sha256=digest,
        basis="identical emitted code (lowered.llvm.mlir bytes)",
    )


def sweep_table(
    sweep,
    before_cases,
    build,
    *,
    mechanism,
    verdicts,
    spaces,
    capsules_root,
    baseline_cases=None,
    verify,
    compare,
    _against,
):
    """Per-shape rows for the frozen after-compiler builds; baselines pair by capsule name."""
    rows = []
    for capsule, entry in sorted(sweep.items()):
        after = build(entry)
        shape = declared_shape(capsules_root, capsule, after)
        row = dict(
            capsule=capsule,
            build=entry[1],
            shape=shape,
            capacity=capacity_conditions(shape, spaces),
            numerically_correct=numerical(verdicts.get(capsule)),
            route_receipts=_routes(after),
            storage_valid=_storage_status(after),
        )
        if capsule in before_cases:
            pair = compare(build(before_cases[capsule]), after, mechanism=mechanism)
            row.update(
                baseline=before_cases[capsule][1],
                intervention_applied=pair["intervention_applied"],
                traffic_reduced=pair["traffic_reduced"],
                traffic_delta=pair["traffic_delta"],
            )
        else:
            row.update(
                baseline=None,
                traffic=after.get("traffic"),
                intervention_applied=dict(status=UNKNOWN, reason="no before build of this capsule supplied"),
                traffic_reduced=dict(status="unknown", reason="no before build of this capsule supplied"),
            )
        base = (baseline_cases or {}).get(capsule)
        seed = build(base) if base is not None else None
        if seed is not None:
            row["against_baseline"] = _against(seed, after, base[1])
        row["fell_back_to_seed_or_before"] = _identical(
            after, build(before_cases[capsule]) if capsule in before_cases else None, seed
        )
        refusals = [dict(family=r["family"], **f) for r in row["route_receipts"] for f in r.get("refusals", [])]
        if refusals:
            row["blockers"] = dict(legality=refusals)
            row["blocker_source"] = "compiler-emitted route refusal records; capacity from RTL facts"
        over = [t for t in row["capacity"]["tensors"] if t["exceeds"]]
        if over:
            row.setdefault("blockers", {})["capacity"] = over
        rows.append(row)
    region = {APPLIED: [], FALLBACK: [], UNKNOWN: []}
    for row in rows:
        region[row["intervention_applied"]["status"]].append(
            dict(
                capsule=row["capsule"],
                operation=row["shape"]["operation"],
                shapes={k: v["shape"] for k, v in row["shape"]["tensors"].items()},
                correct=row["numerically_correct"]["status"],
                traffic=row["traffic_reduced"]["status"],
                blockers=sorted((row.get("blockers") or {}).keys()),
            )
        )
    return dict(
        rows=rows,
        applicability_region=dict(
            applied=region[APPLIED],
            fallback=region[FALLBACK],
            unknown=region[UNKNOWN],
            counts={k: len(v) for k, v in region.items()},
            identical_emitted_code=_region_by_dimensions(rows),
            frozen_after_builds=sorted({r["build"] for r in rows}),
            extrapolation="none: only the listed capsules' emitted builds are reported",
        ),
    )


def _dimensions(row):
    """Declared tensor extents of one sweep row as named axes ``<tensor>[i]``; nothing is inferred."""
    out = {}
    for name, tensor in sorted(row["shape"]["tensors"].items()):
        for axis, extent in enumerate(tensor.get("shape") or ()):
            if type(extent) is int:
                out[f"{name}[{axis}]"] = extent
    return out


def _region_by_dimensions(rows):
    """Listed shapes split by identical emitted code, with each declared axis's observed values per side.

    Only the listed capsules are summarized; no boundary between them is inferred."""
    sides = dict(fell_back=[], emitted_new_code=[], unknown=[])
    for row in rows:
        flag = row["fell_back_to_seed_or_before"]["status"]
        side = "fell_back" if flag is True else "emitted_new_code" if flag is False else "unknown"
        sides[side].append(
            dict(
                capsule=row["capsule"],
                dimensions=_dimensions(row),
                identical_to=row["fell_back_to_seed_or_before"].get("identical_to", []),
            )
        )
    axes = {}
    for side, items in sides.items():
        for item in items:
            for axis, extent in item["dimensions"].items():
                axes.setdefault(axis, {}).setdefault(side, set()).add(extent)
    return dict(
        **sides,
        counts={k: len(v) for k, v in sides.items()},
        observed_values_by_axis={
            axis: {side: sorted(values) for side, values in by.items()} for axis, by in sorted(axes.items())
        },
        extrapolation="none: values are the listed capsules' declared extents, not a derived boundary",
    )


# Preserve established callable pickle identities through the explicit public exports.
for _export in (
    _digest,
    stream_facts,
    stream_changes,
    _flag,
    regressions,
    _routes,
    _applied,
    _stream_applied,
    _undeclared_applied,
    _stream_fact_names,
    _brief,
    verdict_rows,
    numerical,
    declared_shape,
    _baseline_entry,
    _identical,
    _dimensions,
    _region_by_dimensions,
):
    _export.__module__ = "merlin_experiments.phase1.feedback.candidate_verification"
