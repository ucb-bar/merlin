"""Bind a complete development-readiness roster to bounded original-owner work.

This operator plan is neither admission nor an execution runner. Existing sealed
capture, complete-reference, role/exclusion and realization owners remain required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

INITIAL = ("original190:166", "original190:34", "original190:35", "engineering5:0")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class _Intake:
    def __init__(self):
        self.files = {}
        self.bytes = 0

    def read(self, pin):
        path = Path(pin["path"])
        if not path.is_absolute() or path.is_symlink() or path.resolve() != path or not path.is_file():
            raise ValueError("admission plan requires canonical direct input leaves")
        if path.stat().st_size > 8 * 1024**2:
            raise ValueError("admission plan input file exceeds metadata bound")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != pin["sha256"] or ("bytes" in pin and len(raw) != pin["bytes"]):
            raise ValueError("admission plan input bytes changed")
        prior = self.files.get(str(path))
        if prior and prior != dict(path=str(path), bytes=len(raw), sha256=pin["sha256"]):
            raise ValueError("conflicting input identities")
        if not prior:
            self.bytes += len(raw)
            if self.bytes > 32 * 1024**2:
                raise ValueError("admission plan aggregate metadata bound exceeded")
            self.files[str(path)] = dict(path=str(path), bytes=len(raw), sha256=pin["sha256"])
        return raw

    def json(self, pin):
        from merlin.common import strict_json

        return strict_json.loads(self.read(pin), max_bytes=8 * 1024**2)

    def finish(self):
        for pin in tuple(self.files.values()):
            self.read(pin)


def _relative(root, name):
    p = Path(name)
    if p.is_absolute() or ".." in p.parts or p.as_posix() != name:
        raise ValueError("reference path escapes original inventory")
    return str(root / p)


def _references(intake, member, ledger):
    ref = member["complete_independent_reference_stage"]
    if isinstance(ref, dict) and ref.get("kind") == "supplement":
        index = intake.json(ref["index"])
        ordinal = int(member["identity"].rsplit(":", 1)[1])
        selected = [r for r in index["members"] if r["ordinal"] == ordinal]
        if (
            len(selected) != 1
            or index["source_manifest_sha256"] != ledger["original_cohort_metadata"]["sha256"]
            or selected[0]["request_sha256"] != member["original_request_sha256"]
            or selected[0]["status"] != "complete_exact_reference_materialized"
            or selected[0]["numeric_policy"] != member["numerical_policy"]
            or selected[0]["outputs"] != ref["outputs"]
        ):
            raise ValueError("original supplemental reference correspondence changed")
        pins = [{**r, "path": _relative(Path(ref["index"]["path"]).parent, r["path"])} for r in ref["outputs"]]
    else:
        origin = ledger["original_cohort_metadata"]
        manifest = intake.json(origin)
        ordinal = int(member["identity"].rsplit(":", 1)[1])
        rows = [r for r in manifest["members"] if r["ordinal"] == ordinal]
        if len(rows) != 1 or rows[0]["request_sha256"] != member["original_request_sha256"]:
            raise ValueError("original reference/source request correspondence changed")
        row = rows[0]
        pins = [dict(row["reference_identity"], path=_relative(Path(origin["path"]).parent, row["reference"]))]
    for pin in pins:
        intake.read(pin)
    return pins


def _assemble(intake, members, requirements, validation, history, ledger, limits):
    names = {r["original_requirement"]["requirement"] for r in requirements}
    unmapped = {r["identity"]: r for r in validation["members_unmapped_to_retained90"]}
    rows = []
    for m in members:
        identity = m["identity"]
        original = history[identity]
        if (
            m["prospective_role"] != original["prospective_setup_role"]
            or m["numerical_policy"] != ledger[identity]["numerical_policy"]
            or m["historical_V23"]["original_stages"] != ledger[identity]["original_stages_preserved"]
            or m["original_role"]
            != dict(
                counts_toward205=ledger[identity]["counts_toward_original205"],
                guard=ledger[identity]["guard_not_performance_objective"],
                value=ledger[identity]["role"],
            )
            or m["original_request_sha256"] != original["original_request_sha256"]
            or m["prospective_role_sha256_canonical_json"] != _digest(m["prospective_role"])
            or m["numerical_policy_sha256_canonical_json"] != _digest(m["numerical_policy"])
        ):
            raise ValueError("source, numerical or operator role binding changed")
        coverage = m["required_coverage_classes"]
        purpose = (unmapped.get(identity, {}).get("original_source_request") or {}).get("purposes", [])
        exact = sorted(set(purpose) & names)
        coverage_status = (
            "original_requirement_binding"
            if coverage
            else ("exact_original_declared_purpose_binding" if exact else "REFUSED_UNMAPPED_REQUIREMENT")
        )
        row = dict(
            identity=identity,
            original_role=m["original_role"],
            prospective_role=m["prospective_role"],
            request_sha256=m["original_request_sha256"],
            declaration_sha256=m["source_declaration_sha256"],
            numerical_policy=m["numerical_policy"],
            numerical_policy_sha256=m["numerical_policy_sha256_canonical_json"],
            required_work=m["required_complete_accelerator_work"],
            required_coverage=coverage,
            exact_original_purpose_matches=exact,
            coverage_status=coverage_status,
            historical_refusals=m["historical_V23"]["all_refusals_retained"],
            historical_stages=ledger[identity]["original_stages_preserved"],
            missing_pipeline=m["current_V26"],
            reference=m["complete_independent_reference_stage"],
            source_pins=m["source_pins_recorded_not_live_verified"],
            status="NOT_SELECTED_INITIAL_BATCH",
        )
        if identity in INITIAL:
            if coverage_status.startswith("REFUSED") or not row["source_pins"] or m["numerical_policy"] is None:
                row["status"] = "REFUSED_INITIAL_PLAN_PREREQUISITE"
            else:
                for source in row["source_pins"]:
                    intake.read(source)
                row["reference_pins"] = _references(intake, m, ledger[identity])
                row.update(
                    status="PLANNED_ORIGINAL_OWNERS_REQUIRED",
                    job=dict(
                        order=INITIAL.index(identity),
                        operation="original_V26_source_profile_and_live_admission_preflight",
                        capture_owner="merlin_experiments.phase0.development_sealed_batch.prepare",
                        continuation="original complete compiler/native/source-realization/writer owners only",
                        limits=limits,
                        cost_prediction="UNKNOWN",
                        depends_on=[] if identity == INITIAL[0] else [INITIAL[0] + ":complete_current_cost_and_result"],
                        authorized=False,
                    ),
                )
        rows.append(row)
    return rows


def plan(path, *, sha256, limits):
    """Reopen original custody before/after one complete, deterministic planning operation."""
    if (
        set(limits) != {"cpu_seconds", "wall_seconds", "rss_bytes", "file_bytes", "output_bytes"}
        or any(type(v) is not int or v < 1 for v in limits.values())
        or limits["file_bytes"] > 64 * 1024**2
    ):
        raise ValueError("explicit finite original limits required; actual trace ceiling cannot increase")
    intake = _Intake()
    final = intake.json(dict(path=str(Path(path)), sha256=sha256))
    for pin in final.values():
        intake.read(pin)
    handoff = intake.json(final["existing_handoff"])
    for pin in handoff.values():
        intake.read(pin)
    pins = intake.json(handoff["pins.json"])
    for pin in pins.values():
        intake.read(pin)
    members = intake.json(pins["all243-members.json"])["members"]
    requirements = intake.json(pins["all90-requirements.json"])["requirements"]
    summary = intake.json(pins["summary.json"])
    validation = intake.json(handoff["validation.json"])
    # The unchanged source inventories are the authority for membership/roles, not a supplied list.
    inputs = {k: intake.json(v) for k, v in summary["input_pins"].items()}
    history = {m["identity"]: m for m in inputs["all243"]["members"]}
    ledger = {m["identity"]: m for m in inputs["ledger"]["members"]}
    ids = [m["identity"] for m in members]
    if (
        len(ids) != 243
        or len(set(ids)) != 243
        or set(ids) != set(history)
        or set(ids) != set(ledger)
        or len(requirements) != 90
        or not set(INITIAL) <= set(ids)
    ):
        raise ValueError("complete original243/90 roster required; no member may be dropped")
    if requirements != inputs["all90"]["requirements"]:
        # The readiness overlay adds stage fields, so compare the original requirements only.
        if [r["original_requirement"] for r in requirements] != [
            r["original_requirement"] for r in inputs["all90"]["requirements"]
        ]:
            raise ValueError("original coverage requirements changed")
    for p in inputs["full16_handoff"]["compiler"]["files"].values():
        intake.read(p)
    rows = _assemble(intake, members, requirements, validation, history, ledger, dict(limits))
    intake.finish()
    return dict(
        schema="merlin.phase0.development_admission_job_plan.v1",
        readiness_sha256=sha256,
        compiler_sha256=inputs["full16_handoff"]["compiler"]["sha256"],
        members=rows,
        jobs=[
            dict(identity=identity, **row["job"])
            for identity in INITIAL
            for row in rows
            if row["identity"] == identity and "job" in row
        ],
        initial_order=list(INITIAL),
        input_pins=list(intake.files.values()),
        source_sealed=False,
        writer_admitted=False,
        jobs_executed=False,
        jobs_authorized=False,
        performance_objective_changed=False,
        formal_cohort_frozen=False,
        physical_capacity_or_timing_qualified=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--readiness-pins", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from merlin.common.cpu_pool import shared_compute_pool

    with shared_compute_pool():
        result = plan(
            args.readiness_pins,
            sha256=args.sha256,
            limits=dict(
                cpu_seconds=90, wall_seconds=300, rss_bytes=1024**3, file_bytes=64 * 1024**2, output_bytes=128 * 1024**2
            ),
        )
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")


if __name__ == "__main__":
    main()
