"""Pinned public source/archive readiness, without execution or owner issuance.

This host-only report preserves every declared member and original archive row.
It reuses original intake, source recognition and held checks. A report is data;
normal queue inference still requires its original live owners and fresh counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from xdsl.utils.exceptions import ParseError, VerifyException

from merlin.common.cpu_pool import shared_compute_pool

from . import _archive_feedback as HISTORY
from . import _archive_source_domain as DOMAIN
from . import archive_screen_review as REVIEW
from . import calibration_validation as CALIBRATION
from .contracts import StageGateError

INPUT_SCHEMA = "merlin.phase2.public_timing_readiness_inputs.v1"
SCHEMA = "merlin.phase2.public_timing_readiness.v1"
MAX_MEMBERS = 4096


def _read(pin, *, limit=64 << 20):
    if (type(pin) is not dict or not {"path", "sha256"} <= set(pin)
            or set(pin) - {"path", "sha256", "bytes"} or type(pin["path"]) is not str):
        raise ValueError("readiness needs an exact ordinary file pin")
    path = Path(pin["path"])
    if (not path.is_absolute() or path.resolve() != path or path.is_symlink() or not path.is_file()
            or path.stat().st_size > limit):
        raise ValueError("readiness input is indirect, absent or exceeds its byte ceiling")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if (any(getattr(before, key) != getattr(after, key) for key in fields)
            or hashlib.sha256(raw).hexdigest() != pin["sha256"]
            or "bytes" in pin and (type(pin["bytes"]) is not int or len(raw) != pin["bytes"])):
        raise ValueError("readiness pinned input changed")
    return raw


def _metadata(pin):
    return json.loads(_read(pin))


def _review(dataset):
    from merlin_timing import validation as V

    try:
        reviewed = REVIEW._review_dataset(dataset, None, "event_timing")
    except (ValueError, StageGateError) as error:
        return {"status": "REFUSED", "reason": str(error),
                "statistical_report": V.validate(dataset, prediction_kind="event_timing")}, None
    return {"status": "DATA_PREFLIGHT_PASS_REQUIRES_LIVE_OWNERS", "statistical_report": reviewed["report"]}, reviewed


def _component(spec):
    from merlin_timing import archive as A

    if type(spec) is not dict or set(spec) != {"archive", "binding", "observations"}:
        raise ValueError("readiness archive selection needs original archive/binding/observations")
    # Validate all wrapper pins before reopening producer artifacts. Binding and
    # report JSON can never issue a runtime, Count or timing-semantics owner.
    binding, observations = _metadata(spec["binding"]), _metadata(spec["observations"])
    _read(spec["archive"], limit=256 << 20)
    inventory = A.inspect_archive(spec["archive"]["path"], spec["archive"]["sha256"])
    plan = A.prepare(inventory, binding)
    dataset = A.attach(plan, observations)
    catalog = HISTORY.prepare(archive=Path(spec["archive"]["path"]),
        archive_sha256=spec["archive"]["sha256"], binding_path=Path(spec["binding"]["path"]), dataset=dataset)
    review, reviewed = _review(dataset)
    result = {"inputs": spec, "domain": dataset["domain"], "profile": dataset["profile"],
              "cases": dataset["cases"], "records": catalog["cases"], "review": review,
              "original_record_count": len(inventory["records"]),
              "original_raw_report_count": len(observations["reports"]),
              "eligible_dataset_records": len(dataset["records"])}
    # Reopen exact metadata again; no mutation may be hidden behind the source catalog.
    if binding != _metadata(spec["binding"]) or observations != _metadata(spec["observations"]):
        raise ValueError("readiness original archive wrappers changed")
    return result, dataset, catalog, reviewed


def _members(cohort, catalogs):
    rows = cohort.get("members")
    if (type(rows) is not list or not 0 < len(rows) <= MAX_MEMBERS
            or any(type(row) is not dict or type(row.get("identity")) is not str
                   or not row["identity"] or type(row.get("role")) is not str for row in rows)
            or len({row["identity"] for row in rows}) != len(rows)):
        raise ValueError("readiness requires every unique original member identity and role")
    result = []
    for row in rows:
        demand = row.get("source_requirements") or {}
        parsed = demand.get("parsed_operations") or {}
        source = parsed.get("source")
        item = {"identity": row["identity"], "original_role": row["role"],
                "request_sha256": row.get("request_sha256"), "complete_inputs": demand.get("complete_inputs"),
                "complete_outputs": demand.get("complete_outputs"), "scalar_products": demand.get("scalar_products"),
                "contraction_count": demand.get("contraction_count"), "timing_qualified": False,
                "measured_cycles": None, "promotion_authority": "NONE"}
        if source is None:
            item.update(status="SOURCE_UNAVAILABLE_OR_REFUSED", reason=parsed.get("reason") or demand.get("reason")
                        or "original complete typed source unavailable")
        else:
            raw = _read(source, limit=1 << 20)
            item["source"] = source
            try:
                profile = DOMAIN.typed_profile(raw.decode())
            except (ValueError, KeyError, TypeError, NotImplementedError,
                    UnicodeError, ParseError, VerifyException) as error:
                item.update(status="UNSUPPORTED_COMPLETE_SEMANTICS", reason=str(error))
            else:
                matches = [DOMAIN.train_support(catalog, profile) for catalog in catalogs]
                item.update(status="EXACT_TRAIN_SOURCE_SUPPORTED_COUNTS_PENDING" if any(matches)
                            else "EXACT_TRAIN_GEOMETRY_MISSING", source_profile=profile,
                            exact_train_groups_by_archive=matches,
                            reason="Exact complete semantics/geometry only; "
                                   "live source, native and raw activity gates pending")
            if _read(source, limit=1 << 20) != raw:
                raise ValueError("readiness source changed during profile recognition")
        result.append(item)
    return result


def analyze(inputs):
    """Reopen original public evidence; report failures without shrinking obligations."""
    from merlin_timing import archive_union as UNION

    if (type(inputs) is not dict or inputs.get("schema") != INPUT_SCHEMA
            or set(inputs) - {"schema", "cohort", "archives", "calibration_plan"}
            or type(inputs.get("archives")) is not list or not 1 <= len(inputs["archives"]) <= UNION.MAX_ARCHIVES):
        raise ValueError("readiness needs its explicit bounded input schema")
    cohort = _metadata(inputs["cohort"])
    components, datasets, catalogs, reviewed = [], [], [], []
    for spec in inputs["archives"]:
        component, dataset, catalog, review = _component(spec)
        components.append(component)
        datasets.append(dataset)
        catalogs.append(catalog)
        reviewed.append(review)
    members = _members(cohort, catalogs)
    joint = None
    if len(datasets) > 1:
        try:
            joined = UNION.join(tuple(datasets))
            joint = {"case_count": len(joined["cases"]), "record_count": len(joined["records"]),
                     "review": _review(joined)[0]}
        except ValueError as error:
            joint = {"status": "REFUSED", "reason": str(error)}
    # Primary archive is explicit, never selected after seeing residual errors.
    ranges = reviewed[0]["raw_activity_ranges"] if reviewed[0] is not None else None
    for component, dataset in zip(components, datasets, strict=True):
        observed = _metadata(component["inputs"]["observations"])["reports"]
        cases = {row["record_id"]: case["case"]["case_id"]
                 for case in component["records"] for row in case["records"]}
        eligible = {row["record_id"] for row in dataset["records"]}
        component["primary_raw_activity"] = [
            {"record_id": record_id, "case_id": cases[record_id], "eligible_dataset_record": record_id in eligible,
             "reason": "primary archive preflight unavailable" if ranges is None
                       else REVIEW.domain_reason(row["report"], ranges)}
            for record_id, row in observed.items()
        ]
    calibration = None
    if "calibration_plan" in inputs:
        plan = _metadata(inputs["calibration_plan"])
        if plan.get("schema") != CALIBRATION.SCHEMA:
            raise ValueError("readiness calibration obligations need the original versioned plan")
        calibration = {"input": inputs["calibration_plan"], "points": plan["points"],
                       "unavailable": plan["unavailable"], "status": "ORIGINAL_MEASUREMENTS_STILL_REQUIRED"}
    if _metadata(inputs["cohort"]) != cohort:
        raise ValueError("complete readiness roster changed")
    return {"schema": SCHEMA, "inputs": inputs, "members": members, "archives": components, "joint": joint,
            "calibration_validation": calibration, "declared_members": len(members), "no_member_dropped": True,
            "scope": "Pinned declared public roster only; no live owner, execution or timing qualification issued",
            "live_qualification": False, "new_measurements": 0, "new_fits": 0, "promotion_authority": "NONE"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink() or args.output.resolve() != args.output:
            raise ValueError("fresh canonical readiness output required")
        pin = {"path": str(args.inputs), "sha256": args.inputs_sha256}
        document = _metadata(pin)
        with shared_compute_pool(timeout_s=120):
            result = analyze(document)
            if _metadata(pin) != document:
                raise ValueError("readiness selection changed during analysis")
        with args.output.open("x") as stream:
            stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n")
        return 0
    except (OSError, ValueError, KeyError, TypeError, StageGateError) as error:
        print("merlin-phase2-timing-readiness: "+str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
