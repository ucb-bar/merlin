"""Reconstruct public archive eligibility, raw activity support and held gates.

This standalone preflight supplies no Phase 1, corpus, queue or timing owner.
"""

import json

from merlin_timing import archive as A
from merlin_timing import validation as V

from merlin.runtime import _native_files as F

from . import archive_spike_runtime as RT
from .contracts import StageGateError


def domain_reason(report, ranges):
    counts = report.get("feature_counts") or {}
    if set(counts) != set(ranges):
        return "complete raw activity vocabulary differs from the archive"
    for name, (lo, hi) in ranges.items():
        value = counts[name]
        if type(value) not in (int, float) or not lo <= value <= hi:
            return "raw activity outside public training support: " + name
    return None


def review_archive(
    *,
    runtime,
    archive,
    archive_sha256,
    binding_path,
    observations_path,
    screen_path,
    prediction_kind="empirical_screen",
    additional_archives=(),
):
    """Reopen actual original records; no saved qualified flag substitutes."""
    if type(additional_archives) is not tuple:
        raise StageGateError("additional native archives require an immutable declaration tuple")
    if additional_archives:
        if prediction_kind != "event_timing" or screen_path is not None:
            raise StageGateError("joint archives require explicit native event exploration")
        from . import archive_evidence as JOINT

        runtime.verify()
        dataset, components = JOINT.review(
            runtime,
            {
                "archive": archive,
                "archive_sha256": archive_sha256,
                "binding_path": binding_path,
                "observations_path": observations_path,
            },
            additional_archives,
        )
        result = _review_dataset(dataset, None, prediction_kind)
        result["archive_components"] = components
        return result
    runtime.verify()
    model = json.loads(runtime.model_json)
    inventory = A.inspect_archive(archive, archive_sha256)
    binding, _b = RT.metadata(binding_path)
    observations, _o = RT.metadata(observations_path)
    if prediction_kind not in {"empirical_screen", "event_timing"}:
        raise StageGateError("archive review requires an explicit supported prediction kind")
    if (screen_path is None) != (prediction_kind == "event_timing"):
        raise StageGateError("native event review cannot consume a fitted linear screen")
    screen = RT.metadata(screen_path)[0] if screen_path is not None else None
    selected_domain = dict(model["domain"], target_model_sha256=F.digest(runtime.model_path))
    if (
        binding["domain"] != selected_domain
        or binding["review"]["timing_selection"] != json.loads(runtime.timing_json)
        or inventory.get("export_scope") != "training_eligible_public"
    ):
        raise StageGateError("archive timing/runtime/source selection differs from the original reviewed public intake")
    plan = A.prepare(inventory, binding)
    dataset = A.attach(plan, observations)
    return _review_dataset(dataset, screen, prediction_kind, original_reports=observations["reports"])


def _review_dataset(dataset, screen, prediction_kind, *, original_reports=None):
    report = V.validate(dataset, screen, prediction_kind=prediction_kind)
    if (
        not report["qualified"]
        or report["event_engine_qualified"] != (prediction_kind == "event_timing")
        or dataset["profile"] != "phase1-public"
    ):
        raise StageGateError("original archive " + prediction_kind + " held gates did not qualify")
    cases = {c["case_id"]: c for c in dataset["cases"]}
    if original_reports is None:
        raw = [
            (cases[r["case_id"]], r["event_report"]["report"])
            for r in dataset["records"]
            if r["event_report"] is not None
        ]
    else:
        raw = [
            (cases[r["case_id"]], original_reports[r["record_id"]]["report"])
            for r in dataset["records"]
            if r["record_id"] in original_reports
        ]
    training = [r["feature_counts"] for case, r in raw if case["split"] == "train"]
    names = set(training[0])
    if any(set(r) != names for r in training):
        raise StageGateError("archive training raw activity vocabularies disagree")
    ranges = {n: [min(r[n] for r in training), max(r[n] for r in training)] for n in sorted(names)}
    held = {case["case_id"] for case, r in raw if case["split"] == "test" and domain_reason(r, ranges) is None}
    declared = {case["case_id"] for case in dataset["cases"] if case["split"] == "test"}
    if held != declared:
        raise StageGateError("additional raw-activity guard lacks complete original held coverage")
    return {
        "dataset": dataset,
        "screen": screen,
        "report": report,
        "raw_activity_ranges": ranges,
        "families": sorted(
            {case["mechanism_family"] for case in dataset["cases"] if case["split"] == "train"}
            & (
                {family for family, evidence in report["families"].items() if evidence["eligible_pairs"] > 0}
                if prediction_kind == "event_timing"
                else {case["mechanism_family"] for case in dataset["cases"]}
            )
        ),
        "authority": "preflight only; original ordinary policy required for installation",
    }
