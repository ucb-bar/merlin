"""Publicly evidenced provider implementation ancestry, never fit authority."""

from __future__ import annotations

import copy
import json

from .feedback import joint_performance as PERF
from .joint_scoped_revision import _denial
from .joint_structural_revision import _verify_file

SCHEMA = "merlin.provider_implementation_review.v1"
KIND = "provisional_provider_implementation"


def verify(old, new, evidence, *, previous, current):
    """Permit one provider binary correction with immutable observer and prices."""
    if (
        evidence.get("schema") != SCHEMA
        or evidence.get("status") != "provisional_untuned"
        or evidence.get("previous_runtime") != previous
        or evidence.get("runtime") != current
        or evidence.get("fit_authorized") is not False
        or evidence.get("promotion_authorized") is not False
    ):
        raise ValueError("provider correction requires exact provisional review")
    before, after = json.loads(old.model_json), json.loads(new.model_json)
    specs = [PERF._read(previous), PERF._read(current)]
    mutable = {"review_id", "libraries", "model", "model_sha256"}
    if {k: v for k, v in specs[0].items() if k not in mutable} != {
        k: v for k, v in specs[1].items() if k not in mutable
    }:
        raise ValueError("provider correction changes observer, facts or runtime scope")
    expected = copy.deepcopy(before)
    expected["libraries"] = after["libraries"]
    if expected != after:
        raise ValueError("provider correction changes prices, facts or model scope")
    for left, right in ((before["libraries"], after["libraries"]), (specs[0]["libraries"], specs[1]["libraries"])):
        if len(left) != 2 or len(right) != 2 or left[0] != right[0] or left[1] == right[1]:
            raise ValueError("provider correction requires one changed timing provider")
    for key, spec, model in [("previous_model", specs[0], before), ("model", specs[1], after)]:
        if (
            evidence[key] != {"path": spec["model"], "sha256": spec["model_sha256"]}
            or PERF._read(evidence[key]) != model
        ):
            raise ValueError("provider correction model binding differs")
    source = evidence["observer_source"]
    if (
        source["path"] != specs[1]["source_manifest"]
        or PERF._read(source).get("sha256") != after["domain"]["feature_engine_sha256"]
    ):
        raise ValueError("provider correction loses observer source binding")
    build = PERF._read(evidence["provider_build"])
    provider = build.get("provider", {})
    if (
        provider != {"path": specs[1]["libraries"][1], "sha256": after["libraries"][1]}
        or build.get("parameters_unchanged") is not True
        or build.get("calibration_promotion_authorized") is not False
        or not build.get("source_commit")
    ):
        raise ValueError("provider correction build binding differs")
    _verify_file(provider)
    leaves = build.get("sources")
    if not isinstance(leaves, list) or not 1 <= len(leaves) <= 64:
        raise ValueError("provider correction lacks bounded compiled sources")
    for pin in leaves:
        _verify_file(pin)
    _replays(evidence)
    _denial(evidence)


def _replays(evidence):
    replay = PERF._read(evidence["public_replays"])
    rows = replay.get("rows")
    if (
        replay.get("schema") != "merlin.provider_public_replays.v1"
        or replay.get("private_labels_used") is not False
        or replay.get("qualification_issued") is not False
        or not isinstance(rows, list)
        or not 1 <= len(rows) <= 32
    ):
        raise ValueError("provider correction lacks bounded public replay evidence")
    for row in rows:
        reports = [PERF._read(row[k]) for k in ("baseline_report", "report")]
        for report, model in zip(reports, (evidence["previous_model"], evidence["model"])):
            if (
                report.get("model_sha256") != model["sha256"]
                or report.get("execution_completed") is not True
                or report.get("timing_complete") is not True
                or report.get("unresolved") != []
                or report.get("functional_status") != "PASS"
                or report.get("raw_functional_exit_code") != 0
            ):
                raise ValueError("provider correction replay incomplete or unbound")
        for key in ("elf_sha256", "input_sha256", "scope_sha256", "scope_binding_sha256"):
            if not reports[0].get(key) or reports[0][key] != reports[1].get(key):
                raise ValueError("provider correction replay ELF/input/window differs")
        _verify_file(row["elf"])
        if row["elf"]["sha256"] != reports[0]["elf_sha256"]:
            raise ValueError("provider correction replay executable differs")
        for key in ("baseline_stdout", "stdout"):
            _verify_file(row[key])
        if row["baseline_stdout"]["sha256"] != row["stdout"]["sha256"]:
            raise ValueError("provider correction public numerical output differs")
        # Original public RTL label remains pinned; this does not grant accuracy.
        PERF._read(row["physical_evidence"])
