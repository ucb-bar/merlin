"""Explicit live target checks of both completed diagnostic compiler arms.

This joins original retained jobs, never creates a target capability or restores
one from a saved report. Reports remain private; feedback carries only an issued
digest. Numerical, instruction, completion and timing authority stay with their
original owners.
"""

from __future__ import annotations

import inspect
import json
import math
import time
import weakref
from pathlib import Path
from types import MethodType

from merlin.common.cpu_pool import shared_compute_pool
from merlin.targetgen.compiler_pass_selection import CompilerPassSelection

from .contracts import StageGateError, canonical_json, document_sha256, sha256_file, write_json

SCHEMA = "merlin.diagnostic_source_pair_selection.v1"
PROTOCOL = "merlin.diagnostic_source_pair.v1"
SCOPE = "original live diagnostic relation only; no timing, target capability or promotion authority"
_ISSUED = weakref.WeakKeyDictionary()


def _pin(path):
    path = Path(path)
    if (
        not path.is_absolute()
        or path.is_symlink()
        or path.resolve(strict=True) != path
        or not path.is_file()
        or path.stat().st_size > 8 << 20
    ):
        raise StageGateError("diagnostic pair check requires original regular source/product bytes")
    return {"path": str(path), "sha256": sha256_file(path)}


def _closed(value):
    try:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode()
        if len(encoded) > 1 << 20:
            raise ValueError("report exceeds one MiB")
        return json.loads(encoded)
    except (TypeError, ValueError, OverflowError) as error:
        raise StageGateError("diagnostic pair check returned an invalid bounded report") from error


class DiagnosticSourcePairCheck:
    """Selected only by an original live preparation and source registry."""

    def __init__(self, passes, source_passes):
        if (
            type(passes) is not CompilerPassSelection
            or type(source_passes) is not CompilerPassSelection
            or len(passes.mechanisms) != 1
            or source_passes.mechanisms != ("source_execution_diagnostic",)
            or passes.registry is not source_passes.registry
            or passes.binding != source_passes.binding
            or passes.factory is None
            or source_passes.factory is None
            or passes.factory.record() != source_passes.factory.record()
        ):
            raise StageGateError("diagnostic pair check requires the same original source/preparation registry")
        self._passes, self._source_passes = passes, source_passes
        cap = passes.resolve(passes.mechanisms[0])
        if cap is None:
            raise StageGateError("diagnostic pair check lacks its original live preparation capability")
        callback = getattr(cap.owner, "check_source_pair", None)
        if type(callback) is not MethodType or callback.__self__ is not cap.owner:
            raise StageGateError("diagnostic pair check requires an original bound target checker")
        source = inspect.getsourcefile(callback.__func__)
        if source is None:
            raise StageGateError("diagnostic pair checker has no original source")
        self._owner, self._function, self._code = cap.owner, callback.__func__, callback.__func__.__code__
        self._pin = _pin(Path(source).resolve())
        self._record = dict(
            schema=SCHEMA,
            passes=passes.record(),
            source_passes=source_passes.record(),
            callback={**self._pin, "symbol": self._function.__qualname__},
            scope=SCOPE,
        )
        _ISSUED[self] = (canonical_json(self._record), {})
        try:
            self.verify_for(source_passes)
        except BaseException:
            del _ISSUED[self]
            raise

    def verify_for(self, source_passes):
        if type(self) is not DiagnosticSourcePairCheck or self not in _ISSUED:
            raise StageGateError("diagnostic pair check requires its original issued selection")
        selected, _ = _ISSUED[self]
        passes = self._passes
        passes.revalidate()
        self._source_passes.revalidate()
        cap = passes.resolve(passes.mechanisms[0])
        callback = getattr(self._owner, "check_source_pair", None)
        if (
            source_passes is not self._source_passes
            or passes.registry is not source_passes.registry
            or passes.binding != source_passes.binding
            or cap is None
            or cap.owner is not self._owner
            or type(callback) is not MethodType
            or callback.__self__ is not self._owner
            or callback.__func__ is not self._function
            or self._function.__code__ is not self._code
            or _pin(self._pin["path"]) != self._pin
            or passes.record() != self._record["passes"]
            or source_passes.record() != self._record["source_passes"]
            or canonical_json(self._record) != selected
        ):
            raise StageGateError("diagnostic pair check original owner, callback or source changed")

    def record(self):
        self.verify_for(self._source_passes)
        return json.loads(_ISSUED[self][0])


def select_source_pair_check(passes, source_passes):
    return DiagnosticSourcePairCheck(passes, source_passes)


def _evidence(workspace, observation):
    path = Path(workspace) / "ordinary_observation.json"
    row = _pin(path)
    if path.stat().st_size > 8 << 20 or row["sha256"] != observation.get("evidence_sha256"):
        raise StageGateError("diagnostic pair check lost its original complete arm evidence")
    return row


def observe(
    selection, *, policy, member, baseline_workspace, candidate_workspace, baseline, candidate, destination, deadline
):
    """Run once after both original arms join and before feedback publication."""
    remaining = deadline - time.monotonic()
    if not math.isfinite(remaining) or remaining <= 0:
        raise StageGateError("diagnostic pair check exhausted the original job deadline")
    with shared_compute_pool(timeout_s=remaining):
        selection.verify_for(policy._compiler_pass_selection)
        pins = [
            _evidence(work, arm) for work, arm in ((baseline_workspace, baseline), (candidate_workspace, candidate))
        ]
        inputs = dict(
            family=member.family,
            capsule=member.capsule,
            source_sha256=member.source_sha256,
            baseline_evidence_sha256=pins[0]["sha256"],
            candidate_evidence_sha256=pins[1]["sha256"],
        )
        report = {"status": "UNKNOWN", "reason": "both original complete numerical/source arms are required"}
        if all(arm.get("functional_status") == "PASS" for arm in (baseline, candidate)):
            report = _closed(
                selection._owner.check_source_pair(
                    baseline_workspace=Path(baseline_workspace),
                    candidate_workspace=Path(candidate_workspace),
                    capsule=policy.load_capsule(member, deadline=deadline),
                    remaining_s=max(0, deadline - time.monotonic()),
                )
            )
            if type(report) is not dict or report.get("status") != "PASS":
                write_json(destination, {"inputs": inputs, "target_report": report, "scope": SCOPE})
                raise StageGateError("original diagnostic target pair relation refused")
        selection.verify_for(policy._compiler_pass_selection)
        if pins != [
            _evidence(work, arm) for work, arm in ((baseline_workspace, baseline), (candidate_workspace, candidate))
        ]:
            raise StageGateError("diagnostic pair check changed its original complete arm evidence")
        if deadline <= time.monotonic():
            raise StageGateError("diagnostic pair check exceeded the original job deadline")
        receipt = dict(
            schema="merlin.diagnostic_source_pair_observation.v1",
            inputs=inputs,
            selection_sha256=document_sha256(selection.record()),
            target_report=report,
            scope=SCOPE,
        )
        destination = Path(destination)
        if destination.exists():
            raise StageGateError("diagnostic pair check requires a fresh private receipt")
        write_json(destination, receipt)
        pin = _pin(destination)
        projection = dict(status=report["status"], evidence_sha256=pin["sha256"], scope=SCOPE)
        _ISSUED[selection][1][pin["sha256"]] = (pin, canonical_json(inputs), canonical_json(projection))
        return projection


def validate(selection, *, policy, member, baseline, candidate, projection):
    """Saved digests cannot issue results in another original live selection."""
    selection.verify_for(policy._compiler_pass_selection)
    if type(projection) is not dict:
        raise StageGateError("diagnostic pair feedback omitted its original live observation")
    issued = _ISSUED[selection][1].get(projection.get("evidence_sha256"))
    if issued is None:
        raise StageGateError("diagnostic pair feedback was not observed by its original live owner")
    pin, inputs, original = issued
    if (
        canonical_json(projection) != original
        or _pin(pin["path"]) != pin
        or inputs
        != canonical_json(
            dict(
                family=member.family,
                capsule=member.capsule,
                source_sha256=member.source_sha256,
                baseline_evidence_sha256=baseline.get("evidence_sha256"),
                candidate_evidence_sha256=candidate.get("evidence_sha256"),
            )
        )
    ):
        raise StageGateError("diagnostic pair feedback changed original member, arms or private receipt")


def recheck_child_result(selection, *, policy, document, work_root, selected_members, deadline):
    """The original parent reruns the checker; a child receipt issues no owner.

    Workspaces come from the actual admitted job, never feedback-supplied paths.
    This deliberately pays for fresh target checks across the process boundary.
    """
    from .public_source_diagnostic import PAIRED_FEEDBACK_SCHEMA

    selection.verify_for(policy._compiler_pass_selection)
    if (
        type(document) is not dict
        or document.get("schema") != PAIRED_FEEDBACK_SCHEMA
        or document.get("selected_members") != [list(key) for key in selected_members]
        or type(document.get("cells")) is not list
        or len(document["cells"]) != len(policy.search_corpus.capsules)
    ):
        raise StageGateError("diagnostic child pair feedback differs from its original admitted job")
    rows = document["cells"]
    members = sorted(policy.search_corpus.capsules, key=lambda row: (row.family, row.capsule))
    for index, (member, row) in enumerate(zip(members, rows, strict=True)):
        if type(row) is not dict or (row.get("family"), row.get("capsule"), row.get("source_sha256")) != (
            member.family,
            member.capsule,
            member.source_sha256,
        ):
            raise StageGateError("diagnostic child pair feedback changed its original member order")
        if (member.family, member.capsule) not in selected_members:
            continue
        if any(type(row.get(arm)) is not dict for arm in ("baseline", "candidate")):
            raise StageGateError("diagnostic child pair feedback lost original arm evidence")
        projection = observe(
            selection,
            policy=policy,
            member=member,
            baseline_workspace=Path(work_root) / f"m{index:03d}_baseline",
            candidate_workspace=Path(work_root) / f"m{index:03d}_candidate",
            baseline=row["baseline"],
            candidate=row["candidate"],
            destination=Path(work_root) / "parent-pair-checks" / f"m{index:03d}.json",
            deadline=deadline,
        )
        if projection != row.get("diagnostic_pair_check"):
            raise StageGateError("diagnostic child pair differs from the parent's fresh original target check")
