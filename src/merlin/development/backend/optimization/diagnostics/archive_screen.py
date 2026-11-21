"""Native queue inference from original process and full numerical observations.

This consumes an independently qualified event engine; it neither fits prices
nor grants target semantics. Identical executions supply one prediction, while
both original functional arms remain required and visible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path

from merlin.compiler.lower.source.reviewed import ReviewedPassCapability
from merlin.development.backend.optimization.contracts import StageGateError, canonical_json
from merlin.development.backend.optimization.diagnostics import spike_runtime as RT
from merlin.development.backend.optimization.diagnostics.archive_evidence import AdditionalArchiveEvidence
from merlin.development.backend.optimization.diagnostics.spike_runtime import ArchiveSpikeRuntime
from merlin.development.backend.optimization.ordinary.ordinary_archive_screen import issue_archive_screen
from merlin.runtime.views import native_simulator as F

# ---- archive_native_screen ---------------------------------------------------------------------


def verify_semantics(capability, binding):
    if (
        type(capability) is not ReviewedPassCapability
        or capability.mechanism != "native_timing_semantics"
        or capability.binding != binding
    ):
        raise StageGateError("native archive screening needs independently reviewed exact event/resource semantics")
    capability.revalidate()


def original_numerical_observation(directory, value, *, checked_source=None):
    """Reopen the original queued receipt and its live selection, never a verdict argument."""
    path = directory / "ordinary_observation.json"
    pin = F.pin(path, max_bytes=64 << 20)
    if pin["sha256"] != value.get("evidence_sha256"):
        raise StageGateError("native inference lost its original full numerical observation")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise StageGateError("native numerical observation changed during reopening")
    original = json.loads(raw)
    public = {name: item for name, item in original.items() if name != "original_grade"}
    if checked_source is not None:
        from merlin.development.backend.optimization.ordinary.ordinary_checked_source import record
        from merlin.development.targetgen.execution.host.checked_source_diagnostics import (
            NUMERICAL_COUNTS_PROTOCOL,
            numerical_grade_matches_selection,
        )

        if type(checked_source) is not tuple or len(checked_source) != 2:
            raise StageGateError("native diagnostic inference lost its original live complete-source selection")
        selected = record(*checked_source)
        if (
            not numerical_grade_matches_selection(selected, original.get("original_grade", {}))
            or public.pop("checked_source_diagnostic", None) != selected
        ):
            raise StageGateError("native diagnostic inference lost its original live complete-source selection")
        if selected.get("verification_protocol") == NUMERICAL_COUNTS_PROTOCOL:
            from merlin.development.targetgen.execution.host.checked_source_diagnostics import (
                reopen_numerical_counts_grade,
            )

            try:
                reopen_numerical_counts_grade(checked_source[0], original["original_grade"])
            except (OSError, ValueError, TypeError, KeyError) as error:
                raise StageGateError("native fast inference lost original source/ISA/full-output witnesses") from error
    if canonical_json(public) != canonical_json({k: v for k, v in value.items() if k != "evidence_sha256"}):
        raise StageGateError("native inference numerical receipt differs from its original returned observation")
    tiers = original.get("original_grade", {}).get("tiers", {})
    if original.get("functional_status") != "PASS" or any(
        tiers.get(tier, {}).get("status") != "pass" for tier in ("L0", "L1", "L2")
    ):
        raise StageGateError("native inference requires original complete L0/L1/L2 numerical checks")
    if F.pin(path, max_bytes=64 << 20) != pin:
        raise StageGateError("native numerical observation changed during consumption")
    return original, pin


def inference_case(*, key, member, family, dataset, manifest_sha256, report, directory, value, checked_source=None):
    """Reopen the original private numerical verdict for this exact queued arm."""
    from merlin_timing import contracts as C

    original, pin = original_numerical_observation(directory, value, checked_source=checked_source)
    path = directory / "ordinary_observation.json"
    if original.get("compiled", {}).get("elf_sha256") != report["elf_sha256"]:
        raise StageGateError("native numerical observation belongs to another executable")
    case = {
        "case_id": key,
        "source_group_sha256": C.digest({"original_public_source": member.source_sha256}),
        "mechanism_family": family,
        "source": {
            "visibility": "public",
            "phase": "phase2",
            "manifest_sha256": manifest_sha256,
            "source_sha256": member.source_sha256,
        },
        **{name: report[name] for name in ("elf_sha256", "input_sha256", "scope_sha256", "device_sha256")},
    }
    case["workload_sha256"] = C.workload(case)
    row = {
        "case": case,
        "features": {name: report["feature_counts"][name] for name in dataset["feature_names"]},
        "event_report": {"sha256": C.digest(report), "report": report},
        "numeric": {
            "status": "pass",
            "receipt_sha256": pin["sha256"],
            "execution_sha256": C.execution_identity(case),
            "measured_now": True,
        },
    }
    if F.pin(path, max_bytes=64 << 20) != pin:
        raise StageGateError("native numerical observation changed during consumption")
    return row


def compare(requests, *, dataset, qualification):
    from merlin_timing import contracts as C
    from merlin_timing import feedback as B
    from merlin_timing import validation as V

    declaration = {
        "schema": C.NATIVE_CASES,
        "profile": dataset["profile"],
        "domain": dataset["domain"],
        "feature_names": dataset["feature_names"],
    }
    unique, aliases, evidence, signatures = {}, {}, {}, {}
    for row in requests:
        C.native_cases({**declaration, "cases": [row]})
        case = row["case"]
        if case["case_id"] in aliases:
            raise StageGateError("native queue inference requires a distinct identity for each queued arm")
        identity = tuple(case[name] for name in ("elf_sha256", "input_sha256", "scope_sha256"))
        prediction, reason = V.native_prediction(row, case, declaration)
        numeric = row["numeric"]
        if numeric is None or numeric["status"] != "pass" or numeric["measured_now"] is not True:
            raise StageGateError("native duplicate inference requires every original current numerical receipt")
        if row["event_report"] is None:
            raise StageGateError("native queue inference requires each original event observation")
        report = row["event_report"]["report"]
        scope = report.get("scope_binding")
        signature = (
            C.execution_identity(case),
            case["mechanism_family"],
            row["features"],
            prediction,
            reason,
            {name: scope.get(name) for name in ("begin_pc", "end_pc", "occurrence")} if scope else None,
        )
        prior = unique.setdefault(identity, row)
        # Two fresh process runs of the same bytes must agree on their actual
        # scoped prediction. Independent review/process receipts and whole-trace
        # diagnostics need not have identical hashes. Preserve each one's pins.
        # Never choose a faster repeat or average conflicting scoped results.
        if signatures.setdefault(identity, signature) != signature:
            raise StageGateError("identical native executions disagree; no representative may be selected")
        aliases[case["case_id"]] = prior["case"]["case_id"]
        evidence[case["case_id"]] = {
            "event_report_sha256": row["event_report"]["sha256"],
            "numeric_receipt_sha256": numeric["receipt_sha256"],
        }
    result = B.compare_native(
        {
            "schema": C.NATIVE_CASES,
            "profile": dataset["profile"],
            "domain": dataset["domain"],
            "feature_names": dataset["feature_names"],
            "cases": list(unique.values()),
        },
        qualification=qualification,
    )
    predictions = {row["case_id"]: row for row in result["cases"]}
    return {
        **result,
        "cases": [
            {
                **predictions[original],
                "case_id": key,
                "prediction_kind": "event_timing",
                "original_execution_evidence": evidence[key],
            }
            for key, original in aliases.items()
        ],
    }


# ---- archive_screen_review ---------------------------------------------------------------------
# Reconstruct public archive eligibility, raw activity support and held gates.
#
# This standalone preflight supplies no Phase 1, corpus, queue or timing owner.


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
    from merlin_timing import archive as A

    if type(additional_archives) is not tuple:
        raise StageGateError("additional native archives require an immutable declaration tuple")
    if additional_archives:
        if prediction_kind != "event_timing" or screen_path is not None:
            raise StageGateError("joint archives require explicit native event exploration")
        from merlin.development.backend.optimization.diagnostics import archive_evidence as JOINT

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
    from merlin_timing import validation as V

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
    if original_reports is None:
        # A union must not hide an out-of-domain repeat behind another repeat
        # of the same case. Legacy single-archive coverage keeps its behavior.
        held.difference_update(
            case["case_id"] for case, r in raw if case["split"] == "test" and domain_reason(r, ranges) is not None
        )
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


# ---- archive_screen_selection ------------------------------------------------------------------
# Explicit original-host archive selection for the normal queued stage.
#
# This object carries live reviewed owners, never candidate JSON callbacks or
# qualification flags. Ordinary Phase 1, corpus and final gates run unchanged.


@dataclass(frozen=True)
class ArchiveScreenSelection:
    review_id: str
    archive: Path
    archive_sha256: str
    binding_path: Path
    observations_path: Path
    screen_path: Path | None
    runtime: ArchiveSpikeRuntime
    correspondence: ReviewedPassCapability
    expected_rtl_facts_sha256: str
    expected_observed_contract_sha256: str

    def preflight(self):
        """Review actual public archive support without starting ordinary search."""

        self.correspondence.revalidate()
        native = type(self) is NativeArchiveScreenSelection
        if native:
            from merlin.development.backend.optimization.diagnostics.archive_evidence import declarations
            from merlin.development.backend.optimization.ordinary._ordinary_service_effects import select

            verify_semantics(self.native_semantics, self.correspondence.binding)
            select(self.service_effects, self.correspondence.binding)
            declarations(self.additional_archives)
        return review_archive(
            runtime=self.runtime,
            archive=self.archive,
            archive_sha256=self.archive_sha256,
            binding_path=self.binding_path,
            observations_path=self.observations_path,
            screen_path=self.screen_path,
            prediction_kind="event_timing" if native else "empirical_screen",
            **({"additional_archives": self.additional_archives} if native and self.additional_archives else {}),
        )

    def install(self, policy):
        """Rebuild original eligibility/held gates before admitting any job."""
        if (
            type(self) not in (ArchiveScreenSelection, NativeArchiveScreenSelection)
            or type(self.runtime) is not ArchiveSpikeRuntime
            or type(self.correspondence) is not ReviewedPassCapability
        ):
            raise StageGateError("normal archive screening requires explicit original live host owners")
        if type(self) is NativeArchiveScreenSelection:
            verify_semantics(self.native_semantics, self.correspondence.binding)
            if self.screen_path is not None:
                raise StageGateError("native event selection cannot consume a fitted linear screen")
        arguments = {field.name: getattr(self, field.name) for field in fields(self)}
        if type(self) is NativeArchiveScreenSelection:
            if arguments.pop("additional_archives"):
                raise StageGateError("joint native archives currently require explicit public diagnostic exploration")
        owner = issue_archive_screen(policy, **arguments)
        policy.install_screen(owner)
        return owner


@dataclass(frozen=True)
class NativeArchiveScreenSelection(ArchiveScreenSelection):
    """Explicit separately versioned native selection; screen_path must be None.

    Both a live resource-semantics review and native held gates are required.
    This selection cannot promote predictions to measured objective values.
    """

    native_semantics: ReviewedPassCapability
    service_effects: ReviewedPassCapability | None = None
    additional_archives: tuple[AdditionalArchiveEvidence, ...] = ()


def install(selection, policy):
    """Called only after the original policy/build selection and before its pool."""
    if selection is not None:
        if type(selection) not in (ArchiveScreenSelection, NativeArchiveScreenSelection):
            raise StageGateError("archive-screen JSON cannot replace original host selection")
        return selection.install(policy)
    return None
