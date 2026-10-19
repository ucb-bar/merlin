"""Reviewed public archive screening in ordinary search, independent of incumbents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.llvmlower.reviewed_pass_capabilities import PassCapabilityBinding, ReviewedPassCapability
from merlin.perf import work_volume as WORK
from merlin.runtime import _native_files as F
from merlin.targetgen.contract import interface_emit as INTERFACE

from . import _archive_feedback as HISTORY
from . import _archive_feedback_context as HISTORY_CONTEXT
from . import _archive_source_domain as SOURCE_DOMAIN
from . import archive_count_observation as OBS
from . import archive_native_screen as NATIVE
from . import archive_screen_review as REVIEW
from . import archive_spike_runtime as RT
from . import ordinary_capsule_feedback as FEEDBACK
from .archive_screen_review import domain_reason as _domain_reason
from .contracts import StageGateError, canonical_json, document_sha256, write_json

_ISSUER = object()
_FAMILIES = {"matmul": "matmul", "linear": "matmul", "conv2d": "conv", "resident_reuse": "resident_chain"}


def context(policy):
    binding = dict(policy._binding())
    binding.pop("ordinary_screen_sha256", None)
    return document_sha256(binding)


def _family(member):
    operation = member.descriptor.get("operation") or {}
    return _FAMILIES.get(operation.get("op")) if isinstance(operation, dict) else None


def _root(policy, document):
    pool = policy.evaluation_pool
    if pool is None:
        raise StageGateError("archive screening needs its original completed queued job")
    core = {k: v for k, v in document.items() if k != "screening"}
    matches = []
    for row in pool.observations():
        result = row.get("result") or {}
        if row["state"] == "completed" and canonical_json(
            {k: v for k, v in result.items() if k != "screening"}
        ) == canonical_json(core):
            matches.append(pool._get(row["job_id"]).root / "rtl")
    if len(matches) != 1:
        raise StageGateError("archive screening lacks one original completed feedback owner")
    return matches[0]


@dataclass(frozen=True, eq=False)
class OrdinaryArchiveScreen:
    runtime: RT.ArchiveSpikeRuntime
    correspondence: ReviewedPassCapability
    correspondence_identity: str
    context_sha256: str
    review_id: str
    dataset_json: bytes
    screen_json: bytes
    report_json: bytes
    ranges_json: bytes
    files: tuple[tuple[str, str, tuple], ...]
    families: tuple[str, ...]
    _issuer: object
    native_semantics: ReviewedPassCapability | None = None
    native_semantics_identity: str | None = None
    retained_feedback_json: bytes = b"{}"

    @property
    def sha256(self):
        return document_sha256(
            {
                "schema": (
                    "merlin.ordinary_archive_screen.v2"
                    if self.native_semantics
                    else "merlin.ordinary_archive_screen.v1"
                ),
                **({"native_semantics_identity": self.native_semantics_identity} if self.native_semantics else {}),
                "review_id": self.review_id,
                "context_sha256": self.context_sha256,
                "runtime_sha256": self.runtime.sha256,
                "correspondence_identity": self.correspondence_identity,
                "dataset_sha256": document_sha256(json.loads(self.dataset_json)),
                "screen_sha256": document_sha256(json.loads(self.screen_json)),
                "report_sha256": document_sha256(json.loads(self.report_json)),
                "raw_activity_ranges": json.loads(self.ranges_json),
                "retained_feedback_sha256": document_sha256(json.loads(self.retained_feedback_json)),
                "files": self.files,
            }
        )

    def verify(self, policy):
        if self._issuer is not _ISSUER or self.context_sha256 != context(policy):
            raise StageGateError("archive screen changed its original normal policy selection")
        self.runtime.verify()
        self.correspondence.revalidate()
        if self.correspondence.identity != self.correspondence_identity:
            raise StageGateError("original count-model correspondence owner changed")
        if self.native_semantics is not None:
            NATIVE.verify_semantics(self.native_semantics, self.correspondence.binding)
            if self.native_semantics.identity != self.native_semantics_identity:
                raise StageGateError("original native event semantics owner changed")
        for path, _digest, identity in self.files:
            selected = F.canonical(Path(path))
            if F.identity(selected.stat()) != identity:
                raise StageGateError("reviewed original archive/screen evidence changed")

    def summary(self):
        report = json.loads(self.report_json)
        return {
            "status": "qualified_public_screening",
            "calibration_sha256": self.sha256,
            "source": "reviewed_phase1_public_archive",
            "metrics": report["metrics"],
            "authority": "exploration ordering only",
            "event_engine_qualified": self.native_semantics is not None,
            **({"prediction_kind": "event_timing"} if self.native_semantics else {}),
            "raw_activity_features_guarded": len(json.loads(self.ranges_json)),
            "domain_limits": [
                "exact runtime/scope/hardware selection",
                "training activity ranges and known semantic families",
                "no capacity extrapolation qualification",
                "public held set revisited diagnostically",
            ],
            "screening_net_benefit": "UNKNOWN until complete route costs are observed",
        }

    def observe_member(self, policy, member, workspace, compiled, *, sandbox, remaining_s, functional):
        self.verify(policy)
        family = _family(member)
        source_domain = None
        if family is None and (member.descriptor.get("operation") or {}).get("op") == "model":
            source_domain = SOURCE_DOMAIN.current(member, compiled, workspace, json.loads(self.retained_feedback_json))
            family = source_domain["family"]
        if family not in self.families or member.descriptor.get("performance", {}).get("capacity"):
            return {"status": "UNKNOWN", "reason": (source_domain or {}).get("reason")
                    or "source semantic/capacity regime outside archive qualification",
                    **({"source_domain": source_domain} if source_domain else {})}
        if compiled.get("status") != "observed" or functional is not True:
            return {"status": "UNKNOWN", "reason": "original complete functional/linked observation unavailable"}
        matches = [
            p
            for p in F.members(workspace, max_files=10000)
            if p.is_file()
            and not p.is_symlink()
            and p.suffix == ".elf"
            and p.stat().st_size <= 64 << 20
            and F.digest(p) == compiled["elf_sha256"]
        ]
        if not matches:
            return {"status": "UNKNOWN", "reason": "original observed executable unavailable"}
        # This identifies the original immutable public source/input closure,
        # rather than candidate C text which also contains changed instructions.
        input_sha = document_sha256({"kind": "original_frozen_public_capsule", "sha256": member.source_sha256})
        result = OBS.observe(
            matches[0],
            runtime=self.runtime,
            policy=sandbox,
            directory=workspace / "archive_counts",
            input_sha256=input_sha,
            remaining_s=remaining_s,
            numerical_pass=functional,
        )
        path = workspace / "archive_counts_receipt.json"
        write_json(path, result)
        return {"status": result["status"], "reason": result.get("reason"), "receipt_sha256": F.digest(path),
                **({"source_domain": source_domain} if source_domain else {})}

    def screen(self, policy, document, *, work_root=None):
        from merlin_timing import contracts as C
        from merlin_timing import feedback as B
        from merlin_timing import observations as O

        from . import ordinary_fast_feedback as FAST

        self.verify(policy)
        if document.get("kind") != FAST.KIND:
            raise StageGateError("archive screen requires the complete original compile/Spike roster")
        root = F.canonical(Path(work_root)) if work_root is not None else _root(policy, document)
        members = sorted(policy.search_corpus.capsules, key=lambda m: (m.family, m.capsule))
        if [(c["family"], c["capsule"], c["source_sha256"]) for c in document["cells"]] != [
            (m.family, m.capsule, m.source_sha256) for m in members
        ]:
            raise StageGateError("archive screen changed its complete original member/source roster")
        screen, dataset, ranges = (
            json.loads(self.screen_json),
            json.loads(self.dataset_json),
            json.loads(self.ranges_json),
        )
        requests, unavailable, raw_reports = [], {}, {}
        for index, (cell, member) in enumerate(zip(document["cells"], members, strict=True)):
            workload = C.digest(
                {
                    "source": member.source_sha256,
                    "scope": dataset["domain"]["scope_sha256"],
                    "device": dataset["domain"]["device_sha256"],
                }
            )
            for arm in ("baseline", "candidate"):
                key = str(index) + ":" + arm
                value = cell[arm]
                reason = "original complete functional screen did not pass"
                if value["functional_status"] == "PASS":
                    directory = root / f"m{index:03d}_{arm}"
                    pin = value.get("archive_counts") or {}
                    reason = pin.get("reason") or "fresh archive count receipt unavailable"
                    if pin.get("status") == "observed":
                        family = _family(member)
                        if family is None:
                            source_domain = SOURCE_DOMAIN.current(
                                member, value["compiled"], directory, json.loads(self.retained_feedback_json),
                            )
                            if (source_domain.get("status") != "source_domain_supported"
                                    or source_domain != pin.get("source_domain")):
                                raise StageGateError("original current typed source/domain binding changed")
                            family = source_domain["family"]
                        if family not in self.families or member.descriptor.get("performance", {}).get("capacity"):
                            raise StageGateError("original source semantic/capacity regime changed")
                        receipt_path = directory / "archive_counts_receipt.json"
                        if F.pin(receipt_path, max_bytes=1 << 20)["sha256"] != pin.get("receipt_sha256"):
                            raise StageGateError("original archive count receipt changed")
                        receipt = json.loads(receipt_path.read_bytes())
                        report_path = directory / "archive_counts/report.json"
                        if F.pin(report_path, max_bytes=16 << 20) != receipt.get("report"):
                            raise StageGateError("original archive count report changed")
                        report = json.loads(report_path.read_bytes())
                        if (
                            receipt.get("runtime_sha256") != self.runtime.sha256
                            or report.get("elf_sha256") != value["compiled"].get("elf_sha256")
                            or report.get("domain") != dataset["domain"]
                            or not O.feature_complete(report)
                            or report.get("input_sha256")
                            != document_sha256(
                                {"kind": "original_frozen_public_capsule", "sha256": member.source_sha256}
                            )
                        ):
                            raise StageGateError("original count report changed its selected execution/domain")
                        raw_reports[key] = report
                        for invocation in receipt["processes"]:
                            path = (
                                directory
                                / "archive_counts"
                                / ("functional" if invocation == receipt["processes"][0] else "annotated")
                            )
                            actual_path = Path(invocation["path"])
                            if not actual_path.is_relative_to(path) or F.digest(actual_path) != invocation["sha256"]:
                                raise StageGateError("original count invocation changed or escaped its owner")
                            actual = I.verify(actual_path)
                            if actual.get("stage") != "archive_screen_count" or not any(
                                p["sha256"] == report["elf_sha256"] for p in actual.get("inputs", [])
                            ):
                                raise StageGateError("original count process changed its executable input")
                            if invocation == receipt["processes"][-1] and not any(
                                p["sha256"] == receipt["report"]["sha256"] for p in actual.get("outputs", [])
                            ):
                                raise StageGateError("original count report lacks its actual process producer")
                        reason = _domain_reason(report, ranges)
                        if reason is None:
                            if self.native_semantics is not None:
                                requests.append(
                                    NATIVE.inference_case(
                                        key=key,
                                        member=member,
                                        family=family,
                                        dataset=dataset,
                                        manifest_sha256=policy.search_corpus.manifest_sha256,
                                        report=report,
                                        directory=directory,
                                        value=value,
                                    )
                                )
                            else:
                                requests.append(
                                    {
                                        "case_id": key,
                                        "workload_sha256": workload,
                                        "features": {n: report["feature_counts"][n] for n in screen["feature_names"]},
                                    }
                                )
                if reason is not None:
                    unavailable[key] = reason
        qualification = {"dataset": dataset, "report": json.loads(self.report_json)}
        comparison = (
            NATIVE.compare(requests, dataset=dataset, qualification=qualification)
            if self.native_semantics is not None
            else B.compare(
                screen,
                {"schema": C.CASES, "profile": dataset["profile"], "domain": dataset["domain"], "cases": requests},
                qualification=qualification,
            )
        )
        predictions = {row["case_id"]: row for row in comparison["cases"]}
        facts, _facts_pin = RT.metadata(policy.feedback_evaluator.rtl_facts_path, maximum=8 << 20)
        machine = FEEDBACK.R.machine_bounds(policy.target_experiment.target, facts=facts)
        machine.update({name: dataset["domain"][name] for name in ("scope_sha256", "device_sha256")})
        retained = json.loads(self.retained_feedback_json)
        evidence = []
        for index, cell in enumerate(document["cells"]):
            arms = [predictions.get(str(index) + ":" + arm) for arm in ("baseline", "candidate")]
            intervals = [row.get("interval") if row else None for row in arms]
            saved = max(0, intervals[0][0] - intervals[1][1]) if all(intervals) else None
            history = {}
            for arm in ("baseline", "candidate"):
                report = raw_reports.get(str(index) + ":" + arm)
                history[arm] = HISTORY.queued(
                    retained, compiled=cell[arm]["compiled"], report=report,
                    directory=root / f"m{index:03d}_{arm}", value=cell[arm],
                ) if report is not None else {
                    "status": "UNKNOWN", "measurement": None, "measured_now": False,
                    "reason": "original complete same-executable count/window observation unavailable",
                }
            evidence.append(
                {
                    "family": cell["family"],
                    "capsule": cell["capsule"],
                    "baseline": arms[0],
                    "candidate": arms[1],
                    "conservative_saved_cycles": saved,
                    "retained_measurements": history,
                    "source_domains": {
                        arm: (cell[arm].get("archive_counts") or {}).get("source_domain")
                        for arm in ("baseline", "candidate")
                    },
                    "capsule_feedback": {
                        arm: FEEDBACK.join_capsule_feedback(
                            members[index].descriptor,
                            cell[arm]["compiled"],
                            raw_reports.get(str(index) + ":" + arm),
                            predictions.get(str(index) + ":" + arm),
                            target=policy.target_experiment.target,
                            facts=facts,
                            machine=machine,
                            measurement=history[arm].get("measurement"),
                        )
                        for arm in ("baseline", "candidate")
                    },
                    "unavailable": {
                        a: unavailable[str(index) + ":" + a]
                        for a in ("baseline", "candidate")
                        if str(index) + ":" + a in unavailable
                    },
                }
            )
        return {
            "status": "qualified_public_screening",
            "calibration_sha256": self.sha256,
            "evidence": evidence,
            "order": [
                [r["family"], r["capsule"]]
                for r in sorted(
                    evidence,
                    key=lambda r: (r["conservative_saved_cycles"] is None, -(r["conservative_saved_cycles"] or 0)),
                )
            ],
            "scope": "public archive exploration only; original incumbent/final/plateau gates unchanged",
            "event_engine_qualified": self.native_semantics is not None,
            **({"prediction_kind": "event_timing"} if self.native_semantics else {}),
            "promotion_authority": "NONE",
        }


def issue_archive_screen(
    policy,
    *,
    review_id,
    archive,
    archive_sha256,
    binding_path,
    observations_path,
    screen_path,
    runtime,
    correspondence,
    expected_rtl_facts_sha256,
    expected_observed_contract_sha256,
    native_semantics=None,
):
    """Reconstruct eligibility and held gates from the exact reviewed public archive.

    The host explicitly reviews correspondence to its original selected target.
    This intake never accepts hidden labels or a saved qualification boolean.
    """
    from merlin_timing import archive as A
    from merlin_timing import contracts as C
    from merlin_timing import validation as V

    if type(runtime) is not RT.ArchiveSpikeRuntime or not isinstance(review_id, str) or not review_id.strip():
        raise StageGateError("archive screen needs its original reviewed runtime and host attribution")
    policy.validate_search_inputs()
    runtime.verify()
    from merlin.llvmlower import reviewed_pass_capabilities as CAP

    model = json.loads(runtime.model_json)
    binding = PassCapabilityBinding(
        hardware=expected_rtl_facts_sha256,
        software=expected_observed_contract_sha256,
        numerical_policy=model["domain"]["feature_schema_sha256"],
        runtime=F.digest(runtime.model_path),
        library=document_sha256(model["libraries"]),
    )
    if (
        type(correspondence) is not ReviewedPassCapability
        or correspondence.mechanism != "timing_model_correspondence"
        or correspondence.binding != binding
    ):
        raise StageGateError("archive screen needs original exact live count-model hardware correspondence")
    correspondence.revalidate()
    if native_semantics is not None:
        NATIVE.verify_semantics(native_semantics, binding)
    if (
        policy._ordinary_screen is not None
        or policy.evaluation_pool is not None
        or F.digest(policy.feedback_evaluator.rtl_facts_path) != expected_rtl_facts_sha256
    ):
        raise StageGateError("archive screen must bind exact original facts before queue admission")
    reviewed = REVIEW.review_archive(
        runtime=runtime,
        archive=archive,
        archive_sha256=archive_sha256,
        binding_path=binding_path,
        observations_path=observations_path,
        screen_path=screen_path,
        prediction_kind="event_timing" if native_semantics is not None else "empirical_screen",
    )
    dataset, screen, report, ranges = (reviewed[key] for key in ("dataset", "screen", "report", "raw_activity_ranges"))
    retained = HISTORY.prepare(
        archive=archive, archive_sha256=archive_sha256, binding_path=binding_path, dataset=dataset,
    )
    from merlin_timing import feedback as B
    from merlin_timing import fitting as FIT
    from merlin_timing import observations as O

    sources = (A, C, V, B, FIT, O, RT, OBS, CAP, REVIEW, FEEDBACK, FEEDBACK.R, NATIVE,
               HISTORY, HISTORY_CONTEXT, SOURCE_DOMAIN, *SOURCE_DOMAIN.source_modules(),
               OBS.ALIASES, *OBS.ALIASES.source_modules(), WORK, INTERFACE)
    paths = tuple(
        dict.fromkeys(
            (
                Path(archive),
                Path(binding_path),
                Path(observations_path),
                *((Path(screen_path),) if screen_path is not None else ()),
                policy.feedback_evaluator.rtl_facts_path,
                *(Path(m.__file__).resolve() for m in sources),
                Path(HISTORY.source_observation.__code__.co_filename).resolve(),
                Path(__file__).resolve(),
            )
        )
    )
    files = tuple((str(p), F.pin(p, max_bytes=256 << 20)["sha256"], F.identity(p.stat())) for p in paths)
    result = OrdinaryArchiveScreen(
        runtime,
        correspondence,
        correspondence.identity,
        context(policy),
        review_id,
        canonical_json(dataset),
        canonical_json(screen),
        canonical_json(report),
        canonical_json(ranges),
        files,
        tuple(reviewed["families"]),
        _ISSUER,
        native_semantics,
        native_semantics.identity if native_semantics else None,
        canonical_json(retained),
    )
    result.verify(policy)
    return result
