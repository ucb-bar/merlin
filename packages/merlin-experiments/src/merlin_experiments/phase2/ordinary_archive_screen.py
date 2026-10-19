"""Reviewed public archive screening in ordinary search, independent of incumbents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.runtime import _native_files as F

from . import archive_count_observation as OBS
from . import archive_spike_runtime as RT
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


def _domain_reason(report, ranges):
    counts = report.get("feature_counts") or {}
    if set(counts) != set(ranges):
        return "complete raw activity vocabulary differs from the archive"
    for name, (lo, hi) in ranges.items():
        value = counts[name]
        if type(value) not in (int, float) or not lo <= value <= hi:
            return "raw activity outside public training support: " + name
    return None


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
    context_sha256: str
    review_id: str
    dataset_json: bytes
    screen_json: bytes
    report_json: bytes
    ranges_json: bytes
    files: tuple[tuple[str, str, tuple], ...]
    families: tuple[str, ...]
    _issuer: object

    @property
    def sha256(self):
        return document_sha256(
            {
                "schema": "merlin.ordinary_archive_screen.v1",
                "review_id": self.review_id,
                "context_sha256": self.context_sha256,
                "runtime_sha256": self.runtime.sha256,
                "dataset_sha256": document_sha256(json.loads(self.dataset_json)),
                "screen_sha256": document_sha256(json.loads(self.screen_json)),
                "report_sha256": document_sha256(json.loads(self.report_json)),
                "raw_activity_ranges": json.loads(self.ranges_json),
                "files": self.files,
            }
        )

    def verify(self, policy):
        if self._issuer is not _ISSUER or self.context_sha256 != context(policy):
            raise StageGateError("archive screen changed its original normal policy selection")
        self.runtime.verify()
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
            "event_engine_qualified": False,
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
        if family not in self.families or member.descriptor.get("performance", {}).get("capacity"):
            return {"status": "UNKNOWN", "reason": "source semantic/capacity regime outside archive qualification"}
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
        return {"status": result["status"], "reason": result.get("reason"), "receipt_sha256": F.digest(path)}

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
        requests, unavailable = [], {}
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
                            requests.append(
                                {
                                    "case_id": key,
                                    "workload_sha256": workload,
                                    "features": {n: report["feature_counts"][n] for n in screen["feature_names"]},
                                }
                            )
                if reason is not None:
                    unavailable[key] = reason
        comparison = B.compare(
            screen,
            {"schema": C.CASES, "profile": dataset["profile"], "domain": dataset["domain"], "cases": requests},
            qualification={"dataset": dataset, "report": json.loads(self.report_json)},
        )
        predictions = {row["case_id"]: row for row in comparison["cases"]}
        evidence = []
        for index, cell in enumerate(document["cells"]):
            arms = [predictions.get(str(index) + ":" + arm) for arm in ("baseline", "candidate")]
            intervals = [row.get("interval") if row else None for row in arms]
            saved = max(0, intervals[0][0] - intervals[1][1]) if all(intervals) else None
            evidence.append(
                {
                    "family": cell["family"],
                    "capsule": cell["capsule"],
                    "baseline": arms[0],
                    "candidate": arms[1],
                    "conservative_saved_cycles": saved,
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
            "scope": "public archive empirical exploration only; original incumbent/final/plateau gates unchanged",
            "event_engine_qualified": False,
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
    expected_rtl_facts_sha256,
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
    if (
        policy._ordinary_screen is not None
        or policy.evaluation_pool is not None
        or F.digest(policy.feedback_evaluator.rtl_facts_path) != expected_rtl_facts_sha256
    ):
        raise StageGateError("archive screen must bind exact original facts before queue admission")
    inventory = A.inspect_archive(archive, archive_sha256)
    binding, _b = RT.metadata(binding_path)
    observations, _o = RT.metadata(observations_path)
    screen, _s = RT.metadata(screen_path)
    model = json.loads(runtime.model_json)
    selected_domain = dict(model["domain"], target_model_sha256=F.digest(runtime.model_path))
    if (
        binding["domain"] != selected_domain
        or binding["review"]["timing_selection"] != json.loads(runtime.timing_json)
        or inventory.get("export_scope") != "training_eligible_public"
    ):
        raise StageGateError("archive timing/runtime/source selection differs from the original reviewed public intake")
    plan = A.prepare(inventory, binding)
    dataset = A.attach(plan, observations)
    report = V.validate(dataset, screen)
    if not report["qualified"] or report["event_engine_qualified"] or dataset["profile"] != "phase1-public":
        raise StageGateError("original archive empirical held gates did not qualify")
    cases = {c["case_id"]: c for c in dataset["cases"]}
    raw = [
        (cases[r["case_id"]], observations["reports"][r["record_id"]]["report"])
        for r in dataset["records"]
        if r["record_id"] in observations["reports"]
    ]
    training = [r["feature_counts"] for case, r in raw if case["split"] == "train"]
    names = set(training[0])
    if any(set(r) != names for r in training):
        raise StageGateError("archive training raw activity vocabularies disagree")
    ranges = {n: [min(r[n] for r in training), max(r[n] for r in training)] for n in sorted(names)}
    held = {case["case_id"] for case, r in raw if case["split"] == "test" and _domain_reason(r, ranges) is None}
    declared = {case["case_id"] for case in dataset["cases"] if case["split"] == "test"}
    if held != declared:
        raise StageGateError("additional raw-activity guard lacks complete original held coverage")
    from merlin_timing import feedback as B
    from merlin_timing import fitting as FIT
    from merlin_timing import observations as O

    sources = (A, C, V, B, FIT, O, RT, OBS)
    paths = tuple(
        dict.fromkeys(
            (
                Path(archive),
                Path(binding_path),
                Path(observations_path),
                Path(screen_path),
                policy.feedback_evaluator.rtl_facts_path,
                *(Path(m.__file__).resolve() for m in sources),
                Path(__file__).resolve(),
            )
        )
    )
    files = tuple((str(p), F.pin(p, max_bytes=256 << 20)["sha256"], F.identity(p.stat())) for p in paths)
    result = OrdinaryArchiveScreen(
        runtime,
        context(policy),
        review_id,
        canonical_json(dataset),
        canonical_json(screen),
        canonical_json(report),
        canonical_json(ranges),
        files,
        tuple(sorted({case["mechanism_family"] for case in dataset["cases"] if case["split"] == "train"})),
        _ISSUER,
    )
    result.verify(policy)
    return result
