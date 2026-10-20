"""Bounded development search over the actual host-issued evaluation queue.

Ranking consumes complete paired RTL observations, including non-contraction
members. Prediction order remains screening only. This owner never materializes
a compiler or grants lineage, numerical qualification or final acceptance.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import asdict, dataclass

from .contracts import StageGateError, document_sha256
from .search_inputs import ORDINARY_KIND
from .search_priorities import measured_priorities


@dataclass(frozen=True)
class SearchSettings:
    beam_width: int = 3
    soft_seconds: int = 14_400
    hard_seconds: int = 28_800
    final_reserve_seconds: int = 2_700
    minimum_improvement: float = 0.01
    plateau_batches: int = 3
    queued_job_timeout_seconds: int | None = None

    def verify(self):
        if (
            any(
                type(value) is not int or value < 1
                for value in (
                    self.beam_width,
                    self.soft_seconds,
                    self.hard_seconds,
                    self.final_reserve_seconds,
                    self.plateau_batches,
                )
            )
            or self.beam_width > 3
            or not self.final_reserve_seconds < self.soft_seconds <= self.hard_seconds <= 28_800
            or type(self.minimum_improvement) not in (int, float)
            or not math.isfinite(self.minimum_improvement)
            or not 0 < self.minimum_improvement < 1
            or self.queued_job_timeout_seconds is not None
            and (
                type(self.queued_job_timeout_seconds) is not int
                or not 1 <= self.queued_job_timeout_seconds <= 2400
                or self.queued_job_timeout_seconds > self.soft_seconds - self.final_reserve_seconds
            )
        ):
            raise StageGateError("search requires a bounded beam, time reserve and positive improvement threshold")


def rank_measured_contenders(
    observations, *, roster, corpus_sha256, configuration_sha256, timing_kind="component-rtl-feedback"
):
    """Replay complete development observations; repeated timings retain their range.

    This pure projection is diagnostic. Only the coordinator can obtain input
    observations from its live issued pool. Bounds are observed ranges, not
    statistical confidence intervals or claims about complete held-out programs.
    """
    from .component_workflow import validate_component_feedback

    roster = tuple(sorted(roster))
    if not roster or len(set(roster)) != len(roster):
        raise StageGateError("search objective requires the exact nonempty generated roster")
    counts = Counter(family for family, _ in roster)
    weights = {identity: 1 / len(counts) / counts[identity[0]] for identity in roster}
    groups, excluded = {}, []
    for observation in observations:
        if observation["state"] != "completed" or observation["kind"] != timing_kind:
            continue
        if timing_kind == ORDINARY_KIND:
            from .corpus_feedback import validate_redacted_feedback

            feedback = validate_redacted_feedback(observation["result"])
            identity = (feedback["candidate_sha256"], feedback["tuning_corpus_sha256"], feedback["certificate_sha256"])
            cells = feedback["cells"]
        elif timing_kind == "component-rtl-feedback":
            feedback = validate_component_feedback(observation["result"])
            if feedback["tier"] != "certified_component_rtl":
                raise StageGateError("search observation lacks its selected certified timing tier")
            identity = (feedback["candidate_sha256"], feedback["corpus_sha256"], feedback["configuration_sha256"])
            cells = feedback["evidence"]["cells"]
        else:
            raise StageGateError("search timing route is outside its fixed original evaluators")
        if identity != (observation["source_sha256"], corpus_sha256, configuration_sha256):
            raise StageGateError("search timing observation differs from its issued source/corpus/engine")
        identities = [(row["family"], row["capsule"]) for row in cells]
        if sorted(identities) != list(roster):
            raise StageGateError("search observation changes the complete objective roster")
        if not all(row["measured"] and row["comparable"] for row in cells):
            excluded.append({"job_id": observation["job_id"], "reason": "incomplete or incorrect RTL roster"})
            continue
        source = observation["source_sha256"]
        group = groups.setdefault(source, {"observations": [], "members": {identity: [] for identity in roster}})
        group["observations"].append(observation)
        for identity, row in zip(identities, cells, strict=True):
            group["members"][identity].append(row)
    ranked = []
    for source, group in groups.items():

        def weighted_cycles(arm, aggregate):
            return sum(
                weights[key] * aggregate(row[f"{arm}_gsim_cycles"] for row in values)
                for key, values in group["members"].items()
            )

        baseline_lo, baseline_hi = weighted_cycles("baseline", min), weighted_cycles("baseline", max)
        candidate_lo, candidate_hi = weighted_cycles("candidate", min), weighted_cycles("candidate", max)
        first = group["observations"][0]
        ranked.append(
            {
                "source_sha256": source,
                "revision_id": first["revision_id"],
                "edit_family": first["options"].get("edit_family"),
                "jobs": [row["job_id"] for row in group["observations"]],
                "ratio_lo": candidate_lo / baseline_hi,
                "ratio_hi": candidate_hi / baseline_lo,
                "baseline_weighted_cycles": {"lo": baseline_lo, "hi": baseline_hi},
                "candidate_weighted_cycles": {"lo": candidate_lo, "hi": candidate_hi},
                "priorities": measured_priorities(group["members"], weights),
            }
        )
    return {
        "ranked": sorted(ranked, key=lambda row: (row["ratio_hi"], row["source_sha256"])),
        "excluded": excluded,
        "weights_sha256": document_sha256([[*key, weights[key]] for key in roster]),
        "weight_basis": "equal generated families and equal members within family",
        "uncertainty_basis": "range of every observed repeat; no confidence or unseen-program claim",
    }


class ComponentSearchCoordinator:
    """Single promotion owner for a live pool; decisions remain development only."""

    timing_kind = "component-rtl-feedback"

    def __init__(self, pool, *, settings=SearchSettings()):
        from .evaluation_pool import ComponentEvaluationPool

        if type(pool) is not ComponentEvaluationPool or type(settings) is not SearchSettings:
            raise StageGateError("search must bind the actual fixed evaluation pool")
        settings.verify()
        if pool.inputs.rtl is None:
            raise StageGateError("search requires an explicitly selected certified timing evaluator")
        self.pool, self.settings = pool, settings
        self.roster = tuple(sorted(pool.inputs.validate()))
        self.corpus_sha256 = pool.inputs.corpus.capsules_sha256
        self.configuration_sha256 = pool.inputs.rtl.certificate.sha256
        self.timing_kind = pool.inputs.timing_kind
        self.started = time.monotonic()
        self._incumbent = pool.baseline_revision.public_record()
        self._plateau = 0
        self._finalized = False
        self.selected = None
        self.calibration, self.calibration_installations, self.calibration_wall_seconds = None, 0, None
        from .contracts import canonical_json
        from .ordinary_queue_calibration import partition

        selected_partition = partition(pool.policy) if pool.inputs.ordinary else None
        self.calibration_partition = canonical_json(selected_partition) if selected_partition is not None else None

    def admit_submission(self, *, timeout_s):
        elapsed = time.monotonic() - self.started
        if (
            self._finalized
            or type(timeout_s) not in (int, float)
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
            or elapsed + timeout_s > self.settings.soft_seconds - self.settings.final_reserve_seconds
        ):
            raise StageGateError("search reserves the remaining time for draining and final checks")
        if self.status()["stop_reason"] is not None:
            raise StageGateError("search has stopped new exploration")

    def submit(self, revision_id, *, timeout_s, **selection):
        """Keep a host-selected queued lifetime separate from its broker acknowledgment.

        Absent the explicit setting, retain the original request timeout. The
        selected lifetime includes a fixed campaign completion boundary, so
        setup and queue waiting cannot move work into the verification reserve.
        This selects no evaluator, resource grant or scientific qualification.
        """
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise StageGateError("queued submission requires its original finite acknowledgment budget")
        selected = self.settings.queued_job_timeout_seconds
        lifetime = timeout_s if selected is None else selected
        self.admit_submission(timeout_s=lifetime)
        boundary = (
            {}
            if selected is None
            else {
                "completion_deadline": self.started + self.settings.soft_seconds - self.settings.final_reserve_seconds
            }
        )
        # Recheck after the original status/owner traversal, which can consume
        # time. Pool admission and dispatch independently enforce this boundary.
        if selected is not None and time.monotonic() + lifetime > boundary["completion_deadline"]:
            raise StageGateError("search reserves the remaining time for draining and final checks")
        return self.pool.submit(revision_id, timeout_s=lifetime, **selection, **boundary)

    def status(self):
        observations = self.pool.observations()
        projection = rank_measured_contenders(
            observations,
            roster=self.roster,
            corpus_sha256=self.corpus_sha256,
            configuration_sha256=self.configuration_sha256,
            timing_kind=self.timing_kind,
        )
        ranked = projection["ranked"]
        eligible = {row["source_sha256"]: row for row in ranked}
        baseline = self.pool.baseline_revision.public_record()
        incumbent = next((row for row in ranked if row["source_sha256"] == self._incumbent["source_sha256"]), None)
        if incumbent is None or incumbent["ratio_hi"] >= 1 - self.settings.minimum_improvement:
            # Every repeat is reopened. A previously winning source can lose
            # its conservative superiority without becoming incorrect.
            self._incumbent = baseline
            incumbent = None
        incumbent_lo = incumbent["ratio_lo"] if incumbent else 1.0
        if ranked and ranked[0]["ratio_hi"] < incumbent_lo * (1 - self.settings.minimum_improvement):
            selected = self.pool.revisions.get(ranked[0]["revision_id"])
            self._incumbent = selected.public_record()
            self._plateau = 0
        # Recompute from current bounds rather than remembering old wins or
        # plateaus after later repeats have widened their measured ranges.
        processed, pending, plateau, batch_best = set(), [], 0, 1.0
        for observation in observations:
            source = observation["source_sha256"]
            if source != baseline["source_sha256"] and source not in processed and source in eligible:
                processed.add(source)
                pending.append(source)
        while len(pending) >= self.settings.beam_width:
            batch = pending[: self.settings.beam_width]
            del pending[: self.settings.beam_width]
            rows = [eligible[source] for source in batch if source in eligible]
            labels = {row["edit_family"] for row in rows if row["edit_family"]}
            best = min((row["ratio_hi"] for row in rows), default=math.inf)
            improved = best < batch_best * (1 - self.settings.minimum_improvement)
            batch_best = min(batch_best, best)
            # Labels demonstrate declared diversity only. Missing/incomplete
            # batches cannot count towards even this scoped plateau heuristic.
            plateau = 0 if improved or len(labels) < 2 else plateau + 1
        self._plateau = plateau
        elapsed = time.monotonic() - self.started
        stop = None
        if self._finalized:
            stop = "exploration finalized"
        elif elapsed >= self.settings.hard_seconds:
            stop = "hard wall budget exhausted"
        elif elapsed >= self.settings.soft_seconds - self.settings.final_reserve_seconds:
            stop = "final drain and verification reserve reached"
        elif len(observations) >= self.pool.max_jobs:
            stop = "retained evaluation job budget exhausted"
        elif self._plateau >= self.settings.plateau_batches:
            stop = "scoped plateau across completed batches with declared edit diversity"
        beam, seen, labels = [self._incumbent], {self._incumbent["source_sha256"]}, set()
        choices = [row for row in ranked if row["source_sha256"] not in seen]
        for row in choices:
            label = row["edit_family"]
            if label and label not in labels and len(beam) < self.settings.beam_width:
                beam.append(self.pool.revisions.get(row["revision_id"]).public_record())
                seen.add(row["source_sha256"])
                labels.add(label)
        for row in choices:
            if row["source_sha256"] not in seen and len(beam) < self.settings.beam_width:
                beam.append(self.pool.revisions.get(row["revision_id"]).public_record())
                seen.add(row["source_sha256"])
        screening = [
            {"revision_id": row["revision_id"], "screening": row["result"]["screening"]}
            for row in observations
            if row["state"] == "completed"
            and row["kind"] in ("component-analytical-feedback", "corpus-compile-spike-feedback")
            and "screening" in row["result"]
        ]
        priorities = next(
            (row["priorities"] for row in ranked if row["source_sha256"] == self._incumbent["source_sha256"]),
            {"status": "UNKNOWN", "reason": "incumbent has no complete measured roster"},
        )
        from .evaluation_priorities import project as project_costs

        return {
            "schema": "merlin.component_search_status.v1",
            "corpus_sha256": self.corpus_sha256,
            "settings": asdict(self.settings),
            "elapsed_seconds": elapsed,
            "incumbent": self._incumbent,
            "baseline_fallback": baseline,
            "beam": beam,
            **projection,
            "screening": screening,
            "provisional": self.provisional(observations),
            "priorities": priorities,
            "evaluation_priorities": project_costs(
                priorities,
                observations,
                source_sha256=self._incumbent["source_sha256"],
                baseline_sha256=baseline["source_sha256"],
                corpus_sha256=self.corpus_sha256,
                certificate_sha256=self.configuration_sha256,
            ),
            "plateau_batches": self._plateau,
            "stop_reason": stop,
            "promotion": "NO_FINAL_ACCEPTANCE",
            "objective_scope": (
                "complete generated roster of measured simulator cycles; complete host and model costs UNKNOWN"
            ),
            "finalized": self._finalized,
            "selected": self.selected,
            "final_requirements": "observed author lineage plus fresh complete compile/static/numerical checks",
        }

    def provisional(self, observations=None):
        from .provisional_search import project

        if self.timing_kind != ORDINARY_KIND:
            return {"status": "UNAVAILABLE", "candidates": [], "history": []}
        return project(self, self.pool.observations() if observations is None else observations)

    def resume_parent(self, revision_id):
        from .provisional_search import resume

        return resume(self, revision_id)

    def finalize(self, *, timeout_s):
        """Stop exploration and finish or reap queued work before source selection."""
        remaining = self.settings.hard_seconds - self.settings.final_reserve_seconds - (time.monotonic() - self.started)
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0 or remaining <= 0:
            raise StageGateError("search finalization cannot consume its final verification reserve")
        self.pool.drain(timeout_s=min(timeout_s, remaining))
        self._finalized = True
        return self.status()

    def select(self):
        """Restore only the measured incumbent during the observed author session."""
        if not self._finalized:
            raise StageGateError("search source selection requires completed evaluation cleanup")
        status = self.status()
        source = status["incumbent"]["source_sha256"]
        ranked = next((row for row in status["ranked"] if row["source_sha256"] == source), None)
        if ranked is None:
            raise StageGateError("search selection feedback is outside its complete current objective")
        observed = next(
            (
                row
                for row in self.pool.observations()
                if row["state"] == "completed"
                and row["kind"] == self.timing_kind
                and row["source_sha256"] == source
                and row["job_id"] in ranked["jobs"]
            ),
            None,
        )
        if observed is None:
            raise StageGateError(
                "search selection requires complete measured feedback, including its baseline fallback"
            )
        self.selected = self.pool.revisions.restore(status["incumbent"]["revision_id"], self.pool.policy.candidate)
        return {
            "schema": "merlin.component_search_selection.v1",
            "selected": self.selected,
            "feedback": observed["result"],
            "promotion": "NO_FINAL_ACCEPTANCE",
        }
