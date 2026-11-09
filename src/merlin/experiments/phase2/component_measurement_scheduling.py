"""Live, explicitly selected development measurement deferral.

The fixed analytical producer and existing independent measurement issuer own
all predictions and applicability. Saved reports and these data classes cannot
issue a plan. This owner neither fits a model nor changes correctness/held/final
rosters. Unqualified current runtimes remain unavailable.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

from merlin.benchharness import hash_tree
from merlin.perf.component_cost import complete_component_cost
from merlin.perf.component_measurement_plan import plan_development_measurements
from merlin.perf.component_screen import ComponentOpportunity
from merlin.perf.execution_policy import ITERATION_MAX_SECONDS

from .component_applicability import applicability_observation
from .component_feature_inputs import plain
from .component_runtime import require_independent_runtime
from .contracts import StageGateError, document_sha256, exact_tree_record, sha256_file, write_json

_ISSUED = WeakKeyDictionary()


def _live(binding, evaluator):
    from .component_analytical import ComponentAnalyticalBinding, _independent_calibration
    from .component_measurement_qualification import IndependentMeasurementQualification
    from .development_feedback import DevelopmentGsimFeedback

    if type(binding) is not ComponentAnalyticalBinding or type(evaluator) is not DevelopmentGsimFeedback:
        raise StageGateError("development scheduling requires the fixed analytical and certified evaluator owners")
    runtime = require_independent_runtime(
        binding.independent_runtime,
        required_roles=("feature_provider", "rtl_executor"),
        target_descriptor=binding.target_descriptor,
    )
    qualification = runtime.qualification
    if type(qualification) is not IndependentMeasurementQualification:
        raise StageGateError("development scheduling lacks independent physical and held qualification")
    calibration = _independent_calibration(binding.calibration_adapter, runtime, binding.qualification)
    domains = binding.validate(calibration)
    if (
        evaluator.corpus is not binding.corpus
        or evaluator.baseline != binding.baseline
        or evaluator.baseline_sha256 != binding.baseline_sha256
        or evaluator.target_experiment.path != binding.target_descriptor
        or evaluator.executor is not runtime.services.rtl_executor
    ):
        raise StageGateError("development scheduling selected different original inputs or qualified execution")
    return qualification, calibration, domains


def _configuration(evaluator):
    from . import gsim_gate as G
    from .component_runtime_authority import _callback_identity
    from .paired_measurement import gsim_workload

    selected = evaluator.certificate
    current = G.load_certificate(selected.path, expected_sha256=selected.sha256)
    if current.target != evaluator.target_experiment.target:
        raise StageGateError("development measurement certificate targets another selection")
    if set(evaluator.decisions) != {(member.family, member.capsule) for member in evaluator.corpus.capsules}:
        raise StageGateError("development measurement certified membership is incomplete")
    for member in evaluator.corpus.capsules:
        expected = G.plan_evaluation(
            current,
            gsim_workload(member),
            phase="development_correctness",
            gsim_available=True,
        )
        if (
            not expected.admitted
            or not expected.use_gsim
            or (expected.to_dict() != evaluator.decisions[member.family, member.capsule].to_dict())
        ):
            raise StageGateError("development measurement lacks an original exact certified workload")
    identity = _callback_identity(evaluator.executor)
    return document_sha256(
        {
            "certificate_sha256": current.sha256,
            "target_sha256": sha256_file(evaluator.target_experiment.path),
            "executor": [identity[2], identity[3]],
            "rtl_identity": evaluator.rtl_identity,
            "work_root": str(plain(evaluator.work_root)),
            "decisions": [[list(key), value.to_dict()] for key, value in sorted(evaluator.decisions.items())],
        }
    )


def _replay(binding, evaluator, observations, applicability, seconds, *, candidate_sha256, max_measurements):
    from .component_analytical import _verify_artifacts

    qualification, calibration, domains = _live(binding, evaluator)
    opportunities, status = [], {}
    for member, pair, derived in zip(binding.corpus.capsules, observations, applicability, strict=True):
        if type(pair) is not tuple or len(pair) != 2 or pair[0].inputs_sha256 != pair[1].inputs_sha256:
            raise StageGateError("development measurement original arm/input membership changed")
        costs, in_domain = [], True
        for feature, coordinate, compiler_sha256 in zip(
            pair, derived, (binding.baseline_sha256, candidate_sha256), strict=True
        ):
            _verify_artifacts(feature)
            if (
                feature.compiler_sha256,
                feature.member_sha256,
                feature.corpus_sha256,
                feature.target_sha256,
                feature.scope_sha256,
            ) != (
                compiler_sha256,
                member.source_sha256,
                binding.corpus.capsules_sha256,
                binding.target_sha256,
                binding.scope.sha256,
            ):
                raise StageGateError("development measurement feature source/member/context changed")
            lookup, _pins = applicability_observation(
                observation=coordinate,
                features=feature,
                domain=qualification.applicability_domain,
                runtime=qualification.functional_runtime,
                owner=coordinate.product.parent,
            )
            totals, _report = complete_component_cost(
                feature,
                calibration,
                scope=binding.scope,
                qualified_domains=domains,
                applicability_domain=qualification.applicability_domain,
                applicability_coordinates=coordinate.coordinates,
            )
            costs.append(totals[binding.objective])
            in_domain &= lookup["status"] == "IN_DOMAIN"
        ident = document_sha256([member.family, member.capsule])
        status[ident] = "IN_DOMAIN" if in_domain else "UNKNOWN"

        def paired_gate(field):
            values = [getattr(feature, field) for feature in pair]
            return (
                True if values == ["PASS", "PASS"] else False if any(v in ("FAIL", "REFUSAL") for v in values) else None
            )

        opportunities.append(
            ComponentOpportunity(
                ident,
                member.family,
                binding.objective,
                *costs,
                seconds[ident],
                paired_gate("legality_status"),
                paired_gate("functional_status"),
            )
        )
    return plan_development_measurements(
        tuple(opportunities),
        applicability=status,
        owner_available=True,
        purpose="development_performance",
        max_measurements=max_measurements,
    )


@dataclass(frozen=True, eq=False)
class DevelopmentMeasurementPlan:
    binding: object
    evaluator: object
    candidate_sha256: str
    configuration_sha256: str
    observations: tuple
    applicability: tuple
    seconds: dict
    decisions: tuple
    selected: tuple[str, ...]
    max_measurements: int
    member_timeout_s: int
    record_path: Path
    record_sha256: str
    source_pins: tuple[tuple[Path, str], ...]

    def _identity(self):
        return document_sha256(
            {
                "candidate_sha256": self.candidate_sha256,
                "configuration_sha256": self.configuration_sha256,
                "binding_sha256": self.binding.sha256,
                "observations": [[asdict(feature) for feature in pair] for pair in self.observations],
                "applicability": [[coordinate.product_sha256 for coordinate in pair] for pair in self.applicability],
                "seconds": self.seconds,
                "decisions": [asdict(row) for row in self.decisions],
                "selected": self.selected,
                "max_measurements": self.max_measurements,
                "member_timeout_s": self.member_timeout_s,
                "record_path": str(self.record_path),
                "record_sha256": self.record_sha256,
                "source_pins": [(str(path), digest) for path, digest in self.source_pins],
            }
        )

    def verify(self, evaluator, *, candidate=None, package=None, arm=None, member=None):
        issued = _ISSUED.get(self)
        if issued is None:
            raise StageGateError("development measurement plan was not issued by the live qualified producer")
        if evaluator is not self.evaluator or issued != (self._identity(), id(self.binding), id(self.evaluator)):
            raise StageGateError("development measurement plan or executor selection changed")
        if sha256_file(plain(self.record_path)) != self.record_sha256:
            raise StageGateError("development measurement plan product changed")
        for path, digest in self.source_pins:
            if sha256_file(plain(path)) != digest:
                raise StageGateError("development measurement source or tool bytes changed")
        if _configuration(evaluator) != self.configuration_sha256:
            raise StageGateError("development measurement context changed")
        decisions, selected = _replay(
            self.binding,
            evaluator,
            self.observations,
            self.applicability,
            self.seconds,
            candidate_sha256=self.candidate_sha256,
            max_measurements=self.max_measurements,
        )
        if (decisions, selected) != (self.decisions, self.selected):
            raise StageGateError("development measurement applicability or predictions changed")
        if candidate is not None:
            exact_tree_record(plain(candidate))
            if str(hash_tree(candidate)["sha256"]) != self.candidate_sha256:
                raise StageGateError("development measurement candidate bytes changed")
        if package is not None:
            if arm not in ("baseline", "candidate"):
                raise StageGateError("development measurement arm is unsupported")
            expected = self.binding.baseline_sha256 if arm == "baseline" else self.candidate_sha256
            exact_tree_record(plain(package))
            if str(hash_tree(package)["sha256"]) != expected:
                raise StageGateError("development measurement consumed a different compiler arm")
        if member is not None and (
            not any(member is original for original in self.binding.corpus.capsules)
            or document_sha256([member.family, member.capsule]) not in self.selected
        ):
            raise StageGateError("development measurement member is outside its exact selected original roster")
        return issued

    def reason(self, member):
        ident = document_sha256([member.family, member.capsule])
        return next(row.reason for row in self.decisions if row.id == ident)


def prepare_development_measurements(
    *,
    binding,
    evaluator,
    candidate,
    evidence_root,
    timeout_s,
    max_measurements,
    member_timeout_s,
):
    """Execute the fixed qualified feature producer, then issue its live plan.

    Feature extraction may itself need execution until an independently
    qualified compile-only provider exists. This function never silently turns
    static site counts into physical features or measured predictions.
    """
    from merlin.perf import component_cost as CC
    from merlin.perf import component_measurement_plan as MP
    from merlin.perf import component_screen as CS

    from . import component_analytical as A
    from . import component_applicability as CA
    from . import component_measurement_scheduling as owner
    from . import development_feedback as D
    from .component_workflow import _corpus

    if (
        type(timeout_s) not in (int, float)
        or not math.isfinite(timeout_s)
        or not 0 < timeout_s <= ITERATION_MAX_SECONDS
        or type(member_timeout_s) is not int
        or not 0 < member_timeout_s <= ITERATION_MAX_SECONDS
        or type(max_measurements) is not int
        or not 0 < max_measurements <= 10000
    ):
        raise StageGateError("development measurements require explicit bounded native and preparation budgets")
    qualification, _calibration, _domains = _live(binding, evaluator)
    if not 0 < len(binding.corpus.capsules) <= 10000:
        raise StageGateError("development measurement complete member roster exceeds its bound")
    candidate, root = plain(candidate), plain(evidence_root)
    if root.exists() or any(
        root.is_relative_to(path) or path.is_relative_to(root)
        for path in (candidate, binding.baseline, binding.corpus.root)
    ):
        raise StageGateError("development measurement preparation requires a fresh disjoint private owner")
    exact_tree_record(candidate)
    candidate_sha = str(hash_tree(candidate)["sha256"])
    _corpus(binding.corpus)
    root.mkdir(parents=True, mode=0o700)
    results = A._evaluate(
        binding,
        binding.calibration_adapter,
        candidate=candidate,
        corpus=binding.corpus,
        timeout_s=timeout_s,
    )
    if type(results) is not A.ComponentAnalyticalResults or results.observations is None:
        raise StageGateError("development measurement preparation has no actual fixed feature products")
    observations, coordinates, seconds = [], [], {}
    timing = {row["id"]: row["evaluation_seconds"] for row in results.screening["evidence"]}
    for index, member in enumerate(binding.corpus.capsules):
        pair = results.observations[member.family, member.capsule]
        observed = []
        for arm, feature in enumerate(pair):
            workspace = root / "applicability" / str(index) / str(arm)
            observed.append(qualification.context.observe_component_applicability(feature, workspace))
        observations.append(pair)
        coordinates.append(tuple(observed))
        ident = document_sha256([member.family, member.capsule])
        seconds[ident] = timing[ident]
    observations, coordinates = tuple(observations), tuple(coordinates)
    decisions, selected = _replay(
        binding,
        evaluator,
        observations,
        coordinates,
        seconds,
        candidate_sha256=candidate_sha,
        max_measurements=max_measurements,
    )
    _corpus(binding.corpus)
    if str(hash_tree(candidate)["sha256"]) != candidate_sha:
        raise StageGateError("development measurement preparation changed its original candidate")
    source_pins = tuple(
        sorted(
            set(
                binding.independent_runtime.source_pins
                + tuple(
                    (Path(module.__file__).resolve(), sha256_file(Path(module.__file__).resolve()))
                    for module in (A, CA, D, CC, MP, CS, owner)
                )
            )
        )
    )
    record = root / "development_measurement_plan.json"
    configuration = _configuration(evaluator)
    write_json(
        record,
        {
            "schema": "merlin.development_measurement_plan.v1",
            "purpose": "development_performance",
            "candidate_sha256": candidate_sha,
            "configuration_sha256": configuration,
            "binding_sha256": binding.sha256,
            "decisions": [asdict(row) for row in decisions],
            "selected": selected,
            "max_measurements": max_measurements,
            "member_timeout_s": member_timeout_s,
            "promotion": "NONE",
            "measured": False,
        },
    )
    plan = DevelopmentMeasurementPlan(
        binding,
        evaluator,
        candidate_sha,
        configuration,
        observations,
        coordinates,
        seconds,
        decisions,
        selected,
        max_measurements,
        member_timeout_s,
        record,
        sha256_file(record),
        source_pins,
    )
    _ISSUED[plan] = (plan._identity(), id(binding), id(evaluator))
    plan.verify(evaluator, candidate=candidate)
    return plan
