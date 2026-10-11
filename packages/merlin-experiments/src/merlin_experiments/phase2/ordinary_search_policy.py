"""Queued development feedback on the normal Phase 1 functional fork.

This adapter retains the corpus workflow's scientific and receipt owners. It
adds immutable revision dispatch, not component-runtime qualification or final
acceptance. The selected ordinary executor still performs its original fresh
Spike/numerical and certified simulator admission for every measured member.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass, replace
from functools import partial
from pathlib import Path

from merlin.perf.execution_policy import ITERATION_MAX_SECONDS
from merlin.targetgen.target_experiment import TargetExperiment

from . import campaign as PC
from . import corpus as CORPUS
from . import corpus_feedback as CF
from . import development_feedback as DF
from . import functional_inputs as FI
from . import gsim_gate as GATE
from . import measurement_support as MS
from . import paired_measurement as PME
from . import queued_search_actions as QSA
from .broker_policy import BrokerServices
from .contracts import StageGateError, canonical_json, document_sha256, exact_tree_record, mapping_file, sha256_file
from .edit_authority import FrozenEditAuthority
from .retained_revisions import RetainedRevision, executing_revisions

_ISSUER = object()
_FUNCTIONAL_FILES = (
    "environment.yaml",
    "qa_loop_summary.yaml",
    "freeze.json",
    "run_manifest.yaml",
    "grading_public/score_capsule.json",
    "grading_hidden/score_capsule.json",
    "input_bundle_manifest.yaml",
)


def _data(value):
    """Project original selected identities to JSON without arbitrary reconstruction."""
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _data(asdict(value))
    if type(value) is dict:
        return {key: _data(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_data(item) for item in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise StageGateError("ordinary search selection contains an unsupported identity value")


def _real(path, *, label, directory=True):
    path = Path(path)
    if path.resolve() != path or path.is_symlink() or not (path.is_dir() if directory else path.is_file()):
        raise StageGateError(f"ordinary search {label} is absent, linked or noncanonical")
    return path


def _phase1_target(environment, target_experiment):
    scope = environment.get("task_scope") or {}
    corpus = environment.get("public_corpus_input") or {}
    if (
        scope.get("target") != target_experiment.target
        or corpus.get("descriptor_sha256") != target_experiment.descriptor_sha256
    ):
        raise StageGateError("ordinary search target/descriptor differs from the original Phase 1 environment")


class OrdinarySearchPolicy(CF.CorpusFeedbackPolicy):
    """Live normal corpus policy selected by ``prepare_ordinary_search_policy``.

    The queue calls only ``evaluate_retained``. Structural broker actions retain
    their original separate handlers; their documents cannot become timing.
    """

    def __init__(
        self,
        *,
        functional,
        corpus,
        feedback,
        edit_authority,
        contract_root,
        candidate,
        receipt_path,
        feedback_round,
        services,
        e2e_sentinel,
        _issuer,
    ):
        if _issuer is not _ISSUER:
            raise StageGateError("ordinary search needs original normal functional and feedback admission")
        super().__init__(
            candidate=candidate,
            target_experiment=feedback.target_experiment,
            receipt_path=receipt_path,
            services=services,
            feedback_evaluator=feedback,
            feedback_round=feedback_round,
            functional_base=feedback.baseline,
            e2e_sentinel=e2e_sentinel,
        )
        self.edit_authority = edit_authority
        self._functional, self._corpus, self._selected_feedback = functional, corpus, feedback
        self._contract_root, self._issuer = contract_root, _issuer
        self.evaluation_pool, self.search_coordinator = None, None
        self._functions = (
            (DF, "development_executor", DF.development_executor),
            (DF.DevelopmentGsimFeedback, "evaluate", DF.DevelopmentGsimFeedback.evaluate),
            (PME, "run_execution", PME.run_execution),
        )
        owners = (FI, PC, CORPUS, CF, DF, GATE, MS, PME, QSA)
        self._source_pins = tuple(
            (str(path), sha256_file(path))
            for path in (Path(__file__).resolve(), *(Path(owner.__file__).resolve() for owner in owners))
        )
        self._selection = canonical_json(self._binding())
        self.validate_search_inputs()

    @property
    def search_corpus(self):
        return self._corpus

    @property
    def search_baseline(self):
        return self._selected_feedback.baseline

    def enable_evaluation_pool(self, *, revisions, resources, job_resources, root, **kwargs):
        from .evaluation_pool import ComponentEvaluationPool

        if self.evaluation_pool is not None or revisions.authority is not self.edit_authority:
            raise StageGateError("normal corpus queue requires its original edit authority once")
        self.evaluation_pool = ComponentEvaluationPool(
            policy=self, revisions=revisions, resources=resources, job_resources=job_resources, root=root, **kwargs
        )
        return self.evaluation_pool

    def build_registry(self):
        original = super().build_registry()
        return (*original, *QSA.actions()) if self.search_coordinator is not None else original

    def validate_actions(self, actions):
        super().validate_actions(actions)
        expected = {action.name: action for action in self.build_registry()}
        if set(actions) != set(expected) or any(
            actions[name].advertised() != expected[name].advertised() for name in expected
        ):
            raise StageGateError("normal search action contract differs from its original selected owners")

    def execute(self, request, action_name, rendered, call_index, timeout_s, started):
        if action_name in {action.name for action in QSA.actions()}:
            return QSA.execute(self, action_name, rendered, timeout_s, started)
        return super().execute(request, action_name, rendered, call_index, timeout_s, started)

    def _binding(self):
        feedback, functional = self._selected_feedback, self._functional
        executor = feedback.executor
        if (
            type(executor) is not partial
            or executor.func is not DF.development_executor
            or executor.args
            or executor.keywords != {"contract_root": self._contract_root}
        ):
            raise StageGateError("ordinary search requires the original fixed development executor")
        return {
            "schema": "merlin.ordinary_corpus_search_binding.v1",
            "workflow_id": self.workflow_id,
            "scope": "frozen_development_corpus_simulator_cycles",
            "functional_origin": _data(functional),
            "functional_files": [
                {"path": str(functional.run_dir / name), "sha256": sha256_file(functional.run_dir / name)}
                for name in _FUNCTIONAL_FILES
            ],
            "baseline": {
                "path": str(feedback.baseline),
                "sha256": feedback.baseline_sha256,
                "tree": exact_tree_record(feedback.baseline),
            },
            "candidate": str(self.candidate),
            "corpus": _data(self._corpus),
            "target": _data(self.target_experiment),
            "target_descriptor_sha256": sha256_file(self.target_experiment.path),
            "certificate": _data(feedback.certificate),
            "rtl_identity": _data(feedback.rtl_identity),
            "rtl_facts_path": str(feedback.rtl_facts_path),
            "decisions": [
                {"family": identity[0], "capsule": identity[1], "decision": decision.to_dict()}
                for identity, decision in sorted(feedback.decisions.items())
            ],
            "contract": {"path": str(self._contract_root), "tree": exact_tree_record(self._contract_root)},
            "edit_authority": _data(self.edit_authority.binding),
            "feedback_configuration": {
                "work_root": str(feedback.work_root),
                "round": self.feedback_round,
                "executed_profile_timeout_s": feedback.executed_profile_timeout_s,
                "machine_bounds": _data(feedback.machine_bounds),
                "seed_points": _data(feedback.seed_points),
                "member_cost": _data(dict(feedback.member_cost)),
                "tuning_call_budget": feedback.tuning_call_budget,
            },
            "runtime_source_pins": [{"path": path, "sha256": digest} for path, digest in self._source_pins],
        }

    def validate_search_inputs(self):
        """Reopen original admission pins without reading the author's mutable bytes."""
        if (
            self._issuer is not _ISSUER
            or type(self._selected_feedback) is not DF.DevelopmentGsimFeedback
            or self.feedback_evaluator is not self._selected_feedback
            or self.functional_base != self.search_baseline
            or self.target_experiment is not self._selected_feedback.target_experiment
            or self._selected_feedback.corpus is not self._corpus
            or type(self.edit_authority) is not FrozenEditAuthority
            or self.edit_authority.initial_source != self.candidate
        ):
            raise StageGateError("ordinary search original owner selection changed")
        self.edit_authority.check_integrity()
        for owner, name, function in self._functions:
            if getattr(owner, name) is not function:
                raise StageGateError("ordinary search original execution owner changed")
        for path, digest in self._source_pins:
            if sha256_file(Path(path)) != digest:
                raise StageGateError("ordinary search original runtime source changed")
        CORPUS.verify_frozen_performance_corpus(self._corpus)
        roster = {(member.family, member.capsule) for member in self._corpus.capsules}
        if not roster or len(roster) != len(self._corpus.capsules) or roster != set(self._selected_feedback.decisions):
            raise StageGateError("ordinary search requires the exact nonempty frozen member roster")
        for member in self._corpus.capsules:
            if member.descriptor != mapping_file(member.source_dir / "capsule.yaml", yaml_file=True):
                raise StageGateError("ordinary search frozen member descriptor changed")
        if sha256_file(self.target_experiment.path) != self.target_experiment.descriptor_sha256:
            raise StageGateError("ordinary search target differs from the descriptor loaded by Phase 1")
        FI.verify_functional_host_lane_snapshot(self._functional.model_host_lane_snapshot)
        environment = mapping_file(self._functional.run_dir / "environment.yaml", yaml_file=True)
        _phase1_target(environment, self.target_experiment)
        PC.verify_private_input_snapshot(self._functional.run_dir, environment)
        if not PC.check_fork(PC.functional_fork(self._functional), self.search_baseline).ok:
            raise StageGateError("ordinary search baseline no longer matches its original functional fork")
        loaded = GATE.load_certificate(
            self._selected_feedback.certificate.path, expected_sha256=self._selected_feedback.certificate.sha256
        )
        if _data(loaded) != _data(self._selected_feedback.certificate):
            raise StageGateError("ordinary search certificate selection changed")
        if (
            MS.load_rtl_identity(self._selected_feedback.rtl_facts_path, self.target_experiment.target)
            != self._selected_feedback.rtl_identity
        ):
            raise StageGateError("ordinary search original elaborated RTL identity changed")
        if canonical_json(self._binding()) != self._selection:
            raise StageGateError("ordinary search original functional/corpus/target/executor binding changed")
        return roster

    def binding(self):
        """Private closed selection data; no new semantic or physical authority."""
        self.validate_search_inputs()
        return json.loads(self._selection)

    @property
    def search_closure_sha256(self):
        return document_sha256(self.binding())

    def evaluate_retained(self, revision, baseline_revision, *, work_root, call_index, timeout_s, workers=1):
        """Run the original complete tuning roster from issued immutable sources."""
        if (
            type(revision) is not RetainedRevision
            or type(baseline_revision) is not RetainedRevision
            or type(call_index) is not int
            or call_index < 0
            or type(timeout_s) not in (int, float)
            or not 1 <= timeout_s <= ITERATION_MAX_SECONDS
            or type(workers) is not int
            or workers < 1
        ):
            raise StageGateError("ordinary queued feedback needs issued revisions and a reduced-witness deadline")
        work_root = Path(work_root)
        if work_root.resolve() != work_root or work_root.exists() or work_root.is_symlink():
            raise StageGateError("ordinary queued feedback requires a fresh canonical private work root")
        if any(
            work_root.is_relative_to(path) or path.is_relative_to(work_root)
            for path in (self.candidate, self.search_baseline, self.search_corpus.root, self.edit_authority.output)
        ):
            raise StageGateError("ordinary queued feedback output overlaps its original input owners")
        self.validate_search_inputs()
        revision.verify(authority=self.edit_authority)
        baseline_revision.verify(authority=self.edit_authority)
        if baseline_revision.source_sha256 != self._selected_feedback.baseline_sha256:
            raise StageGateError("ordinary queued feedback baseline differs from its Phase 1 functional fork")
        if DF.sweep_workers() != workers:
            raise StageGateError("ordinary retained feedback differs from its explicit member worker grant")
        deadline = time.monotonic() + timeout_s
        feedback = replace(
            self._selected_feedback,
            baseline=baseline_revision.source,
            work_root=work_root,
            _baseline_cache=None,
            _profile_baseline_cache=None,
            _whole_model_baseline=None,
            _prefetched={},
            _totals=None,
            _spend=None,
            _achievable_points=self._selected_feedback.seed_points,
        )
        with executing_revisions(revision, baseline_revision):
            document = CF.validate_redacted_feedback(
                feedback.evaluate(
                    revision.source,
                    round_index=self.feedback_round,
                    call_index=call_index,
                    timeout_s=int(timeout_s),
                    retained_revision=revision,
                    complete_roster=True,
                )
            )
        revision.verify(authority=self.edit_authority)
        baseline_revision.verify(authority=self.edit_authority)
        roster = self.validate_search_inputs()
        if time.monotonic() > deadline:
            raise StageGateError("ordinary queued feedback exceeded its original deadline")
        if (
            {(row["family"], row["capsule"]) for row in document["cells"]} != roster
            or document["candidate_sha256"] != revision.source_sha256
            or document["tuning_corpus_sha256"] != self.search_corpus.capsules_sha256
            or document["certificate_sha256"] != self._selected_feedback.certificate.sha256
        ):
            raise StageGateError("ordinary queued feedback differs from its exact original roster or identity")
        return document


def prepare_ordinary_search_policy(
    *,
    functional_runs_root,
    functional_run_id,
    functional_submission_sha256,
    corpus,
    baseline,
    candidate,
    target_experiment,
    receipt_path,
    work_root,
    edit_authority,
    contract_root,
    certificate_path,
    certificate_sha256,
    rtl_facts_path,
    feedback_round=0,
    tuning_call_budget=None,
    waive_functional_gate=(),
    services=BrokerServices(),
    e2e_sentinel=None,
    whole_model_inputs=None,
):
    """Admit the same Phase 1 functional fork and ordinary corpus feedback route.

    No executable callback, backend name, imported provider or saved verdict can
    substitute for either original admission owner.
    """
    if (
        type(corpus) is not CORPUS.FrozenPerformanceCorpus
        or type(target_experiment) is not TargetExperiment
        or type(edit_authority) is not FrozenEditAuthority
        or not edit_authority.configured
        or type(feedback_round) is not int
        or feedback_round < 0
    ):
        raise StageGateError("ordinary search requires normal frozen corpus, target and edit authority")
    candidate = _real(candidate, label="candidate")
    baseline = _real(baseline, label="functional baseline")
    contract_root = _real(contract_root, label="contract resources")
    edit_authority.check_integrity()
    if (
        edit_authority.initial_source != candidate
        or edit_authority.binding["initial_candidate_sha256"] != functional_submission_sha256
    ):
        raise StageGateError("ordinary search edit authority must start from the selected Phase 1 functional compiler")
    functional = FI.inspect_stage_functional_run(
        functional_runs_root,
        functional_run_id,
        functional_submission_sha256,
        waive=frozenset(waive_functional_gate),
    )
    _phase1_target(mapping_file(functional.run_dir / "environment.yaml", yaml_file=True), target_experiment)
    if not PC.check_fork(PC.functional_fork(functional), baseline).ok:
        raise StageGateError("ordinary search baseline is not the original admitted functional fork")
    FI._require_read_only_tree(baseline, label="ordinary search functional baseline")
    CORPUS.verify_frozen_performance_corpus(corpus)
    feedback = DF.prepare_development_feedback(
        contract_root=contract_root,
        certificate_path=certificate_path,
        certificate_sha256=certificate_sha256,
        rtl_facts_path=rtl_facts_path,
        corpus=corpus,
        baseline=baseline,
        baseline_sha256=functional.digest,
        target_experiment=target_experiment,
        work_root=Path(work_root),
        tuning_call_budget=tuning_call_budget,
        functional_run_dir=functional.run_dir,
        whole_model_inputs=whole_model_inputs,
    )
    return OrdinarySearchPolicy(
        functional=functional,
        corpus=corpus,
        feedback=feedback,
        edit_authority=edit_authority,
        contract_root=contract_root,
        candidate=candidate,
        receipt_path=Path(receipt_path),
        feedback_round=feedback_round,
        services=services,
        e2e_sentinel=e2e_sentinel,
        _issuer=_ISSUER,
    )
