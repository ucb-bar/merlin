"""Original-host public source diagnostics, separate from formal Phase 2 admission.

This owner selects the existing retained, boxed, checked-source L2 route only.
It issues no certificate, timing, placement, incumbent or complete-cohort claim.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import math
import sys
import weakref
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from time import monotonic

from merlin.common.cpu_pool import shared_compute_pool
from merlin.targetgen.common_transforms import CommonTransformService
from merlin.targetgen.compiler_library import CompilerLibrarySelection
from merlin.targetgen.compiler_pass_selection import CompilerPassSelection
from merlin.targetgen.target_experiment import TargetExperiment

from . import campaign as CAMPAIGN
from . import compiler_libraries as LIBRARIES
from . import corpus as CORPUS
from . import ordinary_checked_source as CHECKED
from . import ordinary_fast_feedback as FAST
from . import public_diagnostic_corpus as PUBLIC_CORPUS
from .contracts import StageGateError, canonical_json, document_sha256, sha256_file
from .edit_authority import FrozenEditAuthority
from .retained_revisions import RetainedRevision, source_inventory

SCHEMA = "merlin.public_source_diagnostic_policy.v1"
FEEDBACK_SCHEMA = "merlin.public_source_diagnostic_feedback.v1"
_ISSUER = object()
_ISSUED = weakref.WeakKeyDictionary()


def _source(path):
    path = Path(path)
    if path.is_symlink() or path.resolve(strict=True) != path or not path.is_file():
        raise StageGateError("public diagnostics require original ordinary source leaves")
    return {"path": str(path), "sha256": sha256_file(path)}


def _portable(value):
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _portable(asdict(value))
    if type(value) in (tuple, list):
        return [_portable(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _portable(item) for key, item in value.items()}
    if value is None or type(value) in (str, bool, int) or type(value) is float and math.isfinite(value):
        return value
    raise StageGateError("public diagnostic target declaration is not closed data")


def _verify_corpus(corpus):
    if type(corpus) is PUBLIC_CORPUS.FrozenPublicDiagnosticCorpus:
        corpus.verify()
    elif type(corpus) is CORPUS.FrozenPerformanceCorpus:
        CORPUS.verify_frozen_performance_corpus(corpus)
    else:
        raise StageGateError("public diagnostics require their exact original frozen source owner")


def _capsule(corpus, member, contract):
    if type(corpus) is PUBLIC_CORPUS.FrozenPublicDiagnosticCorpus:
        return corpus.execution_capsule(member, contract=contract)
    return FAST.CR.load_capsule(member.source_dir, contract=str(contract))


def _capsules(corpus, contract):
    if type(corpus) is PUBLIC_CORPUS.FrozenPublicDiagnosticCorpus:
        return corpus.execution_capsules(contract=contract)
    return (_capsule(corpus, member, contract) for member in corpus.capsules)


def _inventory(path, *, readonly=False):
    return source_inventory(path, max_bytes=32 << 20, max_files=8192, readonly=readonly)


@dataclass(frozen=True)
class PublicDiagnosticExecution:
    """The actual original profile service, never a synthetic timing evaluator."""

    _compiler_profile_service: CommonTransformService
    work_root: Path
    rtl_facts_path: Path


class PublicSourceDiagnosticPolicy:
    def __init__(self, *, _issuer, **inputs):
        if _issuer is not _ISSUER:
            raise StageGateError("public diagnostics require original host issuance")
        self.__dict__.update(inputs)
        self._ordinary_screen = None
        self._selected_owners = self._owners()
        from merlin_experiments.phase1.feedback import compiler_profile_runtime as PROFILE

        self._functions = tuple(
            (owner, name, getattr(owner, name))
            for owner, name in (
                (FAST, "evaluate_retained"),
                (FAST.CR, "load_capsule"),
                (FAST.CR, "run_capsule"),
                (CAMPAIGN, "package_sandbox_policy"),
                (CAMPAIGN, "boxed_entrypoints"),
                (FAST, "validate_feedback"),
                (CHECKED, "record"),
                (CHECKED, "for_capsule"),
                (CORPUS, "verify_frozen_performance_corpus"),
                (CommonTransformService, "_verify"),
                (PublicSourceDiagnosticPolicy, "_binding"),
                (PublicSourceDiagnosticPolicy, "validate_search_inputs"),
                (PublicSourceDiagnosticPolicy, "binding"),
                (PublicSourceDiagnosticPolicy, "compiler_execution"),
                (PublicSourceDiagnosticPolicy, "evaluate_retained"),
                (PublicSourceDiagnosticPolicy, "load_capsule"),
                (PUBLIC_CORPUS.FrozenPublicDiagnosticCorpus, "verify"),
                (PUBLIC_CORPUS.FrozenPublicDiagnosticCorpus, "execution_capsule"),
                (PUBLIC_CORPUS.FrozenPublicDiagnosticCorpus, "execution_capsules"),
                (PROFILE, "original_compiler_execution"),
            )
        )
        self._source_pins = tuple(
            _source(Path(owner.__file__).resolve()) for owner in (FAST, CHECKED, CORPUS, PUBLIC_CORPUS, PROFILE)
        ) + (
            _source(Path(__file__).resolve()),
            _source(Path(inspect.getsourcefile(CommonTransformService)).resolve()),
        )
        self._selection = canonical_json(self._binding())
        _ISSUED[self] = self._selection
        try:
            self.validate_search_inputs()
        except BaseException:
            del _ISSUED[self]
            raise

    def _owners(self):
        return (
            self.feedback_evaluator._compiler_profile_service,
            self.target_experiment,
            self.edit_authority,
            self._corpus,
            self._compiler_library_selection,
            self._compiler_pass_selection,
            self._checked_source_diagnostic,
        )

    @property
    def search_corpus(self):
        return self._corpus

    @property
    def search_baseline(self):
        return self._baseline

    def _binding(self, *, deadline=None):
        service = self.feedback_evaluator._compiler_profile_service
        service._verify()
        self.edit_authority.check_integrity()
        _verify_corpus(self._corpus)
        library = self._compiler_library_selection
        if _source(self.target_experiment.path)["sha256"] != self.target_experiment.descriptor_sha256:
            raise StageGateError("public diagnostic target changed its selected descriptor")
        if self._compiler_pass_selection.binding.library != library.contract.sha256:
            raise StageGateError("public diagnostics changed the selected compiler library release")
        baseline = _inventory(self._baseline, readonly=True)
        if baseline != self._binding_baseline:
            raise StageGateError("public diagnostic original baseline binding changed")
        return {
            "schema": SCHEMA,
            "formal_admission": False,
            "promotion_authority": "NONE",
            "timing": "UNKNOWN",
            "target": _source(self.target_experiment.path),
            "target_name": self.target_experiment.target,
            "target_declaration": _portable(self.target_experiment),
            "execution_root": str(self.feedback_evaluator.work_root),
            "rtl_facts": _source(self.feedback_evaluator.rtl_facts_path),
            "contract": _inventory(self._contract_root),
            "corpus": [self._corpus.manifest_sha256, self._corpus.capsules_sha256],
            "public_execution_views": (
                [cap["public_diagnostic_execution_view"] for cap in _capsules(self._corpus, self._contract_root)]
                if type(self._corpus) is PUBLIC_CORPUS.FrozenPublicDiagnosticCorpus
                else None
            ),
            "baseline": baseline,
            "initial_seed": _inventory(self.edit_authority.seed, readonly=True),
            "edit_authority": self.edit_authority.binding,
            "profile": {
                "archive": _source(service.bundle_manifest),
                "profile_sha256": service.profile["sha256"],
                "snapshot_sha256": service.snapshot_digest,
                "workspace": str(service.workspace),
                "repo": str(service.repo),
            },
            "compiler_library": library.record(),
            "compiler_pass_selection": self._compiler_pass_selection.record(),
            "checked_source_diagnostic": CHECKED.record(self._checked_source_diagnostic, self._compiler_pass_selection),
            "runtime_source_pins": self._source_pins,
        }

    def validate_search_inputs(self, *, deadline=None):
        if type(self) is not PublicSourceDiagnosticPolicy or self not in _ISSUED:
            raise StageGateError("public diagnostic policy was not issued by its original host")
        if (
            self._ordinary_screen is not None
            or any(
                actual is not original for actual, original in zip(self._owners(), self._selected_owners, strict=True)
            )
            or type(self.feedback_evaluator) is not PublicDiagnosticExecution
            or type(self.feedback_evaluator._compiler_profile_service) is not CommonTransformService
            or type(self._compiler_library_selection) is not CompilerLibrarySelection
            or type(self._compiler_pass_selection) is not CompilerPassSelection
            or type(self.target_experiment) is not TargetExperiment
            or type(self._corpus) not in (CORPUS.FrozenPerformanceCorpus, PUBLIC_CORPUS.FrozenPublicDiagnosticCorpus)
            or type(self.edit_authority) is not FrozenEditAuthority
            or self.edit_authority.initial_source != self.candidate
            or any(getattr(owner, name) is not function for owner, name, function in self._functions)
            or any(_source(row["path"]) != row for row in self._source_pins)
        ):
            raise StageGateError("public diagnostic original execution owner or source changed")
        limit = CHECKED._deadline(deadline)
        with shared_compute_pool(timeout_s=CHECKED._queue_wait(limit)):
            current = canonical_json(self._binding(deadline=limit))
        CHECKED._completed(limit)
        if current != self._selection or current != _ISSUED[self]:
            raise StageGateError("public diagnostic original profile/source/corpus selection changed")
        return {(row.family, row.capsule) for row in self._corpus.capsules}

    def binding(self):
        self.validate_search_inputs()
        return json.loads(self._selection)

    def load_capsule(self, member, *, deadline):
        limit = CHECKED._deadline(deadline)
        with shared_compute_pool(timeout_s=CHECKED._queue_wait(limit)):
            result = _capsule(self._corpus, member, self._contract_root)
        CHECKED._completed(limit)
        return result

    @property
    def search_closure_sha256(self):
        return document_sha256(self.binding())

    @contextlib.contextmanager
    def compiler_execution(self, package, workspace, *, deadline=None):
        from merlin_experiments.phase1.feedback.compiler_profile_runtime import original_compiler_execution

        self.validate_search_inputs(deadline=deadline)
        execution = original_compiler_execution(
            self.feedback_evaluator._compiler_profile_service,
            target_experiment=self.target_experiment,
            candidate=package,
            work_root=workspace / "original-profile",
        )
        # Setup/teardown are metadata work. The package and native child own
        # their leaf grants; never retain this parent lease while waiting.
        entered = False
        try:
            limit = CHECKED._deadline(deadline)
            with shared_compute_pool(timeout_s=CHECKED._queue_wait(limit)):
                execution.__enter__()
                entered = True
            CHECKED._completed(limit)
            yield
        finally:
            error = sys.exc_info()
            if entered:
                # Cleanup retains its own bounded admission even after the
                # observation deadline. The existing guardian owns hard stop
                # and descendant cleanup; late metadata never yields feedback.
                with shared_compute_pool(timeout_s=120):
                    execution.__exit__(*error)
            self.validate_search_inputs(deadline=deadline)

    def publish_completed_artifacts(self, job_root):
        """Keep a cleaned diagnostic unscored and outside reusable build caches."""
        return {"status": "UNAVAILABLE", "published": 0}

    def discard_cleaned_artifacts(self, job_root):
        """No cache ownership was issued; the queue retains its cleanup receipt."""
        return {"status": "UNAVAILABLE", "discarded": 0}

    @LIBRARIES.scoped_policy
    def evaluate_retained(
        self,
        revision,
        baseline_revision,
        *,
        work_root,
        call_index,
        timeout_s,
        workers=1,
        kind=FAST.SELECTED_KIND,
        selected_members=(),
        **unsupported,
    ):
        if (
            unsupported
            or kind != FAST.SELECTED_KIND
            or type(timeout_s) not in (int, float)
            or not math.isfinite(timeout_s)
            or not 1 <= timeout_s <= 900
            or type(workers) is not int
            or not 1 <= workers <= 2
            or type(revision) is not RetainedRevision
            or type(baseline_revision) is not RetainedRevision
        ):
            raise StageGateError("public diagnostics allow only bounded selected compile/Spike feedback")
        deadline = monotonic() + timeout_s
        self.validate_search_inputs(deadline=deadline)
        root = Path(work_root)
        if (
            root.exists()
            or root.is_symlink()
            or root.resolve() != root
            or not root.is_relative_to(self.feedback_evaluator.work_root)
        ):
            raise StageGateError("public diagnostics require fresh output under their original private root")
        revision.verify(authority=self.edit_authority)
        baseline_revision.verify(authority=self.edit_authority)
        if baseline_revision.inventory != self._binding_baseline:
            raise StageGateError("public diagnostic baseline differs from its original seed")
        return FAST.evaluate_retained(
            self,
            revision,
            baseline_revision,
            work_root=work_root,
            call_index=call_index,
            timeout_s=max(0, deadline - monotonic()),
            workers=workers,
            selected_members=selected_members,
        )


def prepare_public_source_diagnostic(
    *,
    candidate,
    baseline,
    corpus,
    target_experiment,
    edit_authority,
    service,
    compiler_library_selection,
    compiler_pass_selection,
    checked_source_diagnostic,
    contract_root,
    work_root,
):
    """Issue public development execution only; the formal constructor is unchanged."""
    if (
        type(corpus) not in (CORPUS.FrozenPerformanceCorpus, PUBLIC_CORPUS.FrozenPublicDiagnosticCorpus)
        or not corpus.capsules
        or type(target_experiment) is not TargetExperiment
        or type(edit_authority) is not FrozenEditAuthority
        or not edit_authority.configured
        or type(service) is not CommonTransformService
        or type(compiler_library_selection) is not CompilerLibrarySelection
        or type(compiler_pass_selection) is not CompilerPassSelection
    ):
        raise StageGateError("public diagnostics require original live source, profile, edit and pass owners")
    candidate, baseline, work_root = map(Path, (candidate, baseline, work_root))
    if candidate != edit_authority.initial_source or work_root.exists() or work_root.resolve() != work_root:
        raise StageGateError("public diagnostic output and edit authority must have their original owners")
    if any(
        work_root.is_relative_to(path) or path.is_relative_to(work_root)
        for path in (candidate, baseline, corpus.root, edit_authority.output, service.workspace)
    ):
        raise StageGateError("public diagnostic output overlaps an immutable input or original workspace")
    with shared_compute_pool(timeout_s=120):
        baseline_inventory = _inventory(baseline, readonly=True)
        if baseline_inventory != _inventory(edit_authority.seed, readonly=True):
            raise StageGateError("public diagnostics initially compare their exact original frozen seed")
        for cap in _capsules(corpus, contract_root):
            if (
                cap.get("kind") != "model"
                or cap.get("label") not in {"public", "dev"}
                or not {"L0", "L1", "L2", "L3"} <= set(cap.get("required_oracle_tiers") or ())
            ):
                raise StageGateError("public diagnostics retain complete public model and mandatory tier obligations")
        return PublicSourceDiagnosticPolicy(
            _issuer=_ISSUER,
            candidate=candidate,
            _baseline=baseline,
            _corpus=corpus,
            target_experiment=target_experiment,
            edit_authority=edit_authority,
            feedback_evaluator=PublicDiagnosticExecution(
                service,
                work_root,
                Path(CHECKED.record(checked_source_diagnostic, compiler_pass_selection)["rtl_facts"]["path"]),
            ),
            _compiler_library_selection=compiler_library_selection,
            _compiler_pass_selection=compiler_pass_selection,
            _checked_source_diagnostic=checked_source_diagnostic,
            _contract_root=Path(contract_root),
            _binding_baseline=baseline_inventory,
            receipt_path=work_root / "diagnostic-policy.json",
        )


def selected(policy):
    return type(policy) is PublicSourceDiagnosticPolicy
