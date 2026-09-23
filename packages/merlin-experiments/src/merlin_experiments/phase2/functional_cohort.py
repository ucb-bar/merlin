"""Exact public/hidden functional cohort admission for performance experiments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.common.digest import sha256_file as _sha_file
from merlin.targetgen import capsule_runner as CAPSULES
from merlin.targetgen.capsule_common import discover_capsules
from merlin_experiments.phase2 import functional_inputs as FI
from merlin_experiments.phase2 import gsim_gate as GATE
from merlin_experiments.phase2 import gsim_workload as WORKLOAD
from merlin_experiments.phase2.broker_evidence import _is_sha256 as _is_sha
from merlin_experiments.phase2.contracts import PerformanceExperimentError as ExperimentError


@dataclass(frozen=True)
class FunctionalCapsule:
    """One exact descriptor admitted by the canonical formal grading policy."""

    name: str
    kind: str
    manifest: Path
    manifest_sha256: str
    workload_sha256: str


@dataclass(frozen=True)
class FunctionalGradeCohort:
    """The public and hidden descriptors that actually enter the scored denominator."""

    public: tuple[FunctionalCapsule, ...]
    hidden: tuple[FunctionalCapsule, ...]
    public_source_count: int
    hidden_source_count: int
    #: Capsules THIS submission stated it does not lower. They stay in the cohort -- they are part of
    #: the scored denominator and the grade reports them as ``declined`` -- and they are excluded from
    #: the one-ELF certificate envelope, because a declined capsule produced no ELF to cross-validate.
    #: Empty when no run was consulted, which keeps the previous behaviour exactly.
    declined: tuple[str, ...] = ()
    #: Frozen Phase-1 inputs used when this cohort came from an inspected functional run.  They are
    #: deliberately optional so descriptor-only tooling can still derive today's live cohort.
    frozen_contract: Path | None = None
    frozen_hidden_source_dirs: tuple[Path, ...] = ()
    public_admission: Mapping[str, Any] | None = None
    admission_descriptor_sha256: str | None = None


def functional_capsule(cap: Mapping[str, Any]) -> FunctionalCapsule:
    name = cap.get("name")
    kind = cap.get("kind")
    directory = cap.get("__dir__")
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(kind, str)
        or not kind
        or not isinstance(directory, str)
        or not directory
    ):
        raise ExperimentError("functional capsule discovery returned a malformed name/kind/directory")
    manifest = (Path(directory) / "capsule.yaml").resolve()
    if not manifest.is_file():
        raise ExperimentError(f"functional capsule descriptor is absent: {manifest}")
    workload = WORKLOAD.derive_workload(manifest)
    return FunctionalCapsule(
        name=name,
        kind=kind,
        manifest=manifest,
        manifest_sha256=_sha_file(manifest),
        workload_sha256=GATE.workload_sha256(workload),
    )


def functional_grade_cohort(target: object, *, contract_root: Path) -> FunctionalGradeCohort:
    """Reproduce the official grader's descriptor-driven public and hidden admission.

    Public grading consumes ``public_capsules_for(target)``: that canonical materializer discovers the
    descriptor's graded roots with the public/dev label policy and applies ``graded_exclude``.  Hidden
    grading consumes the descriptor's hidden roots and asks ``capsule_grade.grade`` for capability
    admission, whose selector is ``capsule_runner._split_ineligible`` over non-model capsules.  Derive
    those same decisions here without materializing a cache, so preflight remains read-only.
    """
    contract = contract_root
    public_source = discover_capsules(target.graded_roots(), labels={"public", "dev"}, contract=contract)
    public_names = [str(cap.get("name")) for cap in public_source]
    if len(public_names) != len(set(public_names)):
        raise ExperimentError("functional public roots contain duplicate capsule names")
    excluded = set(getattr(target, "graded_exclude", ()) or ())
    unknown = sorted(excluded - set(public_names))
    if unknown:
        raise ExperimentError(f"functional public exclusions name absent capsules: {unknown}")
    public_selected = [cap for cap in public_source if str(cap.get("name")) not in excluded]

    hidden_source = discover_capsules(target.hidden_roots(), labels={"hidden"}, contract=contract)
    hidden_names = [str(cap.get("name")) for cap in hidden_source]
    if len(hidden_names) != len(set(hidden_names)):
        raise ExperimentError("functional hidden roots contain duplicate capsule names")
    hidden_ops = [cap for cap in hidden_source if cap.get("kind") != "model"]
    _eligible, hidden_ineligible = CAPSULES._split_ineligible(hidden_ops, target.target)
    hidden_excluded = {str(row.get("capsule")) for row in hidden_ineligible}
    hidden_selected = [cap for cap in hidden_source if str(cap.get("name")) not in hidden_excluded]

    expected = (
        ("public source", len(public_source), getattr(target, "graded_expected_source_capsules", None)),
        ("public admitted", len(public_selected), getattr(target, "graded_expected_admitted_capsules", None)),
        ("hidden source", len(hidden_source), getattr(target, "hidden_expected_source_capsules", None)),
        ("hidden admitted", len(hidden_selected), getattr(target, "hidden_expected_admitted_capsules", None)),
    )
    drift = [
        f"{label}={observed}, expected={declared}"
        for label, observed, declared in expected
        if declared is not None and observed != declared
    ]
    if drift:
        raise ExperimentError("functional grade cohort drifted from descriptor: " + "; ".join(drift))
    if not public_selected or not hidden_selected:
        raise ExperimentError("full functional grade needs nonempty admitted public and hidden cohorts")
    return FunctionalGradeCohort(
        public=tuple(functional_capsule(cap) for cap in public_selected),
        hidden=tuple(functional_capsule(cap) for cap in hidden_selected),
        public_source_count=len(public_source),
        hidden_source_count=len(hidden_source),
    )


def name_set_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(str(name) for name in names), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def score_cohort_boundary(
    score: Mapping[str, Any], *, label: str, source_names: Sequence[str], recorded_count: int
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Verify the exact admitted names written by one frozen Phase-1 grade."""
    rows = score.get("per_capsule")
    if not isinstance(rows, list) or not rows:
        raise ExperimentError(f"frozen {label} score has no per-capsule denominator")
    admitted = tuple(str(row.get("capsule")) for row in rows if isinstance(row, Mapping))
    if (
        len(admitted) != len(rows)
        or any(not name or name == "None" for name in admitted)
        or len(set(admitted)) != len(admitted)
        or score.get("n_capsules") != len(admitted)
        or recorded_count != len(admitted)
    ):
        raise ExperimentError(f"frozen {label} score has an inconsistent admitted name set")
    source = tuple(source_names)
    if len(source) != len(set(source)) or not set(admitted).issubset(source):
        raise ExperimentError(f"frozen {label} score is not a subset of its input snapshot")
    admission = score.get("cohort_admission")
    if not isinstance(admission, Mapping):
        raise ExperimentError(f"frozen {label} score has no cohort-admission record")
    counts = tuple(
        admission.get(field)
        for field in ("n_source_capsules", "n_admitted_capsules", "n_capability_excluded", "n_resource_excluded")
    )
    if (
        any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts)
        or counts[0] != len(source)
        or counts[1] != len(admitted)
        or counts[0] != counts[1] + counts[2] + counts[3]
        or admission.get("admitted_name_set_sha256") != name_set_sha256(admitted)
        or admission.get("excluded_name_set_sha256") != name_set_sha256(tuple(set(source) - set(admitted)))
    ):
        raise ExperimentError(f"frozen {label} cohort-admission record does not close")
    return admitted, dict(admission)


def functional_grade_cohort_from_run(
    target: object, functional: FI.StageFunctionalRun, *, source_root: Path
) -> FunctionalGradeCohort:
    """Reconstruct the scored cohort from Phase 1's immutable input snapshot and score names.

    The live descriptor may legitimately acquire new capsules after Phase 1.  Such growth belongs to a
    later functional run; it must not silently change the denominator of a performance campaign whose
    compiler and functional verdict were frozen earlier.
    """
    snapshot_root = Path(functional.bundle_input_snapshot["path"])
    snapshot_repo = (snapshot_root / "repo").resolve(strict=True)
    live_repo = source_root.resolve()
    try:
        primary_relative = Path(target.capsule_corpus).resolve().relative_to(live_repo)
    except (AttributeError, ValueError) as exc:
        raise ExperimentError("target capsule corpus cannot be mapped into the functional snapshot") from exc
    primary = snapshot_repo / primary_relative
    if primary.is_symlink() or not primary.is_dir():
        raise ExperimentError("functional snapshot lacks the target's primary capsule corpus")
    parent = primary.parent
    public_roots = [
        primary,
        *(
            directory
            for directory in sorted(parent.iterdir())
            if directory.is_dir()
            and not directory.is_symlink()
            and directory != primary
            and directory.name != "hidden"
            and not directory.name.startswith(("_", "."))
            and next(directory.glob("*/capsule.yaml"), None) is not None
        ),
    ]
    hidden_root = parent / "hidden"
    if hidden_root.is_symlink() or not hidden_root.is_dir():
        raise ExperimentError("functional snapshot lacks the target's hidden capsule corpus")
    contract = snapshot_repo / "merlin/contract"
    public_source = discover_capsules(public_roots, labels={"public", "dev"}, contract=contract)
    hidden_source = discover_capsules([hidden_root], labels={"hidden"}, contract=contract)
    public_index = {str(cap.get("name")): cap for cap in public_source}
    hidden_index = {str(cap.get("name")): cap for cap in hidden_source}
    if len(public_index) != len(public_source) or len(hidden_index) != len(hidden_source):
        raise ExperimentError("functional snapshot contains duplicate target capsule names")
    public_names, public_admission = score_cohort_boundary(
        functional.public_score,
        label="public",
        source_names=tuple(public_index),
        recorded_count=functional.public_capsules,
    )
    hidden_names, _hidden_admission = score_cohort_boundary(
        functional.hidden_score,
        label="hidden",
        source_names=tuple(hidden_index),
        recorded_count=functional.hidden_capsules,
    )
    descriptor_sha = public_admission.get("descriptor_sha256")
    if not _is_sha(descriptor_sha):
        raise ExperimentError("frozen public admission does not pin its target descriptor")
    return FunctionalGradeCohort(
        public=tuple(functional_capsule(public_index[name]) for name in public_names),
        hidden=tuple(functional_capsule(hidden_index[name]) for name in hidden_names),
        public_source_count=len(public_source),
        hidden_source_count=len(hidden_source),
        frozen_contract=contract.resolve(),
        frozen_hidden_source_dirs=tuple(Path(hidden_index[name]["__dir__"]).resolve() for name in sorted(hidden_index)),
        public_admission=public_admission,
        admission_descriptor_sha256=str(descriptor_sha),
    )


def declined_names(functional: object) -> tuple[str, ...]:
    """Capsule names this submission's own grade recorded as DECLINED, public and hidden.

    Read from the score the run already wrote, never re-derived: "the backend stated it does not lower
    this" is a fact about the SUBMISSION, and recomputing it from the descriptor would answer a
    different question.
    """
    names: set[str] = set()
    for attribute in ("public_score", "hidden_score"):
        score = getattr(functional, attribute, None)
        if not isinstance(score, dict):
            continue
        for row in score.get("declined") or ():
            name = row.get("capsule") if isinstance(row, dict) else row
            if isinstance(name, str) and name:
                names.add(name)
    return tuple(sorted(names))


def functional_gsim_cases(cohort: FunctionalGradeCohort) -> tuple[FunctionalCapsule, ...]:
    """Exact admitted descriptors eligible for strict one-ELF GSIM/Verilator capture.

    An operator capsule lowers to one ELF, so the pre-authoring certificate can run that identical ELF
    on both engines. A model capsule is an ordered host/accelerator program whose formal evidence is a
    dynamic ledger of many tile ELFs; pretending it has one descriptor-level ELF would certify an
    artifact that does not exist. Models therefore remain in ``FunctionalGradeCohort`` and the complete
    post-candidate functional regrade, where the pinned GSIM runs every emitted tile and the model
    execution checker validates their ledger. They are excluded only from this *prebuilt one-ELF*
    certificate envelope.

    A capsule the submission DECLINED is excluded for the same reason, and it IS the same reason: it
    produced no ELF at all. The backend stated it does not lower these -- "rather than emitting a
    program that writes nothing" -- so there is nothing for the two engines to disagree about, while
    demanding a capture for one makes the whole certificate unbuildable over a fact that has nothing to
    do with GSIM/Verilator agreement. Measured on the g4p1 submission: 10 host-lane bf16/f32 capsules,
    every one already recorded ``declined`` by its own grade, blocked all 90 cases.

    The exclusion is per-SUBMISSION and never a property of the corpus: a later submission that lowers
    them puts them straight back into the envelope.
    """
    declined = frozenset(cohort.declined)
    return tuple(
        capsule
        for capsule in (*cohort.public, *cohort.hidden)
        if capsule.kind != "model" and capsule.name not in declined
    )
