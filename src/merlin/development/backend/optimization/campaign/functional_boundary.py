"""Fail-closed boundary between the completed functional Arm-4 run and performance.

The performance campaign is allowed to consume exactly one content-addressed functional
submission.  It copies that submission into a run-private workspace, makes the copy read-only,
records the functional fork, and executes every untrusted package entrypoint in a credential-free,
networkless bwrap sandbox.  Pure helpers in this module make the refusal paths testable without
starting a simulator.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from pathlib import Path

import yaml

from merlin.analysis.perf.search.fork import ForkPoint, candidate_states, check_invariants, fork_from
from merlin.common.digest import hash_tree
from merlin.development.backend.optimization.author_interface.prompt_inputs import (
    PerfCell,
    PerfPromptContractError,
)
from merlin.development.targetgen.oracles.schedule import CapsuleState, Verdict
from merlin.development.targetgen.packaging import loading as oot_runner
from merlin.development.targetgen.sandbox import grants as BW
from merlin.development.targetgen.sandbox import toolchain as TC
from merlin.development.targetgen.sandbox.answer_surfaces import (
    AnswerSurface,
    CapsuleMetadata,
    answer_surfaces,
    audit_hit_is_violation,
)
from merlin.target.registry.experiment_descriptor import TargetExperiment

# ---- campaign ----------------------------------------------------------------------------------


class CampaignGateError(RuntimeError):
    """A phase-boundary or completion condition was not proved."""


def verify_private_input_snapshot(run_dir: Path, environment: Mapping) -> dict:
    """Verify V4 ownership, original bundle and payload against host-owned provenance.

    This is deliberately not a marker-only verifier: the marker cannot authenticate its own
    private-input policy. Native runs pin the archived bundle before authoring; a re-freeze pins
    its new grading bundle separately without rewriting the original authoring evidence.
    """
    snapshot = environment.get("bundle_input_snapshot")
    if not isinstance(snapshot, dict) or snapshot.get("version") != 4:
        raise CampaignGateError("verified execution requires bundle snapshot V4 ownership; create a newly frozen run")
    name = environment.get("bundle_manifest_file", "input_bundle_manifest.yaml")
    if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
        raise CampaignGateError("functional bundle manifest must be archived in the run directory")
    bundle_path = run_dir / name
    bundle = _mapping_file(bundle_path, yaml_file=True)
    bundle_digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    if environment.get("bundle_manifest_sha256") != bundle_digest:
        raise CampaignGateError("functional archived bundle bytes disagree with host provenance")
    from merlin.development.targetgen.authoring.tool_profile import verify_tool_profile

    try:
        profile = verify_tool_profile(bundle, tools=tuple(environment.get("resolved_tools") or ()))
    except ValueError as error:
        raise CampaignGateError(str(error)) from error
    if profile is not None and environment.get("compiler_tool_profile") != profile:
        raise CampaignGateError("functional run changed its immutable compiler tool profile")
    root = Path(str(snapshot.get("path") or ""))
    ws = Path(str(environment.get("workspace_path") or ""))
    if not root.is_absolute() or not ws.is_absolute() or root.is_symlink() or BW.bundle_snapshot_root(ws) != root:
        raise CampaignGateError("functional snapshot is not bound to its original workspace")
    marker_path = root / "snapshot.json"
    marker = _mapping_file(marker_path)
    if hashlib.sha256(marker_path.read_bytes()).hexdigest() != snapshot.get("manifest_sha256"):
        raise CampaignGateError("functional snapshot marker bytes disagree with host provenance")
    repo = Path(str(marker.get("repo") or ""))
    if not repo.is_absolute() or ".." in repo.parts:
        raise CampaignGateError("functional snapshot has no safe original repository")
    try:
        BW.verify_snapshot_binding(ws, bundle, snapshot, repo=repo)
        verified = BW.verify_bundle_snapshot(ws, bundle, repo=repo)
    except (OSError, RuntimeError, ValueError) as exc:
        raise CampaignGateError("functional private input snapshot verification failed") from exc
    for field in ("version", "content_sha256", "n_files", "n_bytes"):
        if verified.get(field) != snapshot.get(field):
            raise CampaignGateError("functional snapshot payload disagrees with host provenance")
    return {
        "workspace": ws,
        "repo": repo,
        "bundle": bundle,
        "bundle_path": bundle_path,
        "bundle_sha256": bundle_digest,
        "marker": verified,
    }


UNWAIVABLE: frozenset[str] = frozenset(
    {
        "sandbox_not_bwrap",
        "isolation_audit_unclean",
        "bundle_input_snapshot_incomplete",
        "answer_mask_vacuous",
        "round_answer_access_unclean",
        "finalize_answer_access_unclean",
        "manifest_integrity_gate_failed",
        "score_integrity_failed",
        "capsule_identity_reused",
        "freeze_not_immutable",
        "cohort_admission_missing",
        "cohort_admission_malformed",
        "cohort_admission_does_not_close",
        "excluded_name_set_unpinned",
    }
)

_WAIVABLE_PREDICATES: frozenset[str] = frozenset(
    {
        "manifest_not_gradeable",
        "score_incomplete",
        "score_not_gradeable",
        "score_evidence_incomplete",
        "capsule_without_passing_verdict",
        "score_repeats_capsule",
        "capsule_tier_not_earned",
        "qa_loop_not_converged",
        "no_completed_round",
        "finalize_missing",
        "finalize_regrade_not_pass",
        "phase_functional_pass_missing",
        # A phase grade whose passed/total ratio is SHORT but well-formed. Added deliberately (see the
        # note above): it is a completeness fact -- every declared capsule was graded and some did not
        # pass -- and it carries the observed ratio, so a reader of a waived result knows exactly what
        # was short. A MALFORMED or VACUOUS ratio is broken evidence and still raises in `_graded_ratio`,
        # so this name can never admit one.
        "phase_grade_incomplete",
        "public_tier_below_cert",
        "task_scope_public_mismatch",
        "task_scope_hidden_mismatch",
        "graded_count_not_admitted_cohort",
        "operator_sealed_before_convergence",
    }
)


@dataclass(frozen=True)
class Deviation:
    """One gate predicate that failed and was explicitly waived, with what was actually observed."""

    predicate: str
    detail: str

    def to_dict(self) -> dict:
        return {"predicate": self.predicate, "detail": self.detail}


@dataclass(frozen=True)
class FunctionalRun:
    run_dir: Path
    submission_dir: Path
    run_id: str
    digest: str
    public_capsules: int
    hidden_capsules: int
    public_score: dict
    hidden_score: dict
    frozen_at: str
    # Empty tuple means the gate passed on its own terms. A non-empty tuple must travel with every
    # number derived from this run: a result produced under a waiver can be reported, but it can never
    # be cited as though the gate had passed.
    deviations: tuple[Deviation, ...] = ()

    @property
    def gate_clean(self) -> bool:
        return not self.deviations


@dataclass(frozen=True)
class PackageSandboxInputs:
    """Resolved construction inputs, without ambient simulator or answer discovery.

    An empty harness or surface tuple explicitly selects absence. This is a runtime
    selection, not a serialized snapshot or a guarantee that the selected bytes are frozen.
    """

    paths: TC.ToolchainPaths
    sim: TC.SimToolchain
    harness: str
    surfaces: tuple[AnswerSurface, ...]


@dataclass(frozen=True)
class FrozenPackageSandboxInputs:
    """Verified declaration-owned resources; no original providers are consulted."""

    root: Path
    record: dict
    repo: Path
    argv: tuple[str, ...]
    probes: tuple[TC.ToolProbe, ...]
    env_prefix: str
    surfaces: tuple[AnswerSurface, ...]


class PackagePolicyScope:
    """What every package entrypoint policy of one execution scope may share.

    One grade, self-check request or probe owns one instance for its executor lifetime and passes it
    to :func:`package_sandbox_policy`. Capsule metadata parses are keyed by freshly read bytes.
    Filesystem membership and linked-inode facts may share the existing kernel roster guard;
    any mutation invalidates reuse and unsupported trees retain fresh traversal. Live selections,
    mount translations, privacy classifications and grants are derived on every call. No policy
    result or cross-request authority is cached.
    """

    def __init__(self) -> None:
        self.capsules = CapsuleMetadata()


def select_package_sandbox_inputs(
    te: TargetExperiment, *, scope: PackagePolicyScope | None = None
) -> PackageSandboxInputs:
    """Resolve the live selection once, before freezing an execution declaration."""
    answers = answer_surfaces(te) if scope is None else answer_surfaces(te, capsules=scope.capsules)
    return PackageSandboxInputs(
        TC.ToolchainPaths.from_checkout(), TC._sim(te), TC.curated_harness_dir(te), tuple(answers)
    )


@dataclass(frozen=True)
class PackageSandboxPolicy:
    argv: tuple[str, ...]
    coverage_gap: tuple[str, ...]
    required_tools: tuple[TC.ToolProbe, ...]
    workspace: Path
    package: Path
    target_experiment: TargetExperiment
    env_prefix: str | None = None
    frozen_inputs: FrozenPackageSandboxInputs | None = None
    selected_inputs: PackageSandboxInputs | FrozenPackageSandboxInputs | None = dataclass_field(
        default=None, repr=False, compare=False
    )

    compiler_fork_sha256: str | None = None

    def __post_init__(self) -> None:
        # Preserve the six-argument constructor while capturing its defaults once.
        # Descriptor-free policies are also used to exercise thread-local routing.
        if self.env_prefix is None:
            prefix = (
                TC.sandbox_env(self.target_experiment, self.workspace) if self.target_experiment is not None else ""
            )
            object.__setattr__(self, "env_prefix", prefix)


def _mapping_file(path: Path, *, yaml_file: bool = False) -> dict:
    if not path.is_file():
        raise CampaignGateError(f"required campaign evidence is absent: {path}")
    try:
        doc = (
            yaml.safe_load(path.read_text(encoding="utf-8"))
            if yaml_file
            else json.loads(path.read_text(encoding="utf-8"))
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise CampaignGateError(f"campaign evidence is unreadable at {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise CampaignGateError(f"campaign evidence must be a mapping: {path}")
    return doc


def _graded_ratio(value: object, *, label: str) -> tuple[int, list[Deviation]]:
    """The DECLARED total of a ``passed/total`` phase grade, plus any completeness deviation.

    Two different failures used to be folded into one raise here, and that is why an honest run could
    not be consumed at all: the same shortfall that `inspect_functional_run` already reports as the
    WAIVABLE ``phase_functional_pass_missing`` aborted the inspection before any waiver could be
    acknowledged. The two are not the same kind of fact —

    * a ratio that is MALFORMED, non-numeric, VACUOUS (``total <= 0``), or arithmetically impossible
      (``passed`` negative, or larger than the total that was graded) is BROKEN EVIDENCE. There is no
      grade to reason about, so it still RAISES and no waiver reaches it;
    * a ratio that is merely SHORT (``passed < total``) is a COMPLETENESS gap of exactly the kind the
      waiver mechanism exists for — every declared capsule was graded and some did not pass — so it
      becomes the named, waivable ``phase_grade_incomplete`` deviation carrying the observed ratio.

    THE RETURNED TOTAL IS ALWAYS THE DECLARED TOTAL, never ``passed``. Downstream this number is the
    ``expected=`` denominator for :func:`_validate_score` and the coverage figure for
    :func:`_validate_scope_coverage`; returning ``passed`` would shrink the denominator and make an
    83/96 run read as a complete 83/83 one — a worse defect than the refusal it replaces.
    """
    parts = str(value or "").split("/")
    if len(parts) != 2:
        raise CampaignGateError(f"{label} must state an explicit passed/total ratio")
    try:
        passed, total = (int(x) for x in parts)
    except ValueError as exc:
        raise CampaignGateError(f"{label} has a malformed passed/total ratio: {value!r}") from exc
    if total <= 0:
        raise CampaignGateError(f"{label} must be non-vacuous, not {passed}/{total}")
    if passed < 0 or passed > total:
        raise CampaignGateError(f"{label} has an impossible passed/total ratio: {passed}/{total}")
    if passed != total:
        return total, [
            Deviation(
                "phase_grade_incomplete",
                f"{label} is incomplete: {passed}/{total} ({total - passed} of {total} graded capsule(s) did not pass)",
            )
        ]
    return total, []


def _validate_score(score: dict, *, label: str, expected: int) -> list[Deviation]:
    """Findings for one graded phase. Returns them rather than raising on the first.

    A caller cannot acknowledge a deviation it was never shown, so every predicate is evaluated and
    every failure is named. The gate's strictness is unchanged — `inspect_functional_run` still raises
    on anything not explicitly waived — but it now reports the whole picture at once instead of one
    blocker per run.
    """
    found: list[Deviation] = []
    n_capsules = score.get("n_capsules")
    n_passed = score.get("n_passed")
    rows = score.get("per_capsule")
    if (
        not isinstance(n_capsules, int)
        or isinstance(n_capsules, bool)
        or not isinstance(n_passed, int)
        or isinstance(n_passed, bool)
        or n_capsules <= 0
        or n_passed != n_capsules
        or n_capsules != expected
    ):
        found.append(
            Deviation(
                "score_incomplete",
                f"{label} score is not a full non-vacuous {expected}/{expected} grade "
                f"(n_capsules={n_capsules!r}, n_passed={n_passed!r})",
            )
        )
    if score.get("functional_pass") != 1 or score.get("gradeable") is not True:
        found.append(Deviation("score_not_gradeable", f"{label} score is not gradeable and functionally complete"))
    if score.get("integrity_status") != "clean" or score.get("integrity_exempt") is not False:
        found.append(
            Deviation(
                "score_integrity_failed",
                f"{label} score did not pass the integrity gate "
                f"(status={score.get('integrity_status')!r}, "
                f"exempt={score.get('integrity_exempt')!r})",
            )
        )
    if not isinstance(rows, list) or len(rows) != expected:
        found.append(Deviation("score_evidence_incomplete", f"{label} score has no complete per-capsule evidence"))
        return found
    names: set[str] = set()
    unverdicted, repeated, untiered = [], [], []
    for row in rows:
        if not isinstance(row, dict) or not row.get("capsule") or row.get("status") != "pass":
            unverdicted.append(str((row or {}).get("capsule") if isinstance(row, dict) else row))
            continue
        name = str(row["capsule"])
        if name in names:
            repeated.append(name)
        names.add(name)
        tiers = row.get("tiers") or {}
        if tiers.get("L2") != "pass" or tiers.get("L3") != "pass":
            untiered.append(f"{name}(L2={tiers.get('L2')!r},L3={tiers.get('L3')!r})")
    if unverdicted:
        found.append(
            Deviation(
                "capsule_without_passing_verdict",
                f"{label} contains a capsule without a passing verdict: {', '.join(unverdicted[:6])}",
            )
        )
    if repeated:
        found.append(Deviation("score_repeats_capsule", f"{label} repeats capsule {repeated[0]!r}"))
    if untiered:
        found.append(
            Deviation(
                "capsule_tier_not_earned", f"{label} capsule(s) did not earn both L2 and L3: {', '.join(untiered[:6])}"
            )
        )
    return found


def _audit_violations(hits: object) -> list[str]:
    """The answer-access audit hits that actually mean withheld content reached the agent.

    The QA audit records BOTH violations and advisories, and the difference is the whole point: a
    ``blocked_probe`` is a read the mask REFUSED -- evidence the protection worked -- and a
    ``recon_probe`` returned filenames, not content. Demanding an EMPTY hit list therefore disqualified,
    unwaivably, rounds the audit itself judged clean (measured: round 0 of merlincirct_g4p1_20260905 was
    ``answer_access_clean: true`` with two advisory hits). The advisory/violation vocabulary is DECLARED
    once, in :mod:`merlin.development.targetgen.sandbox.answer_surfaces`, and consumed here -- so this gate cannot
    drift from the audit that produced the hits.

    FAIL CLOSED in every other direction: a missing or non-list ``audit_hits`` field, a hit that is not a
    mapping, and any hit kind not explicitly declared advisory all count as violations. A new hit kind is
    disqualifying until someone deliberately declares it benign.
    """
    if not isinstance(hits, list):
        return ["audit_hits is missing or not a list"]
    bad = []
    for hit in hits:
        if audit_hit_is_violation(hit):
            kind = hit.get("kind") if isinstance(hit, dict) else None
            bad.append(str(kind) if kind else f"malformed hit {hit!r}"[:120])
    return bad


def _validate_clean_run(environment: dict, summary: dict) -> list[Deviation]:
    """Findings about how the run was CONDUCTED, separated by kind (see UNWAIVABLE)."""
    found: list[Deviation] = []
    if environment.get("sandbox") != "bwrap":
        found.append(Deviation("sandbox_not_bwrap", "functional run was not executed in the required bwrap sandbox"))
    if environment.get("isolation_violations") != []:
        found.append(Deviation("isolation_audit_unclean", "functional run lacks a clean isolation-violation audit"))
    inputs = environment.get("bundle_input_snapshot")
    snapshot_digest = inputs.get("content_sha256") if isinstance(inputs, dict) else None
    if (
        not isinstance(inputs, dict)
        or inputs.get("version") != 4
        or not isinstance(snapshot_digest, str)
        or len(snapshot_digest) != 64
        or any(c not in "0123456789abcdef" for c in snapshot_digest)
        or not isinstance(inputs.get("n_files"), int)
        or isinstance(inputs.get("n_files"), bool)
        or inputs["n_files"] <= 0
        or not isinstance(inputs.get("n_bytes"), int)
        or isinstance(inputs.get("n_bytes"), bool)
        or inputs["n_bytes"] <= 0
    ):
        found.append(
            Deviation(
                "bundle_input_snapshot_incomplete",
                "functional run requires a complete V4 ownership snapshot; create a newly frozen run",
            )
        )
    mask = environment.get("golden_mask_selftest")
    if (
        not isinstance(mask, dict)
        or mask.get("leaked_answer_files") != []
        or int(mask.get("n_answer_files_masked") or 0) <= 0
    ):
        found.append(Deviation("answer_mask_vacuous", "functional run did not prove a non-vacuous clean answer mask"))
    if summary.get("converged") is not True:
        found.append(
            Deviation(
                "qa_loop_not_converged", f"functional QA loop did not converge (converged={summary.get('converged')!r})"
            )
        )
    operator_seal = summary.get("operator_seal")
    if isinstance(operator_seal, dict) and operator_seal.get("requested") is True:
        found.append(
            Deviation(
                "operator_sealed_before_convergence",
                "functional authoring was explicitly stopped at completed round "
                f"{operator_seal.get('last_completed_round')!r}; checkpoint submission "
                f"{operator_seal.get('checkpoint_submission_sha256')!r} was sent through the ordinary "
                "official grade and immutable freeze",
            )
        )
    rounds = summary.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        found.append(Deviation("no_completed_round", "functional QA evidence has no completed round"))
    else:
        for row in rounds:
            if not isinstance(row, dict) or row.get("answer_access_clean") is not True:
                found.append(
                    Deviation("round_answer_access_unclean", "functional QA round failed the answer-access audit")
                )
                break
            bad = _audit_violations(row.get("audit_hits"))
            if bad:
                found.append(
                    Deviation(
                        "round_answer_access_unclean", f"functional QA round carries answer-access VIOLATIONS: {bad}"
                    )
                )
                break
    # THE FINALIZE BLOCK IS TWO DIFFERENT QUESTIONS, and folding them into one predicate made a
    # SKIPPED finalize indistinguishable from an unaudited one. A submission-identity pin skips the
    # finalizer by design (`run_baseline_qa_loop`: "docs-only finalizer would change certified
    # bytes"), so a pinned run has no finalize turn at all — there is no agent activity to audit, and
    # demanding an answer-access verdict for a turn that never happened is a category error. What such
    # a run genuinely LACKS is the post-finalize regrade, and that is a completeness gap.
    finalize = summary.get("finalize")
    if not isinstance(finalize, dict):
        found.append(Deviation("finalize_missing", "functional run recorded no finalization block"))
    elif finalize.get("skipped") is True:
        found.append(
            Deviation(
                "finalize_regrade_not_pass",
                f"finalization was SKIPPED, so there is no post-finalize clean regrade "
                f"({finalize.get('reason') or 'no reason recorded'}); the rounds' own "
                f"answer-access audits still apply and no un-audited turn was run",
            )
        )
    else:
        bad = _audit_violations(finalize.get("audit_hits"))
        if finalize.get("answer_access_clean") is not True or bad:
            found.append(
                Deviation(
                    "finalize_answer_access_unclean",
                    "functional finalization did not pass its answer-access audit"
                    + (f" (violations: {bad})" if bad else ""),
                )
            )
        if finalize.get("regrade_all_pass") is not True:
            found.append(
                Deviation(
                    "finalize_regrade_not_pass",
                    f"functional finalization did not pass a clean regrade "
                    f"(regrade_all_pass={finalize.get('regrade_all_pass')!r})",
                )
            )
    return found


def _has_symlink(root: Path) -> Path | None:
    if root.is_symlink():
        return root
    for path in root.rglob("*"):
        if path.is_symlink():
            return path
    return None


def _digest_excluded_path(root: Path) -> Path | None:
    """Return executable state omitted by the functional harness's historical tree hash."""
    excluded = {"build", "__pycache__", ".git"}
    for path in root.rglob("*"):
        if excluded & set(path.relative_to(root).parts):
            return path
    return None


def _admission_shortfall(score: dict, graded: int, *, label: str) -> tuple[int, list[Deviation]]:
    """How many source capsules this phase did NOT grade, and whether that is properly accounted for.

    Returns ``(accounted_exclusions, findings)``. A phase's ``cohort_admission`` record states the
    source pool, the admitted cohort and the exclusions by class, with a digest over the excluded name
    set. Requiring the arithmetic to CLOSE is what makes an exclusion evidence rather than a gap: a
    capsule may be absent from a grade only if the record says so and the numbers add up.
    """
    found: list[Deviation] = []
    admission = score.get("cohort_admission")
    if not isinstance(admission, dict):
        return 0, [
            Deviation(
                "cohort_admission_missing",
                f"{label} carries no cohort-admission record, so a capsule absent from "
                f"the grade cannot be distinguished from one silently dropped",
            )
        ]

    def _n(field: str) -> int | None:
        v = admission.get(field)
        return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None

    source, admitted = _n("n_source_capsules"), _n("n_admitted_capsules")
    cap_ex, res_ex = _n("n_capability_excluded"), _n("n_resource_excluded")
    if None in (source, admitted, cap_ex, res_ex):
        return 0, [
            Deviation("cohort_admission_malformed", f"{label} cohort-admission counts are malformed: {admission!r}")
        ]
    if admitted + cap_ex + res_ex != source:
        found.append(
            Deviation(
                "cohort_admission_does_not_close",
                f"{label} admission arithmetic does not close: admitted={admitted} + "
                f"capability={cap_ex} + resource={res_ex} != source={source}",
            )
        )
    if cap_ex or res_ex:
        digest = admission.get("excluded_name_set_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            found.append(
                Deviation(
                    "excluded_name_set_unpinned",
                    f"{label} excluded {cap_ex + res_ex} capsule(s) without pinning the excluded name set to a digest",
                )
            )
    if graded != admitted:
        found.append(
            Deviation(
                "graded_count_not_admitted_cohort",
                f"{label} graded {graded} capsule(s) but its admission record admits {admitted}",
            )
        )
    return cap_ex + res_ex, found


def _validate_scope_coverage(
    task_scope: dict, public_score: dict, hidden_score: dict, *, n_public: int, n_hidden: int
) -> list[Deviation]:
    """Does each grade cover the sealed scope, once recorded exclusions are reconciled?

    THE TWO SCOPE FIELDS COUNT DIFFERENT THINGS, and asserting one rule for both is why an honest run
    was refused. ``required_public_dev_capsules`` is the ADMITTED cohort: `_task_runtime_scope` filters
    the public pool by the descriptor's named `graded_exclude`, so it already nets out exclusions.
    ``held_out_capsules`` is the SEALED POOL: hidden names are deliberately absent from the public
    descriptor (they would leak the holdout), so that count cannot be name-filtered and is instead tied
    to the hidden snapshot digest.

    Comparing a sealed pool count directly to the admitted count can refuse a valid
    run with recorded capability or resource exclusions. Reconcile those exclusions
    from the frozen evidence rather than assuming both scope fields count the same set.

    So reconcile against the phase's own admission record instead of comparing raw totals. This is
    STRICTER where it counts: the exclusions must be recorded, digest-pinned, and add up, whereas a
    raw-total comparison would pass silently whenever two unrelated numbers happened to agree.
    """
    found: list[Deviation] = []
    pub_excluded, pub_found = _admission_shortfall(public_score, n_public, label="public functional grade")
    hid_excluded, hid_found = _admission_shortfall(hidden_score, n_hidden, label="hidden functional grade")
    found += pub_found + hid_found

    required_public = task_scope.get("required_public_dev_capsules")
    if (
        not isinstance(required_public, int)
        or isinstance(required_public, bool)
        or required_public <= 0
        or required_public != n_public
    ):
        found.append(
            Deviation(
                "task_scope_public_mismatch",
                f"public functional grade does not cover the sealed task scope: "
                f"grade={n_public}, required={required_public!r}",
            )
        )
    required_hidden = task_scope.get("held_out_capsules")
    if not isinstance(required_hidden, int) or isinstance(required_hidden, bool) or required_hidden <= 0:
        found.append(
            Deviation(
                "task_scope_hidden_mismatch",
                f"sealed task scope declares no positive held-out count: {required_hidden!r}",
            )
        )
    elif required_hidden != n_hidden + hid_excluded:
        found.append(
            Deviation(
                "task_scope_hidden_mismatch",
                f"hidden functional grade does not cover the sealed task scope: "
                f"graded={n_hidden} + accounted exclusions={hid_excluded} "
                f"!= sealed={required_hidden}",
            )
        )
    return found
