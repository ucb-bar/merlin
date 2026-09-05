"""Fail-closed boundary between the completed functional Arm-4 run and performance.

The performance campaign is allowed to consume exactly one content-addressed functional
submission.  It copies that submission into a run-private workspace, makes the copy read-only,
records the functional fork, and executes every untrusted package entrypoint in a credential-free,
networkless bwrap sandbox.  Pure helpers in this module make the refusal paths testable without
starting a simulator.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import yaml

from merlin.benchharness import hash_tree
from merlin.perf.fork import ForkPoint, candidate_states, check_invariants, fork_from
from merlin.targetgen import oot_runner
from merlin.targetgen.oracle_schedule import CapsuleState, Verdict
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox import toolchain as TC
from merlin.targetgen.sandbox.answer_surfaces import answer_surfaces, audit_hit_is_violation
from merlin.targetgen.target_experiment import TargetExperiment
from perf_prompt import PerfCell, PerfPromptContractError


class CampaignGateError(RuntimeError):
    """A phase-boundary or completion condition was not proved."""


# ----------------------------------------------------------------------------------------------
# WAIVERS. The gate's predicates are not all the same KIND of requirement, and treating them as one
# undifferentiated wall is why the only way past it was to edit the gate.
#
# A COMPLETENESS predicate says the functional baseline is fully established. Waiving one yields a
# WEAKER result: a cycle measurement resting on a baseline that is incomplete in a named way. That is
# a legitimate research trade-off — it is still a real measurement, and a reader who is told exactly
# what was incomplete can judge it.
#
# An INTEGRITY predicate says the numbers mean anything at all: the sandbox held, the answer mask was
# clean, no answer-key identity crossed between public and hidden, the bytes are the bytes. Waiving
# one does not yield a weaker result, it yields a WORTHLESS one, because the measurement could be
# fabricated or leaked. So these are UNWAIVABLE, and asking to waive one is itself an error rather
# than a permission — a blanket `--force` cannot exist here, only named waivers.
UNWAIVABLE: frozenset[str] = frozenset({
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
})


# Derived from the two validators below: every predicate they can emit, minus the unwaivable ones.
# Declared as a literal so a NEW predicate is not silently waivable the moment someone adds it --
# adding one here is a deliberate act, and `inspect_functional_run` refuses an unknown name.
_WAIVABLE_PREDICATES: frozenset[str] = frozenset({
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
})


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
class PackageSandboxPolicy:
    argv: tuple[str, ...]
    coverage_gap: tuple[str, ...]
    required_tools: tuple[TC.ToolProbe, ...]
    workspace: Path
    package: Path
    target_experiment: TargetExperiment


def _mapping_file(path: Path, *, yaml_file: bool = False) -> dict:
    if not path.is_file():
        raise CampaignGateError(f"required campaign evidence is absent: {path}")
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) if yaml_file else json.loads(
            path.read_text(encoding="utf-8"))
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
        raise CampaignGateError(
            f"{label} has an impossible passed/total ratio: {passed}/{total}")
    if passed != total:
        return total, [Deviation("phase_grade_incomplete",
                                 f"{label} is incomplete: {passed}/{total} "
                                 f"({total - passed} of {total} graded capsule(s) did not pass)")]
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
    if (not isinstance(n_capsules, int) or isinstance(n_capsules, bool)
            or not isinstance(n_passed, int) or isinstance(n_passed, bool)
            or n_capsules <= 0 or n_passed != n_capsules or n_capsules != expected):
        found.append(Deviation("score_incomplete",
                               f"{label} score is not a full non-vacuous {expected}/{expected} grade "
                               f"(n_capsules={n_capsules!r}, n_passed={n_passed!r})"))
    if score.get("functional_pass") != 1 or score.get("gradeable") is not True:
        found.append(Deviation("score_not_gradeable",
                               f"{label} score is not gradeable and functionally complete"))
    if score.get("integrity_status") != "clean" or score.get("integrity_exempt") is not False:
        found.append(Deviation("score_integrity_failed",
                               f"{label} score did not pass the integrity gate "
                               f"(status={score.get('integrity_status')!r}, "
                               f"exempt={score.get('integrity_exempt')!r})"))
    if not isinstance(rows, list) or len(rows) != expected:
        found.append(Deviation("score_evidence_incomplete",
                               f"{label} score has no complete per-capsule evidence"))
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
        found.append(Deviation("capsule_without_passing_verdict",
                               f"{label} contains a capsule without a passing verdict: "
                               f"{', '.join(unverdicted[:6])}"))
    if repeated:
        found.append(Deviation("score_repeats_capsule",
                               f"{label} repeats capsule {repeated[0]!r}"))
    if untiered:
        found.append(Deviation("capsule_tier_not_earned",
                               f"{label} capsule(s) did not earn both L2 and L3: "
                               f"{', '.join(untiered[:6])}"))
    return found


def _audit_violations(hits: object) -> list[str]:
    """The answer-access audit hits that actually mean withheld content reached the agent.

    The QA audit records BOTH violations and advisories, and the difference is the whole point: a
    ``blocked_probe`` is a read the mask REFUSED -- evidence the protection worked -- and a
    ``recon_probe`` returned filenames, not content. Demanding an EMPTY hit list therefore disqualified,
    unwaivably, rounds the audit itself judged clean (measured: round 0 of merlincirct_g4p1_20260905 was
    ``answer_access_clean: true`` with two advisory hits). The advisory/violation vocabulary is DECLARED
    once, in :mod:`merlin.targetgen.sandbox.answer_surfaces`, and consumed here -- so this gate cannot
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
        found.append(Deviation("sandbox_not_bwrap",
                               "functional run was not executed in the required bwrap sandbox"))
    if environment.get("isolation_violations") != []:
        found.append(Deviation("isolation_audit_unclean",
                               "functional run lacks a clean isolation-violation audit"))
    inputs = environment.get("bundle_input_snapshot")
    snapshot_digest = inputs.get("content_sha256") if isinstance(inputs, dict) else None
    if (not isinstance(inputs, dict) or inputs.get("version") != 2
            or not isinstance(snapshot_digest, str) or len(snapshot_digest) != 64
            or any(c not in "0123456789abcdef" for c in snapshot_digest)
            or not isinstance(inputs.get("n_files"), int) or isinstance(inputs.get("n_files"), bool)
            or inputs["n_files"] <= 0
            or not isinstance(inputs.get("n_bytes"), int) or isinstance(inputs.get("n_bytes"), bool)
            or inputs["n_bytes"] <= 0):
        found.append(Deviation("bundle_input_snapshot_incomplete",
                               "functional run lacks a complete immutable bundle-input snapshot v2 "
                               "record"))
    mask = environment.get("golden_mask_selftest")
    if (not isinstance(mask, dict) or mask.get("leaked_answer_files") != []
            or int(mask.get("n_answer_files_masked") or 0) <= 0):
        found.append(Deviation("answer_mask_vacuous",
                               "functional run did not prove a non-vacuous clean answer mask"))
    if summary.get("converged") is not True:
        found.append(Deviation("qa_loop_not_converged",
                               f"functional QA loop did not converge "
                               f"(converged={summary.get('converged')!r})"))
    rounds = summary.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        found.append(Deviation("no_completed_round", "functional QA evidence has no completed round"))
    else:
        for row in rounds:
            if not isinstance(row, dict) or row.get("answer_access_clean") is not True:
                found.append(Deviation("round_answer_access_unclean",
                                       "functional QA round failed the answer-access audit"))
                break
            bad = _audit_violations(row.get("audit_hits"))
            if bad:
                found.append(Deviation("round_answer_access_unclean",
                                       f"functional QA round carries answer-access VIOLATIONS: {bad}"))
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
        found.append(Deviation("finalize_regrade_not_pass",
                               f"finalization was SKIPPED, so there is no post-finalize clean regrade "
                               f"({finalize.get('reason') or 'no reason recorded'}); the rounds' own "
                               f"answer-access audits still apply and no un-audited turn was run"))
    else:
        bad = _audit_violations(finalize.get("audit_hits"))
        if finalize.get("answer_access_clean") is not True or bad:
            found.append(Deviation("finalize_answer_access_unclean",
                                   "functional finalization did not pass its answer-access audit"
                                   + (f" (violations: {bad})" if bad else "")))
        if finalize.get("regrade_all_pass") is not True:
            found.append(Deviation("finalize_regrade_not_pass",
                                   f"functional finalization did not pass a clean regrade "
                                   f"(regrade_all_pass={finalize.get('regrade_all_pass')!r})"))
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
        return 0, [Deviation("cohort_admission_missing",
                             f"{label} carries no cohort-admission record, so a capsule absent from "
                             f"the grade cannot be distinguished from one silently dropped")]
    def _n(field: str) -> int | None:
        v = admission.get(field)
        return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None
    source, admitted = _n("n_source_capsules"), _n("n_admitted_capsules")
    cap_ex, res_ex = _n("n_capability_excluded"), _n("n_resource_excluded")
    if None in (source, admitted, cap_ex, res_ex):
        return 0, [Deviation("cohort_admission_malformed",
                             f"{label} cohort-admission counts are malformed: {admission!r}")]
    if admitted + cap_ex + res_ex != source:
        found.append(Deviation("cohort_admission_does_not_close",
                               f"{label} admission arithmetic does not close: admitted={admitted} + "
                               f"capability={cap_ex} + resource={res_ex} != source={source}"))
    if cap_ex or res_ex:
        digest = admission.get("excluded_name_set_sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            found.append(Deviation("excluded_name_set_unpinned",
                                   f"{label} excluded {cap_ex + res_ex} capsule(s) without pinning "
                                   f"the excluded name set to a digest"))
    if graded != admitted:
        found.append(Deviation("graded_count_not_admitted_cohort",
                               f"{label} graded {graded} capsule(s) but its admission record admits "
                               f"{admitted}"))
    return cap_ex + res_ex, found


def _validate_scope_coverage(task_scope: dict, public_score: dict, hidden_score: dict, *,
                             n_public: int, n_hidden: int) -> list[Deviation]:
    """Does each grade cover the sealed scope, once recorded exclusions are reconciled?

    THE TWO SCOPE FIELDS COUNT DIFFERENT THINGS, and asserting one rule for both is why an honest run
    was refused. ``required_public_dev_capsules`` is the ADMITTED cohort: `_task_runtime_scope` filters
    the public pool by the descriptor's named `graded_exclude`, so it already nets out exclusions.
    ``held_out_capsules`` is the SEALED POOL: hidden names are deliberately absent from the public
    descriptor (they would leak the holdout), so that count cannot be name-filtered and is instead tied
    to the hidden snapshot digest.

    Measured on gemmini (2026-09-03): public 48 source -> 11 capability + 3 resource excluded -> 34
    admitted, and the scope says 34, so it matched. Hidden 11 source -> 1 capability excluded ->
    10 admitted, and the scope says 11, so `held_out_capsules != n_hidden` refused a run whose only
    "gap" was `GH0_softmax_bf16_hidden` -- a bf16 capsule this target's frozen capability cannot
    execute, excluded by the same derived hardware predicate that excluded its 11 public bf16 siblings,
    and recorded with a digest over the excluded name set.

    So reconcile against the phase's own admission record instead of comparing raw totals. This is
    STRICTER where it counts: the exclusions must be recorded, digest-pinned, and add up, whereas a
    raw-total comparison would pass silently whenever two unrelated numbers happened to agree.
    """
    found: list[Deviation] = []
    pub_excluded, pub_found = _admission_shortfall(public_score, n_public,
                                                   label="public functional grade")
    hid_excluded, hid_found = _admission_shortfall(hidden_score, n_hidden,
                                                   label="hidden functional grade")
    found += pub_found + hid_found

    required_public = task_scope.get("required_public_dev_capsules")
    if (not isinstance(required_public, int) or isinstance(required_public, bool)
            or required_public <= 0 or required_public != n_public):
        found.append(Deviation("task_scope_public_mismatch",
                               f"public functional grade does not cover the sealed task scope: "
                               f"grade={n_public}, required={required_public!r}"))
    required_hidden = task_scope.get("held_out_capsules")
    if (not isinstance(required_hidden, int) or isinstance(required_hidden, bool)
            or required_hidden <= 0):
        found.append(Deviation("task_scope_hidden_mismatch",
                               f"sealed task scope declares no positive held-out count: "
                               f"{required_hidden!r}"))
    elif required_hidden != n_hidden + hid_excluded:
        found.append(Deviation("task_scope_hidden_mismatch",
                               f"hidden functional grade does not cover the sealed task scope: "
                               f"graded={n_hidden} + accounted exclusions={hid_excluded} "
                               f"!= sealed={required_hidden}"))
    return found


def inspect_functional_run(run_root: Path, run_id: str, expected_digest: str, *,
                           waive: "frozenset[str] | tuple[str, ...] | None" = None) -> FunctionalRun:
    """Validate one explicitly named, fully graded Arm-4 functional run.

    No directory search is performed: the caller supplies both the run ID and the whole-submission
    SHA-256, and all independently recorded digests plus the bytes on disk must agree with it.

    ``waive`` names COMPLETENESS predicates to accept as known gaps rather than refusals, so a
    campaign can be launched against a baseline that is incomplete in a stated way and still produce
    a real measurement. It is a set of predicate NAMES, never a boolean: a blanket force would waive
    whatever failed next, including something nobody looked at. Three rules keep it honest —

    * a predicate in :data:`UNWAIVABLE` cannot be waived at all, and asking is an error (those
      predicates decide whether the numbers mean anything, not how complete they are);
    * a waiver naming a predicate that did NOT fail is an error, because a stale waiver silently
      pre-authorises a future failure nobody has seen;
    * every accepted waiver is recorded on the returned :class:`FunctionalRun` as a
      :class:`Deviation` carrying what was actually observed, and travels with the result.

    Identity and structure (run id, path safety, digest agreement, absent evidence files) are never
    waivable and still raise immediately: without them there is no run to reason about.
    """
    requested = frozenset(waive or ())
    forbidden = sorted(requested & UNWAIVABLE)
    if forbidden:
        raise CampaignGateError(
            f"these gate predicates are UNWAIVABLE and cannot be overridden: {', '.join(forbidden)}. "
            f"They decide whether the measurement means anything (sandbox, answer mask, "
            f"answer-access audit, integrity, public/hidden identity separation), not how complete "
            f"the baseline is. A result produced with one of them waived would not be a weaker "
            f"result, it would be an unattributable one.")
    unknown = sorted(requested - _WAIVABLE_PREDICATES)
    if unknown:
        raise CampaignGateError(
            f"unknown gate predicate(s) in waive: {', '.join(unknown)}; "
            f"waivable predicates are: {', '.join(sorted(_WAIVABLE_PREDICATES))}")
    if not run_id:
        raise CampaignGateError("an explicit functional run id is required")
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise CampaignGateError("functional run id must be a simple directory name")
    if (len(expected_digest) != 64 or expected_digest.lower() != expected_digest
            or any(c not in "0123456789abcdef" for c in expected_digest)):
        raise CampaignGateError("functional submission digest must be an explicit lowercase SHA-256")

    arm_root = (Path(run_root) / "merlin_assisted").resolve()
    run_dir = arm_root / run_id
    if run_dir.is_symlink() or not run_dir.is_dir() or arm_root not in run_dir.resolve().parents:
        raise CampaignGateError(f"explicit functional run does not resolve safely: {run_dir}")
    run_dir = run_dir.resolve()
    submission = run_dir / "submission"
    if not submission.is_dir():
        raise CampaignGateError(f"functional submission is absent: {submission}")
    linked = _has_symlink(submission)
    if linked is not None:
        raise CampaignGateError(f"functional submission contains a live symlink: {linked}")
    excluded = _digest_excluded_path(submission)
    if excluded is not None:
        raise CampaignGateError(
            f"functional submission contains a digest-excluded path: {excluded}")

    environment = _mapping_file(run_dir / "environment.yaml", yaml_file=True)
    summary = _mapping_file(run_dir / "qa_loop_summary.yaml", yaml_file=True)
    freeze = _mapping_file(run_dir / "freeze.json")
    manifest = _mapping_file(run_dir / "run_manifest.yaml", yaml_file=True)
    public_score = _mapping_file(run_dir / "grading_public" / "score_capsule.json")
    hidden_score = _mapping_file(run_dir / "grading_hidden" / "score_capsule.json")

    if environment.get("run_id") != run_id or manifest.get("run_id") != run_id:
        raise CampaignGateError("functional evidence names a different run id")
    if not str(environment.get("bundle_id") or "").startswith("merlin_assisted_rtlchecks_"):
        raise CampaignGateError("functional run is not from the Arm-4 RTL-checks bundle")
    found: list[Deviation] = list(_validate_clean_run(environment, summary))
    if (manifest.get("integrity_status") != "clean" or manifest.get("integrity_exempt") is not False
            or manifest.get("gradeable") is not True):
        found.append(Deviation("manifest_integrity_gate_failed",
                               "functional run did not pass integrity and gradeability gates"))
    public = manifest.get("public_dev") or {}
    hidden = manifest.get("hidden") or {}
    if public.get("functional_pass") != 1 or hidden.get("functional_pass") != 1:
        found.append(Deviation("phase_functional_pass_missing",
                               "functional run did not pass both public and hidden grades"))
    if public.get("highest_tier") != "L3":
        found.append(Deviation("public_tier_below_cert",
                               f"functional public run did not reach the required L3 tier "
                               f"(highest_tier={public.get('highest_tier')!r})"))
    n_public, _pub_ratio_found = _graded_ratio(public.get("passed"),
                                               label="public functional grade")
    n_hidden, _hid_ratio_found = _graded_ratio(hidden.get("passed"),
                                               label="hidden functional grade")
    found += _pub_ratio_found + _hid_ratio_found
    task_scope = environment.get("task_scope")
    if not isinstance(task_scope, dict):
        raise CampaignGateError("functional run lacks its sealed task scope")
    found += _validate_scope_coverage(task_scope, public_score, hidden_score,
                                      n_public=n_public, n_hidden=n_hidden)
    found += _validate_score(public_score, label="public functional grade", expected=n_public)
    found += _validate_score(hidden_score, label="hidden functional grade", expected=n_hidden)
    _pub_rows = public_score.get("per_capsule") if isinstance(public_score, dict) else None
    _hid_rows = hidden_score.get("per_capsule") if isinstance(hidden_score, dict) else None
    if isinstance(_pub_rows, list) and isinstance(_hid_rows, list):
        _shared = ({str(r["capsule"]) for r in _pub_rows if isinstance(r, dict) and r.get("capsule")}
                   & {str(r["capsule"]) for r in _hid_rows if isinstance(r, dict) and r.get("capsule")})
        if _shared:
            found.append(Deviation("capsule_identity_reused",
                                   f"public and hidden grades reuse a capsule identity: "
                                   f"{', '.join(sorted(_shared)[:4])}"))
    if freeze.get("workspace_mutable_after_freeze") is not False or not freeze.get("frozen_at"):
        found.append(Deviation("freeze_not_immutable",
                               "functional freeze did not record an immutable workspace and "
                               "timestamp"))

    # Refuse everything not explicitly named, and name ALL of it in one message.
    blocking = [d for d in found if d.predicate not in requested]
    if blocking:
        _lines = "; ".join(f"[{d.predicate}] {d.detail}" for d in blocking)
        raise CampaignGateError(f"functional run refused by {len(blocking)} gate predicate(s): {_lines}")
    stale = sorted(requested - {d.predicate for d in found})
    if stale:
        raise CampaignGateError(
            f"waive names predicate(s) that did not fail: {', '.join(stale)}. A waiver for a "
            f"predicate nobody saw fail would silently pre-authorise a future failure; drop it.")

    observed = hash_tree(submission)["sha256"]
    recorded = {
        "requested": expected_digest,
        "run_manifest": manifest.get("submission_sha256"),
        "freeze": freeze.get("submission_sha256"),
        "freeze_recheck": freeze.get("submission_sha256_recheck"),
        "submission_tree": observed,
    }
    if any(value != expected_digest for value in recorded.values()):
        raise CampaignGateError(
            f"explicit functional submission digest does not match every record: {recorded}")
    return FunctionalRun(
        deviations=tuple(sorted(found, key=lambda d: d.predicate)),
        run_dir=run_dir,
        submission_dir=submission.resolve(),
        run_id=run_id,
        digest=expected_digest,
        public_capsules=n_public,
        hidden_capsules=n_hidden,
        public_score=public_score,
        hidden_score=hidden_score,
        frozen_at=str(freeze.get("frozen_at") or manifest.get("frozen_at") or ""),
    )


def materialize_perf_workspace(record: FunctionalRun, perf_root: Path) -> Path:
    """Copy the exact functional package into a new performance-only workspace."""
    perf_root = Path(perf_root).resolve()
    workspace = perf_root / "workspace"
    snapshot = workspace / "submission"
    observed = materialize_readonly_tree(record.submission_dir, snapshot)
    if observed != record.digest:
        raise CampaignGateError(
            f"copied performance submission digest {observed} does not match {record.digest}")
    return snapshot


def materialize_readonly_tree(source: Path, snapshot: Path) -> str:
    """Copy a symlink-free, fully hashed tree once and make the destination read-only."""
    source = Path(source).resolve()
    snapshot = Path(snapshot).resolve()
    if not source.is_dir():
        raise CampaignGateError(f"immutable campaign source is not a directory: {source}")
    if snapshot.exists() or snapshot.is_symlink():
        raise CampaignGateError(f"immutable campaign snapshot already exists: {snapshot}")
    linked = _has_symlink(source)
    if linked is not None:
        raise CampaignGateError(f"immutable campaign source contains a live symlink: {linked}")
    excluded = _digest_excluded_path(source)
    if excluded is not None:
        raise CampaignGateError(f"immutable campaign source contains a digest-excluded path: {excluded}")
    before = hash_tree(source)["sha256"]
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, snapshot, symlinks=False)
    linked = _has_symlink(snapshot)
    if linked is not None:
        raise CampaignGateError(f"immutable campaign snapshot contains a symlink: {linked}")
    observed = hash_tree(snapshot)["sha256"]
    if observed != before:
        raise CampaignGateError(
            f"immutable campaign copy changed during materialization: {before} != {observed}")
    for path in sorted(snapshot.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir():
            path.chmod(0o555)
        elif path.is_file():
            path.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)
    snapshot.chmod(0o555)
    return str(observed)


def functional_fork(record: FunctionalRun) -> ForkPoint:
    """Convert the public+hidden L2/L3 evidence into the immutable Phase-F fork record."""
    states: list[CapsuleState] = []
    tiers: set[str] = set()
    for score in (record.public_score, record.hidden_score):
        for row in score["per_capsule"]:
            passed = {str(tier): Verdict(str(status), record.digest)
                      for tier, status in (row.get("tiers") or {}).items() if status == "pass"}
            tiers.update(passed)
            states.append(CapsuleState(str(row["capsule"]), record.digest, verdicts=passed))
    if not states or not tiers:
        raise CampaignGateError("cannot fork from an empty functional verdict set")
    return fork_from(states, tier_order=sorted(tiers), digest=record.digest,
                     recorded_at=record.frozen_at or None)


def check_fork(fork: ForkPoint, snapshot: Path):
    """Check the copied compiler's current bytes against every Phase-F invariant."""
    digest = hash_tree(Path(snapshot))["sha256"]
    states = candidate_states(fork, digest=digest)
    return check_invariants(fork, states, provenance=fork.provenance)


def _remove_agent_home_mounts(argv: Sequence[str]) -> list[str]:
    """Remove the agent-launch credential bind from a package-only bwrap prefix.

    ``base_argv`` also serves paid agent launches and therefore binds ``~/.claude``.  A compiled
    submission never needs that state.  Strip both that bind and its nested projects mask, then clear
    the inherited environment so API/provider credentials cannot enter through environment variables.
    """
    home_claude = str(Path(os.path.expanduser("~/.claude")))
    out: list[str] = []

    def is_agent_home_path(value: str) -> bool:
        path = Path(value)
        home = Path(home_claude)
        return path == home or home in path.parents

    i = 0
    while i < len(argv):
        if argv[i] in ("--bind", "--ro-bind", "--bind-try", "--ro-bind-try") and i + 2 < len(argv):
            if is_agent_home_path(argv[i + 2]):
                i += 3
                continue
        if argv[i] == "--tmpfs" and i + 1 < len(argv) and is_agent_home_path(argv[i + 1]):
            i += 2
            continue
        out.append(argv[i])
        i += 1
    if any(is_agent_home_path(value) for value in out if value.startswith("/")):
        raise CampaignGateError("package sandbox still exposes the agent credential directory")
    return out


def package_sandbox_policy(te: TargetExperiment, workspace: Path,
                           package: Path) -> PackageSandboxPolicy:
    """Derive a credential-free, answer-closed bwrap policy for untrusted package entrypoints."""
    workspace = Path(workspace).resolve()
    package = Path(package).resolve()
    if not workspace.is_dir() or not package.is_dir() or workspace == package:
        raise CampaignGateError("performance sandbox workspace and copied package must be distinct directories")
    # Empty bundle is intentional: the package sees its input files and the derived toolchain, not the
    # experiment repository.  The policy-only escape avoids requiring an agent-input snapshot when no
    # bundle grants exist and does not expose any live input itself.
    argv = BW.base_argv(workspace, {}, _policy_test_live_inputs=True)
    argv = _remove_agent_home_mounts(argv)
    argv += ["--unshare-net", "--clearenv", "--setenv", "HOME", "/tmp",
             "--setenv", "XDG_RUNTIME_DIR", "/tmp/.xdg"]
    argv += TC.toolchain_binds(te)
    argv = _remove_agent_home_mounts(argv)
    surfaces = answer_surfaces(te)
    argv = BW.apply_answer_masks(argv, surfaces)
    argv += ["--ro-bind", str(package), str(package)]
    gaps = tuple(str(surface.path) for surface in BW.coverage_gap(argv, surfaces))
    if gaps:
        raise CampaignGateError(f"package sandbox exposes derived answer surfaces: {gaps}")
    probes = tuple(TC.required_tool_probes(te))
    if not probes:
        raise CampaignGateError("package sandbox derives zero required tool probes")
    return PackageSandboxPolicy(tuple(argv), gaps, probes, workspace, package, te)


def run_tool_probes(policy: PackageSandboxPolicy, *, timeout: int = 60) -> list[dict]:
    """Run every descriptor-derived tool probe in the exact package sandbox."""
    rows: list[dict] = []
    for probe in policy.required_tools:
        proc = subprocess.run(
            [*policy.argv, "bash", "-c", TC.sandbox_env(policy.target_experiment, policy.workspace)
             + probe.cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        row = {"label": probe.label, "returncode": proc.returncode,
               "stdout": (proc.stdout or "")[-400:], "stderr": (proc.stderr or "")[-400:]}
        rows.append(row)
        if proc.returncode != 0:
            raise CampaignGateError(
                f"required sandbox tool probe {probe.label!r} failed with rc={proc.returncode}")
    if len(rows) != len(policy.required_tools):
        raise CampaignGateError("not every required package-sandbox tool probe ran")
    return rows


def _inside(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


#: The policy the CALLING THREAD is currently boxed under, and the bookkeeping that lets several
#: threads be boxed at once. The patch below replaces two module globals in `oot_runner`, and
#: save/restore of a global is not composable: with two threads, the second saves the FIRST's
#: replacement as its "original", and when the first exits it restores the true originals -- leaving
#: the second thread's remaining package invocations running with NO sandbox at all, credentials
#: unmasked and the answer masks bypassed. Nothing would say so; the measurement would simply be
#: taken outside the box. So the policy is per-thread and the patch is installed once, under a lock,
#: for as long as any thread is inside it.
_ACTIVE = threading.local()
_PATCH_LOCK = threading.Lock()
_PATCH_DEPTH = 0
_ORIGINALS: dict = {}


def active_sandbox_policy() -> "PackageSandboxPolicy | None":
    """The policy boxing the calling thread, or None when it is not inside one."""
    return getattr(_ACTIVE, "policy", None)


@contextlib.contextmanager
def boxed_entrypoints(policy: PackageSandboxPolicy) -> Iterator[None]:
    """Route oot_runner's untrusted package execution through ``policy`` for the CALLING THREAD.

    Safe to enter from several threads at once: each one sees its own policy, and the module-level
    patch is installed once and removed only when the last of them leaves.
    """
    global _PATCH_DEPTH

    previous = getattr(_ACTIVE, "policy", None)
    _ACTIVE.policy = policy
    with _PATCH_LOCK:
        if _PATCH_DEPTH == 0:
            _ORIGINALS["run_entrypoint"] = oot_runner.run_entrypoint
            _ORIGINALS["build_package"] = oot_runner.build_package
            oot_runner.build_package = _refuse_build
            oot_runner.run_entrypoint = _run_boxed
        _PATCH_DEPTH += 1
    try:
        yield
    finally:
        _ACTIVE.policy = previous
        with _PATCH_LOCK:
            _PATCH_DEPTH -= 1
            if _PATCH_DEPTH == 0:
                oot_runner.run_entrypoint = _ORIGINALS["run_entrypoint"]
                oot_runner.build_package = _ORIGINALS["build_package"]


def _refuse_build(pkg, *, timeout: int = 1800) -> None:
    build = pkg.manifest.get("build") or {}
    if any(build.get(key) for key in ("configure", "command")):
        raise CampaignGateError(
            "Arm-4 performance package declares an untrusted build step; no host build is allowed")


def _run_boxed(pkg, name: str, input_mlir: Path, output_json: Path | None = None,
               *, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run one untrusted package entrypoint inside the CALLING THREAD's sandbox policy.

    Refusing when no policy is set is the whole point of reading it from the thread rather than from
    a closure: an invocation that arrives on a thread nobody boxed must not fall through to an
    unsandboxed execution, which is exactly what the previous save/restore could produce.
    """
    policy = active_sandbox_policy()
    if policy is None:
        raise CampaignGateError(
            "untrusted package execution reached a thread that is not inside a sandbox policy")
    if pkg.directory.resolve() != policy.package:
        raise CampaignGateError("capsule runner attempted to execute a package outside the snapshot")
    input_mlir = Path(input_mlir).resolve()
    output_json = Path(output_json).resolve() if output_json is not None else None
    if not _inside(input_mlir, policy.workspace):
        raise CampaignGateError("untrusted package input is outside the performance workspace")
    if output_json is not None and not _inside(output_json, policy.workspace):
        raise CampaignGateError("untrusted package output is outside the performance workspace")
    argv = oot_runner._resolve_argv(pkg, name, input_mlir, output_json)
    if oot_runner._needs_interpreter(pkg, argv):
        argv = [sys.executable, *argv]
    shell = TC.sandbox_env(policy.target_experiment, policy.workspace) + 'exec "$@"'
    return subprocess.run(
        [*policy.argv, "--chdir", str(policy.package), "bash", "-c", shell,
         "perf-package", *argv],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def completion_report(results: Sequence[Mapping], expected: Sequence[PerfCell]) -> dict:
    """Validate exact Phase-P cells without treating functional-simulator cycles as performance."""
    expected_rows = tuple(expected)
    for identity in expected_rows:
        if not isinstance(identity, PerfCell):
            raise CampaignGateError("performance expected identity must be a PerfCell")
        try:
            identity.validate()
        except PerfPromptContractError as exc:
            raise CampaignGateError(f"invalid expected performance identity: {exc}") from exc
    expected_cells = set(expected_rows)
    if not expected_cells:
        raise CampaignGateError("performance campaign has zero expected Arm-4 cells")
    if len(expected_cells) != len(expected_rows):
        raise CampaignGateError("performance expected cell identities contain duplicates")
    by_identity: dict[PerfCell, Mapping] = {}
    for row in results:
        if not isinstance(row, Mapping):
            raise CampaignGateError("performance result must be a mapping with an exact identity")
        identity = PerfCell(
            row.get("family"), row.get("capsule"), row.get("simulator"), row.get("replicate"))
        try:
            identity.validate()
        except PerfPromptContractError as exc:
            raise CampaignGateError(f"invalid reported performance identity: {exc}") from exc
        if identity in by_identity:
            raise CampaignGateError(f"performance results repeat cell identity {identity.label!r}")
        by_identity[identity] = row

    extras = sorted(set(by_identity) - expected_cells)
    if extras:
        raise CampaignGateError(
            f"performance results contain unexpected cell identities: {[cell.label for cell in extras]}")

    reported = correct = cycles_measured = 0
    failed = 0
    for identity in sorted(expected_cells):
        cell = by_identity.get(identity)
        if cell is None:
            continue
        reported += 1
        is_correct = cell.get("correct") is True
        cycles = cell.get("cycles")
        has_cycles = (identity.simulator == "verilator"
                      and isinstance(cycles, int) and not isinstance(cycles, bool) and cycles > 0)
        if identity.simulator == "spike":
            valid = is_correct and cycles is None
        else:
            provenance = cell.get("provenance") or {}
            valid = (is_correct and has_cycles and isinstance(provenance, Mapping)
                     and provenance.get("tier") == "L3"
                     and provenance.get("simulator") == "verilator"
                     and provenance.get("derived_from_rtl") is True
                     and provenance.get("cycle_accurate") is True)
        correct += int(is_correct)
        cycles_measured += int(has_cycles)
        failed += int(not valid)
    missing = len(expected_cells) - reported
    counts = {"expected": len(expected_cells), "reported": reported, "correct": correct,
              "cycles_measured": cycles_measured, "failed": failed, "missing": missing,
              "complete": not missing and not failed}
    return counts


def completion_counts(results: Sequence[Mapping], expected: Sequence[PerfCell]) -> dict:
    """Require every exact Arm-4 Phase-P identity to carry valid simulator-specific evidence."""
    counts = completion_report(results, expected)
    if not counts["complete"]:
        raise CampaignGateError(
            f"Arm-4 performance reported {counts['reported']} of {counts['expected']} expected cells; "
            f"{counts['failed']} reported cell(s) failed simulator-specific completion evidence")
    return counts
