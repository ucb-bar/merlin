"""Host-side grading of immutable candidate copies during Phase 1 authoring.

Owns snapshot, grade, evidence, promotion and redacted publication ordering.
Numerical grading remains in feedback.qa and the independent capsule evaluators.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import shutil
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import yaml

from merlin.common.cpu_pool import AUXILIARY, shared_compute_pool, with_priority
from merlin.common.tree_hash import hash_tree
from merlin.targetgen.target_experiment import load_target_experiment
from merlin.targetgen.tier_policy import required_tiers
from merlin_experiments.phase1 import oot_history as OH
from merlin_experiments.phase1 import run_inputs as RI
from merlin_experiments.phase1 import treatments as T
from merlin_experiments.phase1.context import InvocationContext, oracle_kwargs, readback_kwargs
from merlin_experiments.phase1.feedback import lifecycle as FL
from merlin_experiments.phase1.feedback import joint_test_scope as JTS


@dataclass(frozen=True)
class GradingInputs:
    """One invocation's public view, descriptor policy and grading contract.

    Public and policy roots remain distinct for fullsuite treatments.
    promotion_root retains the existing promotion API's explicit frozen-view
    override; None preserves its descriptor-owned legacy multi-root selection.
    Frozen runs pass an admitted public Path. Legacy diagnostic runs may supply
    a resolver: it is invoked only after the manifest/language gate, preserving
    the historical no-submission path without materializing a corpus.
    No construction/import discovers a target or publishes a corpus.
    """

    context: InvocationContext
    arm: str
    public_root: Path | Callable[[], Path]
    policy_roots: tuple[Path, ...]
    contract: Path | None
    promotion_root: Path | None = None
    additional_forbidden: tuple[str, ...] = ()
    public_build_service: object | None = None
    public_build_selection: dict | None = None
    public_build_budget_s: int = 0
    compiler_profile_service: object | None = None
    verify_grade_inputs: Callable[[], None] | None = None
    driver_grade_completed: Callable[[object], None] | None = None
    rtl_facts: Path | None = None


def _history_runs_roots(run_dir: Path) -> tuple[Path, ...]:
    """This run's earlier grade and selfcheck runs roots; they only order capsule work."""
    work = Path(run_dir) / "_qa_work"
    roots = [*sorted(work.glob("runs_*")), *sorted(work.glob("fruns_*")), *sorted(work.glob("selfchecks/*"))]
    return tuple(root for root in roots if root.is_dir() and not root.is_symlink())


def _attribute_source(verdict: dict, ws: Path, snapshot_digest, *, stage: str) -> dict:
    """Separate immutable measured rows from eligibility of the observed live source."""
    verdict = dict(verdict)
    verdict.setdefault(
        "snapshot_result",
        {
            "scope": stage,
            **{
                key: verdict.get(key)
                for key in (
                    "all_pass",
                    "gradeable",
                    "n_passed",
                    "n_capsules",
                    "structural_evidence_complete",
                )
            },
        },
    )
    try:
        current = hash_tree(ws / "submission")["sha256"]
    except OSError:
        current = None
    matches = snapshot_digest is not None and current == snapshot_digest
    verdict.update(
        graded_source_sha256=snapshot_digest,
        current_source_sha256=current,
        source_hash_algorithm="merlin.common.tree_hash.hash_tree",
        source_checked_at=_dt.datetime.now(_dt.UTC).isoformat(),
        snapshot_current=matches,
        current_source_eligible=matches,
        current_source_eligibility_scope="source identity only; all mandatory checks still apply",
        count_scope="immutable_snapshot",
        current_source_n_passed=verdict.get("n_passed") if matches else None,
    )
    if not matches:
        verdict.update(all_pass=None, gradeable=False)
    return verdict


def compiler_execution(inputs, *, candidate, run_dir):
    from .compiler_profile_runtime import original_compiler_execution

    return original_compiler_execution(
        inputs.compiler_profile_service,
        target_experiment=(
            None if inputs.compiler_profile_service is None else load_target_experiment(inputs.context.descriptor)
        ),
        candidate=candidate,
        work_root=Path(run_dir).resolve() / "_qa_work/profile_package_execution",
    )


def declared_loop_tiers(policy_roots: tuple[Path, ...]) -> set:
    """The oracle tiers THIS target's graded corpus declares in ``required_oracle_tiers``.

    Read from the corpus roots (not the materialized subset, which is derived FROM this) and handed to
    ``qa_loop_adapters`` so the per-round loop tier is one the capsules actually declared, rather than
    whichever endpoint tier happens to be fastest."""
    from merlin.targetgen.contract.materialize import declared_oracle_tiers

    try:
        return declared_oracle_tiers(*policy_roots)
    except Exception:  # noqa: BLE001 — no resolvable corpus -> legacy fastest-tier behavior
        return set()


def cert_tiers_beyond_loop(
    *,
    context: InvocationContext,
    policy_roots: tuple[Path, ...],
    public_roots: Callable[[], Path | tuple[Path, ...] | list[Path]],
) -> tuple[set, set]:
    """``(cert tiers held back from the fast loop, the subset the pilot corpus makes MANDATORY)``.

    One derivation, shared by the checkpoint gate and by the fast first grade, so the two can never
    disagree about which tier a verdict is missing. Target-agnostic: the ladders come from
    ``qa_checkpoint_adapters`` / ``qa_loop_adapters`` for THIS target and the mandatory set from the
    capsules' own ``required_oracle_tiers`` — no sim name is written down here.

    Resolve public roots only after finding an additional cert tier. Native
    diagnostic callers may lazily materialize their legacy corpus at that point;
    admitted grading callers return their already-resolved immutable public view.
    """
    from merlin.targetgen import capsule_runner as _CR

    try:
        te = load_target_experiment(context.descriptor)
        ck = _CR.qa_checkpoint_adapters(te.target, te.sim_via, **oracle_kwargs(context))
        loop = _CR.qa_loop_adapters(
            te.target, te.sim_via, declared_tiers=declared_loop_tiers(policy_roots), **oracle_kwargs(context)
        )
    except Exception:  # noqa: BLE001 — no resolvable checkpoint oracle -> nothing to gate on
        if context.readback_policy is not None:
            # An explicit transport must never silently remove certification tiers.
            raise
        ck, loop = {}, {}
    cert_tiers = set(ck) - set(loop)  # the cycle-accurate cert tiers held back from the fast loop
    if not cert_tiers:
        return set(), set()
    mandatory: set = set()
    _roots = public_roots()
    for _r in _roots if isinstance(_roots, (list, tuple)) else [_roots]:
        for cf in Path(_r).rglob("capsule.yaml"):
            try:
                doc = yaml.safe_load(cf.read_text()) or {}
            except Exception:  # noqa: BLE001
                continue
            mandatory |= set(required_tiers(doc) or [])
    return cert_tiers, cert_tiers & mandatory


def submission_language_ok(submission_dir, arm: str) -> tuple[bool, str]:
    """ENFORCE the arm language contract on an EMITTED submission: a merlin arm (arm-3/arm-4) must build
    its dialect with the xDSL/Python kit — NOT a hand C++/TableGen backend. Returns (ok, reason). Reads
    the submission's manifest.yaml: for a merlin arm it fails if ``language: cpp`` or a build block that
    compiles a C++ tool (cmake / mlir-tblgen / a *-opt binary) is present. The C++ arms are exempt (that
    is their mandated method). Pure + target-agnostic — the grader/driver calls this to reject a
    non-compliant round with an actionable reason instead of grading a forbidden backend."""
    from pathlib import Path

    import yaml

    if "merlin_assisted" not in arm:  # only the merlin (xDSL) arms are constrained
        return True, "not a merlin arm (no xDSL mandate)"
    mpath = Path(submission_dir) / "mlir_oot" / "manifest.yaml"
    if not mpath.is_file():
        mpath = next(iter(sorted(Path(submission_dir).rglob("manifest.yaml"))), None)
    if not mpath or not mpath.is_file():
        return False, "no manifest.yaml in submission"
    m = yaml.safe_load(mpath.read_text()) or {}
    lang = str(m.get("language", "")).strip().lower()
    if lang in ("cpp", "c++", "cxx"):
        return False, f"merlin arm must use xDSL/Python, manifest declares language: {lang}"
    build = m.get("build") or {}
    blob = " ".join(str(v) for v in (build.values() if isinstance(build, dict) else [build])).lower()
    for marker in ("cmake", "mlir-tblgen", "tblgen", "clang++", "g++"):
        if marker in blob:
            return False, f"merlin arm must use xDSL/Python, build block invokes {marker!r} (C++ toolchain)"
    return True, f"xDSL/Python (language={lang or 'python'})"


def _language_ok(submission_dir: Path, arm: str) -> tuple[bool, str]:
    """Enforce the current arm's language mandate on the emitted submission (merlin arms => xDSL/Python,
    not a hand C++ backend). Uses the submission language policy; degrades to OK if the
    check is unavailable so it never spuriously blocks a run."""
    try:
        return submission_language_ok(submission_dir, arm)
    except Exception:  # noqa: BLE001 — never let the compliance check itself break grading
        return True, "language check unavailable"


def grade(
    ws: Path,
    run_dir: Path,
    rnd: int,
    no_oracle: bool,
    timeout: int,
    label: str = "round",
    *,
    inputs: GradingInputs,
    scratch_key: str | None = None,
    previous_scratch_key: str | None = None,
    qa_runner: T.QARunner | None = None,
    advisory_background: bool = False,
) -> dict:
    """Copy the agent's submission to an operator-only scratch, grade it, return + persist the
    redacted verdict (into ws/qa/verdict.json for the next round, and archived per round).

    ``label`` names the archive namespace. It stays ``"round"`` — byte-identical file names — for every
    round grade, so the per-round trajectory readers (`gen_trajectory_v2`, `abc_status`, which glob
    ``verdict_round_*.json``) keep seeing exactly the rounds. A grade taken WHILE a turn is still running
    is not a round and is filed under its own label, so it cannot be mistaken for one."""
    if advisory_background and JTS.focused() and not no_oracle:
        import time
        # Interval grades use the same bounded native Spike path as first-grade.
        # A unique private snapshot avoids clashing with the first-grade tick.
        from .snapshot_publication import current_token

        return fast_grade(ws, run_dir, int(current_token() or time.time_ns()), timeout, inputs=inputs)
    # In-turn tick numbers restart for every agent round. Keying scratch only by ``rnd`` therefore
    # reuses the previous round's sealed, read-only ``cand_901/submission`` and fails before grading
    # with PermissionError. Keep the numeric id for logs, but let the caller supply a run-wide scratch
    # identity that includes both the agent round and its tick.
    _key = scratch_key or (f"{rnd:02d}" if label == "round" else str(rnd))
    if inputs.driver_grade_completed is not None:
        inputs.driver_grade_completed(None)
    background_guard = None
    progress_rows = {}
    progress_path = ws / "qa" / "progress.json"

    def publish_progress(row):
        # A separate redacted stream cannot replace the final corpus verdict.
        from .qualification_summary import summarize

        progress_rows[row["capsule"]] = row
        qualification = summarize(list(progress_rows.values()))
        current = background_guard.current()
        progress_path.parent.mkdir(exist_ok=True)
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema": "merlin.phase1.grade_progress.v1",
                    "state": "running",
                    "all_pass": None,
                    "certified_complete": False,
                    "graded_source_sha256": snapshot_digest,
                    "snapshot_current": current,
                    "count_scope": "immutable_snapshot",
                    "current_source_n_passed": sum(r.get("status") == "pass" for r in progress_rows.values())
                    if current
                    else None,
                    "n_completed": len(progress_rows),
                    "n_passed": sum(r.get("status") == "pass" for r in progress_rows.values()),
                    "n_execution_passed": qualification["execution_passed"],
                    "qualification_summary": qualification,
                    "per_capsule": list(progress_rows.values()),
                    "updated_at": _dt.datetime.now(_dt.UTC).isoformat(),
                    "scope": "completed redacted rows; final shape, model and certification gates remain pending",
                },
                indent=2,
            )
            + "\n"
        )
        temporary.replace(progress_path)

    def publish_pending_shape():
        # Capsule rows for this snapshot are complete; the mandatory shape gate is not. The interim
        # document can never converge anything: all_pass stays unknown (or a failure stays a failure)
        # until the probes land and the final verdict replaces this one.
        interim = dict(verdict)
        if interim.get("all_pass") is not False:
            interim["all_pass"] = None
        interim["shape_coverage"] = {
            "ran": False,
            "status": "pending",
            "note": "the required shape-coverage probes are still running for this snapshot; "
            "all_pass is unknown until they finish and the final verdict replaces this one.",
        }
        interim["graded_request_sha256"] = background_guard.source
        interim = _attribute_source(interim, ws, snapshot_digest, stage="capsule_evidence_before_shape_probes")
        write_verdict(ws / "qa" / "verdict.json", interim)
        if progress_rows:
            progress = json.loads(progress_path.read_text())
            progress.update(
                state="shape_coverage_pending",
                snapshot_current=interim["snapshot_current"],
                current_source_n_passed=interim["current_source_n_passed"],
            )
            temporary = progress_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(progress, indent=2) + "\n")
            temporary.replace(progress_path)

    background_workers = 0
    if advisory_background:
        from .background_admission import BackgroundGuard, source_digest

        public = inputs.public_root() if callable(inputs.public_root) else inputs.public_root
        background_guard = BackgroundGuard(ws, public, inputs, source_digest(ws),
                                           finish_bounded_snapshot=JTS.focused())
        raw = os.environ.get("MERLIN_BACKGROUND_GRADE_WORKERS", "0")
        if not raw.isascii() or not raw.isdecimal() or not 0 <= int(raw) <= 64:
            raise ValueError("MERLIN_BACKGROUND_GRADE_WORKERS must be an integer between 0 and 64")
        background_workers = int(raw)
    oot_commit = None
    snapshot_digest = None
    cand = run_dir / "_qa_work" / f"cand_{_key}" / "submission"
    if cand.exists():
        shutil.rmtree(cand.parent)
    _lok, _lwhy = _language_ok(ws / "submission", inputs.arm)
    _indirect = None
    if (ws / "submission" / "manifest.yaml").exists() and _lok:
        try:
            with shared_compute_pool(timeout_s=timeout):
                RI.copy_candidate_tree(ws / "submission", cand, ignore=RI.SUBMISSION_IGNORED)
        except RI.IndirectSubmission as exc:
            _indirect = str(exc)
    if _indirect is not None:
        verdict = {
            "all_pass": False,
            "n_passed": 0,
            "n_capsules": 0,
            "package_failure": {"plane": "schema", "detail": _indirect},
            "per_capsule": [],
            "note": "replace every symbolic link or special file in submission/ with an ordinary file.",
        }
    elif not (ws / "submission" / "manifest.yaml").exists():
        # ZERO, NOT A PLACEHOLDER. These two stubs are "nothing was graded", and they used to say
        # `n_capsules: 4`, which reads as a real cohort of four: `0/4` appears in the round log
        # identically to a genuine grade, and `per_capsule: []` is the only thing distinguishing them.
        # Measured cost of that: a reader tracked a live run for an hour believing its graded
        # denominator had collapsed from 96 to 4. A count of nothing graded is 0.
        verdict = {
            "all_pass": False,
            "n_passed": 0,
            "n_capsules": 0,
            "package_failure": {"plane": "schema", "detail": "no submission/manifest.yaml"},
            "per_capsule": [],
            "note": "write submission/manifest.yaml first.",
        }
    elif not _lok:
        # ENFORCE the arm language mandate: a merlin arm must build with the xDSL kit, not a C++ tool.
        # A non-compliant submission gets a failing verdict with the fix reason (never silently graded).
        verdict = {
            "all_pass": False,
            "n_passed": 0,
            "n_capsules": 0,
            "package_failure": {"plane": "language", "detail": _lwhy},
            "per_capsule": [],
            "note": f"language mandate: {_lwhy}. Use the xDSL/Python kit "
            "(oot_starterkit + xdsl_dialects), declare language: python, no C++/cmake build.",
        }
    else:
        with shared_compute_pool(timeout_s=timeout):
            RI.strip_build_state(cand)  # clean, relocatable build per grade (abc9 L3-build bug)
            snapshot_digest = hash_tree(cand)["sha256"]
        # The harness commits exactly the bytes about to be graded; the agent never touches the repo.
        oot_commit = OH.commit_graded(run_dir, cand, label=label, key=_key, sandbox_roots=(ws.parent,))
        with shared_compute_pool(timeout_s=timeout):
            candidate_sha256 = snapshot_digest if inputs.public_build_service is not None else None
        public_root = inputs.public_root() if callable(inputs.public_root) else inputs.public_root
        public_resources = None
        if not JTS.focused() and (background_guard is None or background_guard.current()):
            from .public_scalability_gate import observe_authoring

            public_resources = observe_authoring(
                owner=inputs.compiler_profile_service,
                candidate=cand,
                context=inputs.context,
                capsules_root=public_root,
                contract=inputs.contract,
                facts_path=inputs.rtl_facts,
                work_root=run_dir.absolute() / "_qa_work" / f"public_scalability_{_key}",
            )
            if public_resources is not None:
                # Publish this bounded source-only result before the first
                # simulator. It is not a capsule verdict or numerical grade.
                feedback = ws / "qa" / "public_codegen_resources.json"
                feedback.parent.mkdir(exist_ok=True)
                pending = feedback.with_suffix(".pending")
                pending.write_text(json.dumps(public_resources, indent=2) + "\n")
                pending.replace(feedback)
        out = run_dir / "qa_history" / f"verdict_{label}_{_key}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        argv = [
            "--submission",
            str(cand),
            "--capsules-root",
            str(public_root),
            "--out",
            str(out),
            "--runs-root",
            str(run_dir / "_qa_work" / f"runs_{_key}"),
            "--timeout",
            str(timeout),
        ]
        if no_oracle:
            argv.append("--no-oracle")
        from merlin.targetgen.capsule_scheduling import timing_history
        from merlin_experiments.phase1.feedback import qa

        with compiler_execution(inputs, candidate=cand, run_dir=run_dir), timing_history(_history_runs_roots(run_dir)):
            verdict = (qa_runner or partial(qa.run, context=inputs.context))(
                str(cand),
                str(public_root),
                run_dir / "_qa_work" / f"runs_{_key}",
                {"public", "dev"},
                no_oracle,
                timeout,
                **({"contract": inputs.contract} if inputs.contract is not None else {}),
                **({"additional_forbidden": inputs.additional_forbidden} if inputs.additional_forbidden else {}),
                **(
                    {"defer_pending": background_guard.pending_reason, "defer_running": False}
                    if background_guard is not None
                    else {}
                ),
                **(
                    {"on_capsule_complete": publish_progress}
                    if background_guard is not None and os.environ.get("MERLIN_BACKGROUND_PROGRESS") == "1"
                    else {}
                ),
                **({"max_workers": background_workers} if background_workers else {}),
            )
            if public_resources is not None:
                verdict["public_codegen_resources"] = public_resources
                if (
                    public_resources["required"]
                    and not public_resources["resource_complete"]
                    and verdict.get("all_pass") is True
                ):
                    verdict["all_pass"] = False
                    verdict["not_converged_reason"] = "selected public compilation resource ladder is incomplete"
            from .public_scalability import attach as attach_scalability

            if JTS.focused():
                verdict["public_scalability"] = {"status": "pending", "reason": "use focused scalability selfchecks; full ladder at milestone"}
            else:
                attach_scalability(
                    verdict,
                    cand,
                    context=inputs.context,
                    output=run_dir / "_qa_work" / f"scalability_{_key}",
                    timeout=timeout,
                    contract=inputs.contract,
                    additional_forbidden=inputs.additional_forbidden,
                    max_workers=background_workers,
                )
            verdict = _attribute_source(verdict, ws, snapshot_digest, stage="capsule_evidence_before_shape_probes")
            if background_guard is not None and background_guard.reason() is not None:
                verdict.update(
                    all_pass=None,
                    gradeable=False,
                    n_capsules=len(verdict.get("per_capsule") or []),
                    background_grade={
                        "status": "superseded_snapshot" if background_guard.finish_bounded_snapshot else "deferred",
                        "reason": background_guard.deferred,
                        "source_sha256": background_guard.source,
                    },
                )
            if candidate_sha256 is not None:
                verdict["graded_candidate_sha256"] = candidate_sha256
            verdict["graded_source_sha256"] = snapshot_digest
            verdict = write_verdict(out, verdict)
            _write_stage_ledger(
                run_dir,
                rnd,
                cand,
                run_dir / "_qa_work" / f"runs_{_key}",
                verdict,
                artifact_key=scratch_key,
                previous_artifact_key=previous_scratch_key,
            )
            shape_skip = _shape_skip_reason(verdict)
            if background_guard is not None and not background_guard.current():
                verdict.update(all_pass=None, gradeable=False)
                verdict["shape_coverage"] = {
                    "ran": False,
                    "reason": "superseded source; required probes deferred to current or final grade",
                }
            elif shape_skip is not None:
                _skip_shape_generalization(verdict, shape_skip)
            else:
                _attach_shape_generalization(
                    verdict,
                    cand,
                    run_dir,
                    rnd,
                    timeout=timeout,
                    artifact_key=scratch_key,
                    context=inputs.context,
                    contract=inputs.contract,
                    additional_forbidden=inputs.additional_forbidden,
                    **(
                        {"compiler_profile_service": inputs.compiler_profile_service}
                        if inputs.compiler_profile_service is not None
                        else {}
                    ),
                    **(
                        {"before_probe": publish_pending_shape}
                        if background_guard is not None and os.environ.get("MERLIN_BACKGROUND_PROGRESS") == "1"
                        else {}
                    ),
                )
            if JTS.focused() or (background_guard is not None and (
                not background_guard.current() or os.environ.get("MERLIN_BACKGROUND_CODEGEN_SCALABILITY") == "0"
            )):
                verdict["codegen_scalability"] = {"ran": False, "reason": "advisory diagnostic deferred to final grade"}
            else:
                _attach_codegen_scalability(
                    verdict,
                    cand,
                    run_dir,
                    rnd,
                    timeout=timeout,
                    artifact_key=scratch_key,
                    context=inputs.context,
                    contract=inputs.contract,
                    additional_forbidden=inputs.additional_forbidden,
                    **(
                        {
                            "build_service": inputs.public_build_service,
                            "build_selection": inputs.public_build_selection,
                            "build_budget_s": inputs.public_build_budget_s,
                            "candidate_sha256": candidate_sha256,
                        }
                        if inputs.public_build_service is not None
                        else {}
                    ),
                )
            verdict = _attribute_source(verdict, ws, snapshot_digest, stage="capsule_evidence_before_shape_probes")
            completed_out = out.with_name(f"completed_{label}_{_key}.json")
            verdict = write_verdict(completed_out, verdict)
            if inputs.driver_grade_completed is not None and not no_oracle:
                from .grade_reuse import retain_driver_grade

                with shared_compute_pool(timeout_s=timeout):
                    receipt = retain_driver_grade(
                        candidate=cand,
                        runs=run_dir / "_qa_work" / f"runs_{_key}",
                        public=public_root,
                        verdict=verdict,
                        inputs=inputs,
                        run_dir=run_dir,
                        oot_commit=oot_commit.commit if oot_commit is not None else None,
                        verdict_path=completed_out,
                        evidence_paths=tuple(
                            path
                            for path in (
                                out.parent / f"shape_coverage_round_{scratch_key or f'{rnd:02d}'}.json",
                                out.parent / f"codegen_scalability_round_{scratch_key or f'{rnd:02d}'}.json",
                                out.parent / f"codegen_scalability_public_round_{scratch_key or f'{rnd:02d}'}.json",
                                run_dir / "_qa_work" / f"public_scalability_{_key}",
                            )
                            if path.exists()
                        ),
                    )
                inputs.driver_grade_completed(receipt)
        _record_plateau(run_dir, current_source_sha256=verdict.get("current_source_sha256"))
        FL.record_channel_health(ws, run_dir)
    # PROMOTE off the round grade too. Promotion is hooked into both BROKERS, but a broker only sees a
    # verdict the agent ASKED for -- and a converged agent stops asking. Measured on the run that
    # motivated this: 24 self-checks in round 0, then ZERO in rounds 1 and 2 once it reached the corpus
    # ceiling, so the only verdict produced in those rounds was this one and promotion had nothing to fire
    # on. Three paths produce a verdict; all three must consider promotion, or the deeper tier is only
    # ever reached while the agent is still struggling -- which is exactly backwards, since a converged
    # submission is the one worth certifying.
    # Runs AFTER _attach_shape_generalization so promotion considers the completed verdict.
    try:
        import sys as _sys

        from merlin_experiments.phase1.feedback.promotion import promote as _promote
        from merlin_experiments.phase1.feedback.promotion import resolve_tiers as _resolve

        _loop, _cert, _cover = _resolve(ws, context=inputs.context, capsules_root=inputs.promotion_root)
        if (
            _loop
            and _cert
            and isinstance(verdict, dict)
            and verdict.get("per_capsule")
            and (background_guard is None or background_guard.deferred is None)
        ):
            _p = _promote(
                ws,
                ws / ".qa_channel",
                verdict,
                _loop,
                _cert,
                _cover,
                _sys.stderr,
                source_ws=cand.parent,
                context=inputs.context,
                capsules_root=inputs.promotion_root,
            )
            if _p:
                print(f"  [promote] round grade -> {_cert}: {_p}", flush=True)
    except Exception as _pe:  # noqa: BLE001 -- promotion is an optimisation, never a gate
        print(f"  [promote] skipped: {type(_pe).__name__}: {_pe}", flush=True)

    # hand the redacted verdict to the agent for the next round
    qa_dir = ws / "qa"
    qa_dir.mkdir(exist_ok=True)
    if background_guard is not None:
        verdict["graded_request_sha256"] = background_guard.source
    verdict = _attribute_source(verdict, ws, snapshot_digest, stage="capsule_evidence_before_shape_probes")
    # The immutable snapshot has a valid result even when the agent has since edited
    # its source. Suppressing that result left R52 without qa/verdict.json for hours.
    verdict = write_verdict(qa_dir / "verdict.json", verdict)
    if progress_rows:
        progress = json.loads(progress_path.read_text())
        progress.update(
            state="complete",
            snapshot_current=verdict.get("snapshot_current"),
            current_source_n_passed=verdict.get("current_source_n_passed"),
        )
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, indent=2) + "\n")
        temporary.replace(progress_path)
    if oot_commit is None:
        return verdict
    OH.record_round(run_dir, oot_commit, label=label, key=_key, verdict=verdict)
    return {**verdict, "oot_commit": oot_commit.commit}  # host-side round record; not the agent's copy


# --- the agent must never run a whole turn BLIND -------------------------------------------------
# MEASURED on merlincirct_arm4_func_20260901_codex1: the agent ran its ENTIRE 6184-second turn with no
# `qa/` directory at all, and said so itself in its closing message ("qa/verdict.json was not present").
# The round's verdict is produced AFTER the turn, and the mandatory ladder that produces it includes the
# cycle-accurate cert tier — one capsule (GM0) alone cost 1818s of Verilator — so under a long turn
# (`--schedule continuous --round-timeout 43200`) the first feedback an agent can read is hours away, or
# never. A blind agent is not a measurement of the agent.
#
# The fix is to make the FIRST grade fast and cheap: the LOOP tier only, published within minutes, then
# the expensive full-ladder grade on the normal --grade-interval. What the fast grade must never do is
# claim more than it ran: the cert tiers it withheld are NAMED in `tiers_not_run`, and `all_pass` is
# `null` (UNKNOWN) whenever a MANDATORY tier was withheld — never `true`. A check that could not run
# reading as SUCCESS is a recurring bug in this repo; this is not another instance of it.


def _fast_loop_verdict_doc(
    red: dict,
    tiers_graded,
    tiers_withheld,
    *,
    structural_evidence=None,
    cohort_complete=False,
) -> dict:
    """Build the fast first verdict from the per-capsule readback. PURE, so its honesty is testable.

    `n_passed` counts capsules that passed THE TIERS THAT RAN, and the document says so. `all_pass` is a
    bool only when nothing mandatory was withheld; otherwise it is None — the JSON null that reads as
    UNKNOWN to every consumer and is falsy to every gate, so no downstream check can mistake a partial
    ladder for a converged run.
    """
    rows: list = []
    npass = 0
    for name, info in sorted(red.items()):
        info = info or {}
        npass += int(info.get("status") == "pass")
        rows.append(
            {
                "capsule": name,
                "status": info.get("status"),
                "tiers": info.get("tiers") or {},
                "numeric_status": info.get("numeric_status"),
                "mismatch_count": info.get("mismatch_count"),
                "trace_status": info.get("trace_status"),
                "trace_requirement_status": info.get("trace_requirement_status"),
                "trace_unavailable_requirements": info.get("trace_unavailable_requirements") or [],
                "trace_violations": info.get("trace_violations") or [],
                "failure_plane": info.get("failure_plane"),
                "failure_category": info.get("failure_category"),
                "failure_detail": info.get("failure_detail"),
            }
        )
    graded, withheld = sorted(tiers_graded), sorted(tiers_withheld)
    complete = not withheld
    # The grader joins original trace, provided-host, model and refusal scopes.
    # A trace-only shortcut would invent obligations for host/refusal rows.
    structural_complete = structural_evidence is True
    nc = len(rows)
    return {
        "qa_gate": "capsule_bench_v0_pilot",
        "stage": "first_grade_loop_tier",
        "gradeable": True,
        "tiers_graded": graded,
        "tiers_not_run": withheld,
        "mandatory_tiers_complete": complete,
        # UNKNOWN, never success: a mandatory tier that did not run may not be reported as passed.
        "all_pass": (
            bool(nc > 0 and npass == nc and structural_complete and cohort_complete is True) if complete else None
        ),
        "original_cohort_complete": cohort_complete is True,
        "structural_evidence_complete": structural_complete,
        "n_passed": npass,
        "n_capsules": nc,
        "per_capsule": rows,
        "note": (
            "FAST FIRST GRADE — the LOOP tier only, published early so you are not working blind. "
            f"Tiers graded: {graded or 'none'}. Tiers NOT run: {withheld or 'none'} — those are "
            "UNKNOWN, not passed; `all_pass` is null while any mandatory tier is outstanding. The "
            "full mandatory-ladder grade follows on the normal interval and is the one that "
            "converges the run. Contains NO reference output values."
        ),
    }


def _completed_loop_promoter(ws, frozen_source, runs, loop, cert, cover, *, inputs):
    """Promote only closed, durable selected public/dev functional passes."""
    import sys

    from merlin.targetgen.capsule_common import discover_capsules
    from merlin_experiments.phase1.feedback import promotion, qa, tier_boundary_join

    te = load_target_experiment(inputs.context.descriptor)
    root = inputs.public_root() if callable(inputs.public_root) else inputs.public_root
    with shared_compute_pool():
        caps = discover_capsules((root,), labels={"public", "dev"}, contract=inputs.contract)
    excluded = set(te.effective_exclusions(cap.get("name") for cap in caps))
    allowed = {str(cap["name"]) for cap in caps if cap.get("name") and cap["name"] not in excluded}
    by_name = {str(cap["name"]): cap for cap in caps if cap.get("name")}

    def complete(result):
        name = result.get("capsule")
        from . import joint_physical_correctness as physical
        from .joint_curriculum import _passed
        expected_refusal = bool(physical.roster()) and _passed(result) == 'expected_refusal'
        if name not in allowed or ((result.get("tiers", {}).get(loop) or {}).get("status") != "pass"
                                   and not expected_refusal):
            return
        # A callback observes a completed result, not an in-flight file or candidate
        # supplied row. Reopen exactly its published record before attributing it.
        path = Path(runs) / "runs" / _suite_for(te.target) / name / "capsule_result.json"
        try:
            if json.loads(path.read_text()) != json.loads(json.dumps(result, allow_nan=False)):
                raise ValueError("completed capsule differs from its durable result")
            if result.get("kind") == "model" and loop == "L2" and cert == "L3":
                from merlin_experiments.phase1.feedback import model_gate_vouchers

                model_gate_vouchers.issue(
                    ws,
                    frozen_source,
                    result,
                    capsule=by_name[name],
                    context=inputs.context,
                    cert_tier=cert,
                    receipt_path=path,
                    contract=inputs.contract,
                )
            promotion.promote(
                ws,
                Path(ws) / ".qa_channel",
                {
                    "per_capsule": [
                        {
                            "capsule": name,
                            "pass": True,
                            "tiers": {loop: "pass"},
                            "execution_digest": qa._execution_digest_from_result(path),
                            "tier_boundary": tier_boundary_join.optional_observation(path, tier=loop)
                                if name in physical.roster() and not expected_refusal else None,
                        }
                    ]
                },
                loop,
                cert,
                cover,
                sys.stderr,
                source_ws=frozen_source,
                context=inputs.context,
                capsules_root=inputs.promotion_root,
                closed_loop_rows={name: result},
            )
        except Exception as exc:  # noqa: BLE001 -- completed feedback survives queue faults
            print(f"[first-grade] completed promotion skipped: {type(exc).__name__}: {exc}", flush=True)

    return complete


def _suite_for(target):
    from merlin.targetgen.capsule_runner import suite_for

    return suite_for(target)


def fast_grade(
    ws: Path,
    run_dir: Path,
    tick: int,
    timeout: int,
    *,
    inputs: GradingInputs,
) -> dict:
    """Grade the current submission at the LOOP tier ONLY and publish it as the agent's verdict.

    Cheap by construction: `qa_loop_adapters` is the fast functional tier the corpus itself declares, so
    this costs seconds-to-minutes where the checkpoint ladder costs tens of minutes per capsule. Same
    snapshot-copy discipline as `qa_grade`, so it is safe to run while the agent is editing.
    """
    from .background_admission import SubmissionNotReady

    if not (ws / "submission" / "manifest.yaml").is_file():
        raise SubmissionNotReady("no submission/manifest.yaml to grade yet")

    from merlin.targetgen import capsule_grade as _CG
    from merlin.targetgen import capsule_runner as _CR
    from merlin_experiments.phase1.feedback import qa as _qc

    from .background_admission import BackgroundGuard, source_digest
    from .grade_progress import SnapshotProgress

    cand = run_dir / "_qa_work" / f"fcand_{tick}" / "submission"
    public_root = inputs.public_root() if callable(inputs.public_root) else inputs.public_root
    background_guard = BackgroundGuard(ws, public_root, inputs, source_digest(ws),
                                       finish_bounded_snapshot=JTS.focused())
    with shared_compute_pool(timeout_s=timeout):
        if cand.parent.exists():
            shutil.rmtree(cand.parent)
        RI.copy_candidate_tree(ws / "submission", cand, ignore=RI.SUBMISSION_IGNORED)
        RI.strip_build_state(cand)
    snapshot_digest = hash_tree(cand)["sha256"]
    fruns = run_dir / "_qa_work" / f"fruns_{tick}"
    te = load_target_experiment(inputs.context.descriptor)
    from .. import joint_control
    adapters = _CR.qa_loop_adapters(
        te.target,
        te.sim_via,
        declared_tiers={"L2"} if joint_control.selected() else declared_loop_tiers(inputs.policy_roots),
        **oracle_kwargs(inputs.context),
    )
    _, _withheld = cert_tiers_beyond_loop(
        context=inputs.context,
        policy_roots=inputs.policy_roots,
        public_roots=lambda: inputs.public_root() if callable(inputs.public_root) else inputs.public_root,
    )
    # Resolve once before the suite: every completed functional result can feed
    # the existing queue without waiting for the last capsule in this batch.
    completed_promotion = None
    try:
        from merlin_experiments.phase1.feedback import promotion as _promotion

        loop, cert, cover = _promotion.resolve_tiers(ws, context=inputs.context, capsules_root=inputs.promotion_root)
        if loop and cert:
            completed_promotion = _completed_loop_promoter(ws, cand.parent, fruns, loop, cert, cover, inputs=inputs)
    except Exception as exc:  # noqa: BLE001 -- scheduling cannot invalidate completed loop evidence
        print(f"[first-grade] streaming promotion unavailable: {type(exc).__name__}: {exc}", flush=True)
    from merlin.targetgen.capsule_scheduling import timing_history

    progress = SnapshotProgress(ws, fruns, snapshot_digest, background_guard)
    from . import joint_curriculum
    from .rolling_scalability import Selection as ScalabilitySelection, PerformanceFeedback
    scalability = ScalabilitySelection(inputs.context) if JTS.focused() else None
    scalability_cases = scalability.cases if scalability is not None else {}
    scale_feedback = (PerformanceFeedback(scalability, compiler=cand, runs_root=fruns, timeout=timeout)
                      if scalability_cases else None)
    curriculum = joint_curriculum.native(run_dir, inputs=inputs, compiler_sha256=snapshot_digest,
        target=te.target, runs_root=fruns, runtime_config=readback_kwargs(inputs.context),
        additional_identity=scalability.identity, verify_additional=scalability.verify) if scalability is not None else None

    def process_completed(row):
        if scale_feedback is not None:
            scale_feedback.completed(row)
        # Automatic checks retain the same public products as explicit selfchecks.
        # This passive path launches no timing observer and grants no fit authority.
        try:
            from . import completed_calibration
            if row.get('capsule') not in scalability_cases:
                completed_calibration.register(
                    row, result_path=fruns / "runs" / _suite_for(te.target)
                        / str(row.get("capsule", "")) / "capsule_result.json",
                    compiler=cand, capsules_root=public_root, contract=inputs.contract,
                )
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            print(f"[first-grade] calibration intake unavailable: {type(error).__name__}: {error}", flush=True)
        if curriculum is not None:
            try:
                curriculum.completed(row)
            except (OSError, ValueError, KeyError, TypeError) as error:
                print(f"[first-grade] curriculum receipt unavailable: {type(error).__name__}", flush=True)
        from .joint_physical_correctness import roster as physical_roster
        if (row.get('capsule') not in scalability_cases and completed_promotion is not None
                and (physical_roster() or background_guard.reason() is None)):
            completed_promotion(row)

    def completed(row):
        # Publish the native durable row immediately. Remaining consumers reopen
        # the same evidence on a request-owned worker, without blocking refill.
        progress.completed(row)
        completion_processing.submit(row)

    selected_roots, test_scope = JTS.select(public_root, submission=cand, curriculum=curriculum,
                                          additional_cases=scalability_cases)
    if joint_control.selected() is not None:
        remaining = joint_control.remaining()
        if remaining <= 0:
            selected_roots = []
            test_scope['execution_status'] = 'original_campaign_deadline_exhausted'
        else:
            timeout = min(timeout, max(1, math.ceil(remaining)))
    history = _history_runs_roots(run_dir)
    rolling = (joint_curriculum.admission(curriculum, timeout=timeout,
        remaining_s=joint_control.remaining(drain=True), history=history)
        if curriculum is not None else nullcontext(timeout))
    from .completion_processing import CompletionProcessing

    try:
        with (rolling as case_timeout,
              compiler_execution(inputs, candidate=cand, run_dir=run_dir),
              timing_history(history),
              CompletionProcessing(process_completed) as completion_processing):
            _fast_score = {} if not selected_roots else (
                _CG.grade(
                    str(cand),
                    capsules_root=selected_roots,
                    runs_root=str(fruns),
                    model_snapshot_root=fruns.resolve() / ".private_model_sources",
                    labels={"public", "dev"},
                    contract=str(
                        inputs.contract if inputs.contract is not None else inputs.context.repo / "merlin/contract"
                    ),
                    oracle_adapters=adapters,
                    timeout=case_timeout,
                    max_workers=_CG.default_grade_workers(),
                    target=te.target,
                    additional_forbidden=inputs.additional_forbidden,
                    defer_pending=background_guard.pending_reason,
                    defer_running=False,
                    on_capsule_complete=completed,
                    on_cohort_ready=progress.selected,
                )
                or {}
            )
            if scalability is not None:
                scalability.verify()
    except Exception as e:  # noqa: BLE001 — a partial readback is still feedback, and every tier the
        # grade did not reach stays UNKNOWN in the document below rather than reading as clean.
        print(f"[first-grade] grade error: {type(e).__name__}: {str(e)[:200]}", flush=True)
        _fast_score = {}
    package_not_ready = (_fast_score.get("package_readiness") or {}).get("status") == "not_ready"
    red = {} if package_not_ready else _qc._per_capsule_from_results(fruns)
    for row in _fast_score.get("per_capsule") or []:
        if row.get("background_deferred") and row.get("capsule") not in red:
            red[row["capsule"]] = {"status": "gated", "tiers": {}}
    verdict = _fast_loop_verdict_doc(
        red,
        sorted(adapters),
        sorted(_withheld),
        structural_evidence=_fast_score.get("structural_evidence_all_pass"),
        cohort_complete=_qc._original_cohort_complete(_fast_score),
    )
    if scale_feedback is not None:
        scale_feedback.project(verdict)
    if _fast_score.get("failure"):
        failure = _fast_score["failure"]
        verdict.update(
            all_pass=False, gradeable=False, integrity_status=_fast_score.get("integrity_status"),
            package_failure={"plane": failure.get("plane"), "category": failure.get("category"),
                             "detail": _qc._redact_detail(str(failure.get("detail", "")))},
            note="Compiler package rejected before capsule execution; selected obligations remain pending.",
        )
    if _fast_score.get("package_precheck_refusal"):
        verdict["package_precheck_refusal"] = {
            **_fast_score["package_precheck_refusal"], "source_sha256": snapshot_digest,
        }
    if package_not_ready:
        # Package bootstrap ran; no capsule or oracle did. This is readiness
        # feedback, never a reduced cohort or an empty passing grade.
        verdict.update(
            all_pass=False,
            gradeable=False,
            n_passed=0,
            n_capsules=0,
            per_capsule=[],
            mandatory_tiers_complete=False,
            tiers_graded=[],
            tiers_not_run=sorted(set(adapters) | set(_withheld)),
            package_readiness=_fast_score["package_readiness"],
            package_failure=_fast_score.get("failure"),
            note="The declared compiler tool is missing after package build; no capsules executed. "
            "Finish the declared tool; the first grade will retry shortly. The selected cohort is unchanged.",
        )
    verdict = _attribute_source(verdict, ws, snapshot_digest, stage="loop_screen")
    if background_guard.reason() is not None:
        verdict.update(
            all_pass=None,
            gradeable=False,
            mandatory_tiers_complete=False,
            background_grade={
                "status": "superseded_snapshot" if background_guard.finish_bounded_snapshot else "deferred",
                "reason": background_guard.deferred,
                "source_sha256": background_guard.source,
            },
        )
    verdict["graded_request_sha256"] = background_guard.source
    verdict = _attribute_source(verdict, ws, snapshot_digest, stage="loop_screen")
    if curriculum is not None and test_scope is not None:
        test_scope['curriculum'] = curriculum.report()
        test_scope['pending'] = test_scope['curriculum']['pending']
    JTS.annotate(verdict, test_scope)
    progress.finish(verdict)
    out = run_dir / "qa_history" / f"verdict_fast_{tick}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    verdict = write_verdict(out, verdict)
    (ws / "qa").mkdir(exist_ok=True)
    if verdict["snapshot_current"]:
        verdict = write_verdict(ws / "qa" / "verdict.json", verdict)
    _record_plateau(run_dir, current_source_sha256=verdict.get("current_source_sha256"))
    FL.record_channel_health(ws, run_dir)
    # The first successful loop verdict unlocks cert work immediately. The existing
    # broker queue owns native parallelism, deduplication and certification; never
    # wait for the periodic full-grade cadence or attribute these bytes to live edits.
    try:
        import sys as _sys

        from merlin_experiments.phase1.feedback.promotion import promote as _promote
        from merlin_experiments.phase1.feedback.promotion import resolve_tiers as _resolve

        _loop, _cert, _cover = _resolve(ws, context=inputs.context, capsules_root=inputs.promotion_root)
        if _loop and _cert and verdict.get("per_capsule") and background_guard.deferred is None:
            _promote(
                ws,
                ws / ".qa_channel",
                {**verdict, 'per_capsule': [row for row in verdict['per_capsule']
                                           if row.get('capsule') not in scalability_cases]},
                _loop,
                _cert,
                _cover,
                _sys.stderr,
                source_ws=cand.parent,
                context=inputs.context,
                capsules_root=inputs.promotion_root,
            )
    except Exception as exc:  # noqa: BLE001 -- optional cert scheduling cannot erase loop feedback
        print(f"[first-grade] promotion skipped: {type(exc).__name__}: {exc}", flush=True)
    return verdict


def _shape_skip_reason(verdict: dict) -> str | None:
    """Why this grade's shape probes need not run, or None when its gate needs them.

    Only a grade that is not gradeable and cannot converge skips: nothing it reports may terminate
    authoring, so the probes wait for a gradeable grade. A structure-only grade whose ``all_pass`` is
    still true keeps its probes, because that gate depends on them.
    """
    if verdict.get("gradeable") is False and verdict.get("all_pass") is not True:
        return "grade is not gradeable; required shape probes run on the next gradeable grade"
    return None


def _skip_shape_generalization(verdict: dict, reason: str) -> None:
    verdict["shape_coverage"] = {
        "ran": False,
        "skipped": True,
        "reason": reason,
        "note": "the shape-coverage probes did NOT run for this grade; this is NOT a pass.",
    }
    if verdict.get("all_pass"):
        verdict["all_pass"] = False
        verdict["not_converged_reason"] = "the required shape-coverage probes did not run; not a pass"


# The public suite is the set of shapes an agent can see, so a backend that keys on those shapes passes
# it by construction and the loop has nothing left to say. Measured: a submission converged at 14/26 with
# every failure a shape it had never implemented, its self-check clean on everything it HAD, and no round
# feedback able to point at the gap -- the first instrument to see it was a post-freeze holdout.
#
# These probes are derived from the target's own declared capability closure and its DERIVED tile edge
# (never the corpus), so they leak nothing an agent could not compute itself, and they run at the cheap
# loop tier. Restricted to the multi-tile corners: the question is "does this backend generalize past ONE
# tile, and along WHICH axis" -- per-axis, because a backend that loops over K and N but not M passes two
# of the three, and only naming the axis makes the result actionable.
@with_priority(AUXILIARY)  # Advisory probe CPU yields to author requests and grade capsules.
def _attach_shape_generalization(
    verdict: dict,
    cand,
    run_dir,
    rnd: int,
    *,
    timeout: int,
    artifact_key: str | None = None,
    context: InvocationContext,
    contract: Path | None = None,
    additional_forbidden: tuple[str, ...] = (),
    compiler_profile_service=None,
    before_probe: Callable[[], None] | None = None,
) -> None:
    """Probe whether this round's candidate LOWERS shapes past a single tile, and fold it into the gate.

    Structural, not numerical: it runs only the emit half of the contract, so it costs no oracle,
    needs no golden, and works on an operand format that has no CPU reference. Artifact size is
    reported separately as an advisory observation; a larger shape may use a smaller runtime loop.

    Failure to run is RECORDED, never treated as clean -- a probe that did not run reading as a pass is
    the same class of bug as an unavailable oracle scoring as one.

    ``MERLIN_SHAPE_COVERAGE_WORKERS`` runs corners concurrently and ``MERLIN_SHAPE_COVERAGE_CACHE=1``
    reuses a sweep of byte-identical inputs (see :func:`shape_coverage.measure`). ``before_probe`` runs
    only when probes are actually about to run, so a caller can publish its pending capsule verdict first.
    """
    from . import shape_coverage

    key = artifact_key or f"{rnd:02d}"
    out = run_dir / "qa_history" / f"shape_coverage_round_{key}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    workers, use_cache = shape_coverage.selected_workers(), shape_coverage.cache_selected()
    try:
        cov = shape_coverage.measure(
            cand,
            run_dir,
            context=context,
            contract=str(contract if contract is not None else context.repo / "merlin/contract"),
            timeout=min(timeout, 300),
            additional_forbidden=additional_forbidden,
            service=compiler_profile_service,
            before_probe=before_probe,
            workers=workers,
            use_cache=use_cache,
        )
        out.write_text(json.dumps(cov, indent=2))
    except Exception as e:  # noqa: BLE001 -- record it; never let it read as clean
        verdict["shape_coverage"] = {
            "ran": False,
            "error": f"{type(e).__name__}: {e}",
            "note": "the shape-coverage probes did NOT run this round; this is NOT a pass.",
        }
        if verdict.get("all_pass"):
            verdict["all_pass"] = False
            verdict["not_converged_reason"] = (
                "every public capsule passes, but the required shape-coverage probes did not run; "
                "shape generalization is unmeasured, not a pass"
            )
        return

    verdict["shape_coverage"] = {
        "ran": True,
        "scope": "public_emit_only",
        "tile_edge": cov.get("tile_edge"),
        "baseline_tile_lowered": cov.get("baseline_tile_lowered"),
        "per_corner": {c["corner"]: c["outcome"] for c in cov.get("corners", [])},
        "emitted_work": cov.get("emitted_work"),
        "smaller_emitted_artifacts": cov.get("smaller_emitted_artifacts") or [],
        "multi_tile_axes_uncovered": cov.get("multi_tile_axes_uncovered") or [],
        "tail_cases_uncovered": cov.get("tail_cases_uncovered") or [],
        "tail_axes_uncovered": cov.get("tail_axes_uncovered") or [],
        "all_covered": bool(cov.get("all_covered")),
        "unmeasured": cov.get("unmeasured"),
        "detail": {c["corner"]: c.get("detail") for c in cov.get("corners", []) if c.get("detail")},
        "note": (
            "DERIVED shape probes, not corpus capsules: the SAME contraction at one tile and at two "
            "tiles in each of M, K and N, at this target's derived tile edge. `emitted_work` is the "
            "size of the program you emitted for each. Smaller text is advisory: a runtime loop can "
            "correctly be more compact. An uncovered axis or tail means its emit probe declined, "
            "errored or returned no commands; it is not a numerical verdict. If you cannot lower a "
            "shape, DECLARE it (`declined` on the command buffer) instead of emitting a terminator. "
            "Compilation and native numerical checks remain separate."
        ),
    }
    # THE GATE. Passing every public capsule while lowering only the shapes they happen to use is exactly
    # the state that shipped at 14/26, so the loop must not call that converged.
    if verdict.get("all_pass") and not verdict["shape_coverage"]["all_covered"]:
        verdict["all_pass"] = False
        verdict["not_converged_reason"] = (
            "every public capsule passes, but the derived shape probes did not confirm lowering "
            + (
                f"past one tile on axis/axes {verdict['shape_coverage']['multi_tile_axes_uncovered']}"
                if verdict["shape_coverage"]["multi_tile_axes_uncovered"]
                else (
                    f"for tail case(s) {verdict['shape_coverage']['tail_cases_uncovered']}"
                    if verdict["shape_coverage"]["tail_cases_uncovered"]
                    else "the baseline tile itself (nothing about shape can be concluded yet)"
                )
            )
        )


def _public_object_build_projection(observation: dict, *, budget_s: int, candidate_sha256: str) -> dict:
    """Copy only public measurement scalars into the agent-visible verdict."""
    if observation.get("schema") != "merlin.codegen_scalability_probe.v2":
        raise ValueError("public object-build observation has the wrong schema")
    samples = observation.get("samples")
    if not isinstance(samples, list) or len(samples) != 4:
        raise ValueError("public object-build observation has no complete scale roster")
    projected = []
    for row, scale in zip(samples, (1, 2, 4, 8), strict=True):
        build = row.get("build_only") if isinstance(row, dict) else None
        if not isinstance(build, dict) or row.get("extent_scale") != scale:
            raise ValueError("public object-build sample is malformed")
        outcome = row.get("outcome")
        status = build.get("status")
        if outcome not in ("lowered", "empty", "declined", "error") or status not in (
            "compiled",
            "compile_error",
            "not_emitted",
            "selection_changed",
            "source_exceeds_probe_budget",
        ):
            raise ValueError("public object-build status is not recognized")
        item = {"extent_scale": scale, "emit_outcome": outcome, "build_status": status}
        for source, dest in (("artifact_bytes", "emitted_artifact_bytes"), ("object_bytes", "object_bytes")):
            value = build.get(source)
            if type(value) is int and value >= 0:
                item[dest] = value
        elapsed = build.get("compile_wall_s")
        if type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0:
            item["compile_wall_s"] = elapsed
        projected.append(item)
    return {
        "ran": True,
        "schema": "merlin.public_object_build_feedback.v1",
        "scope": "public_emit_and_build_only_advisory",
        "candidate_sha256": candidate_sha256,
        "per_sample_budget_s": budget_s,
        "samples": projected,
        "note": "Public object compilation only; no execution, numerical result, or certification claim.",
    }


@with_priority(AUXILIARY)  # Advisory probe CPU yields to author requests and grade capsules.
def _attach_codegen_scalability(
    verdict: dict,
    cand,
    run_dir,
    rnd: int,
    *,
    timeout: int,
    context: InvocationContext,
    artifact_key: str | None = None,
    contract: Path | None = None,
    additional_forbidden: tuple[str, ...] = (),
    build_service=None,
    build_selection: dict | None = None,
    build_budget_s: int = 0,
    candidate_sha256: str | None = None,
) -> None:
    """Publish answer-free public size observations without changing scientific grades."""
    from . import codegen_scalability

    key = artifact_key or f"{rnd:02d}"
    out = run_dir / "qa_history" / f"codegen_scalability_round_{key}.json"
    public_out = run_dir / "qa_history" / f"codegen_scalability_public_round_{key}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        if build_service is not None:
            if (
                type(build_budget_s) is not int
                or not 0 < build_budget_s <= 120
                or not isinstance(build_selection, dict)
                or not isinstance(candidate_sha256, str)
                or len(candidate_sha256) != 64
                or hash_tree(Path(cand))["sha256"] != candidate_sha256
            ):
                raise ValueError("frozen public object-build selection or candidate changed")
            if codegen_scalability._selected_build_inputs(build_service, context.target) != build_selection:
                raise ValueError("frozen public object-build source/tool selection changed")
        observation = codegen_scalability.run(
            cand,
            target=context.target,
            contract=contract,
            timeout=min(timeout, 30),
            additional_forbidden=additional_forbidden,
            **(
                {"build_service": build_service, "build_timeout_s": build_budget_s} if build_service is not None else {}
            ),
        )
        if build_service is not None:
            observation["graded_candidate_sha256"] = candidate_sha256
        out.write_text(json.dumps(observation, indent=2))
        if build_service is None:
            verdict["codegen_scalability"] = {"ran": True, **observation}
        else:
            if (
                hash_tree(Path(cand))["sha256"] != candidate_sha256
                or codegen_scalability._selected_build_inputs(build_service, context.target) != build_selection
                or any(
                    build.get("build_selection") != build_selection
                    for row in observation.get("samples", [])
                    if isinstance(row, dict)
                    for build in (row.get("build_only"),)
                    if isinstance(build, dict) and build.get("status") == "compiled"
                )
            ):
                raise ValueError("frozen public object-build selection changed during observation")
            visible = _public_object_build_projection(
                observation, budget_s=build_budget_s, candidate_sha256=candidate_sha256
            )
            public_out.write_text(json.dumps(visible, indent=2))
            verdict["codegen_scalability"] = visible
    except Exception as exc:  # noqa: BLE001 -- missing advisory measurements are not numerical failures
        if build_service is None:
            verdict["codegen_scalability"] = {
                "ran": False,
                "error": f"{type(exc).__name__}: {exc}",
                "scope": "public_emit_only",
                "note": "No scalability measurement was obtained; this is not a pass or a numerical verdict.",
            }
        else:
            if not out.is_file():
                out.write_text(
                    json.dumps(
                        {
                            "scope": "host_private_public_build_diagnostic",
                            "candidate_sha256": candidate_sha256,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        indent=2,
                    )
                )
            visible = {
                "ran": False,
                "schema": "merlin.public_object_build_feedback.v1",
                "scope": "public_emit_and_build_only_advisory",
                "candidate_sha256": candidate_sha256,
                "error": "public object-build measurement unavailable",
                "note": "No object-build result was obtained; the numerical grade is unchanged.",
            }
            public_out.write_text(json.dumps(visible, indent=2))
            verdict["codegen_scalability"] = visible


def write_verdict(path: Path, verdict: dict) -> dict:
    """Write a verdict WITH the time it was produced, and return it.

    Verdicts carried no timestamp of any kind. Under the continuous schedule the grader refreshes
    them while the agent works, so the sequence of verdicts IS the run's progress record -- and with
    no stamp inside, the only clock a reader has is the file's mtime. An mtime does not survive a
    copy, and several of these run trees have been copied between worktrees, so every progress curve
    drawn from them rests on filesystem metadata rather than on anything the run recorded.

    Stamping costs one field. `graded_at` is UTC and ISO-8601, matching the spelling every other
    record in the run tree uses. Existing verdicts are unaffected: readers fall back to the mtime and
    say so.
    """
    from .. import joint_control
    verdict = joint_control.stage_evidence(dict(verdict))
    if path.name == 'verdict.json' and path.parent.name == 'qa':
        from .joint_physical_correctness import attach
        verdict = attach(path, verdict)
    verdict.setdefault("graded_at", _dt.datetime.now(_dt.UTC).isoformat())
    if path.name == "verdict.json" and path.parent.name == "qa":
        from .visible_model_gate import publish

        from .snapshot_publication import publish_live

        return publish_live(path.parent.parent, lambda: publish(path, verdict), default=verdict)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verdict, indent=2))
    return verdict


def _record_plateau(run_dir: Path, *, current_source_sha256: str | None = None) -> None:
    """Assess this run's progress across GRADES and record it OPERATOR-SIDE. Never raises.

    OUT OF BAND, for the same reason `_write_stage_ledger` is: telling the agent it has plateaued, or
    naming the capsules that have never passed, is FEEDBACK -- and feedback defines an arm. Handing it
    over would change the treatment and make the run incomparable with every earlier one. This is for
    whoever decides whether to keep paying for the run.

    It exists because the detector that was here could not fire. `--plateau-rounds` counts consecutive
    ROUNDS with no progress, and the default schedule is `continuous` -- one long session, so exactly
    one round. Its threshold was unreachable in the mode every real run uses, and opt-in besides.
    Measured on merlincirct_g4p1_biasabi_20260906: 92/96 reached at 2.19 h, then 3.91 h more -- 64% of
    the run -- with the score never moving and nothing saying so. A grade is the unit the continuous
    schedule actually produces, so that is the unit here.
    """
    try:
        from merlin.targetgen import plateau as _PL

        history = run_dir / "qa_history"
        # Keep pre-shape archives, but use the completed record for that exact
        # snapshot when available. A snapshot must never count as two grades.
        selected = {path.name.removeprefix("verdict_"): path for path in history.glob("verdict_*.json")}
        selected.update({path.name.removeprefix("completed_"): path for path in history.glob("completed_*.json")})
        hist = sorted(selected.values(), key=lambda q: q.stat().st_mtime)
        grades = []
        for q in hist:
            try:
                grades.append(json.loads(q.read_text()))
            except Exception:  # noqa: BLE001 -- one unreadable grade must not blind the assessment
                continue
        current_grades = [
            g
            for g in grades
            if current_source_sha256 is not None
            and g.get("graded_source_sha256") == current_source_sha256
            and g.get("snapshot_current") is True
        ]
        got = _PL.assess(current_grades)
        historical = _PL.assess(grades)
        doc = {
            "scope": "grades of the observed current source only",
            "current_source_sha256": current_source_sha256,
            "current_source_grade_available": bool(current_grades),
            "stuck": got.stuck,
            "reason": got.reason,
            "stalled_grades": got.stalled_grades,
            "n_grades": got.n_grades,
            "best_passed": got.best_passed if current_grades else None,
            "latest_passed": got.latest_passed if current_grades else None,
            "n_capsules": got.n_capsules,
            "never_passed": list(got.never_passed),
            "regressed": list(got.regressed),
            "sentence": got.sentence() if current_grades else "No grade for the observed current source is available.",
            "snapshot_history": {
                "scope": "immutable snapshots; not current-source qualification",
                "n_grades": historical.n_grades,
                "best_passed": historical.best_passed,
                "latest_passed": historical.latest_passed,
            },
            "note": (
                "operator-side only: this is never handed to the agent, because new feedback "
                "would change the arm's treatment"
            ),
        }
        (run_dir / "plateau.json").write_text(json.dumps(doc, indent=2))
        if got.stuck or got.regressed:
            print(f"[plateau] {got.sentence()}", flush=True)
    except Exception:  # noqa: BLE001 -- diagnostics may never fail a grade
        return


def _write_stage_ledger(
    run_dir,
    rnd: int,
    cand,
    runs_root,
    verdict,
    *,
    artifact_key: str | None = None,
    previous_artifact_key: str | None = None,
) -> None:
    """Record per-round artifact fingerprints beside the verdict — OUT OF BAND.

    Answers "did the agent's edit reach what was graded?" from bytes, so a plateau is one line instead of
    an investigation. Written into the run dir (a sandbox-DENIED path) and never into the agent's
    ``qa/verdict.json``: telling the agent its edit was inert is feedback, and feedback defines an arm.

    Never raises. This is diagnostics — it must not be able to fail a round that otherwise graded fine.
    """
    try:
        from merlin.targetgen import stage_ledger as SL

        led_dir = run_dir / "rounds"
        led_dir.mkdir(parents=True, exist_ok=True)
        key = artifact_key or f"{rnd:02d}"
        if artifact_key is None:
            previous_key = f"{rnd - 1:02d}" if rnd else None
        else:
            previous_key = previous_artifact_key
        prev_p = led_dir / f"round_{previous_key}.stage_ledger.json" if previous_key is not None else None
        prev = json.loads(prev_p.read_text()) if prev_p is not None and prev_p.is_file() else None

        # Per-capsule emit dirs, found by SHAPE at any depth: a dir named for the emit output whose
        # PARENT is a graded capsule dir (it holds the capsule's own result/manifest). Depth and the
        # intervening dir names are never assumed -- no target, no suite, and no generated-root literal --
        # so a target that lays its run tree out differently is still discovered, and one that lays it out
        # unrecognizably simply yields none rather than a wrong answer.
        marks = ("capsule_result.json", "run_manifest.yaml")
        roots = {
            d.parent.name: d
            for d in Path(runs_root).rglob("generated")
            if d.is_dir() and any((d.parent / m).is_file() for m in marks)
        }
        led = SL.build(submission_dir=cand, emitted_roots=roots, previous=prev)
        led["round"] = rnd
        led["failing_and_frozen"] = SL.failing_and_frozen(led, verdict)
        (led_dir / f"round_{key}.stage_ledger.json").write_text(json.dumps(led, indent=2))
        print(f"  {SL.summarize(led)}", flush=True)
        if led["failing_and_frozen"]:
            print(
                f"  stage_ledger: FAILING AND FROZEN ({len(led['failing_and_frozen'])}): "
                f"{', '.join(led['failing_and_frozen'][:10])}"
                f"{' ...' if len(led['failing_and_frozen']) > 10 else ''}",
                flush=True,
            )
    except Exception as e:  # noqa: BLE001 - diagnostics never gate
        print(f"  stage_ledger: unavailable ({type(e).__name__}: {e})", flush=True)
