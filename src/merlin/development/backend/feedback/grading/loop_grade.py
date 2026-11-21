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

from merlin.common.digest import hash_tree
from merlin.common.process.cpu_pool import shared_compute_pool
from merlin.development.backend.campaign.calibration.activation import admit_grade, grade_model_scope
from merlin.development.backend.context import InvocationContext, readback_kwargs
from merlin.development.backend.feedback.campaign import correctness as JTS
from merlin.development.backend.feedback.execution.snapshot_evaluator import measure
from merlin.development.backend.feedback.grading import completed_grade, package_rejection
from merlin.development.backend.feedback.grading import lifecycle as FL
from merlin.development.backend.feedback.grading.loop_probes import (
    _attach_codegen_scalability,
    _attach_shape_generalization,
    _shape_skip_reason,
    _skip_shape_generalization,
)
from merlin.development.backend.session import treatments as T
from merlin.development.backend.session import workspaces as OH
from merlin.development.backend.session.inputs import run_record as RI
from merlin.evaluation.capsule.evidence import certificates as capsule_availability
from merlin.target.registry.descriptor_loading import load_target_experiment

# ---- loop_grading ------------------------------------------------------------------------------


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
        source_hash_algorithm="merlin.common.digest.hash_tree",
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
    from merlin.development.backend.feedback.execution.profile_runtime import original_compiler_execution

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
    from merlin.development.targetgen.corpus.materialization import declared_oracle_tiers

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

    try:
        te = load_target_experiment(context.descriptor)
        ck = capsule_availability.qa_checkpoint_adapters(te.target, te.sim_via, **readback_kwargs(context))
        loop = capsule_availability.qa_loop_adapters(
            te.target, te.sim_via, declared_tiers=declared_loop_tiers(policy_roots), **readback_kwargs(context)
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
            mandatory |= set(doc.get("required_oracle_tiers") or [])
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


@grade_model_scope
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
    from merlin.development.backend.feedback.grading.fast_grade import (
        _record_plateau,
        _write_stage_ledger,
        fast_grade,
        write_verdict,
    )

    if advisory_background and JTS.focused() and not no_oracle:
        import time

        # Interval grades use the same bounded native Spike path as first-grade.
        # A unique private snapshot avoids clashing with the first-grade tick.
        from merlin.development.backend.feedback.execution.snapshots import current_token

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
        from merlin.development.backend.feedback.verdicts import summarize
        from merlin.development.backend.feedback.disclosure import project

        progress_rows[row["capsule"]] = row
        qualification = summarize(list(progress_rows.values()))
        current = background_guard.current()
        progress_path.parent.mkdir(exist_ok=True)
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                project(
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
                        # Seeded grades: passes carried in full from the seed run, labelled per row.
                        **(
                            {"n_carried_passed": carried}
                            if (carried := sum(bool(r.get("carried_from")) for r in progress_rows.values()))
                            else {}
                        ),
                        "qualification_summary": qualification,
                        "per_capsule": list(progress_rows.values()),
                        "updated_at": _dt.datetime.now(_dt.UTC).isoformat(),
                        "scope": "completed redacted rows; final shape, model and certification gates remain pending",
                    },
                    workspace=ws,
                ),
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
        from merlin.development.backend.feedback.grading.background_admission import (
            BackgroundGuard,
            source_digest,
        )

        public = inputs.public_root() if callable(inputs.public_root) else inputs.public_root
        background_guard = BackgroundGuard(ws, public, inputs, source_digest(ws), finish_bounded_snapshot=JTS.focused())
        raw = os.environ.get("MERLIN_BACKGROUND_GRADE_WORKERS")
        if raw is None:
            # No optional worker selection means no background worker pool.
            raw = "0"
        if not raw.isascii() or not raw.isdecimal() or not 0 <= int(raw) <= 64:
            raise ValueError("MERLIN_BACKGROUND_GRADE_WORKERS must be an integer between 0 and 64")
        background_workers = int(raw)
    oot_commit = None
    snapshot_digest = None
    cand = run_dir / "_qa_work" / f"cand_{_key}" / "submission"
    if cand.exists():
        shutil.rmtree(cand.parent)
    _lok, _lwhy = _language_ok(ws / "submission", inputs.arm)
    if not (ws / "submission" / "manifest.yaml").exists():
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
            shutil.copytree(ws / "submission", cand, ignore=shutil.ignore_patterns(*RI.SNAPSHOT_IGNORED))
            RI.strip_build_state(cand)  # clean, relocatable build per grade (abc9 L3-build bug)
            snapshot_digest = hash_tree(cand)["sha256"]
        if background_guard is not None:
            background_guard.bind_snapshot(cand)
        admit_grade(
            candidate=cand,
            source_sha256=snapshot_digest,
            output=run_dir,
            kind="full",
            identity={"label": label, "scratch_key": _key},
        )
        from merlin.development.backend.feedback.guidance.construction_plan import (
            require_ready as require_construction_plan,
        )

        require_construction_plan(cand)
        # The harness commits exactly the bytes about to be graded; the agent never touches the repo.
        oot_commit = OH.commit_graded(run_dir, cand, label=label, key=_key, sandbox_roots=(ws.parent,))
        with shared_compute_pool(timeout_s=timeout):
            candidate_sha256 = snapshot_digest if inputs.public_build_service is not None else None
        public_root = inputs.public_root() if callable(inputs.public_root) else inputs.public_root
        public_resources = None
        if not JTS.focused() and (background_guard is None or background_guard.current()):
            from merlin.development.backend.feedback.scalability.resource_gate import observe_authoring

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
                from merlin.development.backend.feedback.disclosure import project

                pending.write_text(json.dumps(project(public_resources, workspace=ws), indent=2) + "\n")
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
        from merlin.development.backend.feedback.grading import grade_lanes, qa
        from merlin.development.targetgen.grading.scheduling import admission_relevance, timing_history
        from merlin.development.targetgen.grading.seeded_grade import workspace_gap

        relevance = (
            admission_relevance(
                source_sha256=snapshot_digest,
                reach=background_guard.case_reach,
                superseded=grade_lanes.relevance(),
                gap=workspace_gap(ws),
            )
            if background_guard is not None
            else nullcontext()
        )
        with (
            compiler_execution(inputs, candidate=cand, run_dir=run_dir),
            timing_history(_history_runs_roots(run_dir)),
            relevance,
        ):
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
            from merlin.development.backend.feedback.scalability.public_scalability import (
                attach as attach_scalability,
            )

            if JTS.focused():
                verdict["public_scalability"] = {
                    "status": "pending",
                    "reason": "use focused scalability selfchecks; full ladder at milestone",
                }
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
            if JTS.focused() or (
                background_guard is not None
                and (not background_guard.current() or os.environ.get("MERLIN_BACKGROUND_CODEGEN_SCALABILITY") == "0")
            ):
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
                from merlin.development.backend.feedback.grading.sampling import retain_driver_grade

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

        from merlin.development.backend.feedback.certification.promotion import promote as _promote
        from merlin.development.backend.feedback.certification.promotion import resolve_tiers as _resolve

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


def _verdict_row(name: str, info: dict | None) -> dict:
    """One redacted per-test-case verdict row from its readback (``qa._per_capsule_from_results``)."""
    info = info or {}
    return {
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
        **({"carried": info["carried"]} if info.get("carried") else {}),
        **({"carried_from": info["carried_from"]} if info.get("carried_from") else {}),
    }
