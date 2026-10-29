"""At most two owned grades: one admission window may run beside an older drain."""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time

from merlin.targetgen.capsule_scheduling import observe_admission
from merlin.common.cpu_pool import GRADE, cpu_priority, preparation_cpus, selected_pool
from .. import joint_control
from ..compiler_library import join_background, library_thread
from . import snapshot_publication
from .background_admission import SubmissionNotReady


def _completed_development_source(verdict, current_source):
    """Cheap coverage deduplication only; withheld physical/final gates remain mandatory."""
    curriculum = (verdict.get('joint_test_selection') or {}).get('curriculum') or {}
    required = curriculum.get('original_development_required')
    reused = curriculum.get('exact_correctness_reused')
    actual = verdict.get('graded_request_sha256')
    if (verdict.get('snapshot_current') is True and not verdict.get('package_failure')
            and curriculum.get('schema') == 'merlin.joint_correctness_curriculum.v1'
            and curriculum.get('pending') == [] and curriculum.get('original_unresolved') == 0
            and type(required) is int and required > 0
            and isinstance(reused, list) and len(reused) >= required
            and isinstance(actual, str) and len(actual) == 64
            and all(c in '0123456789abcdef' for c in actual) and actual == current_source):
        return actual
    return None


def _partial_counts(run_dir, tick):
    """Counts a snapshot had reached (completed redacted rows only), from its own progress stream."""
    path = Path(run_dir) / "_qa_work" / f"fruns_{tick}" / "snapshot_progress.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    rows = [r for r in data.get("per_capsule") or [] if isinstance(r, dict)]
    return {
        "n_selected": data.get("n_selected"),
        "n_completed": data.get("n_completed"),
        "n_passed": data.get("n_passed"),
        "n_failed": sum(r.get("status") == "fail" for r in rows),
        "state": data.get("state"),
    }


def run(
    workspace,
    run_dir,
    stop,
    *,
    interval_s,
    first_grade,
    grade,
    source_identity,
    on_finished=None,
    poll_s=1.0,
    first_required=True,
):
    """Own/join every worker. No scratch recycling or cancellation on ordinary edits.

    Same-source overlap is allowed only after the older admission queue closes;
    the curriculum's exact in-flight ledger excludes its admitted capsules.

    Joint campaigns add preemption (``joint_control.snapshot_preemption``): once the author's source
    has moved past a snapshot (and the next snapshot is due), the stale snapshot stops admitting,
    its queued work is cancelled and, in ``abort`` mode, its in-flight capsules are revoked at their
    next cooperative checkpoint after a short grace, so the newest snapshot gets the grader at once.
    At most two non-abandoned snapshots exist; a revoked owner that fails to unwind within its grace
    is abandoned (its daemon thread keeps unwinding, owning nothing the scheduler waits on). At the
    freeze boundary every in-flight snapshot is revoked and joined only for ``freeze_grace_s``.
    """
    directory = Path(run_dir) / '_qa_work/background_snapshot_owners'
    directory.mkdir(parents=True, exist_ok=True)
    active = []
    abandoned = []
    completed_source = None
    next_start = 0.0
    first_pending = first_required
    readiness_state = object()

    def readiness(reason):
        nonlocal readiness_state
        if reason == readiness_state:
            return
        readiness_state = reason
        document = dict(schema="merlin.background_snapshot_readiness.v1",
                        status="waiting_for_manifest" if reason else "manifest_present",
                        reason=reason, checked_at_epoch=time.time(),
                        grade_attempted=False, all_pass=None,
                        scope="admission observation only; no package or numerical qualification")
        path = Path(run_dir) / "background_snapshot_readiness.json"
        temporary = path.with_suffix(".next.json")
        temporary.write_text(json.dumps(document, indent=2) + "\n")
        temporary.replace(path)

    def persist(job):
        document = {key: job[key] for key in ("token", "source", "started_at_epoch", "tick")}
        document.update(
            schema="merlin.background_snapshot_owner.v1",
            admission_closed=job["closed"].is_set(),
            snapshot_dir=str(Path(run_dir) / "_qa_work" / f"fcand_{job['tick']}" / "submission"),
            runs_dir=str(Path(run_dir) / "_qa_work" / f"fruns_{job['tick']}"),
            admission_reason=job.get("admission_reason"),
            state=(
                "abandoned"
                if job.get("abandoned")
                else "deferred"
                if job.get("deferred")
                else "complete"
                if job["done"].is_set()
                else "draining"
                if job["closed"].is_set()
                else "admitting"
            ),
            preemption=(
                {
                    "by_source": job.get("preempted_by"),
                    "at_epoch": job.get("preempted_at"),
                    "abort_reason": job.get("abort_reason"),
                    "abort_at_epoch": job.get("aborted_at"),
                    "partial_counts": _partial_counts(run_dir, job["tick"]),
                }
                if job["preempt"].is_set() or job["abort"].is_set()
                else None
            ),
            deferred_reason=job.get("deferred"),
            error=job.get("error"),
            source_scope="immutable grade owns its actual source/result pins",
        )
        path = directory / (job["token"] + ".json")
        temporary = path.with_suffix(".next.json")
        temporary.write_text(json.dumps(document, indent=2) + "\n")
        temporary.replace(path)

    def launch(source):
        nonlocal next_start, first_pending
        tick = time.time_ns()
        job = dict(
            token=str(tick),
            tick=tick,
            source=source,
            started_at_epoch=time.time(),
            closed=threading.Event(),
            done=threading.Event(),
            first=first_pending,
            preempt=threading.Event(),
            abort=threading.Event(),
        )
        first_pending = False
        snapshot_publication.claim(workspace, job['token'], source)

        def closed(reason):
            job['admission_reason'] = reason
            job['closed'].set()

        def preempted_reason():
            return (
                f"snapshot preempted: author source moved past {job['source'][:12]} and a newer snapshot is due"
                if job["preempt"].is_set()
                else None
            )

        def abort_reason():
            return job.get("abort_reason") if job["abort"].is_set() else None

        def work():
            try:
                with (
                    selected_pool(preparation_cpus()),
                    cpu_priority(GRADE),
                    snapshot_publication.scope(workspace, job["token"], preempted=preempted_reason),
                    observe_admission(
                        closed,
                        stop_requested=lambda: stop.is_set() or job["preempt"].is_set(),
                        abort_requested=abort_reason,
                    ),
                ):
                    callback = first_grade if job["first"] else grade
                    job["verdict"] = callback(tick)
            except SubmissionNotReady as error:
                job['deferred'] = str(error)[:500] or 'submission manifest not ready'
            except Exception as error:  # owned grade failure cannot strand the other owner
                detail = ''.join(char if char.isprintable() else ' ' for char in str(error)[:500])
                job['error'] = f'{type(error).__name__}: {detail}'[:500]
                print(f"[snapshot {tick}] grade failed: {job['error']}", flush=True)
            finally:
                job['closed'].set()
                job['done'].set()

        job['thread'] = library_thread(work, name=f'grade-snapshot-{tick}', daemon=True)
        active.append(job)
        persist(job)
        job['thread'].start()
        next_start = time.monotonic() + interval_s

    def preempt(job, newer):
        job["preempted_by"], job["preempted_at"] = newer, time.time()
        job["preempt"].set()
        job["admission_reason"] = job.get("admission_reason") or "preempted by a newer snapshot"
        job["closed"].set()  # it can admit nothing more, so the newest snapshot may start at once
        persist(job)

    def revoke(job, reason):
        if not job["abort"].is_set():
            job["abort_reason"], job["aborted_at"] = reason, time.time()
            job["abort"].set()
            persist(job)

    def preemption_step(policy):
        """Stale-snapshot preemption; returns nothing. Only called inside a joint campaign."""
        now = time.time()
        for job in tuple(active):
            if job["preempt"].is_set() and policy["mode"] == "abort":
                if not job["abort"].is_set() and now - job["preempted_at"] >= policy["abort_grace_s"]:
                    revoke(job, "snapshot preempted by a newer snapshot")
                elif (
                    job["abort"].is_set()
                    and not job["done"].is_set()
                    and now - job["aborted_at"] >= max(15, policy["abort_grace_s"])
                ):
                    # Unwinding too slowly: stop counting it against the two-snapshot limit.
                    job["abandoned"] = True
                    active.remove(job)
                    abandoned.append(job)
                    persist(job)
        due = (
            time.monotonic() >= next_start
            and (Path(workspace) / "submission/manifest.yaml").is_file()
            and any(not job["done"].is_set() for job in active)
        )
        if not due:
            return
        source = source_identity()
        if source is None or source == completed_source:
            return
        for job in tuple(active):
            if (
                not job["done"].is_set()
                and not job["preempt"].is_set()
                and source != job["source"]
                and now - job["started_at_epoch"] >= policy["min_age_s"]
            ):
                preempt(job, source)
        if policy["mode"] == "abort" and len(active) >= 2:
            # Newest cannot start while two owners exist: revoke the oldest stale one now.
            stale = [job for job in active if job["preempt"].is_set() and not job["done"].is_set()]
            if stale:
                revoke(
                    min(stale, key=lambda job: job["started_at_epoch"]),
                    "snapshot preempted; slot needed for the newest snapshot",
                )

    try:
        while not stop.is_set():
            policy = joint_control.snapshot_preemption()
            if policy is not None and policy["mode"] != "off":
                preemption_step(policy)
            for job in tuple(active):
                persist(job)
                if not job['done'].is_set():
                    continue
                join_background(job['thread'])
                active.remove(job)
                if job.get('deferred') or (job.get('error') and not job.get('verdict')):
                    # It never graded, so it must not keep supersede-marking the older grade.
                    others = [j['token'] for j in active]
                    snapshot_publication.release(workspace, job['token'], max(others, key=int) if others else None)
                if job.get('deferred'):
                    first_pending = first_pending or job['first']
                    next_start = 0.0
                    readiness(job['deferred'])
                    continue  # neither a completed grade nor a grade-health failure
                verdict = job.get('verdict') or {}
                covered = _completed_development_source(verdict, source_identity())
                if covered is not None:
                    completed_source = covered
                if on_finished is not None:
                    on_finished(job['tick'], verdict, job.get('error'))
            # A second owner is eligible only after the first cannot admit more.
            can_launch = len(active) < 2 and all(job['closed'].is_set() for job in active)
            if can_launch and time.monotonic() >= next_start:
                if not (Path(workspace) / 'submission/manifest.yaml').is_file():
                    readiness('submission/manifest.yaml not present; author construction in progress')
                    stop.wait(poll_s)
                    continue
                readiness(None)
                source = source_identity()
                if source is not None and source != completed_source and not stop.is_set():
                    launch(source)
            stop.wait(poll_s)
    finally:
        # Preserve every started result and source scope across native turn handoff, except at the
        # joint freeze boundary: the frozen submission's final coverage needs the grader and its
        # budget, so superseded in-flight snapshots are revoked and joined only for a short grace.
        frozen = joint_control.selected() is not None and joint_control.freeze_reached()
        if frozen:
            for job in active:
                job["preempt"].set()
                job.setdefault("preempted_at", time.time())
                revoke(job, "joint campaign freeze boundary; final coverage owns the grader")
            until = time.monotonic() + joint_control.freeze_grace_s()
        for job in active:
            if frozen:
                job["thread"].join(max(0.0, until - time.monotonic()))
                if job["thread"].is_alive():
                    job["abandoned"] = True
                    abandoned.append(job)
                    persist(job)
                    continue
            join_background(job["thread"])
            persist(job)
