"""Owned background grades: one admission window may run beside older drains.

Normally at most two owners (one admitting, one draining). When many shared CPU slots are free a
third may start, so a newer snapshot is not left waiting behind two slow drains while CPUs idle
(``MERLIN_BACKGROUND_MAX_OWNERS``, default 3; ``MERLIN_BACKGROUND_EXTRA_OWNER_FREE_SLOTS``, default a
quarter of the pool, at least 2). Only one owner ever admits at a time.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time

from merlin.targetgen.capsule_scheduling import observe_admission
from merlin.common.cpu_pool import GRADE, PREPARATION, MutablePriority, cpu_priority, preparation_cpus, selected_pool
from ..compiler_library import library_thread, join_background
from . import idle_performance, scratch_retention, snapshot_publication, source_quiet
from .background_admission import SubmissionNotReady


#: How long a grade refused for low disk space waits before the volume is checked again.
LOW_SPACE_RETRY_S = 60.0
#: Seconds between two checks of the shared pool for an extra owner.
EXTRA_OWNER_CHECK_S = 15.0


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw.isascii() and raw.isdecimal() and int(raw) > 0 else default


def max_owners(free_slots, pool_size) -> int:
    """Two owners, or up to ``MERLIN_BACKGROUND_MAX_OWNERS`` while enough shared slots are free."""
    ceiling = _positive_int("MERLIN_BACKGROUND_MAX_OWNERS", 3)
    if ceiling <= 2 or free_slots is None:
        return min(2, ceiling)
    needed = _positive_int("MERLIN_BACKGROUND_EXTRA_OWNER_FREE_SLOTS", max(2, (pool_size or 0) // 4))
    return ceiling if free_slots >= needed else 2


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


def run(workspace, run_dir, stop, *, interval_s, first_grade, grade, source_identity,
        on_finished=None, poll_s=1.0, first_required=True):
    """Own/join every worker. No scratch recycling or cancellation on ordinary edits.

    Same-source overlap is allowed only after the older admission queue closes;
    the curriculum's exact in-flight ledger excludes its admitted capsules.
    When a grade of a different source starts, an older grade still draining
    keeps every result it started but leases CPUs at :data:`PREPARATION`, so
    the newer compiler's capsules take free slots first.
    """
    directory = Path(run_dir) / '_qa_work/background_snapshot_owners'
    directory.mkdir(parents=True, exist_ok=True)
    active = []
    measurer = idle_performance.IdleMeasurer(workspace, run_dir)
    completed_source = None
    next_start = 0.0
    capacity = {"checked": -EXTRA_OWNER_CHECK_S, "owners": 2}
    first_pending = first_required
    readiness_state = object()
    # A due grade waits for a quiet source (bounded); see source_quiet.
    quiet = source_quiet.for_workspace(workspace, source_identity, poll_s=poll_s)
    announced = False  # one log line per wait, not per poll

    def readiness(reason, status=None, **detail):
        nonlocal readiness_state
        key = (reason, status, tuple(sorted(detail.items())))
        if key == readiness_state:
            return
        readiness_state = key
        document = dict(schema="merlin.background_snapshot_readiness.v1",
                        status=status or ("waiting_for_construction_plan" if reason and reason.startswith("construction plan")
                                          else "waiting_for_manifest" if reason else "manifest_present"),
                        reason=reason, checked_at_epoch=time.time(),
                        grade_attempted=False, all_pass=None,
                        scope="admission observation only; no package or numerical qualification", **detail)
        path = Path(run_dir) / "background_snapshot_readiness.json"
        temporary = path.with_suffix(".next.json")
        try:
            temporary.write_text(json.dumps(document, indent=2) + "\n")
            temporary.replace(path)
        except OSError as error:  # a full volume loses this observation, not the grader
            scratch_retention.say(f"[snapshot] readiness record unavailable: {type(error).__name__}: {error}")

    def persist(job):
        document = {key: job[key] for key in ('token', 'source', 'started_at_epoch', 'tick')}
        document.update(schema='merlin.background_snapshot_owner.v1',
                        admission_closed=job['closed'].is_set(),
                        snapshot_dir=str(Path(run_dir) / '_qa_work' / f"fcand_{job['tick']}" / 'submission'),
                        runs_dir=str(Path(run_dir) / '_qa_work' / f"fruns_{job['tick']}"),
                        admission_reason=job.get('admission_reason'),
                        state='deferred' if job.get('deferred') else 'complete' if job['done'].is_set() else 'draining' if job['closed'].is_set() else 'admitting',
                        deferred_reason=job.get('deferred'),
                        lease_priority=job['priority'].level,
                        error=job.get('error'), source_scope='immutable grade owns its actual source/result pins')
        path = directory / (job['token'] + '.json')
        temporary = path.with_suffix('.next.json')
        try:
            temporary.write_text(json.dumps(document, indent=2) + '\n')
            temporary.replace(path)
        except OSError as error:  # a full volume loses this observation, not the grader
            scratch_retention.say(f"[snapshot {job['tick']}] owner record unavailable: {type(error).__name__}: {error}")

    def prune():
        # Superseded grades keep their results and receipts; their build trees go (scratch_retention).
        try:
            scratch_retention.prune_superseded_grades(run_dir, workspace)
        except Exception as error:  # noqa: BLE001 -- retention never stops the grader
            scratch_retention.say(f"[retention] pruning failed: {type(error).__name__}: {error}")

    def launch(source):
        nonlocal next_start, first_pending
        tick = time.time_ns()
        job = dict(token=str(tick), tick=tick, source=source, started_at_epoch=time.time(),
                   closed=threading.Event(), done=threading.Event(), first=first_pending,
                   priority=MutablePriority(GRADE))
        for older in active:
            if older['source'] != source and not older['done'].is_set():
                older['priority'].lower(PREPARATION)
        snapshot_publication.claim(workspace, job['token'])
        first_pending = False

        def closed(reason):
            job['admission_reason'] = reason
            job['closed'].set()

        def work():
            try:
                with selected_pool(preparation_cpus()), cpu_priority(job['priority']), \
                        snapshot_publication.scope(workspace, job['token']), \
                        observe_admission(closed, stop_requested=stop.is_set, current=lambda: source_identity() == source):
                    callback = first_grade if job['first'] else grade
                    job['verdict'] = callback(tick)
            except SubmissionNotReady as error:
                job['deferred'] = str(error)[:500] or 'submission manifest not ready'
            except Exception as error:  # owned grade failure cannot strand the other owner
                detail = ''.join(char if char.isprintable() else ' ' for char in str(error)[:500])
                job['error'] = f'{type(error).__name__}: {detail}'[:500]
                scratch_retention.say(f"[snapshot {tick}] grade failed: {job['error']}")
            finally:
                job['closed'].set()
                job['done'].set()

        job['thread'] = library_thread(work, name=f'grade-snapshot-{tick}', daemon=True)
        active.append(job)
        persist(job)
        job['thread'].start()
        next_start = time.monotonic() + interval_s

    try:
        while not stop.is_set():
            for job in tuple(active):
                persist(job)
                if not job['done'].is_set():
                    continue
                join_background(job['thread'])
                active.remove(job)
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
                prune()
            # A further owner is eligible only after every active one stopped admitting.
            if len(active) >= 2 and time.monotonic() - capacity["checked"] >= EXTRA_OWNER_CHECK_S:
                capacity["checked"] = time.monotonic()
                try:
                    capacity["owners"] = max_owners(idle_performance.free_slots(), idle_performance.pool_size())
                except Exception:  # noqa: BLE001 -- an unreadable pool keeps the two-owner default
                    capacity["owners"] = 2
            can_launch = (len(active) < (2 if len(active) < 2 else capacity["owners"])
                          and all(job['closed'].is_set() for job in active))
            if can_launch and time.monotonic() >= next_start:
                if not (Path(workspace) / 'submission/manifest.yaml').is_file():
                    readiness('submission/manifest.yaml not present; author construction in progress')
                    stop.wait(poll_s)
                    continue
                from .construction_plan import readiness as construction_readiness
                plan_reason = construction_readiness(Path(workspace) / "submission")
                if plan_reason is not None:
                    readiness(plan_reason)
                    stop.wait(poll_s)
                    continue
                settled, source = quiet.ready()
                if quiet.waiting_s is None:
                    readiness(None)
                    announced = False
                elif source is not None:  # a sample that did not settle: at most once per check interval
                    if not announced:
                        announced = True
                        scratch_retention.say(
                            f"[snapshot] grade due; waiting for the source to stay unchanged {quiet.quiet:g} s "
                            f"(at most {quiet.max_delay:g} s)")
                    readiness("source changed within the quiet period", status="waiting_for_quiet_source",
                              waited_s=round(quiet.waiting_s, 1), quiet_s=quiet.quiet,
                              max_delay_s=quiet.max_delay)
                if settled and source is not None and source != completed_source and not stop.is_set():
                    if not scratch_retention.admit_grade(run_dir, workspace):
                        next_start = time.monotonic() + LOW_SPACE_RETRY_S
                        stop.wait(poll_s)
                        continue
                    try:
                        launch(source)
                        quiet.launched()
                    except OSError as error:  # e.g. a full volume: retry later, never end the grader
                        scratch_retention.say(f"[snapshot] grade not started: {type(error).__name__}: {error}")
                        next_start = time.monotonic() + LOW_SPACE_RETRY_S
            # Free reserved CPUs refresh the performance cohort's timing estimates (idle_performance).
            # While a grade admits, only slots beyond its reserve are used; the AUXILIARY lease class
            # keeps every grade's capsules ahead of this work.
            measurer.tick(grading=any(not job['closed'].is_set() for job in active))
            stop.wait(poll_s)
    finally:
        # Preserve every started result and source scope across native turn handoff.
        for job in active:
            join_background(job['thread'])
            persist(job)
