"""At most two owned grades: one admission window may run beside an older drain."""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time

from merlin.targetgen.capsule_scheduling import observe_admission
from merlin.common.cpu_pool import GRADE, cpu_priority, preparation_cpus, selected_pool
from ..compiler_library import library_thread, join_background
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


def run(workspace, run_dir, stop, *, interval_s, first_grade, grade, source_identity,
        on_finished=None, poll_s=1.0, first_required=True):
    """Own/join every worker. No scratch recycling or cancellation on ordinary edits.

    Same-source overlap is allowed only after the older admission queue closes;
    the curriculum's exact in-flight ledger excludes its admitted capsules.
    """
    directory = Path(run_dir) / '_qa_work/background_snapshot_owners'
    directory.mkdir(parents=True, exist_ok=True)
    active = []
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
                        status=("waiting_for_construction_plan" if reason and reason.startswith("construction plan")
                                else "waiting_for_manifest" if reason else "manifest_present"),
                        reason=reason, checked_at_epoch=time.time(),
                        grade_attempted=False, all_pass=None,
                        scope="admission observation only; no package or numerical qualification")
        path = Path(run_dir) / "background_snapshot_readiness.json"
        temporary = path.with_suffix(".next.json")
        temporary.write_text(json.dumps(document, indent=2) + "\n")
        temporary.replace(path)

    def persist(job):
        document = {key: job[key] for key in ('token', 'source', 'started_at_epoch', 'tick')}
        document.update(schema='merlin.background_snapshot_owner.v1',
                        admission_closed=job['closed'].is_set(),
                        snapshot_dir=str(Path(run_dir) / '_qa_work' / f"fcand_{job['tick']}" / 'submission'),
                        runs_dir=str(Path(run_dir) / '_qa_work' / f"fruns_{job['tick']}"),
                        admission_reason=job.get('admission_reason'),
                        state='deferred' if job.get('deferred') else 'complete' if job['done'].is_set() else 'draining' if job['closed'].is_set() else 'admitting',
                        deferred_reason=job.get('deferred'),
                        error=job.get('error'), source_scope='immutable grade owns its actual source/result pins')
        path = directory / (job['token'] + '.json')
        temporary = path.with_suffix('.next.json')
        temporary.write_text(json.dumps(document, indent=2) + '\n')
        temporary.replace(path)

    def launch(source):
        nonlocal next_start, first_pending
        tick = time.time_ns()
        job = dict(token=str(tick), tick=tick, source=source, started_at_epoch=time.time(),
                   closed=threading.Event(), done=threading.Event(), first=first_pending)
        first_pending = False
        snapshot_publication.claim(workspace, job['token'])

        def closed(reason):
            job['admission_reason'] = reason
            job['closed'].set()

        def work():
            try:
                with selected_pool(preparation_cpus()), cpu_priority(GRADE), \
                        snapshot_publication.scope(workspace, job['token']), observe_admission(closed, stop_requested=stop.is_set):
                    callback = first_grade if job['first'] else grade
                    job['verdict'] = callback(tick)
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
            # A second owner is eligible only after the first cannot admit more.
            can_launch = len(active) < 2 and all(job['closed'].is_set() for job in active)
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
                readiness(None)
                source = source_identity()
                if source is not None and source != completed_source and not stop.is_set():
                    launch(source)
            stop.wait(poll_s)
    finally:
        # Preserve every started result and source scope across native turn handoff.
        for job in active:
            join_background(job['thread'])
            persist(job)
