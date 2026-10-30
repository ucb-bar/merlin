"""Host-owned scheduling admission for advisory background grades.

An active marker identifies an authorized foreground full self-check, never
numerical authority. It is issued by the real broker in private storage and
expires with its actual process owners. Background grades no longer pause for
it: CPU ordering between author and grade work is the shared pool's lease
priority (``merlin.common.cpu_pool``); only a superseded source defers work.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
from pathlib import Path

SCHEMA = "merlin.background_foreground_admission.v1"


class SubmissionNotReady(RuntimeError):
    """No grade was attempted because the initial package manifest is absent."""


def _identity(pid):
    root = Path("/proc") / str(pid)
    stat = (root / "stat").read_text().rpartition(") ")[2].split()
    if stat[0] == "Z":
        raise ValueError("process is terminal")
    return {
        "pid": int(pid),
        "birth": stat[19],
        "parent": int(stat[1]),
        "pid_namespace": os.readlink(root / "ns/pid"),
        "mount_namespace": os.readlink(root / "ns/mnt"),
    }


def _marker(ws, *, create=False):
    base = os.environ.get("TMPDIR")
    if not base or not Path(base).is_absolute():
        return None
    key = hashlib.sha256(str(Path(ws).resolve()).encode()).hexdigest()[:24]
    home = Path(base) / "merlin_background_admission" / key
    if home.is_symlink() or home.parent.is_symlink():
        return None
    if create:
        home.mkdir(mode=0o700, parents=True, exist_ok=True)
        home.chmod(0o700)
    return home / "foreground.json"


def _read(path):
    if path is None or path.is_symlink() or path.stat().st_size > 16384:
        return None
    value = json.loads(path.read_text())
    return value if isinstance(value, dict) else None


def _archive(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("original profile archive is unavailable")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@contextlib.contextmanager
def active_full_selfcheck(*, ws, process, request, source_sha256, public, context, contract, bundle_manifest, sim,
                         dispatcher_pid=None):
    """Publish only an actual ordinary full-ladder worker launched by this broker."""
    path, record = None, None
    try:
        from merlin.targetgen import capsule_runner as CR

        from ..brokers.selfcheck import FORWARDED_PROBES
        from .selfcheck import _adapters, _target_sim_via

        ordinary = (
            request.get("capsules", "all") == "all"
            and not request.get("caller_layout")
            and not request.get("action")
            and not any(request.get(k) for k in FORWARDED_PROBES)
        )
        valid_source = (
            isinstance(source_sha256, str)
            and len(source_sha256) == 64
            and all(c in "0123456789abcdef" for c in source_sha256)
        )
        if ordinary and bundle_manifest is not None and valid_source:
            target, via = _target_sim_via(context)
            adapters, _ = _adapters(sim, target, via)
            selected = set(str(request.get("tiers") or "").split(",")) - {""}
            if selected and selected <= set(adapters):
                adapters = {tier: value for tier, value in adapters.items() if tier in selected}
            elif selected:
                adapters = {}
            full = set(CR.qa_checkpoint_adapters(target, via))
            if full and set(adapters) == full:
                worker, broker = _identity(process.pid), _identity(os.getpid())
                # Only the original one-request handler selects this extra hop;
                # a request body or candidate environment cannot supply it.
                dispatcher = _identity(dispatcher_pid) if dispatcher_pid is not None else None
                manager = _identity(dispatcher["parent"] if dispatcher is not None else os.getppid())
                parent = dispatcher if dispatcher is not None else manager
                if (worker["parent"] == broker["pid"] and broker["parent"] == parent["pid"]
                        and (dispatcher is None or dispatcher["parent"] == manager["pid"])):
                    path = _marker(ws, create=True)
                    if path is not None:
                        record = {
                            "schema": SCHEMA,
                            "nonce": secrets.token_hex(16),
                            "worker": worker,
                            "broker": broker,
                            "manager": manager,
                            "source_sha256": source_sha256,
                            "public": str(Path(public).resolve()),
                            "descriptor": _archive(context.descriptor),
                            "contract": str(Path(contract).resolve()) if contract is not None else None,
                            "profile": _archive(bundle_manifest),
                            "tiers": sorted(full),
                        }
                        if dispatcher is not None:
                            record["dispatcher"] = dispatcher
                        temporary = path.with_suffix(f".{record['nonce']}.tmp")
                        temporary.write_text(json.dumps(record, sort_keys=True))
                        temporary.chmod(0o600)
                        os.replace(temporary, path)
    except Exception:  # noqa: BLE001 -- missing priority identity never changes grading
        path, record = None, None
    try:
        yield
    finally:
        if path is not None and record is not None:
            try:
                if _read(path) == record:
                    path.unlink()
            except (OSError, ValueError):
                pass


def _live_process_chain(record):
    """Reopen every original process hop; dead/reparented owners grant no priority."""
    manager, broker, worker = (record[key] for key in ("manager", "broker", "worker"))
    if (manager != _identity(os.getpid()) or broker != _identity(broker["pid"])
            or worker != _identity(worker["pid"])):
        return False
    parent = manager
    dispatcher = record.get("dispatcher")
    if dispatcher is not None:
        if dispatcher != _identity(dispatcher["pid"]) or dispatcher["parent"] != manager["pid"]:
            return False
        parent = dispatcher
    return broker["parent"] == parent["pid"] and worker["parent"] == broker["pid"]


def foreground_covers(ws, source_sha256, public, inputs):
    """Same-source/cohort priority from a live authorized worker, not QA counts."""
    try:
        record = _read(_marker(ws))
        service = inputs.compiler_profile_service
        return bool(
            record
            and service is not None
            and record.get("schema") == SCHEMA
            and record.get("source_sha256") == source_sha256
            and record.get("public") == str(Path(public).resolve())
            and record.get("descriptor") == _archive(inputs.context.descriptor)
            and record.get("contract")
            == (str(Path(inputs.contract).resolve()) if inputs.contract is not None else None)
            and record.get("profile") == _archive(service.bundle_manifest)
            and record.get("tiers")
            and _live_process_chain(record)
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def source_digest(ws):
    from ..brokers.selfcheck import _submission_digest

    try:
        source = Path(ws) / "submission"
        return _submission_digest(source) if source.is_dir() else None
    except (OSError, ValueError):
        return None


class BackgroundGuard:
    #: How a superseded bounded snapshot treats work after a newer snapshot of different source appears:
    #: ``interrupt`` skips unstarted capsules and gates the next compiler command of started ones;
    #: ``drain`` only skips unstarted capsules (started ones run to their verdict).
    STALE_POLICIES = ("interrupt", "drain")

    def __init__(self, ws, public, inputs, source_sha256, *, finish_bounded_snapshot=False):
        self.ws, self.public, self.inputs, self.source = ws, public, inputs, source_sha256
        self.deferred = None
        self.finish_bounded_snapshot = finish_bounded_snapshot
        from .snapshot_publication import current_token
        from . import snapshot_smoke

        self.token = current_token()
        self.skip_noted = None  # the reason work was actually skipped, once any was
        self.prescreen_failure = None  # why the rest of this snapshot is skipped (smoke), else None
        self.screen = snapshot_smoke.LoadFailureScreen() if snapshot_smoke.enabled() else None
        self.stale_policy = os.environ.get('MERLIN_SNAPSHOT_STALE_POLICY', 'interrupt')
        if self.stale_policy not in self.STALE_POLICIES:
            raise ValueError('MERLIN_SNAPSHOT_STALE_POLICY must be interrupt or drain')
        selected = os.environ.get('MERLIN_BACKGROUND_FINISH_SNAPSHOT', '0')
        if selected not in {'0', '1'}:
            raise ValueError('MERLIN_BACKGROUND_FINISH_SNAPSHOT must be 0 or 1')
        # Legacy selection remains accepted, but may never keep queuing obsolete work.

    def newer_snapshot(self):
        """Token of a newer snapshot of DIFFERENT source (this one is superseded), else None."""
        if not self.finish_bounded_snapshot or self.token is None:
            return None
        from .snapshot_publication import superseding_snapshot

        return superseding_snapshot(self.ws, self.token)

    def smoke(self, tree):
        """Static smoke of the copied compiler before any capsule dispatch."""
        from . import snapshot_smoke

        if self.screen is not None and self.prescreen_failure is None:
            self.prescreen_failure = snapshot_smoke.static_smoke(tree)
        return self.prescreen_failure

    def observe(self, row):
        """Feed a completed redacted row to the load-failure screen."""
        if self.screen is not None:
            self.screen.observe(row)
            if self.prescreen_failure is None and self.screen.reason is not None:
                self.prescreen_failure = self.screen.reason

    def skip_reason(self):
        """Why the rest of this snapshot should not run: a failed smoke or a newer snapshot.

        Handing out a reason means work was (or is about to be) skipped for it: remember that, so a
        snapshot that finished its whole roster before being superseded is not relabelled."""
        if self.prescreen_failure is not None:
            self.skip_noted = self.prescreen_failure
            return self.prescreen_failure
        newer = self.newer_snapshot()
        if newer is not None:
            self.skip_noted = f"superseded by newer snapshot {newer} (different source)"
            return self.skip_noted
        return None

    def interrupt_reason(self):
        """Reason for work that has not begun its next command; ``None`` lets started work drain."""
        if self.prescreen_failure is not None or self.stale_policy == 'interrupt':
            return self.skip_reason()
        return None

    def terminal_status(self):
        """Verdict label for a snapshot that did not run its roster: superseded vs smoke vs deferred."""
        if self.prescreen_failure is not None:
            return 'prescreen_failed'
        return 'superseded_snapshot' if self.finish_bounded_snapshot else 'deferred'

    def ended_early(self):
        """The grade is not a complete verdict: source moved on, or work was skipped as superseded/smoke."""
        return self.reason() is not None or self.skip_noted is not None

    def ended_reason(self):
        return self.deferred or self.skip_noted

    def pending_reason(self):
        """A selected bounded joint smoke finishes its immutable copied source.

        Live edits still revoke current-source attribution and promotion through
        reason/current. They do not invalidate computations on the frozen copy.
        Existing invocation deadlines and the shared CPU lease priorities remain.
        A failed smoke, or a newer snapshot of different source, stops starting more.
        """
        if self.finish_bounded_snapshot:
            from .. import joint_control

            # A trusted evaluator/library handoff or final drain is distinct
            # from ordinary edits. Finish running jobs; do not start more.
            stop = joint_control.stop_reason()
            if stop is not None:
                return stop
            # A newer snapshot owns the grader now: no further admission from this one, queued work
            # is cancelled, and the verdict is recorded as superseded with its partial counts.
            from .snapshot_publication import preempted_reason

            preempted = preempted_reason()
            if preempted is not None:
                self.deferred = preempted
                return preempted
            if joint_control.remaining() <= 0:
                return "joint campaign deadline reached"
            return self.skip_reason()
        return self.reason() or self.prescreen_failure

    def reason(self):
        # Only obsolescence defers pending work. A concurrent same-source author
        # self-check no longer pauses this grade: its leases outrank the grade's
        # in the shared CPU pool (cpu_pool.AUTHOR before cpu_pool.GRADE).
        # Operator patch (diagnostic fast flow): the grade runs on a private copy of the submission, so a live
        # edit must not cancel the unstarted capsules (an author that edits continuously otherwise never gets a
        # complete grade: 320/321 rows were stamped gated). Attribution is still withheld through current().
        return self.deferred

    def current(self):
        return self.source is not None and source_digest(self.ws) == self.source

    def superseded_record(self, verdict):
        """The ``background_grade`` record of a superseded snapshot, with its partial counts."""
        from .snapshot_publication import preempted_reason

        rows = [row for row in (verdict.get("per_capsule") or []) if isinstance(row, dict)]
        status = lambda row: row.get("status")  # noqa: E731
        counts = {
            "n_rows": len(rows),
            "n_pass": sum(status(r) == "pass" for r in rows),
            "n_fail": sum(status(r) == "fail" for r in rows),
            "n_not_run": sum(status(r) == "gated" for r in rows),
            "n_other": sum(status(r) not in {"pass", "fail", "gated"} for r in rows),
            "n_aborted_in_flight": sum(r.get("execution_status") == "aborted" for r in rows),
        }
        return {
            "status": "superseded_snapshot" if self.finish_bounded_snapshot else "deferred",
            "reason": self.deferred,
            "source_sha256": self.source,
            "preempted": preempted_reason() is not None,
            "partial_counts": counts,
        }
