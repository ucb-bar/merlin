"""Host-owned scheduling admission for advisory background grades.

An active marker identifies an authorized foreground full self-check, never
numerical authority. It is issued by the real broker in private storage and
expires with its actual process owners. Background grades no longer pause for
it: CPU ordering between author and grade work is the shared pool's lease
priority (``merlin.common.cpu_pool``); a source edit never defers or closes it either.
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
    """One background grade's immutable snapshot: what it graded, what its cases reach, when it stops.

    An edit never cancels, closes or narrows a grade's admission. Every grade runs its own snapshot to
    the end, like a helper working on its own copy, and its per-case results are reconciled with the
    current source when they are published (:mod:`.reconcile`). Measured on fresh7refcca: closing the
    unstarted queue on every relevant edit gated all 306 cases of all 76 grades in 2.6 h, so the agent
    received no grade results at all. Waste is avoided by lease priority and cross-snapshot
    deduplication (:mod:`.grade_lanes`), not by cancellation. Only final handoff and the joint
    campaign deadline stop new admission.
    """

    def __init__(self, ws, public, inputs, source_sha256, *, finish_bounded_snapshot=False):
        self.ws, self.public, self.inputs, self.source = ws, public, inputs, source_sha256
        self.deferred = None
        self.finish_bounded_snapshot = finish_bounded_snapshot
        self._components = None
        selected = os.environ.get('MERLIN_BACKGROUND_FINISH_SNAPSHOT', '0')
        if selected not in {'0', '1'}:
            raise ValueError('MERLIN_BACKGROUND_FINISH_SNAPSHOT must be 0 or 1')
        # Legacy selection remains accepted; every grade now finishes its snapshot.

    def bind_snapshot(self, candidate):
        """Bind reach and reconciliation identity to the copied compiler, never live edits."""
        from .promotion import submission_digests
        candidate = Path(candidate)
        if candidate.name != "submission":
            raise ValueError("background dependency binding requires original submission snapshot")
        try:
            self._components = dict(submission_digests(candidate.parent)[1]) or None
        except (OSError, ValueError, TypeError):
            self._components = None

    def case_reach(self, capsule):
        """Components of the bound snapshot that ``capsule``'s commands reach; ``None`` is unknown.

        Every case runs every declared command (``capsule_common.run_entrypoints``), so each case
        reaches every component of the decomposition, ``UNATTRIBUTED`` included. A snapshot without
        a decomposition has no attributable reach.
        """
        return frozenset(self._components) if self._components else None

    def reach_identity(self):
        """The snapshot's reconciliation identity (:func:`.reconcile.reach_identity`), or ``None``."""
        from .reconcile import reach_identity

        return reach_identity(self._components)

    def pending_reason(self):
        """Why this grade admits no further case: final handoff or the joint deadline, else ``None``.

        Running work drains on its immutable snapshot either way. A source edit is never a reason.
        """
        if self.finish_bounded_snapshot:
            from .. import joint_control
            stop = joint_control.stop_reason()
            if stop is not None:
                return stop
            if joint_control.remaining() <= 0:
                return "joint campaign deadline reached"
        return None

    def reason(self):
        """Whether the author edited since this snapshot: labels the verdict as an older version's.

        Only a label: admission continues, and each case is reconciled when published.
        """
        current = source_digest(self.ws)
        if (self.source is not None
                and current is not None and current != self.source):
            self.deferred = "background source superseded"
        return self.deferred

    def current(self):
        return self.source is not None and source_digest(self.ws) == self.source
