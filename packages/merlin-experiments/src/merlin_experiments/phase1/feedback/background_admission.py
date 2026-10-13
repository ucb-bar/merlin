"""Host-owned scheduling admission for advisory background grades.

An active marker grants priority, never numerical authority. It is issued by
the real broker in private storage and expires with its actual process owners.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
from pathlib import Path

SCHEMA = "merlin.background_foreground_admission.v1"


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
def active_full_selfcheck(*, ws, process, request, source_sha256, public, context, contract, bundle_manifest, sim):
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
                worker, broker, manager = _identity(process.pid), _identity(os.getpid()), _identity(os.getppid())
                if worker["parent"] == broker["pid"] and broker["parent"] == manager["pid"]:
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
            and record["manager"] == _identity(os.getpid())
            and record["broker"] == _identity(record["broker"]["pid"])
            and record["worker"] == _identity(record["worker"]["pid"])
            and record["broker"]["parent"] == record["manager"]["pid"]
            and record["worker"]["parent"] == record["broker"]["pid"]
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
    def __init__(self, ws, public, inputs, source_sha256):
        self.ws, self.public, self.inputs, self.source = ws, public, inputs, source_sha256
        self.deferred = None

    def reason(self):
        current = source_digest(self.ws)
        if self.source is not None and current is not None and current != self.source:
            self.deferred = "background source superseded"
        elif foreground_covers(self.ws, self.source, self.public, self.inputs):
            self.deferred = "same-source full selfcheck has foreground priority"
        return self.deferred

    def current(self):
        return self.source is not None and source_digest(self.ws) == self.source


def foreground_active(ws, inputs):
    try:
        public = inputs.public_root() if callable(inputs.public_root) else inputs.public_root
        source = source_digest(ws)
        return source is not None and foreground_covers(ws, source, public, inputs)
    except Exception:  # noqa: BLE001 -- uncertain scheduling identity does not pause feedback
        return False
