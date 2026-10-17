"""Optional same-byte simulator verdict reuse, never an inferred hardware source revision.

Only an original native factory adapter and original compiler owner may issue a
request. All semantic inputs and the selected byte-bound engine are reopened at
lookup/publication. Unsupported or incomplete selections always miss.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
import types
import weakref
from pathlib import Path

from merlin.common.bundle_snapshot import verified_snapshot_scope
from merlin.runtime.backends import base
from merlin.runtime.backends.chipyard_rocc_bound import BoundRoCCBackend

ENV = "MERLIN_SIMULATION_CERT_CACHE"
NAMESPACE = "same-byte-simulator-verdict-v2"
_ISSUED = weakref.WeakSet()


def _digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) > 32 << 20:
        raise ValueError("simulation certificate request exceeds its bound")
    return hashlib.sha256(raw).hexdigest()


def _request_digest(request):
    # Discovery stamps a copy-specific location. Bind the complete source tree
    # bytes at that location so targeted cohorts can move without changing their
    # meaning. Reopening it at each lookup also rejects changed sidecar inputs.
    capsule = request["capsule"]
    if "__dir__" not in capsule:
        return _digest(request)
    from merlin_experiments.phase2.retained_revisions import source_inventory

    directory = Path(capsule["__dir__"])
    if directory.resolve() != directory or directory.is_symlink():
        raise ValueError("simulation certificate capsule source must be canonical")
    tree = source_inventory(directory, max_bytes=32 << 20, max_files=8192)
    return _digest({**request, "capsule": {**capsule, "__dir__": {"source_tree_sha256": tree["sha256"]}}})


def _instrument(target):
    from .tier_cache import grading_path

    files = grading_path(target)
    if not files:
        raise ValueError("simulation certificate has no complete grading path")
    digest = hashlib.sha256()
    for path in sorted(files):
        if path.is_symlink() or not path.is_file():
            raise ValueError("simulation certificate grading member is absent or indirect")
        digest.update(str(path).encode())
        digest.update(path.read_bytes())  # Unreadable is a miss, never a synthetic digest marker.
    return digest.hexdigest()  # Never use the process-lifetime legacy memo.


@dataclasses.dataclass(frozen=True, eq=False)
class _Request:
    process: int
    thread: object
    executor: object
    adapter: object
    backend: object
    target: str
    tier: str
    selection: dict
    engine_identity: dict
    package_dir: Path
    source_sha256: str
    profile: dict
    request: dict
    request_sha256: str
    instrument: str


def _selected(adapter, target, tier):
    from merlin_experiments.phase1.feedback.operator_child import adapter_selection

    from . import capsule_runner as runner

    original_code = next(value for value in runner.simulator_adapter.__code__.co_consts
                         if isinstance(value, types.CodeType) and value.co_name == "run")
    if type(adapter) is not types.FunctionType or adapter.__code__ is not original_code:
        raise ValueError("simulation certificate requires an original native factory adapter")
    selected = adapter_selection({tier: adapter}, target=target)[tier]
    closure = dict(zip(adapter.__code__.co_freevars, (cell.cell_contents for cell in adapter.__closure__)))
    policy = closure["readback_policy"]
    if (closure["target"] != target or closure["sim"] != selected["engine"]
            or closure["selection"] != selected["selection"]
            or (policy.record() if policy is not None else None) != selected["readback_policy"]):
        raise ValueError("simulation certificate adapter selection changed")
    return selected


def capture(*, capsule, cb, gold, reference, direct_oracle_golden, policy, llvm_text,
            adapter, target, tier, package_dir):
    """Snapshot existing trusted operator inputs; no new corpus/reference reads."""
    if os.environ.get(ENV) != "1":
        return None
    try:
        from merlin_experiments.phase1.feedback import compiler_profile_child as child

        executor = child.P.active_package_executor()
        if type(executor) is not child.R._Executor or type(executor.owner) is not child.CommonTransformService:
            return None
        owner = executor.owner
        with verified_snapshot_scope(owner.workspace, owner.bundle, repo=owner.repo):
            if child._original_executor() is not executor:
                return None
            profile = child._owner_record(owner)
        selected = _selected(adapter, target, tier)
        backend = base.get_backend(target)
        if type(backend) is not BoundRoCCBackend:
            return None
        if (type(capsule) is not dict or capsule.get("kind") == "model" or type(cb) is not dict
                or not (cb.get("outputs") or any(isinstance(spec, dict) and spec.get("role") == "output"
                    for spec in (cb.get("tensors") or {}).values()))
                or type(gold) is not dict or not gold or type(policy) is not dict
                or type(llvm_text) is not str or not llvm_text
                or type(direct_oracle_golden) is not bool
                or (not direct_oracle_golden and (type(reference) is not dict or not reference))):
            return None
        source = child._source(package_dir)
        if source != executor.source_sha256:
            return None
        request = {"capsule": capsule, "command_buffer_and_stimulus": cb, "gold": gold, "reference": reference,
                   "direct_oracle_golden": direct_oracle_golden, "numeric_policy": policy, "llvm_text": llvm_text}
        value = _Request(os.getpid(), threading.current_thread(), executor, adapter, backend, target, tier,
                         selected, backend.execution_engine_identity(selected["engine"]), Path(package_dir),
                         source, profile, request, _request_digest(request), _instrument(target))
        _ISSUED.add(value)
        return value
    except Exception:  # noqa: BLE001 -- incomplete authority/data is an ordinary cache miss
        return None


def key(request, executables, *, expected=None):
    """Freshly reopen every bound identity before a lookup or accepted cache record."""
    try:
        from merlin_experiments.phase1.feedback import compiler_profile_child as child

        if (os.environ.get(ENV) != '1' or type(request) is not _Request or request not in _ISSUED
                or request.process != os.getpid()
                or request.thread is not threading.current_thread()
                or child.P.active_package_executor() is not request.executor):
            return None, None
        if expected is not None and expected != (request.request['capsule'].get('name'), request.tier, request.target):
            return None, None
        owner = request.executor.owner
        with verified_snapshot_scope(owner.workspace, owner.bundle, repo=owner.repo):
            if child._original_executor() is not request.executor or child._owner_record(owner) != request.profile:
                return None, None
        backend = base.get_backend(request.target)
        if (type(backend) is not BoundRoCCBackend
                or backend.execution_engine_identity(request.selection["engine"]) != request.engine_identity
                or _selected(request.adapter, request.target, request.tier) != request.selection
                or child._source(request.package_dir) != request.source_sha256
                or _request_digest(request.request) != request.request_sha256
                or _instrument(request.target) != request.instrument):
            return None, None
        programs = {}
        for item in executables:
            path = Path(item)
            if path.is_symlink() or not path.is_file() or path.name in programs:
                return None, None
            with path.open("rb") as stream:
                if stream.read(4) != b"\x7fELF":
                    return None, None
                stream.seek(0)
                programs[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        if not programs:
            return None, None
        identity = _digest({"namespace": NAMESPACE, "target": request.target, "tier": request.tier,
                            "programs": programs,
                            "profile": request.profile, "request": request.request_sha256,
                            "engine_identity": request.engine_identity})
        instrument = _digest({"namespace": NAMESPACE, "grading_path": request.instrument,
                              "adapter": request.selection})
        return identity, instrument
    except Exception:  # noqa: BLE001 -- unreadable/changed authority or bytes never produce a hit
        return None, None


def cache_root():
    from .tier_cache import cache_root as selected_root

    try:
        root = selected_root()
        if root is not None:
            root = root / NAMESPACE
            root.mkdir(parents=True, exist_ok=True)
        return root
    except OSError:
        return None


def scope():
    return {"certificate_scope": NAMESPACE, "hardware_source_revision_claim": False}
