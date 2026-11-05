"""Compare fresh ordinary oracle artifacts without inventing replay authority.

The normal engine's unspecified reset/seed still requires fresh execution. An
optional closed native filesystem view and parent-published archive retain
actual observations; reports never become executable cache entries.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import inspect
import json
import os
import threading
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.runtime import _native_files as F
from merlin.runtime.native_runtime import active_native_runtime

from ._artifact_index import ArtifactRetestIndex, prepare_artifact_retest_index

__all__ = [
    "ArtifactRetestIndex",
    "prepare_artifact_retest_index",
    "active_artifact_retests",
    "compare_selected_artifacts",
    "retain_selected_artifacts",
    "observe_public_artifact_retests",
]

_ACTIVE = contextvars.ContextVar("ordinary_artifact_retesting", default=None)
_MAX_ENTRIES = 128
_MAX_DOCUMENT_BYTES = 1024 * 1024
_MISSING_RUNTIME = (
    "the selected native simulator does not execute under original-owner frozen "
    "system runtime mounts; transitive loader/library dependencies remain UNKNOWN"
)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _pin(path):
    path = Path(path)
    if path.resolve() != path or path.is_symlink() or not path.is_file():
        raise ValueError("artifact comparison requires a canonical regular file")
    before = F.identity(path.stat())
    result = {"path": str(path), "sha256": F.digest(path)}
    if before != F.identity(path.stat()):
        raise ValueError("artifact changed during comparison")
    return result


class _ArtifactRetests:
    def __init__(self, root, context, index=None):
        self.root = Path(root)
        if (
            self.root.resolve() != self.root
            or self.root.is_symlink()
            or not self.root.is_dir()
            or self.root.stat().st_uid != os.getuid()
            or self.root.stat().st_mode & 0o077
        ):
            raise ValueError("artifact observations require the original canonical job output")
        self.context_sha256 = _sha(_json(context).encode())
        self.entries = {}
        self.last_groups = {}
        self.lock = threading.RLock()
        self.index = index
        if index is not None:
            if type(index) is not ArtifactRetestIndex:
                raise ValueError("native artifact observation requires the exact live original index")
            index.verify(context=context, runtime_owner=active_native_runtime())

    def compare(
        self, *, elf, cb, inputs, lowered_mlir, build, backend, provenance, capture_bytes=False, memory_transport=False
    ):
        """Read the fresh whole build and selected tools before actual dispatch."""
        elf = Path(elf)
        if not elf.is_relative_to(self.root):
            raise ValueError("fresh artifact is outside its original job output")
        if type(build) is not dict or build.get("elf_sha256") != _pin(elf)["sha256"]:
            raise ValueError("fresh artifact has no verified complete harness receipt")
        paths = [elf, elf.parent / "harness.c", elf.parent / build.get("kernel_object_name", "kernel.o")]
        pins = [_pin(path) for path in paths]
        if pins[1]["sha256"] != build.get("harness_sha256") or pins[2]["sha256"] != build.get("kernel_object_sha256"):
            raise ValueError("fresh artifact differs from its completed harness receipt")
        source = inspect.getsourcefile(backend.run_elf)
        if source is None:
            raise ValueError("selected process owner source is unavailable")
        owner = _pin(Path(source).resolve())
        tools = []
        for path in sorted(elf.parent.rglob("invocation.json")):
            record = I.verify(path)
            if record["stage"] == "execution":
                continue
            tools.append(
                {
                    "stage": record["stage"],
                    "kind": record["kind"],
                    "executable": record["executable"],
                    "dependencies": record["dependencies"],
                    "environment": record.get("environment"),
                }
            )
        tools.sort(key=_json)
        closure = {
            "original_context_sha256": self.context_sha256,
            "build": build,
            "inputs_sha256": _sha(_json(inputs).encode()),
            "command_buffer_sha256": _sha(_json(cb).encode()),
            "lowered_mlir_sha256": _sha(lowered_mlir.encode()),
            "selected_process_owner": owner,
            "selected_engine": provenance,
            "effective_environment": I.environment_identity(dict(os.environ)),
            "observed_build_tools": tools,
        }
        runtime = active_native_runtime()
        native_identity = None
        if runtime is not None and not memory_transport:
            try:
                native_identity = runtime.execution_identity(backend, elf, capture_bytes=capture_bytes)
                closure["native_runtime"] = native_identity
            except (ValueError, OSError, KeyError, TypeError):
                pass
        encoded = _json(closure).encode()
        if len(encoded) > _MAX_DOCUMENT_BYTES:
            raise ValueError("artifact comparison exceeds its bounded observation size")
        group = _json({"target": build["target"], "readback_policy": build["readback_policy"]})
        current = _sha(encoded)
        key = current  # Complete artifact/input/native identity; capsules cannot overwrite each other.
        report = {
            "schema": "merlin.ordinary_artifact_retesting.v1",
            "comparison": "first_observation",
            "artifact_context_sha256": current,
            "execution_reused": False,
            "decision": "execute",
            "runtime_dependency_closure": {"status": "UNKNOWN", "reason": _MISSING_RUNTIME},
            "scope": "fresh artifact and known selected input comparison; no replay, timing or numerical verdict",
        }
        if native_identity is not None:
            report["runtime_filesystem_view"] = native_identity["runtime"]
            report["runtime_dependency_closure"] = native_identity["reset_seed_closure"]
        if self.index is not None and native_identity is not None:
            try:
                archived = self.index.original_observation(key)
                if archived is not None:
                    body, _ = archived
                    report["comparison"] = "unchanged_artifacts"
                    report["previous_observation"] = body["invocation"]["path"]
                    report["archive"] = "verified_original_native_observation"
                    # Filesystem closure does not close the original engine's
                    # unspecified reset/seed. Never manufacture a replay hit.
            except (ValueError, OSError, TypeError, KeyError):
                report["archive"] = "UNKNOWN"
        with self.lock:
            previous = self.entries.get(key) or self.entries.get(self.last_groups.get(group))
            if previous is not None:
                try:
                    for pin in previous["pins"]:
                        if _pin(Path(pin["path"])) != pin:
                            raise ValueError("previous artifact changed")
                    record = I.verify(Path(previous["invocation"]))
                    if record["stage"] != "execution" or record["kind"] != "python_call":
                        raise ValueError("previous dispatch is unrelated")
                    report["comparison"] = (
                        "unchanged_artifacts" if previous["closure_sha256"] == current else "changed_artifacts"
                    )
                    report["previous_observation"] = previous["invocation"]
                except (ValueError, OSError, TypeError, KeyError):
                    report["comparison"] = "uncertain_previous_observation"
            self.last_groups[group] = key
            if len(self.last_groups) > _MAX_ENTRIES:
                self.last_groups.pop(next(iter(self.last_groups)))
        return key, current, pins, report

    def retain(self, observation, *, invocation, console):
        key, current, pins, report = observation
        path = Path(invocation)
        record = I.verify(path)
        payload = console if type(console) is bytes else console.encode("utf-8")
        if (
            not path.is_relative_to(self.root)
            or record["stage"] != "execution"
            or record["kind"] != "python_call"
            or record.get("outcome") != "returned"
            or pins[0] not in record["inputs"]
            or Path(record["stdout"]["path"]).read_bytes() != payload
            or any(_pin(Path(pin["path"])) != pin for pin in pins)
        ):
            raise ValueError("artifact comparison requires its actual completed original dispatch")
        with self.lock:
            if key not in self.entries and len(self.entries) >= _MAX_ENTRIES:
                self.entries.pop(next(iter(self.entries)))
            self.entries[key] = {"closure_sha256": current, "pins": pins, "invocation": str(path)}
        if self.index is not None:
            try:
                owner = active_native_runtime()
                if owner is not self.index.runtime_owner:
                    raise ValueError("native artifact observation lost its original runtime owner")
                native = owner.completed_observation(elf=Path(pins[0]["path"]), console=console)
                self.index.stage_observation(self.root, key=key, pins=pins, invocation=path, native=native)
                report["archive_publication"] = "pending_original_guardian_cleanup"
            except (ValueError, OSError, TypeError, KeyError):
                report["archive_publication"] = "UNKNOWN"
        (Path(pins[0]["path"]).parent / "artifact_retesting.json").write_text(_json(report) + "\n")
        return report


def active_artifact_retests():
    """Return only the explicit live job observer, never a saved report."""
    return _ACTIVE.get()


def compare_selected_artifacts(*, simulator, service, **kwargs):
    observer = active_artifact_retests()
    if observer is None or simulator != "gsim" or service is not None:
        return None
    try:
        return observer, observer.compare(**kwargs)
    except (ValueError, OSError, TypeError, KeyError):
        return observer, None


def retain_selected_artifacts(selection, *, invocation, console):
    observer, observation = selection
    if observation is not None:
        try:
            return observer.retain(observation, invocation=invocation, console=console)
        except (ValueError, OSError, TypeError, KeyError):
            pass
    return {
        "schema": "merlin.ordinary_artifact_retesting.v1",
        "comparison": "uncertain_artifact_observation",
        "execution_reused": False,
        "decision": "execute",
        "runtime_dependency_closure": {"status": "UNKNOWN", "reason": _MISSING_RUNTIME},
    }


@contextlib.contextmanager
def observe_public_artifact_retests(root, *, context, index=None):
    """Observe one public development job; no cross-job/worker replay is supplied.

    The host selects this after original normal functional/corpus admission.
    Protected final and qualification callers never select it. Copied thread
    contexts may share this bounded observer under the same original job root.
    """
    observer = _ArtifactRetests(root, context, index=index)
    token = _ACTIVE.set(observer)
    try:
        yield observer
    finally:
        _ACTIVE.reset(token)
