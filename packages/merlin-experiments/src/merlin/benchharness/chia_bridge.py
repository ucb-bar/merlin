"""Route a CHIA (``ucb-bar/chia``) workflow's output into an aet-managed run directory.

CHIA is a Ray-based workflow-graph framework for agentic hardware/software co-design. It
orchestrates; it does not own where results land. This module is the seam that makes a CHIA
loop write into the canonical ``runs/<target>/<suite>/<run-id>/`` layout (CLAUDE.md
"Generated-output convention") via :func:`merlin.common.artifacts.start_run`, with the CHIA
profiler JSONL and scalar metrics dropped in a ``chia/`` subdir of the same run.

CHIA is an optional, separately qualified environment (distribution ``chialoops``).
Every ``chia``/``ray`` import here is function-local; importing the bridge never
starts services or requires the optional dependency.

Both :class:`ChiaRun` members that matter here (the aet ``RunHandle`` and CHIA's
``MetricsBackend``) are non-serializable, so they stay on the Ray driver and must never be
captured by a ``@ChiaFunction`` body.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from merlin.common.paths import build_dir, checkout_root

__all__ = [
    "AET_BACKEND",
    "ChiaRun",
    "chia_available",
    "chia_get",
    "chia_python",
    "chia_run",
    "driver_python",
    "require_chia",
]

# ChiaProfileCollector hard-codes this filename under its log_dir (chia/trace/profiler.py).
_PROFILE_LOG_NAME = "ChiaProfileCollector.log"

#: Historical public label for Merlin's directly constructed metrics backend.
AET_BACKEND = "aet"

_backend_cls = None


def _check_public_api() -> None:
    """Reject incompatible optional installs without private compatibility patches."""
    import importlib

    required = {
        "ray": (
            "init",
            "shutdown",
            "is_initialized",
            "cluster_resources",
            "kill",
            "ObjectRef",
            "wait",
            "get",
            "cancel",
        ),
        "chia.trace.metrics": ("MetricsBackend",),
        "chia.trace.profiler": ("reset_profiler", "start_collector", "stop_collector", "get_collector"),
        "chia.base.ChiaFunction": ("get",),
    }
    for module_name, names in required.items():
        module = importlib.import_module(module_name)
        for name in names:
            if not callable(getattr(module, name, None)):
                raise RuntimeError(f"missing public API {module_name}.{name}")


def chia_available() -> bool:
    """Whether this interpreter supplies the required public Chia/Ray contract."""
    try:
        _check_public_api()
    except Exception:
        return False
    return True


def require_chia() -> None:
    """Raise with the fix instructions when the interpreter lacks CHIA."""
    try:
        _check_public_api()
    except Exception as exc:
        raise RuntimeError(
            f"CHIA public API unavailable in {sys.executable}: {exc}. "
            "Install merlin-experiments[chia] in an isolated chia-venv and set "
            "MERLIN_CHIA_PYTHON to its bin/python; do not modify a shared environment."
        ) from exc


def _configured_python(variable: str, *, explicit: str | Path | None = None) -> str | None:
    if explicit is None and variable not in os.environ:
        return None
    value = str(explicit) if explicit is not None else os.environ[variable]
    if not value or not Path(value).is_file() or not os.access(value, os.X_OK):
        source = "explicit interpreter" if explicit is not None else variable
        raise RuntimeError(f"{source} must name an existing executable Python: {value!r}")
    # Do not resolve venv symlinks: their invocation path selects the environment.
    return value


def chia_python(explicit: str | Path | None = None) -> str:
    """CLI override, then MERLIN_CHIA_PYTHON, then canonical out/build/chia-venv."""
    configured = _configured_python("MERLIN_CHIA_PYTHON", explicit=explicit)
    if configured is not None:
        return configured
    path = build_dir() / "chia-venv" / "bin" / "python"
    if path.is_file() and os.access(path, os.X_OK):
        return str(path)
    raise RuntimeError("No chia-venv interpreter; set MERLIN_CHIA_PYTHON explicitly.")


def driver_python() -> str:
    """Interpreter for shelling out to the QA-loop drivers: the main ``.venv``, not this one.

    A CHIA loop script runs under ``build/chia-venv``. The drivers it launches
    (``run_baseline_qa_loop.py`` and friends) must keep running under the main ``.venv`` so no
    ray/mcp/pydantic-2.12 ever crosses into the agent's process tree. Falls back to the current
    interpreter when the main venv is missing.
    """
    configured = _configured_python("MERLIN_EXPERIMENT_PYTHON")
    if configured is not None:
        return configured
    checkout = checkout_root()
    if checkout is not None:
        p = checkout / ".venv" / "bin" / "python"
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return sys.executable


def chia_get(refs, **kwargs):
    """Resolve ordered batches through public scalar get, sharing one timeout budget.

    Official Chia main only unwraps profiled scalar results. No private envelope
    inspection is needed here; callbacks still receive the complete ordered batch.
    """
    from chia.base.ChiaFunction import get as _get

    timeout = kwargs.get("timeout")
    if timeout is not None and (
        isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0
    ):
        raise ValueError("timeout must be finite and nonnegative")
    if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes, bytearray)):
        return _get(refs, **kwargs)
    callback = kwargs.pop("callback", None)
    timeout = kwargs.pop("timeout", None)
    deadline = None if timeout is None else time.monotonic() + timeout
    values = []
    for ref in refs:
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        values.append(_get(ref, timeout=remaining, **kwargs))
    return callback(values) if callback is not None else values


def _aet_backend_cls():
    """Build (once) the ``MetricsBackend`` subclass. Deferred: the base class lives in chia."""
    global _backend_cls
    if _backend_cls is not None:
        return _backend_cls

    from chia.trace.metrics import MetricsBackend

    class AetMetricsBackend(MetricsBackend):
        """Sink CHIA scalars into an aet run dir as ``chia/metrics.jsonl``.

        This lives in *merlin*, not in chia: merlin may depend on both aet (Apache-2.0) and
        chia (BSD-3), so nothing has to be upstreamed and no license boundary is crossed.
        """

        def __init__(self, *, run_dir: str | Path, **_ignored):
            self._path = Path(run_dir) / "chia" / "metrics.jsonl"
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self._path.open("a")

        def log_scalar(self, tag: str, value: float, step: int) -> None:
            self._fh.write(json.dumps({"tag": tag, "value": value, "step": step}) + "\n")

        def flush(self) -> None:
            if not self._fh.closed:
                self._fh.flush()

        def close(self) -> None:
            if not self._fh.closed:
                self._fh.close()

    _backend_cls = AetMetricsBackend
    return _backend_cls


def _record_trace(handle, *, accounting: str) -> None:
    """Read the public collector without replacing the canonical AET run."""
    import ray
    from chia.trace.profiler import get_collector

    from merlin.benchharness.chia_trace import record_trace

    collector = get_collector()
    if collector is None:
        raise RuntimeError("owned Chia collector disappeared before trace collection")
    events = ray.get(collector.get_events.remote(), timeout=10)
    record_trace(handle, events, accounting=accounting)


def _stop_owned_collector(*, started: bool) -> None:
    """Disable implicit inherited-identity flush, including on partial startup."""
    import ray
    from chia.trace.profiler import get_collector, stop_collector

    previous = os.environ.get("CHIA_AET_SINK")
    os.environ["CHIA_AET_SINK"] = "0"
    try:
        stop_collector()
        if not started:
            # start_collector can create its named actor then fail before caching it.
            # Public lookup still finds that orphan in a borrowed Ray cluster.
            orphan = get_collector()
            if orphan is not None:
                ray.kill(orphan)
    finally:
        if previous is None:
            os.environ.pop("CHIA_AET_SINK", None)
        else:
            os.environ["CHIA_AET_SINK"] = previous


@dataclass
class ChiaRun:
    """Driver-side handle for one CHIA workflow bound to an aet run directory.

    ``handle``/``metrics`` are not Ray-serializable — keep them out of ``@ChiaFunction`` bodies.
    Populate :attr:`summary` during the run; :func:`chia_run` writes it via ``finish_run`` on exit.
    """

    handle: object  # merlin.common.artifacts.RunHandle
    metrics: object  # Merlin's chia.trace.metrics.MetricsBackend implementation
    profile_path: Path  # chia/ChiaProfileCollector.log (JSONL) — feeds `chia viz-profile`
    summary: dict = field(default_factory=dict)
    _failed: bool = field(default=False, init=False, repr=False)

    def mark_failed(self) -> None:
        """Record a failed child outcome without discarding its normal result/evidence."""
        self._failed = True

    @property
    def run_dir(self) -> Path:
        return self.handle.run_dir

    @property
    def run_id(self) -> str:
        """Canonical AET run identity used by every subordinate artifact path."""
        return str(self.handle.run_id)


@contextmanager
def chia_run(
    *,
    suite: str,
    method: str,
    target: str,
    seed: int = 0,
    run_id: str | None = None,
    extra: dict | None = None,
    ray_resources: dict | None = None,
    accounting: str = "trace",
):
    """Open an aet run, wire CHIA's profiler + metrics into it, yield a :class:`ChiaRun`.

    Ray is initialized here (once) so ``ray_resources`` — the *logical* resources that
    ``@ChiaFunction(resources=...)`` gates on, e.g. ``{"verilator": 2}`` to cap concurrent
    Verilator-heavy tasks — are declared before any task is dispatched. Ray's own session dir is
    left at its OS default (``/tmp/ray``), outside the repo: its spill/plasma files would
    otherwise pollute the run directory.

    Driver-side only. Exits with ``status="ok"`` or, on any exception, ``status="error"``.
    """
    if accounting not in {"trace", "child-ledgers"}:
        raise ValueError("accounting must be trace or child-ledgers")
    require_chia()

    import ray
    from chia.trace.profiler import get_collector, reset_profiler, start_collector

    from merlin.common.artifacts import finish_run, start_run

    handle = start_run(suite=suite, method=method, target=target, seed=seed, run_id=run_id, extra=extra)
    owned_ray = owned_collector = collector_started = False
    metrics = run = None
    primary = None
    cleanup_errors = []
    publish_trace = os.environ.get("CHIA_AET_SINK") == "1"

    def cleanup(operation):
        try:
            operation()
        except BaseException as exc:
            cleanup_errors.append(exc)

    try:
        chia_dir = handle.run_dir / "chia"
        chia_dir.mkdir(parents=True, exist_ok=True)
        if get_collector() is not None:
            raise RuntimeError("A Chia collector already exists; finish its owning workflow first.")
        if ray.is_initialized():
            available = ray.cluster_resources()
            missing = {key: value for key, value in (ray_resources or {}).items() if available.get(key, 0) < value}
            if missing:
                raise RuntimeError(f"Borrowed Ray cluster lacks requested resources: {missing}")
        else:
            owned_ray = True  # Also clean up partially successful initialization.
            ray.init(address="local", resources=dict(ray_resources or {}), ignore_reinit_error=True)
        reset_profiler()
        owned_collector = True
        start_collector(log_dir=str(chia_dir))
        collector_started = True
        metrics = _aet_backend_cls()(run_dir=handle.run_dir)
        run = ChiaRun(handle=handle, metrics=metrics, profile_path=chia_dir / _PROFILE_LOG_NAME)
        yield run
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if metrics is not None:
            cleanup(metrics.close)
        if owned_collector:
            if collector_started and publish_trace:
                cleanup(lambda: _record_trace(handle, accounting=accounting))
            cleanup(lambda: _stop_owned_collector(started=collector_started))
            cleanup(reset_profiler)
        if owned_ray:
            cleanup(ray.shutdown)
        # The canonical logger is finalized once, after all owned cleanup attempts.
        status = "error" if primary is not None or cleanup_errors or (run is not None and run._failed) else "ok"
        cleanup(lambda: finish_run(handle, status=status, summary=run.summary or None if run else None))
        if cleanup_errors:
            error = primary if primary is not None else cleanup_errors[0]
            for other in cleanup_errors:
                if other is not error:
                    error.add_note(f"Chia cleanup also failed: {type(other).__name__}: {other}")
            if primary is None:
                raise error
