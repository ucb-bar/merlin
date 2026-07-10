"""Route a CHIA (``ucb-bar/chia``) workflow's output into an aet-managed run directory.

CHIA is a Ray-based workflow-graph framework for agentic hardware/software co-design. It
orchestrates; it does not own where results land. This module is the seam that makes a CHIA
loop write into the canonical ``runs/<target>/<suite>/<run-id>/`` layout (CLAUDE.md
"Generated-output convention") via :func:`merlin.common.artifacts.start_run`, with the CHIA
profiler JSONL and scalar metrics dropped in a ``chia/`` subdir of the same run.

CHIA lives in its own virtualenv (it hard-pins ``pydantic``/``ray[default]`` and is not
pip-installable under the name ``chia``), so **every** ``chia``/``ray`` import here is
function-local. Importing this module from the main ``.venv``, where Ray is absent, must keep
working — that is what :func:`chia_available` reports and what the unit tests assert. Same
pattern as the lazy ``aet`` imports in ``merlin.common.artifacts`` and the guarded
``import anthropic`` in ``merlin.common.llm``.

Both :class:`ChiaRun` members that matter here (the aet ``RunHandle`` and CHIA's
``MetricsLogger``) are non-serializable, so they stay on the Ray driver and must never be
captured by a ``@ChiaFunction`` body.
"""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from merlin.common.paths import repo_root

__all__ = [
    "AET_BACKEND", "ChiaRun", "chia_available", "chia_get", "chia_run", "driver_python",
    "require_chia",
]

# ChiaProfileCollector hard-codes this filename under its log_dir (chia/trace/profiler.py).
_PROFILE_LOG_NAME = "ChiaProfileCollector.log"

#: Name our metrics backend registers under, for ``MetricsLogger(backend=AET_BACKEND, ...)``.
AET_BACKEND = "aet"

_backend_cls = None


def chia_available() -> bool:
    """True when ``chia`` and ``ray`` are importable in the running interpreter."""
    try:
        import chia.trace  # noqa: F401
        import ray  # noqa: F401
    except Exception:
        return False
    return True


def require_chia() -> None:
    """Raise with the fix instructions when the interpreter lacks CHIA."""
    if not chia_available():
        raise RuntimeError(
            "CHIA is not importable in this interpreter. CHIA lives in its own venv because it "
            "hard-pins pydantic/ray[default]; installing it into the main .venv would downgrade "
            f"pydantic and perturb concurrent sessions. Run this script with "
            f"{repo_root() / 'build' / 'chia-venv' / 'bin' / 'python'} (create it with: "
            "uv venv build/chia-venv --python 3.13 && "
            "uv pip install --python build/chia-venv -e /scratch/agustin/projects/chia -e .)"
        )


def driver_python() -> str:
    """Interpreter for shelling out to the QA-loop drivers: the main ``.venv``, not this one.

    A CHIA loop script runs under ``build/chia-venv``. The drivers it launches
    (``run_baseline_qa_loop.py`` and friends) must keep running under the main ``.venv`` so no
    ray/mcp/pydantic-2.12 ever crosses into the agent's process tree. Falls back to the current
    interpreter when the main venv is missing.
    """
    p = repo_root() / ".venv" / "bin" / "python"
    return str(p) if p.is_file() else sys.executable


def chia_get(refs, **kwargs):
    """``chia.base.ChiaFunction.get`` that also unwraps a *list* of profiled results.

    Upstream bug (chia 4fd7f3c): ``get()`` post-processes ``ray.get``'s return value with
    ``profiler.on_remote_complete``, which only unwraps a lone ``_ProfiledResult``. Hand it a
    sequence of refs while the profiler is enabled and every element comes back as a
    ``_ProfiledResult`` — contradicting ``get``'s own overload
    ``Sequence[ObjectRef[R]] -> List[R]``. The per-element ``_register_result`` call is skipped
    too, so the profiler's dependency edges are lost for batched gets. CHIA's own examples never
    trip it because they only ``get()`` one ref at a time; a fan-out does nothing else.

    Unwrapping element-wise through ``on_remote_complete`` restores both the value and the
    dependency registration. Drop this shim once the fix lands upstream — it is already a no-op
    against a fixed ``get()``.
    """
    from chia.base.ChiaFunction import get as _get
    from chia.trace.profiler import _ProfiledResult, get_profiler

    value = _get(refs, **kwargs)
    if isinstance(value, list) and any(isinstance(v, _ProfiledResult) for v in value):
        profiler = get_profiler()
        return [profiler.on_remote_complete(v) for v in value]
    return value


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


def _register_backend() -> None:
    """Register :data:`AET_BACKEND` with CHIA's backend table.

    ``chia.trace.metrics._BACKENDS`` is private and CHIA exposes no ``register_backend()``, so an
    out-of-tree backend has to insert itself. That gap is the motivation for the upstream
    ``register_backend(name, cls)`` patch; switch to it once it lands.
    """
    from chia.trace import metrics as _metrics

    register = getattr(_metrics, "register_backend", None)
    if callable(register):  # upstream added the public API
        register(AET_BACKEND, _aet_backend_cls())
    else:
        _metrics._BACKENDS.setdefault(AET_BACKEND, _aet_backend_cls())


@dataclass
class ChiaRun:
    """Driver-side handle for one CHIA workflow bound to an aet run directory.

    ``handle``/``metrics`` are not Ray-serializable — keep them out of ``@ChiaFunction`` bodies.
    Populate :attr:`summary` during the run; :func:`chia_run` writes it via ``finish_run`` on exit.
    """

    handle: object              # merlin.common.artifacts.RunHandle
    metrics: object            # chia.trace.MetricsLogger
    profile_path: Path         # chia/ChiaProfileCollector.log (JSONL) — feeds `chia viz-profile`
    summary: dict = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return self.handle.run_dir


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
):
    """Open an aet run, wire CHIA's profiler + metrics into it, yield a :class:`ChiaRun`.

    Ray is initialized here (once) so ``ray_resources`` — the *logical* resources that
    ``@ChiaFunction(resources=...)`` gates on, e.g. ``{"verilator": 2}`` to cap concurrent
    Verilator-heavy tasks — are declared before any task is dispatched. Ray's own session dir is
    left at its OS default (``/tmp/ray``), outside the repo: its spill/plasma files would
    otherwise pollute the run directory.

    Driver-side only. Exits with ``status="ok"`` or, on any exception, ``status="error"``.
    """
    require_chia()

    import ray

    from merlin.common.artifacts import finish_run, start_run

    handle = start_run(suite=suite, method=method, target=target, seed=seed,
                       run_id=run_id, extra=extra)
    # "chia" is not a RunPaths attribute, so it cannot go through start_run(make_subdirs=...).
    chia_dir = handle.run_dir / "chia"
    chia_dir.mkdir(parents=True, exist_ok=True)

    if not ray.is_initialized():
        ray.init(resources=dict(ray_resources or {}), ignore_reinit_error=True)

    from chia.trace import MetricsLogger, start_collector

    start_collector(log_dir=str(chia_dir))
    _register_backend()
    metrics = MetricsLogger(backend=AET_BACKEND, run_dir=handle.run_dir)

    run = ChiaRun(handle=handle, metrics=metrics, profile_path=chia_dir / _PROFILE_LOG_NAME)
    status = "error"
    try:
        yield run
        status = "ok"
    finally:
        try:
            metrics.close()
        except Exception:
            pass
        finish_run(handle, status=status, summary=run.summary or None)
