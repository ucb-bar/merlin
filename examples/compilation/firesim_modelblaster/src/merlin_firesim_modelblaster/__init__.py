"""ModelBlaster-owned FireSim execution, separate from Merlin's compiler core."""

from __future__ import annotations

import os
import sys
from mmap import ACCESS_READ, mmap
from pathlib import Path

import yaml

from merlin.common.paths import env


def _checkout() -> Path:
    configured = env("MERLIN_MODELBLASTER")
    if not configured:
        raise RuntimeError("MERLIN_MODELBLASTER is unset; select a ModelBlaster checkout in .env or the environment")
    root = Path(configured).resolve()
    if not root.is_dir():
        raise RuntimeError(f"MERLIN_MODELBLASTER does not name a directory: {root}")
    return root


def _runner_file(root: Path) -> Path:
    for path in (
        root / "src" / "modelblaster" / "validation" / "firesim_runner.py",
        root / "validation" / "firesim_runner.py",
    ):
        if path.is_file():
            return path
    raise RuntimeError(f"MERLIN_MODELBLASTER has no validation/firesim_runner.py under {root}")


def _check_workload(firesim_root: str, workload: str) -> None:
    """Without queue-owned config, reject a stale workload before touching FPGA."""
    cfg = Path(firesim_root) / "deploy" / "config_runtime.yaml"
    if not cfg.is_file():
        return
    declared = ((yaml.safe_load(cfg.read_text()) or {}).get("workload", {}) or {}).get("workload_name")
    expected = f"{workload}.json"
    if declared != expected:
        raise RuntimeError(
            f"{cfg} declares workload_name={declared!r} but this run stages into {workload!r}; "
            f"set workload_name to {expected!r} or use the queue, which owns its per-job config"
        )


class ModelBlasterRunner:
    """FireSim adapter retaining the established ModelBlaster run behavior."""

    completion_metric_prefix = "=== MODELBLASTER_WALL_CYCLES ==="

    def preflight(self) -> str:
        """Read-only check of the selected checkout and runner source."""
        return str(_runner_file(_checkout()))

    def __call__(
        self,
        elf: str,
        *,
        firesim_root: str,
        firesim_env: str,
        timeout: int,
        queue: bool,
    ) -> str:
        root = _checkout()
        _runner_file(root)
        with Path(elf).open("rb") as image, mmap(image.fileno(), 0, access=ACCESS_READ) as contents:
            if contents.find(self.completion_metric_prefix.encode("ascii")) < 0:
                raise RuntimeError(
                    f"{elf}: missing {self.completion_metric_prefix!r}; build with the selected runner's "
                    "completion_metric_prefix before submitting to FireSim"
                )
        # ModelBlaster supports both package and historical flat layouts. Its
        # own imports require these roots on sys.path; the mutation is contained
        # in this opt-in adapter, never in Merlin's shared runtime.
        for path in (str(root / "src"), str(root)):
            if path not in sys.path:
                sys.path.insert(0, path)
        # ModelBlaster's runner binds these names at import time.
        os.environ.setdefault("FIRESIM_WORKLOAD_NAME", "merlin-oscar")
        os.environ.setdefault("FIRESIM_PROJECT", "merlin-oscar")
        socket = os.environ.get("FIRESIM_SSH_AUTH_SOCK", "/tmp/firesim_ssh_agent.sock")
        current = os.environ.get("SSH_AUTH_SOCK", "")
        if os.path.exists(socket) and (not current or not os.path.exists(current)):
            os.environ["SSH_AUTH_SOCK"] = socket
        if queue:
            os.environ["FIRESIM_QUEUE"] = "1"
            os.environ.setdefault("FIRESIM_QUEUE_TIMEOUT", str(timeout))
        else:
            _check_workload(firesim_root, os.environ["FIRESIM_WORKLOAD_NAME"])
        try:
            from modelblaster.validation.firesim_runner import run_firesim
        except ModuleNotFoundError as exc:
            if exc.name not in {"modelblaster", "modelblaster.validation", "modelblaster.validation.firesim_runner"}:
                raise
            from validation.firesim_runner import run_firesim
        module = sys.modules.get(run_firesim.__module__)
        origin = getattr(module, "__file__", None)
        if origin is None or not Path(origin).resolve().is_relative_to(root):
            raise RuntimeError(f"ModelBlaster runner imported from outside selected MERLIN_MODELBLASTER: {origin}")
        return run_firesim(
            str(elf), models=None, firesim_root=firesim_root, firesim_env=firesim_env, timeout=float(timeout)
        )


runner = ModelBlasterRunner()
