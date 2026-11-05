"""Find, build and start the Bubble Tea frontend; fall back to the plain renderer without it.

The frontend binary is looked up as ``$MERLIN_CONSOLE_BIN``, then ``<out/build>/console/merlin-console``
(:func:`merlin.common.paths.build_dir`), then ``merlin-console`` on ``PATH``. :func:`build` compiles it
from ``packages/merlin-experiments/console`` with the Go toolchain named by ``$MERLIN_CONSOLE_GO`` or
found on ``PATH``. The frontend receives this interpreter as ``--python`` so its data process imports the
same installation.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

BINARY = "merlin-console"


def build_root() -> Path:
    from merlin.common.paths import build_dir

    return build_dir() / "console"


def source_dir() -> Path:
    from merlin.common.paths import repo_root

    return repo_root() / "packages" / "merlin-experiments" / "console"


def binary() -> Path | None:
    candidates = [os.environ.get("MERLIN_CONSOLE_BIN")]
    try:
        candidates.append(str(build_root() / BINARY))
    except Exception:  # noqa: BLE001 -- no checkout: only the environment and PATH remain
        pass
    candidates.append(shutil.which(BINARY))
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return Path(candidate)
    return None


def go_tool() -> str | None:
    return os.environ.get("MERLIN_CONSOLE_GO") or shutil.which("go")


def build(*, go: str | None = None, quiet: bool = False) -> Path:
    """Compile the frontend into the build root and return the binary path."""
    go = go or go_tool()
    if not go:
        raise SystemExit("building the console needs Go >= 1.26 on PATH or $MERLIN_CONSOLE_GO")
    source = source_dir()
    if not (source / "go.mod").is_file():
        raise SystemExit(f"console source not found at {source}")
    out = build_root() / BINARY
    out.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "GOFLAGS": os.environ.get("GOFLAGS", "-mod=readonly")}
    result = subprocess.run(
        [go, "build", "-trimpath", "-o", str(out), "."],
        cwd=source,
        env=env,
        capture_output=quiet,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"console build failed ({result.returncode}){': ' + result.stderr[-800:] if quiet else ''}")
    return out


def frontend_argv(
    path: Path,
    *,
    operator_private: bool = False,
    tasks=(),
    run_dirs=(),
    experiment=None,
    tasks_root=None,
    interval=None,
) -> list[str]:
    argv = [str(path), "--python", sys.executable]
    if operator_private:
        argv.append("--operator-private")
    for task in tasks or ():
        argv += ["--task", str(task)]
    for run in run_dirs or ():
        argv += ["--run-dir", str(run)]
    if experiment:
        argv += ["--experiment", str(experiment)]
    if tasks_root:
        argv += ["--tasks-root", str(tasks_root)]
    if interval:
        argv += ["--interval", str(interval)]
    return argv


def exec_frontend(argv: list[str]) -> int:
    """Replace this process with the frontend (the terminal belongs to it from here)."""
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(argv[0], argv)
    return 0  # pragma: no cover -- execv does not return


__all__ = ["BINARY", "binary", "build", "build_root", "exec_frontend", "frontend_argv", "go_tool", "source_dir"]
