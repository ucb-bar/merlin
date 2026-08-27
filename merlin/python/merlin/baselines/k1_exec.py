"""Generic K1 deploy/run for external baselines + a board lock (single physical board).

The frameworks build their OWN binaries (not merlin's model.o), so we can't route them through
``merlin.mining.k1.run_on_k1`` (which is merlin-package-specific). This module provides a thin,
framework-agnostic push/run over the same SSH plumbing and env vars, plus:

  * a **board lock** — one physical K1, so parallel per-framework agents must serialize their
    *on-board* runs. A cross-process ``flock`` on a lockfile under ``artifacts/cache/baselines/``
    makes "builds parallel, board runs serial" safe.
  * **fail-closed** — if the toolchain or board is unavailable, :func:`board_available` is False and
    :func:`push`/:func:`run` raise :class:`BoardUnavailable`, so the runner records ``not_run`` with
    a reason instead of fabricating a timing.
"""
from __future__ import annotations

import contextlib
import fcntl
import subprocess
from pathlib import Path

from merlin.common.artifacts import cache_dir
from merlin.mining import k1

# Reuse merlin's board config verbatim (env: MERLIN_K1_HOST / MERLIN_K1_SSH_KEY / MERLIN_K1_REMOTE_DIR).
K1_HOST = k1.K1_HOST
K1_SSH_KEY = k1.K1_SSH_KEY
K1_REMOTE_DIR = k1.K1_REMOTE_DIR

_SSH_OPTS = ["-i", K1_SSH_KEY, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]


class BoardUnavailable(RuntimeError):
    """Raised when the K1 board/toolchain is not reachable (fail-closed)."""


def board_available() -> bool:
    """True iff the SpacemiT toolchain is installed AND the board answers SSH (delegates to k1)."""
    return k1.available()


def board_vlenb() -> int | None:
    """RVV VLEN in bytes as reported by the board (256-bit K1 -> 32), or None if unreachable."""
    try:
        return k1.board_vlenb()
    except Exception:  # noqa: BLE001
        return None


@contextlib.contextmanager
def board_lock(timeout: float | None = None):
    """Serialize on-board execution across processes (single physical K1).

    Blocking by default; pass ``timeout`` seconds to fail fast if another agent holds the board.
    """
    lock_path = cache_dir("baselines") / "k1_board.lock"
    fh = open(lock_path, "w")
    try:
        if timeout is None:
            fcntl.flock(fh, fcntl.LOCK_EX)
        else:
            import time
            deadline = time.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"K1 board busy > {timeout}s (held by another run)")
                    time.sleep(1.0)
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def _require_board() -> None:
    if not board_available():
        raise BoardUnavailable(
            "K1 board unavailable (set MERLIN_K1_HOST to a reachable board and "
            "MERLIN_K1_TOOLCHAIN to the SpacemiT toolchain)")


def push(local: str | Path, remote: str | None = None, *, timeout: int = 300) -> str:
    """scp a file to the board. Returns the remote path. Fail-closed if the board is down."""
    _require_board()
    local = Path(local)
    remote = remote or f"{K1_REMOTE_DIR}/{local.name}"
    subprocess.run(["ssh", *_SSH_OPTS, K1_HOST, f"mkdir -p {K1_REMOTE_DIR}"],
                   check=True, capture_output=True, timeout=60)
    r = subprocess.run(["scp", *_SSH_OPTS, str(local), f"{K1_HOST}:{remote}"],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise BoardUnavailable(f"scp {local} -> {remote} failed: {r.stderr[:200]}")
    return remote


def run(argv: list[str], *, timeout: int = 600, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run a command on the board over SSH and return the completed process (stdout/stderr captured).

    Does NOT take the board lock itself — wrap the whole build+push+run+parse in ``board_lock()``.
    """
    _require_board()
    prefix = ""
    if env:
        prefix = " ".join(f"{k}={v}" for k, v in env.items()) + " "
    cmd = prefix + " ".join(argv)
    return subprocess.run(["ssh", *_SSH_OPTS, K1_HOST, cmd],
                          capture_output=True, text=True, timeout=timeout)
