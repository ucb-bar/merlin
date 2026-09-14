"""Run a subprocess and turn a failure into the caller's own error: the helper nine modules re-wrote.

Measured 2026-09-14: baselines/{buddy,executorch,exo}, runtime/{kernel_spike,elf_audit},
runtime/backends/{spike_model,zephyr_model}, llvmlower/codegen and mining/k1 each carried a private
``_run`` doing the same thing -- stringify the argv, capture text output, raise the module's own error on
a non-zero exit with a tail of the output -- with nine different tail lengths and three different answers
to "what happens on a timeout". The error class, the tail and the timeout behaviour stay each caller's
choice; the mechanics live here once.

``wrap_timeout=False`` lets ``subprocess.TimeoutExpired`` propagate unchanged, for callers that relied on
catching it; ``True`` turns it into the caller's error with the limit that fired.
"""
from __future__ import annotations

import subprocess
from collections.abc import Sequence
from typing import Any

_SHOWN_CMD_CHARS = 1000


def run_checked(cmd: Sequence[Any], *, error: type[Exception] = RuntimeError, timeout: float | None = None,
                timeout_hint: str = "", wrap_timeout: bool = True, tail: int = 2000,
                **kw: Any) -> subprocess.CompletedProcess:
    """Run ``cmd`` (each element ``str()``-ed), capturing text output; raise ``error`` on a non-zero exit.

    The message names the exit status, the command (capped at 1000 characters) and the last ``tail``
    characters of stdout and stderr. ``kw`` passes through to :func:`subprocess.run` (``cwd``, ``env``,
    ``input``...).
    """
    argv = [str(c) for c in cmd]
    shown = " ".join(argv)[:_SHOWN_CMD_CHARS]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, **kw)
    except subprocess.TimeoutExpired:
        if not wrap_timeout:
            raise
        raise error(f"command timed out after {timeout}s{timeout_hint}: {shown}") from None
    if proc.returncode != 0:
        raise error(f"command failed (rc {proc.returncode}): {shown}\n"
                    f"STDOUT:{(proc.stdout or '')[-tail:]}\nSTDERR:{(proc.stderr or '')[-tail:]}")
    return proc
