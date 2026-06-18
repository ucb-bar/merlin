"""SpacemiT K1 board adapter — real RVV silicon (VLEN=256, Bianbu Linux/glibc).

K1 is the fast real-hardware cycle target that pairs with spike (spike = correctness +
instruction evidence; K1 = cycle truth). It is Linux-hosted (no HTIF/bare-metal), so it needs a
Linux userspace runtime cross-compiled with the SpacemiT toolchain, scp'd to the board, run, and
parsed for the same OUT/METRIC/DONE markers via zephyr_model._parse_console.

This module is intentionally fail-closed: when the toolchain or the board is unavailable,
:func:`available` returns False and the runner records the K1 rung as ``not_run`` (NEVER a false
pass). The cross-compile + deploy path is built out in S2.4-S2.6; until then ``available()`` is
False so the coupled runner still works on spike alone.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

# Board access (memory: spacemit-k1-board-access). IP is a DHCP lease — override via env.
K1_SSH_KEY = os.environ.get("MERLIN_K1_SSH_KEY", "/scratch2/agustin/DIMA_SLICE")
K1_HOST = os.environ.get("MERLIN_K1_HOST", "")  # e.g. root@10.44.97.186; empty => discover/unset
# SpacemiT cross-toolchain (copied into build_tools/SpacemiT/ in S2.4).
_REPO = Path(__file__).resolve().parents[4]
K1_TOOLCHAIN = Path(os.environ.get(
    "MERLIN_K1_TOOLCHAIN", str(_REPO / "build_tools" / "SpacemiT" / "riscv-tools-spacemit")))

VLEN = 256  # K1 X60 vector length, bits; recorded per-run from `vlenb` once the runtime lands.


def toolchain_cc() -> Path | None:
    """Path to the SpacemiT linux-gnu clang/gcc, or None if the toolchain isn't installed yet."""
    for rel in ("bin/clang", "bin/riscv64-unknown-linux-gnu-gcc"):
        c = K1_TOOLCHAIN / rel
        if c.is_file():
            return c
    return None


def available() -> bool:
    """True only if BOTH the cross-toolchain is installed AND the board is reachable over SSH.

    Fail-closed: any uncertainty -> False -> runner records ``not_run`` (never a false pass).
    """
    if toolchain_cc() is None or not K1_HOST or not Path(K1_SSH_KEY).is_file():
        return False
    if shutil.which("ssh") is None:
        return False
    import subprocess
    try:
        r = subprocess.run(
            ["ssh", "-i", K1_SSH_KEY, "-o", "BatchMode=yes",
             "-o", "ConnectTimeout=5", K1_HOST, "true"],
            capture_output=True, timeout=15)
        return r.returncode == 0
    except Exception:
        return False


def run_on_k1(model_dir: str | Path, work: str | Path, pkg, *, timeout: int = 600) -> dict[str, Any]:
    """Cross-compile the workload for K1, deploy, run, and parse OUT/METRIC/DONE.

    NOT YET IMPLEMENTED (S2.5/S2.6). Raises ``NotImplementedError`` so the runner records the K1
    rung as ``not_run`` with a clear reason rather than fabricating a result.
    """
    raise NotImplementedError(
        "K1 Linux runtime (cross-compile + scp + rdcycle) lands in S2.5/S2.6; "
        "until then certify_rvv records the K1 rung as not_run.")
