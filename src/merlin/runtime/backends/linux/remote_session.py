"""Remote Linux board adapter: cross-compile for a RISC-V Linux board, deploy over SSH, run, pull outputs.

A board whose catalog entry says ``flow: linux`` is real RVV silicon running a glibc userspace. It has
no HTIF or bare-metal console, so it needs a cross-compiled Linux runtime that reports the same
OUT/METRIC/DONE markers ``zephyr_execution._parse_console`` reads. Every fact about the board -- its
``-march``/``-mabi``/target triple, vector length, timebase and core clock, model-thread stack, remote
staging directories and their limits, OpenMP binding -- is a field of its catalog entry
(:class:`merlin.runtime.boards.catalog.Board`). Host-specific access -- SSH key, host, port and the
cross-toolchain install -- comes from per-board environment variables ``MERLIN_<BOARD>_<WHAT>``
(:func:`env_name`), read through ``paths.env`` so a repo-local ``.env`` configures them.

This module is intentionally fail-closed: when the toolchain or the board is unavailable,
:func:`available` returns False and the runner records the board rung as ``not_run`` (NEVER a false
pass). The cross-compile + deploy path reuses the SAME portable artifacts as the spike build -- the
LLVM IR ``model.ll`` (``llvmlower.lower``) and the data-driven C runtime (``llvmlower.c_runtime`` +
``src/merlin/runtime/native/c/merlin_model.c``); only the harness differs:

  * spike/Zephyr -- bare-metal: ``-mcmodel=medany -ffreestanding -nostdlib``, HTIF console, mcycle.
  * a Linux board -- glibc userspace: the board's ``-march``/``-mabi`` at ``-O2`` (NO
    medany/freestanding), a ``main()`` that prints OUT/METRIC/DONE, timed with the user-readable
    ``time`` CSR at the board's declared ``timebase_hz``.

Every public entry point takes the board explicitly; nothing here selects a board by default.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

from merlin.common import external_tools as _toolchains
from merlin.common.paths import env as _env
from merlin.common.paths import target_env_name
from merlin.common.process import invocation_records as _proc

# ---- remote_linux ------------------------------------------------------------------------------

if TYPE_CHECKING:
    from merlin.runtime.boards.catalog import Board


class RemoteLinuxError(RuntimeError):
    """A build, deploy or run step for a remote Linux board failed."""


class VlenMismatch(RuntimeError):
    """The board reported a vector length different from the one codegen pinned."""


# ---- the board profile ---------------------------------------------------------------


def env_name(board: Board, what: str) -> str:
    """The per-board environment variable ``MERLIN_<BOARD>_<WHAT>`` (``paths.target_env_name``)."""
    return target_env_name(board.name, what)


def binary_name(board: Board) -> str:
    """The deployed binary's name, also the harness banner and the lock/probe file prefix."""
    return f"merlin_{board.name}"


def substrate(board: Board) -> str:
    """The measurement-substrate label this board's numbers are recorded under.

    Target contracts name the same label as a measurement authority (``measurement.wall_from``) and
    :func:`merlin.analysis.perf.measurement.authority.pick` matches measurements on it, so callers read it from the
    board instead of restating it. Defaults to the board name.
    """
    return _require_linux(board).substrate or board.name


def _require_linux(board: Board) -> Board:
    from merlin.runtime.boards.catalog import FLOW_LINUX

    if getattr(board, "flow", None) != FLOW_LINUX:
        raise RemoteLinuxError(f"board {getattr(board, 'name', board)!r} is not a `flow: {FLOW_LINUX}` board")
    return board


def resolve_board(name: str, catalog: str | Path | None = None) -> Board:
    """The ``flow: linux`` board ``name`` from ``catalog``, else from the selected board catalog."""
    from merlin.runtime.boards import catalog as boards

    if catalog is None:
        selected = boards.board(name)
    else:
        found = boards.load_boards(catalog)
        if name not in found:
            raise boards.BoardRegistryError(f"board {name!r} is absent from {catalog}")
        selected = found[name]
    return _require_linux(selected)


@dataclass(frozen=True)
class Profile:
    """One ``flow: linux`` board with this host's settings for it resolved.

    ``march``, ``vlen``, ``omp_threads`` and ``remote_dir`` are catalog facts a host may override
    (``MERLIN_<BOARD>_MARCH`` / ``_VLEN`` / ``_OMP_THREADS`` / ``_REMOTE_DIR``); the access fields
    exist only on the host. Empty access fields read as "unset" and fail closed.
    """

    board: Board
    march: str
    vlen: int
    omp_threads: int
    remote_dir: str
    host: str
    ssh_key: str
    ssh_port: str
    toolchain_install: Path | None

    @property
    def name(self) -> str:
        return self.board.name

    @property
    def binary(self) -> str:
        return binary_name(self.board)

    def env(self, what: str) -> str:
        return env_name(self.board, what)

    @property
    def triple(self) -> str:
        return str(self.board.triple)

    @property
    def target_flag(self) -> str:
        return f"--target={self.triple}"

    @property
    def ssh_port_opts(self) -> list[str]:
        return ["-p", self.ssh_port] if self.ssh_port else []  # ssh uses -p

    @property
    def scp_port_opts(self) -> list[str]:
        return ["-P", self.ssh_port] if self.ssh_port else []  # scp uses -P (capital)

    def scratch_path(self, name: str) -> str:
        return f"{str(self.board.scratch_dir).rstrip('/')}/{name}"


def profile(board: Board | Profile) -> Profile:
    """Resolve ``board``'s profile against this host's environment (process env, then ``.env``)."""
    if isinstance(board, Profile):
        return board
    _require_linux(board)
    toolchain_var = env_name(board, "TOOLCHAIN")
    # The toolchain's layout -- the vendor's extracted-release directory naming, and the install used
    # when the variable is unset -- is declared by the owning target contract under
    # ``runtime.toolchain`` (keyed by the variable) and read through merlin.common.external_tools.
    install = _env(toolchain_var, "") or _toolchains.default_install(toolchain_var)
    return Profile(
        board=board,
        march=_env(env_name(board, "MARCH"), board.march),
        vlen=int(_env(env_name(board, "VLEN"), str(board.vlen))),
        omp_threads=int(_env(env_name(board, "OMP_THREADS"), str(board.harts))),
        remote_dir=_env(env_name(board, "REMOTE_DIR"), board.remote_dir),
        host=_env(env_name(board, "HOST"), "") or "",  # e.g. root@<board-ip>; empty => unreachable
        ssh_key=_env(env_name(board, "SSH_KEY"), "") or "",  # path to the SSH private key
        # Optional non-standard SSH port, for a network that filters inbound :22 to the board's
        # segment while passing high ports. Empty => the default port 22.
        ssh_port=_env(env_name(board, "SSH_PORT"), "") or "",
        toolchain_install=Path(install) if install else None,
    )


def verify_vlen(board: Board | Profile, reported_vlenb: int, vlen: int | None = None) -> None:
    """Fail closed when the board's actual ``vlenb`` contradicts the VLEN we compiled against.

    ``vlenb`` is VLEN in BYTES and the harness banner already prints it, so the true value is
    observable on every run -- there is no reason to keep trusting a constant. Pinning a VLEN larger
    than the hardware's is not a lost optimization but a MISCOMPILE (the backend sizes register
    groups for lanes that do not exist), so a mismatch must stop the run rather than be measured."""
    p = profile(board)
    want = int(vlen or p.vlen)
    got = int(reported_vlenb) * 8
    if got != want:
        raise VlenMismatch(
            f"board reports VLEN={got} bits (vlenb={reported_vlenb}) but codegen pinned {want}. "
            f"Set {p.env('VLEN')}={got} for this board, or drop the pin. Pinning a VLEN the hardware "
            f"lacks miscompiles; pinning a smaller one silently idles the datapath."
        )


def codegen_march(board: Board | Profile, march: str | None = None, vlen: int | None = None) -> str:
    """``march`` with the board's REAL vector length pinned (``_zvl<VLEN>b``).

    Why this is not cosmetic. ``-march=rv64gcv`` only promises the RVV *minimum* VLEN of 128 bits
    (``zvl128b``), so when our codegen emits a FIXED-width vector the LLVM backend must size the
    register group for that worst case: on a VLEN=256 core a ``vector<16xf32>`` (512 bits) becomes
    ``e32,m4`` and a ``vector<32xf32>`` becomes ``e32,m8`` — exactly DOUBLE the LMUL the hardware
    needs. Two costs follow, and together they were the whole "vector width vs IPC" tradeoff
    (measured, f32 GEMM 128^3):

      * REGISTER PRESSURE doubles. An ``m8`` value occupies 8 of the 32 architectural vector
        registers, so an MR=4/NR=32 register block wants 32 registers for accumulators alone and the
        allocator spills inside the K loop (objdump: 25 inner-loop instructions with 4 ``vl8r.v``/
        ``vs8r.v``, vs 12 and none once VLEN is pinned). MR=8/NR=32 went 92 -> 46.
      * HALF THE DATAPATH IS IDLE. ``vsetivli zero, 16, e32, m4`` on a VLEN=256 core sets ``vl=16``
        against a ``VLMAX`` of 32 — the instruction reserves a 4-register group and uses half of it.

    An expert kernel never hits this because it sizes to the vector length it *queries* at run time
    (XNNPACK's ``__riscv_vsetvl_e32m4``), which is why its advantage looked like "lanes per issue".
    Pinning the known target VLEN recovers the same lanes-per-issue by CODE GENERATION.

    NOTE ON GENERALITY: pinning a known VLEN is a STOPGAP that trades portability for lanes. The
    target-agnostic fix is VL-agnostic codegen -- `MicrokernelSpec.vl_strategy = VL_DYNAMIC`, a
    vsetvli loop that sizes to whatever VLEN the hardware reports at run time. Until that axis is
    emitted, this pin recovers the same lanes for boards whose VLEN we can state and verify
    (:func:`verify_vlen`).

    Kept separate from the board's plain ``march`` (which the C glue and the cross-framework baseline
    arms compile against) so pinning VLEN for our own codegen never silently re-flags a baseline
    mid-campaign.
    """
    p = profile(board)
    m = march or p.march
    v = int(vlen or p.vlen)
    return m if f"zvl{v}b" in m else f"{m}_zvl{v}b"


def _board_lock_path(p: Profile) -> Path:
    """A host-wide lockfile identifying THIS board. Keyed by the board's host so two different boards
    do not block each other, but every process on this machine (concurrent beams, other sessions)
    contends on the same file when hitting the same board."""
    tag = hashlib.sha1((p.host or "unset").encode()).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"{p.binary}_board_{tag}.lock"


@contextmanager
def board_lock(board: Board | Profile, timeout: int = 1800):
    """Serialize physical-board access across ALL processes on this host (a board has one set of
    deploy paths and one set of cores; concurrent forks would corrupt each other's scp'd binaries and
    contend for cores, poisoning the measurement). A file ``flock`` is the cross-process lock that a
    ThreadPool max_workers=1 (single-beam only) and a Ray resource gate (single cluster only) both
    miss. ``MERLIN_<BOARD>_NO_BOARD_LOCK=1`` disables it (single-user runs)."""
    p = profile(board)
    if os.environ.get(p.env("NO_BOARD_LOCK")) == "1":
        yield
        return
    lockf = _board_lock_path(p)
    fh = open(lockf, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)  # blocks until the board is free
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def toolchain_root(board: Board | Profile) -> Path | None:
    """Resolve the directory that actually contains ``bin/clang``.

    Accepts either ``$MERLIN_<BOARD>_TOOLCHAIN`` pointing directly at a toolchain prefix
    (``<prefix>/bin/clang``) OR at a parent holding an extracted vendor release, whose directory
    naming the owning target contract declares (see :mod:`merlin.common.external_tools`).
    """
    p = profile(board)
    if p.toolchain_install is None:
        return None
    return _toolchains.find_prefix(p.toolchain_install, p.env("TOOLCHAIN"), tools=("clang",))


def toolchain_cc(board: Board | Profile) -> Path | None:
    """Path to the board's linux-gnu cross clang/gcc, or None if the toolchain isn't installed."""
    p = profile(board)
    root = toolchain_root(p)
    if root is None:
        return None
    for rel in ("bin/clang", f"bin/{p.triple}-gcc"):
        c = root / rel
        if c.is_file():
            return c
    return None


def available(board: Board | Profile, *, deadline_ns: int | None = None) -> bool:
    """True only if BOTH the cross-toolchain is installed AND the board is reachable over SSH.

    Fail-closed: any uncertainty -> False -> runner records ``not_run`` (never a false pass).
    """
    p = profile(board)
    if toolchain_cc(p) is None or not p.host or not Path(p.ssh_key).is_file():
        return False
    if shutil.which("ssh") is None:
        return False
    try:
        timeout = 15
        if deadline_ns is not None:
            remaining_ns = int(deadline_ns) - time.monotonic_ns()
            if remaining_ns <= 0:
                return False
            timeout = min(timeout, max(1, math.ceil(remaining_ns / 1_000_000_000)))
        r = subprocess.run(
            [
                "ssh",
                "-i",
                p.ssh_key,
                *p.ssh_port_opts,
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                "-o",
                "StrictHostKeyChecking=no",
                p.host,
                "true",
            ],
            capture_output=True,
            timeout=timeout,
        )
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


# ---- build / deploy / run -----------------------------------------------------------

# A bounded wall clock for the cross-compile/link/scp steps. A pathological schedule (e.g. the
# outer-product contraction feature at a large square regime) makes clang spin for many minutes on
# one object file, hanging a serial beam; time it out so the fork fails-closed as a board build error
# the certify ladder records. Board-run (_ssh) already carries its own timeout. Override/disable (0)
# with MERLIN_COMPILE_TIMEOUT_S — the same knob the host/spike codegen path honors. Default unified at
# 900s across all four compile wrappers (was 300): a per-object 300s ceiling tripped legitimate
# whole-model builds (bitvla int8 fails even the baseline at 600s). For a whole-model beam launch
# set MERLIN_COMPILE_TIMEOUT_S=3600; 0 (or empty) disables the ceiling.
_CMD_TIMEOUT_S = int(os.environ.get("MERLIN_COMPILE_TIMEOUT_S", "900") or "0")
_ACTIVE_CELL_DEADLINE_NS: ContextVar[int | None] = ContextVar("merlin_remote_linux_cell_deadline_ns", default=None)


def _bounded_timeout(requested: int | float | None) -> int | float | None:
    deadline = _ACTIVE_CELL_DEADLINE_NS.get()
    if deadline is None:
        return requested
    remaining_ns = deadline - time.monotonic_ns()
    if remaining_ns <= 0:
        raise RemoteLinuxError("paper cell exhausted its whole-cell deadline")
    remaining = max(1, math.ceil(remaining_ns / 1_000_000_000))
    return remaining if requested is None else min(requested, remaining)


def _whole_cell_deadline(function):
    """Bind one optional monotonic deadline across build, deploy, perf, and validation calls."""

    @wraps(function)
    def wrapped(*args, deadline_ns: int | None = None, **kwargs):
        token = _ACTIVE_CELL_DEADLINE_NS.set(int(deadline_ns) if deadline_ns is not None else None)
        try:
            return function(*args, **kwargs)
        finally:
            _ACTIVE_CELL_DEADLINE_NS.reset(token)

    return wrapped


def _run(cmd: list, **kw) -> subprocess.CompletedProcess:
    timeout = _bounded_timeout(kw.pop("timeout", _CMD_TIMEOUT_S or None))
    return _proc.run_checked(
        cmd, error=RemoteLinuxError, timeout=timeout, timeout_hint=" within the bounded compile/cell deadline", **kw
    )


def _ssh(p: Profile, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "ssh",
            "-i",
            p.ssh_key,
            *p.ssh_port_opts,
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "StrictHostKeyChecking=no",
            p.host,
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=_bounded_timeout(timeout),
    )


def _scp(p: Profile, source, destination) -> list:
    """The ``scp`` command copying ``source`` to ``destination`` (either may be ``<host>:<path>``)."""
    return [
        "scp",
        "-i",
        p.ssh_key,
        *p.scp_port_opts,
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        source,
        destination,
    ]


def _remote_bounded(command: str, timeout: int) -> str:
    """Make the board own the deadline so a lost SSH client cannot orphan an inference.

    A host-side ``subprocess.run(timeout=...)`` kills only the local ssh process.  The remote child
    keeps running after that connection disappears (observed: runs consumed board CPU for more than
    a day after their cells timed out).  The board's userland must provide GNU timeout; put the
    measured command under it and quote the complete shell fragment as one argument.
    """
    seconds = int(timeout)
    if seconds < 1:
        raise ValueError("remote timeout must be positive")
    return f"timeout --signal=TERM --kill-after=5s {seconds}s sh -c {shlex.quote(command)}"


def board_vlenb(board: Board | Profile) -> int | None:
    """Read VLENB (bytes) from the board, or None if unreachable. VLEN_bits = vlenb*8."""
    p = profile(board)
    try:
        r = _ssh(p, "cat /proc/cpuinfo | grep -m1 -i vlenb || true", timeout=20)
        for line in r.stdout.splitlines():
            if "vlenb" in line.lower():
                return int(line.split(":")[-1].strip())
    except Exception:  # noqa: BLE001
        pass
    return None


def board_conditions(board: Board | Profile) -> dict[str, Any]:
    """Read the board conditions that can otherwise confound cross-framework latency.

    This is deliberately a separate, untimed SSH probe. Paper adapters call it immediately before
    and after a headline session and preserve both observations; they never infer frequency from
    the requested governor or from Merlin's cycle-conversion constant.
    """
    p = profile(board)
    command = (
        "p=/sys/devices/system/cpu/cpufreq/policy0; "
        "printf 'governor=%s\\n' \"$(cat $p/scaling_governor 2>/dev/null)\"; "
        "printf 'current_khz=%s\\n' \"$(cat $p/scaling_cur_freq 2>/dev/null)\"; "
        "printf 'max_khz=%s\\n' \"$(cat $p/cpuinfo_max_freq 2>/dev/null)\"; "
        "printf 'max_thermal_millic=%s\\n' \"$(cat /sys/class/thermal/thermal_zone*/temp "
        '2>/dev/null | sort -nr | head -1)"'
    )
    proc = _ssh(p, command, timeout=30)
    if proc.returncode:
        raise RemoteLinuxError(f"{p.name} board-condition probe failed: {proc.stderr.strip()}")
    values: dict[str, Any] = {}
    for line in proc.stdout.splitlines():
        key, sep, value = line.partition("=")
        if not sep or not key or not value:
            continue
        try:
            values[key] = int(value)
        except ValueError:
            values[key] = value
    required = {"governor", "current_khz", "max_khz", "max_thermal_millic"}
    if set(values) != required:
        raise RemoteLinuxError(f"incomplete {p.name} board-condition observation: {values}")
    return values


def run_arch_probe(board: Board | Profile, source: str | Path, *, timeout: int = 60) -> dict[str, Any]:
    """Cross-compile and run a small architecture-only probe on the board.

    The source is part of the target contract, not a benchmark candidate: it may read architectural
    CSRs and operating-system facts but must not time a kernel.  The returned source digest and raw
    console make the preflight evidence reproducible.  Deployment is serialized with every other user
    of the board and the temporary remote binary is removed on exit.
    """
    from merlin.runtime.backends.linux.cross_build import _require_toolchain

    p = profile(board)
    source = Path(source)
    if not source.is_file():
        raise RemoteLinuxError(f"{p.name} architecture probe does not exist: {source}")
    cc, _root = _require_toolchain(p)
    if not p.host or not Path(p.ssh_key).is_file():
        raise RemoteLinuxError(f"{p.name} SSH endpoint or key is unavailable ({p.env('HOST')}, {p.env('SSH_KEY')})")
    source_bytes = source.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    remote = p.scratch_path(f"{p.binary}_arch_probe_{digest[:12]}")
    with tempfile.TemporaryDirectory(prefix=f"merlin-{p.name}-probe-") as tmp:
        binary = Path(tmp) / "probe"
        _run(
            [
                str(cc),
                p.target_flag,
                f"-march={p.march}",
                f"-mabi={p.board.mabi}",
                "-O2",
                str(source),
                "-static",
                "-o",
                str(binary),
            ]
        )
        with board_lock(p):
            _run(_scp(p, str(binary), f"{p.host}:{remote}"))
            try:
                _ssh(p, f"chmod +x {remote}", timeout=30)
                proc = _ssh(p, remote, timeout=timeout)
            finally:
                try:
                    _ssh(p, f"rm -f {remote}", timeout=30)
                except Exception:  # noqa: BLE001 -- cleanup must not hide the probe verdict
                    pass
    if proc.returncode:
        raise RemoteLinuxError(f"{p.name} architecture probe failed ({proc.returncode}): {proc.stderr.strip()}")
    values: dict[str, Any] = {}
    for line in proc.stdout.splitlines():
        key, sep, value = line.partition("=")
        if not sep or not key:
            continue
        value = value.strip()
        try:
            values[key.strip()] = int(value)
        except ValueError:
            values[key.strip()] = value
    return {
        "source": str(source),
        "source_sha256": digest,
        "values": values,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "returncode": proc.returncode,
    }


def _remote_binary_path(p: Profile, binary: str | Path, name: str) -> str:
    """Where to stage the binary on the board: the scratch directory for small ones, ``remote_dir``
    for large.

    Above the board's ``scratch_limit_bytes`` the binary goes to ``remote_dir`` on real storage. A
    scratch directory that is a tmpfs is RAM, so a weights-embedded binary there lands in the same
    memory the model needs to run in, and two of them may not fit at all. This is the SAME reason the
    mmap path stages weights.bin under ``remote_dir``; a binary with the weights linked in IS the
    weights. A full tmpfs makes scp fail with a bare "write remote ... Failure" that looks like a
    network error rather than ENOSPC.
    """
    try:
        big = Path(binary).stat().st_size > int(p.board.scratch_limit_bytes)
    except OSError:
        big = False
    return f"{p.remote_dir}/{name}" if big else p.scratch_path(name)


def _pull_full_output(p: Profile, remote_out: str, bwork: Path, result: dict[str, Any]) -> None:
    """Replace the empty console output with the complete binary tensor from the board.

    The generated harness deliberately leaves an ``OUT 0`` framing record in stdout so the
    existing parser still proves that the process reached ``DONE``.  The element count printed
    beside the file is the independent completeness witness: a short or missing transfer fails
    before any accuracy gate can mistake it for the model's answer.
    """
    local_out = bwork / "board_output.bin"
    command = _scp(p, f"{p.host}:{remote_out}", local_out)
    # The inference and its output file already exist on the board.  A transient network/SSH close
    # while pulling that file must retry BEFORE the caller's finally block deletes it; otherwise
    # a multi-minute, correct run is irretrievably discarded for a transport failure unrelated to
    # the model.  Three immediate attempts are bounded and preserve the original exception if all
    # fail.  The element-count check below still rejects partial transfers.
    for attempt in range(3):
        try:
            _run(command)
            break
        except RemoteLinuxError:
            local_out.unlink(missing_ok=True)
            if attempt == 2:
                raise
            # A dropped SSH transport can remain unusable for the next immediate
            # connection attempt while sshd tears the old session down.  Back off
            # briefly; the remote result is immutable and still present until the
            # caller's cleanup, so retrying the pull is safe and idempotent.
            time.sleep(1 << attempt)
    import numpy as _np

    output = _np.fromfile(local_out, dtype=_np.float32)
    expected = int(result.get("metrics", {}).get("output_file_elems") or 0)
    if not expected or len(output) != expected:
        raise RemoteLinuxError(f"board output file has {len(output)} elements; expected {expected}")
    result["outputs"] = output
    result["prefix"] = output
    result["output_complete"] = True


def _deploy_weights(p: Profile, bwork: Path, remote: str) -> tuple[str, str | None, list[str]]:
    """Stage the mmap'd weight blob(s) a build left markers for; return ``(env prefix, single blob,
    per-program blobs)``. Weights go to ``remote_dir`` on real storage, never the scratch directory."""
    marker = bwork / "USE_MMAP_WEIGHTS"
    multi_marker = bwork / "USE_MMAP_WEIGHTS.json"
    wenv, remote_w = "", None
    remote_weights: list[str] = []
    if multi_marker.is_file():
        _ssh(p, f"mkdir -p {p.remote_dir}", timeout=30)
        by_digest: dict[str, str] = {}
        for item in json.loads(multi_marker.read_text(encoding="utf-8")):
            digest, program = str(item["sha256"]), int(item["program"])
            remote_stage = by_digest.get(digest)
            if remote_stage is None:
                remote_stage = f"{p.remote_dir}/{Path(remote).name}.weights.{digest[:16]}.bin"
                _run(_scp(p, str(item["path"]), f"{p.host}:{remote_stage}"))
                by_digest[digest] = remote_stage
                remote_weights.append(remote_stage)
            wenv += f"MERLIN_WEIGHTS_{program}={remote_stage} "
    elif marker.is_file():
        _ssh(p, f"mkdir -p {p.remote_dir}", timeout=30)
        remote_w = f"{p.remote_dir}/{Path(remote).name}.weights.bin"
        _run(_scp(p, marker.read_text().strip(), f"{p.host}:{remote_w}"))
        wenv = f"MERLIN_WEIGHTS={remote_w} "
    return wenv, remote_w, remote_weights


def _omp_env(p: Profile, threads: int) -> str:
    bind = p.board.omp_proc_bind
    return f"OMP_NUM_THREADS={threads} " + (f"OMP_PROC_BIND={bind} " if bind else "")


def run_binary(
    board: Board | Profile,
    model_dir: str | Path,
    bwork: str | Path,
    pkg,
    binary: str | Path,
    *,
    env: dict[str, str] | None = None,
    timeout: int = 600,
    capture_full_output: bool = False,
    parallel_harts: int | None = None,
) -> dict[str, Any]:
    """Deploy and run an ALREADY-BUILT board binary, with an explicit environment.

    Split out of :func:`run`'s build+run ladder so a measurement can run the SAME binary
    repeatedly under different settings — the multicore scaling curve varies only
    ``OMP_NUM_THREADS``, and rebuilding per point would put a different object under each
    measurement and make the curve unattributable.

    Takes the board lock around deploy+run only (the cross-compile is CPU-local), matching
    :func:`run`: concurrent agents can build in parallel while the physical board serializes.
    """
    from merlin.runtime.backends.zephyr import execution as zm

    p = profile(board)
    if not p.host:
        raise RemoteLinuxError(f"{p.env('HOST')} unset — board unreachable")
    if parallel_harts is not None and int(parallel_harts) < 1:
        raise ValueError("parallel_harts must be positive when provided")
    bwork = Path(bwork)
    remote = _remote_binary_path(p, binary, f"{Path(model_dir).name}_{pkg.run_id}_envrun_{p.binary}")
    remote_out = f"{p.remote_dir}/{Path(remote).name}.output.bin" if capture_full_output else None
    with board_lock(p):
        if remote.startswith(p.remote_dir) or remote_out:
            _ssh(p, f"mkdir -p {p.remote_dir}", timeout=30)
        _run(_scp(p, str(binary), f"{p.host}:{remote}"))
        wenv, remote_w, remote_weights = _deploy_weights(p, bwork, remote)
        output_env = f"MERLIN_OUTPUT_FILE={remote_out} " if remote_out else ""
        run_env = dict(env or {})
        if parallel_harts is not None:
            run_env["OMP_NUM_THREADS"] = str(int(parallel_harts))
            if p.board.omp_proc_bind:
                run_env["OMP_PROC_BIND"] = p.board.omp_proc_bind
        envs = "".join(f"{k}={v} " for k, v in run_env.items())
        affinity_list = (
            f"0-{int(parallel_harts) - 1}" if parallel_harts is not None and int(parallel_harts) > 1 else None
        )
        taskset = f"taskset -c {affinity_list} " if affinity_list else ""
        try:
            _ssh(p, f"chmod +x {remote}", timeout=30)
            conditions_before = board_conditions(p)
            command = f"{wenv}{output_env}{envs}{taskset}{remote}"
            proc = _ssh(p, _remote_bounded(command, timeout), timeout=timeout + 15)
            result = zm._parse_console(proc.stdout + proc.stderr, proc.returncode)
            if remote_out:
                _pull_full_output(p, remote_out, bwork, result)
            result["board_conditions"] = {"before": conditions_before, "after": board_conditions(p)}
            result["affinity_mask"] = affinity_list
            result["requested_core_count"] = int(parallel_harts or 1)
            result["core_count"] = int(parallel_harts or 1)
            result["affinity_available_cpus"] = int(result.get("metrics", {}).get("affinity_cpus") or 0)
            return result
        finally:
            try:
                cleanup = [remote]
                if remote_w:
                    cleanup.append(remote_w)
                cleanup.extend(remote_weights)
                if remote_out:
                    cleanup.append(remote_out)
                _ssh(p, "rm -f " + " ".join(cleanup), timeout=30)
            except Exception:  # noqa: BLE001
                pass


def _is_multi_program(model_dir: str | Path) -> bool:
    """True when this bundle is a VERSION-2 session, i.e. several programs in one image.

    Version 1 declares a step count and stays a single program (the harness loops its steps
    internally), so it takes the ordinary single-program build path. Only version 2 routes to the
    multi-program builder. Kept as its own predicate so a caller can ask the question without
    duplicating the version check -- and so a capability that path does not have can be refused
    before anything is built.
    """
    path = Path(model_dir) / "session_contract.yaml"
    if not path.is_file():
        return False
    from merlin.common.record_formats import load_yaml

    value = load_yaml(path)
    return isinstance(value, dict) and int(value.get("version", 0) or 0) == 2


@_whole_cell_deadline
def run(
    board: Board | Profile,
    model_dir: str | Path,
    work: str | Path,
    pkg,
    *,
    timeout: int = 600,
    kernel_backend: str | None = None,
    dispatch_timing: bool = False,
    op_profile: bool = False,
    force_scalar: bool = False,
    parallel_harts: int | None = None,
    fallback_policy: str = "allow",
    host_effects=None,
    require_csr_vlen: bool = False,
    iters: int = 1,
    warmup: int = 0,
    session_repeats: int | None = None,
    dump_cap: int | None = 4096,
    max_session_steps: int | None = None,
) -> dict[str, Any]:
    """Cross-compile the workload for ``board``, deploy over scp, run, and parse OUT/METRIC/DONE.

    Returns the ``zephyr_execution._parse_console`` result dict augmented with ``vlen`` (the bit-width
    the board reports — vlenb*8, falling back to the declared VLEN). The ``cycles`` metric is an
    ESTIMATE from the board's ``time`` CSR scaled by ``chip_freq_hz / timebase_hz`` (not
    cycle-accurate); ``wall_ns`` is the silicon authority.

    ``dump_cap`` is the number of output elements the board prints, and therefore the CEILING on
    how much of the answer an accuracy verdict can cover -- the host gate scores the printed prefix
    against the leading elements of the reference, so a short console produces a real-looking
    cos/rel over a slice nobody asked about. ``dump_cap=None`` prints the model's whole output
    (derived from its own ``MERLIN_OUT_ELEMS``). The default is unchanged at 4096, so an existing
    caller gets a byte-identical binary; callers that grade accuracy should pass ``None`` and check
    ``comparison_complete`` on the gate. The value used is echoed back as ``dump_cap``.
    """
    from merlin.runtime.backends.linux.cross_build import (
        _require_worker_census,
        _resolve_openmp_selection,
        build_binary,
        build_session_binary,
    )
    from merlin.runtime.backends.zephyr import application as zm
    from merlin.runtime.backends.zephyr import execution as zephyr_execution

    p = profile(board)
    if fallback_policy not in ("allow", "forbid"):
        raise ValueError("fallback_policy must be 'allow' or 'forbid'")
    openmp = _resolve_openmp_selection(parallel=False, parallel_harts=parallel_harts, host_effects=host_effects)
    parallel_harts = openmp.parallel_harts
    if force_scalar and host_effects is not None:
        raise RemoteLinuxError("force_scalar=True conflicts with host-policy parallel lowering effects")

    if not p.host:
        raise RemoteLinuxError(f"{p.env('HOST')} unset — board unreachable")

    if op_profile and _is_multi_program(model_dir):
        # FAIL CLOSED. The multi-program session build path does not take an ``op_profile``
        # parameter, so asking for a per-op profile of a version-2 session builds the
        # UN-instrumented binary, writes no ``opprof_table.json``, and returns a result with no
        # ``op_profile`` key. Downstream that reads as an empty op table, and the profiler reports a
        # breakdown of a model with no ops -- "nothing was measured" wearing the clothes of "there
        # was nothing to measure". It also costs a full build and a board slot to produce that.
        raise RemoteLinuxError(
            "op_profile=True is not supported for a version-2 multi-program session: the session "
            "build path does not instrument the IR, so the run would produce no PROF lines. "
            "Profile each program's bundle (the `stages/<name>` directories) on its own instead."
        )

    requested_mode = "scalar" if force_scalar else "rvv_openmp" if parallel_harts else "rvv"
    multi_program = _is_multi_program(model_dir)

    def _build_deploy_run(mode: str, tag: str) -> dict:
        # mode: "rvv" vectorized (fixed-width RVV); "omp" scalar int8 + OpenMP across the board's
        # cores; "scalar" single-core scalar int8 (last-resort correctness).
        bwork = Path(work) / tag
        # the cross-compile is CPU-local (no board) — do it OUTSIDE the board lock so concurrent
        # forks still build in parallel; only the deploy+run below serializes on the physical board.
        if multi_program:
            if mode == "omp":
                raise RemoteLinuxError("multi-program scalar OpenMP fallback is not implemented")
            binary = build_session_binary(
                p,
                model_dir,
                bwork,
                pkg,
                force_scalar=(mode == "scalar"),
                parallel_harts=(parallel_harts if mode == "rvv_openmp" else None),
                host_effects=(host_effects if mode == "rvv_openmp" else None),
                fallback_policy=fallback_policy,
                kernel_backend=kernel_backend,
                dump_cap=dump_cap,
            )
        else:
            binary = build_binary(
                p,
                model_dir,
                bwork,
                pkg,
                force_scalar=(mode == "scalar"),
                parallel=(mode == "omp"),
                parallel_harts=(parallel_harts if mode == "rvv_openmp" else None),
                host_effects=(host_effects if mode == "rvv_openmp" else None),
                fallback_policy=fallback_policy,
                kernel_backend=kernel_backend,
                dispatch_timing=dispatch_timing,
                op_profile=op_profile,
                dump_cap=dump_cap,
                max_session_steps=max_session_steps,
            )
        with board_lock(p):
            result = _deploy_run(mode, tag, bwork, binary)
        result["local_binary"] = str(binary)
        # RECORD the ceiling the console was produced under. `_gate` already reports
        # n_compared/n_reference/compared_fraction/comparison_complete from the arrays, but those
        # say only THAT a run was short, never why. Without this a truncated verdict and a complete
        # one are indistinguishable in the artifact after the fact.
        result["dump_cap"] = None if dump_cap is None else int(dump_cap)
        result["output_complete"] = dump_cap is None or len(result.get("outputs", ())) < int(dump_cap)
        result["memory_policy"] = (
            "mmap"
            if ((bwork / "USE_MMAP_WEIGHTS").is_file() or (bwork / "USE_MMAP_WEIGHTS.json").is_file())
            else "resident"
        )
        result["execution_mode"] = mode
        result["requested_execution_mode"] = requested_mode
        result["fallback_used"] = mode != requested_mode
        requested_cores = int(parallel_harts) if mode == "rvv_openmp" else p.omp_threads if mode == "omp" else 1
        # A serial image has exactly one calling thread but deliberately remains unpinned, so its
        # affinity mask may contain every CPU. A parallel image is taskset-bounded and must observe
        # exactly the requested mask; fail closed instead of publishing a mislabeled paper cell.
        result["requested_core_count"] = requested_cores
        affinity_cpus = int(result.get("metrics", {}).get("affinity_cpus") or 0)
        result["affinity_available_cpus"] = affinity_cpus
        if mode in ("omp", "rvv_openmp"):
            if affinity_cpus != requested_cores:
                raise RemoteLinuxError(
                    f"requested {requested_cores} cores but sched_getaffinity exposed {affinity_cpus}"
                )
            result["core_count"] = affinity_cpus
        else:
            result["core_count"] = 1
        result["affinity_source"] = "sched_getaffinity"
        if mode == "rvv_openmp" and openmp.provider == "merlin_pthread_pool":
            worker_cpus = _require_worker_census(result.get("metrics", {}), requested=int(parallel_harts))
            result["openmp_runtime"] = openmp.provider
            result["worker_cpus"] = list(worker_cpus)
            result["worker_affinity_ok"] = True
        return result

    def _deploy_run(mode: str, tag: str, bwork: Path, binary) -> dict:
        remote = _remote_binary_path(p, binary, f"{Path(model_dir).name}_{pkg.run_id}_{tag}_{p.binary}")
        if remote.startswith(p.remote_dir):
            _ssh(p, f"mkdir -p {p.remote_dir}", timeout=30)
        _run(_scp(p, str(binary), f"{p.host}:{remote}"))
        # Big-model mmap path: build_binary left a marker with the weights.bin to deploy
        # alongside; the binary mmaps it (resident RAM = working set, not the whole blob).
        wenv, remote_w, remote_weights = _deploy_weights(p, bwork, remote)
        # Fan the OpenMP loops across the board's cores (bound per the board's omp_proc_bind).
        thread_count = int(parallel_harts) if mode == "rvv_openmp" else p.omp_threads
        env = _omp_env(p, thread_count) if mode in ("omp", "rvv_openmp") else ""
        affinity_count = thread_count if mode in ("omp", "rvv_openmp") else 1
        # PIN ONLY WHEN PINNING IS THE POINT. `taskset` here exists to bound a MULTICORE run to a
        # known core count. A single-threaded run does not need it, and pinning one to cpu0
        # specifically is actively wrong for a comparison: cpu0 commonly carries the board's
        # interrupt load, and pinning a single-threaded run there has measured several times slower
        # than unpinned on a BYTE-IDENTICAL binary (model.ll, model.o and the linked ELF all
        # unchanged; only the launch command differed).
        #
        # It is also an ASYMMETRY, which is the part that makes a ratio wrong rather than merely
        # slow: the ExecuTorch arm takes `--cpu_threads=1` and is never tasksetted, so its one
        # thread migrates freely over every core. Pinning ours and not theirs compares a pinned
        # arm against a floating one. Single-threaded therefore gets the same treatment they get --
        # one thread, unpinned -- and the mask is recorded either way so the choice is auditable
        # from the artifact instead of having to be re-derived from an anomaly.
        affinity_list = None if affinity_count == 1 else f"0-{affinity_count - 1}"
        taskset = "" if affinity_list is None else f"taskset -c {affinity_list} "
        # Sustained mode: warmup + N timed passes against the same buffers. Only emitted when
        # asked for, so the default console stays byte-identical to the single-shot path.
        if session_repeats is not None:
            if int(session_repeats) < 1:
                raise ValueError("session_repeats must be positive when provided")
            env += f"MERLIN_SESSION_REPEATS={int(session_repeats)} MERLIN_SESSION_WARMUPS={int(warmup)} "
        elif int(iters) > 1 or int(warmup) > 0:
            env += f"MERLIN_ITERS={int(iters)} MERLIN_WARMUP={int(warmup)} "
        remote_out = f"{p.remote_dir}/{Path(remote).name}.output.bin" if dump_cap is None else None
        output_env = f"MERLIN_OUTPUT_FILE={remote_out} " if remote_out else ""
        if remote_out:
            _ssh(p, f"mkdir -p {p.remote_dir}", timeout=30)
        try:
            _ssh(p, f"chmod +x {remote}", timeout=30)
            conditions_before = board_conditions(p)
            command = f"{wenv}{output_env}{env}{taskset}{remote}"
            proc = _ssh(p, _remote_bounded(command, timeout), timeout=timeout + 15)
            r = zephyr_execution._parse_console(proc.stdout + proc.stderr, proc.returncode)
            if remote_out:
                _pull_full_output(p, remote_out, bwork, r)
            conditions_after = board_conditions(p)
            r["board_conditions"] = {"before": conditions_before, "after": conditions_after}
            # The mask this wall was produced under. Recorded even when absent (None = unpinned),
            # because "no taskset" and "pinned to cpu0" can differ several-fold and a bare number
            # cannot tell them apart after the fact.
            r["affinity_mask"] = affinity_list
            if (bwork / "HAS_SESSION_QUALITY").is_file():
                quality_env = "MERLIN_VALIDATE_SESSION=1 MERLIN_SESSION_REPEATS=1 MERLIN_SESSION_WARMUPS=0 "
                quality_command = f"{wenv}{env}{quality_env}{taskset}{remote}"
                qproc = _ssh(p, _remote_bounded(quality_command, timeout), timeout=timeout + 15)
                qres = zephyr_execution._parse_console(qproc.stdout + qproc.stderr, qproc.returncode)
                qm = qres.get("metrics", {})
                steps = int(qm.get("trajectory_steps") or 0)
                r["trajectory_quality"] = {
                    "scope": "trajectory",
                    "steps": steps,
                    "min_cosine": ((qm.get("trajectory_min_cos_ppm") or 0) / 1_000_000.0),
                    "max_relative_error": ((qm.get("trajectory_max_rel_ppm") or 0) / 1_000_000.0),
                    "top1_matches": int(qm.get("trajectory_top1_matches") or 0),
                    "top1_agreement": ((int(qm.get("trajectory_top1_matches") or 0) / steps) if steps else None),
                    "validation_console": qres.get("console", ""),
                }
                correctness_steps = int(qm.get("correctness_trajectory_steps") or 0)
                r["trajectory_correctness"] = {
                    "scope": "trajectory",
                    "steps": correctness_steps,
                    "min_cosine": ((qm.get("correctness_trajectory_min_cos_ppm") or 0) / 1_000_000.0),
                    "max_relative_error": ((qm.get("correctness_trajectory_max_rel_ppm") or 0) / 1_000_000.0),
                    "top1_matches": int(qm.get("correctness_trajectory_top1_matches") or 0),
                    "top1_agreement": (
                        int(qm.get("correctness_trajectory_top1_matches") or 0) / correctness_steps
                        if correctness_steps
                        else None
                    ),
                    "validation_console": qres.get("console", ""),
                }
            ni = bwork / "N_OUTLINED_INT8_ROUTED"
            if ni.is_file():
                r["n_outlined_int8_routed"] = int(ni.read_text().strip())
            # Per-op profile: join the board's `PROF <id> <ticks> <hits>` lines against the
            # id->op table the build emitted. Absent (or empty) unless op_profile=True.
            tbl = bwork / "opprof_table.json"
            if op_profile and tbl.is_file():
                from merlin.compiler.lower.model.profile import parse_prof_lines

                table = json.loads(tbl.read_text())
                ticks = parse_prof_lines(r["console"])
                for rec in table:
                    t, h = ticks.get(rec["id"], (0, 0))
                    rec["ticks"], rec["hits"] = t, h
                r["op_profile"] = table
            return r
        finally:
            try:
                cleanup = ([remote_w] if remote_w else []) + remote_weights
                if remote_out:
                    cleanup.append(remote_out)
                _ssh(p, f"rm -f {remote}" + (" " + " ".join(cleanup) if cleanup else ""), timeout=30)
            except Exception:  # noqa: BLE001
                pass

    if force_scalar:
        # Single-core scalar int8: the SAME lowering minus vectorization. Its only use is as a
        # differential oracle — when a vectorized board run disagrees with the host, running the
        # scalar build on the same silicon says whether the fault is in the RVV codegen or in
        # everything else (harness, toolchain, arg table, board). Slow; never a perf path.
        res = _build_deploy_run("scalar", "scalar")
    elif parallel_harts:
        # Exact vector+multicore mode. A paper run must never silently turn this into scalar OpenMP
        # or single-core scalar; the caller selects fallback_policy='forbid' for that contract.
        res = _build_deploy_run("rvv_openmp", "rvv_openmp")
    else:
        try:
            res = _build_deploy_run("rvv", "rvv")
        except zm.ZephyrModelError:
            if fallback_policy == "forbid":
                raise
            # The vectorized int8 lowering OOB-stores / crashes for some shapes (e.g. rdt's
            # diffusion transformer) while smaller models vectorize fine. Retry with the MULTICORE
            # lowering (scalar int8 datapath + OpenMP parallel loops) — correct AND fast enough to
            # finish on a multicore board, where single-core scalar times out. Final fall-back:
            # single-core scalar.
            try:
                res = _build_deploy_run("omp", "omp")
            except (zm.ZephyrModelError, RemoteLinuxError):
                res = _build_deploy_run("scalar", "scalar")

    # Prefer the vlenb the harness printed (=== <binary> vlenb=N ===); else probe; else the declaration.
    vlen_bits = p.vlen
    vlen_source = "declared"
    for line in res["console"].splitlines():
        if "vlenb=" in line:
            try:
                vlen_bits = int(line.split("vlenb=")[1].split()[0]) * 8
                vlen_source = "csr"
            except (IndexError, ValueError):
                pass
            break
    if vlen_source == "declared":
        probed = board_vlenb(p)
        if probed is not None:
            vlen_bits = int(probed) * 8
            vlen_source = "procfs"
    if require_csr_vlen and vlen_source != "csr":
        raise RemoteLinuxError("paper run requires vlenb read from the harness CSR; no CSR value was parsed")
    if vlen_source in ("csr", "procfs"):
        verify_vlen(p, vlen_bits // 8)
    res["vlen"] = vlen_bits
    res["vlen_source"] = vlen_source
    return res
